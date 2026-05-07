#include "utils.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used by element_add / pytorch_linear / etc.
#include "kernel_profiler.h"

#include <clblast.h>       // clblast::Gemm — used by pytorch_linear (dtype-templated dispatch).

#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <cstdint>

// ──────────────────────────────────────────────
// gemv_m1 fast path for the M=1 decode GEMV.
//
// CLBlast's HGemm M=1 path is "1 thread per output column, no vec4, no
// cooperative reduction" — measured at ~6× the realistic memory-BW ceiling on
// Adreno 620. The replacement kernels in kernels/gemv_m1.cl use cooperative
// WG=64, vec4 fp16 loads, and __local-mem tree-reduce; the multi-output _no4
// variant additionally does 4 outputs per WG with 4 independent fp32
// accumulators (register-level parallelism).
//
// Lazy-builds kernels/gemv_m1.cl on first call, caches kernels for the
// process. Returns false if the K/N pair is not eligible — caller falls
// through to CLBlast (e.g. prefill M>1 paths).
// ──────────────────────────────────────────────
static bool try_gemv_m1_fast_path(cl_command_queue queue,
                                  int N, int K,
                                  cl_mem x, cl_mem W, cl_mem out);

// gemv_m1 fast path with FUSED RESIDUAL ADD. See pytorch_linear_radd().
// Reads `hidden[n]`, computes the dot product, writes `hidden[n] = sum + r`.
// Returns false if the K/N pair is not eligible (M=1 implied, K%64==0, N%4==0).
static bool try_gemv_m1_radd_fast_path(cl_command_queue queue,
                                       int N, int K,
                                       cl_mem x, cl_mem W, cl_mem hidden);

// ──────────────────────────────────────────────
// IEEE 754 binary16 codec (host-side).
// Bit-exact: handles subnormals, Inf, NaN, saturating overflow on encode.
// Returns float32 on decode. Branch-light implementation, no compiler-half
// intrinsic dependence so it compiles identically across NDK / Linux hosts.
// ──────────────────────────────────────────────

float nnopt_f16_to_f32(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits >> 15) & 0x1u;
    uint32_t exp  = (uint32_t)(bits >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)(bits      ) & 0x3FFu;
    uint32_t out_sign = sign << 31;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = out_sign;                              // ±0
        } else {
            // Subnormal: normalize.
            int e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            uint32_t out_exp = (uint32_t)(127 - 15 - e);
            out = out_sign | (out_exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        // Inf or NaN.
        out = out_sign | 0x7F800000u | (mant << 13);
    } else {
        uint32_t out_exp = (uint32_t)(exp - 15 + 127);
        out = out_sign | (out_exp << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

uint16_t nnopt_f32_to_f16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    uint32_t sign = (bits >> 31) & 0x1u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    uint16_t out_sign = (uint16_t)(sign << 15);
    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        // Inf / NaN
        uint16_t out_mant = mant ? (uint16_t)((mant >> 13) | 0x200u) : 0u; // preserve NaN-ness
        return (uint16_t)(out_sign | 0x7C00u | out_mant);
    }
    if (exp >= 0x1F) {
        // Saturating overflow → ±Inf
        return (uint16_t)(out_sign | 0x7C00u);
    }
    if (exp <= 0) {
        // Subnormal or underflow.
        if (exp < -10) return out_sign;                     // → ±0
        mant |= 0x800000u;                                  // restore implicit 1
        uint32_t shift = (uint32_t)(14 - exp);
        // Round to nearest even
        uint32_t round_bit = mant & (1u << (shift - 1));
        uint32_t sticky    = mant & ((1u << (shift - 1)) - 1u);
        uint16_t out_mant  = (uint16_t)(mant >> shift);
        if (round_bit && (sticky || (out_mant & 1u))) out_mant++;
        return (uint16_t)(out_sign | out_mant);
    }
    // Normal — round to nearest even.
    uint32_t round_bit = mant & 0x1000u;
    uint32_t sticky    = mant & 0x0FFFu;
    uint16_t out_mant  = (uint16_t)((mant >> 13) & 0x3FFu);
    uint16_t out_exp   = (uint16_t)(exp & 0x1Fu);
    uint16_t out       = (uint16_t)(out_sign | (out_exp << 10) | out_mant);
    if (round_bit && (sticky || (out_mant & 1u))) {
        out++;  // may carry into exp; that's ok per IEEE 754 round-half-to-even.
    }
    return out;
}

float compute_mse(const float* a, const float* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = (double)a[i] - (double)b[i];
        sum += diff * diff;
    }
    return (float)(sum / n);
}

float compute_max_diff(const float* a, const float* b, size_t n) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

std::vector<float> load_npy_float32(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return {};

    // Skip numpy header (simplified parser)
    char magic[6];
    file.read(magic, 6);
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);

    std::string header(header_len, '\0');
    file.read(&header[0], header_len);

    // Read remaining data as float32
    auto pos = file.tellg();
    file.seekg(0, std::ios::end);
    auto end_pos = file.tellg();
    file.seekg(pos);

    size_t num_bytes = end_pos - pos;
    size_t num_floats = num_bytes / sizeof(float);

    std::vector<float> data(num_floats);
    file.read(reinterpret_cast<char*>(data.data()), num_bytes);

    return data;
}

void save_npy_float32(const std::string& path, const float* data, const std::vector<size_t>& shape) {
    // Minimal .npy writer for float32
    std::ofstream file(path, std::ios::binary);

    // Magic
    file.write("\x93NUMPY", 6);
    uint8_t major = 1, minor = 0;
    file.write(reinterpret_cast<char*>(&major), 1);
    file.write(reinterpret_cast<char*>(&minor), 1);

    // Header
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        shape_str += std::to_string(shape[i]);
        if (i < shape.size() - 1) shape_str += ", ";
    }
    shape_str += ")";

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + "}";
    // Pad to multiple of 64
    while ((10 + header.size() + 1) % 64 != 0) header += ' ';
    header += '\n';

    uint16_t header_len = (uint16_t)header.size();
    file.write(reinterpret_cast<char*>(&header_len), 2);
    file.write(header.c_str(), header.size());

    // Data
    size_t total = 1;
    for (auto s : shape) total *= s;
    file.write(reinterpret_cast<const char*>(data), total * sizeof(float));
}

// In-place add: a[i] += b[i]. Kernel object cached per program so repeat
// calls don't pay clCreateKernel. Use this for residual adds at decode
// to keep the M=1 hot path allocation-free (Rule FUSE-DECODE-01).
bool element_add_inplace(cl_command_queue queue, cl_program utils_program,
                         cl_mem a, cl_mem b, size_t n) {
    static cl_program s_cached_program = nullptr;
    static cl_kernel  s_cached_kernel  = nullptr;
    if (s_cached_program != utils_program) {
        if (s_cached_kernel) { clReleaseKernel(s_cached_kernel); s_cached_kernel = nullptr; }
        cl_int kerr = CL_SUCCESS;
        s_cached_kernel = clCreateKernel(utils_program, "element_add", &kerr);
        if (kerr != CL_SUCCESS || !s_cached_kernel) {
            NNOPT_ERROR_FMT("element_add_inplace: clCreateKernel failed (%d)", kerr);
            s_cached_kernel = nullptr;
            return false;
        }
        s_cached_program = utils_program;
    }
    int n_int = (int)n;
    cl_int err = clSetKernelArg(s_cached_kernel, 0, sizeof(cl_mem), &a);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("element_add_inplace arg0: %d", err); return false; }
    err = clSetKernelArg(s_cached_kernel, 1, sizeof(cl_mem), &b);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("element_add_inplace arg1: %d", err); return false; }
    err = clSetKernelArg(s_cached_kernel, 2, sizeof(int), &n_int);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("element_add_inplace arg2: %d", err); return false; }
    size_t gws = n;
    cl_event* evt = KernelProfiler::event_for("element_add_inplace");
    err = clEnqueueNDRangeKernel(queue, s_cached_kernel, 1, nullptr, &gws, nullptr, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("element_add_inplace: clEnqueueNDRangeKernel failed (%d)", err);
        return false;
    }
    return true;
}

cl_mem element_add(cl_command_queue queue, cl_program utils_program, cl_mem a, cl_mem b, size_t n) {
    cl_int err;
    cl_context ctx;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);

    // Allocate output buffer (storage_t: cl_half under fp16, float under fp32).
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("element_add: clCreateBuffer failed (%d)", err);
        return nullptr;
    }

    // Copy a into out
    err = clEnqueueCopyBuffer(queue, a, out, 0, 0, n * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("element_add: clEnqueueCopyBuffer failed (%d)", err);
        clReleaseMemObject(out);
        return nullptr;
    }

    // Dispatch element_add kernel: out[i] += b[i]
    cl_kernel kernel = clCreateKernel(utils_program, "element_add", &err);
    if (err != CL_SUCCESS || !kernel) {
        NNOPT_ERROR_FMT("element_add: clCreateKernel(\"element_add\") failed (%d)", err);
        clReleaseMemObject(out);
        return nullptr;
    }

    int n_int = (int)n;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &out);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
    clSetKernelArg(kernel, 2, sizeof(int), &n_int);

    size_t global_size = n;
    cl_event* evt = KernelProfiler::event_for("element_add");
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("element_add: clEnqueueNDRangeKernel failed (%d)", err);
    }

    clReleaseKernel(kernel);
    return out;
}

bool split_last_dim_2(cl_command_queue queue, cl_program utils_program,
                      cl_mem src, cl_mem first, cl_mem second,
                      int rows, int half_cols) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(utils_program, "split_last_dim_2", &err);
    if (err != CL_SUCCESS || !kernel) {
        NNOPT_ERROR_FMT("split_last_dim_2: clCreateKernel(\"split_last_dim_2\") failed (%d)", err);
        return false;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &first);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &second);
    clSetKernelArg(kernel, 3, sizeof(int), &rows);
    clSetKernelArg(kernel, 4, sizeof(int), &half_cols);

    size_t global_size = (size_t)rows * (size_t)half_cols;
    cl_event* evt = KernelProfiler::event_for("split_last_dim_2");
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("split_last_dim_2: clEnqueueNDRangeKernel failed (%d)", err);
        clReleaseKernel(kernel);
        return false;
    }

    // SYNC-01: queue is in-order; downstream kernels see this output without
    // explicit sync. NNOPT_DEBUG_SYNC strips to no-op in release.
    NNOPT_DEBUG_SYNC(queue);
    clReleaseKernel(kernel);
    return true;
}

bool pytorch_linear(cl_command_queue queue,
                    int M, int N, int K,
                    cl_mem x, cl_mem W, cl_mem out) {
    // out[M, N] = x[M, K] @ W[N, K]^T  where W is nn.Linear weight [N, K].
    //
    // M=1 fast path: kernels/gemv_m1.cl. All 5 OpenELM GEMV sites at decode
    // (qkv_proj K=1280, out_proj K∈{768,1024,1280}, ffn proj_1 K=1280,
    // ffn proj_2 K∈{768..5120}, lm_head K=1280) satisfy K%64==0 and N%4==0,
    // so they hit the cooperative WG=64 + vec4 + 4-output _no4 variant.
    if (M == 1 && K >= 64 && (K % 64) == 0) {
        if (try_gemv_m1_fast_path(queue, N, K, x, W, out)) return true;
        // else fall through to CLBlast (rare — only on first-call build failure)
    }
    //
    // CLBlast RowMajor GEMM signature:
    //   C[M,N] = alpha * op(A)[M,K] * op(B)[K,N] + beta * C[M,N]
    // With TransposeA=kNo, TransposeB=kYes, op(B) treats W[N,K] as B[K,N]^T.
    //
    // Leading dimensions for RowMajor:
    //   lda = K (A's stride between rows of A[M,K])
    //   ldb = K (B's stored stride between rows of W[N,K])  ← gotcha
    //   ldc = N (C's stride between rows of C[M,N])
    //
    // Dtype-templated dispatch: HGemm under fp16, SGemm under fp32. Internal
    // accumulation in CLBlast Hgemm is fp32 (verified — square sanity test).
#ifdef NNOPT_USE_FP16
    // Use the portable host-side IEEE 754 fp16 encoder defined above.
    // clblast::FloatToHalf is not portable across CLBlast builds (some
    // versions only expose it when cl_khr_fp16 was enabled at CLBlast
    // build time, leading to "no member named 'FloatToHalf'" link errors).
    cl_half h_one  = static_cast<cl_half>(nnopt_f32_to_f16(1.0f));
    cl_half h_zero = static_cast<cl_half>(nnopt_f32_to_f16(0.0f));
    auto status = clblast::Gemm<cl_half>(
        clblast::Layout::kRowMajor,
        clblast::Transpose::kNo,
        clblast::Transpose::kYes,
        M, N, K,
        h_one,
        x, 0, K,
        W, 0, K,
        h_zero,
        out, 0, N,
        &queue,
        nullptr);
#else
    auto status = clblast::Gemm<float>(
        clblast::Layout::kRowMajor,
        clblast::Transpose::kNo,
        clblast::Transpose::kYes,
        M, N, K,
        1.0f,
        x, 0, K,
        W, 0, K,
        0.0f,
        out, 0, N,
        &queue,
        nullptr);
#endif
    if (status != clblast::StatusCode::kSuccess) {
        NNOPT_ERROR_FMT("pytorch_linear: CLBlast Gemm failed status=%d (M=%d N=%d K=%d)",
                        (int)status, M, N, K);
        return false;
    }
    // SYNC-01: queue is in-order. Removing this clFinish (downstream kernels
    // see the GEMM output without explicit sync) is worth ~30% of decode
    // throughput on a 30-layer transformer at M=1 (measured 0.40 -> 1.71
    // tok/s on SmolLM2-135M, of which ~half came from removing this site
    // alone — pytorch_linear is called 7 times per layer per token).
    NNOPT_DEBUG_SYNC(queue);
    return true;
}

bool pytorch_linear_radd(cl_command_queue queue,
                         int M, int N, int K,
                         cl_mem x, cl_mem W, cl_mem hidden) {
    // Eligibility: M=1, K%64==0, N%4==0. Anything else falls back to caller's
    // pytorch_linear + element_add_inplace path.
    if (M != 1 || K < 64 || (K % 64) != 0 || (N % 4) != 0) return false;
    return try_gemv_m1_radd_fast_path(queue, N, K, x, W, hidden);
}

bool pytorch_conv1d(cl_command_queue queue,
                    int M, int N, int K,
                    cl_mem x, cl_mem W, cl_mem out) {
    // out[M, N] = x[M, K] @ W[K, N]  where W is HF Conv1D weight [K, N] = [in, out].
    //
    // HF Conv1D forward (transformers.pytorch_utils.Conv1D.forward):
    //   y = x @ self.weight + self.bias
    // No transpose. Weight is allocated as nn.Parameter(torch.empty(in, out)).
    // This is the OPPOSITE of nn.Linear, which stores [out, in] and forwards
    // as x @ W^T. The two GEMM wrappers differ by exactly one transpose flag
    // and one leading-dim convention — pick correctly per layer contract's
    // weight_key_parent_classes field.
    //
    // CLBlast RowMajor GEMM signature:
    //   C[M,N] = alpha * op(A)[M,K] * op(B)[K,N] + beta * C[M,N]
    // With TransposeA=kNo, TransposeB=kNo, op(B) reads W[K,N] directly.
    //
    // Leading dimensions for RowMajor:
    //   lda = K (A's stride between rows of A[M,K])
    //   ldb = N (B's stride between rows of W[K,N])  ← differs from pytorch_linear
    //   ldc = N (C's stride between rows of C[M,N])
#ifdef NNOPT_USE_FP16
    cl_half h_one  = static_cast<cl_half>(nnopt_f32_to_f16(1.0f));
    cl_half h_zero = static_cast<cl_half>(nnopt_f32_to_f16(0.0f));
    auto status = clblast::Gemm<cl_half>(
        clblast::Layout::kRowMajor,
        clblast::Transpose::kNo,
        clblast::Transpose::kNo,
        M, N, K,
        h_one,
        x, 0, K,
        W, 0, N,
        h_zero,
        out, 0, N,
        &queue,
        nullptr);
#else
    auto status = clblast::Gemm<float>(
        clblast::Layout::kRowMajor,
        clblast::Transpose::kNo,
        clblast::Transpose::kNo,
        M, N, K,
        1.0f,
        x, 0, K,
        W, 0, N,
        0.0f,
        out, 0, N,
        &queue,
        nullptr);
#endif
    if (status != clblast::StatusCode::kSuccess) {
        NNOPT_ERROR_FMT("pytorch_conv1d: CLBlast Gemm failed status=%d (M=%d N=%d K=%d)",
                        (int)status, M, N, K);
        return false;
    }
    NNOPT_DEBUG_SYNC(queue);
    return true;
}

// ──────────────────────────────────────────────
// gemv_m1 fast-path implementation. See declaration above pytorch_linear.
// ──────────────────────────────────────────────
// Shared program for all gemv_m1 kernels (fast-path + radd). Building the
// program once per process eliminates a per-program-pair scheduling cost on
// Adreno where switching kernel objects across cl_program boundaries
// appears to flush some driver state.
static cl_program s_gemv_program = nullptr;
static bool s_gemv_init_failed = false;
static cl_kernel s_k_768           = nullptr;
static cl_kernel s_k_1536          = nullptr;
static cl_kernel s_k_768_no4       = nullptr;
static cl_kernel s_k_1280_no4      = nullptr;
static cl_kernel s_k_1536_no4      = nullptr;
static cl_kernel s_k_generic       = nullptr;
static cl_kernel s_k_generic_no4   = nullptr;
static cl_kernel s_k_generic_no4_coalesced = nullptr;
static cl_kernel s_k_1280_no4_radd    = nullptr;
static cl_kernel s_k_generic_no4_radd = nullptr;
static cl_kernel s_k_generic_no4_radd_coalesced = nullptr;

static bool ensure_gemv_program(cl_command_queue queue) {
    if (s_gemv_init_failed) return false;
    if (s_gemv_program) return true;

    cl_context ctx = nullptr;
    cl_device_id dev = nullptr;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);
    clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr);
    if (!ctx || !dev) { s_gemv_init_failed = true; return false; }

    std::ifstream f("kernels/gemv_m1.cl");
    if (!f.is_open()) {
        NNOPT_ERROR("gemv_m1: cannot open kernels/gemv_m1.cl");
        s_gemv_init_failed = true;
        return false;
    }
    std::stringstream buf; buf << f.rdbuf();
    std::string src = buf.str();
    const char* src_ptr = src.c_str();
    size_t src_len = src.size();

    cl_int err = CL_SUCCESS;
    s_gemv_program = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS || !s_gemv_program) {
        NNOPT_ERROR_FMT("gemv_m1: clCreateProgramWithSource failed (%d)", (int)err);
        s_gemv_init_failed = true;
        return false;
    }
#ifdef NNOPT_USE_FP16
    const char* opts = "-D USE_FP16=1";
#else
    const char* opts = "";
#endif
    err = clBuildProgram(s_gemv_program, 1, &dev, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(s_gemv_program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size + 1, 0);
        clGetProgramBuildInfo(s_gemv_program, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        NNOPT_ERROR_FMT("gemv_m1: build failed (%d): %s", (int)err, log.data());
        clReleaseProgram(s_gemv_program);
        s_gemv_program = nullptr;
        s_gemv_init_failed = true;
        return false;
    }

    s_k_768              = clCreateKernel(s_gemv_program, "gemv_m1_k768",       &err);
    s_k_1536             = clCreateKernel(s_gemv_program, "gemv_m1_k1536",      &err);
    s_k_768_no4          = clCreateKernel(s_gemv_program, "gemv_m1_k768_no4",   &err);
    s_k_1280_no4         = clCreateKernel(s_gemv_program, "gemv_m1_k1280_no4",  &err);
    s_k_1536_no4         = clCreateKernel(s_gemv_program, "gemv_m1_k1536_no4",  &err);
    s_k_generic          = clCreateKernel(s_gemv_program, "gemv_m1",            &err);
    s_k_generic_no4      = clCreateKernel(s_gemv_program, "gemv_m1_no4",        &err);
    s_k_generic_no4_coalesced       = clCreateKernel(s_gemv_program, "gemv_m1_no4_coalesced",       &err);
    s_k_1280_no4_radd    = clCreateKernel(s_gemv_program, "gemv_m1_k1280_no4_radd", &err);
    s_k_generic_no4_radd = clCreateKernel(s_gemv_program, "gemv_m1_no4_radd",   &err);
    s_k_generic_no4_radd_coalesced  = clCreateKernel(s_gemv_program, "gemv_m1_no4_radd_coalesced",  &err);
    if (!s_k_generic || !s_k_generic_no4 || !s_k_generic_no4_radd) {
        NNOPT_ERROR("gemv_m1: clCreateKernel for one of the generic kernels failed");
        s_gemv_init_failed = true;
        return false;
    }
    return true;
}

static bool try_gemv_m1_fast_path(cl_command_queue queue,
                                  int N, int K,
                                  cl_mem x, cl_mem W, cl_mem out) {
    if (!ensure_gemv_program(queue)) return false;

    // Pick the most specialized kernel for this (K, N) pair.
    //   K=768  with N%4==0 → gemv_m1_k768_no4
    //   K=1536 with N%4==0 → gemv_m1_k1536_no4
    //   K=768  / K=1536    → 1-output specialized
    //   any K  with N%4==0 → gemv_m1_no4 (generic 4-output)
    //   else                → gemv_m1 (generic 1-output)
    cl_kernel kernel = nullptr;
    bool is_no4 = (N % 4 == 0);
    bool needs_K_arg = false;

    if (K == 768  && is_no4)      { kernel = s_k_768_no4;  }
    else if (K == 1280 && is_no4) { kernel = s_k_1280_no4; }
    else if (K == 1536 && is_no4) { kernel = s_k_1536_no4; }
    else if (K == 768)            { kernel = s_k_768;      }
    else if (K == 1536)           { kernel = s_k_1536;     }
    else if (is_no4 && (K % 256 == 0)) { kernel = s_k_generic_no4_coalesced; needs_K_arg = true; }
    else if (is_no4)              { kernel = s_k_generic_no4; needs_K_arg = true; }
    else                          { kernel = s_k_generic;     needs_K_arg = true; }

    cl_int err = CL_SUCCESS;
    cl_uint a = 0;
    err |= clSetKernelArg(kernel, a++, sizeof(cl_mem), &x);
    err |= clSetKernelArg(kernel, a++, sizeof(cl_mem), &W);
    err |= clSetKernelArg(kernel, a++, sizeof(cl_mem), &out);
    if (needs_K_arg) {
        err |= clSetKernelArg(kernel, a++, sizeof(int), &K);
    }
    err |= clSetKernelArg(kernel, a++, sizeof(int), &N);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1: clSetKernelArg failed (%d)", (int)err);
        return false;
    }

    constexpr size_t WG = 64;
    size_t gws, lws = WG;
    if (kernel == s_k_768_no4 || kernel == s_k_1280_no4 || kernel == s_k_1536_no4 ||
        kernel == s_k_generic_no4 || kernel == s_k_generic_no4_coalesced) {
        gws = (size_t)(N / 4) * WG;
    } else {
        gws = (size_t)N * WG;
    }
    char prof_label[48];
    std::snprintf(prof_label, sizeof(prof_label), "gemv_m1_K%d_N%d", K, N);
    cl_event* evt = KernelProfiler::event_for(prof_label);
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &gws, &lws, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1: clEnqueueNDRangeKernel failed (%d) (K=%d N=%d)", (int)err, K, N);
        return false;
    }
    NNOPT_DEBUG_SYNC(queue);
    return true;
}

// ──────────────────────────────────────────────
// gemv_m1 fused-residual-add fast-path implementation. Shares the same
// cl_program as try_gemv_m1_fast_path so successive launches don't bounce
// across program objects (Adreno appears to flush some scheduler state
// when alternating cl_kernels from different programs).
// ──────────────────────────────────────────────
static bool try_gemv_m1_radd_fast_path(cl_command_queue queue,
                                       int N, int K,
                                       cl_mem x, cl_mem W, cl_mem hidden) {
    if (!ensure_gemv_program(queue)) return false;

    // Pick the K-specialized radd variant when available; otherwise prefer
    // the coalesced generic when K%256==0; fall back to the un-coalesced
    // generic only for K%64==0 && K%256!=0 (no decode-path K hits this on
    // OpenELM, but keep the path for safety).
    cl_kernel kernel = nullptr;
    bool needs_K_arg = false;
    if (K == 1280) {
        kernel = s_k_1280_no4_radd;
    } else if (K % 256 == 0) {
        kernel = s_k_generic_no4_radd_coalesced;
        needs_K_arg = true;
    } else {
        kernel = s_k_generic_no4_radd;
        needs_K_arg = true;
    }

    cl_int err = CL_SUCCESS;
    cl_uint a = 0;
    err |= clSetKernelArg(kernel, a++, sizeof(cl_mem), &x);
    err |= clSetKernelArg(kernel, a++, sizeof(cl_mem), &W);
    err |= clSetKernelArg(kernel, a++, sizeof(cl_mem), &hidden);
    if (needs_K_arg) {
        err |= clSetKernelArg(kernel, a++, sizeof(int), &K);
    }
    err |= clSetKernelArg(kernel, a++, sizeof(int), &N);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_radd: clSetKernelArg failed (%d)", (int)err);
        return false;
    }

    constexpr size_t WG = 64;
    const size_t gws = (size_t)(N / 4) * WG;
    const size_t lws = WG;

    char prof_label[64];
    std::snprintf(prof_label, sizeof(prof_label), "gemv_m1_K%d_N%d_radd", K, N);
    cl_event* evt = KernelProfiler::event_for(prof_label);
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &gws, &lws, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_radd: clEnqueueNDRangeKernel failed (%d) (K=%d N=%d)", (int)err, K, N);
        return false;
    }
    NNOPT_DEBUG_SYNC(queue);
    return true;
}
