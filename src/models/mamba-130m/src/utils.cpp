#include "utils.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used by element_add / pytorch_linear / etc.
#include "kernel_profiler.h"

#include <clblast.h>       // clblast::Gemm — used by pytorch_linear (dtype-templated dispatch).

#include <cstdio>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>

// Forward decl: M=1 GEMV fast path that replaces CLBlast HGemm at decode.
// Lazy-builds kernels/gemv_m1.cl on first call, caches across the process.
// Returns false if the kernel can't be built (caller falls back to CLBlast).
static bool try_gemv_m1_fast_path(cl_command_queue queue,
                                  int N, int K,
                                  cl_mem x, cl_mem W, cl_mem out);

// Same as above, but for the K=1536 fused-residual variant. Reads `hidden`
// for the residual and writes hidden = sum + hidden. Returns false if the
// shape doesn't qualify (K!=1536 || N%4!=0) or the kernel can't be built.
static bool try_gemv_m1_k1536_radd(cl_command_queue queue,
                                   int N,
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
    cl_event* prof_evt = KernelProfiler::event_for("residual_add");
    err = clEnqueueNDRangeKernel(queue, s_cached_kernel, 1, nullptr, &gws, nullptr, 0, nullptr, prof_evt);
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
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("element_add: clCreateKernel failed (%d)", err);
        clReleaseMemObject(out);
        return nullptr;
    }

    int n_int = (int)n;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &out);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
    clSetKernelArg(kernel, 2, sizeof(int), &n_int);

    size_t global_size = n;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
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
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("split_last_dim_2: clCreateKernel failed (%d)", err);
        return false;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &first);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &second);
    clSetKernelArg(kernel, 3, sizeof(int), &rows);
    clSetKernelArg(kernel, 4, sizeof(int), &half_cols);

    size_t global_size = (size_t)rows * (size_t)half_cols;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
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
    // M=1 fast path: replace CLBlast HGemm (M=1 path is a naive single-thread-
    // per-output GEMV with no vec4 / cooperative reduction) with custom
    // gemv_m1 from kernels/gemv_m1.cl. Caller-side eligibility:
    //   - M == 1                                  (decode hot path)
    //   - K == 48                                  → gemv_m1_k48 (dt_proj)
    //   - K >= 64 && K % 64 == 0                   → gemv_m1[/k768/k1536]
    // If gemv_m1 build fails the call returns false and we fall through to
    // CLBlast — never silently regress.
    if (M == 1 && (K == 48 || (K >= 64 && (K % 64) == 0))) {
        if (try_gemv_m1_fast_path(queue, N, K, x, W, out)) {
            return true;
        }
        // Fall through to CLBlast (rare — only if program/kernel build failed).
    }

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
    // CLBlast accepts an output event ptr as its last arg — we wire it up
    // when NNOPT_KERNEL_PROFILE is on, otherwise nullptr (no event capture).
    cl_event* prof_evt = nullptr;
    char prof_label[64];
    if (KernelProfiler::enabled()) {
        std::snprintf(prof_label, sizeof(prof_label), "clblast_gemm_M%d_K%d_N%d", M, K, N);
        prof_evt = KernelProfiler::event_for(prof_label);
    }

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
        prof_evt);
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
        prof_evt);
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

// ──────────────────────────────────────────────────────────────────────
// gemv_m1 fast path
// ──────────────────────────────────────────────────────────────────────
// Lazy-builds kernels/gemv_m1.cl on first call. Same LOAD/STORE/USE_FP16
// macros as the rest of the kernel suite. WG_SIZE = 64 fixed inside the
// kernel; we set local_size = 64 here to match.
//
// We don't piggyback on OpenCLContext::utils_program because pytorch_linear
// has no handle to OpenCLContext. Self-bootstrapping via clGetCommandQueueInfo
// keeps the call sites untouched.

static bool try_gemv_m1_fast_path(cl_command_queue queue,
                                  int N, int K,
                                  cl_mem x, cl_mem W, cl_mem out) {
    static cl_program s_program             = nullptr;
    static cl_kernel  s_kernel_gen          = nullptr;   // generic any-K
    static cl_kernel  s_kernel_48           = nullptr;   // specialized K=48 (dt_proj)
    static cl_kernel  s_kernel_768          = nullptr;   // specialized K=768
    static cl_kernel  s_kernel_1536         = nullptr;   // specialized K=1536
    static cl_kernel  s_kernel_768_no4      = nullptr;   // K=768, 4 outputs per WG
    static cl_kernel  s_kernel_1536_no4     = nullptr;   // K=1536, 4 outputs per WG
    static cl_kernel  s_kernel_768_no8      = nullptr;   // K=768, 8 outputs per WG (preferred when N%8==0)
    static cl_kernel  s_kernel_768_no4_img  = nullptr;   // K=768, no4, image1d_buffer W
    static cl_kernel  s_kernel_1536_no4_img = nullptr;   // K=1536, no4, image1d_buffer W
    static cl_context s_ctx                 = nullptr;   // stashed for image lookup
    static bool       s_failed              = false;     // sticky: don't retry after build failure
    // Per-buffer image1d_buffer_t cache. Keyed on the underlying cl_mem
    // buffer pointer. Same backing memory — clCreateImage(desc.buffer=W)
    // creates a view, no copy. Adreno's texture cache is a separate L1
    // from the buffer cache; using image reads typically yields 1.4-1.7×
    // BW for hot streaming weight reads.
    static std::unordered_map<cl_mem, cl_mem> s_image_cache;
    static size_t     s_image_max_pixels    = 0;         // CL_DEVICE_IMAGE_MAX_BUFFER_SIZE; 0 = unprobed
    static bool       s_image_supported     = true;      // sticky; flips false if any clCreateImage fails

    if (s_failed) return false;

    if (!s_kernel_gen) {
        cl_int err = CL_SUCCESS;
        cl_context ctx = nullptr;
        clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);
        cl_device_id dev = nullptr;
        clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr);
        if (!ctx || !dev) {
            NNOPT_ERROR("gemv_m1: failed to read queue context/device");
            s_failed = true;
            return false;
        }

        std::ifstream f("kernels/gemv_m1.cl");
        if (!f.is_open()) {
            NNOPT_ERROR("gemv_m1: cannot open kernels/gemv_m1.cl");
            s_failed = true;
            return false;
        }
        std::stringstream ss; ss << f.rdbuf();
        std::string src = ss.str();
        const char* src_ptr = src.c_str();
        size_t src_len = src.size();
        s_program = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS || !s_program) {
            NNOPT_ERROR_FMT("gemv_m1: clCreateProgramWithSource failed: %d", (int)err);
            s_failed = true;
            return false;
        }

        // Only enable the image-kernel section of the .cl source if env opt-in.
        // Keeping the image kernels out of the OpenCL program when not used
        // avoids any cross-kernel optimization side effects (Adreno's compiler
        // changes register allocation for the buffer kernels when image
        // intrinsics are present in the same program).
        const char* env_use_image_build = std::getenv("NNOPT_GEMV_USE_IMAGE");
        const bool image_at_build = env_use_image_build && env_use_image_build[0] != '0';
        std::string opts_str;
#ifdef NNOPT_USE_FP16
        opts_str = "-DUSE_FP16=1";
#endif
        if (image_at_build) {
            if (!opts_str.empty()) opts_str += " ";
            opts_str += "-DENABLE_IMAGE_KERNELS=1";
        }
        const char* opts = opts_str.c_str();
        err = clBuildProgram(s_program, 1, &dev, opts, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(s_program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size + 1, 0);
            clGetProgramBuildInfo(s_program, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            NNOPT_ERROR_FMT("gemv_m1: clBuildProgram failed: %d\n%s", (int)err, log.data());
            clReleaseProgram(s_program);
            s_program = nullptr;
            s_failed = true;
            return false;
        }

        s_kernel_gen      = clCreateKernel(s_program, "gemv_m1",           &err);
        s_kernel_48       = clCreateKernel(s_program, "gemv_m1_k48",       &err);
        s_kernel_768      = clCreateKernel(s_program, "gemv_m1_k768",      &err);
        s_kernel_1536     = clCreateKernel(s_program, "gemv_m1_k1536",     &err);
        s_kernel_768_no4  = clCreateKernel(s_program, "gemv_m1_k768_no4",  &err);
        s_kernel_1536_no4 = clCreateKernel(s_program, "gemv_m1_k1536_no4", &err);
        s_kernel_768_no8  = clCreateKernel(s_program, "gemv_m1_k768_no8",  &err);
        if (err != CL_SUCCESS || !s_kernel_gen || !s_kernel_48 || !s_kernel_768 ||
            !s_kernel_1536 || !s_kernel_768_no4 || !s_kernel_1536_no4 || !s_kernel_768_no8) {
            NNOPT_ERROR_FMT("gemv_m1: clCreateKernel failed: %d", (int)err);
            if (s_kernel_gen)      clReleaseKernel(s_kernel_gen);
            if (s_kernel_48)       clReleaseKernel(s_kernel_48);
            if (s_kernel_768)      clReleaseKernel(s_kernel_768);
            if (s_kernel_1536)     clReleaseKernel(s_kernel_1536);
            if (s_kernel_768_no4)  clReleaseKernel(s_kernel_768_no4);
            if (s_kernel_1536_no4) clReleaseKernel(s_kernel_1536_no4);
            if (s_kernel_768_no8)  clReleaseKernel(s_kernel_768_no8);
            s_kernel_gen = s_kernel_48 = s_kernel_768 = s_kernel_1536 = nullptr;
            s_kernel_768_no4 = s_kernel_1536_no4 = s_kernel_768_no8 = nullptr;
            clReleaseProgram(s_program);
            s_program = nullptr;
            s_failed = true;
            return false;
        }

        // Image variants — opt-in via env NNOPT_GEMV_USE_IMAGE=1. Default
        // off because (a) on Adreno 620 the texture path was neutral-to-
        // slightly-slower vs the no4 buffer path in measurement, and
        // (b) building the image kernels alongside the buffer kernels
        // was empirically observed to slow the buffer kernels (~5%).
        // Hypothesis: the OpenCL program-wide optimizer changes register
        // allocation or scheduling for the buffer kernels when image
        // intrinsics are also present. Keep code paths intact for a
        // future driver re-evaluation.
        const char* env_use_image = std::getenv("NNOPT_GEMV_USE_IMAGE");
        const bool   image_opt_in = env_use_image && env_use_image[0] != '0';
        cl_int img_err = CL_SUCCESS;
        if (image_opt_in) {
            s_kernel_768_no4_img  = clCreateKernel(s_program, "gemv_m1_k768_no4_img",  &img_err);
            if (img_err != CL_SUCCESS) {
                NNOPT_ERROR_FMT("gemv_m1: image kernel build failed (img path disabled): %d", (int)img_err);
                s_image_supported = false;
                s_kernel_768_no4_img = nullptr;
            }
            if (s_image_supported) {
                s_kernel_1536_no4_img = clCreateKernel(s_program, "gemv_m1_k1536_no4_img", &img_err);
                if (img_err != CL_SUCCESS) {
                    NNOPT_ERROR_FMT("gemv_m1: image kernel build failed (img path disabled): %d", (int)img_err);
                    s_image_supported = false;
                    if (s_kernel_768_no4_img)  clReleaseKernel(s_kernel_768_no4_img);
                    s_kernel_768_no4_img = s_kernel_1536_no4_img = nullptr;
                }
            }
            s_ctx = ctx;
            if (s_image_supported) {
                cl_int qerr = clGetDeviceInfo(dev, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
                                              sizeof(s_image_max_pixels), &s_image_max_pixels, nullptr);
                if (qerr != CL_SUCCESS) s_image_max_pixels = 0;
            }
        } else {
            s_image_supported = false;  // gate dispatch off
        }
    }

    // Pick the most-specialized kernel for this K. Specializations have
    // hard-unrolled inner loops (clang only fully unrolls the dot-chain when
    // the loop bound is a compile-time constant) — generic fallback handles
    // the rest at slightly lower throughput.
    //
    // Dispatch shape:
    //   gemv_m1_k{768,1536}_no4  → cooperative WG=64, 4 outputs per WG.
    //                              gws = (N/4) * 64, lws = 64. Preferred when
    //                              N%4==0 — shares x reads across 4 outputs,
    //                              4× fewer WGs, register-level parallelism on
    //                              4 dot-product chains.
    //   gemv_m1 / gemv_m1_k768 / gemv_m1_k1536  → cooperative WG=64, 1 output
    //                                              per WG. gws = N * 64, lws = 64.
    //   gemv_m1_k48                             → 1 thread per output (K too
    //                                              small to make cooperative
    //                                              reduction worthwhile).
    //                                              gws = N (no lws).
    enum DispatchMode { COOP_1, COOP_4, COOP_8, NONCOOP };
    cl_kernel    k    = nullptr;
    DispatchMode mode = COOP_1;
    // Empirical (Adreno 620): no8 K=768 is a 9% REGRESSION vs no4 (22.4 vs
    // 24.6 tok/s). Likely register pressure (8 fp32 acc + 8 float4 W + 1 float4
    // x ≈ 37 32-bit registers per thread) forcing spills to local memory.
    // no8 kernel kept in .cl for documentation; dispatch never selects it
    // (opt-in via env NNOPT_GEMV_USE_NO8=1 for future re-evaluation).
    static const bool s_use_no8 = []() {
        const char* e = std::getenv("NNOPT_GEMV_USE_NO8");
        return e && e[0] != '0';
    }();
    if (K == 48)                                       { k = s_kernel_48;       mode = NONCOOP; }
    else if (K == 768  && s_use_no8 && (N % 8) == 0)   { k = s_kernel_768_no8;  mode = COOP_8;  }
    else if (K == 768  && (N % 4) == 0)                { k = s_kernel_768_no4;  mode = COOP_4;  }
    else if (K == 1536 && (N % 4) == 0)                { k = s_kernel_1536_no4; mode = COOP_4;  }
    else if (K == 768)                                 { k = s_kernel_768;      mode = COOP_1;  }
    else if (K == 1536)                                { k = s_kernel_1536;     mode = COOP_1;  }
    else                                               { k = s_kernel_gen;      mode = COOP_1;  }

    // Texture-cache fast path: when in COOP_4 with K=768 or K=1536, try to
    // wrap W as an image1d_buffer_t and dispatch the _img variant. If the
    // image creation fails (size too big, driver quirk, etc.) silently fall
    // back to the buffer kernel.
    //
    // Empirical note (Adreno 620, Snapdragon 765G): the image1d_buffer path
    // is ESSENTIALLY NEUTRAL vs the buffer no4 path here — the no4 buffer
    // kernel is already so well coalesced (wave-stride 64-byte access) that
    // Adreno's texture L1 doesn't beat the buffer L1. Default-off; opt in
    // via env var NNOPT_GEMV_USE_IMAGE=1 for diagnosis or future-driver
    // re-evaluation.
    static const bool s_use_image = []() {
        const char* e = std::getenv("NNOPT_GEMV_USE_IMAGE");
        return e && e[0] != '0';
    }();
    cl_mem W_img = nullptr;
    if (s_use_image && mode == COOP_4 && s_image_supported &&
        (k == s_kernel_768_no4 || k == s_kernel_1536_no4)) {
        const size_t needed_pixels = (size_t)N * (size_t)K / 4;
        if (s_image_max_pixels == 0 || needed_pixels <= s_image_max_pixels) {
            auto it = s_image_cache.find(W);
            if (it != s_image_cache.end()) {
                W_img = it->second;
            } else {
                cl_image_format fmt;
                fmt.image_channel_order     = CL_RGBA;
                fmt.image_channel_data_type = CL_HALF_FLOAT;
                cl_image_desc desc;
                std::memset(&desc, 0, sizeof(desc));
                desc.image_type   = CL_MEM_OBJECT_IMAGE1D_BUFFER;
                desc.image_width  = needed_pixels;
                desc.buffer       = W;
                cl_int ierr = CL_SUCCESS;
                W_img = clCreateImage(s_ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &ierr);
                if (ierr == CL_SUCCESS && W_img) {
                    s_image_cache[W] = W_img;
                } else {
                    NNOPT_ERROR_FMT("gemv_m1: clCreateImage1dBuffer failed: %d (N=%d K=%d, needed=%zu, max=%zu) — falling back to buffer path",
                                    (int)ierr, N, K, needed_pixels, s_image_max_pixels);
                    W_img = nullptr;
                    // Don't disable globally — this size may exceed device limit but other weights still work.
                }
            }
        }
        if (W_img) {
            k = (K == 768) ? s_kernel_768_no4_img : s_kernel_1536_no4_img;
        }
    }

    cl_int err = clSetKernelArg(k, 0, sizeof(cl_mem), &x);
    if (err == CL_SUCCESS) {
        if (W_img) err = clSetKernelArg(k, 1, sizeof(cl_mem), &W_img);
        else       err = clSetKernelArg(k, 1, sizeof(cl_mem), &W);
    }
    if (err == CL_SUCCESS) err = clSetKernelArg(k, 2, sizeof(cl_mem), &out);
    if (k == s_kernel_gen) {
        if (err == CL_SUCCESS) err = clSetKernelArg(k, 3, sizeof(int), &K);
        if (err == CL_SUCCESS) err = clSetKernelArg(k, 4, sizeof(int), &N);
    } else {
        if (err == CL_SUCCESS) err = clSetKernelArg(k, 3, sizeof(int), &N);
    }
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1: setKernelArg failed: %d", (int)err);
        return false;
    }

    cl_event* prof_evt = nullptr;
    char prof_label[64];
    if (KernelProfiler::enabled()) {
        const char* tag = W_img ? "_img"
                                : (mode == COOP_8 ? "_no8"
                                : (mode == COOP_4 ? "_no4" : ""));
        std::snprintf(prof_label, sizeof(prof_label), "gemv_m1_K%d_N%d%s", K, N, tag);
        prof_evt = KernelProfiler::event_for(prof_label);
    }

    if (mode == COOP_8) {
        const size_t WG = 64;
        size_t gws = (size_t)(N / 8) * WG;
        size_t lws = WG;
        err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, prof_evt);
    } else if (mode == COOP_4) {
        const size_t WG = 64;
        size_t gws = (size_t)(N / 4) * WG;
        size_t lws = WG;
        err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, prof_evt);
    } else if (mode == COOP_1) {
        const size_t WG = 64;
        size_t gws = (size_t)N * WG;
        size_t lws = WG;
        err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, prof_evt);
    } else {
        size_t gws = (size_t)N;
        err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, nullptr, 0, nullptr, prof_evt);
    }
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1: enqueue failed: %d (N=%d K=%d)", (int)err, N, K);
        return false;
    }
    NNOPT_DEBUG_SYNC(queue);
    return true;
}

// ──────────────────────────────────────────────────────────────────────
// Fused gemv_m1 + residual_add (K=1536 only)
// ──────────────────────────────────────────────────────────────────────
// Reuses the kernel program and lazy-build path of try_gemv_m1_fast_path
// — calls it first as a no-op build trigger, then dispatches its own
// _radd variant. Caches the kernel handle in a static across calls.
static bool try_gemv_m1_k1536_radd(cl_command_queue queue,
                                   int N,
                                   cl_mem x, cl_mem W, cl_mem hidden) {
    static cl_kernel s_kernel_radd = nullptr;
    static bool      s_failed      = false;

    if (s_failed) return false;
    if ((N % 4) != 0) return false;

    if (!s_kernel_radd) {
        // Trigger lazy program build by calling try_gemv_m1_fast_path with a
        // sentinel write that we then immediately overwrite. Cleaner: call
        // it once with a real K=1536 dispatch (doing the work twice is fine
        // on first call). Simpler still: read the program out via clGetKernelInfo.
        // Easiest: just build the kernel by hand via the same path as try_gemv_m1.
        // We piggyback on the static program already built by try_gemv_m1_fast_path.
        // To get a kernel for "gemv_m1_k1536_no4_radd" we need access to s_program
        // from that function. Make sure try_gemv_m1_fast_path has been called
        // at least once for K=1536. If not, kick a fake build:
        cl_mem dummy = hidden;  // valid cl_mem
        // First-call bootstrap: dispatch the existing fused path (no-op) just
        // to ensure the program is compiled and cached. We can't access the
        // static s_program from here, so we cl_program-build a fresh one.
        cl_int err = CL_SUCCESS;
        cl_context ctx = nullptr;
        clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);
        cl_device_id dev = nullptr;
        clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr);

        std::ifstream f("kernels/gemv_m1.cl");
        if (!f.is_open()) {
            NNOPT_ERROR("gemv_m1_radd: cannot open kernels/gemv_m1.cl");
            s_failed = true;
            return false;
        }
        std::stringstream ss; ss << f.rdbuf();
        std::string src = ss.str();
        const char* src_ptr = src.c_str();
        size_t src_len = src.size();
        cl_program prog = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("gemv_m1_radd: clCreateProgramWithSource failed: %d", (int)err);
            s_failed = true;
            return false;
        }
#ifdef NNOPT_USE_FP16
        const char* opts = "-DUSE_FP16=1";
#else
        const char* opts = "";
#endif
        err = clBuildProgram(prog, 1, &dev, opts, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size + 1, 0);
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            NNOPT_ERROR_FMT("gemv_m1_radd: clBuildProgram failed: %d\n%s", (int)err, log.data());
            clReleaseProgram(prog);
            s_failed = true;
            return false;
        }
        s_kernel_radd = clCreateKernel(prog, "gemv_m1_k1536_no4_radd", &err);
        clReleaseProgram(prog);  // kernel retains program
        if (err != CL_SUCCESS || !s_kernel_radd) {
            NNOPT_ERROR_FMT("gemv_m1_radd: clCreateKernel failed: %d", (int)err);
            s_failed = true;
            return false;
        }
        (void)dummy;
    }

    cl_int err = clSetKernelArg(s_kernel_radd, 0, sizeof(cl_mem), &x);
    if (err == CL_SUCCESS) err = clSetKernelArg(s_kernel_radd, 1, sizeof(cl_mem), &W);
    if (err == CL_SUCCESS) err = clSetKernelArg(s_kernel_radd, 2, sizeof(cl_mem), &hidden);
    if (err == CL_SUCCESS) err = clSetKernelArg(s_kernel_radd, 3, sizeof(int),    &N);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_radd: setKernelArg failed: %d", (int)err);
        return false;
    }

    cl_event* prof_evt = nullptr;
    char prof_label[64];
    if (KernelProfiler::enabled()) {
        std::snprintf(prof_label, sizeof(prof_label), "gemv_m1_K1536_N%d_no4_radd", N);
        prof_evt = KernelProfiler::event_for(prof_label);
    }

    const size_t WG = 64;
    size_t gws = (size_t)(N / 4) * WG;
    size_t lws = WG;
    err = clEnqueueNDRangeKernel(queue, s_kernel_radd, 1, nullptr, &gws, &lws, 0, nullptr, prof_evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_radd: enqueue failed: %d (N=%d)", (int)err, N);
        return false;
    }
    NNOPT_DEBUG_SYNC(queue);
    return true;
}

bool pytorch_linear_radd(cl_command_queue queue,
                         int M, int N, int K,
                         cl_mem x, cl_mem W, cl_mem hidden) {
    if (M == 1 && K == 1536 && (N % 4) == 0) {
        if (try_gemv_m1_k1536_radd(queue, N, x, W, hidden)) {
            return true;
        }
    }
    return false;  // caller falls back to pytorch_linear + element_add_inplace
}
