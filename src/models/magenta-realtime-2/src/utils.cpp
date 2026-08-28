#include <cerrno>
#include <unistd.h>
#include <sys/resource.h>
#include <sched.h>
#include "utils.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used by element_add / pytorch_linear / etc.
#include "profiler.h"      // KernelProfiler::event_for — dormant unless NNOPT_PROFILE=1.

#include <clblast.h>       // clblast::Gemm — used by pytorch_linear (dtype-templated dispatch).

#include <fstream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cstdint>

// ──────────────────────────────────────────────
// set_arg_checked — public clSetKernelArg wrapper.
// One definition used by every op (src/ops/*.cpp). Returns false on failure
// after surfacing a descriptive NNOPT_ERROR_FMT (kernel name slot, arg index,
// CL error code). Model-agnostic.
// ──────────────────────────────────────────────
bool set_arg_checked(cl_kernel kernel,
                     unsigned int arg_index,
                     size_t arg_size,
                     const void* arg_value,
                     const char* arg_name) {
    const cl_int err = clSetKernelArg(kernel, arg_index, arg_size, arg_value);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("clSetKernelArg(%s) idx=%u failed: %d",
                        arg_name ? arg_name : "(unnamed)",
                        (unsigned)arg_index,
                        (int)err);
        return false;
    }
    return true;
}

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
    err = clEnqueueNDRangeKernel(queue, s_cached_kernel, 1, nullptr, &gws, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("util_element_add_inplace"));
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
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("util_element_add"));
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
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("util_split_last_dim_2"));
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

// CLBlast status code → human-readable name + meaning. Generated from
// /Users/<...>/.nnopt/deps/clblast/include/clblast.h:55-120 (kept in sync
// with the CLBlast header we link against). Keeps the failure message
// actionable: "status=-1010" tells nobody anything, but
// "kInsufficientMemoryB (Matrix B's OpenCL buffer is too small)" plus the
// requested M/N/K and actual buffer size lets the agent identify the
// caller-vs-weight dim mismatch on the FIRST failing cycle.
static const char* nnopt_clblast_status_name(int s) {
    switch (s) {
        case 0:     return "kSuccess";
        // OpenCL-passthrough errors (most useful: -38, -46, -54, -55).
        case -11:   return "kOpenCLBuildProgramFailure (OpenCL kernel compile failed)";
        case -30:   return "kInvalidValue (CL_INVALID_VALUE)";
        case -36:   return "kInvalidCommandQueue (CL_INVALID_COMMAND_QUEUE)";
        case -38:   return "kInvalidMemObject (CL_INVALID_MEM_OBJECT — buffer is null or released)";
        case -42:   return "kInvalidBinary";
        case -43:   return "kInvalidBuildOptions";
        case -44:   return "kInvalidProgram";
        case -45:   return "kInvalidProgramExecutable";
        case -46:   return "kInvalidKernelName (CL_INVALID_KERNEL_NAME — host name doesn't match a __kernel in the .cl)";
        case -47:   return "kInvalidKernelDefinition";
        case -48:   return "kInvalidKernel";
        case -49:   return "kInvalidArgIndex";
        case -50:   return "kInvalidArgValue";
        case -51:   return "kInvalidArgSize";
        case -52:   return "kInvalidKernelArgs (CL_INVALID_KERNEL_ARGS — some kernel arg unset)";
        case -53:   return "kInvalidLocalNumDimensions (CL_INVALID_WORK_DIMENSION)";
        case -54:   return "kInvalidLocalThreadsTotal (CL_INVALID_WORK_GROUP_SIZE — local size exceeds device limit)";
        case -55:   return "kInvalidLocalThreadsDim (CL_INVALID_WORK_ITEM_SIZE — per-dim local size exceeds device limit)";
        case -56:   return "kInvalidGlobalOffset";
        case -57:   return "kInvalidEventWaitList";
        case -58:   return "kInvalidEvent";
        case -59:   return "kInvalidOperation (CL_INVALID_OPERATION)";
        case -61:   return "kInvalidBufferSize";
        case -63:   return "kInvalidGlobalWorkSize";
        // CLBlast / clBLAS shared error codes — matrix/dim/buffer issues.
        case -1024: return "kNotImplemented";
        case -1022: return "kInvalidMatrixA (Matrix A is not a valid OpenCL buffer)";
        case -1021: return "kInvalidMatrixB (Matrix B is not a valid OpenCL buffer)";
        case -1020: return "kInvalidMatrixC (Matrix C is not a valid OpenCL buffer)";
        case -1019: return "kInvalidVectorX";
        case -1018: return "kInvalidVectorY";
        case -1017: return "kInvalidDimension (M, N, K must be > 0)";
        case -1016: return "kInvalidLeadDimA (lda smaller than A's first dimension)";
        case -1015: return "kInvalidLeadDimB (ldb smaller than B's first dimension)";
        case -1014: return "kInvalidLeadDimC (ldc smaller than C's first dimension)";
        case -1013: return "kInvalidIncrementX";
        case -1012: return "kInvalidIncrementY";
        case -1011: return "kInsufficientMemoryA (Matrix A's OpenCL buffer is too small for the requested GEMM)";
        case -1010: return "kInsufficientMemoryB (Matrix B's OpenCL buffer is too small for the requested GEMM — check that the WEIGHT shape on disk matches the (N, K) the caller is passing)";
        case -1009: return "kInsufficientMemoryC (Matrix C's OpenCL buffer is too small)";
        case -1008: return "kInsufficientMemoryX";
        case -1007: return "kInsufficientMemoryY";
        // CLBlast-specific.
        case -2050: return "kInsufficientMemoryTemp";
        case -2049: return "kInvalidBatchCount";
        case -2048: return "kInvalidOverrideKernel";
        case -2047: return "kMissingOverrideParameter";
        case -2046: return "kInvalidLocalMemUsage (not enough device local memory)";
        case -2045: return "kNoHalfPrecision (device doesn't support fp16)";
        case -2044: return "kNoDoublePrecision";
        case -2043: return "kInvalidVectorScalar";
        case -2042: return "kInsufficientMemoryScalar";
        case -2041: return "kDatabaseError";
        case -2040: return "kUnknownError";
        case -2039: return "kUnexpectedError";
        default:    return "<unrecognized CLBlast status — see clblast.h>";
    }
}

// Query the actual byte-size of an OpenCL buffer object. Used to enrich
// CLBlast failure messages with what the buffer ACTUALLY holds vs what the
// GEMM dimensions require — turns "status=-1010" into "buffer is X bytes,
// GEMM needs Y bytes (off by Z)" which the agent can act on.
static size_t nnopt_cl_mem_size_bytes(cl_mem m) {
    if (!m) return 0;
    size_t sz = 0;
    cl_int e = clGetMemObjectInfo(m, CL_MEM_SIZE, sizeof(sz), &sz, nullptr);
    return (e == CL_SUCCESS) ? sz : 0;
}

bool pytorch_linear(cl_command_queue queue,
                    int M, int N, int K,
                    cl_mem x, cl_mem W, cl_mem out) {
    // out[M, N] = x[M, K] @ W[N, K]^T  where W is nn.Linear weight [N, K].
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
        // Enrich the failure message with: (1) the CLBlast status name,
        // (2) the requested M/N/K, (3) the actual buffer sizes for x/W/out
        // alongside what the GEMM dimensions imply they should be. This
        // turns an opaque "status=-1010" into a one-line diagnosis the
        // agent can act on without needing to look up the CLBlast header.
        const size_t elem = sizeof(nnopt_storage_t);
        const size_t x_bytes_actual = nnopt_cl_mem_size_bytes(x);
        const size_t W_bytes_actual = nnopt_cl_mem_size_bytes(W);
        const size_t out_bytes_actual = nnopt_cl_mem_size_bytes(out);
        const size_t x_bytes_need = (size_t)M * (size_t)K * elem;
        const size_t W_bytes_need = (size_t)N * (size_t)K * elem;
        const size_t out_bytes_need = (size_t)M * (size_t)N * elem;
        // Compute on-disk weight dim implied by the buffer (W is fp16; rows
        // count = bytes / (cols * elem). When cols == K, rows == intermediate
        // for nn.Linear). This lets the agent compare actual-vs-expected N at
        // a glance instead of dividing in their head.
        const long long W_rows_actual = (K > 0) ? (long long)(W_bytes_actual / ((size_t)K * elem)) : -1;
        NNOPT_ERROR_FMT(
            "pytorch_linear: CLBlast Gemm failed status=%d %s | M=%d N=%d K=%d | "
            "x=%zub need=%zub (M*K*elem) | "
            "W=%zub need=%zub (N*K*elem; weight stored as [N,K] for nn.Linear; rows-implied N=%lld) | "
            "out=%zub need=%zub. "
            "DIAGNOSTIC FLOW (no prescription — diagnose the root): "
            "(1) Read .nnport/model_info.json::tensor_shapes for this layer's weight key (whichever the caller passed to op_Linear). "
            "(2) Compare the on-disk shape against the M/N/K above to find which dim disagrees. "
            "(3) Consult .nnport/dimensions_audit.json for the MODEL_CONFIG constant feeding the disagreeing dim — "
            "if source=weights and consensus matches the on-disk shape, the bug is NOT the constant; the bug is "
            "wrong weight_key / wrong layer_idx / merged-projection assumed split (e.g. Phi-3 gate_up_proj has "
            "rows=2*intermediate) / transposed layout. "
            "(4) If audit source=config (no evidence keys matched), the family's weight-key layout is novel — "
            "add an entry to dimensionsAudit.ts::DIM_EVIDENCE and re-port.",
            (int)status, nnopt_clblast_status_name((int)status),
            M, N, K,
            x_bytes_actual, x_bytes_need,
            W_bytes_actual, W_bytes_need, W_rows_actual,
            out_bytes_actual, out_bytes_need);
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
        // Same enrichment pattern as pytorch_linear, with conv1d's weight
        // layout: HF Conv1D stores W as [K, N] (in, out), so W's required
        // size is K*N*elem (NOT N*K*elem like nn.Linear).
        const size_t elem = sizeof(nnopt_storage_t);
        const size_t x_bytes_actual = nnopt_cl_mem_size_bytes(x);
        const size_t W_bytes_actual = nnopt_cl_mem_size_bytes(W);
        const size_t out_bytes_actual = nnopt_cl_mem_size_bytes(out);
        const size_t x_bytes_need = (size_t)M * (size_t)K * elem;
        const size_t W_bytes_need = (size_t)K * (size_t)N * elem;
        const size_t out_bytes_need = (size_t)M * (size_t)N * elem;
        NNOPT_ERROR_FMT(
            "pytorch_conv1d: CLBlast Gemm failed status=%d %s | M=%d N=%d K=%d | "
            "x=%zub need=%zub (M*K*elem) | "
            "W=%zub need=%zub (K*N*elem; HF Conv1D stores [K,N] = [in,out]) | "
            "out=%zub need=%zub. "
            "If a buffer is SMALLER than 'need', the caller's dim disagrees with the actual weight shape — "
            "look up the weight's shape in weights/model.meta.json and fix the caller's MODEL_CONFIG value.",
            (int)status, nnopt_clblast_status_name((int)status),
            M, N, K,
            x_bytes_actual, x_bytes_need,
            W_bytes_actual, W_bytes_need,
            out_bytes_actual, out_bytes_need);
        return false;
    }
    NNOPT_DEBUG_SYNC(queue);
    return true;
}

// ── nnopt_gemv — see utils.h. fp16 and q4 share one dispatch shape. ──────────
#include "opencl_context.h"
#include "weights.h"
#include <map>
#include <vector>
#include <cmath>
#include <chrono>

// nnopt_gemv routes its int8-eligible shapes into nnopt_gemv_fused (the fused kernel is the only
// one with an INT8 specialization, and a plain GEMV is just that kernel with PRENORM/GELU off).
// Both directions have fallbacks, so without this flag the two would bounce a shape between them
// forever. Set only around the delegating call; nnopt_gemv_fused reads it and returns null instead
// of calling back.
// See utils.h: ONE reader, because backbone and MLP_forward must agree or the norm disappears.
// Level 1 (GELU epilogue only) is the default: folding the pre-norm as well removes 36 dispatches
// per frame but measured +47% on dense1, which is the larger number by far.
// ── CPU affinity: put the dispatch thread on a performance core ─────────────────────────────────
// The AR spends 63 us of CPU inside every clEnqueueNDRangeKernel — 5-10x what a dispatch should
// cost — and the GPU sits 61% idle waiting for them. That cost is the vendor driver's command
// building, which is ordinary CPU work, so WHICH CORE runs it matters.
//
// The engine is a subprocess the app spawns, and Android schedules spawned/background processes on
// efficiency cores. Nothing in the engine or the app had ever set an affinity or a priority.
//
// Two things get logged rather than assumed:
//   * the mask BEFORE — if it already excludes the big cores, we are inside a restricted cpuset and
//     sched_setaffinity cannot escape it. That is an app-side fix (spawn in a foreground cgroup),
//     and the log says so instead of silently doing nothing.
//   * whether the call succeeded, and the niceness, which an app process usually may not lower.
//
// NNOPT_CPUPIN=0 disables.
void nnopt_pin_perf_cores(const char* who) {
    static int on = -1;
    // OFF by default. The affinity idea was wrong for this chip (every core is fast, so narrowing
    // the mask can only remove options) and the priority raise could not be shown to help against
    // the session's drift. It stays behind NNOPT_CPUPIN=1 rather than shipping unverified.
    if (on < 0) { const char* e = std::getenv("NNOPT_CPUPIN"); on = (e && e[0] == '1') ? 1 : 0; }
    if (!on) return;

    const int ncpu = (int)sysconf(_SC_NPROCESSORS_CONF);
    if (ncpu <= 1) return;
    std::vector<long> khz((size_t)ncpu, 0);
    long best = 0;
    for (int i = 0; i < ncpu; ++i) {
        char path[128];
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        if (FILE* f = std::fopen(path, "r")) {
            long v = 0; if (std::fscanf(f, "%ld", &v) == 1) khz[(size_t)i] = v;
            std::fclose(f);
        }
        if (khz[(size_t)i] > best) best = khz[(size_t)i];
    }

    cpu_set_t before; CPU_ZERO(&before);
    const bool got_before = (sched_getaffinity(0, sizeof(before), &before) == 0);

    // "Big" = at least 70% of the fastest core, NOT within 10% of it.
    //
    // The 10% rule was wrong and measurably harmful. This is an 8-Elite-class part: 2 prime cores at
    // 4.7 GHz plus 6 performance cores, and NO little cores. "Within 10% of max" selected the two
    // prime cores only, so it took a thread that had 8 cores and confined it to 2 — sharing them
    // with the codec thread and the app's UI thread. A restriction dressed up as an optimisation.
    //
    // 70% keeps every genuinely fast core and excludes only a true efficiency cluster (typically
    // ~50% of prime). And when every core qualifies, we do not call setaffinity AT ALL: there is
    // nothing to gain from narrowing a mask that is already the whole machine, and narrowing it can
    // only take scheduling options away.
    cpu_set_t want; CPU_ZERO(&want);
    int n_want = 0, n_avail = 0;
    for (int i = 0; i < ncpu; ++i) {
        if (got_before && !CPU_ISSET(i, &before)) continue;   // outside our cpuset — not offerable
        ++n_avail;
        if (best > 0 && khz[(size_t)i] * 10 >= best * 7) { CPU_SET(i, &want); ++n_want; }
    }
    int rc = 0;
    const char* why = "pinned";
    if (n_want == 0) {
        why = "no core above the 70% threshold — leaving the mask alone";
    } else if (n_want == n_avail) {
        why = "every available core is already a fast one — leaving the mask alone";
    } else {
        rc = sched_setaffinity(0, sizeof(want), &want);
    }
    errno = 0;
    const int old_nice = getpriority(PRIO_PROCESS, 0);
    const int prc = setpriority(PRIO_PROCESS, 0, -10);
    const int new_nice = getpriority(PRIO_PROCESS, 0);

    char blist[64] = {0}, wlist[64] = {0};
    for (int i = 0; i < ncpu && i < 16; ++i) {
        char b[4]; std::snprintf(b, sizeof(b), "%d", (got_before && CPU_ISSET(i,&before)) ? 1 : 0);
        std::strncat(blist, b, sizeof(blist)-std::strlen(blist)-1);
        std::snprintf(b, sizeof(b), "%d", CPU_ISSET(i,&want) ? 1 : 0);
        std::strncat(wlist, b, sizeof(wlist)-std::strlen(wlist)-1);
    }
    std::fprintf(stderr, "CPUPIN %s: cpus=%d maxfreq=%ldMHz before=[%s] want=[%s] %s "
                         "setaffinity=%d nice %d->%d (setpriority=%d)\n",
                 who, ncpu, best/1000, blist, wlist, why, rc, old_nice, new_nice, prc);
    std::fflush(stderr);
}

// Codec GPU span for the last render, in ms. Reported next to the AR's so "GPU busy" finally
// counts both queues instead of only the one the AR runs on.
namespace { double g_codec_gpu_ms = 0.0; }
void   nnopt_set_codec_gpu_ms(double ms) { g_codec_gpu_ms = ms; }
double nnopt_codec_gpu_ms()              { return g_codec_gpu_ms; }

static int    g_codec_fp16_force = -1;
static double g_codec_ab_cos = 0.0, g_codec_ab_err = 0.0;
static int    g_codec_ab_n = 0;

void nnopt_codec_fp16_force(int mode) { g_codec_fp16_force = mode; }
int  nnopt_codec_fp16_forced()        { return g_codec_fp16_force; }

void nnopt_set_codec_ab(double cosine, double max_abs_err, int n_chunks) {
    g_codec_ab_cos = cosine; g_codec_ab_err = max_abs_err; g_codec_ab_n = n_chunks;
}
void nnopt_get_codec_ab(double* cosine, double* max_abs_err, int* n_chunks) {
    if (cosine)      *cosine      = g_codec_ab_cos;
    if (max_abs_err) *max_abs_err = g_codec_ab_err;
    if (n_chunks)    *n_chunks    = g_codec_ab_n;
}

static int g_cfp16 = 0, g_cfp16_skip = 0, g_cfp16_fail = 0;
void nnopt_codec_fp16_tally(int fp16, int skipped, int failed) {
    g_cfp16 += fp16; g_cfp16_skip += skipped; g_cfp16_fail += failed;
}
void nnopt_codec_fp16_get(int* fp16, int* skipped, int* failed) {
    if (fp16)    *fp16    = g_cfp16;
    if (skipped) *skipped = g_cfp16_skip;
    if (failed)  *failed  = g_cfp16_fail;
}

void nnopt_codec_stats_reset() {
    g_codec_ab_cos = g_codec_ab_err = 0.0; g_codec_ab_n = 0;
    g_cfp16 = g_cfp16_skip = g_cfp16_fail = 0;
    g_codec_gpu_ms = 0.0;
}

int nnopt_fuse_level() {
    static int v = -1, ep = -1;
    if (ep != nnopt_toggle_epoch()) {
        // 2 = fold the MLP pre-norm only. Level 3 also folds the ATTENTION pre-norms, which
        // removes 48 dispatches a frame and MEASURED SLOWER on the 840 (AR 2.907 -> 3.143): the
        // PRENORM variant makes each of those 48 GEMVs read a scale vector and run a second
        // reduction, and that cost more than the 48 dispatches it saved. Kept behind level 3
        // because that verdict flips the moment the per-dispatch host cost comes down.
        // Default 2. Folding the pre-norm into dense1 costs a little GPU time (+2.4 ms/frame
        // measured on the 620) and removes 36 dispatches per frame. That was a bad trade while the
        // AR was GPU-bound; it is a good one now that it measures 94% HOST-bound at ~40 us of host
        // cost per dispatch. Proven bit-identical output on the 620 (rms 3682.7, peak 28545 both ways).
        const char* e = std::getenv("NNOPT_FUSE");
        v = e ? std::atoi(e) : 2;
        ep = nnopt_toggle_epoch();
    }
    return v;
}

static bool g_gemv_delegating = false;
static bool nnopt_int8_eligible(const std::string& k);
static int  nnopt_int8_scope();
static cl_command_queue nnopt_live_queue();

cl_mem nnopt_gemv(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                  cl_mem x, const std::string& w_key,
                  int rows, int in_dim, int out_dim, cl_mem bias) {
    cl_mem W = weights.get_buffer(w_key);
    if (!W) { NNOPT_ERROR_FMT("nnopt_gemv: missing %s", w_key.c_str()); return nullptr; }
    // Built ONCE. This used to be concatenated twice per call — here and again for the q4 check
    // below — which is two heap allocations per GEMV, 192 GEMVs a frame, 19,200 a render. Not the
    // reason the AR is slow (see HOST_COST), but there is no argument for paying it.
    const std::string s_key = w_key + ".scale";
    const bool q4 = weights.has_tensor(s_key);
    // int8 first: the fused kernel's INT8 specialization halves the weight traffic, and this call
    // is bandwidth-bound on exactly that operand. A q4 bundle already ships quantized weights and
    // keeps its own path below.
    if (!q4 && !g_gemv_delegating &&
        (in_dim & 3) == 0 && nnopt_int8_eligible(w_key)) {
        g_gemv_delegating = true;
        cl_mem r = nnopt_gemv_fused(cl_ctx, weights, queue, x, w_key, rows, in_dim, out_dim, bias,
                                    std::string(), false);
        g_gemv_delegating = false;
        if (r) return r;
    }
    (void)queue;
    cl_mem S = q4 ? weights.get_buffer(s_key) : nullptr;
    if (q4 && !S) { NNOPT_ERROR_FMT("nnopt_gemv: %s present but unreadable", s_key.c_str()); return nullptr; }

    static cl_kernel k_f16 = nullptr, k_f16b = nullptr, k_q4 = nullptr, k_q4b = nullptr;
    cl_int err = CL_SUCCESS;
    cl_kernel use = nullptr;
    if (q4) {
        cl_kernel& slot = bias ? k_q4b : k_q4;
        if (!slot) {
            cl_program p = cl_ctx.build_program_from_file("kernels/gemv_q4_f32.cl");
            if (!p) { NNOPT_ERROR("nnopt_gemv: build kernels/gemv_q4_f32.cl"); return nullptr; }
            slot = clCreateKernel(p, bias ? "gemv_q4_bias_f32" : "gemv_q4_f32", &err);
            if (!slot) { NNOPT_ERROR_FMT("nnopt_gemv: q4 clCreateKernel (err=%d)", err); return nullptr; }
        }
        use = slot;
    } else {
        // NNOPT_GEMV=v8 selects 128-BIT weight loads (vload_half8) instead of the default 64-bit
        // vload_half4. The Adreno guide names 128 bits as the maximum per-transaction width and
        // calls out memory-bound kernels specifically (80-NB295-11 Rev. C §6.3); the AR reads ~95 GB
        // of weights per 2 s chunk, so the weight transaction width IS the kernel. Behind a toggle
        // because the default shape was picked by a sweep on an Adreno 620, and tuning verdicts from
        // one Adreno have repeatedly failed to transfer to another — this gets A/B'd on the target.
        static int v8 = -1, v8_epoch = -1;
        if (v8_epoch != nnopt_toggle_epoch()) {
            const char* g = std::getenv("NNOPT_GEMV");
            v8 = (g && std::strcmp(g, "v8") == 0) ? 1 : 0;
            v8_epoch = nnopt_toggle_epoch();
        }
        // Bias-less path only (the biased kernel is a separate file and is not the hot one), and
        // only when in_dim splits into 8s — every AR shape here is a multiple of 64, but the guard
        // keeps a future shape from silently reading past the row.
        const bool use_v8 = (v8 == 1) && !bias && (in_dim % 8 == 0);
        static cl_kernel k_f16_v8 = nullptr;
        cl_kernel& slot = use_v8 ? k_f16_v8 : (bias ? k_f16b : k_f16);
        if (!slot) {
            const char* file = use_v8 ? "kernels/linear_f32_w16_v8.cl"
                                      : (bias ? "kernels/linear_bias_f32.cl" : "kernels/linear_f32_w16.cl");
            cl_program p = cl_ctx.build_program_from_file(file);
            if (!p) { NNOPT_ERROR_FMT("nnopt_gemv: build %s", file); return nullptr; }
            slot = clCreateKernel(p, use_v8 ? "linear_f32_w16_v8"
                                            : (bias ? "linear_bias_f32" : "linear_f32_w16"), &err);
            if (!slot) { NNOPT_ERROR_FMT("nnopt_gemv: fp16 clCreateKernel (err=%d)", err); return nullptr; }
        }
        use = slot;

        // NNOPT_SG (default ON) — swap the local-memory tree reduction for a subgroup reduction.
        // gemv is 75.6% of AR GPU time and its combine costs six barriers for three accumulate
        // iterations, so the barriers, not the math, are what it spends its time on.
        //
        // Default-on but self-disabling: the device must advertise cl_khr_subgroups AND the program
        // must actually compile, otherwise this silently keeps the existing kernel. A device without
        // subgroups must never fail here — it must just run the old path.
        static int sg_on = -1, sg_epoch = -1;
        if (sg_epoch != nnopt_toggle_epoch()) {
            const char* g = std::getenv("NNOPT_SG");
            sg_on = (g && g[0] == '0') ? 0 : 1;
            sg_epoch = nnopt_toggle_epoch();
        }
        static int sg_state = -1;            // -1 untried, 0 unusable, 1 usable
        static cl_kernel k_sg = nullptr;
        if (sg_on && !use_v8 && !bias) {
            if (sg_state < 0) {
                size_t eb = 0;
                clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_EXTENSIONS, 0, nullptr, &eb);
                std::string dext(eb ? eb - 1 : 0, '\0');
                if (eb) clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_EXTENSIONS, eb, &dext[0], nullptr);
                const bool has_sg = dext.find("cl_khr_subgroups") != std::string::npos;
                if (has_sg) {
                    // Subgroup built-ins are OpenCL 2.0 and the default build is CL1.2, so without
                    // an explicit -cl-std the compiler rejects them with "OpenCL 2.0 built-in is not
                    // supported" — the device advertising cl_khr_subgroups is not enough. Try the
                    // newest std first and fall back, since the 840 reports 3.x and the 620 2.0.
                    // NNOPT_SGSTD overrides the order so the two build modes can be compared: the
                    // OpenCL 2.0+ std is REQUIRED for subgroup built-ins, but it also switches the
                    // whole kernel into generic-address-space codegen, which the guide (8.8) calls
                    // out as slower. That makes "is the reduce slow" and "is the build mode slow"
                    // two different questions.
                    const char* sgstd = std::getenv("NNOPT_SGSTD");
                    const char* kStds3[] = {"-cl-std=CL3.0", "-cl-std=CL2.0"};
                    const char* kStds2[] = {"-cl-std=CL2.0", "-cl-std=CL3.0"};
                    const char** kStds = (sgstd && std::strcmp(sgstd, "2") == 0) ? kStds2 : kStds3;
                    for (int si = 0; si < 2; ++si) {
                        const char* std_opt = kStds[si];
                        cl_program sp = cl_ctx.build_program_from_file("kernels/linear_f32_w16_sg.cl", std_opt);
                        if (sp) { k_sg = clCreateKernel(sp, "linear_f32_w16_sg", &err); }
                        if (k_sg) { std::fprintf(stderr, "GEMV_REDUCE built with %s\n", std_opt); break; }
                    }
                }
                sg_state = k_sg ? 1 : 0;

                // Self-check on the device, against this call's REAL weights. The wave size cannot
                // be pinned here (no cl_qcom_reqd_sub_group_size), so the multi-subgroup path is
                // reasoning about a number this code does not control — and getting it wrong yields
                // a partial sum, which is silently plausible audio rather than a crash. Run both
                // kernels once and compare; anything past float reassociation means fall back.
                if (sg_state == 1) {
                    const size_t obytes = (size_t)rows * out_dim * sizeof(float);
                    cl_mem o_ref = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, obytes, nullptr, &err);
                    cl_mem o_sg  = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, obytes, nullptr, &err);
                    if (o_ref && o_sg) {
                        const size_t nwg_c = ((size_t)out_dim + 7) / 8;
                        const size_t g_c[2] = {(size_t)rows, nwg_c * 64}, l_c[2] = {1, 64};
                        auto run = [&](cl_kernel k, cl_mem o) {
                            int a = 0;
                            clSetKernelArg(k, a++, sizeof(cl_mem), &x);
                            clSetKernelArg(k, a++, sizeof(cl_mem), &W);
                            clSetKernelArg(k, a++, sizeof(cl_mem), &o);
                            clSetKernelArg(k, a++, sizeof(int), &in_dim);
                            clSetKernelArg(k, a++, sizeof(int), &out_dim);
                            clEnqueueNDRangeKernel(queue, k, 2, nullptr, g_c, l_c, 0, nullptr, nullptr);
                        };
                        run(slot, o_ref);
                        run(k_sg,  o_sg);
                        std::vector<float> a_ref((size_t)rows * out_dim), a_sg((size_t)rows * out_dim);
                        clEnqueueReadBuffer(queue, o_ref, CL_TRUE, 0, obytes, a_ref.data(), 0, nullptr, nullptr);
                        clEnqueueReadBuffer(queue, o_sg,  CL_TRUE, 0, obytes, a_sg.data(),  0, nullptr, nullptr);
                        // Scale the error by the magnitude of the OUTPUT VECTOR, not per element.
                        // A per-element ratio divides by values that are legitimately near zero and
                        // reports a huge relative error for a rounding-sized absolute one — that is
                        // a broken metric, not a broken kernel.
                        double max_abs = 0.0, max_ref = 0.0;
                        for (size_t i = 0; i < a_ref.size(); ++i) {
                            const double d = std::fabs((double)a_ref[i] - (double)a_sg[i]);
                            if (d > max_abs) max_abs = d;
                            const double m = std::fabs((double)a_ref[i]);
                            if (m > max_ref) max_ref = m;
                        }
                        const double worst = max_abs / (max_ref > 1e-12 ? max_ref : 1.0);
                        // 1e-3 still sits ~500x below a structural mis-reduction (a dropped subgroup
                        // is ~50% off) while leaving room for fp32 reassociation under fast-math.
                        if (worst > 1e-3) {
                            std::fprintf(stderr,
                                "GEMV_REDUCE self-check FAILED (%.3g of scale; maxabs %.3g, maxref %.3g "
                                "on [%d,%d]) — using local-tree\n",
                                worst, max_abs, max_ref, in_dim, out_dim);
                            sg_state = 0;
                        } else {
                            std::fprintf(stderr,
                                "GEMV_REDUCE self-check ok (%.3g of scale; maxabs %.3g, maxref %.3g)\n",
                                worst, max_abs, max_ref);
                            // Correct is not the same as faster. Removing six barriers helps only if
                            // the wave-level reduce is native AND the OpenCL 2.0+ build mode this
                            // kernel needs does not pessimize the rest of it (generic address space,
                            // guide 8.8). On an Adreno 620 the subgroup build measured 2.4x SLOWER;
                            // on an 840 it may not. Neither device gets to decide for the other, so
                            // time both here and keep the winner. Same principle as XGEMM_TUNE.
                            const int kIters = 30;
                            auto bench = [&](cl_kernel k, cl_mem o) {
                                run(k, o); clFinish(queue);                    // warm
                                auto t0 = std::chrono::steady_clock::now();
                                for (int it = 0; it < kIters; ++it) run(k, o);
                                clFinish(queue);
                                return std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() - t0).count() / kIters;
                            };
                            const double ms_ref = bench(slot,  o_ref);
                            const double ms_sg  = bench(k_sg,  o_sg);
                            // Require a REAL margin. Measured on the 840 these two land within 2 us
                            // of each other and the winner flipped between runs — switching on that
                            // is noise, and the subgroup path is not free (it needs an OpenCL 2.0+
                            // build, i.e. generic-address-space codegen for that kernel). Keep the
                            // default unless subgroups win by more than the run-to-run spread.
                            const bool clear_win = ms_sg < ms_ref * 0.97;
                            if (!clear_win) sg_state = 0;
                            std::fprintf(stderr,
                                "GEMV_REDUCE bench [%d,%d]: local-tree %.3f ms, subgroup %.3f ms "
                                "(%.1f%%) -> %s\n",
                                in_dim, out_dim, ms_ref, ms_sg,
                                (ms_sg / ms_ref - 1.0) * 100.0,
                                sg_state ? "subgroup" : "local-tree (no clear win)");
                        }
                    }
                    if (o_ref) pool_free(o_ref);
                    if (o_sg)  pool_free(o_sg);
                }

                std::fprintf(stderr, "GEMV_REDUCE %s\n",
                             sg_state ? "subgroup (no barriers)"
                                      : (has_sg ? "local-tree (subgroup path rejected)"
                                                : "local-tree (no cl_khr_subgroups)"));
                std::fflush(stderr);
            }
            if (sg_state == 1) use = k_sg;
        }
    }

    // NNOPT_GEMVT=1 — K-major thread-per-output GEMV. The default kernel gives each workgroup 8
    // outputs and has its 64 threads stride the input, which costs a tree reduction: 6 barriers over
    // local memory. For the AR's shapes that reduction is the bulk of the kernel — in_dim 768 means
    // the accumulate loop runs THREE times before those six barriers. K-major removes the reduction
    // entirely (one thread owns one output) while keeping wave reads coalesced.
    //
    // Costs one transpose per weight, done once and cached, so it never lands on the hot path.
    // Bias-less rows==1 only, which is every AR GEMV.
    static int g_gemvt_v4 = 0;
    static int gt = -1, gt_epoch = -1;
    if (gt_epoch != nnopt_toggle_epoch()) {
        const char* g = std::getenv("NNOPT_GEMVT");
        gt = (g && g[0] != '0') ? 1 : 0;
        gt_epoch = nnopt_toggle_epoch();
    }
    if (gt && !q4 && !bias && rows == 1) {
        static std::map<std::string, cl_mem> tcache;
        static cl_kernel k_tr = nullptr, k_gt = nullptr;
        if (!k_gt) {
            cl_program pt = cl_ctx.build_program_from_file("kernels/transpose_nk_to_kn_f16.cl");
            cl_program pg = cl_ctx.build_program_from_file("kernels/gemv_transposed_f16.cl");
            if (pt) k_tr = clCreateKernel(pt, "transpose_nk_to_kn_f16", &err);
            // NNOPT_GEMVT=v4 picks the four-outputs-per-thread variant; =1 the scalar one.
            const char* gv = std::getenv("NNOPT_GEMVT");
            const bool want_v4 = gv && std::strcmp(gv, "v4") == 0;
            if (pg) k_gt = clCreateKernel(pg, want_v4 ? "gemv_transposed_f16_v4"
                                                      : "gemv_transposed_f16", &err);
            g_gemvt_v4 = want_v4 ? 1 : 0;
        }
        cl_mem Wt = nullptr;
        if (k_tr && k_gt) {
            auto it = tcache.find(w_key);
            if (it != tcache.end()) Wt = it->second;
            else {
                cl_int te = CL_SUCCESS;
                Wt = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                    (size_t)in_dim * out_dim * sizeof(cl_half), nullptr, &te);
                if (Wt) {
                    int ta = 0;
                    clSetKernelArg(k_tr, ta++, sizeof(cl_mem), &W);
                    clSetKernelArg(k_tr, ta++, sizeof(cl_mem), &Wt);
                    clSetKernelArg(k_tr, ta++, sizeof(int), &out_dim);   // N
                    clSetKernelArg(k_tr, ta++, sizeof(int), &in_dim);    // K
                    size_t tg[2] = {(size_t)in_dim, (size_t)out_dim};
                    if (cl_ctx.profEnqueue(k_tr, 2, tg, nullptr, "transpose_w") != CL_SUCCESS) {
                        clReleaseMemObject(Wt); Wt = nullptr;
                    }
                }
                tcache[w_key] = Wt;   // cache the failure too, so a bad shape is not retried per call
            }
        }
        if (Wt) {
            cl_mem outT = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE,
                                     (size_t)out_dim * sizeof(float), nullptr, &err);
            if (outT) {
                // Per-site instance, same reason as the main path below. (k_tr above needs none —
                // the transpose runs once per weight at prep time, never inside a recorded frame.)
                cl_kernel k_gt_i = nnopt_kernel_instance(k_gt, w_key, ".gemvt");
                int ga = 0;
                clSetKernelArg(k_gt_i, ga++, sizeof(cl_mem), &x);
                clSetKernelArg(k_gt_i, ga++, sizeof(cl_mem), &Wt);
                clSetKernelArg(k_gt_i, ga++, sizeof(cl_mem), &outT);
                clSetKernelArg(k_gt_i, ga++, sizeof(int), &in_dim);    // K
                clSetKernelArg(k_gt_i, ga++, sizeof(int), &out_dim);   // N
                const size_t nthread = g_gemvt_v4 ? (size_t)((out_dim + 3) / 4) : (size_t)out_dim;
                const size_t gwsT = (nthread + 63) / 64 * 64, lwsT = 64;
                if (cl_ctx.profEnqueue(k_gt_i, 1, &gwsT, &lwsT, "gemv_t") == CL_SUCCESS) return outT;
                pool_free(outT);
            }
        }
    }

    cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE,
                            (size_t)rows * out_dim * sizeof(float), nullptr, &err);
    if (!out) { NNOPT_ERROR_FMT("nnopt_gemv: out alloc [%d,%d]", rows, out_dim); return nullptr; }

    // One kernel object per call site. `use` up to here is the shared prototype that the variant
    // selection above picked (q4 / fp16 / v8 / subgroup); from here it is this site's own instance,
    // so the args below are not clobbered by the next layer's dispatch and a capture stays valid.
    use = nnopt_kernel_instance(use, w_key);

    int ai = 0;
    clSetKernelArg(use, ai++, sizeof(cl_mem), &x);
    clSetKernelArg(use, ai++, sizeof(cl_mem), &W);
    if (q4)   clSetKernelArg(use, ai++, sizeof(cl_mem), &S);
    if (bias) clSetKernelArg(use, ai++, sizeof(cl_mem), &bias);
    clSetKernelArg(use, ai++, sizeof(cl_mem), &out);
    clSetKernelArg(use, ai++, sizeof(int), &in_dim);
    clSetKernelArg(use, ai++, sizeof(int), &out_dim);
    // 8, not NOUT: this is the UNFUSED path (linear_f32_w16 / gemv_q4_f32), whose kernels compute a
    // fixed 8 outputs per workgroup. NOUT only parameterises linear_bias_fused_f32.
    const size_t nwg = ((size_t)out_dim + 7) / 8;
    const size_t gws[2] = {(size_t)rows, nwg * 64}, lws[2] = {1, 64};
    if (cl_ctx.profEnqueue(use, 2, gws, lws, "gemv") != CL_SUCCESS) {
        NNOPT_ERROR_FMT("nnopt_gemv: enqueue %s", w_key.c_str());
        pool_free(out);
        return nullptr;
    }
    return out;
}

bool nnopt_weight_dims(Weights& weights, const std::string& w_key, int* out_dim, int* in_dim) {
    const std::vector<int> sh = weights.get_shape(w_key);
    if (sh.size() != 2) { NNOPT_ERROR_FMT("nnopt_weight_dims: %s is not rank-2", w_key.c_str()); return false; }
    const bool packed = weights.has_tensor(w_key + ".scale");
    if (out_dim) *out_dim = sh[0];
    if (in_dim)  *in_dim  = packed ? sh[1] * 2 : sh[1];
    return true;
}

// ── per-request toggle epoch (see utils.h) ──────────────────────────────────────────────────────
namespace {
int g_toggle_epoch = 0;
}
int  nnopt_toggle_epoch() { return g_toggle_epoch; }
void nnopt_toggle_bump()  { ++g_toggle_epoch; }

// ── per-call-site kernel instances (see utils.h) ─────────────────────────────────────────────────
// A cl_qcom_recordable_queues recording holds a REFERENCE to a cl_kernel, not a snapshot of its
// arguments. This engine shares one cl_kernel per kernel TYPE, so all 192 gemv dispatches in a frame
// set args on the same object; a replay then runs every one of them with whatever args were set
// last — wrong buffers, out-of-bounds reads, GPU fault, device reset. That is the whole reason
// recording is off. Giving every dispatch its own kernel object, with args set once at capture and
// never rewritten, is what makes a capture self-consistent.
//
// The site key CANNOT be the weight prefix alone. run_depth_step runs 12x per frame (once per RVQ
// codebook) over the SAME weights, so a prefix-keyed instance would still be shared 12 ways inside
// one recorded frame — the original bug, unchanged. The key is therefore (prototype, prefix, frame
// dispatch ordinal): the ordinal makes it unique WITHIN a frame, and the prefix makes a divergence
// between frames detectable instead of silent.
namespace {
// Indexed by the frame dispatch ordinal, not a map keyed by a built-up string: this runs ~455 times
// per frame on the path whose whole problem is HOST cost, so a per-dispatch heap allocation plus a
// std::map lookup would be spending exactly the resource this change exists to save. The ordinal is
// deterministic (identical dispatch sequence every frame), so a flat vector plus a prototype-pointer
// compare resolves the common case in O(1) with no allocation.
struct KInstSlot { cl_kernel proto = nullptr; cl_kernel inst = nullptr; };
// ALL per-thread. The codec now runs on its own host thread while the AR generates, and it shares
// helpers (residual_add and friends) with the AR — on shared globals its dispatches would land in
// the middle of the AR frame's ordinal sequence, which is the exact divergence that makes a frame
// capture invalid. A worker thread gets its own inert copy: not in a frame, so no ordinals, no
// trace, no per-site kernels.
thread_local std::vector<KInstSlot>   g_kinst_slots[2];       // [generation][ordinal]
thread_local size_t                   g_kinst_made   = 0;     // distinct kernel objects created
thread_local long                     g_kinst_seq    = 0;     // dispatch ordinal within the frame
thread_local bool                     g_kinst_stable = true;  // cleared if the sequence diverges
// base + suffix kept SEPARATE: concatenating them per dispatch would heap-allocate ~455 times a
// frame on the one path whose entire problem is host cost. Comparison is by value, allocation-free.
struct KInstTrace { std::string base; const char* suffix; };
thread_local std::vector<KInstTrace>  g_kinst_trace;          // frame 0's sequence, to compare later frames
thread_local bool                     g_kinst_have_trace = false;
thread_local long                     g_kinst_frames = 0;
thread_local int                      g_kinst_gen    = 0;     // see nnopt_kinst_set_generation
thread_local bool                     g_kinst_in_frame = false; // inside an AR frame? (see nnopt_kinst_frame_end)
thread_local std::map<cl_kernel,cl_kernel> g_kinst_proto;        // instance -> prototype (for LWS tuning)
thread_local long                     g_kinst_dispatches = 0;   // NDRange dispatches seen inside the frame
thread_local long                     g_kinst_requests   = 0;   // per-site instance lookups inside the frame
}

// Once a frame is captured, the kernel objects it references must NEVER have their args rewritten —
// a later live frame allocates fresh output buffers from the pool and would silently repoint the
// recording at them. So a live pass that has to run AFTER a capture (the replay verification) bumps
// the generation, which vends it a disjoint set of kernel objects and leaves the capture intact.
void nnopt_kinst_set_generation(int g) { g_kinst_gen = (g > 0) ? 1 : 0; }
int  nnopt_kinst_generation()          { return g_kinst_gen; }

// Outside an AR frame the ordinal must not advance. The CODEC calls some of the same helpers
// (residual_add among them) and runs at chunk boundaries INSIDE the AR's frame loop, so without a
// closed window its dispatches append to the AR frame's sequence — which is exactly the
// "DIVERGED at #480 (saw 'residual_add')" failure: frame 0 has no codec, frame 4 does. Codec
// dispatches simply share kernel objects, as they always did; the codec is never recorded.
void nnopt_kinst_frame_end() {
    // Every dispatch inside a recorded frame must have come through nnopt_kernel_instance. If any
    // did not, it is still sharing a kernel object across call sites — the exact defect that reset
    // the GPU — and no amount of grepping proves the list is complete. So the engine counts both
    // and refuses to record when they disagree, naming the gap instead of replaying stale args.
    if (g_kinst_in_frame && g_kinst_dispatches != g_kinst_requests) {
        static bool told = false;
        if (!told) {
            told = true;
            std::fprintf(stderr,
                "KINST %ld dispatches vs %ld per-site lookups in one AR frame — %ld dispatch(es) "
                "bypassed nnopt_kernel_instance and still share a kernel object\n",
                g_kinst_dispatches, g_kinst_requests, g_kinst_dispatches - g_kinst_requests);
        }
        nnopt_record_mark_unsafe("a dispatch in the frame bypassed the per-site kernel pool");
    }
    g_kinst_in_frame = false;
}

// Called from OpenCLContext::profEnqueue for every NDRange dispatch.
void nnopt_kinst_note_dispatch() { if (g_kinst_in_frame) ++g_kinst_dispatches; }

// The kernel TYPE behind a per-site instance. Workgroup tuning keys on this: keying on the
// instance would give every one of the ~519 sites its own cache entry, i.e. no tuning at all.
cl_kernel nnopt_kinst_proto_for(cl_kernel k) {
    if (!k) return k;
    auto it = g_kinst_proto.find(k);
    return (it == g_kinst_proto.end()) ? k : it->second;
}

void nnopt_kinst_frame_begin() {
    g_kinst_in_frame = true;
    // The first frame that ran to completion becomes the reference every later frame is checked
    // against.
    if (!g_kinst_have_trace && g_kinst_frames > 0 && !g_kinst_trace.empty()) g_kinst_have_trace = true;
    g_kinst_seq = 0;
    g_kinst_dispatches = 0;
    g_kinst_requests   = 0;
    ++g_kinst_frames;
}

bool nnopt_kinst_stable() { return g_kinst_stable; }

void nnopt_kinst_reset() {
    // Deliberately does NOT drop the vended kernel objects: they are valid for the process and
    // rebuilding ~455 of them per render would be pure waste. Only the per-render bookkeeping goes.
    g_kinst_seq = 0; g_kinst_frames = 0; g_kinst_stable = true; g_kinst_gen = 0;
    g_kinst_in_frame = false;
    g_kinst_trace.clear(); g_kinst_have_trace = false;
}

// ── record-safety latch ─────────────────────────────────────────────────────────────────────────
// Any code path that bakes a host-computed value into a kernel argument makes a capture invalid for
// every later frame — the replay would reuse frame N's value forever. Such a path must veto
// recording rather than hope it is never taken. Latches once and reports why.
namespace { bool g_rec_safe = true; }
void nnopt_record_mark_unsafe(const char* why) {
    if (!g_rec_safe) return;
    g_rec_safe = false;
    std::fprintf(stderr, "RECORDQ unsafe: %s — staying on live dispatch\n", why ? why : "?");
    std::fflush(stderr);
}
bool nnopt_record_safe() { return g_rec_safe && g_kinst_stable; }

// ── buffer copy as a KERNEL (see kernels/copy_f32.cl) ──────────────────────────────────────────
// A recording captures only clEnqueueNDRangeKernel. Any clEnqueueCopyBuffer inside the captured
// frame is therefore NOT replayed, and every replayed frame silently reuses whatever the captured
// frame left in the destination — a wrong-but-plausible token, not a crash. Offsets are in ELEMENTS.
bool nnopt_copy_buffer(OpenCLContext& cl_ctx, cl_mem src, cl_mem dst,
                       int src_off, int dst_off, int n,
                       const std::string& site, const char* suffix) {
    static cl_kernel k = nullptr; cl_int err = CL_SUCCESS;
    if (!k) {
        cl_program p = cl_ctx.build_program_from_file("kernels/copy_f32.cl");
        if (!p) { NNOPT_ERROR("nnopt_copy_buffer: build copy_f32 failed"); return false; }
        k = clCreateKernel(p, "copy_f32", &err);
        if (!k) { NNOPT_ERROR_FMT("nnopt_copy_buffer: clCreateKernel (err=%d)", err); return false; }
    }
    cl_kernel ki = nnopt_kernel_instance(k, site, suffix);
    clSetKernelArg(ki,0,sizeof(cl_mem),&src);     clSetKernelArg(ki,1,sizeof(cl_mem),&dst);
    clSetKernelArg(ki,2,sizeof(int),&src_off);    clSetKernelArg(ki,3,sizeof(int),&dst_off);
    clSetKernelArg(ki,4,sizeof(int),&n);
    const size_t g = (size_t)((n + 63) / 64 * 64), l = 64;
    return cl_ctx.profEnqueue(ki,1,&g,&l,"copy") == CL_SUCCESS;
}

cl_kernel nnopt_kernel_instance(cl_kernel proto, const std::string& base, const char* suffix) {
    if (!proto) return nullptr;

    // Epoch-keyed, never a plain function-local static: a static is evaluated during the warm-up
    // render, before a serve request sets its env, so every later experiment would silently measure
    // the warm-up's value. NNOPT_KINST=0 restores the shared-kernel behaviour for an A/B.
    static int on = -1, on_epoch = -1;
    if (on_epoch != nnopt_toggle_epoch()) {
        const char* e = std::getenv("NNOPT_KINST");
        on = (e && e[0] == '0') ? 0 : 1;
        on_epoch = nnopt_toggle_epoch();
    }
    if (!on) return proto;

    // Not inside an AR frame (codec, warm-up, op-tests): share the kernel object as before and do
    // NOT advance the ordinal. Only the AR frame is recorded, so only it needs per-site objects.
    if (!g_kinst_in_frame) return proto;

    ++g_kinst_requests;
    const size_t seq = (size_t)g_kinst_seq++;

    // Divergence check. Every frame must issue the identical dispatch sequence, or a capture taken
    // on one frame is not valid for the next. Compared without building a key: a std::string
    // equality test is a length check plus a memcmp, whereas `base + suffix` would malloc.
    if (!g_kinst_have_trace) {
        g_kinst_trace.push_back(KInstTrace{base, suffix});
    } else if (g_kinst_stable) {
        const bool same = seq < g_kinst_trace.size()
                       && g_kinst_trace[seq].base == base
                       && ((g_kinst_trace[seq].suffix == nullptr && suffix == nullptr) ||
                           (g_kinst_trace[seq].suffix && suffix &&
                            std::strcmp(g_kinst_trace[seq].suffix, suffix) == 0));
        if (!same) {
            g_kinst_stable = false;
            std::fprintf(stderr,
                "KINST dispatch sequence DIVERGED at #%zu (saw '%s%s') — recording is not safe\n",
                seq, base.c_str(), suffix ? suffix : "");
            std::fflush(stderr);
        }
    }

    std::vector<KInstSlot>& slots = g_kinst_slots[g_kinst_gen];
    if (seq < slots.size() && slots[seq].proto == proto) return slots[seq].inst;

    // Recover the program and entry name from the prototype. Both clGetKernelInfo queries are core
    // OpenCL 1.2 and both are forwarded by Edgi's libOpenCL shim. clCloneKernel would be the obvious
    // call but it is OpenCL 2.1 AND absent from the shim's forwarder list, so it is not an option.
    cl_program prog = nullptr;
    char       fname[256] = {0};
    cl_kernel  inst = nullptr;
    if (clGetKernelInfo(proto, CL_KERNEL_PROGRAM, sizeof(prog), &prog, nullptr) == CL_SUCCESS &&
        clGetKernelInfo(proto, CL_KERNEL_FUNCTION_NAME, sizeof(fname) - 1, fname, nullptr) == CL_SUCCESS &&
        prog && fname[0]) {
        cl_int err = CL_SUCCESS;
        inst = clCreateKernel(prog, fname, &err);
        if (!inst) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::fprintf(stderr, "KINST clCreateKernel(%s) failed (%d) — sharing kernel objects, "
                                     "recording disabled\n", fname, err);
            }
        }
    }
    if (!inst) {
        // Falling back to the shared object is correct for LIVE dispatch but fatal for a replay, so
        // it must also veto recording rather than quietly reintroduce the original bug.
        nnopt_record_mark_unsafe("could not create a per-site kernel object");
        inst = proto;
    } else {
        ++g_kinst_made;
    }
    if (inst != proto) g_kinst_proto[inst] = proto;
    if (seq >= slots.size()) slots.resize(seq + 1);
    // Replacing a slot must RELEASE what was there. Slots persist across renders on purpose (see
    // nnopt_kinst_reset), and they are only replaced when the kernel at a given ordinal CHANGES —
    // which is precisely what switching precision does: int8 <-> fp16 repoints ~200 GEMV sites at a
    // different program, so every toggle rebuilt and abandoned a frame's worth of kernel objects.
    // A cl_kernel holds a reference to every buffer bound to it, so the leak is not 40 bytes per
    // object, it is the arguments they pin — and it grows with each switch until allocation fails.
    if (slots[seq].inst && slots[seq].inst != inst && slots[seq].inst != slots[seq].proto) {
        g_kinst_proto.erase(slots[seq].inst);
        clReleaseKernel(slots[seq].inst);
    }
    slots[seq].proto = proto;
    slots[seq].inst  = inst;
    return inst;
}

// Number of distinct kernel objects vended — the ~455-per-frame figure, observable.
size_t nnopt_kinst_count() { return g_kinst_made; }

// ── int8 weights for the depth MLPs ─────────────────────────────────────────
// depth_body runs ONCE PER RVQ LEVEL — 12x per frame — so its weights are re-read 12x while the
// temporal body's are read once. Its per-level working set is 22.8 MB against the 840's 18 MB HPM:
// it misses by 27%, so every level goes to DRAM. The four MLP tensors are 18.88 MB of that; at int8
// they are 9.44 MB, which puts the working set at ~13.4 MB and inside the cache.
//
// Symmetric per-OUTPUT-ROW quantization: each row gets its own absmax/127 scale, so a row whose
// weights are small keeps its resolution instead of being crushed by some other row's outlier. This
// is the property q4 lacked when it wrecked the audio here. Done once at first use, from the fp16
// buffer already on the device, so there is no new weight file to convert, ship, or download.
// rowsum = the per-output-row sum of the QUANTIZED weights, precomputed here.
//
// qcom_dot8_acc multiplies SIGNED by UNSIGNED bytes, so the activation has to carry a zero point of
// 128 to become unsigned. Expanding that:
//     sum_d Wq[d]*(xq_u[d]-128) = sum_d Wq[d]*xq_u[d]  -  128 * sum_d Wq[d]
// The second term depends only on the weight row, so it is a constant computed once at quantization
// time rather than a correction the kernel recomputes on every call.
struct NnoptInt8W { cl_mem w = nullptr; cl_mem scale = nullptr; cl_mem rowsum = nullptr;
                    cl_mem wimg = nullptr; };   // wimg: an image2d VIEW of w, see nnopt_int8_weight

namespace { int g_i8_n = 0; double g_i8_mb = 0.0; }
void nnopt_int8_note(double mb) { ++g_i8_n; g_i8_mb += mb; }
void nnopt_int8_summary(char* buf, size_t n) {
    std::snprintf(buf, n, "int8 scope=%d tensors=%d %.0fMB (from %.0fMB fp16)",
                  nnopt_int8_scope(), g_i8_n, g_i8_mb, g_i8_mb * 2);
}

static NnoptInt8W* nnopt_int8_weight(OpenCLContext& cl_ctx, cl_command_queue queue,
                                     Weights& weights, const std::string& w_key,
                                     int in_dim, int out_dim) {
    static std::map<std::string, NnoptInt8W> cache;
    auto it = cache.find(w_key);
    if (it != cache.end()) return it->second.w ? &it->second : nullptr;

    NnoptInt8W e;
    cache[w_key] = e;                        // cache failure too, so a bad key is not retried per call
    cl_mem Wf16 = weights.get_buffer(w_key);
    if (!Wf16) return nullptr;

    const size_t n = (size_t)in_dim * out_dim;
    std::vector<uint16_t> h16(n);
    // Always the LIVE queue. A blocking read on the recordable queue would either be captured or
    // fail — a recordable queue does not execute — and frame 1 is the capture frame. Frame 0
    // quantizes everything, so in practice this runs before any capture, but the queue the caller
    // handed down is the frame queue and must not be assumed live.
    cl_command_queue rq = nnopt_live_queue() ? nnopt_live_queue() : queue;
    cl_int err = clEnqueueReadBuffer(rq, Wf16, CL_TRUE, 0, n * sizeof(uint16_t), h16.data(),
                                     0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("int8: read %s (err=%d)", w_key.c_str(), (int)err); return nullptr; }

    std::vector<int8_t> q(n);
    std::vector<float>  sc((size_t)out_dim);
    std::vector<int32_t> rs((size_t)out_dim, 0);
    for (int r = 0; r < out_dim; ++r) {
        const size_t off = (size_t)r * in_dim;
        float amax = 0.0f;
        for (int k = 0; k < in_dim; ++k) {
            const float v = std::fabs(nnopt_f16_to_f32(h16[off + k]));
            if (v > amax) amax = v;
        }
        const float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        sc[r] = scale;
        const float inv = 1.0f / scale;
        int32_t rsum = 0;
        for (int k = 0; k < in_dim; ++k) {
            float t = nnopt_f16_to_f32(h16[off + k]) * inv;
            t = t < -127.0f ? -127.0f : (t > 127.0f ? 127.0f : t);
            const int8_t qv = (int8_t)std::lrintf(t);
            q[off + k] = qv;
            rsum += qv;                      // the -128*sum_d Wq[d] correction term, per row
        }
        rs[(size_t)r] = rsum;
    }

    e.w = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         n * sizeof(int8_t), q.data(), &err);
    if (e.w) e.scale = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      (size_t)out_dim * sizeof(float), sc.data(), &err);
    if (e.w) e.rowsum = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       (size_t)out_dim * sizeof(int32_t), rs.data(), &err);
    if (!e.w || !e.scale || !e.rowsum) {
        if (e.w)     { clReleaseMemObject(e.w); e.w = nullptr; }
        if (e.scale) { clReleaseMemObject(e.scale); e.scale = nullptr; }
        if (e.rowsum){ clReleaseMemObject(e.rowsum); e.rowsum = nullptr; }
        NNOPT_ERROR_FMT("int8: alloc failed for %s", w_key.c_str());
        return nullptr;
    }
    // ── an image2d view of the weight buffer (guide §6.2 / §7.1.5.3) ────────────────────────────
    // "Adreno GPUs have a powerful texture engine and dedicated level 1 cache that can load data in
    // image objects effectively" — a GEMV streaming weights out of a plain buffer never touches
    // that L1 at all. cl_khr_image2d_from_buffer makes the image a VIEW over the bytes already
    // uploaded: no copy, no second allocation.
    //
    // CL_RGBA + CL_SIGNED_INT32 = 128 bits per texel, which §6.3 names as the widest fetch, and
    // reinterprets as sixteen int8 weights via as_char16. Width is in_dim/16 texels, height is one
    // row per output.
    //
    // Every constraint is checked rather than assumed, and a failure leaves wimg null so the buffer
    // path runs — an image is a performance choice, never a correctness one.
    if (e.w && (in_dim % 16) == 0) {
        cl_uint pitch_align = 0; size_t max_w = 0, max_h = 0;
        clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_IMAGE_PITCH_ALIGNMENT, sizeof(pitch_align), &pitch_align, nullptr);
        clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_IMAGE2D_MAX_WIDTH,  sizeof(max_w), &max_w, nullptr);
        clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(max_h), &max_h, nullptr);
        const size_t tex_w = (size_t)in_dim / 16;        // 16 int8 per RGBA-int32 texel
        const size_t tex_h = (size_t)out_dim;
        const bool pitch_ok = (pitch_align == 0) || (tex_w % pitch_align == 0);
        if (tex_w && tex_w <= max_w && tex_h <= max_h && pitch_ok) {
            cl_image_format fmt{}; fmt.image_channel_order = CL_RGBA;
                                   fmt.image_channel_data_type = CL_SIGNED_INT32;
            cl_image_desc d{}; d.image_type = CL_MEM_OBJECT_IMAGE2D;
            d.image_width = tex_w; d.image_height = tex_h;
            d.image_row_pitch = (size_t)in_dim;          // bytes per weight row
            d.buffer = e.w;
            cl_int ie = CL_SUCCESS;
            e.wimg = clCreateImage(cl_ctx.context(), CL_MEM_READ_ONLY, &fmt, &d, nullptr, &ie);
            if (!e.wimg) {
                static bool warned = false;
                if (!warned) { warned = true;
                    std::fprintf(stderr, "IMG2D unavailable (err=%d, %zux%zu pitch=%d align=%u) — "
                                         "weights stream from the buffer\n",
                                 (int)ie, tex_w, tex_h, in_dim, pitch_align);
                    std::fflush(stderr); }
            }
        }
    }
    cache[w_key] = e;
    // One running total, not 40 lines: what matters in a log is which SCOPE ran and how much of the
    // model it actually covered, so a report can be tied back to a configuration after the fact.
    // Counted, not printed. This used to emit one line PER TENSOR — 93 lines of identical shape
    // per launch, which is most of why the engine log has to be scrolled to find anything. The
    // totals are reported once, with the render summary.
    nnopt_int8_note(n / 1e6);
    return &cache[w_key];
}

// The depth MLPs, and only those: the temporal body is read once per frame so halving it saves
// bandwidth but cannot change cache residency, and the logits head is already sliced per level.
static bool nnopt_is_depth_mlp(const std::string& k) {
    return k.find("depth_body.") != std::string::npos &&
           (k.find(".body.layers.1.inner._linear.weight") != std::string::npos ||
            k.find(".body.layers.3.inner._linear.weight") != std::string::npos);
}

// ── how much of the AR runs in int8 ─────────────────────────────────────────────────────────────
// NNOPT_INT8SCOPE, a LADDER so a quality regression can be bisected without a rebuild — the serve
// request forwards NNOPT_* keys straight to setenv, so every rung is one line typed into the app:
//
//   0  all fp16 — the reference
//   1  depth MLPs only                        (what shipped before v14)
//   2  + depth attention projections          => the whole depth body
//   3  + temporal MLPs
//   4  + temporal attention, incl. the cross-attention kv_proj that reads the style conditioning
//
// The rungs are ordered by how much damage a quantization error there can do, cheapest first. The
// depth body refines a frame that already exists; the temporal body carries the long-range
// structure, and its cross-attention kv_proj is the ONLY place the style conditioning enters the
// model — error there does not sound like noise, it sounds like the prompt not being followed.
//
// Attention projections deserve the suspicion: per-output-row weight scaling handles an outlier
// ROW, but the classic int8 failure in a transformer is an outlier ACTIVATION CHANNEL, which this
// scheme does not address at all. That is why they sit on the last two rungs.
//
// Everything outside the two transformer bodies stays fp16 at every rung: the logits head decides
// which token is SAMPLED (an error there changes the note, not its loudness) and the embeddings are
// a gather, not a GEMV.
static int nnopt_int8_scope() {
    static int v = -1, ep = -1;
    if (ep != nnopt_toggle_epoch()) {
        const char* e = std::getenv("NNOPT_INT8SCOPE");
        v = e ? std::atoi(e) : 4;
        ep = nnopt_toggle_epoch();
    }
    return v;
}

static bool nnopt_int8_eligible(const std::string& k) {
    const int s = nnopt_int8_scope();
    if (s <= 0) return false;
    const bool depth    = k.find("depth_body.")    != std::string::npos;
    const bool temporal = k.find("temporal_body.") != std::string::npos;
    if (!depth && !temporal) return false;
    if (k.find("logit") != std::string::npos) return false;
    if (k.find("embed") != std::string::npos) return false;
    // The two feed-forward denses of a Residual block; everything else under a body is attention
    // (qkv_proj / q_proj / kv_proj) or its output projection.
    const bool mlp = k.find(".body.layers.1.inner._linear.weight") != std::string::npos ||
                     k.find(".body.layers.3.inner._linear.weight") != std::string::npos;
    switch (s) {
        case 1:  return depth && mlp;
        case 2:  return depth;
        case 3:  return depth || (temporal && mlp);
        default: return true;
    }
}

// ── nnopt_gemv_fused — see utils.h ──────────────────────────────────────────
cl_mem nnopt_gemv_fused(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                        cl_mem x, const std::string& w_key,
                        int rows, int in_dim, int out_dim, cl_mem bias,
                        const std::string& rms_key, bool apply_gelu) {
    cl_mem W = weights.get_buffer(w_key);
    if (!W) { NNOPT_ERROR_FMT("nnopt_gemv_fused: missing %s", w_key.c_str()); return nullptr; }
    // Quantized weights keep the unfused path — the fused kernel reads fp16 rows directly.
    if (weights.has_tensor(w_key + ".scale"))
        return nnopt_gemv(cl_ctx, weights, queue, x, w_key, rows, in_dim, out_dim, bias);

    const bool want_rms = !rms_key.empty();
    cl_mem rms_scale = nullptr;
    if (want_rms) {
        rms_scale = weights.get_buffer(rms_key);
        if (!rms_scale) { NNOPT_ERROR_FMT("nnopt_gemv_fused: missing %s", rms_key.c_str()); return nullptr; }
    }
    // The fold assumes the row divides into float4s, matching the base kernel's `in_dim>>2` loop.
    if ((in_dim & 3) != 0)
        return nnopt_gemv(cl_ctx, weights, queue, x, w_key, rows, in_dim, out_dim, bias);

    // NNOPT_INT8SCOPE is the single control now. There used to be a second one here
    // (NNOPT_INT8DEPTH) plus a runtime autotuner that could override both — three ways to answer
    // one question, which is how a run ends up labelled with a configuration it did not use.
    cl_int err = CL_SUCCESS;
    NnoptInt8W* i8 = nullptr;
    if (nnopt_int8_eligible(w_key))
        i8 = nnopt_int8_weight(cl_ctx, queue, weights, w_key, in_dim, out_dim);
    const bool use_i8 = (i8 != nullptr);
    // Called from nnopt_gemv purely to reach the INT8 specialization. If int8 did not materialise
    // there is nothing here to fuse — PRENORM and GELU are both off — so hand the shape back
    // rather than quietly running a different fp16 kernel than the one nnopt_gemv's own toggles
    // (subgroup reduce, v8 loads) would have selected.
    if (g_gemv_delegating && !use_i8) return nullptr;

    // 128-bit weight loads, OFF by default. The Adreno guide names 128 bits as the widest
    // transaction and this kernel is weight-bandwidth-bound, so the wide step should have been free
    // money — measured on the 620 it is a LOSS: int8 417.8 us/call wide against 333.6 us narrow.
    // The narrow loop has adjacent lanes reading adjacent float4s, which the memory system already
    // merges into full transactions, so widening the per-lane load buys no bandwidth and costs
    // registers. Kept behind NNOPT_GEMVWIDE=1 because that verdict is a 620 verdict and the 840 has
    // a different compiler and a different cache hierarchy — it is one env var and one run to check.
    static int wide_on = -1, wide_epoch = -1;
    if (wide_epoch != nnopt_toggle_epoch()) {
        // ON for int8 now: the wide path finally loads the way the guide prescribes (uint4 +
        // as_char16 = one 128-bit transaction for sixteen weights) instead of vload16 on a char*,
        // which left the width to the compiler and measured slower. NNOPT_GEMVWIDE=0 to A/B.
        const char* gw = std::getenv("NNOPT_GEMVWIDE");
        wide_on = (gw && gw[0] == '0') ? 0 : 1;
        wide_epoch = nnopt_toggle_epoch();
    }

    // Outputs per workgroup. workgroups = out_dim/NOUT, and the driver's enqueue cost tracks
    // workgroup count on this device — see the note in the kernel. NNOPT_GEMV_NOUT=8|16|32.
    static int nout = 0, nout_epoch = -1;
    if (nout_epoch != nnopt_toggle_epoch()) {
        // Back to 8. NOUT=32 was a test of whether the driver's enqueue cost scales with WORKGROUP
        // count: it launches 4x fewer workgroups for the same dispatch. Measured on the 840 it does
        // not — host us/dispatch went 63.4 -> 68.6 with a quarter of the workgroups, so the cost is
        // flat per dispatch. It also cost GPU time (32 live accumulators is real register pressure).
        // Kept reachable via NNOPT_GEMV_NOUT for a device that answers differently.
        const char* e = std::getenv("NNOPT_GEMV_NOUT");
        nout = e ? std::atoi(e) : 8;
        if (nout != 8 && nout != 16 && nout != 32) nout = 32;
        nout_epoch = nnopt_toggle_epoch();
    }
    const int nout_bit = (nout == 32) ? 2 : (nout == 16 ? 1 : 0);

    // ── qcom_dot8_acc: DISABLED. IT PRODUCES BROKEN AUDIO. ──────────────────────────────────────
    //
    // Not a toggle. Flipping this to true ships wrong output, so it is a constant and the kernel
    // path it selects (DOT8 in linear_bias_fused_f32.cl) is dead code until the defect below is
    // fixed and the result is checked against the fp16 reference.
    //
    // What it was for: guide §9.5.1's one-instruction int8 dot product with int32 accumulate,
    // replacing four int-to-float converts and a float dot(). §8 notes Adreno has no general 8-bit
    // ALU, so those converts exist only to reach a float instruction.
    //
    // What is wrong: almost certainly the OPERAND SIGNEDNESS. §9.5.1 states
    // qcom_dot8_acc(uint p0, uint p1, int acc) takes "four signed 8-bit components and four
    // unsigned 8-bit components, respectively" — p0 signed — and this code passes the signed
    // weights as p0. But the same section's example builds p0 from `uchar p0a = -11;`, i.e. an
    // UNSIGNED type holding a negative value, which says the opposite. If p0 is really the unsigned
    // operand, every negative weight is read as W+256: a kernel that runs, does not crash, and
    // returns garbage — which is exactly what it did on the 840.
    //
    // The fix that removes the question rather than guessing at it: use qcom_udot8_acc, where both
    // operands are unsigned, and carry both zero points —
    //     sum (Wu-128)(xq-128) = sum Wu*xq - 128*sum Wu - 128*sum xq + 128*128*in_dim
    // sum Wu is per weight row (NnoptInt8W::rowsum already holds sum Wq; add 128*in_dim), sum xq
    // falls out of the kernel's first pass for free, and the last term is a constant.
    //
    // Worth noting before anyone spends time on it: this cannot speed up the render as it stands.
    // It reduces GPU time, and the AR measures 12.11 ms of GPU work inside a 30.7 ms frame — the
    // GPU is idle 61% of the time. It only pays once dispatch count comes down.
    // DISABLED. Measured on the Razr against an int8 control, same weights, same everything:
    //     int8, no dot8      rms 1.22x reference   spectrum cosine 0.9989
    //     int8 + dot8 (A)    rms 1.56x             0.9886
    //     int8 + dot8 (B)    rms 3.22x             0.9917
    // Both operand orders degrade, so the defect is NOT the p0/p1 signedness the guide contradicts
    // itself about — that was my first theory and the device disproved it. It is somewhere in the
    // zero-point expansion, and three readings of the kernel have not found it.
    //
    // Whoever picks this up: the control row above is the bar (0.9989), the harness is two renders
    // on the Razr with NNOPT_INT8SCOPE=4 and NNOPT_GEMVWIDE=0/1, and the arithmetic to check is
    //     sw*sx*( sum Wq*xq - 128*sum Wq )
    // with sx = max|x|/127 reduced over the workgroup and sum Wq precomputed per row (rowsum).
    const bool use_dot8 = false;

    // One specialization per (nout,wide,int8,prenorm,bias,gelu) combination, built once and cached.
    // Compile-time flags keep an unused feature out of the generated code entirely.
    // Weights through the texture engine when an image view exists (see nnopt_int8_weight). No
    // toggle: the image is either creatable on this device or it is not, and the answer is the
    // same on every call.
    const bool use_img = use_i8 && wide_on && i8->wimg != nullptr;

    // Activation stays in __global. __constant was tried (guide §6.4) and MEASURED WORSE: host cost
    // went 59.1 -> 66.7 us per dispatch the moment it went on, and nothing else in that build
    // touches the host. The likely mechanism is the driver copying the buffer into constant memory
    // on every enqueue — which is per-dispatch host work, and per-dispatch host work is the entire
    // frame time here. §6.4's benefit assumes UNIFORM access (all work items reading the same
    // address so it can broadcast); our lanes each read a different element of x, so the broadcast
    // never applied in the first place.
    static int constx = 0;

    const int variant = (constx ? 512 : 0) | (use_img ? 256 : 0) | (use_dot8 ? 128 : 0) |
                        (nout_bit << 5) | (wide_on ? 16 : 0) |
                        (use_i8 ? 8 : 0) | (want_rms ? 4 : 0) | (bias ? 2 : 0) | (apply_gelu ? 1 : 0);
    static cl_kernel k_var[1024] = {nullptr};
    static int       k_bad[1024] = {0};
    if (!k_var[variant] && !k_bad[variant]) {
        char opts[160];
        std::snprintf(opts, sizeof(opts),
                      "-D PRENORM=%d -D BIAS=%d -D GELU=%d -D INT8=%d -D WIDE=%d -D NOUT=%d "
                      "-D DOT8=%d -D IMG=%d -D CONSTX=%d -D DOT8_SWAP=%d",
                      want_rms ? 1 : 0, bias ? 1 : 0, apply_gelu ? 1 : 0, use_i8 ? 1 : 0, wide_on, nout,
                      use_dot8 ? 1 : 0, use_img ? 1 : 0, constx, 0);
        cl_program p = cl_ctx.build_program_from_file("kernels/linear_bias_fused_f32.cl", opts);
        if (p) k_var[variant] = clCreateKernel(p, "linear_fused_f32", &err);
        if (!k_var[variant]) {
            k_bad[variant] = 1;
            std::fprintf(stderr, "GEMV_FUSED variant %s failed to build — using unfused path\n", opts);
        }
    }
    // nnopt_gemv delegates HERE for its int8 shapes, so bouncing back would recurse forever.
    // g_gemv_delegating says "you were called from nnopt_gemv"; return null and let it run its own
    // fp16 path instead.
    if (!k_var[variant]) {
        if (g_gemv_delegating) return nullptr;
        return nnopt_gemv(cl_ctx, weights, queue, x, w_key, rows, in_dim, out_dim, bias);
    }

    cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE,
                            (size_t)rows * out_dim * sizeof(float), nullptr, &err);
    if (!out) { NNOPT_ERROR_FMT("nnopt_gemv_fused: out alloc [%d,%d]", rows, out_dim); return nullptr; }

    // Per-call-site instance of the selected variant, so this site's args survive until replay.
    cl_kernel k = nnopt_kernel_instance(k_var[variant], w_key, ".fused");
    const float eps = 1e-6f;   // same constant the standalone RMSNorm uses
    int ai = 0;
    // Arg 0 is where a __constant refusal shows up. Catch it, disable constx for the process, and
    // let the next call build the __global variant — no flag, no rebuild, no wrong answer.
    if (clSetKernelArg(k, ai++, sizeof(cl_mem), &x) != CL_SUCCESS && constx) {
        constx = 0;
        std::fprintf(stderr, "CONSTX rejected by the driver on a pooled buffer — activation stays "
                             "in __global\n");
        std::fflush(stderr);
        pool_free(out);
        return nnopt_gemv_fused(cl_ctx, weights, queue, x, w_key, rows, in_dim, out_dim, bias,
                                rms_key, apply_gelu);
    }
    clSetKernelArg(k, ai++, sizeof(cl_mem), use_img ? &i8->wimg : (use_i8 ? &i8->w : &W));
    if (use_i8)   clSetKernelArg(k, ai++, sizeof(cl_mem), &i8->scale);
    if (use_dot8) clSetKernelArg(k, ai++, sizeof(cl_mem), &i8->rowsum);
    if (bias)     clSetKernelArg(k, ai++, sizeof(cl_mem), &bias);
    if (want_rms) { clSetKernelArg(k, ai++, sizeof(cl_mem), &rms_scale);
                    clSetKernelArg(k, ai++, sizeof(float),  &eps); }
    clSetKernelArg(k, ai++, sizeof(cl_mem), &out);
    clSetKernelArg(k, ai++, sizeof(int), &in_dim);
    clSetKernelArg(k, ai++, sizeof(int), &out_dim);
    const size_t nwg = ((size_t)out_dim + nout - 1) / nout;
    const size_t gws[2] = {(size_t)rows, nwg * 64}, lws[2] = {1, 64};
    const cl_int eq = cl_ctx.profEnqueue(k, 2, gws, lws, use_i8 ? "gemv_i8" : "gemv_fused");
    if (eq != CL_SUCCESS) {
        NNOPT_ERROR_FMT("nnopt_gemv_fused: enqueue %s err=%d gws=%zux%zu lws=%zux%zu in=%d out=%d",
                        w_key.c_str(), (int)eq, gws[0], gws[1], lws[0], lws[1], in_dim, out_dim);
        pool_free(out);
        return nullptr;
    }
    return out;
}

// ── position buffers ────────────────────────────────────────────────────────
// Two kinds, because a recording pins buffer HANDLES and replays them unchanged:
//
//   varying  — the temporal step's frame index. One buffer whose contents are rewritten between
//              replays, on the LIVE queue. Writing it on the recordable queue would capture the
//              write (or fail): a recordable queue does not execute.
//   constant — the depth body's RVQ level. All 12 levels occur INSIDE a single recorded frame, so
//              one shared buffer cannot serve them — level 3's dispatch would read whatever the
//              last write left. Their values are identical in every frame, so each level gets its
//              own immutable buffer and the recording pins it safely.
static cl_command_queue g_live_queue = nullptr;
void nnopt_set_live_queue(cl_command_queue q) { g_live_queue = q; }
static cl_command_queue nnopt_live_queue() { return g_live_queue; }

cl_mem nnopt_pos_buffer(OpenCLContext& cl_ctx, cl_command_queue queue, int pos, bool constant) {
    cl_int err = CL_SUCCESS;
    if (constant) {
        static std::map<int, cl_mem> consts;
        auto it = consts.find(pos);
        if (it != consts.end()) return it->second;
        cl_mem b = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(int), &pos, &err);
        consts[pos] = b;
        return b;
    }
    static cl_mem buf = nullptr;
    static int last = -0x7fffffff;
    if (!buf) {
        buf = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(int), nullptr, &err);
        if (!buf) return nullptr;
    }
    cl_command_queue wq = g_live_queue ? g_live_queue : queue;
    if (pos != last) {
        // NON-blocking: a blocking write drains the queue and this is hit once per frame — it was
        // CL_TRUE first and cost +70% on AR wall. The staging ring keeps the source alive, since an
        // async write must not point at a dead stack variable.
        static int stage[256];
        static int si = 0;
        stage[si] = pos;
        const cl_int e = clEnqueueWriteBuffer(wq, buf, CL_FALSE, 0, sizeof(int),
                                              &stage[si], 0, nullptr, nullptr);
        si = (si + 1) % 256;
        if (e != CL_SUCCESS) return nullptr;
        last = pos;
    }
    return buf;
}
