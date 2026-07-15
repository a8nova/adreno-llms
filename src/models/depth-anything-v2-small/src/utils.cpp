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
