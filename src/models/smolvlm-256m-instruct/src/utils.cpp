#include "utils.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used by element_add / pytorch_linear / etc.
#include "opencl_context.h"  // nnopt_build_program_cached — kernel-binary disk cache

#include <clblast.h>       // clblast::Gemm — used by pytorch_linear (dtype-templated dispatch).

#include <fstream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <vector>

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
    err = clEnqueueNDRangeKernel(queue, s_cached_kernel, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("element_add_inplace: clEnqueueNDRangeKernel failed (%d)", err);
        return false;
    }
    return true;
}

cl_mem element_add(cl_command_queue queue, cl_program utils_program, cl_mem a, cl_mem b, size_t n) {
    // element_add is used for residual connections: out = a + b.
    // It MUST treat `n` as authoritative and fail-fast if the caller passes
    // a buffer that is smaller than n * sizeof(nnopt_storage_t).
    //
    // Previous code attempted to "clamp" the copy to avoid CL_INVALID_VALUE.
    // That hides real shape bugs and then the add kernel reads beyond the
    // copied region (undefined), producing garbage while the run appears to
    // succeed.
    cl_int err;
    cl_context ctx;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);

    if (!a || !b) {
        NNOPT_ERROR_FMT("element_add: null input buffer a=%p b=%p", (void*)a, (void*)b);
        return nullptr;
    }

    const size_t elem = sizeof(nnopt_storage_t);
    const size_t requested_bytes = n * elem;

    size_t a_bytes = 0;
    size_t b_bytes = 0;
    (void)clGetMemObjectInfo(a, CL_MEM_SIZE, sizeof(a_bytes), &a_bytes, nullptr);
    (void)clGetMemObjectInfo(b, CL_MEM_SIZE, sizeof(b_bytes), &b_bytes, nullptr);

    if (a_bytes < requested_bytes || b_bytes < requested_bytes) {
        NNOPT_ERROR_FMT(
            "element_add: size mismatch: n=%zu elem=%zu requested=%zu bytes; a_bytes=%zu b_bytes=%zu. "
            "Caller passed wrong n or wrong-shaped buffers.",
            n, elem, requested_bytes, a_bytes, b_bytes);
        return nullptr;
    }

    // Allocate output buffer (storage_t: cl_half under fp16, float under fp32).
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, requested_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !out) {
        NNOPT_ERROR_FMT("element_add: clCreateBuffer failed (%d)", (int)err);
        return nullptr;
    }

    // Copy a into out.
    err = clEnqueueCopyBuffer(queue, a, out, 0, 0, requested_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("element_add: clEnqueueCopyBuffer failed (%d)", (int)err);
        clReleaseMemObject(out);
        return nullptr;
    }

    // Dispatch element_add kernel: out[i] += b[i]
    cl_kernel kernel = clCreateKernel(utils_program, "element_add", &err);
    if (err != CL_SUCCESS || !kernel) {
        NNOPT_ERROR_FMT("element_add: clCreateKernel(\"element_add\") failed (%d)", (int)err);
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
        NNOPT_ERROR_FMT("element_add: clEnqueueNDRangeKernel failed (%d)", (int)err);
        clReleaseKernel(kernel);
        clReleaseMemObject(out);
        return nullptr;
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

#ifdef NNOPT_USE_FP16
// ──────────────────────────────────────────────
// Custom GEMV for the M=1 fp16 decode path.
// CLBlast Hgemm at M=1 is dominated by dispatch + auto-tuning overhead on
// Adreno (measured 3-7 ms / GEMV for our shapes). This singleton kernel runs
// a work-group per output row, lanes reduce across K via half8 vector loads.
// ──────────────────────────────────────────────
static const char* k_gemv_m1_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_SIZE 64

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_fp16(
    __global const half* x,
    __global const half* W,
    __global half* y,
    const int N,
    const int K) {
  const int row = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  if (row >= N) return;
  __global const half* w_row = W + (size_t)row * (size_t)K;

  float acc = 0.0f;
  const int K8 = K >> 3;
  for (int i = lid; i < K8; i += WG_SIZE) {
    float8 xv = convert_float8(vload8(i, x));
    float8 wv = convert_float8(vload8(i, w_row));
    acc += dot(xv.s0123, wv.s0123) + dot(xv.s4567, wv.s4567);
  }
  for (int i = (K8 << 3) + lid; i < K; i += WG_SIZE) {
    acc += (float)x[i] * (float)w_row[i];
  }

  __local float partial[WG_SIZE];
  partial[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) partial[lid] += partial[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) y[row] = (half)partial[0];
}

// Variant: y[row] = (W·x)[row] + residual[row]. Fuses element-wise residual-add
// into the GEMV write so callers can skip a separate add_residual dispatch.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_residual_fp16(
    __global const half* x,
    __global const half* W,
    __global const half* residual,
    __global half* y,
    const int N,
    const int K) {
  const int row = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  if (row >= N) return;
  __global const half* w_row = W + (size_t)row * (size_t)K;

  float acc = 0.0f;
  const int K8 = K >> 3;
  for (int i = lid; i < K8; i += WG_SIZE) {
    float8 xv = convert_float8(vload8(i, x));
    float8 wv = convert_float8(vload8(i, w_row));
    acc += dot(xv.s0123, wv.s0123) + dot(xv.s4567, wv.s4567);
  }
  for (int i = (K8 << 3) + lid; i < K; i += WG_SIZE) {
    acc += (float)x[i] * (float)w_row[i];
  }

  __local float partial[WG_SIZE];
  partial[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) partial[lid] += partial[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) y[row] = (half)(partial[0] + (float)residual[row]);
}
)CLC";

namespace {
struct GemvState {
    bool tried_init = false;
    bool ok = false;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_kernel kernel_residual = nullptr;  // gemv_m1_residual_fp16 — fuses +residual
};

static GemvState& gemv_state() { static GemvState s; return s; }

static bool gemv_ensure_init(cl_command_queue queue) {
    auto& s = gemv_state();
    if (s.tried_init) return s.ok;
    s.tried_init = true;

    cl_context ctx = nullptr;
    cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS) return false;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) return false;

    cl_int err = CL_SUCCESS;
    s.program = nnopt_build_program_cached(ctx, dev, std::string(k_gemv_m1_src),
                                           "-cl-std=CL2.0 -cl-fast-relaxed-math");
    if (!s.program) return false;
    s.kernel = clCreateKernel(s.program, "gemv_m1_fp16", &err);
    if (err != CL_SUCCESS || !s.kernel) return false;
    s.kernel_residual = clCreateKernel(s.program, "gemv_m1_residual_fp16", &err);
    if (err != CL_SUCCESS || !s.kernel_residual) return false;
    s.ok = true;
    return true;
}
}  // end anon namespace for gemv_m1 state

#ifdef NNOPT_USE_FP16
// Public dispatch: fused GEMV + element-wise residual add. Decode-only fast path
// for down_proj (lets caller skip a separate add_residual kernel dispatch).
bool gemv_m1_residual_fp16_dispatch(cl_command_queue queue,
                                    int N, int K,
                                    cl_mem x, cl_mem W, cl_mem residual, cl_mem out) {
    if (!gemv_ensure_init(queue)) return false;
    cl_kernel k = gemv_state().kernel_residual;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &x, "x")) return false;
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &W, "W")) return false;
    if (!set_arg_checked(k, 2, sizeof(cl_mem), &residual, "residual")) return false;
    if (!set_arg_checked(k, 3, sizeof(cl_mem), &out, "y")) return false;
    if (!set_arg_checked(k, 4, sizeof(int), &N, "N")) return false;
    if (!set_arg_checked(k, 5, sizeof(int), &K, "K")) return false;
    const size_t lws[1] = {64};
    const size_t gws[1] = {(size_t)N * lws[0]};
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_residual_fp16 dispatch failed: %d", (int)err);
        return false;
    }
    return true;
}
#endif

// ──────────────────────────────────────────────
// Image2D-backed GEMV — Adreno L1 texture-cache path.
// ──────────────────────────────────────────────
static const char* k_gemv_m1_image_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_SIZE 64

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_image_fp16(
    __global const half* x,
    __read_only image2d_t W,
    __global half* y,
    const int N,
    const int K) {
  const int row = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  if (row >= N) return;

  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
  const int K4 = K >> 2;
  float acc = 0.0f;
  for (int k4 = lid; k4 < K4; k4 += WG_SIZE) {
    float4 xv = convert_float4(vload4(k4, x));
    half4  wh = read_imageh(W, smp, (int2)(k4, row));
    float4 wv = convert_float4(wh);
    acc += dot(xv, wv);
  }

  __local float partial[WG_SIZE];
  partial[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) partial[lid] += partial[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) y[row] = (half)partial[0];
}

// Image-backed GEMV + element-wise residual add (fused-down_proj variant).
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_image_residual_fp16(
    __global const half* x,
    __read_only image2d_t W,
    __global const half* residual,
    __global half* y,
    const int N,
    const int K) {
  const int row = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  if (row >= N) return;

  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
  const int K4 = K >> 2;
  float acc = 0.0f;
  for (int k4 = lid; k4 < K4; k4 += WG_SIZE) {
    float4 xv = convert_float4(vload4(k4, x));
    half4  wh = read_imageh(W, smp, (int2)(k4, row));
    float4 wv = convert_float4(wh);
    acc += dot(xv, wv);
  }

  __local float partial[WG_SIZE];
  partial[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) partial[lid] += partial[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) y[row] = (half)(partial[0] + (float)residual[row]);
}

// Image-backed fused gate+up+SwiGLU (M=1). Two image inputs.
inline float silu_image_f(float v) { return v / (1.0f + native_exp(-v)); }

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_image_swiglu_fp16(
    __global const half* x,
    __read_only image2d_t W_gate,
    __read_only image2d_t W_up,
    __global half* y,
    const int N,
    const int K) {
  const int row = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  if (row >= N) return;

  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
  const int K4 = K >> 2;
  float ag = 0.0f, au = 0.0f;
  for (int k4 = lid; k4 < K4; k4 += WG_SIZE) {
    float4 xv = convert_float4(vload4(k4, x));
    float4 wg = convert_float4(read_imageh(W_gate, smp, (int2)(k4, row)));
    float4 wu = convert_float4(read_imageh(W_up,   smp, (int2)(k4, row)));
    ag += dot(xv, wg);
    au += dot(xv, wu);
  }

  __local float pg[WG_SIZE];
  __local float pu[WG_SIZE];
  pg[lid] = ag; pu[lid] = au;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) { pg[lid] += pg[lid + s]; pu[lid] += pu[lid + s]; }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) y[row] = (half)(silu_image_f(pg[0]) * pu[0]);
}

// Image-backed fused QKV projection (M=1 decode). Three image weights; branch
// on row index into Y_q / Y_k / Y_v.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_qkv_image_fp16(
    __global const half* x,
    __read_only image2d_t W_q,
    __read_only image2d_t W_k,
    __read_only image2d_t W_v,
    __global half* Y_q,
    __global half* Y_k,
    __global half* Y_v,
    const int N_q,
    const int N_kv,
    const int K) {
  const int row = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  const int total = N_q + 2 * N_kv;
  if (row >= total) return;

  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
  int local_row;
  int which;  // 0=q, 1=k, 2=v
  __global half* y_out;
  if (row < N_q) {
    which = 0; local_row = row; y_out = Y_q;
  } else if (row < N_q + N_kv) {
    which = 1; local_row = row - N_q; y_out = Y_k;
  } else {
    which = 2; local_row = row - N_q - N_kv; y_out = Y_v;
  }

  const int K4 = K >> 2;
  float acc = 0.0f;
  for (int k4 = lid; k4 < K4; k4 += WG_SIZE) {
    float4 xv = convert_float4(vload4(k4, x));
    half4  wh;
    // Image2D reads have to be statically dispatched on the image handle.
    // The compiler hoists this branch out of the loop.
    if (which == 0)      wh = read_imageh(W_q, smp, (int2)(k4, local_row));
    else if (which == 1) wh = read_imageh(W_k, smp, (int2)(k4, local_row));
    else                 wh = read_imageh(W_v, smp, (int2)(k4, local_row));
    float4 wv = convert_float4(wh);
    acc += dot(xv, wv);
  }

  __local float partial[WG_SIZE];
  partial[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) partial[lid] += partial[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) y_out[local_row] = (half)partial[0];
}
)CLC";

namespace {
struct GemvImageState {
    bool tried_init = false;
    bool ok = false;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;            // gemv_m1_image_fp16
    cl_kernel kernel_residual = nullptr;   // gemv_m1_image_residual_fp16
    cl_kernel kernel_swiglu = nullptr;     // gemv_m1_image_swiglu_fp16
    cl_kernel kernel_qkv = nullptr;        // gemv_m1_qkv_image_fp16
};
static GemvImageState& gemv_image_state() { static GemvImageState s; return s; }

static bool gemv_image_ensure_init(cl_command_queue queue) {
    auto& s = gemv_image_state();
    if (s.tried_init) return s.ok;
    s.tried_init = true;
    cl_context ctx = nullptr; cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS) return false;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) return false;
    cl_int err = CL_SUCCESS;
    s.program = nnopt_build_program_cached(ctx, dev, std::string(k_gemv_m1_image_src),
                                           "-cl-std=CL2.0 -cl-fast-relaxed-math");
    if (!s.program) return false;
    s.kernel = clCreateKernel(s.program, "gemv_m1_image_fp16", &err);
    if (err != CL_SUCCESS || !s.kernel) return false;
    s.kernel_residual = clCreateKernel(s.program, "gemv_m1_image_residual_fp16", &err);
    if (err != CL_SUCCESS || !s.kernel_residual) return false;
    s.kernel_swiglu = clCreateKernel(s.program, "gemv_m1_image_swiglu_fp16", &err);
    if (err != CL_SUCCESS || !s.kernel_swiglu) return false;
    s.kernel_qkv = clCreateKernel(s.program, "gemv_m1_qkv_image_fp16", &err);
    if (err != CL_SUCCESS || !s.kernel_qkv) return false;
    s.ok = true;
    return true;
}
}  // anon namespace

// Cache of image-view per weight buffer. Created lazily on first request.
static std::unordered_map<cl_mem, cl_mem>& weight_image_cache() {
    static std::unordered_map<cl_mem, cl_mem> c;
    return c;
}

cl_mem get_or_create_weight_image(cl_context ctx, cl_command_queue queue,
                                  cl_mem weight_buf, int N, int K) {
    if (!weight_buf || N <= 0 || K <= 0) return nullptr;
    auto& cache = weight_image_cache();
    auto it = cache.find(weight_buf);
    if (it != cache.end()) return it->second;

    if ((K & 3) != 0) {
        // Image2D layout is CL_RGBA: 4 fp16 per pixel. K must be divisible by 4.
        cache[weight_buf] = nullptr;
        return nullptr;
    }

    // Query device image2d max height — bail (with cache=nullptr) if N exceeds it.
    cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr) != CL_SUCCESS) {
        return nullptr;
    }
    size_t max_h = 0;
    clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(max_h), &max_h, nullptr);
    if (max_h > 0 && (size_t)N > max_h) {
        fprintf(stderr, "[weight_image] N=%d exceeds CL_DEVICE_IMAGE2D_MAX_HEIGHT=%zu — skipping image-backing for this weight\n",
                N, max_h);
        cache[weight_buf] = nullptr;
        return nullptr;
    }

    cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
    cl_image_desc desc{};
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = (size_t)(K / 4);
    desc.image_height = (size_t)N;
    desc.image_depth = 0;

    cl_int err = CL_SUCCESS;
    cl_mem img = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
    if (err != CL_SUCCESS || !img) {
        fprintf(stderr, "[weight_image] clCreateImage failed (err=%d) for N=%d K=%d\n", (int)err, N, K);
        cache[weight_buf] = nullptr;
        return nullptr;
    }

    // Copy buffer → image. Buffer layout matches the natural image row layout
    // (K fp16 per row = K/4 pixels × 4 fp16, same byte stride).
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {(size_t)(K / 4), (size_t)N, 1};
    err = clEnqueueCopyBufferToImage(queue, weight_buf, img, /*src_offset=*/0,
                                     origin, region, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[weight_image] clEnqueueCopyBufferToImage failed (err=%d)\n", (int)err);
        clReleaseMemObject(img);
        cache[weight_buf] = nullptr;
        return nullptr;
    }

    cache[weight_buf] = img;
    return img;
}

#ifdef NNOPT_USE_FP16
bool gemv_m1_image_fp16_dispatch(cl_command_queue queue,
                                 int N, int K,
                                 cl_mem x, cl_mem W_image, cl_mem out) {
    if (!gemv_image_ensure_init(queue)) return false;
    cl_kernel k = gemv_image_state().kernel;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &x, "x")) return false;
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &W_image, "W")) return false;
    if (!set_arg_checked(k, 2, sizeof(cl_mem), &out, "y")) return false;
    if (!set_arg_checked(k, 3, sizeof(int), &N, "N")) return false;
    if (!set_arg_checked(k, 4, sizeof(int), &K, "K")) return false;
    const size_t lws[1] = {64};
    const size_t gws[1] = {(size_t)N * lws[0]};
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_image_fp16 dispatch failed: %d", (int)err);
        return false;
    }
    return true;
}

bool gemv_m1_image_residual_fp16_dispatch(cl_command_queue queue,
                                          int N, int K,
                                          cl_mem x, cl_mem W_image,
                                          cl_mem residual, cl_mem out) {
    if (!gemv_image_ensure_init(queue)) return false;
    cl_kernel k = gemv_image_state().kernel_residual;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &x, "x")) return false;
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &W_image, "W")) return false;
    if (!set_arg_checked(k, 2, sizeof(cl_mem), &residual, "residual")) return false;
    if (!set_arg_checked(k, 3, sizeof(cl_mem), &out, "y")) return false;
    if (!set_arg_checked(k, 4, sizeof(int), &N, "N")) return false;
    if (!set_arg_checked(k, 5, sizeof(int), &K, "K")) return false;
    const size_t lws[1] = {64};
    const size_t gws[1] = {(size_t)N * lws[0]};
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_image_residual_fp16 dispatch failed: %d", (int)err);
        return false;
    }
    return true;
}

bool gemv_m1_qkv_image_fp16_dispatch(cl_command_queue queue,
                                     int N_q, int N_kv, int K,
                                     cl_mem x,
                                     cl_mem W_q_image, cl_mem W_k_image, cl_mem W_v_image,
                                     cl_mem Y_q, cl_mem Y_k, cl_mem Y_v) {
    if (!gemv_image_ensure_init(queue)) return false;
    cl_kernel k = gemv_image_state().kernel_qkv;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &x, "x")) return false;
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &W_q_image, "W_q")) return false;
    if (!set_arg_checked(k, 2, sizeof(cl_mem), &W_k_image, "W_k")) return false;
    if (!set_arg_checked(k, 3, sizeof(cl_mem), &W_v_image, "W_v")) return false;
    if (!set_arg_checked(k, 4, sizeof(cl_mem), &Y_q, "Y_q")) return false;
    if (!set_arg_checked(k, 5, sizeof(cl_mem), &Y_k, "Y_k")) return false;
    if (!set_arg_checked(k, 6, sizeof(cl_mem), &Y_v, "Y_v")) return false;
    if (!set_arg_checked(k, 7, sizeof(int), &N_q, "N_q")) return false;
    if (!set_arg_checked(k, 8, sizeof(int), &N_kv, "N_kv")) return false;
    if (!set_arg_checked(k, 9, sizeof(int), &K, "K")) return false;
    const int total = N_q + 2 * N_kv;
    const size_t lws[1] = {64};
    const size_t gws[1] = {(size_t)total * lws[0]};
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_qkv_image_fp16 dispatch failed: %d", (int)err);
        return false;
    }
    return true;
}

bool gemv_m1_image_swiglu_fp16_dispatch(cl_command_queue queue,
                                        int N, int K,
                                        cl_mem x,
                                        cl_mem W_gate_image, cl_mem W_up_image,
                                        cl_mem out) {
    if (!gemv_image_ensure_init(queue)) return false;
    cl_kernel k = gemv_image_state().kernel_swiglu;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &x, "x")) return false;
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &W_gate_image, "W_gate")) return false;
    if (!set_arg_checked(k, 2, sizeof(cl_mem), &W_up_image, "W_up")) return false;
    if (!set_arg_checked(k, 3, sizeof(cl_mem), &out, "y")) return false;
    if (!set_arg_checked(k, 4, sizeof(int), &N, "N")) return false;
    if (!set_arg_checked(k, 5, sizeof(int), &K, "K")) return false;
    const size_t lws[1] = {64};
    const size_t gws[1] = {(size_t)N * lws[0]};
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_image_swiglu_fp16 dispatch failed: %d", (int)err);
        return false;
    }
    return true;
}
#endif

// Fused gate+up+SwiGLU, TILE_N=2 variant. Each workgroup handles 2 output rows
// to halve the workgroup count (and the per-WG dispatch overhead) while
// keeping the same total compute. Requires N % 2 == 0.
static const char* k_gemv_m1_swiglu_tile2_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_SIZE 64

inline float silu_f(float v) {
  return v / (1.0f + native_exp(-v));
}

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_swiglu_tile2_fp16(
    __global const half* x,
    __global const half* W_gate,
    __global const half* W_up,
    __global half* y,
    const int N,
    const int K) {
  const int wg = (int)get_group_id(0);
  const int row0 = wg * 2;
  const int row1 = row0 + 1;
  const int lid = (int)get_local_id(0);
  if (row0 >= N) return;

  __global const half* wg0 = W_gate + (size_t)row0 * (size_t)K;
  __global const half* wg1 = W_gate + (size_t)row1 * (size_t)K;
  __global const half* wu0 = W_up   + (size_t)row0 * (size_t)K;
  __global const half* wu1 = W_up   + (size_t)row1 * (size_t)K;

  float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;
  const int K8 = K >> 3;
  for (int i = lid; i < K8; i += WG_SIZE) {
    float8 xv = convert_float8(vload8(i, x));
    float8 g0v = convert_float8(vload8(i, wg0));
    float8 g1v = convert_float8(vload8(i, wg1));
    float8 u0v = convert_float8(vload8(i, wu0));
    float8 u1v = convert_float8(vload8(i, wu1));
    ag0 += dot(xv.s0123, g0v.s0123) + dot(xv.s4567, g0v.s4567);
    ag1 += dot(xv.s0123, g1v.s0123) + dot(xv.s4567, g1v.s4567);
    au0 += dot(xv.s0123, u0v.s0123) + dot(xv.s4567, u0v.s4567);
    au1 += dot(xv.s0123, u1v.s0123) + dot(xv.s4567, u1v.s4567);
  }
  for (int i = (K8 << 3) + lid; i < K; i += WG_SIZE) {
    float xv = (float)x[i];
    ag0 += xv * (float)wg0[i];
    ag1 += xv * (float)wg1[i];
    au0 += xv * (float)wu0[i];
    au1 += xv * (float)wu1[i];
  }

  __local float pg0[WG_SIZE], pg1[WG_SIZE], pu0[WG_SIZE], pu1[WG_SIZE];
  pg0[lid] = ag0; pg1[lid] = ag1; pu0[lid] = au0; pu1[lid] = au1;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) {
      pg0[lid] += pg0[lid + s];
      pg1[lid] += pg1[lid + s];
      pu0[lid] += pu0[lid + s];
      pu1[lid] += pu1[lid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    y[row0] = (half)(silu_f(pg0[0]) * pu0[0]);
    if (row1 < N) y[row1] = (half)(silu_f(pg1[0]) * pu1[0]);
  }
}
)CLC";

namespace {
struct GemvSwigluTile2State {
    bool tried_init = false;
    bool ok = false;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
};

static GemvSwigluTile2State& swiglu_tile2_state() { static GemvSwigluTile2State s; return s; }

static bool swiglu_tile2_ensure_init(cl_command_queue queue) {
    auto& s = swiglu_tile2_state();
    if (s.tried_init) return s.ok;
    s.tried_init = true;
    cl_context ctx = nullptr; cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS) return false;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) return false;
    cl_int err = CL_SUCCESS;
    s.program = nnopt_build_program_cached(ctx, dev, std::string(k_gemv_m1_swiglu_tile2_src),
                                           "-cl-std=CL2.0 -cl-fast-relaxed-math");
    if (!s.program) return false;
    s.kernel = clCreateKernel(s.program, "gemv_m1_swiglu_tile2_fp16", &err);
    if (err != CL_SUCCESS || !s.kernel) return false;
    s.ok = true;
    return true;
}
} // anon namespace

bool gemv_m1_swiglu_tile2_fp16_dispatch(cl_command_queue queue,
                                         int N, int K,
                                         cl_mem x, cl_mem W_gate, cl_mem W_up, cl_mem out) {
    if (!swiglu_tile2_ensure_init(queue)) return false;
    cl_kernel k = swiglu_tile2_state().kernel;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &x, "x")) return false;
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &W_gate, "W_gate")) return false;
    if (!set_arg_checked(k, 2, sizeof(cl_mem), &W_up, "W_up")) return false;
    if (!set_arg_checked(k, 3, sizeof(cl_mem), &out, "y")) return false;
    if (!set_arg_checked(k, 4, sizeof(int), &N, "N")) return false;
    if (!set_arg_checked(k, 5, sizeof(int), &K, "K")) return false;
    const size_t lws[1] = {64};
    const size_t gws[1] = {(size_t)((N + 1) / 2) * lws[0]};
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_swiglu_tile2_fp16 dispatch failed: %d", (int)err);
        return false;
    }
    return true;
}

// Fused gate_proj + up_proj + SwiGLU for M=1 decode.
// Computes out[n] = silu(gate_W · x)[n] * (up_W · x)[n]  for n in [0, N).
// Reads x[K] once into local memory and reuses it across both projections.
static const char* k_gemv_m1_swiglu_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_SIZE 64

inline float silu_f(float v) {
  return v / (1.0f + native_exp(-v));
}

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_swiglu_fp16(
    __global const half* x,
    __global const half* W_gate,
    __global const half* W_up,
    __global half* y,
    const int N,
    const int K) {
  const int row = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  if (row >= N) return;
  __global const half* wg_row = W_gate + (size_t)row * (size_t)K;
  __global const half* wu_row = W_up   + (size_t)row * (size_t)K;

  float acc_g = 0.0f;
  float acc_u = 0.0f;
  const int K8 = K >> 3;
  for (int i = lid; i < K8; i += WG_SIZE) {
    float8 xv  = convert_float8(vload8(i, x));
    float8 wgv = convert_float8(vload8(i, wg_row));
    float8 wuv = convert_float8(vload8(i, wu_row));
    acc_g += dot(xv.s0123, wgv.s0123) + dot(xv.s4567, wgv.s4567);
    acc_u += dot(xv.s0123, wuv.s0123) + dot(xv.s4567, wuv.s4567);
  }
  for (int i = (K8 << 3) + lid; i < K; i += WG_SIZE) {
    float xv = (float)x[i];
    acc_g += xv * (float)wg_row[i];
    acc_u += xv * (float)wu_row[i];
  }

  __local float pg[WG_SIZE];
  __local float pu[WG_SIZE];
  pg[lid] = acc_g;
  pu[lid] = acc_u;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) {
      pg[lid] += pg[lid + s];
      pu[lid] += pu[lid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) y[row] = (half)(silu_f(pg[0]) * pu[0]);
}
)CLC";

namespace {
struct GemvSwigluState {
    bool tried_init = false;
    bool ok = false;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
};

GemvSwigluState& swiglu_state() { static GemvSwigluState s; return s; }

bool swiglu_ensure_init(cl_command_queue queue) {
    auto& s = swiglu_state();
    if (s.tried_init) return s.ok;
    s.tried_init = true;
    cl_context ctx = nullptr; cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS) return false;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) return false;
    cl_int err = CL_SUCCESS;
    s.program = nnopt_build_program_cached(ctx, dev, std::string(k_gemv_m1_swiglu_src),
                                           "-cl-std=CL2.0 -cl-fast-relaxed-math");
    if (!s.program) return false;
    s.kernel = clCreateKernel(s.program, "gemv_m1_swiglu_fp16", &err);
    if (err != CL_SUCCESS || !s.kernel) return false;
    s.ok = true;
    return true;
}
} // namespace

// On-device argmax over fp16 logits[V]. Writes a single int32 winner index.
// Single workgroup, 64 lanes; each lane scans a stride of V and tracks its
// local max; final tree reduce. Avoids a ~200 KB device-to-host readback per
// decode token when the sampler is greedy.
static const char* k_argmax_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_SIZE 64

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void argmax_fp16(
    __global const half* logits,
    __global int* out_idx,
    const int N) {
  const int lid = (int)get_local_id(0);
  __local float lmax[WG_SIZE];
  __local int   lidx[WG_SIZE];

  float my_max = -3.402823466e+38f;
  int my_idx = 0;
  for (int i = lid; i < N; i += WG_SIZE) {
    float v = (float)logits[i];
    if (v > my_max) { my_max = v; my_idx = i; }
  }
  lmax[lid] = my_max;
  lidx[lid] = my_idx;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) {
      if (lmax[lid + s] > lmax[lid]) {
        lmax[lid] = lmax[lid + s];
        lidx[lid] = lidx[lid + s];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) out_idx[0] = lidx[0];
}
)CLC";

namespace {
struct ArgmaxState {
    bool tried_init = false;
    bool ok = false;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
};

static ArgmaxState& argmax_state() { static ArgmaxState s; return s; }

static bool argmax_ensure_init(cl_command_queue queue) {
    auto& s = argmax_state();
    if (s.tried_init) return s.ok;
    s.tried_init = true;
    cl_context ctx = nullptr; cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS) return false;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) return false;
    cl_int err = CL_SUCCESS;
    s.program = nnopt_build_program_cached(ctx, dev, std::string(k_argmax_src),
                                           "-cl-std=CL2.0 -cl-fast-relaxed-math");
    if (!s.program) return false;
    s.kernel = clCreateKernel(s.program, "argmax_fp16", &err);
    if (err != CL_SUCCESS || !s.kernel) return false;
    s.ok = true;
    return true;
}
}  // anon namespace

bool argmax_fp16_dispatch(cl_command_queue queue, int N,
                          cl_mem logits, cl_mem out_idx) {
    if (!argmax_ensure_init(queue)) return false;
    cl_kernel k = argmax_state().kernel;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &logits, "logits")) return false;
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &out_idx, "out_idx")) return false;
    if (!set_arg_checked(k, 2, sizeof(int), &N, "N")) return false;
    const size_t lws[1] = {64};
    const size_t gws[1] = {64};
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("argmax_fp16 dispatch failed: %d", (int)err);
        return false;
    }
    return true;
}

// Fused decode-only attention (seq_q=1). One workgroup per query head,
// online softmax over seq_k, head_dim lanes write head_dim outputs in one pass.
// Eliminates 2 of 3 attention dispatches + the scores scratch buffer.
static const char* k_decode_attn_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_SIZE 64

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void decode_attn_fp16(
    __global const half* Q,         // [num_q_heads, head_dim]
    __global const half* K_cache,   // [seq_k_max, num_kv_heads, head_dim]
    __global const half* V_cache,   // [seq_k_max, num_kv_heads, head_dim]
    __global half* out,             // [num_q_heads, head_dim]
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int seq_k,
    const float scale) {
  const int h = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  if (h >= num_q_heads) return;
  const int nrep = num_q_heads / num_kv_heads;
  const int kv_h = h / nrep;
  const int kv_dim = num_kv_heads * head_dim;

  // Load q[h, 0..head_dim] into local memory (head_dim assumed <= WG_SIZE = 64)
  __local float q_local[WG_SIZE];
  q_local[lid] = (lid < head_dim) ? (float)Q[h * head_dim + lid] : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);

  float out_acc = 0.0f;
  float m = -3.402823466e+38f;
  float l = 0.0f;

  __local float reduce[WG_SIZE];
  __local float ab[2];  // ab[0] = a (scale-old), ab[1] = b (current weight)

  for (int s = 0; s < seq_k; ++s) {
    // Each lane computes q[lid] * K[s, kv_h, lid]; reduce across lanes.
    const int k_off = s * kv_dim + kv_h * head_dim;
    float partial = (lid < head_dim) ? q_local[lid] * (float)K_cache[k_off + lid] : 0.0f;
    reduce[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int r = WG_SIZE >> 1; r > 0; r >>= 1) {
      if (lid < r) reduce[lid] += reduce[lid + r];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
      float score = reduce[0] * scale;
      float m_new = fmax(m, score);
      float a_old = native_exp(m - m_new);
      float b_cur = native_exp(score - m_new);
      m = m_new;
      l = l * a_old + b_cur;
      ab[0] = a_old;
      ab[1] = b_cur;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float a_old = ab[0];
    float b_cur = ab[1];
    if (lid < head_dim) {
      const int v_off = s * kv_dim + kv_h * head_dim;
      out_acc = out_acc * a_old + b_cur * (float)V_cache[v_off + lid];
    }
  }

  // Broadcast l and write output.
  __local float l_final;
  if (lid == 0) l_final = l;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < head_dim) {
    out[h * head_dim + lid] = (half)(out_acc / l_final);
  }
}
)CLC";

namespace {
struct DecodeAttnState {
    bool tried_init = false;
    bool ok = false;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
};

static DecodeAttnState& decode_attn_state() { static DecodeAttnState s; return s; }

static bool decode_attn_ensure_init(cl_command_queue queue) {
    auto& s = decode_attn_state();
    if (s.tried_init) return s.ok;
    s.tried_init = true;
    cl_context ctx = nullptr; cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS) return false;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) return false;
    cl_int err = CL_SUCCESS;
    s.program = nnopt_build_program_cached(ctx, dev, std::string(k_decode_attn_src),
                                           "-cl-std=CL2.0 -cl-fast-relaxed-math");
    if (!s.program) return false;
    s.kernel = clCreateKernel(s.program, "decode_attn_fp16", &err);
    if (err != CL_SUCCESS || !s.kernel) return false;
    s.ok = true;
    return true;
}
}  // anon namespace

bool decode_attn_fp16_dispatch(cl_command_queue queue,
                                int num_q_heads, int num_kv_heads, int head_dim, int seq_k,
                                float scale,
                                cl_mem Q, cl_mem K_cache, cl_mem V_cache, cl_mem out) {
    if (!decode_attn_ensure_init(queue)) return false;
    cl_kernel k = decode_attn_state().kernel;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &Q, "Q")) return false;
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &K_cache, "K_cache")) return false;
    if (!set_arg_checked(k, 2, sizeof(cl_mem), &V_cache, "V_cache")) return false;
    if (!set_arg_checked(k, 3, sizeof(cl_mem), &out, "out")) return false;
    if (!set_arg_checked(k, 4, sizeof(int), &num_q_heads, "num_q_heads")) return false;
    if (!set_arg_checked(k, 5, sizeof(int), &num_kv_heads, "num_kv_heads")) return false;
    if (!set_arg_checked(k, 6, sizeof(int), &head_dim, "head_dim")) return false;
    if (!set_arg_checked(k, 7, sizeof(int), &seq_k, "seq_k")) return false;
    if (!set_arg_checked(k, 8, sizeof(float), &scale, "scale")) return false;
    const size_t lws[1] = {64};
    const size_t gws[1] = {(size_t)num_q_heads * lws[0]};
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("decode_attn_fp16 dispatch failed: %d", (int)err);
        return false;
    }
    return true;
}

// Fused Q/K/V projection + RoPE + KV-cache write for M=1 decode.
// Each workgroup handles ONE rope pair (rows `head*head_dim + d_pair` and
// `head*head_dim + d_pair + head_dim/2`). Q rows are RoPE-rotated and written
// to Y_q. K rows are RoPE-rotated and written directly to K_cache at the
// start_pos slot. V rows are written un-rotated to V_cache. Eliminates the
// separate RoPE dispatch AND the `clEnqueueCopyBuffer(K|V → cache)` pair.
static const char* k_gemv_m1_qkv_fused_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_SIZE 64

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_qkv_fused_fp16(
    __global const half* x,
    __global const half* W_q,
    __global const half* W_k,
    __global const half* W_v,
    __global half* Y_q,
    __global half* K_cache,
    __global half* V_cache,
    __global const half* rope_cos,   // [max_seq, head_dim]
    __global const half* rope_sin,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int K,                     // hidden_size
    const int kv_cache_offset,       // = start_pos * (num_kv_heads * head_dim)
    const int cos_row_offset) {      // = start_pos * head_dim
  const int group = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  const int half_dim = head_dim >> 1;
  const int pairs_per_head = half_dim;
  const int q_groups = num_q_heads * pairs_per_head;
  const int k_groups = num_kv_heads * pairs_per_head;

  int head;
  int pair_idx;
  int row0;
  int row1;
  __global const half* w0;
  __global const half* w1;
  __global half* y0;
  __global half* y1;
  int apply_rope;

  if (group < q_groups) {
    head = group / pairs_per_head;
    pair_idx = group - head * pairs_per_head;
    row0 = head * head_dim + pair_idx;
    row1 = row0 + half_dim;
    w0 = W_q + (size_t)row0 * (size_t)K;
    w1 = W_q + (size_t)row1 * (size_t)K;
    y0 = Y_q + row0;
    y1 = Y_q + row1;
    apply_rope = 1;
  } else if (group < q_groups + k_groups) {
    int gi = group - q_groups;
    head = gi / pairs_per_head;
    pair_idx = gi - head * pairs_per_head;
    row0 = head * head_dim + pair_idx;
    row1 = row0 + half_dim;
    w0 = W_k + (size_t)row0 * (size_t)K;
    w1 = W_k + (size_t)row1 * (size_t)K;
    y0 = K_cache + kv_cache_offset + row0;
    y1 = K_cache + kv_cache_offset + row1;
    apply_rope = 1;
  } else {
    int gi = group - q_groups - k_groups;
    head = gi / pairs_per_head;
    pair_idx = gi - head * pairs_per_head;
    row0 = head * head_dim + pair_idx;
    row1 = row0 + half_dim;
    w0 = W_v + (size_t)row0 * (size_t)K;
    w1 = W_v + (size_t)row1 * (size_t)K;
    y0 = V_cache + kv_cache_offset + row0;
    y1 = V_cache + kv_cache_offset + row1;
    apply_rope = 0;
  }

  float acc0 = 0.0f, acc1 = 0.0f;
  const int K8 = K >> 3;
  for (int i = lid; i < K8; i += WG_SIZE) {
    float8 xv = convert_float8(vload8(i, x));
    float8 wv0 = convert_float8(vload8(i, w0));
    float8 wv1 = convert_float8(vload8(i, w1));
    acc0 += dot(xv.s0123, wv0.s0123) + dot(xv.s4567, wv0.s4567);
    acc1 += dot(xv.s0123, wv1.s0123) + dot(xv.s4567, wv1.s4567);
  }
  for (int i = (K8 << 3) + lid; i < K; i += WG_SIZE) {
    float xv = (float)x[i];
    acc0 += xv * (float)w0[i];
    acc1 += xv * (float)w1[i];
  }

  __local float p0[WG_SIZE];
  __local float p1[WG_SIZE];
  p0[lid] = acc0;
  p1[lid] = acc1;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) {
      p0[lid] += p0[lid + s];
      p1[lid] += p1[lid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    float v0 = p0[0];
    float v1 = p1[0];
    if (apply_rope) {
      float c = (float)rope_cos[cos_row_offset + pair_idx];
      float s = (float)rope_sin[cos_row_offset + pair_idx];
      float r0 = v0 * c - v1 * s;
      float r1 = v0 * s + v1 * c;
      *y0 = (half)r0;
      *y1 = (half)r1;
    } else {
      *y0 = (half)v0;
      *y1 = (half)v1;
    }
  }
}
)CLC";

namespace {
struct GemvQkvFusedState {
    bool tried_init = false;
    bool ok = false;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
};

static GemvQkvFusedState& qkv_fused_state() { static GemvQkvFusedState s; return s; }

static bool qkv_fused_ensure_init(cl_command_queue queue) {
    auto& s = qkv_fused_state();
    if (s.tried_init) return s.ok;
    s.tried_init = true;
    cl_context ctx = nullptr; cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS) return false;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) return false;
    cl_int err = CL_SUCCESS;
    s.program = nnopt_build_program_cached(ctx, dev, std::string(k_gemv_m1_qkv_fused_src),
                                           "-cl-std=CL2.0 -cl-fast-relaxed-math");
    if (!s.program) return false;
    s.kernel = clCreateKernel(s.program, "gemv_m1_qkv_fused_fp16", &err);
    if (err != CL_SUCCESS || !s.kernel) return false;
    s.ok = true;
    return true;
}
}  // anon namespace

bool gemv_m1_qkv_fused_fp16_dispatch(cl_command_queue queue,
                                      int num_q_heads, int num_kv_heads, int head_dim, int K_hidden,
                                      int start_pos,
                                      cl_mem x,
                                      cl_mem W_q, cl_mem W_k, cl_mem W_v,
                                      cl_mem Y_q, cl_mem K_cache, cl_mem V_cache,
                                      cl_mem rope_cos, cl_mem rope_sin) {
    if (!qkv_fused_ensure_init(queue)) return false;
    cl_kernel k = qkv_fused_state().kernel;
    const int kv_dim = num_kv_heads * head_dim;
    const int kv_cache_offset = start_pos * kv_dim;
    const int cos_row_offset = start_pos * head_dim;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &x, "x")) return false;
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &W_q, "W_q")) return false;
    if (!set_arg_checked(k, 2, sizeof(cl_mem), &W_k, "W_k")) return false;
    if (!set_arg_checked(k, 3, sizeof(cl_mem), &W_v, "W_v")) return false;
    if (!set_arg_checked(k, 4, sizeof(cl_mem), &Y_q, "Y_q")) return false;
    if (!set_arg_checked(k, 5, sizeof(cl_mem), &K_cache, "K_cache")) return false;
    if (!set_arg_checked(k, 6, sizeof(cl_mem), &V_cache, "V_cache")) return false;
    if (!set_arg_checked(k, 7, sizeof(cl_mem), &rope_cos, "rope_cos")) return false;
    if (!set_arg_checked(k, 8, sizeof(cl_mem), &rope_sin, "rope_sin")) return false;
    if (!set_arg_checked(k, 9, sizeof(int), &num_q_heads, "num_q_heads")) return false;
    if (!set_arg_checked(k, 10, sizeof(int), &num_kv_heads, "num_kv_heads")) return false;
    if (!set_arg_checked(k, 11, sizeof(int), &head_dim, "head_dim")) return false;
    if (!set_arg_checked(k, 12, sizeof(int), &K_hidden, "K")) return false;
    if (!set_arg_checked(k, 13, sizeof(int), &kv_cache_offset, "kv_cache_offset")) return false;
    if (!set_arg_checked(k, 14, sizeof(int), &cos_row_offset, "cos_row_offset")) return false;
    const int pairs_per_head = head_dim >> 1;
    const int total_groups = (num_q_heads + 2 * num_kv_heads) * pairs_per_head;
    const size_t lws[1] = {64};
    const size_t gws[1] = {(size_t)total_groups * lws[0]};
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_qkv_fused_fp16 dispatch failed: %d", (int)err);
        return false;
    }
    return true;
}

// Fused Q/K/V projection for M=1 decode.
// Reads x[K] once and dispatches across (N_q + N_kv + N_kv) output rows in
// a single kernel. Rows in [0, N_q) write into Q[]; [N_q, N_q+N_kv) into K[];
// [N_q+N_kv, N_q+2*N_kv) into V[].
static const char* k_gemv_m1_qkv_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_SIZE 64

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_qkv_fp16(
    __global const half* x,
    __global const half* W_q,
    __global const half* W_k,
    __global const half* W_v,
    __global half* Y_q,
    __global half* Y_k,
    __global half* Y_v,
    const int N_q,
    const int N_kv,
    const int K) {
  const int row = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  const int total = N_q + 2 * N_kv;
  if (row >= total) return;

  __global const half* w_row;
  __global half* y_out;
  int local_row;
  if (row < N_q) {
    w_row = W_q + (size_t)row * (size_t)K;
    y_out = Y_q;
    local_row = row;
  } else if (row < N_q + N_kv) {
    local_row = row - N_q;
    w_row = W_k + (size_t)local_row * (size_t)K;
    y_out = Y_k;
  } else {
    local_row = row - N_q - N_kv;
    w_row = W_v + (size_t)local_row * (size_t)K;
    y_out = Y_v;
  }

  float acc = 0.0f;
  const int K8 = K >> 3;
  for (int i = lid; i < K8; i += WG_SIZE) {
    float8 xv = convert_float8(vload8(i, x));
    float8 wv = convert_float8(vload8(i, w_row));
    acc += dot(xv.s0123, wv.s0123) + dot(xv.s4567, wv.s4567);
  }
  for (int i = (K8 << 3) + lid; i < K; i += WG_SIZE) {
    acc += (float)x[i] * (float)w_row[i];
  }

  __local float partial[WG_SIZE];
  partial[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) partial[lid] += partial[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) y_out[local_row] = (half)partial[0];
}
)CLC";

namespace {
struct GemvQkvState {
    bool tried_init = false;
    bool ok = false;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
};

static GemvQkvState& qkv_state() { static GemvQkvState s; return s; }

static bool qkv_ensure_init(cl_command_queue queue) {
    auto& s = qkv_state();
    if (s.tried_init) return s.ok;
    s.tried_init = true;
    cl_context ctx = nullptr; cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS) return false;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) return false;
    cl_int err = CL_SUCCESS;
    s.program = nnopt_build_program_cached(ctx, dev, std::string(k_gemv_m1_qkv_src),
                                           "-cl-std=CL2.0 -cl-fast-relaxed-math");
    if (!s.program) return false;
    s.kernel = clCreateKernel(s.program, "gemv_m1_qkv_fp16", &err);
    if (err != CL_SUCCESS || !s.kernel) return false;
    s.ok = true;
    return true;
}
}  // anon namespace

bool gemv_m1_qkv_fp16_dispatch(cl_command_queue queue,
                                int N_q, int N_kv, int K,
                                cl_mem x,
                                cl_mem W_q, cl_mem W_k, cl_mem W_v,
                                cl_mem Y_q, cl_mem Y_k, cl_mem Y_v) {
    if (!qkv_ensure_init(queue)) return false;
    cl_kernel k = qkv_state().kernel;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &x, "x")) return false;
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &W_q, "W_q")) return false;
    if (!set_arg_checked(k, 2, sizeof(cl_mem), &W_k, "W_k")) return false;
    if (!set_arg_checked(k, 3, sizeof(cl_mem), &W_v, "W_v")) return false;
    if (!set_arg_checked(k, 4, sizeof(cl_mem), &Y_q, "Y_q")) return false;
    if (!set_arg_checked(k, 5, sizeof(cl_mem), &Y_k, "Y_k")) return false;
    if (!set_arg_checked(k, 6, sizeof(cl_mem), &Y_v, "Y_v")) return false;
    if (!set_arg_checked(k, 7, sizeof(int), &N_q, "N_q")) return false;
    if (!set_arg_checked(k, 8, sizeof(int), &N_kv, "N_kv")) return false;
    if (!set_arg_checked(k, 9, sizeof(int), &K, "K")) return false;
    const size_t lws[1] = {64};
    const size_t gws[1] = {(size_t)(N_q + 2 * N_kv) * lws[0]};
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_qkv_fp16 dispatch failed: %d", (int)err);
        return false;
    }
    return true;
}

bool gemv_m1_swiglu_fp16_dispatch(cl_command_queue queue,
                                  int N, int K,
                                  cl_mem x, cl_mem W_gate, cl_mem W_up, cl_mem out) {
    if (!swiglu_ensure_init(queue)) return false;
    cl_kernel k = swiglu_state().kernel;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &x, "x")) return false;
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &W_gate, "W_gate")) return false;
    if (!set_arg_checked(k, 2, sizeof(cl_mem), &W_up, "W_up")) return false;
    if (!set_arg_checked(k, 3, sizeof(cl_mem), &out, "y")) return false;
    if (!set_arg_checked(k, 4, sizeof(int), &N, "N")) return false;
    if (!set_arg_checked(k, 5, sizeof(int), &K, "K")) return false;
    const size_t lws[1] = {64};
    const size_t gws[1] = {(size_t)N * lws[0]};
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_swiglu_fp16 dispatch failed: %d", (int)err);
        return false;
    }
    return true;
}

static bool gemv_m1_fp16_dispatch_internal(cl_command_queue queue,
                                            int N, int K,
                                            cl_mem x, cl_mem W, cl_mem out) {
    if (!gemv_ensure_init(queue)) return false;
    cl_kernel k = gemv_state().kernel;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &x, "x")) return false;
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &W, "W")) return false;
    if (!set_arg_checked(k, 2, sizeof(cl_mem), &out, "y")) return false;
    if (!set_arg_checked(k, 3, sizeof(int), &N, "N")) return false;
    if (!set_arg_checked(k, 4, sizeof(int), &K, "K")) return false;
    const size_t lws[1] = {64};
    const size_t gws[1] = {(size_t)N * lws[0]};
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_fp16 dispatch failed: %d", (int)err);
        return false;
    }
    return true;
}
#endif

bool pytorch_linear(cl_command_queue queue,
                    int M, int N, int K,
                    cl_mem x, cl_mem W, cl_mem out) {
#ifdef NNOPT_USE_FP16
    // Custom GEMV beats CLBlast Hgemm at M=1 for our shapes (decode path).
    if (M == 1) {
        if (gemv_m1_fp16_dispatch_internal(queue, N, K, x, W, out)) {
            NNOPT_DEBUG_SYNC(queue);
            return true;
        }
        // fall through to CLBlast on failure
    }
#endif
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
        NNOPT_ERROR_FMT(
            "pytorch_linear: CLBlast Gemm failed status=%d %s | M=%d N=%d K=%d | "
            "x=%zub need=%zub (M*K*elem) | "
            "W=%zub need=%zub (N*K*elem; weight stored as [N,K] for nn.Linear) | "
            "out=%zub need=%zub. "
            "If a buffer is SMALLER than 'need', the caller's dim disagrees with the actual weight shape — "
            "look up the weight's shape in weights/model.meta.json and fix the caller's MODEL_CONFIG / "
            "INTERMEDIATE_SIZE_PER_LAYER value (or whichever constant feeds N or K).",
            (int)status, nnopt_clblast_status_name((int)status),
            M, N, K,
            x_bytes_actual, x_bytes_need,
            W_bytes_actual, W_bytes_need,
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
