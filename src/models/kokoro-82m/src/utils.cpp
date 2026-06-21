#include "utils.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used by element_add / pytorch_linear / etc.
#include "profiler.h"      // KernelProfiler::event_for — dormant unless NNOPT_PROFILE=1.
#include "opencl_context.h" // nnopt_build_program_cached — skinny-GEMM program

#include <clblast.h>       // clblast::Gemm — used by pytorch_linear (dtype-templated dispatch).

#include <fstream>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <vector>
#include <map>
#include <cmath>

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


// ── Skinny-GEMM fast path ──────────────────────────────────────────────
// CLBlast Hgemm is heavily tiled for large square GEMMs; at M<=64 (e.g. the
// 31-token BERT pass: 84 GEMMs per inference) it runs ~4 GFLOPS on Adreno.
// This hand-rolled kernel computes out[M,N] = x[M,K] @ W[N,K]^T with one
// work-item per (4 n, 1 m): both x and W rows are contiguous along K, so the
// inner loop is pure vload4 dot-products. Measured well above CLBlast for
// skinny shapes; CLBlast remains the path for M>64.
static const char* k_skinny_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef NNOPT_USE_FP16
typedef half storage_t;
#define LOAD4(p,i) vload_half4(0, ((__global const half*)(p)) + (i))
#define STORE4(v,p,i) vstore_half4((v), 0, ((__global half*)(p)) + (i))
#define STORE(p,i,v) vstore_half((float)(v), (i), (__global half*)(p))
#else
typedef float storage_t;
#define LOAD4(p,i) vload4(0, (__global const float*)(p) + (i))
#define STORE4(v,p,i) vstore4((v), 0, (__global float*)(p) + (i))
#define STORE(p,i,v) ((p)[(i)] = (v))
#endif
// K-reduction lane count for the red/m4/m4_tex kernels (NNOPT_SK_LT sweeps it;
// 64 = the pre-sweep default). Wave-size attribute via NNOPT_SK_WAVE=half|full
// (same §9.2.1 knob that won -31% on conv1d_ht_t8x4).
#ifndef SK_LANES
#define SK_LANES 64
#endif
#if defined(SK_WAVE_HALF)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define SK_WAVE_ATTR __attribute__((qcom_reqd_sub_group_size("half")))
#elif defined(SK_WAVE_FULL)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define SK_WAVE_ATTR __attribute__((qcom_reqd_sub_group_size("full")))
#else
#define SK_WAVE_ATTR
#endif
// Reduction layout: one 64-lane workgroup per (4 n, 1 m) output tile; lanes
// split K in interleaved float4 chunks (lane i reads bytes 16*i, 16*i+1024...
// -> consecutive lanes hit consecutive 16B chunks = fully coalesced), then
// LDS tree-reduce. This keeps the GPU saturated even at M=1 (recurrent
// projections) where a one-WI-per-output layout is nearly serial.
// Wider-tile variant for M>=8, N>=256 (e.g. the 31-token BERT GEMMs): one
// workgroup computes 64 n-values for one m. Lanes = (16 n-quads x 4 k-lanes);
// the x row is staged in LDS once per group (cuts x traffic 16x), each k-lane
// covers a contiguous quarter of K (coalesced W reads), partials reduce in LDS.
__kernel void linear_skinny_w64(__global const storage_t* x,
                                __global const storage_t* W,
                                __global storage_t* out,
                                int M, int N, int K) {
    int gn = (int)get_group_id(0);
    int m  = (int)get_global_id(1);
    int lt = (int)get_local_id(0);
    int nl = lt >> 2;
    int kl = lt & 3;
    int n0 = (gn << 6) + (nl << 2);
    __local float x_lds[768];
    int xb = mul24(m, K);
    for (int k = lt * 4; k < K; k += 256) {
        float4 v = LOAD4(x, xb + k);
        vstore4(v, 0, (__local float*)x_lds + k);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int nc1 = (n0 + 1 < N) ? n0 + 1 : N - 1;
    int nc2 = (n0 + 2 < N) ? n0 + 2 : N - 1;
    int nc3 = (n0 + 3 < N) ? n0 + 3 : N - 1;
    int nb = (n0 < N) ? n0 : N - 1;
    int w0 = mul24(nb, K), w1 = mul24(nc1, K), w2 = mul24(nc2, K), w3 = mul24(nc3, K);
    int quads = K >> 2;
    int per = (quads + 3) >> 2;          // quads per k-lane
    int qb = mul24(kl, per);
    int qe = min(qb + per, quads);
    float4 acc = (float4)(0.0f);
    for (int qq = qb; qq < qe; ++qq) {
        int k = qq << 2;
        float4 xv = vload4(0, (__local float*)x_lds + k);
        acc.x += dot(xv, LOAD4(W, w0 + k));
        acc.y += dot(xv, LOAD4(W, w1 + k));
        acc.z += dot(xv, LOAD4(W, w2 + k));
        acc.w += dot(xv, LOAD4(W, w3 + k));
    }
    __local float4 red[64];
    red[lt] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (kl == 0) {
        float4 r = red[lt] + red[lt + 1] + red[lt + 2] + red[lt + 3];
        if (n0 < N && m < M) {
            int ob = mul24(m, N) + n0;
            if (n0 + 3 < N) {
                STORE4(r, out, ob);
            } else {
                STORE(out, ob, r.x);
                if (n0 + 1 < N) STORE(out, ob + 1, r.y);
                if (n0 + 2 < N) STORE(out, ob + 2, r.z);
            }
        }
    }
}

// 4n x 4m tile per workgroup: x and W vectors are each reused 4x, cutting
// loads-per-output 2.5x vs the 4n x 1m variant. Used when M >= 8.
SK_WAVE_ATTR
__kernel void linear_skinny_red_m4(__global const storage_t* x,
                                   __global const storage_t* W,
                                   __global storage_t* out,
                                   int M, int N, int K) {
    int n0 = (int)get_group_id(0) * 4;
    int m0 = (int)get_group_id(1) * 4;
    int lt = (int)get_local_id(0);
    int n1 = (n0 + 1 < N) ? n0 + 1 : N - 1;
    int n2 = (n0 + 2 < N) ? n0 + 2 : N - 1;
    int n3 = (n0 + 3 < N) ? n0 + 3 : N - 1;
    int w0 = mul24(n0, K), w1 = mul24(n1, K), w2 = mul24(n2, K), w3 = mul24(n3, K);
    int m1 = (m0 + 1 < M) ? m0 + 1 : M - 1;
    int m2 = (m0 + 2 < M) ? m0 + 2 : M - 1;
    int m3 = (m0 + 3 < M) ? m0 + 3 : M - 1;
    int x0 = mul24(m0, K), x1 = mul24(m1, K), x2 = mul24(m2, K), x3 = mul24(m3, K);
    float4 a0 = (float4)(0.0f), a1 = (float4)(0.0f), a2 = (float4)(0.0f), a3 = (float4)(0.0f);
    for (int k = lt * 4; k < K; k += SK_LANES * 4) {
        float4 wv0 = LOAD4(W, w0 + k);
        float4 wv1 = LOAD4(W, w1 + k);
        float4 wv2 = LOAD4(W, w2 + k);
        float4 wv3 = LOAD4(W, w3 + k);
        float4 xv;
        xv = LOAD4(x, x0 + k);
        a0 += (float4)(dot(xv, wv0), dot(xv, wv1), dot(xv, wv2), dot(xv, wv3));
        xv = LOAD4(x, x1 + k);
        a1 += (float4)(dot(xv, wv0), dot(xv, wv1), dot(xv, wv2), dot(xv, wv3));
        xv = LOAD4(x, x2 + k);
        a2 += (float4)(dot(xv, wv0), dot(xv, wv1), dot(xv, wv2), dot(xv, wv3));
        xv = LOAD4(x, x3 + k);
        a3 += (float4)(dot(xv, wv0), dot(xv, wv1), dot(xv, wv2), dot(xv, wv3));
    }
    __local float4 r0[SK_LANES];
    __local float4 r1[SK_LANES];
    __local float4 r2[SK_LANES];
    __local float4 r3[SK_LANES];
    r0[lt] = a0; r1[lt] = a1; r2[lt] = a2; r3[lt] = a3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = SK_LANES >> 1; off > 0; off >>= 1) {
        if (lt < off) { r0[lt] += r0[lt+off]; r1[lt] += r1[lt+off]; r2[lt] += r2[lt+off]; r3[lt] += r3[lt+off]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lt == 0 && n0 < N) {
        #define RM4_ST(J, MM, R) if (MM == m0 + J && m0 + J < M) { \
            int ob = mul24(MM, N) + n0; \
            if (n0 + 3 < N) { STORE4(R[0], out, ob); } \
            else { STORE(out, ob, R[0].x); if (n0+1 < N) STORE(out, ob+1, R[0].y); if (n0+2 < N) STORE(out, ob+2, R[0].z); } }
        RM4_ST(0, m0, r0) RM4_ST(1, m1, r1) RM4_ST(2, m2, r2) RM4_ST(3, m3, r3)
        #undef RM4_ST
    }
}

// Texture-weight m4 variant: W rows come through the TP/L1 pipe as 4-tap
// texels (image row y = n), x stays on the SP/L2 pipe.
#ifdef NNOPT_USE_FP16
#define READ_W4(img, cx, cy) convert_float4(read_imageh((img), g_smp, (int2)((cx),(cy))))
#else
#define READ_W4(img, cx, cy) read_imagef((img), g_smp, (int2)((cx),(cy)))
#endif
__constant sampler_t g_smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
SK_WAVE_ATTR
__kernel void linear_skinny_m4_tex(__global const storage_t* x,
                                   __read_only image2d_t W,
                                   __global storage_t* out,
                                   int M, int N, int K) {
    int n0 = (int)get_group_id(0) * 4;
    int m0 = (int)get_group_id(1) * 4;
    int lt = (int)get_local_id(0);
    int m1 = (m0 + 1 < M) ? m0 + 1 : M - 1;
    int m2 = (m0 + 2 < M) ? m0 + 2 : M - 1;
    int m3 = (m0 + 3 < M) ? m0 + 3 : M - 1;
    int x0 = mul24(m0, K), x1 = mul24(m1, K), x2 = mul24(m2, K), x3 = mul24(m3, K);
    float4 a0 = (float4)(0.0f), a1 = (float4)(0.0f), a2 = (float4)(0.0f), a3 = (float4)(0.0f);
    for (int k = lt * 4; k < K; k += SK_LANES * 4) {
        int kx = k >> 2;
        float4 wv0 = READ_W4(W, kx, n0);
        float4 wv1 = READ_W4(W, kx, n0 + 1);
        float4 wv2 = READ_W4(W, kx, n0 + 2);
        float4 wv3 = READ_W4(W, kx, n0 + 3);
        float4 xv;
        xv = LOAD4(x, x0 + k);
        a0 += (float4)(dot(xv, wv0), dot(xv, wv1), dot(xv, wv2), dot(xv, wv3));
        xv = LOAD4(x, x1 + k);
        a1 += (float4)(dot(xv, wv0), dot(xv, wv1), dot(xv, wv2), dot(xv, wv3));
        xv = LOAD4(x, x2 + k);
        a2 += (float4)(dot(xv, wv0), dot(xv, wv1), dot(xv, wv2), dot(xv, wv3));
        xv = LOAD4(x, x3 + k);
        a3 += (float4)(dot(xv, wv0), dot(xv, wv1), dot(xv, wv2), dot(xv, wv3));
    }
    __local float4 r0[SK_LANES];
    __local float4 r1[SK_LANES];
    __local float4 r2[SK_LANES];
    __local float4 r3[SK_LANES];
    r0[lt] = a0; r1[lt] = a1; r2[lt] = a2; r3[lt] = a3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = SK_LANES >> 1; off > 0; off >>= 1) {
        if (lt < off) { r0[lt] += r0[lt+off]; r1[lt] += r1[lt+off]; r2[lt] += r2[lt+off]; r3[lt] += r3[lt+off]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lt == 0 && n0 < N) {
        #define TM4_ST(J, MM, R) if (MM == m0 + J && m0 + J < M) { \
            int ob = mul24(MM, N) + n0; \
            if (n0 + 3 < N) { STORE4(R[0], out, ob); } \
            else { STORE(out, ob, R[0].x); if (n0+1 < N) STORE(out, ob+1, R[0].y); if (n0+2 < N) STORE(out, ob+2, R[0].z); } }
        TM4_ST(0, m0, r0) TM4_ST(1, m1, r1) TM4_ST(2, m2, r2) TM4_ST(3, m3, r3)
        #undef TM4_ST
    }
}

SK_WAVE_ATTR
__kernel void linear_skinny_red(__global const storage_t* x,
                                __global const storage_t* W,
                                __global storage_t* out,
                                int M, int N, int K) {
    int n0 = (int)get_group_id(0) * 4;
    int lt = (int)get_local_id(0);
    int m  = (int)get_global_id(1);
    float4 acc = (float4)(0.0f);
    int xb = mul24(m, K);
    // N-tail: clamp the row indices (duplicate reads are discarded at store).
    int n1 = (n0 + 1 < N) ? n0 + 1 : N - 1;
    int n2 = (n0 + 2 < N) ? n0 + 2 : N - 1;
    int n3 = (n0 + 3 < N) ? n0 + 3 : N - 1;
    int w0 = mul24(n0, K);
    int w1 = mul24(n1, K), w2 = mul24(n2, K), w3 = mul24(n3, K);
    for (int k = lt * 4; k < K; k += SK_LANES * 4) {
        float4 xv = LOAD4(x, xb + k);
        acc.x += dot(xv, LOAD4(W, w0 + k));
        acc.y += dot(xv, LOAD4(W, w1 + k));
        acc.z += dot(xv, LOAD4(W, w2 + k));
        acc.w += dot(xv, LOAD4(W, w3 + k));
    }
    __local float4 lacc[SK_LANES];
    lacc[lt] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = SK_LANES >> 1; off > 0; off >>= 1) {
        if (lt < off) lacc[lt] += lacc[lt + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lt == 0 && n0 < N && m < M) {
        int ob = mul24(m, N) + n0;
        if (n0 + 3 < N) {
            STORE4(lacc[0], out, ob);
        } else {
            STORE(out, ob, lacc[0].x);
            if (n0 + 1 < N) STORE(out, ob + 1, lacc[0].y);
            if (n0 + 2 < N) STORE(out, ob + 2, lacc[0].z);
        }
    }
}
)CLC";

static bool nnopt_skinny_linear(cl_command_queue queue, int M, int N, int K,
                                cl_mem x, cl_mem W, cl_mem out) {
    static cl_program prog = nullptr;
    static cl_kernel kern = nullptr;
    static cl_kernel kern_w64 = nullptr;
    static cl_kernel kern_m4 = nullptr;
    static cl_kernel kern_m4_tex = nullptr;
    static cl_context built_ctx = nullptr;
    static std::map<cl_mem, cl_mem> w_img_cache;  // weight buffer -> image2d
    cl_context ctx = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS) return false;
    // WG-geometry / wave-size sweep knobs (§6.1.5/§9.2.1, same family as the
    // conv1d_ht_t8x4 NNOPT_HT_* knobs): NNOPT_SK_LT = K-reduction lanes per
    // workgroup (32/64/128; default 64 = pre-sweep behavior), NNOPT_SK_WAVE =
    // half|full forces the wave size (default: compiler heuristic).
    static int sk_lanes = 0;
    if (sk_lanes == 0) {
        const char* e = std::getenv("NNOPT_SK_LT");
        int v = e ? atoi(e) : 64;
        sk_lanes = (v == 32 || v == 64 || v == 128) ? v : 64;
    }
    if (!kern || ctx != built_ctx) {
        cl_device_id dev = nullptr;
        clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr);
#ifdef NNOPT_USE_FP16
        std::string opts = "-cl-fast-relaxed-math -DNNOPT_USE_FP16";
#else
        std::string opts = "-cl-fast-relaxed-math";
#endif
        if (sk_lanes != 64) opts += " -DSK_LANES=" + std::to_string(sk_lanes);
        if (const char* w = std::getenv("NNOPT_SK_WAVE")) {
            if (strcmp(w, "half") == 0) opts += " -DSK_WAVE_HALF=1";
            else if (strcmp(w, "full") == 0) opts += " -DSK_WAVE_FULL=1";
        }
        cl_int err = CL_SUCCESS;
        prog = nnopt_build_program_cached(ctx, dev, k_skinny_src, opts.c_str(), "skinny_gemm", &err);
        if (!prog) return false;
        kern = clCreateKernel(prog, "linear_skinny_red", &err);
        kern_w64 = clCreateKernel(prog, "linear_skinny_w64", &err);
        kern_m4 = clCreateKernel(prog, "linear_skinny_red_m4", &err);
        kern_m4_tex = clCreateKernel(prog, "linear_skinny_m4_tex", &err);
        if (!kern || !kern_w64 || !kern_m4 || !kern_m4_tex) return false;
        built_ctx = ctx;
    }
    if (M >= 8) {
        // Texture path: build (once per weight) an RGBA16F/RGBA32F image over
        // the [N, K] weight; falls back to the buffer kernel on any failure.
        cl_mem wimg = nullptr;
        auto wit = w_img_cache.find(W);
        if (wit != w_img_cache.end()) {
            wimg = wit->second;
        } else {
            size_t img_w = (size_t)(K / 4);
            size_t img_h = (size_t)N;
            if (img_w >= 1 && img_w <= 8192 && img_h <= 8192) {
                cl_image_format fmt;
                fmt.image_channel_order = CL_RGBA;
#ifdef NNOPT_USE_FP16
                fmt.image_channel_data_type = CL_HALF_FLOAT;
#else
                fmt.image_channel_data_type = CL_FLOAT;
#endif
                cl_image_desc desc;
                memset(&desc, 0, sizeof(desc));
                desc.image_type = CL_MEM_OBJECT_IMAGE2D;
                desc.image_width = img_w;
                desc.image_height = img_h;
                cl_int ierr = CL_SUCCESS;
                cl_mem img = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &ierr);
                if (ierr == CL_SUCCESS && img) {
                    size_t origin[3] = {0, 0, 0};
                    size_t region[3] = {img_w, img_h, 1};
                    if (clEnqueueCopyBufferToImage(queue, W, img, 0, origin, region, 0, nullptr, nullptr) == CL_SUCCESS) {
                        wimg = img;
                    } else {
                        clReleaseMemObject(img);
                    }
                }
            }
            w_img_cache[W] = wimg;  // cache nullptr too (don't retry failures)
        }
        if (wimg) {
            clSetKernelArg(kern_m4_tex, 0, sizeof(cl_mem), &x);
            clSetKernelArg(kern_m4_tex, 1, sizeof(cl_mem), &wimg);
            clSetKernelArg(kern_m4_tex, 2, sizeof(cl_mem), &out);
            clSetKernelArg(kern_m4_tex, 3, sizeof(int), &M);
            clSetKernelArg(kern_m4_tex, 4, sizeof(int), &N);
            clSetKernelArg(kern_m4_tex, 5, sizeof(int), &K);
            size_t n_tiles = (size_t)((N + 3) / 4);
            size_t m_tiles = (size_t)((M + 3) / 4);
            size_t gws[2] = {n_tiles * (size_t)sk_lanes, m_tiles};
            size_t lws[2] = {(size_t)sk_lanes, 1};
            return nnopt_enqueue_profiled(queue, kern_m4_tex, 2, nullptr, gws, lws, 0, nullptr, nullptr) == CL_SUCCESS;
        }
        clSetKernelArg(kern_m4, 0, sizeof(cl_mem), &x);
        clSetKernelArg(kern_m4, 1, sizeof(cl_mem), &W);
        clSetKernelArg(kern_m4, 2, sizeof(cl_mem), &out);
        clSetKernelArg(kern_m4, 3, sizeof(int), &M);
        clSetKernelArg(kern_m4, 4, sizeof(int), &N);
        clSetKernelArg(kern_m4, 5, sizeof(int), &K);
        size_t n_tiles = (size_t)((N + 3) / 4);
        size_t m_tiles = (size_t)((M + 3) / 4);
        size_t gws[2] = {n_tiles * (size_t)sk_lanes, m_tiles};
        size_t lws[2] = {(size_t)sk_lanes, 1};
        return nnopt_enqueue_profiled(queue, kern_m4, 2, nullptr, gws, lws, 0, nullptr, nullptr) == CL_SUCCESS;
    }
    if (0 /* w64 measured worse than red on Adreno 620: LDS staging + barriers lose to many small groups */) {
        clSetKernelArg(kern_w64, 0, sizeof(cl_mem), &x);
        clSetKernelArg(kern_w64, 1, sizeof(cl_mem), &W);
        clSetKernelArg(kern_w64, 2, sizeof(cl_mem), &out);
        clSetKernelArg(kern_w64, 3, sizeof(int), &M);
        clSetKernelArg(kern_w64, 4, sizeof(int), &N);
        clSetKernelArg(kern_w64, 5, sizeof(int), &K);
        size_t n_blocks = (size_t)((N + 63) / 64);
        size_t gws[2] = {n_blocks * 64, (size_t)M};
        size_t lws[2] = {64, 1};
        return nnopt_enqueue_profiled(queue, kern_w64, 2, nullptr, gws, lws, 0, nullptr, nullptr) == CL_SUCCESS;
    }
    clSetKernelArg(kern, 0, sizeof(cl_mem), &x);
    clSetKernelArg(kern, 1, sizeof(cl_mem), &W);
    clSetKernelArg(kern, 2, sizeof(cl_mem), &out);
    clSetKernelArg(kern, 3, sizeof(int), &M);
    clSetKernelArg(kern, 4, sizeof(int), &N);
    clSetKernelArg(kern, 5, sizeof(int), &K);
    size_t n_tiles = (size_t)((N + 3) / 4);
    size_t gws[2] = {n_tiles * (size_t)sk_lanes, (size_t)M};
    size_t lws[2] = {(size_t)sk_lanes, 1};
    return nnopt_enqueue_profiled(queue, kern, 2, nullptr, gws, lws, 0, nullptr, nullptr) == CL_SUCCESS;
}

// ── int8 dot8 skinny GEMM (NNOPT_DOT8_GEMM=1) ──────────────────────────────
// Same hardware-dot8 path that gave the resblock convs 1.61x, applied to the
// fp16 skinny GEMMs (plBERT / predictor / flow — 33% of GPU time). Signed
// per-output-row int8 weights x unsigned per-input-row int8 activations (zp=128),
// qcom_dot8_acc over 4-channel groups, dequant out = sx[m]*sw[n]*(A - 128*wsum[n]).
// fp16 build only (the dot8 path is fp16); falls back to the fp16 skinny kernel
// otherwise. A/B-toggle: NNOPT_DOT8_GEMM=0 reverts.
#ifdef NNOPT_USE_FP16
static const char* k_skinny_i8_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable
#ifndef SK_LANES
#define SK_LANES 64
#endif
// Per-input-row (per-m) unsigned act quant, zp=128. One 64-lane WG per row m.
__kernel void gemm_i8_aquant(__global const half* x, __global uint* xq,
                             __global float* sx, int M, int K, int KQ) {
    int m = (int)get_group_id(0);
    int lt = (int)get_local_id(0);
    int xb = mul24(m, K);
    float mx = 0.0f;
    for (int k = lt; k < K; k += 64) mx = fmax(mx, fabs(vload_half(xb + k, x)));
    __local float red[64];
    red[lt] = mx; barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = 32; o > 0; o >>= 1) { if (lt < o) red[lt] = fmax(red[lt], red[lt+o]); barrier(CLK_LOCAL_MEM_FENCE); }
    float s = fmax(red[0], 1e-12f) / 127.0f;
    float inv = 1.0f / s;
    if (lt == 0) sx[m] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kq = lt; kq < KQ; kq += 64) {
        uint v = 0;
        for (int j = 0; j < 4; ++j) {
            int k = (kq << 2) + j;
            float xv = (k < K) ? vload_half(xb + k, x) : 0.0f;   // pad -> 0 -> zp
            int q = clamp(convert_int_rte(xv * inv) + 128, 0, 255);
            v |= ((uint)q) << (8 * j);
        }
        xq[mul24(m, KQ) + kq] = v;
    }
}
// Per-output-row (per-n) symmetric signed int8 weight pack + per-row scale + Sum.
// One 64-lane WG per row n. Cached (static weights) — built once per W.
__kernel void gemm_i8_wpack(__global const half* W, __global uint* wq,
                            __global float* sw, __global int* wsum, int N, int K, int KQ) {
    int n = (int)get_group_id(0);
    int lt = (int)get_local_id(0);
    int wb = mul24(n, K);
    float mx = 0.0f;
    for (int k = lt; k < K; k += 64) mx = fmax(mx, fabs(vload_half(wb + k, W)));
    __local float red[64];
    red[lt] = mx; barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = 32; o > 0; o >>= 1) { if (lt < o) red[lt] = fmax(red[lt], red[lt+o]); barrier(CLK_LOCAL_MEM_FENCE); }
    float s = fmax(red[0], 1e-12f) / 127.0f;
    float inv = 1.0f / s;
    barrier(CLK_LOCAL_MEM_FENCE);
    int acc = 0;
    for (int kq = lt; kq < KQ; kq += 64) {
        uint v = 0;
        for (int j = 0; j < 4; ++j) {
            int k = (kq << 2) + j;
            float wv = (k < K) ? vload_half(wb + k, W) : 0.0f;   // pad -> 0
            int q = clamp(convert_int_rte(wv * inv), -127, 127);
            acc += q;
            v |= ((uint)(q & 0xff)) << (8 * j);
        }
        wq[mul24(n, KQ) + kq] = v;
    }
    __local int redi[64];
    redi[lt] = acc; barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = 32; o > 0; o >>= 1) { if (lt < o) redi[lt] += redi[lt+o]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (lt == 0) { sw[n] = s; wsum[n] = redi[0]; }
}
// The GEMM: 4 n x 4 m output tile per WG, SK_LANES split KQ, int4 accumulators,
// qcom_dot8_acc. Mirrors linear_skinny_m4_tex's reduction layout.
__kernel void linear_skinny_i8(__global const uint* xq, __global const uint* wq,
                               __global const float* sx, __global const float* sw,
                               __global const int* wsum, __global half* out,
                               int M, int N, int K, int KQ) {
    int n0 = (int)get_group_id(0) * 4;
    int m0 = (int)get_group_id(1) * 4;
    int lt = (int)get_local_id(0);
    int m1 = (m0+1<M)?m0+1:M-1, m2 = (m0+2<M)?m0+2:M-1, m3 = (m0+3<M)?m0+3:M-1;
    int n1 = (n0+1<N)?n0+1:N-1, n2 = (n0+2<N)?n0+2:N-1, n3 = (n0+3<N)?n0+3:N-1;
    int4 a0 = (int4)0, a1 = (int4)0, a2 = (int4)0, a3 = (int4)0;
    for (int kq = lt; kq < KQ; kq += SK_LANES) {
        uint w0 = wq[mul24(n0,KQ)+kq], w1 = wq[mul24(n1,KQ)+kq], w2 = wq[mul24(n2,KQ)+kq], w3 = wq[mul24(n3,KQ)+kq];
        uint xm0 = xq[mul24(m0,KQ)+kq], xm1 = xq[mul24(m1,KQ)+kq], xm2 = xq[mul24(m2,KQ)+kq], xm3 = xq[mul24(m3,KQ)+kq];
        a0.x = qcom_dot8_acc(w0,xm0,a0.x); a0.y = qcom_dot8_acc(w1,xm0,a0.y); a0.z = qcom_dot8_acc(w2,xm0,a0.z); a0.w = qcom_dot8_acc(w3,xm0,a0.w);
        a1.x = qcom_dot8_acc(w0,xm1,a1.x); a1.y = qcom_dot8_acc(w1,xm1,a1.y); a1.z = qcom_dot8_acc(w2,xm1,a1.z); a1.w = qcom_dot8_acc(w3,xm1,a1.w);
        a2.x = qcom_dot8_acc(w0,xm2,a2.x); a2.y = qcom_dot8_acc(w1,xm2,a2.y); a2.z = qcom_dot8_acc(w2,xm2,a2.z); a2.w = qcom_dot8_acc(w3,xm2,a2.w);
        a3.x = qcom_dot8_acc(w0,xm3,a3.x); a3.y = qcom_dot8_acc(w1,xm3,a3.y); a3.z = qcom_dot8_acc(w2,xm3,a3.z); a3.w = qcom_dot8_acc(w3,xm3,a3.w);
    }
    __local int4 r0[SK_LANES], r1[SK_LANES], r2[SK_LANES], r3[SK_LANES];
    r0[lt]=a0; r1[lt]=a1; r2[lt]=a2; r3[lt]=a3; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = SK_LANES>>1; off > 0; off >>= 1) {
        if (lt < off) { r0[lt]+=r0[lt+off]; r1[lt]+=r1[lt+off]; r2[lt]+=r2[lt+off]; r3[lt]+=r3[lt+off]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lt == 0) {
        int z0 = wsum[n0]<<7, z1 = wsum[n1]<<7, z2 = wsum[n2]<<7, z3 = wsum[n3]<<7;
        float c0 = sw[n0], c1 = sw[n1], c2 = sw[n2], c3 = sw[n3];
        #define STROW(MM, R) { float sm = sx[MM]; int ob = mul24(MM, N) + n0; \
            vstore_half(sm*c0*(float)(R.x - z0), ob, out); \
            if (n1 > n0)   vstore_half(sm*c1*(float)(R.y - z1), ob+1, out); \
            if (n2 > n0+1) vstore_half(sm*c2*(float)(R.z - z2), ob+2, out); \
            if (n3 > n0+2) vstore_half(sm*c3*(float)(R.w - z3), ob+3, out); }
        STROW(m0, r0[0])
        if (m1 > m0)   STROW(m1, r1[0])
        if (m2 > m0+1) STROW(m2, r2[0])
        if (m3 > m0+2) STROW(m3, r3[0])
        #undef STROW
    }
}
)CLC";

struct I8GemmW { cl_mem wq = nullptr, sw = nullptr, wsum = nullptr; int N = 0, K = 0; };

static bool nnopt_skinny_linear_i8(cl_command_queue queue, int M, int N, int K,
                                   cl_mem x, cl_mem W, cl_mem out) {
    static cl_program prog = nullptr;
    static cl_kernel k_aq = nullptr, k_wp = nullptr, k_gemm = nullptr;
    static cl_context built_ctx = nullptr;
    static std::map<cl_mem, I8GemmW> wcache;   // weight buffer -> packed int8
    static int sk_lanes = 0;
    cl_context ctx = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS) return false;
    if (sk_lanes == 0) {
        const char* e = std::getenv("NNOPT_SK_LT");
        int v = e ? atoi(e) : 64;
        sk_lanes = (v == 32 || v == 64 || v == 128) ? v : 64;
    }
    if (!prog || ctx != built_ctx) {
        cl_device_id dev = nullptr;
        clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr);
        std::string opts = "-cl-fast-relaxed-math -DNNOPT_USE_FP16";
        if (sk_lanes != 64) opts += " -DSK_LANES=" + std::to_string(sk_lanes);
        cl_int err = CL_SUCCESS;
        prog = nnopt_build_program_cached(ctx, dev, k_skinny_i8_src, opts.c_str(), "skinny_gemm_i8", &err);
        if (!prog) return false;
        k_aq   = clCreateKernel(prog, "gemm_i8_aquant", &err);
        k_wp   = clCreateKernel(prog, "gemm_i8_wpack", &err);
        k_gemm = clCreateKernel(prog, "linear_skinny_i8", &err);
        if (!k_aq || !k_wp || !k_gemm) { prog = nullptr; return false; }
        built_ctx = ctx;
    }
    const int KQ = K >> 2;   // caller guarantees K % 4 == 0

    // Weight pack (cached once per W buffer — weights are static).
    auto wit = wcache.find(W);
    if (wit == wcache.end()) {
        I8GemmW e;
        cl_int err = CL_SUCCESS;
        e.wq   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)N * KQ * sizeof(cl_uint), nullptr, &err);
        e.sw   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)N * sizeof(cl_float), nullptr, &err);
        e.wsum = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)N * sizeof(cl_int), nullptr, &err);
        if (!e.wq || !e.sw || !e.wsum) { wcache[W] = I8GemmW{}; return false; }
        e.N = N; e.K = K;
        clSetKernelArg(k_wp, 0, sizeof(cl_mem), &W);
        clSetKernelArg(k_wp, 1, sizeof(cl_mem), &e.wq);
        clSetKernelArg(k_wp, 2, sizeof(cl_mem), &e.sw);
        clSetKernelArg(k_wp, 3, sizeof(cl_mem), &e.wsum);
        clSetKernelArg(k_wp, 4, sizeof(int), &N);
        clSetKernelArg(k_wp, 5, sizeof(int), &K);
        clSetKernelArg(k_wp, 6, sizeof(int), &KQ);
        size_t gws_wp = (size_t)N * 64, lws_wp = 64;
        if (nnopt_enqueue_profiled(queue, k_wp, 1, nullptr, &gws_wp, &lws_wp, 0, nullptr, nullptr) != CL_SUCCESS) {
            wcache[W] = I8GemmW{}; return false;
        }
        wcache[W] = e;
        wit = wcache.find(W);
    }
    if (!wit->second.wq) return false;   // cached failure

    // Per-call activation quant scratch.
    cl_int err = CL_SUCCESS;
    cl_mem xq = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * KQ * sizeof(cl_uint), nullptr, &err);
    cl_mem sx = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * sizeof(cl_float), nullptr, &err);
    if (!xq || !sx) { if (xq) clReleaseMemObject(xq); if (sx) clReleaseMemObject(sx); return false; }
    clSetKernelArg(k_aq, 0, sizeof(cl_mem), &x);
    clSetKernelArg(k_aq, 1, sizeof(cl_mem), &xq);
    clSetKernelArg(k_aq, 2, sizeof(cl_mem), &sx);
    clSetKernelArg(k_aq, 3, sizeof(int), &M);
    clSetKernelArg(k_aq, 4, sizeof(int), &K);
    clSetKernelArg(k_aq, 5, sizeof(int), &KQ);
    size_t gws_aq = (size_t)M * 64, lws_aq = 64;
    bool ok = nnopt_enqueue_profiled(queue, k_aq, 1, nullptr, &gws_aq, &lws_aq, 0, nullptr, nullptr) == CL_SUCCESS;
    if (ok) {
        clSetKernelArg(k_gemm, 0, sizeof(cl_mem), &xq);
        clSetKernelArg(k_gemm, 1, sizeof(cl_mem), &wit->second.wq);
        clSetKernelArg(k_gemm, 2, sizeof(cl_mem), &sx);
        clSetKernelArg(k_gemm, 3, sizeof(cl_mem), &wit->second.sw);
        clSetKernelArg(k_gemm, 4, sizeof(cl_mem), &wit->second.wsum);
        clSetKernelArg(k_gemm, 5, sizeof(cl_mem), &out);
        clSetKernelArg(k_gemm, 6, sizeof(int), &M);
        clSetKernelArg(k_gemm, 7, sizeof(int), &N);
        clSetKernelArg(k_gemm, 8, sizeof(int), &K);
        clSetKernelArg(k_gemm, 9, sizeof(int), &KQ);
        size_t n_tiles = (size_t)((N + 3) / 4);
        size_t m_tiles = (size_t)((M + 3) / 4);
        size_t gws[2] = { n_tiles * (size_t)sk_lanes, m_tiles };
        size_t lws[2] = { (size_t)sk_lanes, 1 };
        ok = nnopt_enqueue_profiled(queue, k_gemm, 2, nullptr, gws, lws, 0, nullptr, nullptr) == CL_SUCCESS;
    }
    clReleaseMemObject(xq);
    clReleaseMemObject(sx);
    return ok;
}
#endif  // NNOPT_USE_FP16

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
    // Skinny fast path: see k_skinny_src above. NNOPT_SKINNY_GEMM=0 disables.
    {
        // NNOPT_SKINNY_GEMM=0 disables; NNOPT_SKINNY_MAX_M=<n> caps which M
        // routes here (bisect tool). Default: all M — every GEMM in this
        // model is skinny enough that the reduction kernel beats CLBlast,
        // and any CLBlast call at all costs ~1.1s of lazy kernel compilation
        // per process (not disk-cached).
        static int max_m = -2;
        if (max_m == -2) {
            const char* e = std::getenv("NNOPT_SKINNY_GEMM");
            if (e && e[0] == '0') max_m = 0;
            else {
                const char* mm = std::getenv("NNOPT_SKINNY_MAX_M");
                max_m = mm ? atoi(mm) : 1 << 30;
            }
        }
#ifdef NNOPT_USE_FP16
        // int8 dot8 GEMM (NNOPT_DOT8_GEMM=1): try first for the M>=8 texture-range
        // GEMMs (plBERT/predictor/flow); on any failure fall through to the fp16
        // skinny kernel below. K>=64 so the quant overhead amortizes.
        static int dot8_gemm = -1;
        if (dot8_gemm == -1) {
            // Default ON (ear-validated 2026-06-08: int8 clip identical to fp16,
            // all 120 GEMMs cos>=0.998, log-STFT cos 0.9957). NNOPT_DOT8_GEMM=0 reverts.
            const char* e = std::getenv("NNOPT_DOT8_GEMM");
            dot8_gemm = (e && e[0] == '0') ? 0 : 1;
        }
        if (dot8_gemm && M <= max_m && M >= 8 && N >= 8 && K >= 64 && (K % 4) == 0 &&
            nnopt_skinny_linear_i8(queue, M, N, K, x, W, out)) {
            // NNOPT_DOT8_GEMM_VERIFY=1: recompute fp16 + print per-shape cos.
            static int v8 = -1;
            if (v8 == -1) { const char* e = std::getenv("NNOPT_DOT8_GEMM_VERIFY"); v8 = (e && e[0]=='1') ? 1 : 0; }
            if (v8) {
                cl_context vc = nullptr; clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(vc), &vc, nullptr);
                cl_mem ref = clCreateBuffer(vc, CL_MEM_READ_WRITE, (size_t)M*N*sizeof(nnopt_storage_t), nullptr, nullptr);
                if (ref && nnopt_skinny_linear(queue, M, N, K, x, W, ref)) {
                    std::vector<uint16_t> ha((size_t)M*N), hb((size_t)M*N);
                    clEnqueueReadBuffer(queue, out, CL_TRUE, 0, ha.size()*2, ha.data(), 0, nullptr, nullptr);
                    clEnqueueReadBuffer(queue, ref, CL_TRUE, 0, hb.size()*2, hb.data(), 0, nullptr, nullptr);
                    double dot=0,na=0,nb=0,maxd=0;
                    for (size_t i=0;i<ha.size();++i){ float a=nnopt_f16_to_f32(ha[i]),b=nnopt_f16_to_f32(hb[i]); dot+=(double)a*b;na+=(double)a*a;nb+=(double)b*b; double d=std::fabs((double)a-b); if(d>maxd)maxd=d; }
                    fprintf(stderr, "DOT8GEMM_VERIFY M=%d N=%d K=%d cos=%.6f maxd=%.4f\n", M,N,K,(na>0&&nb>0)?dot/std::sqrt(na*nb):-2,maxd);
                }
                if (ref) clReleaseMemObject(ref);
            }
            return true;
        }
#endif
        if (M <= max_m && (K % 4) == 0 &&
            nnopt_skinny_linear(queue, M, N, K, x, W, out)) {
            // NNOPT_SKINNY_VERIFY=1: recompute with CLBlast and print the max
            // relative mismatch (debug-only path; leaks nothing on the normal path).
            static int verify = -1;
            if (verify == -1) { const char* v = std::getenv("NNOPT_SKINNY_VERIFY"); verify = (v && v[0]=='1') ? 1 : 0; }
            if (verify) {
                cl_context vctx = nullptr;
                clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(vctx), &vctx, nullptr);
                cl_mem ref = clCreateBuffer(vctx, CL_MEM_READ_WRITE, (size_t)M*N*sizeof(nnopt_storage_t), nullptr, nullptr);
#ifdef NNOPT_USE_FP16
                cl_half vh_one = static_cast<cl_half>(nnopt_f32_to_f16(1.0f));
                cl_half vh_zero = static_cast<cl_half>(nnopt_f32_to_f16(0.0f));
                clblast::Gemm<cl_half>(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kYes,
                                       M, N, K, vh_one, x, 0, K, W, 0, K, vh_zero, ref, 0, N, &queue, nullptr);
                std::vector<uint16_t> ha((size_t)M*N), hb((size_t)M*N);
                clEnqueueReadBuffer(queue, out, CL_TRUE, 0, ha.size()*2, ha.data(), 0, nullptr, nullptr);
                clEnqueueReadBuffer(queue, ref, CL_TRUE, 0, hb.size()*2, hb.data(), 0, nullptr, nullptr);
                double dot=0, na=0, nb=0; double maxd = 0; int maxi = -1;
                for (size_t i = 0; i < ha.size(); ++i) {
                    float fa = nnopt_f16_to_f32(ha[i]), fb = nnopt_f16_to_f32(hb[i]);
                    dot += (double)fa*fb; na += (double)fa*fa; nb += (double)fb*fb;
                    double d = std::fabs((double)fa - fb);
                    if (d > maxd) { maxd = d; maxi = (int)i; }
                }
                double cos = (na>0&&nb>0) ? dot/std::sqrt(na*nb) : -2;
                fprintf(stderr, "SKINNY_VERIFY M=%d N=%d K=%d cos=%.6f maxdiff=%.5f at %d\n", M, N, K, cos, maxd, maxi);
                clReleaseMemObject(ref);
#endif
            }
            return true;
        }
    }
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
