#include "utils.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used by element_add / pytorch_linear / etc.
#include "profiler.h"      // KernelProfiler::event_for — dormant unless NNOPT_PROFILE=1.

#include <clblast.h>       // clblast::Gemm — used by pytorch_linear (dtype-templated dispatch).

#include <fstream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdlib>

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

// ── Per-token buffer pool (kills clCreateBuffer/clReleaseMemObject churn) ──
// Decode re-runs the same op graph every token, allocating identically-sized
// scratch buffers and freeing them — ~300 alloc/free pairs/token on Adreno, a
// measured ~half of the decoder-loop host wall (GPU was only ~51% busy). The
// pool keeps freed buffers and hands the same one back next token (exact-size
// match), so after token 1 the steady state issues ZERO clCreateBuffer/Release.
// Safety: nnopt_pool_free recycles pool buffers but REAL-releases anything not
// in the pool, so converting every per-token release to it is always correct.
// Semantics are identical to clCreateBuffer: contents are undefined on alloc and
// every op fully overwrites its output before reading (same as before pooling).
// NNOPT_NO_POOL=1 disables (pass-through to clCreateBuffer/clReleaseMemObject).
namespace {
struct PoolEntry { cl_mem mem; size_t bytes; bool in_use; };
std::vector<PoolEntry> g_buf_pool;
bool nnopt_pool_disabled() {
    static int d = -1;
    if (d == -1) { const char* s = std::getenv("NNOPT_NO_POOL"); d = (s && s[0] == '1') ? 1 : 0; }
    return d == 1;
}
} // namespace

cl_mem nnopt_pool_alloc(cl_context ctx, size_t bytes, cl_int* err_out) {
    cl_int err = CL_SUCCESS;
    if (!nnopt_pool_disabled()) {
        for (auto& e : g_buf_pool)
            if (!e.in_use && e.bytes == bytes) { e.in_use = true; if (err_out) *err_out = CL_SUCCESS; return e.mem; }
    }
    cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err_out) *err_out = err;
    if (err != CL_SUCCESS || !m) { NNOPT_ERROR_FMT("pool_alloc: clCreateBuffer %d (%zu bytes)", (int)err, bytes); return nullptr; }
    if (!nnopt_pool_disabled()) g_buf_pool.push_back({m, bytes, true});
    return m;
}

void nnopt_pool_free(cl_mem mem) {
    if (!mem) return;
    if (!nnopt_pool_disabled()) {
        for (auto& e : g_buf_pool)
            if (e.mem == mem) { e.in_use = false; return; }
    }
    clReleaseMemObject(mem);   // not a pool buffer (or pool disabled)
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

// Build + cache the gemma3_ops.cl program (device/context pulled from the
// queue so pytorch_linear keeps its signature). Built once for the process.
static cl_program nnopt_get_ops_program(cl_command_queue queue) {
    static cl_program s_prog = nullptr;
    static bool s_tried = false;
    if (s_prog) return s_prog;
    if (s_tried) return nullptr;
    s_tried = true;

    cl_context ctx = nullptr;
    cl_device_id dev = nullptr;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);
    clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr);
    if (!ctx || !dev) { NNOPT_ERROR("ops program: no context/device from queue"); return nullptr; }

    std::ifstream file("kernels/gemma3_ops.cl");
    if (!file.is_open()) { NNOPT_ERROR("ops program: cannot open kernels/gemma3_ops.cl"); return nullptr; }
    std::string src((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const char* src_ptr = src.c_str();
    size_t src_len = src.size();

    cl_int err = CL_SUCCESS;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS || !prog) { NNOPT_ERROR_FMT("ops program: clCreateProgramWithSource %d", (int)err); return nullptr; }
    std::string opts;
#ifdef NNOPT_USE_FP16
    opts = "-D USE_FP16=1";
#endif
    // NNOPT_SHFL=1: compile the experimental subgroup-shuffle GEMV. Gated so the
    // default program never references subgroup intrinsics (which -11 on this driver).
    if (const char* s = std::getenv("NNOPT_SHFL"); s && s[0] == '1') opts += " -D ENABLE_SHFL=1";
    err = clBuildProgram(prog, 1, &dev, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size + 1, 0);
        if (log_size) clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        NNOPT_ERROR_FMT("ops program: clBuildProgram %d log: %s", (int)err, log.data());
        clReleaseProgram(prog);
        return nullptr;
    }
    s_prog = prog;
    return s_prog;
}

static cl_kernel nnopt_get_named_kernel(cl_command_queue queue, const char* name, cl_kernel* slot) {
    if (*slot) return *slot;
    cl_program prog = nnopt_get_ops_program(queue);
    if (!prog) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(prog, name, &err);
    if (err != CL_SUCCESS || !k) { NNOPT_ERROR_FMT("kernel %s: clCreateKernel %d", name, (int)err); return nullptr; }
    *slot = k;
    return k;
}

// Transposed-weight cache for the coalesced grid-N GEMV (lm_head). The weight
// W[N,K] is physically transposed to Wt[K,N] once on first use and kept, so the
// transposed GEMV's adjacent work-items read adjacent addresses. Keyed by W.
static std::unordered_map<cl_mem, cl_mem> g_wt_cache;
static cl_mem nnopt_get_transposed_weight(cl_command_queue queue, cl_mem W, int N, int K) {
    auto it = g_wt_cache.find(W);
    if (it != g_wt_cache.end()) return it->second;
    cl_context ctx = nullptr;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);
    if (!ctx) { NNOPT_ERROR("transpose: no context"); return nullptr; }
    cl_int err = CL_SUCCESS;
    cl_mem Wt = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                               (size_t)N * (size_t)K * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !Wt) { NNOPT_ERROR_FMT("transpose: alloc Wt %d (N=%d K=%d)", (int)err, N, K); return nullptr; }
    static cl_kernel s_tk = nullptr;
    cl_kernel tk = nnopt_get_named_kernel(queue, "gemma3_transpose", &s_tk);
    if (!tk) { clReleaseMemObject(Wt); return nullptr; }
    if (!set_arg_checked(tk, 0, sizeof(cl_mem), &W,  "W")  ||
        !set_arg_checked(tk, 1, sizeof(cl_mem), &Wt, "Wt") ||
        !set_arg_checked(tk, 2, sizeof(int), &N, "N")      ||
        !set_arg_checked(tk, 3, sizeof(int), &K, "K")) { clReleaseMemObject(Wt); return nullptr; }
    const size_t TILE = 16;   // must match TR_TILE in gemma3_ops.cl
    size_t lws[2] = { TILE, TILE };
    size_t gws[2] = { ((size_t)K + TILE - 1) / TILE * TILE,
                      ((size_t)N + TILE - 1) / TILE * TILE };
    err = clEnqueueNDRangeKernel(queue, tk, 2, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("transpose: dispatch %d", (int)err); clReleaseMemObject(Wt); return nullptr; }
    g_wt_cache[W] = Wt;
    return Wt;
}
// Default ON: the transposed GEMV (one work-item per output, x cached in local,
// lws=256) is globally coalesced — in-flight wavefronts march through Wt row by
// row like the bandwidth microbench. Captures ~9 GB/s where the per-output
// reduction GEMV stalls at ~4 GB/s. (The earlier "transposed is slower" verdict
// was a kernel WITHOUT local-x caching and with a bad NULL workgroup size.)
// NNOPT_GEMV_T=0 reverts to the reduction GEMV.
static bool nnopt_gemv_t_enabled() {
    static int e = -1;
    if (e == -1) { const char* s = std::getenv("NNOPT_GEMV_T"); e = (s && s[0] == '0') ? 0 : 1; }
    return e == 1;
}
// Min N for the transposed path. Default 100000 (lm_head only); small proj GEMVs
// under-occupy a grid-N launch so they stay on the reduction kernel. NNOPT_GEMVT_MINN tunes.
static int nnopt_gemvt_min_n() {
    static int v = -1;
    if (v == -1) { const char* s = std::getenv("NNOPT_GEMVT_MINN"); v = (s ? std::atoi(s) : 100000); if (v <= 0) v = 100000; }
    return v;
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
    // Custom fp16/fp32-accum GEMV (workgroup-per-output reduction), replacing
    // CLBlast (its ~3.6ms/call host setup killed tiny decode GEMMs). fp32
    // accumulation preserves the near-tie argmax. lm_head additionally uses the
    // transposed GEMV below (large N → globally coalesced, ~9 GB/s).
    // Reduction width per output. Default 64; NNOPT_GEMV_WG sweeps {8,16,32,64,128}
    // to find the point that captures the device's streaming bandwidth (the bw
    // microbench shows grid size dominates — too-many tiny work-items stall).
    static size_t WG = 0;
    if (WG == 0) {
        WG = 64;
        if (const char* s = std::getenv("NNOPT_GEMV_WG")) {
            size_t v = (size_t)std::atoi(s);
            if (v == 8 || v == 16 || v == 32 || v == 64 || v == 128) WG = v;
        }
    }
    cl_event* gemm_evt = KernelProfiler::event_for(N >= 100000 ? "op_gemm_lmhead" : "op_gemm_proj");
    KernelProfiler::HostTimer _ht(N >= 100000 ? "host_gemm_lmhead" : "host_gemm_proj");
    // Large-N (lm_head, N≈262144 ≈ grid sweet spot): transposed grid-N GEMV.
    // One work-item per output, x cached in local, fully coalesced → captures the
    // device streaming bandwidth (~9 GB/s) the per-output reduction GEMV can't
    // (~4 GB/s). Small proj GEMVs stay on the reduction kernel (their tiny N
    // would under-occupy a grid-N launch). M==1 only (decode); prefill M>1 uses
    // the reduction path which handles arbitrary M.
    if (nnopt_gemv_t_enabled() && M == 1 && N >= nnopt_gemvt_min_n()) {
        cl_mem Wt = nnopt_get_transposed_weight(queue, W, N, K);
        if (Wt) {
            static cl_kernel s_gt = nullptr;
            cl_kernel gtk = nnopt_get_named_kernel(queue, "gemma3_gemv_t", &s_gt);
            if (gtk &&
                set_arg_checked(gtk, 0, sizeof(cl_mem), &x,   "x")   &&
                set_arg_checked(gtk, 1, sizeof(cl_mem), &Wt,  "Wt")  &&
                set_arg_checked(gtk, 2, sizeof(cl_mem), &out, "out") &&
                set_arg_checked(gtk, 3, sizeof(int), &M, "M")        &&
                set_arg_checked(gtk, 4, sizeof(int), &N, "N")        &&
                set_arg_checked(gtk, 5, sizeof(int), &K, "K")) {
                size_t lws = 256;
                if (const char* s = std::getenv("NNOPT_GEMVT_LWS")) { int v = std::atoi(s); if (v > 0) lws = (size_t)v; }
                size_t gws = ((size_t)N + lws - 1) / lws * lws;   // pad to lws multiple
                cl_int derr = clEnqueueNDRangeKernel(queue, gtk, 1, nullptr, &gws, &lws,
                                                     0, nullptr, gemm_evt);
                if (derr == CL_SUCCESS) { NNOPT_DEBUG_SYNC(queue); return true; }
                NNOPT_ERROR_FMT("pytorch_linear: gemv_t dispatch %d (N=%d K=%d) — falling back", (int)derr, N, K);
            }
        }
        // Any failure falls through to the reduction GEMV below.
    }
    // Persistent-workgroup GEMV for large-N (lm_head), M==1: a fixed number of
    // workgroups grid-stride over outputs, caching x in local. Cuts the 262144
    // workgroup launches of the per-output kernel. NNOPT_GEMV_PERSIST=1 to enable;
    // NNOPT_GEMV_NG sets the workgroup count (default 1024).
    if (const char* ps = std::getenv("NNOPT_GEMV_PERSIST"); ps && ps[0] == '1' && M == 1 && N >= 100000) {
        static cl_kernel s_gp = nullptr;
        cl_kernel gpk = nnopt_get_named_kernel(queue, "gemma3_gemv_persist", &s_gp);
        if (gpk &&
            set_arg_checked(gpk, 0, sizeof(cl_mem), &x,   "x")   &&
            set_arg_checked(gpk, 1, sizeof(cl_mem), &W,   "W")   &&
            set_arg_checked(gpk, 2, sizeof(cl_mem), &out, "out") &&
            set_arg_checked(gpk, 3, sizeof(int), &M, "M")        &&
            set_arg_checked(gpk, 4, sizeof(int), &N, "N")        &&
            set_arg_checked(gpk, 5, sizeof(int), &K, "K")) {
            size_t ng = 1024;
            if (const char* s = std::getenv("NNOPT_GEMV_NG")) { int v = std::atoi(s); if (v > 0) ng = (size_t)v; }
            size_t lws = 64, gws = ng * lws;
            cl_int derr = clEnqueueNDRangeKernel(queue, gpk, 1, nullptr, &gws, &lws, 0, nullptr, gemm_evt);
            if (derr == CL_SUCCESS) { NNOPT_DEBUG_SYNC(queue); return true; }
            NNOPT_ERROR_FMT("pytorch_linear: gemv_persist dispatch %d — falling back", (int)derr);
        }
    }
    // NNOPT_GEMV_SG=1 swaps the 6-barrier reduction for the subgroup-reduction
    // GEMV (sub_group_reduce_add, barrier-free) — same coalesced structure, same
    // args/dispatch. Drop-in for both lm_head and proj.
    static int gemv_variant = -1;  // 0=reduction(default) 1=warp 2=shfl
    if (gemv_variant == -1) {
        gemv_variant = 0;
        if (const char* s = std::getenv("NNOPT_WARP"); s && s[0] == '1') gemv_variant = 1;
        if (const char* s = std::getenv("NNOPT_SHFL"); s && s[0] == '1') gemv_variant = 2;
    }
    static cl_kernel s_gemm = nullptr, s_gemm_warp = nullptr, s_gemm_shfl = nullptr;
    cl_kernel gk =
        gemv_variant == 1 ? nnopt_get_named_kernel(queue, "gemma3_gemm_warp", &s_gemm_warp) :
        gemv_variant == 2 ? nnopt_get_named_kernel(queue, "gemma3_gemm_shfl", &s_gemm_shfl) :
                            nnopt_get_named_kernel(queue, "gemma3_gemm_linear", &s_gemm);
    if (!gk && gemv_variant) gk = nnopt_get_named_kernel(queue, "gemma3_gemm_linear", &s_gemm);  // fallback
    if (!gk) { NNOPT_ERROR("pytorch_linear: gemm kernel unavailable"); return false; }
    if (!set_arg_checked(gk, 0, sizeof(cl_mem), &x,   "x"))   return false;
    if (!set_arg_checked(gk, 1, sizeof(cl_mem), &W,   "W"))   return false;
    if (!set_arg_checked(gk, 2, sizeof(cl_mem), &out, "out")) return false;
    if (!set_arg_checked(gk, 3, sizeof(int), &M, "M"))        return false;
    if (!set_arg_checked(gk, 4, sizeof(int), &N, "N"))        return false;
    if (!set_arg_checked(gk, 5, sizeof(int), &K, "K"))        return false;
    {
        size_t gws[2] = { (size_t)N * WG, (size_t)M };
        size_t lws[2] = { WG, 1 };
        cl_int derr = clEnqueueNDRangeKernel(queue, gk, 2, nullptr, gws, lws,
                                             0, nullptr, gemm_evt);
        if (derr != CL_SUCCESS) {
            NNOPT_ERROR_FMT("pytorch_linear: gemm dispatch %d (M=%d N=%d K=%d)", (int)derr, M, N, K);
            return false;
        }
    }
    NNOPT_DEBUG_SYNC(queue);
    return true;
#else
    cl_event* gemm_evt = KernelProfiler::event_for(N >= 100000 ? "op_clblast_gemm_lmhead" : "op_clblast_gemm_proj");
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
        gemm_evt);
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
#endif
}

#ifdef NNOPT_USE_FP16
// Concatenate 2-3 nn.Linear weights [Ni,K] into one [sum(Ni),K] buffer, once,
// cached by the first weight's cl_mem. Pure row-major byte copies (no transpose).
static std::unordered_map<cl_mem, cl_mem> g_concat_cache;
static cl_mem nnopt_get_concat_weight(cl_command_queue queue,
                                      const cl_mem* Ws, const int* Ns, int count, int K) {
    auto it = g_concat_cache.find(Ws[0]);
    if (it != g_concat_cache.end()) return it->second;
    cl_context ctx = nullptr;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);
    if (!ctx) { NNOPT_ERROR("concat: no context"); return nullptr; }
    int rows = 0; for (int i = 0; i < count; ++i) rows += Ns[i];
    cl_int err = CL_SUCCESS;
    cl_mem cat = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                               (size_t)rows * (size_t)K * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !cat) { NNOPT_ERROR_FMT("concat: alloc %d (rows=%d K=%d)", (int)err, rows, K); return nullptr; }
    size_t off = 0;
    for (int i = 0; i < count; ++i) {
        size_t bytes = (size_t)Ns[i] * (size_t)K * sizeof(nnopt_storage_t);
        err = clEnqueueCopyBuffer(queue, Ws[i], cat, 0, off, bytes, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("concat: copy %d (i=%d)", (int)err, i); clReleaseMemObject(cat); return nullptr; }
        off += bytes;
    }
    g_concat_cache[Ws[0]] = cat;
    return cat;
}
#endif

bool pytorch_linear_fused3(cl_command_queue queue, int M,
                           int N0, int N1, int N2, int K,
                           cl_mem x, cl_mem W0, cl_mem W1, cl_mem W2,
                           cl_mem out0, cl_mem out1, cl_mem out2) {
#ifdef NNOPT_USE_FP16
    const cl_mem Ws[3] = { W0, W1, W2 };
    const int    Ns[3] = { N0, N1, N2 };
    const int count = (N2 > 0) ? 3 : 2;
    cl_mem cat = nnopt_get_concat_weight(queue, Ws, Ns, count, K);
    const int Ntot_all = N0 + N1 + N2;
    // Transposed fused GEMV (globally coalesced) for medium/large Ntot, M==1.
    // NNOPT_FUSED_T=0 disables; NNOPT_FUSEDT_MINN sets the Ntot floor (default
    // 1500 — covers gate/up Ntot=4096 and qkv Ntot=1536; smaller would under-occupy).
    // Default OFF: transposing the medium-N fused proj under-occupies (gate/up
    // Ntot=4096 → too few workgroups) and measured slower than the reduction GEMV.
    static int fused_t = -1, fused_t_minn = -1, fused_t_lws = -1;
    if (fused_t == -1) { const char* s = std::getenv("NNOPT_FUSED_T"); fused_t = (s && s[0] == '1') ? 1 : 0; }
    if (fused_t_minn == -1) { const char* s = std::getenv("NNOPT_FUSEDT_MINN"); fused_t_minn = s ? std::atoi(s) : 1500; if (fused_t_minn <= 0) fused_t_minn = 1500; }
    if (fused_t_lws == -1) { const char* s = std::getenv("NNOPT_FUSEDT_LWS"); fused_t_lws = s ? std::atoi(s) : 256; if (fused_t_lws <= 0) fused_t_lws = 256; }
    // Split-K transposed fused GEMV. Default OFF: it only TIES the WG=64 reduction
    // (~4 GB/s — the device's small-N matvec bandwidth limit, which transposed,
    // warp, and split-K all hit) while costing ~130MB of transposed-weight memory.
    static int tsk = -1, tsk_p = -1;
    if (tsk == -1) { const char* s = std::getenv("NNOPT_TSK"); tsk = (s && s[0] == '1') ? 1 : 0; }
    if (tsk_p == -1) { const char* s = std::getenv("NNOPT_TSK_P"); tsk_p = s ? std::atoi(s) : 8; if (tsk_p < 1 || tsk_p > 16) tsk_p = 8; }
    if (cat && M == 1 && tsk) {
        cl_mem Wt = nnopt_get_transposed_weight(queue, cat, Ntot_all, K);  // [K, Ntot]
        if (Wt) {
            static cl_kernel s_tsk = nullptr;
            cl_kernel tk = nnopt_get_named_kernel(queue, "gemma3_gemv_tsk_fused3", &s_tsk);
            cl_mem o2 = (N2 > 0) ? out2 : out1;
            cl_event* evt = KernelProfiler::event_for("op_gemm_proj");
            KernelProfiler::HostTimer _ht("host_gemm_proj");
            if (tk &&
                set_arg_checked(tk, 0, sizeof(cl_mem), &x,    "x")  &&
                set_arg_checked(tk, 1, sizeof(cl_mem), &Wt,   "Wt") &&
                set_arg_checked(tk, 2, sizeof(cl_mem), &out0, "o0") &&
                set_arg_checked(tk, 3, sizeof(cl_mem), &out1, "o1") &&
                set_arg_checked(tk, 4, sizeof(cl_mem), &o2,   "o2") &&
                set_arg_checked(tk, 5, sizeof(int), &N0, "N0") &&
                set_arg_checked(tk, 6, sizeof(int), &N1, "N1") &&
                set_arg_checked(tk, 7, sizeof(int), &N2, "N2") &&
                set_arg_checked(tk, 8, sizeof(int), &K,  "K")  &&
                set_arg_checked(tk, 9, sizeof(int), &tsk_p, "P")) {
                const size_t TN = 64;          // must match TSK_TN
                size_t lws = TN * (size_t)tsk_p;
                size_t nwg = ((size_t)Ntot_all + TN - 1) / TN;
                size_t gws = nwg * lws;
                cl_int derr = clEnqueueNDRangeKernel(queue, tk, 1, nullptr, &gws, &lws, 0, nullptr, evt);
                if (derr == CL_SUCCESS) { NNOPT_DEBUG_SYNC(queue); return true; }
                NNOPT_ERROR_FMT("pytorch_linear_fused3: tsk dispatch %d — falling back", (int)derr);
            }
        }
    }
    if (cat && M == 1 && fused_t && Ntot_all >= fused_t_minn) {
        cl_mem Wt = nnopt_get_transposed_weight(queue, cat, Ntot_all, K);  // [K, Ntot]
        if (Wt) {
            static cl_kernel s_ft = nullptr;
            cl_kernel ftk = nnopt_get_named_kernel(queue, "gemma3_gemv_t_fused3", &s_ft);
            cl_mem o2 = (N2 > 0) ? out2 : out1;
            cl_event* evt = KernelProfiler::event_for("op_gemm_proj");
            KernelProfiler::HostTimer _ht("host_gemm_proj");
            if (ftk &&
                set_arg_checked(ftk, 0, sizeof(cl_mem), &x,    "x")  &&
                set_arg_checked(ftk, 1, sizeof(cl_mem), &Wt,   "Wt") &&
                set_arg_checked(ftk, 2, sizeof(cl_mem), &out0, "o0") &&
                set_arg_checked(ftk, 3, sizeof(cl_mem), &out1, "o1") &&
                set_arg_checked(ftk, 4, sizeof(cl_mem), &o2,   "o2") &&
                set_arg_checked(ftk, 5, sizeof(int), &N0, "N0") &&
                set_arg_checked(ftk, 6, sizeof(int), &N1, "N1") &&
                set_arg_checked(ftk, 7, sizeof(int), &N2, "N2") &&
                set_arg_checked(ftk, 8, sizeof(int), &K,  "K")) {
                size_t lws = (size_t)fused_t_lws;
                size_t gws = ((size_t)Ntot_all + lws - 1) / lws * lws;
                cl_int derr = clEnqueueNDRangeKernel(queue, ftk, 1, nullptr, &gws, &lws, 0, nullptr, evt);
                if (derr == CL_SUCCESS) { NNOPT_DEBUG_SYNC(queue); return true; }
                NNOPT_ERROR_FMT("pytorch_linear_fused3: gemv_t_fused3 dispatch %d — falling back", (int)derr);
            }
        }
    }
    if (cat) {
        static cl_kernel s_f3 = nullptr;
        cl_kernel fk = nnopt_get_named_kernel(queue, "gemma3_gemm_fused3", &s_f3);
        cl_mem o2 = (N2 > 0) ? out2 : out1;   // valid arg even when unused
        static size_t WG = 0;   // reduction width per output for fused proj
        if (WG == 0) { WG = 64; const char* s = std::getenv("NNOPT_FUSED_WG"); if (s) { int v = std::atoi(s); if (v==8||v==16||v==32||v==64||v==128) WG=(size_t)v; } }
        int Ntot = N0 + N1 + N2;
        cl_event* evt = KernelProfiler::event_for("op_gemm_proj");
        KernelProfiler::HostTimer _ht("host_gemm_proj");
        if (fk &&
            set_arg_checked(fk, 0, sizeof(cl_mem), &x,    "x")   &&
            set_arg_checked(fk, 1, sizeof(cl_mem), &cat,  "W")   &&
            set_arg_checked(fk, 2, sizeof(cl_mem), &out0, "o0")  &&
            set_arg_checked(fk, 3, sizeof(cl_mem), &out1, "o1")  &&
            set_arg_checked(fk, 4, sizeof(cl_mem), &o2,   "o2")  &&
            set_arg_checked(fk, 5, sizeof(int), &M,  "M")  &&
            set_arg_checked(fk, 6, sizeof(int), &N0, "N0") &&
            set_arg_checked(fk, 7, sizeof(int), &N1, "N1") &&
            set_arg_checked(fk, 8, sizeof(int), &N2, "N2") &&
            set_arg_checked(fk, 9, sizeof(int), &K,  "K")) {
            size_t gws[2] = { (size_t)Ntot * WG, (size_t)M };
            size_t lws[2] = { WG, 1 };
            cl_int derr = clEnqueueNDRangeKernel(queue, fk, 2, nullptr, gws, lws, 0, nullptr, evt);
            if (derr == CL_SUCCESS) { NNOPT_DEBUG_SYNC(queue); return true; }
            NNOPT_ERROR_FMT("pytorch_linear_fused3: dispatch %d — falling back", (int)derr);
        }
    }
#endif
    // Fallback (fp32, or any fp16 failure): separate calls.
    if (!pytorch_linear(queue, M, N0, K, x, W0, out0)) return false;
    if (!pytorch_linear(queue, M, N1, K, x, W1, out1)) return false;
    if (N2 > 0 && !pytorch_linear(queue, M, N2, K, x, W2, out2)) return false;
    return true;
}

bool pytorch_linear_fused2(cl_command_queue queue, int M,
                           int N0, int N1, int K,
                           cl_mem x, cl_mem W0, cl_mem W1,
                           cl_mem out0, cl_mem out1) {
    return pytorch_linear_fused3(queue, M, N0, N1, 0, K, x, W0, W1, W1, out0, out1, out1);
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
