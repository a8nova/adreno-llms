// Reference: transformers/models/lfm2/modeling_lfm2.py Lfm2RMSNorm.forward and Lfm2MLP.forward
// Shared LFM2 OpenCL helpers: RMSNorm and SwiGLU feed-forward.

#include "ops/lfm2_common.h"
#include "debug_utils.h"
#include "utils.h"
#include "profiler.h"
#include "model_config.h"

#include <CL/cl.h>
#include <string>

namespace {

static bool set_arg_local(cl_kernel k, cl_uint idx, size_t sz, const void* v, const char* name) {
    cl_int err = clSetKernelArg(k, idx, sz, v);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("lfm2_common: clSetKernelArg(%u,%s) failed (%d)", (unsigned)idx, name, (int)err);
        return false;
    }
    return true;
}

static cl_kernel make_kernel(OpenCLContext& cl_ctx, const char* name) {
    cl_program p = lfm2_program(cl_ctx);
    if (!p) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(p, name, &err);
    if (err != CL_SUCCESS || !k) {
        NNOPT_ERROR_FMT("lfm2_common: clCreateKernel(%s) failed (%d)", name, (int)err);
        return nullptr;
    }
    return k;
}

}  // namespace

cl_program lfm2_program(OpenCLContext& cl_ctx) {
    static cl_program program = nullptr;
    if (!program) {
        program = cl_ctx.build_program_from_file("kernels/lfm2_ops.cl");
        if (!program) NNOPT_ERROR("lfm2_program: failed to build kernels/lfm2_ops.cl");
    }
    return program;
}

// Singleton GEMV kernel handle for the pytorch_linear M==1 fast path. Called
// from src/utils.cpp::pytorch_linear with C-linkage; needs queue→context only
// to bootstrap the program build. The actual cl_ctx singleton lives in
// opencl_context.cpp; we stash it here on first invocation.
static OpenCLContext* g_cl_ctx_for_gemv = nullptr;
void nnopt_register_cl_ctx_for_gemv(OpenCLContext* ctx) { g_cl_ctx_for_gemv = ctx; }

#if 0  // REVERTED: fused gemv helpers (caused sub-buffer + memory issues)
// Cached fused GEMV kernels (lazily built on first use; lifetime = process).
static cl_kernel g_gemv_fused3_kernel = nullptr;
static cl_kernel g_gemv_fused2_kernel = nullptr;

bool gemv_fused3(cl_command_queue queue,
                 cl_mem x, cl_mem wa, cl_mem wb, cl_mem wc,
                 cl_mem ya, cl_mem yb, cl_mem yc,
                 int N_a, int N_b, int N_c, int K,
                 const char* label) {
    if (!g_gemv_fused3_kernel) {
        if (!g_cl_ctx_for_gemv) {
            NNOPT_ERROR("gemv_fused3: g_cl_ctx_for_gemv not registered");
            return false;
        }
        cl_program p = lfm2_program(*g_cl_ctx_for_gemv);
        if (!p) return false;
        cl_int err = CL_SUCCESS;
        g_gemv_fused3_kernel = clCreateKernel(p, "gemv_fused3", &err);
        if (err != CL_SUCCESS || !g_gemv_fused3_kernel) {
            NNOPT_ERROR_FMT("gemv_fused3: clCreateKernel failed (%d)", (int)err);
            g_gemv_fused3_kernel = nullptr;
            return false;
        }
    }
    cl_kernel k = g_gemv_fused3_kernel;
    cl_int e = clSetKernelArg(k, 0, sizeof(cl_mem), &x);
    e |= clSetKernelArg(k, 1, sizeof(cl_mem), &wa);
    e |= clSetKernelArg(k, 2, sizeof(cl_mem), &wb);
    e |= clSetKernelArg(k, 3, sizeof(cl_mem), &wc);
    e |= clSetKernelArg(k, 4, sizeof(cl_mem), &ya);
    e |= clSetKernelArg(k, 5, sizeof(cl_mem), &yb);
    e |= clSetKernelArg(k, 6, sizeof(cl_mem), &yc);
    e |= clSetKernelArg(k, 7, sizeof(int), &N_a);
    e |= clSetKernelArg(k, 8, sizeof(int), &N_b);
    e |= clSetKernelArg(k, 9, sizeof(int), &N_c);
    e |= clSetKernelArg(k, 10, sizeof(int), &K);
    if (e != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_fused3 setArg failed (%d)", (int)e);
        return false;
    }
    size_t global = (size_t)(N_a + N_b + N_c) * 64u;
    size_t local = 64u;
    cl_int re = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global, &local,
                                       0, nullptr, KernelProfiler::event_for(label));
    if (re != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_fused3 dispatch failed (%d)", (int)re);
        return false;
    }
    return true;
}

bool gemv_fused2(cl_command_queue queue,
                 cl_mem x, cl_mem wa, cl_mem wb,
                 cl_mem ya, cl_mem yb,
                 int N_a, int N_b, int K,
                 const char* label) {
    if (!g_gemv_fused2_kernel) {
        if (!g_cl_ctx_for_gemv) {
            NNOPT_ERROR("gemv_fused2: g_cl_ctx_for_gemv not registered");
            return false;
        }
        cl_program p = lfm2_program(*g_cl_ctx_for_gemv);
        if (!p) return false;
        cl_int err = CL_SUCCESS;
        g_gemv_fused2_kernel = clCreateKernel(p, "gemv_fused2", &err);
        if (err != CL_SUCCESS || !g_gemv_fused2_kernel) {
            NNOPT_ERROR_FMT("gemv_fused2: clCreateKernel failed (%d)", (int)err);
            g_gemv_fused2_kernel = nullptr;
            return false;
        }
    }
    cl_kernel k = g_gemv_fused2_kernel;
    cl_int e = clSetKernelArg(k, 0, sizeof(cl_mem), &x);
    e |= clSetKernelArg(k, 1, sizeof(cl_mem), &wa);
    e |= clSetKernelArg(k, 2, sizeof(cl_mem), &wb);
    e |= clSetKernelArg(k, 3, sizeof(cl_mem), &ya);
    e |= clSetKernelArg(k, 4, sizeof(cl_mem), &yb);
    e |= clSetKernelArg(k, 5, sizeof(int), &N_a);
    e |= clSetKernelArg(k, 6, sizeof(int), &N_b);
    e |= clSetKernelArg(k, 7, sizeof(int), &K);
    if (e != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_fused2 setArg failed (%d)", (int)e);
        return false;
    }
    size_t global = (size_t)(N_a + N_b) * 64u;
    size_t local = 64u;
    cl_int re = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global, &local,
                                       0, nullptr, KernelProfiler::event_for(label));
    if (re != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_fused2 dispatch failed (%d)", (int)re);
        return false;
    }
    return true;
}

#endif  // REVERTED block

// ── int8 ── (active) — kernel handles to lfm2_ops.cl::gemv_int8_pytorch_linear
// and dequant_int8_to_fp16. The image-path GEMV in utils.cpp covers the M=1
// hot path (K=1024/4608, image2d-backed); these handles serve fallback / M>1
// dequant. Lazily created on first use, process-lifetime.
extern "C" cl_kernel nnopt_get_gemv_int8_kernel(cl_command_queue queue) {
    (void)queue;
    static cl_kernel kernel = nullptr;
    if (kernel) return kernel;
    if (!g_cl_ctx_for_gemv) return nullptr;
    cl_program p = lfm2_program(*g_cl_ctx_for_gemv);
    if (!p) return nullptr;
    cl_int err = CL_SUCCESS;
    kernel = clCreateKernel(p, "gemv_int8_pytorch_linear", &err);
    if (err != CL_SUCCESS || !kernel) {
        NNOPT_ERROR_FMT("nnopt_get_gemv_int8_kernel: clCreateKernel failed (%d)", (int)err);
        kernel = nullptr;
        return nullptr;
    }
    return kernel;
}

extern "C" cl_kernel nnopt_get_dequant_int8_kernel(cl_command_queue queue) {
    (void)queue;
    static cl_kernel kernel = nullptr;
    if (kernel) return kernel;
    if (!g_cl_ctx_for_gemv) return nullptr;
    cl_program p = lfm2_program(*g_cl_ctx_for_gemv);
    if (!p) return nullptr;
    cl_int err = CL_SUCCESS;
    kernel = clCreateKernel(p, "dequant_int8_to_fp16", &err);
    if (err != CL_SUCCESS || !kernel) {
        NNOPT_ERROR_FMT("nnopt_get_dequant_int8_kernel: clCreateKernel failed (%d)", (int)err);
        kernel = nullptr;
        return nullptr;
    }
    return kernel;
}

extern "C" cl_kernel nnopt_get_gemv_kernel(cl_command_queue queue) {
    (void)queue;  // program/kernel are context-scoped; queue is per-dispatch.
    static cl_kernel kernel = nullptr;
    if (kernel) return kernel;
    if (!g_cl_ctx_for_gemv) return nullptr;
    cl_program p = lfm2_program(*g_cl_ctx_for_gemv);
    if (!p) return nullptr;
    cl_int err = CL_SUCCESS;
    kernel = clCreateKernel(p, "gemv_pytorch_linear", &err);
    if (err != CL_SUCCESS || !kernel) {
        NNOPT_ERROR_FMT("nnopt_get_gemv_kernel: clCreateKernel failed (%d)", (int)err);
        kernel = nullptr;
        return nullptr;
    }
    return kernel;
}

// Buffer pool: eliminates ~120 clCreateBuffer/clReleaseMemObject per decode token.
// Free-list keyed by byte-size. lfm2_alloc pops from pool or creates new;
// clRetainMemObject keeps the buffer alive after the caller's clReleaseMemObject.
// lfm2_pool_reclaim() (called at top of each forward) sweeps outstanding buffers
// back to the free-list. Buffers from a prior seq_len (prefill) that are too
// large for the current seq_len (decode) are released to avoid OOM.
#include <unordered_map>
#include <vector>

struct PoolBuf { size_t bytes; cl_mem mem; };
static std::unordered_map<size_t, std::vector<cl_mem>> s_pool_free;
static std::vector<PoolBuf>                            s_pool_outstanding;
static bool                                            s_pool_active = false;

void lfm2_pool_set_active(bool active) { s_pool_active = active; }

cl_mem lfm2_alloc(OpenCLContext& cl_ctx, size_t elems, const char* label) {
    const size_t bytes = elems * sizeof(nnopt_storage_t);
    if (s_pool_active) {
        auto it = s_pool_free.find(bytes);
        if (it != s_pool_free.end() && !it->second.empty()) {
            cl_mem m = it->second.back();
            it->second.pop_back();
            clRetainMemObject(m);
            s_pool_outstanding.push_back({bytes, m});
            return m;
        }
    }
    cl_int err = CL_SUCCESS;
    cl_mem m = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS || !m) {
        NNOPT_ERROR_FMT("lfm2_alloc(%s): %zu elems failed (%d)", label, elems, (int)err);
        return nullptr;
    }
    if (s_pool_active) {
        clRetainMemObject(m);
        s_pool_outstanding.push_back({bytes, m});
    }
    return m;
}

void lfm2_pool_reclaim() {
    for (auto& p : s_pool_outstanding)
        s_pool_free[p.bytes].push_back(p.mem);
    s_pool_outstanding.clear();
}

// Fully release all pool memory (outstanding + free-list). Use between phases
// when the next phase uses radically different buffer sizes (e.g. vision tower
// finished, now LM prefill starts — vision-sized buffers would just OOM).
void lfm2_pool_clear() {
    for (auto& p : s_pool_outstanding) clReleaseMemObject(p.mem);
    s_pool_outstanding.clear();
    for (auto& kv : s_pool_free)
        for (cl_mem m : kv.second) clReleaseMemObject(m);
    s_pool_free.clear();
}

bool lfm2_kernel1(cl_command_queue queue, cl_kernel kernel, size_t global, const char* label) {
    cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr,
                                        0, nullptr, KernelProfiler::event_for(label));
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("%s: clEnqueueNDRangeKernel failed (%d)", label, (int)err);
        return false;
    }
    NNOPT_DEBUG_SYNC(queue);
    return true;
}

// Variant with explicit local work-group size. Required for kernels with
// `reqd_work_group_size` attribute (rms_norm_rows, rms_norm_heads, gemv) —
// without it, the runtime would pick its own LWS and the dispatch would fail.
bool lfm2_kernel1_lws(cl_command_queue queue, cl_kernel kernel, size_t global, size_t local, const char* label) {
    cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local,
                                        0, nullptr, KernelProfiler::event_for(label));
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("%s: clEnqueueNDRangeKernel(lws=%zu) failed (%d)", label, local, (int)err);
        return false;
    }
    NNOPT_DEBUG_SYNC(queue);
    return true;
}

// Fused: out_sum = a + b ; out_norm = rms_norm(out_sum) * weight.
// Replaces (element_add → lfm2_rms_norm) two-dispatch sequence with one.
// Both outputs are owned by the caller (release both).
bool lfm2_rms_norm_add(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                      cl_mem a, cl_mem b, int rows, int cols,
                      const std::string& weight_key,
                      cl_mem* out_sum_p, cl_mem* out_norm_p) {
    cl_mem w = weights.get_buffer(weight_key);
    if (!w) {
        NNOPT_ERROR_FMT("lfm2_rms_norm_add: missing weight %s", weight_key.c_str());
        return false;
    }
    cl_mem out_sum  = lfm2_alloc(cl_ctx, (size_t)rows * (size_t)cols, "rms_norm_add_sum");
    cl_mem out_norm = lfm2_alloc(cl_ctx, (size_t)rows * (size_t)cols, "rms_norm_add_norm");
    if (!out_sum || !out_norm) {
        if (out_sum) clReleaseMemObject(out_sum);
        if (out_norm) clReleaseMemObject(out_norm);
        return false;
    }
    static cl_kernel kernel = nullptr;
    if (!kernel) kernel = make_kernel(cl_ctx, "rms_norm_rows_add");
    if (!kernel) {
        clReleaseMemObject(out_sum); clReleaseMemObject(out_norm);
        return false;
    }
    const float eps = MODEL_CONFIG::NORM_EPS;
    if (!set_arg_local(kernel, 0, sizeof(cl_mem), &a, "a") ||
        !set_arg_local(kernel, 1, sizeof(cl_mem), &b, "b") ||
        !set_arg_local(kernel, 2, sizeof(cl_mem), &w, "weight") ||
        !set_arg_local(kernel, 3, sizeof(cl_mem), &out_sum, "out_sum") ||
        !set_arg_local(kernel, 4, sizeof(cl_mem), &out_norm, "out_norm") ||
        !set_arg_local(kernel, 5, sizeof(int), &rows, "rows") ||
        !set_arg_local(kernel, 6, sizeof(int), &cols, "cols") ||
        !set_arg_local(kernel, 7, sizeof(float), &eps, "eps")) {
        clReleaseMemObject(out_sum); clReleaseMemObject(out_norm);
        return false;
    }
    if (!lfm2_kernel1_lws(queue, kernel, (size_t)rows * 32u, 32u, "lfm2_rms_norm_add")) {
        clReleaseMemObject(out_sum); clReleaseMemObject(out_norm);
        return false;
    }
    *out_sum_p = out_sum;
    *out_norm_p = out_norm;
    return true;
}

cl_mem lfm2_rms_norm(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                     cl_mem input, int rows, int cols, const std::string& weight_key) {
    cl_mem w = weights.get_buffer(weight_key);
    if (!w) {
        NNOPT_ERROR_FMT("lfm2_rms_norm: missing weight %s", weight_key.c_str());
        return nullptr;
    }
    cl_mem out = lfm2_alloc(cl_ctx, (size_t)rows * (size_t)cols, "rms_norm");
    if (!out) return nullptr;
    static cl_kernel kernel = nullptr;
    if (!kernel) kernel = make_kernel(cl_ctx, "rms_norm_rows");
    if (!kernel) { clReleaseMemObject(out); return nullptr; }
    const float eps = MODEL_CONFIG::NORM_EPS;
    if (!set_arg_local(kernel, 0, sizeof(cl_mem), &input, "input") ||
        !set_arg_local(kernel, 1, sizeof(cl_mem), &w, "weight") ||
        !set_arg_local(kernel, 2, sizeof(cl_mem), &out, "out") ||
        !set_arg_local(kernel, 3, sizeof(int), &rows, "rows") ||
        !set_arg_local(kernel, 4, sizeof(int), &cols, "cols") ||
        !set_arg_local(kernel, 5, sizeof(float), &eps, "eps")) {
        clReleaseMemObject(out); return nullptr;
    }
    if (!lfm2_kernel1_lws(queue, kernel, (size_t)rows * 32u, 32u, "lfm2_rms_norm")) {
        clReleaseMemObject(out); return nullptr;
    }
    return out;
}

cl_mem lfm2_mlp(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                cl_mem input, int rows, int hidden_size, int intermediate_size,
                const std::string& prefix, int layer_idx) {
    // PyTorch: Lfm2MLP.forward(x) = w2(silu(w1(x)) * w3(x)).
    cl_mem w1 = weights.get_buffer(prefix + ".feed_forward.w1.weight");
    cl_mem w2 = weights.get_buffer(prefix + ".feed_forward.w2.weight");
    cl_mem w3 = weights.get_buffer(prefix + ".feed_forward.w3.weight");
    if (!w1 || !w2 || !w3) {
        NNOPT_ERROR_FMT("lfm2_mlp: missing feed_forward weights for %s", prefix.c_str());
        return nullptr;
    }
    cl_mem gate = lfm2_alloc(cl_ctx, (size_t)rows * (size_t)intermediate_size, "mlp_gate");
    cl_mem up   = lfm2_alloc(cl_ctx, (size_t)rows * (size_t)intermediate_size, "mlp_up");
    cl_mem act  = lfm2_alloc(cl_ctx, (size_t)rows * (size_t)intermediate_size, "mlp_act");
    cl_mem out  = lfm2_alloc(cl_ctx, (size_t)rows * (size_t)hidden_size, "mlp_out");
    if (!gate || !up || !act || !out) {
        if (gate) clReleaseMemObject(gate); if (up) clReleaseMemObject(up);
        if (act) clReleaseMemObject(act); if (out) clReleaseMemObject(out);
        return nullptr;
    }
    if (!pytorch_linear(queue, rows, intermediate_size, hidden_size, input, w1, gate) ||
        !pytorch_linear(queue, rows, intermediate_size, hidden_size, input, w3, up)) {
        clReleaseMemObject(gate); clReleaseMemObject(up); clReleaseMemObject(act); clReleaseMemObject(out);
        return nullptr;
    }
    if (layer_idx == 0) {
        NNOPT_LAYER_CHECK("block0_sub_feed_forward_w1_out", queue, gate, (size_t)rows * (size_t)intermediate_size);
        NNOPT_LAYER_CHECK("block0_sub_feed_forward_w3_out", queue, up, (size_t)rows * (size_t)intermediate_size);
    }
    static cl_kernel swiglu_kernel = nullptr;
    if (!swiglu_kernel) swiglu_kernel = make_kernel(cl_ctx, "swiglu");
    if (!swiglu_kernel) {
        clReleaseMemObject(gate); clReleaseMemObject(up); clReleaseMemObject(act); clReleaseMemObject(out);
        return nullptr;
    }
    int n = rows * intermediate_size;
    if (!set_arg_local(swiglu_kernel, 0, sizeof(cl_mem), &gate, "gate") ||
        !set_arg_local(swiglu_kernel, 1, sizeof(cl_mem), &up, "up") ||
        !set_arg_local(swiglu_kernel, 2, sizeof(cl_mem), &act, "act") ||
        !set_arg_local(swiglu_kernel, 3, sizeof(int), &n, "n") ||
        !lfm2_kernel1(queue, swiglu_kernel, (size_t)n, "lfm2_swiglu")) {
        clReleaseMemObject(gate); clReleaseMemObject(up); clReleaseMemObject(act); clReleaseMemObject(out);
        return nullptr;
    }
    if (layer_idx == 0) NNOPT_LAYER_CHECK("block0_sub_feed_forward_w2_in", queue, act, (size_t)n);
    if (!pytorch_linear(queue, rows, hidden_size, intermediate_size, act, w2, out)) {
        clReleaseMemObject(gate); clReleaseMemObject(up); clReleaseMemObject(act); clReleaseMemObject(out);
        return nullptr;
    }
    if (layer_idx == 0) NNOPT_LAYER_CHECK("block0_sub_feed_forward_w2_out", queue, out, (size_t)rows * (size_t)hidden_size);
    clReleaseMemObject(gate);
    clReleaseMemObject(up);
    clReleaseMemObject(act);
    return out;
}
