// Shared primitives used by Kokoro ops: embedding gather (NLC) + LayerNorm.
// Leading-underscore filename signals "support, not a graph node".
//
// Exports (extern "C"):
//   prim_embedding_gather_nlc(...)
//   prim_layernorm(...)
//
// Both kernels are lazy-built on first call (Rule PROG-01) and cached.

#include "opencl_context.h"
#include "debug_utils.h"
#include "profiler.h"

#include <CL/cl.h>
#include <cstdint>
#include <string>

static cl_program g_prim_prog = nullptr;
static cl_kernel  g_k_emb_nlc = nullptr;
static cl_kernel  g_k_layernorm = nullptr;

static const char* k_prim_src = R"CLC(
#ifdef NNOPT_USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)    vload_half((i), (__global const half*)(p))
  #define STORE(p,i,v) vstore_half((float)(v), (i), (__global half*)(p))
#else
  typedef float storage_t;
  #define LOAD(p,i)    ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
#endif

// out[t, c] = emb_table[ids[t], c]
__kernel void embedding_gather_nlc(__global const int* ids,
                                   __global const storage_t* emb,
                                   __global storage_t* out,
                                   int T, int C, int V) {
    int t = get_global_id(0);
    int c = get_global_id(1);
    if (t >= T || c >= C) return;
    int idx = ids[t];
    if (idx < 0 || idx >= V) {
        STORE(out, t * C + c, 0.0f);
        return;
    }
    float v = (float)LOAD(emb, idx * C + c);
    STORE(out, t * C + c, v);
}

// LayerNorm over last dim: y = (x - mean) / sqrt(var + eps) * gamma + beta
// One workgroup per row (M total rows of length N). Local size = N (or a divisor — for now require N small enough).
// For simplicity use a single-thread-per-row implementation; N is small (128/512/768).
__kernel void layernorm_rowwise(__global const storage_t* x,
                                __global const storage_t* gamma,
                                __global const storage_t* beta,
                                __global storage_t* y,
                                int M, int N, float eps) {
    int m = get_global_id(0);
    if (m >= M) return;
    int base = m * N;
    float mean = 0.0f;
    for (int n = 0; n < N; ++n) mean += (float)LOAD(x, base + n);
    mean /= (float)N;
    float var = 0.0f;
    for (int n = 0; n < N; ++n) {
        float d = (float)LOAD(x, base + n) - mean;
        var += d * d;
    }
    var /= (float)N;
    float inv_std = rsqrt(var + eps);
    for (int n = 0; n < N; ++n) {
        float v = ((float)LOAD(x, base + n) - mean) * inv_std;
        v = v * (float)LOAD(gamma, n) + (float)LOAD(beta, n);
        STORE(y, base + n, v);
    }
}
)CLC";

static bool ensure_built(OpenCLContext& cl_ctx) {
    if (g_k_emb_nlc && g_k_layernorm) return true;
    cl_context ctx = cl_ctx.context();
    cl_device_id dev = cl_ctx.device();
    cl_int err = CL_SUCCESS;
    const char* opts =
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1";
#else
        "";
#endif
    g_prim_prog = nnopt_build_program_cached(ctx, dev, k_prim_src, opts, "primitives", &err);
    if (!g_prim_prog) return false;
    g_k_emb_nlc = clCreateKernel(g_prim_prog, "embedding_gather_nlc", &err);
    if (err != CL_SUCCESS || !g_k_emb_nlc) {
        NNOPT_ERROR_FMT("_primitives: clCreateKernel(embedding_gather_nlc) failed (%d)", (int)err);
        return false;
    }
    g_k_layernorm = clCreateKernel(g_prim_prog, "layernorm_rowwise", &err);
    if (err != CL_SUCCESS || !g_k_layernorm) {
        NNOPT_ERROR_FMT("_primitives: clCreateKernel(layernorm_rowwise) failed (%d)", (int)err);
        return false;
    }
    return true;
}

extern "C" int prim_embedding_gather_nlc(OpenCLContext& cl_ctx,
                                         cl_command_queue queue,
                                         cl_mem ids_i32,    // [T] int32
                                         cl_mem emb,        // [V, C] storage_t
                                         cl_mem out,        // [T, C] storage_t
                                         int T, int C, int V) {
    if (!ensure_built(cl_ctx)) return -1;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(g_k_emb_nlc, 0, sizeof(cl_mem), &ids_i32);
    err |= clSetKernelArg(g_k_emb_nlc, 1, sizeof(cl_mem), &emb);
    err |= clSetKernelArg(g_k_emb_nlc, 2, sizeof(cl_mem), &out);
    err |= clSetKernelArg(g_k_emb_nlc, 3, sizeof(int), &T);
    err |= clSetKernelArg(g_k_emb_nlc, 4, sizeof(int), &C);
    err |= clSetKernelArg(g_k_emb_nlc, 5, sizeof(int), &V);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("prim_embedding_gather_nlc: clSetKernelArg failed (%d)", (int)err);
        return -1;
    }
    size_t gws[2] = {(size_t)T, (size_t)C};
    err = nnopt_enqueue_profiled(queue, g_k_emb_nlc, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("prim_embedding_gather_nlc: clEnqueueNDRangeKernel failed (%d)", (int)err);
        return -1;
    }
    return 0;
}

extern "C" int prim_layernorm(OpenCLContext& cl_ctx,
                              cl_command_queue queue,
                              cl_mem x,
                              cl_mem gamma,
                              cl_mem beta,
                              cl_mem y,
                              int M, int N, float eps) {
    if (!ensure_built(cl_ctx)) return -1;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(g_k_layernorm, 0, sizeof(cl_mem), &x);
    err |= clSetKernelArg(g_k_layernorm, 1, sizeof(cl_mem), &gamma);
    err |= clSetKernelArg(g_k_layernorm, 2, sizeof(cl_mem), &beta);
    err |= clSetKernelArg(g_k_layernorm, 3, sizeof(cl_mem), &y);
    err |= clSetKernelArg(g_k_layernorm, 4, sizeof(int), &M);
    err |= clSetKernelArg(g_k_layernorm, 5, sizeof(int), &N);
    err |= clSetKernelArg(g_k_layernorm, 6, sizeof(float), &eps);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("prim_layernorm: clSetKernelArg failed (%d)", (int)err);
        return -1;
    }
    size_t gws = (size_t)M;
    err = nnopt_enqueue_profiled(queue, g_k_layernorm, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("prim_layernorm: clEnqueueNDRangeKernel failed (%d)", (int)err);
        return -1;
    }
    return 0;
}
