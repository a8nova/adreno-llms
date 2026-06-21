// Reference: kokoro/model.py:KModel.__init__ — self.bert_encoder = nn.Linear(bert.config.hidden_size=768, hidden_dim=512)
// Reference: kokoro/model.py:forward_with_tokens line ~104 — d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
//
// Computes: out[T, 512] = bert_dur[T, 768] @ W[512, 768]^T + b[512]
// The .transpose(-1,-2) in the reference is a downstream layout change; this op
// outputs row-major [T, 512] (NLC). Caller transposes if it needs NCL.
//
// Weights:
//   bert_encoder.module.weight  fp16 [512, 768]   (PyTorch nn.Linear layout)
//   bert_encoder.module.bias    fp16 [512]
//
// Validate against reference dump:
//   reference/layers/bert_encoder_output.bin   (dtype fp32, shape [1, T, 512])

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "profiler.h"
#include "model_config.h"
#include "utils.h"

#include <CL/cl.h>
#include <cstdint>
#include <string>

// Persistent program/kernel (lazy-built on first call — Rule PROG-01).
static cl_program g_bert_encoder_prog = nullptr;
static cl_kernel  g_bias_add_kernel   = nullptr;

// Simple broadcast bias-add: y[m, n] += b[n]
static const char* k_bias_add_src = R"CLC(
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

__kernel void bias_add_2d(__global storage_t* y,
                          __global const storage_t* b,
                          int M, int N) {
    int gid = get_global_id(0);
    int total = M * N;
    if (gid >= total) return;
    int n = gid % N;
    float v = (float)LOAD(y, gid) + (float)LOAD(b, n);
    STORE(y, gid, v);
}
)CLC";

static bool ensure_kernel(OpenCLContext& cl_ctx) {
    if (g_bias_add_kernel) return true;
    cl_context ctx = cl_ctx.context();
    cl_device_id dev = cl_ctx.device();
    cl_int err = CL_SUCCESS;
    const char* build_opts =
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1";
#else
        "";
#endif
    g_bert_encoder_prog = nnopt_build_program_cached(ctx, dev, k_bias_add_src, build_opts,
                                                      "bert_encoder", &err);
    if (!g_bert_encoder_prog) return false;
    g_bias_add_kernel = clCreateKernel(g_bert_encoder_prog, "bias_add_2d", &err);
    if (err != CL_SUCCESS || !g_bias_add_kernel) {
        NNOPT_ERROR_FMT("op_bert_encoder: clCreateKernel(bias_add_2d) failed (%d)", (int)err);
        return false;
    }
    return true;
}

// out:    caller-allocated, sizeof(nnopt_storage_t) * T * 512
// bert_h: input [T, 768] in row-major (NLC), nnopt_storage_t
// Returns 0 on success, non-zero on failure.
extern "C" int op_bert_encoder(OpenCLContext& cl_ctx,
                               Weights& weights,
                               cl_command_queue queue,
                               cl_mem bert_h,
                               cl_mem out,
                               int T) {
    if (!bert_h || !out) {
        NNOPT_ERROR("op_bert_encoder: null input/output buffer");
        return -1;
    }
    cl_mem W = weights.get_buffer("bert_encoder.module.weight", /*optional=*/false);
    cl_mem b = weights.get_buffer("bert_encoder.module.bias",   /*optional=*/false);
    if (!W || !b) {
        NNOPT_ERROR("op_bert_encoder: missing bert_encoder weights");
        return -1;
    }
    // Linear: out[T, 512] = bert_h[T, 768] @ W[512, 768]^T
    if (!pytorch_linear(queue, /*M=*/T, /*N=*/512, /*K=*/768, bert_h, W, out)) {
        NNOPT_ERROR("op_bert_encoder: pytorch_linear failed");
        return -1;
    }
    // bias add
    if (!ensure_kernel(cl_ctx)) return -1;
    int M = T;
    int N = 512;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(g_bias_add_kernel, 0, sizeof(cl_mem), &out);
    err |= clSetKernelArg(g_bias_add_kernel, 1, sizeof(cl_mem), &b);
    err |= clSetKernelArg(g_bias_add_kernel, 2, sizeof(int), &M);
    err |= clSetKernelArg(g_bias_add_kernel, 3, sizeof(int), &N);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("op_bert_encoder: clSetKernelArg failed (%d)", (int)err);
        return -1;
    }
    size_t gws = (size_t)M * (size_t)N;
    err = nnopt_enqueue_profiled(queue, g_bias_add_kernel, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("op_bert_encoder: clEnqueueNDRangeKernel failed (%d)", (int)err);
        return -1;
    }
    NNOPT_LAYER_CHECK("bert_encoder", queue, out, (size_t)T * 512);
    return 0;
}
