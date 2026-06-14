// Reference: kokoro/modules.py:TextEncoder.forward
//   embedding[178, 512] → [T, 512] (NLC)
//   transpose → [512, T] (NCL)
//   3x { Conv1d weight_norm 512→512 k=5 pad=2 → LayerNorm → LeakyReLU(0.2) }
//   transpose → [T, 512] (NLC)
//   biLSTM(512, 256, bidir) → [T, 512]
//   transpose → [512, T] (NCL)
//
// Caller wants [T, 512] (NLC) for the alignment gather, so final transpose back to NLC.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "profiler.h"
#include "model_config.h"
#include "utils.h"

#include <CL/cl.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

extern "C" int prim_embedding_gather_nlc(OpenCLContext&, cl_command_queue,
                                         cl_mem, cl_mem, cl_mem, int, int, int);
extern "C" int prim_bilstm(OpenCLContext&, Weights&, cl_command_queue,
                            const std::string&, cl_mem, cl_mem, int, int, int);
extern "C" int dec_conv1d_wn(OpenCLContext&, Weights&, cl_command_queue,
                              const std::string&, cl_mem, cl_mem,
                              int, int, int, int, int, int*, int*);

static cl_program g_te_prog = nullptr;
static cl_kernel  g_k_transpose_NLC_to_NCL = nullptr;
static cl_kernel  g_k_transpose_NCL_to_NLC = nullptr;
static cl_kernel  g_k_layernorm_NCL = nullptr;
static cl_kernel  g_k_leaky = nullptr;

static const char* k_te_src = R"CLC(
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

__kernel void te_transpose_NLC_to_NCL(__global const storage_t* in,
                                      __global storage_t* out, int C, int T) {
    int t = get_global_id(0); int c = get_global_id(1);
    if (t >= T || c >= C) return;
    STORE(out, c*T + t, (float)LOAD(in, t*C + c));
}
__kernel void te_transpose_NCL_to_NLC(__global const storage_t* in,
                                      __global storage_t* out, int C, int T) {
    int c = get_global_id(0); int t = get_global_id(1);
    if (c >= C || t >= T) return;
    STORE(out, t*C + c, (float)LOAD(in, c*T + t));
}
// LayerNorm acting on NCL: normalize over C dim per time step, with affine (gamma, beta).
// Reference module's LayerNorm does transpose to NLC, F.layer_norm over channels, transpose back.
// Equivalent: for each t, normalize across C channels using gamma[c], beta[c].
__kernel void te_layernorm_NCL(__global const storage_t* x,
                               __global const storage_t* gamma,
                               __global const storage_t* beta,
                               __global storage_t* y, int C, int T, float eps) {
    int t = get_global_id(0);
    if (t >= T) return;
    float mean = 0.0f;
    for (int c = 0; c < C; ++c) mean += (float)LOAD(x, c*T + t);
    mean /= (float)C;
    float var = 0.0f;
    for (int c = 0; c < C; ++c) { float d = (float)LOAD(x, c*T + t) - mean; var += d*d; }
    var /= (float)C;
    float inv = rsqrt(var + eps);
    for (int c = 0; c < C; ++c) {
        float v = ((float)LOAD(x, c*T + t) - mean) * inv;
        v = v * (float)LOAD(gamma, c) + (float)LOAD(beta, c);
        STORE(y, c*T + t, v);
    }
}
__kernel void te_leaky(__global storage_t* y, int N, float slope) {
    int i = get_global_id(0); if (i >= N) return;
    float v = (float)LOAD(y, i);
    if (v < 0.0f) v *= slope;
    STORE(y, i, v);
}
)CLC";

static bool ensure_built(OpenCLContext& cl_ctx) {
    if (g_k_transpose_NLC_to_NCL) return true;
    cl_int err = CL_SUCCESS;
    const char* opts =
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1";
#else
        "";
#endif
    cl_device_id dev = cl_ctx.device();
    g_te_prog = nnopt_build_program_cached(cl_ctx.context(), dev, k_te_src, opts, "text_encoder", &err);
    if (!g_te_prog) return false;
    g_k_transpose_NLC_to_NCL = clCreateKernel(g_te_prog, "te_transpose_NLC_to_NCL", &err);
    g_k_transpose_NCL_to_NLC = clCreateKernel(g_te_prog, "te_transpose_NCL_to_NLC", &err);
    g_k_layernorm_NCL = clCreateKernel(g_te_prog, "te_layernorm_NCL", &err);
    g_k_leaky = clCreateKernel(g_te_prog, "te_leaky", &err);
    return g_k_transpose_NLC_to_NCL && g_k_transpose_NCL_to_NLC && g_k_layernorm_NCL && g_k_leaky;
}

static cl_mem alloc(OpenCLContext& cl_ctx, size_t bytes) {
    cl_int e = CL_SUCCESS;
    return clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &e);
}

extern "C" int op_text_encoder_modules(OpenCLContext& cl_ctx,
                                       Weights& weights,
                                       cl_command_queue queue,
                                       cl_mem input_ids_i32,
                                       cl_mem out,            // [T, 512] NLC
                                       int T) {
    if (!ensure_built(cl_ctx)) return -1;
    const int C = 512;

    cl_mem emb_w = weights.get_buffer("text_encoder.module.embedding.weight");
    if (!emb_w) { NNOPT_ERROR("text_encoder: missing embedding.weight"); return -1; }

    // (1) embedding gather → [T, 512] NLC
    cl_mem emb_out = alloc(cl_ctx, sizeof(nnopt_storage_t) * T * C);
    if (prim_embedding_gather_nlc(cl_ctx, queue, input_ids_i32, emb_w, emb_out, T, C, 178) != 0) {
        NNOPT_ERROR("text_encoder: embedding gather failed"); return -1;
    }

    // (2) transpose to NCL [512, T]
    cl_mem x_ncl = alloc(cl_ctx, sizeof(nnopt_storage_t) * C * T);
    int Cn = C, Tn = T;
    clSetKernelArg(g_k_transpose_NLC_to_NCL, 0, sizeof(cl_mem), &emb_out);
    clSetKernelArg(g_k_transpose_NLC_to_NCL, 1, sizeof(cl_mem), &x_ncl);
    clSetKernelArg(g_k_transpose_NLC_to_NCL, 2, sizeof(int), &Cn);
    clSetKernelArg(g_k_transpose_NLC_to_NCL, 3, sizeof(int), &Tn);
    size_t gws_tr[2] = {(size_t)T, (size_t)C};
    nnopt_enqueue_profiled(queue, g_k_transpose_NLC_to_NCL, 2, nullptr, gws_tr, nullptr, 0, nullptr, nullptr);

    // (3) 3-layer Conv1d(weight_norm) + LayerNorm + LeakyReLU(0.2)
    //   cnn[i].0 = Conv1d weight_norm (key prefix: text_encoder.module.cnn.<i>.0)
    //   cnn[i].1 = LayerNorm (key: cnn.<i>.1.gamma, cnn.<i>.1.beta)
    cl_mem conv_out = alloc(cl_ctx, sizeof(nnopt_storage_t) * C * T);
    cl_mem ln_out = alloc(cl_ctx, sizeof(nnopt_storage_t) * C * T);
    cl_mem cur = x_ncl;
    for (int i = 0; i < 3; ++i) {
        std::string ci = "text_encoder.module.cnn." + std::to_string(i);
        int Lo, Co;
        if (dec_conv1d_wn(cl_ctx, weights, queue, ci + ".0", cur, conv_out,
                          T, 1, 2, 1, 1, &Lo, &Co) != 0) {
            NNOPT_ERROR_FMT("text_encoder: conv%d failed", i); return -1;
        }
        // LayerNorm: gamma/beta
        cl_mem gm = weights.get_buffer(ci + ".1.gamma");
        cl_mem bt = weights.get_buffer(ci + ".1.beta");
        if (!gm || !bt) { NNOPT_ERROR_FMT("text_encoder: missing %s.1.gamma/beta", ci.c_str()); return -1; }
        float eps = 1e-5f;
        clSetKernelArg(g_k_layernorm_NCL, 0, sizeof(cl_mem), &conv_out);
        clSetKernelArg(g_k_layernorm_NCL, 1, sizeof(cl_mem), &gm);
        clSetKernelArg(g_k_layernorm_NCL, 2, sizeof(cl_mem), &bt);
        clSetKernelArg(g_k_layernorm_NCL, 3, sizeof(cl_mem), &ln_out);
        clSetKernelArg(g_k_layernorm_NCL, 4, sizeof(int), &Cn);
        clSetKernelArg(g_k_layernorm_NCL, 5, sizeof(int), &Tn);
        clSetKernelArg(g_k_layernorm_NCL, 6, sizeof(float), &eps);
        size_t gws_ln = (size_t)T;
        nnopt_enqueue_profiled(queue, g_k_layernorm_NCL, 1, nullptr, &gws_ln, nullptr, 0, nullptr, nullptr);
        // LeakyReLU(0.2)
        int N = C * T;
        float slope = 0.2f;
        clSetKernelArg(g_k_leaky, 0, sizeof(cl_mem), &ln_out);
        clSetKernelArg(g_k_leaky, 1, sizeof(int), &N);
        clSetKernelArg(g_k_leaky, 2, sizeof(float), &slope);
        size_t gws_lk = (size_t)N;
        nnopt_enqueue_profiled(queue, g_k_leaky, 1, nullptr, &gws_lk, nullptr, 0, nullptr, nullptr);
        // Copy ln_out -> cur for next iteration
        clEnqueueCopyBuffer(queue, ln_out, x_ncl, 0, 0, sizeof(nnopt_storage_t) * C * T, 0, nullptr, nullptr);
        cur = x_ncl;
    }

    // (4) Transpose to NLC for biLSTM
    cl_mem x_nlc = alloc(cl_ctx, sizeof(nnopt_storage_t) * T * C);
    clSetKernelArg(g_k_transpose_NCL_to_NLC, 0, sizeof(cl_mem), &x_ncl);
    clSetKernelArg(g_k_transpose_NCL_to_NLC, 1, sizeof(cl_mem), &x_nlc);
    clSetKernelArg(g_k_transpose_NCL_to_NLC, 2, sizeof(int), &Cn);
    clSetKernelArg(g_k_transpose_NCL_to_NLC, 3, sizeof(int), &Tn);
    size_t gws_tr2[2] = {(size_t)C, (size_t)T};
    nnopt_enqueue_profiled(queue, g_k_transpose_NCL_to_NLC, 2, nullptr, gws_tr2, nullptr, 0, nullptr, nullptr);

    // (5) biLSTM(512, 256, bidir) → [T, 512]
    if (prim_bilstm(cl_ctx, weights, queue, "text_encoder.module.lstm",
                    x_nlc, out, T, C, 256) != 0) {
        NNOPT_ERROR("text_encoder: biLSTM failed"); return -1;
    }

    NNOPT_LAYER_CHECK("text_encoder_full", queue, out, (size_t)T * C);
    for (cl_mem m : {emb_out, x_ncl, conv_out, ln_out, x_nlc}) if (m) clReleaseMemObject(m);
    return 0;
}
