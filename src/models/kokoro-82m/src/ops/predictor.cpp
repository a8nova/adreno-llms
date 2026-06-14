// Reference: kokoro/modules.py::ProsodyPredictor + DurationEncoder
//
// op_predictor_durations: still uses 4 frames/char stub (real DurationEncoder
//   + duration_proj would replace this). For audible output the F0Ntrain side
//   matters more.
// op_predictor_F0N: REAL implementation of ProsodyPredictor.F0Ntrain:
//   x = shared_biLSTM(en.T)            # [T, 640] -> [T, 512]
//   F0_x = x.T                          # [512, T]
//   for block in F0 (3 AdainResBlk1d, block[1] upsamples):
//     F0_x = block(F0_x, s)
//   F0_out = F0_proj(F0_x)              # plain Conv1d 256 -> 1, kernel=1
//   N_out  = same path with N blocks + N_proj
//   return F0_out.squeeze(1), N_out.squeeze(1)

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

// LSTM primitive (lstm.cpp).
extern "C" int prim_bilstm(OpenCLContext&, Weights&, cl_command_queue,
                            const std::string&, cl_mem, cl_mem, int, int, int);

// Decoder helpers exposed for predictor (decoder.cpp).
extern "C" int dec_apply_adainresblk1d(OpenCLContext&, Weights&, cl_command_queue,
                                        const std::string&, cl_mem, cl_mem,
                                        int, int, int, bool, cl_mem, int*);
extern "C" int dec_conv1d_plain(OpenCLContext&, Weights&, cl_command_queue,
                                 const std::string&, cl_mem, cl_mem,
                                 int, int, int, int, int, int*, int*);

static cl_mem alloc(OpenCLContext& cl_ctx, size_t bytes) {
    cl_int e = CL_SUCCESS;
    return clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &e);
}

// Tiny in-line kernels (transpose NCL<->NLC) for the predictor.
static cl_program g_pred_prog = nullptr;
static cl_kernel  g_k_transpose_NCL_to_NLC = nullptr;
static cl_kernel  g_k_transpose_NLC_to_NCL = nullptr;
static cl_kernel  g_k_concat_at = nullptr;
static cl_kernel  g_k_broadcast_style = nullptr;

static const char* k_pred_src = R"CLC(
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

// [C, T] -> [T, C]
__kernel void transpose_NCL_to_NLC(__global const storage_t* in,
                                   __global storage_t* out, int C, int T) {
    int c = get_global_id(0); int t = get_global_id(1);
    if (c >= C || t >= T) return;
    STORE(out, t*C + c, (float)LOAD(in, c*T + t));
}
// [T, C] -> [C, T]
__kernel void transpose_NLC_to_NCL(__global const storage_t* in,
                                   __global storage_t* out, int C, int T) {
    int t = get_global_id(0); int c = get_global_id(1);
    if (t >= T || c >= C) return;
    STORE(out, c*T + t, (float)LOAD(in, t*C + c));
}

// Concat [Ca, T] + [Cb, T] -> [Ca+Cb, T]
__kernel void concat_at_C(__global const storage_t* a, int Ca,
                          __global const storage_t* b, int Cb,
                          __global storage_t* y, int T) {
    int oc = get_global_id(0); int t = get_global_id(1);
    int Ct = Ca + Cb;
    if (oc >= Ct || t >= T) return;
    float v = (oc < Ca) ? (float)LOAD(a, oc*T + t) : (float)LOAD(b, (oc - Ca)*T + t);
    STORE(y, oc*T + t, v);
}

// Broadcast style [C] to [C, T] (replicate across T frames).
__kernel void broadcast_style(__global const storage_t* s,
                              __global storage_t* y, int C, int T) {
    int c = get_global_id(0); int t = get_global_id(1);
    if (c >= C || t >= T) return;
    STORE(y, c*T + t, (float)LOAD(s, c));
}
)CLC";

static bool ensure_built_pred(OpenCLContext& cl_ctx) {
    if (g_k_transpose_NCL_to_NLC) return true;
    cl_int err = CL_SUCCESS;
    const char* opts =
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1";
#else
        "";
#endif
    cl_device_id dev = cl_ctx.device();
    g_pred_prog = nnopt_build_program_cached(cl_ctx.context(), dev, k_pred_src, opts, "predictor", &err);
    if (!g_pred_prog) return false;
    g_k_transpose_NCL_to_NLC = clCreateKernel(g_pred_prog, "transpose_NCL_to_NLC", &err);
    g_k_transpose_NLC_to_NCL = clCreateKernel(g_pred_prog, "transpose_NLC_to_NCL", &err);
    g_k_concat_at = clCreateKernel(g_pred_prog, "concat_at_C", &err);
    g_k_broadcast_style = clCreateKernel(g_pred_prog, "broadcast_style", &err);
    return g_k_transpose_NCL_to_NLC && g_k_transpose_NLC_to_NCL && g_k_concat_at && g_k_broadcast_style;
}

// Real predictor durations: takes d_NLC [T, 640] (output of DurationEncoder),
// runs biLSTM(640→512) → duration_proj Linear(512→50) → sigmoid → sum → round → clamp(1).
extern "C" int op_predictor_durations_real(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                            cl_mem d_NLC,    // [T, 640]
                                            int T, std::vector<int>& pred_dur) {
    if (!ensure_built_pred(cl_ctx)) return -1;
    const int H = 256;  // d_hid//2
    const int twoH = 512;
    const int max_dur = 50;

    // biLSTM(640→512)
    cl_mem x_lstm = alloc(cl_ctx, sizeof(nnopt_storage_t) * T * twoH);
    if (prim_bilstm(cl_ctx, weights, queue, "predictor.module.lstm",
                    d_NLC, x_lstm, T, 640, H) != 0) {
        NNOPT_ERROR("predictor.lstm failed"); return -1;
    }

    // duration_proj.linear_layer: Linear(512, 50)
    cl_mem proj_w = weights.get_buffer("predictor.module.duration_proj.linear_layer.weight");
    cl_mem proj_b = weights.get_buffer("predictor.module.duration_proj.linear_layer.bias");
    if (!proj_w || !proj_b) { NNOPT_ERROR("duration_proj missing"); clReleaseMemObject(x_lstm); return -1; }
    cl_mem dur_out = alloc(cl_ctx, sizeof(nnopt_storage_t) * T * max_dur);
    if (!pytorch_linear(queue, T, max_dur, twoH, x_lstm, proj_w, dur_out)) {
        NNOPT_ERROR("duration_proj pytorch_linear failed"); return -1;
    }
    // add bias
    {
        std::vector<float> b_f32(max_dur), out_f32(T * max_dur);
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> bh(max_dur), oh(T * max_dur);
        clEnqueueReadBuffer(queue, proj_b, CL_TRUE, 0, sizeof(uint16_t)*max_dur, bh.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, dur_out, CL_TRUE, 0, sizeof(uint16_t)*T*max_dur, oh.data(), 0, nullptr, nullptr);
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < max_dur; ++k) {
                float v = nnopt_f16_to_f32(oh[t*max_dur + k]) + nnopt_f16_to_f32(bh[k]);
                oh[t*max_dur + k] = nnopt_f32_to_f16(v);
                out_f32[t*max_dur + k] = v;
            }
        clEnqueueWriteBuffer(queue, dur_out, CL_TRUE, 0, sizeof(uint16_t)*T*max_dur, oh.data(), 0, nullptr, nullptr);
#else
        clEnqueueReadBuffer(queue, proj_b, CL_TRUE, 0, sizeof(float)*max_dur, b_f32.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, dur_out, CL_TRUE, 0, sizeof(float)*T*max_dur, out_f32.data(), 0, nullptr, nullptr);
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < max_dur; ++k) out_f32[t*max_dur + k] += b_f32[k];
        clEnqueueWriteBuffer(queue, dur_out, CL_TRUE, 0, sizeof(float)*T*max_dur, out_f32.data(), 0, nullptr, nullptr);
#endif
        // sigmoid + sum + round + clamp(1)
        pred_dur.resize((size_t)T);
        for (int t = 0; t < T; ++t) {
            float s = 0.0f;
            for (int k = 0; k < max_dur; ++k) {
                float v = out_f32[t*max_dur + k];
                s += 1.0f / (1.0f + std::exp(-v));
            }
            int d = (int)std::round(s);
            if (d < 1) d = 1;
            pred_dur[t] = d;
        }
    }
    clReleaseMemObject(x_lstm);
    clReleaseMemObject(dur_out);
    return 0;
}

extern "C" int op_predictor_durations(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                      cl_mem bert_hidden_T_512, cl_mem ref_s, int T,
                                      std::vector<int>& pred_dur) {
    (void)cl_ctx; (void)weights; (void)queue; (void)bert_hidden_T_512; (void)ref_s;
    pred_dur.assign((size_t)T, 4);
    return 0;
}

// AdaLayerNorm forward: x_NLC [T, C], style [128]
//   h = fc(style) [2C]; gamma, beta = split.
//   x_LN = layer_norm(x, channels=C, no affine, eps=1e-5)
//   y = (1+gamma)*x_LN + beta   per channel
// Operates on NLC (last dim is C).
static int apply_adalayernorm(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                              const std::string& prefix,  // "predictor.module.text_encoder.lstms.<i>"
                              cl_mem x_NLC, cl_mem y_NLC, cl_mem style128, int T, int C) {
    if (!ensure_built_pred(cl_ctx)) return -1;
    // fc weights
    cl_mem fc_w = weights.get_buffer(prefix + ".fc.weight");
    cl_mem fc_b = weights.get_buffer(prefix + ".fc.bias");
    if (!fc_w || !fc_b) { NNOPT_ERROR_FMT("adaln: missing %s.fc", prefix.c_str()); return -1; }
    int two_C = 2*C;
    cl_mem h = alloc(cl_ctx, sizeof(nnopt_storage_t) * two_C);
    // Linear: h[2C] = fc_w[2C, 128] @ style[128] + fc_b[2C].
    // pytorch_linear takes (M, N, K, x[M,K], W[N,K], out[M,N]) — for M=1, this is fine.
    if (!pytorch_linear(queue, 1, two_C, 128, style128, fc_w, h)) {
        NNOPT_ERROR("adaln: pytorch_linear failed"); return -1;
    }
    // add fc_b
    std::vector<float> bias_h(two_C);
#ifdef NNOPT_USE_FP16
    std::vector<uint16_t> bh(two_C), hh(two_C);
    clEnqueueReadBuffer(queue, h, CL_TRUE, 0, sizeof(uint16_t)*two_C, hh.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, fc_b, CL_TRUE, 0, sizeof(uint16_t)*two_C, bh.data(), 0, nullptr, nullptr);
    for (int i = 0; i < two_C; ++i) hh[i] = nnopt_f32_to_f16(nnopt_f16_to_f32(hh[i]) + nnopt_f16_to_f32(bh[i]));
    clEnqueueWriteBuffer(queue, h, CL_TRUE, 0, sizeof(uint16_t)*two_C, hh.data(), 0, nullptr, nullptr);
#else
    std::vector<float> bh(two_C), hh(two_C);
    clEnqueueReadBuffer(queue, h, CL_TRUE, 0, sizeof(float)*two_C, hh.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, fc_b, CL_TRUE, 0, sizeof(float)*two_C, bh.data(), 0, nullptr, nullptr);
    for (int i = 0; i < two_C; ++i) hh[i] += bh[i];
    clEnqueueWriteBuffer(queue, h, CL_TRUE, 0, sizeof(float)*two_C, hh.data(), 0, nullptr, nullptr);
#endif
    // Now do layer_norm (no affine) over last dim of x_NLC, then apply (1+gamma)*x_LN + beta
    // We'll do it host-side for simplicity at small T*C.
    int N = T * C;
    std::vector<float> x_f32(N), y_f32(N), gamma_f32(C), beta_f32(C);
#ifdef NNOPT_USE_FP16
    std::vector<uint16_t> x_h(N), y_h(N);
    clEnqueueReadBuffer(queue, x_NLC, CL_TRUE, 0, sizeof(uint16_t)*N, x_h.data(), 0, nullptr, nullptr);
    for (int i = 0; i < N; ++i) x_f32[i] = nnopt_f16_to_f32(x_h[i]);
    for (int i = 0; i < C; ++i) {
        gamma_f32[i] = nnopt_f16_to_f32(hh[i]);
        beta_f32[i]  = nnopt_f16_to_f32(hh[C + i]);
    }
#else
    clEnqueueReadBuffer(queue, x_NLC, CL_TRUE, 0, sizeof(float)*N, x_f32.data(), 0, nullptr, nullptr);
    for (int i = 0; i < C; ++i) { gamma_f32[i] = hh[i]; beta_f32[i] = hh[C+i]; }
#endif
    float eps = 1e-5f;
    for (int t = 0; t < T; ++t) {
        float mean = 0.0f, sq = 0.0f;
        for (int c = 0; c < C; ++c) mean += x_f32[t*C + c];
        mean /= (float)C;
        for (int c = 0; c < C; ++c) { float d = x_f32[t*C + c] - mean; sq += d*d; }
        float inv = 1.0f / sqrtf(sq / (float)C + eps);
        for (int c = 0; c < C; ++c) {
            float xn = (x_f32[t*C + c] - mean) * inv;
            y_f32[t*C + c] = (1.0f + gamma_f32[c]) * xn + beta_f32[c];
        }
    }
#ifdef NNOPT_USE_FP16
    std::vector<uint16_t> y_h2(N);
    for (int i = 0; i < N; ++i) y_h2[i] = nnopt_f32_to_f16(y_f32[i]);
    clEnqueueWriteBuffer(queue, y_NLC, CL_TRUE, 0, sizeof(uint16_t)*N, y_h2.data(), 0, nullptr, nullptr);
#else
    clEnqueueWriteBuffer(queue, y_NLC, CL_TRUE, 0, sizeof(float)*N, y_f32.data(), 0, nullptr, nullptr);
#endif
    clReleaseMemObject(h);
    return 0;
}

// DurationEncoder forward.
// Input: d_en [B, 512, T] (NCL, B=1).  Style: ref_s_pred [128].
// Output: d_NLC [T, 640] — for ProsodyPredictor's downstream use.
// Reference processes via 3 (biLSTM + AdaLayerNorm) layers. Each iteration:
//   biLSTM(x_NLC[T, 640]) → [T, 512]
//   AdaLayerNorm(NLC[T, 512]) → [T, 512]
//   cat with style broadcast [T, 128] → [T, 640]
extern "C" int op_predictor_duration_encoder(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                              cl_mem d_en_NLC,  // [T, 512]  (NLC, B=1)
                                              cl_mem ref_s,
                                              cl_mem d_out_NLC, // [T, 640]
                                              int T_chars) {
    if (!ensure_built_pred(cl_ctx)) return -1;
    // style[128:256] sub-buffer
    cl_buffer_region r;
    r.origin = (size_t)128 * sizeof(nnopt_storage_t);
    r.size = (size_t)128 * sizeof(nnopt_storage_t);
    cl_int err = CL_SUCCESS;
    cl_mem style128 = clCreateSubBuffer(ref_s, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &r, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR("DE: ref_s sub-buffer failed"); return -1; }
    // Build style broadcast [T, 128] for concat
    cl_mem style_bc = alloc(cl_ctx, sizeof(nnopt_storage_t) * T_chars * 128);
    // Use broadcast_style kernel on transposed? No, broadcast_style writes [C, T]. We want [T, 128].
    // For NLC layout we need [T, 128]. Let's just write directly via a host copy: replicate style across T rows.
    {
        std::vector<float> sf(128);
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> sh(128);
        clEnqueueReadBuffer(queue, style128, CL_TRUE, 0, sizeof(uint16_t)*128, sh.data(), 0, nullptr, nullptr);
        std::vector<uint16_t> rep((size_t)T_chars * 128);
        for (int t = 0; t < T_chars; ++t)
            for (int c = 0; c < 128; ++c) rep[t*128 + c] = sh[c];
        clEnqueueWriteBuffer(queue, style_bc, CL_TRUE, 0, sizeof(uint16_t) * T_chars * 128, rep.data(), 0, nullptr, nullptr);
#else
        clEnqueueReadBuffer(queue, style128, CL_TRUE, 0, sizeof(float)*128, sf.data(), 0, nullptr, nullptr);
        std::vector<float> rep((size_t)T_chars * 128);
        for (int t = 0; t < T_chars; ++t)
            for (int c = 0; c < 128; ++c) rep[t*128 + c] = sf[c];
        clEnqueueWriteBuffer(queue, style_bc, CL_TRUE, 0, sizeof(float) * T_chars * 128, rep.data(), 0, nullptr, nullptr);
#endif
    }

    // Initial: x = cat([d_en[T, 512], style_bc[T, 128]], dim=-1) -> [T, 640] (NLC)
    cl_mem x_640 = alloc(cl_ctx, sizeof(nnopt_storage_t) * T_chars * 640);
    {
        // Manual concat host-side (simpler than wiring a new kernel for NLC concat).
        int ns = T_chars * 512;
        int nb = T_chars * 128;
        std::vector<float> a(ns), b(nb), c(T_chars * 640);
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> ah(ns), bh(nb), ch((size_t)T_chars * 640);
        clEnqueueReadBuffer(queue, d_en_NLC, CL_TRUE, 0, sizeof(uint16_t)*ns, ah.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, style_bc, CL_TRUE, 0, sizeof(uint16_t)*nb, bh.data(), 0, nullptr, nullptr);
        for (int t = 0; t < T_chars; ++t) {
            for (int i = 0; i < 512; ++i) ch[t*640 + i] = ah[t*512 + i];
            for (int i = 0; i < 128; ++i) ch[t*640 + 512 + i] = bh[t*128 + i];
        }
        clEnqueueWriteBuffer(queue, x_640, CL_TRUE, 0, sizeof(uint16_t) * T_chars * 640, ch.data(), 0, nullptr, nullptr);
#else
        clEnqueueReadBuffer(queue, d_en_NLC, CL_TRUE, 0, sizeof(float)*ns, a.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, style_bc, CL_TRUE, 0, sizeof(float)*nb, b.data(), 0, nullptr, nullptr);
        for (int t = 0; t < T_chars; ++t) {
            for (int i = 0; i < 512; ++i) c[t*640 + i] = a[t*512 + i];
            for (int i = 0; i < 128; ++i) c[t*640 + 512 + i] = b[t*128 + i];
        }
        clEnqueueWriteBuffer(queue, x_640, CL_TRUE, 0, sizeof(float) * T_chars * 640, c.data(), 0, nullptr, nullptr);
#endif
    }

    // 3 iterations of biLSTM + AdaLayerNorm + cat_style
    cl_mem x_512 = alloc(cl_ctx, sizeof(nnopt_storage_t) * T_chars * 512);
    cl_mem x_ada = alloc(cl_ctx, sizeof(nnopt_storage_t) * T_chars * 512);
    for (int i = 0; i < 3; ++i) {
        // biLSTM: weight key prefix predictor.module.text_encoder.lstms.<2i>
        std::string lstm_prefix = "predictor.module.text_encoder.lstms." + std::to_string(2*i);
        if (prim_bilstm(cl_ctx, weights, queue, lstm_prefix, x_640, x_512, T_chars, 640, 256) != 0) {
            NNOPT_ERROR_FMT("DE: biLSTM %d failed", i);
            return -1;
        }
        // AdaLayerNorm: predictor.module.text_encoder.lstms.<2i+1>
        std::string ada_prefix = "predictor.module.text_encoder.lstms." + std::to_string(2*i+1);
        if (apply_adalayernorm(cl_ctx, weights, queue, ada_prefix, x_512, x_ada, style128, T_chars, 512) != 0) {
            NNOPT_ERROR_FMT("DE: AdaLN %d failed", i);
            return -1;
        }
        // cat([x_ada, style_bc], -1) into x_640 for next iteration
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> ah((size_t)T_chars * 512), bh((size_t)T_chars * 128), ch((size_t)T_chars * 640);
        clEnqueueReadBuffer(queue, x_ada,    CL_TRUE, 0, sizeof(uint16_t) * T_chars * 512, ah.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, style_bc, CL_TRUE, 0, sizeof(uint16_t) * T_chars * 128, bh.data(), 0, nullptr, nullptr);
        for (int t = 0; t < T_chars; ++t) {
            for (int j = 0; j < 512; ++j) ch[t*640 + j] = ah[t*512 + j];
            for (int j = 0; j < 128; ++j) ch[t*640 + 512 + j] = bh[t*128 + j];
        }
        clEnqueueWriteBuffer(queue, x_640, CL_TRUE, 0, sizeof(uint16_t) * T_chars * 640, ch.data(), 0, nullptr, nullptr);
#else
        std::vector<float> a((size_t)T_chars * 512), b((size_t)T_chars * 128), c((size_t)T_chars * 640);
        clEnqueueReadBuffer(queue, x_ada,    CL_TRUE, 0, sizeof(float) * T_chars * 512, a.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, style_bc, CL_TRUE, 0, sizeof(float) * T_chars * 128, b.data(), 0, nullptr, nullptr);
        for (int t = 0; t < T_chars; ++t) {
            for (int j = 0; j < 512; ++j) c[t*640 + j] = a[t*512 + j];
            for (int j = 0; j < 128; ++j) c[t*640 + 512 + j] = b[t*128 + j];
        }
        clEnqueueWriteBuffer(queue, x_640, CL_TRUE, 0, sizeof(float) * T_chars * 640, c.data(), 0, nullptr, nullptr);
#endif
    }

    // d_out_NLC = x_640 (after 3 iterations the result is [T, 640])
    clEnqueueCopyBuffer(queue, x_640, d_out_NLC, 0, 0, sizeof(nnopt_storage_t) * T_chars * 640, 0, nullptr, nullptr);

    clReleaseMemObject(style128);
    clReleaseMemObject(style_bc);
    clReleaseMemObject(x_640);
    clReleaseMemObject(x_512);
    clReleaseMemObject(x_ada);
    return 0;
}

// Real F0Ntrain.
// en_input expected at [640, T_frames] (NCL).
// F0_out, N_out: caller-allocated [T_frames*2] storage_t (F0Ntrain upsamples by 2 in middle block).
extern "C" int op_predictor_F0N(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                cl_mem en_input,    // [640, T_frames]
                                cl_mem ref_s,       // [256] — we use ref_s[128:] for predictor style
                                cl_mem F0_out,      // [T_frames*2]
                                cl_mem N_out,       // [T_frames*2]
                                int T_frames) {
    if (!ensure_built_pred(cl_ctx)) return -1;
    const int en_C = 640;  // d_hid + style_dim = 512 + 128
    const int H = 256;     // d_hid // 2
    const int two_H = 2 * H;  // = 512 (output of biLSTM)

    // Style for predictor: ref_s[128:] — pass as offset sub-buffer (256-128=128 floats).
    cl_buffer_region region_style_pred;
    region_style_pred.origin = (size_t)128 * sizeof(nnopt_storage_t);
    region_style_pred.size = (size_t)128 * sizeof(nnopt_storage_t);
    cl_int err = CL_SUCCESS;
    cl_mem ref_s_pred = clCreateSubBuffer(ref_s, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                                          &region_style_pred, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR("F0Ntrain: ref_s sub-buffer failed"); return -1; }

    // shared LSTM input: en.T = [T_frames, 640]
    cl_mem en_NLC = alloc(cl_ctx, sizeof(nnopt_storage_t) * T_frames * en_C);
    clSetKernelArg(g_k_transpose_NCL_to_NLC, 0, sizeof(cl_mem), &en_input);
    clSetKernelArg(g_k_transpose_NCL_to_NLC, 1, sizeof(cl_mem), &en_NLC);
    int Cn = en_C, Tn = T_frames;
    clSetKernelArg(g_k_transpose_NCL_to_NLC, 2, sizeof(int), &Cn);
    clSetKernelArg(g_k_transpose_NCL_to_NLC, 3, sizeof(int), &Tn);
    size_t gws_t[2] = {(size_t)en_C, (size_t)T_frames};
    nnopt_enqueue_profiled(queue, g_k_transpose_NCL_to_NLC, 2, nullptr, gws_t, nullptr, 0, nullptr, nullptr);

    // Run shared biLSTM -> [T_frames, 512]
    cl_mem x_lstm = alloc(cl_ctx, sizeof(nnopt_storage_t) * T_frames * two_H);
    if (prim_bilstm(cl_ctx, weights, queue, "predictor.module.shared",
                    en_NLC, x_lstm, T_frames, en_C, H) != 0) {
        NNOPT_ERROR("F0Ntrain: shared biLSTM failed");
        return -1;
    }

    // Transpose to [512, T_frames]
    cl_mem F0_x = alloc(cl_ctx, sizeof(nnopt_storage_t) * two_H * T_frames);
    cl_mem N_x  = alloc(cl_ctx, sizeof(nnopt_storage_t) * two_H * T_frames);
    int Cnl = two_H;
    clSetKernelArg(g_k_transpose_NLC_to_NCL, 0, sizeof(cl_mem), &x_lstm);
    clSetKernelArg(g_k_transpose_NLC_to_NCL, 1, sizeof(cl_mem), &F0_x);
    clSetKernelArg(g_k_transpose_NLC_to_NCL, 2, sizeof(int), &Cnl);
    clSetKernelArg(g_k_transpose_NLC_to_NCL, 3, sizeof(int), &Tn);
    size_t gws_t2[2] = {(size_t)T_frames, (size_t)two_H};
    nnopt_enqueue_profiled(queue, g_k_transpose_NLC_to_NCL, 2, nullptr, gws_t2, nullptr, 0, nullptr, nullptr);
    // Copy F0_x -> N_x (both start from x.T)
    clEnqueueCopyBuffer(queue, F0_x, N_x, 0, 0, sizeof(nnopt_storage_t) * two_H * T_frames, 0, nullptr, nullptr);

    // F0 blocks: AdainResBlk1d(512, 512), AdainResBlk1d(512, 256, upsample=True), AdainResBlk1d(256, 256)
    auto run_blocks = [&](const std::string& base, cl_mem x_in, cl_mem* x_out_p, int* T_out_p) -> int {
        // Block 0: 512 -> 512
        int T_cur = T_frames;
        cl_mem b0_out = alloc(cl_ctx, sizeof(nnopt_storage_t) * 512 * T_cur);
        int Tnew0;
        if (dec_apply_adainresblk1d(cl_ctx, weights, queue, base + ".0", x_in, b0_out,
                                     512, 512, T_cur, false, ref_s_pred, &Tnew0) != 0) return -1;
        // Block 1: 512 -> 256, upsample
        int T_after1 = T_cur * 2;
        cl_mem b1_out = alloc(cl_ctx, sizeof(nnopt_storage_t) * 256 * T_after1);
        int Tnew1;
        if (dec_apply_adainresblk1d(cl_ctx, weights, queue, base + ".1", b0_out, b1_out,
                                     512, 256, T_cur, true, ref_s_pred, &Tnew1) != 0) return -1;
        clReleaseMemObject(b0_out);
        // Block 2: 256 -> 256
        cl_mem b2_out = alloc(cl_ctx, sizeof(nnopt_storage_t) * 256 * Tnew1);
        int Tnew2;
        if (dec_apply_adainresblk1d(cl_ctx, weights, queue, base + ".2", b1_out, b2_out,
                                     256, 256, Tnew1, false, ref_s_pred, &Tnew2) != 0) return -1;
        clReleaseMemObject(b1_out);
        *x_out_p = b2_out;
        *T_out_p = Tnew2;
        return 0;
    };

    cl_mem F0_final = nullptr; int F0_T = 0;
    if (run_blocks("predictor.module.F0", F0_x, &F0_final, &F0_T) != 0) { NNOPT_ERROR("F0Ntrain: F0 blocks"); return -1; }
    cl_mem N_final  = nullptr; int N_T  = 0;
    if (run_blocks("predictor.module.N",  N_x,  &N_final,  &N_T)  != 0) { NNOPT_ERROR("F0Ntrain: N blocks");  return -1; }

    // F0_proj: plain Conv1d(256, 1, k=1)
    cl_mem F0_proj_out = alloc(cl_ctx, sizeof(nnopt_storage_t) * F0_T);
    int Lf, Cf;
    if (dec_conv1d_plain(cl_ctx, weights, queue, "predictor.module.F0_proj",
                          F0_final, F0_proj_out, F0_T, 1, 0, 1, 1, &Lf, &Cf) != 0) {
        NNOPT_ERROR("F0Ntrain: F0_proj");
        return -1;
    }
    cl_mem N_proj_out = alloc(cl_ctx, sizeof(nnopt_storage_t) * N_T);
    int Ln, Cn2;
    if (dec_conv1d_plain(cl_ctx, weights, queue, "predictor.module.N_proj",
                          N_final, N_proj_out, N_T, 1, 0, 1, 1, &Ln, &Cn2) != 0) {
        NNOPT_ERROR("F0Ntrain: N_proj");
        return -1;
    }

    // F0_out/N_out are sized [T_frames] in current backbone, but F0_T = T_frames*2 from real upsample.
    // The decoder's F0_conv has stride=2 — so feeding T_frames*2 -> halves -> T_frames. That matches.
    // Copy F0_proj_out (length F0_T = T_frames*2) into F0_out (caller allocated T_frames*2).
    clEnqueueCopyBuffer(queue, F0_proj_out, F0_out, 0, 0,
                        sizeof(nnopt_storage_t) * F0_T, 0, nullptr, nullptr);
    clEnqueueCopyBuffer(queue, N_proj_out,  N_out,  0, 0,
                        sizeof(nnopt_storage_t) * N_T,  0, nullptr, nullptr);

    for (cl_mem m : {en_NLC, x_lstm, F0_x, N_x, F0_final, N_final, F0_proj_out, N_proj_out, ref_s_pred}) {
        if (m) clReleaseMemObject(m);
    }
    return 0;
}

// Helper used by backbone to build en_640 from d_en[512, T] + style:
// out [640, T_frames] = concat(gathered_d_en[512, T_frames], broadcast(style)[128, T_frames]) along C.
extern "C" int op_predictor_build_en640(OpenCLContext& cl_ctx, cl_command_queue queue,
                                         cl_mem d_en_gathered_512, cl_mem ref_s,
                                         cl_mem en640_out, int T_frames) {
    if (!ensure_built_pred(cl_ctx)) return -1;
    // Broadcast style[128:256] across T_frames frames
    cl_buffer_region r;
    r.origin = (size_t)128 * sizeof(nnopt_storage_t);
    r.size = (size_t)128 * sizeof(nnopt_storage_t);
    cl_int err = CL_SUCCESS;
    cl_mem style_part = clCreateSubBuffer(ref_s, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &r, &err);
    cl_mem style_broadcast = alloc(cl_ctx, sizeof(nnopt_storage_t) * 128 * T_frames);
    int C128 = 128;
    clSetKernelArg(g_k_broadcast_style, 0, sizeof(cl_mem), &style_part);
    clSetKernelArg(g_k_broadcast_style, 1, sizeof(cl_mem), &style_broadcast);
    clSetKernelArg(g_k_broadcast_style, 2, sizeof(int), &C128);
    clSetKernelArg(g_k_broadcast_style, 3, sizeof(int), &T_frames);
    size_t gws_b[2] = {(size_t)128, (size_t)T_frames};
    nnopt_enqueue_profiled(queue, g_k_broadcast_style, 2, nullptr, gws_b, nullptr, 0, nullptr, nullptr);
    // Concat
    int Ca = 512, Cb = 128;
    clSetKernelArg(g_k_concat_at, 0, sizeof(cl_mem), &d_en_gathered_512);
    clSetKernelArg(g_k_concat_at, 1, sizeof(int), &Ca);
    clSetKernelArg(g_k_concat_at, 2, sizeof(cl_mem), &style_broadcast);
    clSetKernelArg(g_k_concat_at, 3, sizeof(int), &Cb);
    clSetKernelArg(g_k_concat_at, 4, sizeof(cl_mem), &en640_out);
    clSetKernelArg(g_k_concat_at, 5, sizeof(int), &T_frames);
    size_t gws_c[2] = {640, (size_t)T_frames};
    nnopt_enqueue_profiled(queue, g_k_concat_at, 2, nullptr, gws_c, nullptr, 0, nullptr, nullptr);
    clReleaseMemObject(style_part);
    clReleaseMemObject(style_broadcast);
    return 0;
}
