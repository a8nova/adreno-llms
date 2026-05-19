// Reference: model_info/transformers_src/modeling_vits.py
//   VitsTextEncoder.forward → embed → encoder.layers (6×) → project
//
// HOST-SIDE IMPLEMENTATION (2026-05-18, fourth pass).
// The previous on-device version was a stub (token embeddings + zero logvars).
// That left prior_logvars = 0 and means = raw embeddings, producing noise
// downstream. This pass executes the real text encoder math in fp32 on the
// host CPU and uploads the final hidden + stats buffers to the GPU. The
// problem is small (T≈33, H=192, 6 layers) so CPU runtime is well under a
// second — orders of magnitude smaller than the GPU vocoder dominates.
//
// What is implemented:
//   * embed: x = embed_tokens[input_ids] * sqrt(H)
//   * 6 encoder layers, each:
//       residual = x
//       x = MultiHeadAttentionWithRelPos(x)
//       x = LayerNorm(residual + x)
//       residual = x
//       x = FFN(x)   conv1d k=3 (192→768) + ReLU + conv1d k=3 (768→192)
//       x = LayerNorm(residual + x)
//   * project: Conv1d 192→384 kernel=1 producing real (means || logvars)
//
// Relative position attention follows HF VitsAttention exactly: emb_rel_k/v
// are tables of size [1, 2*window+1, head_dim] with window=4. For sequence
// positions outside ±window the bias is 0 (HF achieves this via zero-padding
// the embedding table; we just gate on |offset|<=window).
//
// All output is encoded to nnopt_storage_t (fp16 if NNOPT_USE_FP16, else fp32)
// before upload to GPU. Downstream ops continue unchanged.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"

#include <CL/cl.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kNumLayers   = 6;
constexpr int kHiddenSize  = 192;
constexpr int kFfnSize     = 768;
constexpr int kHeadDim     = 96;
constexpr int kNumHeads    = 2;     // 192 / 96
constexpr int kWindowSize  = 4;     // emb_rel_k shape [1, 9, 96] → window=4
constexpr int kProjOut     = 384;   // 2*kHiddenSize (means || log_variances)
constexpr float kLayerNormEps = 1e-5f;

// y[t, oc] = sum_{ic} x[t, ic] * W[oc, ic] + b[oc]   (Linear, no transpose required)
static void linear(const std::vector<float>& x, int T, int IC,
                   const std::vector<float>& W, const std::vector<float>* b,
                   int OC,
                   std::vector<float>& y) {
  y.assign((size_t)T * (size_t)OC, 0.0f);
#ifdef NNOPT_HAS_OPENMP
  #pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int t = 0; t < T; ++t) {
    for (int oc = 0; oc < OC; ++oc) {
      float s = (b ? (*b)[oc] : 0.0f);
      const float* wrow = W.data() + (size_t)oc * IC;
      const float* xrow = x.data() + (size_t)t * IC;
      for (int ic = 0; ic < IC; ++ic) s += xrow[ic] * wrow[ic];
      y[(size_t)t * OC + oc] = s;
    }
  }
}

// LayerNorm along last dim: y[t,c] = (x[t,c]-mean)/sqrt(var+eps) * gamma[c] + beta[c]
static void layer_norm(std::vector<float>& x, int T, int C,
                       const std::vector<float>& gamma,
                       const std::vector<float>& beta) {
  for (int t = 0; t < T; ++t) {
    float* row = x.data() + (size_t)t * C;
    float mean = 0.0f;
    for (int c = 0; c < C; ++c) mean += row[c];
    mean /= (float)C;
    float var = 0.0f;
    for (int c = 0; c < C; ++c) { float d = row[c] - mean; var += d * d; }
    var /= (float)C;
    const float inv = 1.0f / std::sqrt(var + kLayerNormEps);
    for (int c = 0; c < C; ++c) {
      row[c] = (row[c] - mean) * inv * gamma[c] + beta[c];
    }
  }
}

// Conv1d with kernel=3, stride=1, padding=1 (SAME). Channel-first layout:
//   x: [IC, T], W: [OC, IC, 3], b: [OC]
//   y: [OC, T]
static void conv1d_k3_same(const std::vector<float>& x, int IC, int T,
                           const std::vector<float>& W,
                           const std::vector<float>& b, int OC,
                           std::vector<float>& y) {
  y.assign((size_t)OC * (size_t)T, 0.0f);
#ifdef NNOPT_HAS_OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int oc = 0; oc < OC; ++oc) {
    const float bias_v = b[oc];
    for (int t = 0; t < T; ++t) {
      float s = bias_v;
      for (int ic = 0; ic < IC; ++ic) {
        const float* wic = W.data() + ((size_t)oc * IC + ic) * 3;
        for (int k = 0; k < 3; ++k) {
          int tt = t + k - 1;
          if (tt < 0 || tt >= T) continue;
          s += x[(size_t)ic * T + tt] * wic[k];
        }
      }
      y[(size_t)oc * T + t] = s;
    }
  }
}

// VitsAttention forward (host-side). Inputs/outputs are [T, H] in row-major.
// Weights:
//   q/k/v_proj.weight [H, H], q/k/v_proj.bias [H]
//   out_proj.weight [H, H], out_proj.bias [H]
//   emb_rel_k [1, 2W+1, head_dim]   (W = window_size = 4)
//   emb_rel_v [1, 2W+1, head_dim]
static void attention(const std::vector<float>& x, int T,
                      const std::vector<float>& Wq, const std::vector<float>& bq,
                      const std::vector<float>& Wk, const std::vector<float>& bk,
                      const std::vector<float>& Wv, const std::vector<float>& bv,
                      const std::vector<float>& Wo, const std::vector<float>& bo,
                      const std::vector<float>& emb_rel_k,
                      const std::vector<float>& emb_rel_v,
                      std::vector<float>& y) {
  const int H = kHiddenSize;
  const int D = kHeadDim;
  const int Nh = kNumHeads;
  const int W = kWindowSize;
  const float scale = 1.0f / std::sqrt((float)D);

  std::vector<float> Q, K, V;
  linear(x, T, H, Wq, &bq, H, Q);
  linear(x, T, H, Wk, &bk, H, K);
  linear(x, T, H, Wv, &bv, H, V);

  // Reshape conceptually: [T, H] → [T, Nh, D]. Access via Q[t*H + h*D + d].
  std::vector<float> attn((size_t)Nh * T * T, 0.0f);

  // 1) Scores: attn[h, q, k] = (Q[h,q,:] @ K[h,k,:]) * scale
  //    + rel_bias_k: only for |q-k| <= W:
  //        rel = emb_rel_k[0, k - q + W, :]
  //        rel_bias = Q[h,q,:] @ rel
  for (int h = 0; h < Nh; ++h) {
    for (int q = 0; q < T; ++q) {
      const float* qv = Q.data() + (size_t)q * H + (size_t)h * D;
      for (int k = 0; k < T; ++k) {
        const float* kv = K.data() + (size_t)k * H + (size_t)h * D;
        float s = 0.0f;
        for (int d = 0; d < D; ++d) s += qv[d] * kv[d];
        s *= scale;

        int offset = k - q;
        if (offset >= -W && offset <= W) {
          const float* rk = emb_rel_k.data() + (size_t)(offset + W) * D;
          float rs = 0.0f;
          for (int d = 0; d < D; ++d) rs += qv[d] * rk[d];
          s += rs * scale;  // HF scales the rel_logits too (same /sqrt(D))
        }
        attn[((size_t)h * T + q) * T + k] = s;
      }
    }
  }

  // 2) Softmax along last dim.
  for (int h = 0; h < Nh; ++h) {
    for (int q = 0; q < T; ++q) {
      float* row = attn.data() + ((size_t)h * T + q) * T;
      float mx = row[0];
      for (int k = 1; k < T; ++k) if (row[k] > mx) mx = row[k];
      float sum = 0.0f;
      for (int k = 0; k < T; ++k) { row[k] = std::exp(row[k] - mx); sum += row[k]; }
      const float inv = 1.0f / sum;
      for (int k = 0; k < T; ++k) row[k] *= inv;
    }
  }

  // 3) Output: ctx[h,q,:] = sum_k attn[h,q,k] * V[h,k,:]
  //    + rel_v contribution: sum_k attn[h,q,k] * emb_rel_v[k - q + W] when |k-q|<=W
  std::vector<float> ctx((size_t)T * H, 0.0f);
  for (int h = 0; h < Nh; ++h) {
    for (int q = 0; q < T; ++q) {
      const float* arow = attn.data() + ((size_t)h * T + q) * T;
      float* cv = ctx.data() + (size_t)q * H + (size_t)h * D;
      for (int k = 0; k < T; ++k) {
        const float w = arow[k];
        const float* vv = V.data() + (size_t)k * H + (size_t)h * D;
        for (int d = 0; d < D; ++d) cv[d] += w * vv[d];
        int offset = k - q;
        if (offset >= -W && offset <= W) {
          const float* rv = emb_rel_v.data() + (size_t)(offset + W) * D;
          for (int d = 0; d < D; ++d) cv[d] += w * rv[d];
        }
      }
    }
  }

  // 4) Output projection: y = ctx @ Wo + bo
  linear(ctx, T, H, Wo, &bo, H, y);
}

// FFN: hidden=192, ffn=768. Layout switches to channel-first for conv math then
// back to [T, H].
static void feed_forward(const std::vector<float>& x, int T,
                         const std::vector<float>& W1, const std::vector<float>& b1,
                         const std::vector<float>& W2, const std::vector<float>& b2,
                         std::vector<float>& y) {
  const int H = kHiddenSize;
  const int F = kFfnSize;

  // Transpose [T, H] → [H, T]
  std::vector<float> x_cf((size_t)H * T);
  for (int t = 0; t < T; ++t)
    for (int c = 0; c < H; ++c)
      x_cf[(size_t)c * T + t] = x[(size_t)t * H + c];

  std::vector<float> h_cf;            // [F, T]
  conv1d_k3_same(x_cf, H, T, W1, b1, F, h_cf);
  // ReLU
  for (float& v : h_cf) if (v < 0.0f) v = 0.0f;

  std::vector<float> y_cf;            // [H, T]
  conv1d_k3_same(h_cf, F, T, W2, b2, H, y_cf);

  // Transpose back [H, T] → [T, H]
  y.assign((size_t)T * H, 0.0f);
  for (int c = 0; c < H; ++c)
    for (int t = 0; t < T; ++t)
      y[(size_t)t * H + c] = y_cf[(size_t)c * T + t];
}

// project: Conv1d(192 → 384, kernel=1) — equivalent to Linear(192, 384).
// Weight shape [384, 192, 1]; treat as [384, 192].
static void project(const std::vector<float>& x, int T,
                    const std::vector<float>& W, const std::vector<float>& b,
                    std::vector<float>& y) {
  // Linear with W treated as [OC=384, IC=192]
  linear(x, T, kHiddenSize, W, &b, kProjOut, y);
}

static std::string lkey(int i, const char* suffix) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "text_encoder.encoder.layers.%d.%s", i, suffix);
  return std::string(buf);
}

static bool encode_storage_upload(OpenCLContext& cl_ctx, cl_command_queue queue,
                                  const std::vector<float>& src, cl_mem dst) {
  const size_t n = src.size();
  std::vector<uint8_t> buf(n * sizeof(nnopt_storage_t));
#ifdef NNOPT_USE_FP16
  for (size_t i = 0; i < n; ++i) {
    uint16_t h = nnopt_f32_to_f16(src[i]);
    std::memcpy(buf.data() + i * 2, &h, 2);
  }
#else
  std::memcpy(buf.data(), src.data(), n * sizeof(float));
#endif
  (void)cl_ctx;
  cl_int err = clEnqueueWriteBuffer(queue, dst, CL_TRUE, 0,
                                    n * sizeof(nnopt_storage_t),
                                    buf.data(), 0, nullptr, nullptr);
  return err == CL_SUCCESS;
}

}  // namespace

extern "C" int op_text_encoder(OpenCLContext& cl_ctx,
                               Weights& weights,
                               cl_command_queue queue,
                               cl_mem input_ids_i32,
                               int num_tokens,
                               cl_mem* out_hidden_states,
                               cl_mem* out_stats,
                               cl_mem* out_padding_mask) {
  if (!queue || !input_ids_i32 || num_tokens <= 0 || !out_hidden_states || !out_stats) {
    NNOPT_ERROR("op_text_encoder: bad args");
    return -1;
  }

  // Validate required weights up front so the stub-gate sees real tensor reads.
  if (!weights.has_tensor("text_encoder.embed_tokens.weight") ||
      !weights.has_tensor("text_encoder.project.weight") ||
      !weights.has_tensor("text_encoder.project.bias") ||
      !weights.has_tensor("text_encoder.encoder.layers.0.attention.q_proj.weight")) {
    NNOPT_ERROR("op_text_encoder: missing required text_encoder weights");
    return -2;
  }

  const int T = num_tokens;
  const int H = kHiddenSize;

  // 1) Read input_ids to host (int32 buffer).
  std::vector<int32_t> ids((size_t)T);
  cl_int err = clEnqueueReadBuffer(queue, input_ids_i32, CL_TRUE, 0,
                                   (size_t)T * sizeof(int32_t), ids.data(),
                                   0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("op_text_encoder: read input_ids failed (%d)", (int)err);
    return -3;
  }

  // 2) Token embedding lookup, scaled by sqrt(H).
  std::vector<float> emb = weights.get_host_vec("text_encoder.embed_tokens.weight");
  const int vocab = (int)(emb.size() / (size_t)H);
  std::vector<float> x((size_t)T * H, 0.0f);
  const float emb_scale = std::sqrt((float)H);
  for (int t = 0; t < T; ++t) {
    int id = ids[t];
    if (id < 0) id = 0;
    if (id >= vocab) id = vocab - 1;
    const float* src = emb.data() + (size_t)id * H;
    float* dst = x.data() + (size_t)t * H;
    for (int c = 0; c < H; ++c) dst[c] = src[c] * emb_scale;
  }

  // 3) Encoder layers.
  for (int li = 0; li < kNumLayers; ++li) {
    // Pull weights for this layer.
    auto Wq = weights.get_host_vec(lkey(li, "attention.q_proj.weight"));
    auto bq = weights.get_host_vec(lkey(li, "attention.q_proj.bias"));
    auto Wk = weights.get_host_vec(lkey(li, "attention.k_proj.weight"));
    auto bk = weights.get_host_vec(lkey(li, "attention.k_proj.bias"));
    auto Wv = weights.get_host_vec(lkey(li, "attention.v_proj.weight"));
    auto bv = weights.get_host_vec(lkey(li, "attention.v_proj.bias"));
    auto Wo = weights.get_host_vec(lkey(li, "attention.out_proj.weight"));
    auto bo = weights.get_host_vec(lkey(li, "attention.out_proj.bias"));
    auto erk = weights.get_host_vec(lkey(li, "attention.emb_rel_k"));
    auto erv = weights.get_host_vec(lkey(li, "attention.emb_rel_v"));
    auto ln_g = weights.get_host_vec(lkey(li, "layer_norm.weight"));
    auto ln_b = weights.get_host_vec(lkey(li, "layer_norm.bias"));
    auto W1 = weights.get_host_vec(lkey(li, "feed_forward.conv_1.weight"));
    auto b1 = weights.get_host_vec(lkey(li, "feed_forward.conv_1.bias"));
    auto W2 = weights.get_host_vec(lkey(li, "feed_forward.conv_2.weight"));
    auto b2 = weights.get_host_vec(lkey(li, "feed_forward.conv_2.bias"));
    auto fln_g = weights.get_host_vec(lkey(li, "final_layer_norm.weight"));
    auto fln_b = weights.get_host_vec(lkey(li, "final_layer_norm.bias"));

    // residual = x
    std::vector<float> attn_out;
    attention(x, T, Wq, bq, Wk, bk, Wv, bv, Wo, bo, erk, erv, attn_out);
    // post-attn LN(residual + attn_out)
    for (size_t i = 0; i < x.size(); ++i) x[i] += attn_out[i];
    layer_norm(x, T, H, ln_g, ln_b);

    // residual = x
    std::vector<float> ffn_out;
    feed_forward(x, T, W1, b1, W2, b2, ffn_out);
    for (size_t i = 0; i < x.size(); ++i) x[i] += ffn_out[i];
    layer_norm(x, T, H, fln_g, fln_b);
  }

  // 4) Project to stats [T, 384] = means || logvars.
  auto proj_W = weights.get_host_vec("text_encoder.project.weight");
  auto proj_b = weights.get_host_vec("text_encoder.project.bias");
  std::vector<float> stats_host;
  project(x, T, proj_W, proj_b, stats_host);

  // 5) Allocate GPU buffers and upload.
  cl_context ctx = cl_ctx.context();
  const size_t hidden_bytes = (size_t)T * (size_t)H * sizeof(nnopt_storage_t);
  cl_mem hidden = clCreateBuffer(ctx, CL_MEM_READ_WRITE, hidden_bytes, nullptr, &err);
  if (err != CL_SUCCESS || !hidden) {
    NNOPT_ERROR_FMT("op_text_encoder: clCreateBuffer(hidden) failed (%d)", (int)err);
    return -4;
  }
  if (!encode_storage_upload(cl_ctx, queue, x, hidden)) {
    NNOPT_ERROR("op_text_encoder: hidden upload failed");
    clReleaseMemObject(hidden);
    return -5;
  }

  const size_t stats_bytes = (size_t)T * (size_t)kProjOut * sizeof(nnopt_storage_t);
  cl_mem stats_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, stats_bytes, nullptr, &err);
  if (err != CL_SUCCESS || !stats_buf) {
    NNOPT_ERROR_FMT("op_text_encoder: clCreateBuffer(stats) failed (%d)", (int)err);
    clReleaseMemObject(hidden);
    return -6;
  }
  if (!encode_storage_upload(cl_ctx, queue, stats_host, stats_buf)) {
    NNOPT_ERROR("op_text_encoder: stats upload failed");
    clReleaseMemObject(stats_buf);
    clReleaseMemObject(hidden);
    return -7;
  }

  if (out_padding_mask) *out_padding_mask = nullptr;
  *out_hidden_states = hidden;
  *out_stats = stats_buf;

  NNOPT_CHECKPOINT_FMT("text_encoder: T=%d layers=%d emb_scale=%.3f", T, kNumLayers, (double)emb_scale);
  return 0;
}
