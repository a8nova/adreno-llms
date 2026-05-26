// GPU text encoder for VITS. Runs 6 transformer layers entirely on GPU:
//   embed → 6×(attention + LN + FFN_conv1d + LN) → project
// No CPU roundtrips between layers. Uses CLBlast HGEMM for linear projections
// and conv1d_gpu for FFN conv1d, custom OpenCL kernels for attention/LN/etc.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"
#include "conv1d_gpu.h"
#include "profiler.h"

#include <CL/cl.h>
#include <clblast.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int H   = 192;
constexpr int FFN = 768;
constexpr int D   = 96;
constexpr int Nh  = 2;
constexpr int W   = 4;
constexpr int NL  = 6;
constexpr float LN_EPS = 1e-5f;

static cl_kernel k_embed = nullptr;
static cl_kernel k_ln_res = nullptr;
static cl_kernel k_fused_attn = nullptr;
static cl_kernel k_transpose = nullptr;
static cl_kernel k_relu = nullptr;
static cl_kernel k_copy = nullptr;
static cl_kernel k_bias_row = nullptr;
static cl_program g_prog = nullptr;

static bool ensure_kernels(OpenCLContext& cl_ctx) {
  if (g_prog) return true;
  g_prog = cl_ctx.get_program("kernels/text_encoder.cl");
  if (!g_prog) g_prog = cl_ctx.build_program_from_file("kernels/text_encoder.cl");
  if (!g_prog) { NNOPT_ERROR("te: program build failed"); return false; }
  cl_int e;
  k_embed = clCreateKernel(g_prog, "te_embed_scale", &e);
  if (e) { NNOPT_ERROR("te: kernel te_embed_scale"); return false; }
  k_ln_res = clCreateKernel(g_prog, "te_layer_norm_residual", &e);
  if (e) { NNOPT_ERROR("te: kernel te_layer_norm_residual"); return false; }
  k_fused_attn = clCreateKernel(g_prog, "te_fused_attention", &e);
  if (e) { NNOPT_ERROR("te: kernel te_fused_attention"); return false; }
  k_transpose = clCreateKernel(g_prog, "te_transpose", &e);
  if (e) { NNOPT_ERROR("te: kernel te_transpose"); return false; }
  k_relu = clCreateKernel(g_prog, "te_relu", &e);
  if (e) { NNOPT_ERROR("te: kernel te_relu"); return false; }
  k_copy = clCreateKernel(g_prog, "te_copy", &e);
  if (e) { NNOPT_ERROR("te: kernel te_copy"); return false; }
  k_bias_row = clCreateKernel(g_prog, "te_bias_rowmajor", &e);
  if (e) { NNOPT_ERROR("te: kernel te_bias_rowmajor"); return false; }
  return true;
}

static bool set_arg(cl_kernel k, cl_uint i, size_t sz, const void* v) {
  return clSetKernelArg(k, i, sz, v) == CL_SUCCESS;
}

// CLBlast HGEMM: y[T, OC] = x[T, IC] × W^T[IC, OC] + bias
// W stored as [OC, IC], bias as [OC]. bias_buf can be nullptr.
// When bias is present: pre-fill y with replicated bias rows on host (tiny:
// OC ≤ 768, T ≤ 512 → 0.8 MB max), then GEMM with beta=1.
static cl_mem linear_gpu(OpenCLContext& cl_ctx, cl_command_queue queue,
                         cl_mem x, cl_mem w_buf, cl_mem bias_buf,
                         int T, int IC, int OC, const char* label) {
  cl_int err;
  const size_t n = (size_t)T * OC;
  cl_mem y = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                            n * sizeof(nnopt_storage_t), nullptr, &err);
  if (err || !y) return nullptr;

  // GEMM: y = x × W^T (beta=0, no bias yet)
#ifdef NNOPT_USE_FP16
  cl_half h_one  = (cl_half)nnopt_f32_to_f16(1.0f);
  cl_half h_zero = (cl_half)nnopt_f32_to_f16(0.0f);
  auto st = clblast::Gemm<cl_half>(
      clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kYes,
      T, OC, IC,
      h_one, x, 0, IC, w_buf, 0, IC,
      h_zero, y, 0, OC,
      &queue, KernelProfiler::event_for(label));
#else
  auto st = clblast::Gemm<float>(
      clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kYes,
      T, OC, IC,
      1.0f, x, 0, IC, w_buf, 0, IC,
      0.0f, y, 0, OC,
      &queue, KernelProfiler::event_for(label));
#endif
  if (st != clblast::StatusCode::kSuccess) {
    NNOPT_ERROR_FMT("te linear_gpu(%s): GEMM status=%d M=%d N=%d K=%d",
                    label, (int)st, T, OC, IC);
    clReleaseMemObject(y); return nullptr;
  }

  // Bias add: y[t, oc] += b[oc] — fully on GPU, no host roundtrip.
  if (bias_buf) {
    set_arg(k_bias_row, 0, sizeof(cl_mem), &y);
    set_arg(k_bias_row, 1, sizeof(cl_mem), &bias_buf);
    set_arg(k_bias_row, 2, sizeof(int), &T);
    set_arg(k_bias_row, 3, sizeof(int), &OC);
    size_t gws_b[2] = {(size_t)T, (size_t)OC};
    clEnqueueNDRangeKernel(queue, k_bias_row, 2, nullptr, gws_b, nullptr, 0, nullptr, nullptr);
  }
  return y;
}

static std::string lkey(int i, const char* suffix) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "text_encoder.encoder.layers.%d.%s", i, suffix);
  return buf;
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
  if (!ensure_kernels(cl_ctx)) return -2;

  const int T = num_tokens;
  cl_int err;
  cl_context ctx = cl_ctx.context();

  auto t0 = std::chrono::steady_clock::now();

  // 1) Embedding: x[T, H] = embed[ids, :] * sqrt(H)
  cl_mem emb_buf = weights.get_buffer("text_encoder.embed_tokens.weight");
  if (!emb_buf) { NNOPT_ERROR("te: missing embed weight"); return -3; }

  cl_mem x = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                            (size_t)T * H * sizeof(nnopt_storage_t), nullptr, &err);
  if (err || !x) return -4;

  float emb_scale = std::sqrt((float)H);
  set_arg(k_embed, 0, sizeof(cl_mem), &emb_buf);
  set_arg(k_embed, 1, sizeof(cl_mem), &input_ids_i32);
  set_arg(k_embed, 2, sizeof(cl_mem), &x);
  int iT = T, iH = H;
  set_arg(k_embed, 3, sizeof(int), &iT);
  set_arg(k_embed, 4, sizeof(int), &iH);
  set_arg(k_embed, 5, sizeof(float), &emb_scale);
  size_t gws_emb[2] = {(size_t)T, (size_t)H};
  clEnqueueNDRangeKernel(queue, k_embed, 2, nullptr, gws_emb, nullptr, 0, nullptr,
                         KernelProfiler::event_for("te.embed"));

  // Scratch buffers reused across layers
  cl_mem residual = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                   (size_t)T * H * sizeof(nnopt_storage_t), nullptr, &err);
  cl_mem attn_ctx = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                   (size_t)T * H * sizeof(nnopt_storage_t), nullptr, &err);
  cl_mem x_cf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                               (size_t)H * T * sizeof(nnopt_storage_t), nullptr, &err);
  cl_mem ffn_cf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 (size_t)FFN * T * sizeof(nnopt_storage_t), nullptr, &err);
  if (!residual || !attn_ctx || !x_cf || !ffn_cf) {
    NNOPT_ERROR("te: scratch alloc failed");
    return -5;
  }

  // 2) Encoder layers
  for (int li = 0; li < NL; ++li) {
    // Save residual
    size_t n_th = (size_t)T * H;
    set_arg(k_copy, 0, sizeof(cl_mem), &x);
    set_arg(k_copy, 1, sizeof(cl_mem), &residual);
    int iN = (int)n_th;
    set_arg(k_copy, 2, sizeof(int), &iN);
    clEnqueueNDRangeKernel(queue, k_copy, 1, nullptr, &n_th, nullptr, 0, nullptr, nullptr);

    // Fused QKV projection: single GEMM [T, H] × [3H, H]^T = [T, 3H]
    // Concatenated weights cached per layer on first call.
    static cl_mem g_qkv_w[NL] = {};
    static cl_mem g_qkv_b[NL] = {};
    if (!g_qkv_w[li]) {
      cl_mem wq = weights.get_buffer(lkey(li, "attention.q_proj.weight"));
      cl_mem wk = weights.get_buffer(lkey(li, "attention.k_proj.weight"));
      cl_mem wv = weights.get_buffer(lkey(li, "attention.v_proj.weight"));
      if (!wq || !wk || !wv) { NNOPT_ERROR_FMT("te: missing attn weights layer %d", li); return -6; }
      const size_t wh = (size_t)H * H * sizeof(nnopt_storage_t);
      g_qkv_w[li] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 3 * wh, nullptr, &err);
      clEnqueueCopyBuffer(queue, wq, g_qkv_w[li], 0, 0,     wh, 0, nullptr, nullptr);
      clEnqueueCopyBuffer(queue, wk, g_qkv_w[li], 0, wh,    wh, 0, nullptr, nullptr);
      clEnqueueCopyBuffer(queue, wv, g_qkv_w[li], 0, 2*wh,  wh, 0, nullptr, nullptr);
      cl_mem bq = weights.get_buffer(lkey(li, "attention.q_proj.bias"));
      cl_mem bk = weights.get_buffer(lkey(li, "attention.k_proj.bias"));
      cl_mem bv = weights.get_buffer(lkey(li, "attention.v_proj.bias"));
      const size_t bh = (size_t)H * sizeof(nnopt_storage_t);
      g_qkv_b[li] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 3 * bh, nullptr, &err);
      clEnqueueCopyBuffer(queue, bq, g_qkv_b[li], 0, 0,    bh, 0, nullptr, nullptr);
      clEnqueueCopyBuffer(queue, bk, g_qkv_b[li], 0, bh,   bh, 0, nullptr, nullptr);
      clEnqueueCopyBuffer(queue, bv, g_qkv_b[li], 0, 2*bh, bh, 0, nullptr, nullptr);
    }

    const int H3 = 3 * H;
    cl_mem QKV = linear_gpu(cl_ctx, queue, x, g_qkv_w[li], g_qkv_b[li], T, H, H3, "te.qkv_proj");
    if (!QKV) { NNOPT_ERROR("te: QKV projection failed"); return -7; }

    // Fused attention: reads QKV[T, 3H], writes attn_ctx[T, H]
    cl_mem rel_k = weights.get_buffer(lkey(li, "attention.emb_rel_k"));
    cl_mem rel_v = weights.get_buffer(lkey(li, "attention.emb_rel_v"));
    float scale = 1.0f / std::sqrt((float)D);
    int iD = D, iNh = Nh, iW = W;
    set_arg(k_fused_attn, 0, sizeof(cl_mem), &QKV);
    set_arg(k_fused_attn, 1, sizeof(cl_mem), &rel_k);
    set_arg(k_fused_attn, 2, sizeof(cl_mem), &rel_v);
    set_arg(k_fused_attn, 3, sizeof(cl_mem), &attn_ctx);
    set_arg(k_fused_attn, 4, sizeof(int), &iT);
    set_arg(k_fused_attn, 5, sizeof(int), &iH);
    set_arg(k_fused_attn, 6, sizeof(int), &iD);
    set_arg(k_fused_attn, 7, sizeof(int), &iNh);
    set_arg(k_fused_attn, 8, sizeof(int), &iW);
    set_arg(k_fused_attn, 9, sizeof(float), &scale);
    // Pad gws[0] to multiple of wave size for full utilization.
    // Adreno 620 wave size is 64; lws=(1,64) aligns dim1 to wave boundary.
    // H=192 = 64×3 → clean division. T may not divide evenly but the kernel
    // has a bounds check (if q >= T return).
    size_t gws_attn[2] = {(size_t)T, (size_t)H};
    size_t lws_attn[2] = {1, 64};
    clEnqueueNDRangeKernel(queue, k_fused_attn, 2, nullptr, gws_attn, lws_attn, 0, nullptr,
                           KernelProfiler::event_for("te.fused_attn"));

    // Out projection
    cl_mem wo = weights.get_buffer(lkey(li, "attention.out_proj.weight"));
    cl_mem bo = weights.get_buffer(lkey(li, "attention.out_proj.bias"));
    cl_mem attn_out = linear_gpu(cl_ctx, queue, attn_ctx, wo, bo, T, H, H, "te.out_proj");
    if (!attn_out) { NNOPT_ERROR("te: out_proj failed"); return -8; }

    clReleaseMemObject(QKV);

    // LayerNorm(x + attn_out): x = LN(residual + attn_out)
    // Copy attn_out into x, then fuse LN with residual
    clEnqueueCopyBuffer(queue, attn_out, x, 0, 0, n_th * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
    clReleaseMemObject(attn_out);

    cl_mem ln_g = weights.get_buffer(lkey(li, "layer_norm.weight"));
    cl_mem ln_b = weights.get_buffer(lkey(li, "layer_norm.bias"));
    set_arg(k_ln_res, 0, sizeof(cl_mem), &x);
    set_arg(k_ln_res, 1, sizeof(cl_mem), &residual);
    set_arg(k_ln_res, 2, sizeof(cl_mem), &ln_g);
    set_arg(k_ln_res, 3, sizeof(cl_mem), &ln_b);
    set_arg(k_ln_res, 4, sizeof(int), &iT);
    set_arg(k_ln_res, 5, sizeof(int), &iH);
    set_arg(k_ln_res, 6, sizeof(float), &LN_EPS);
    size_t gws_ln = (size_t)T;
    clEnqueueNDRangeKernel(queue, k_ln_res, 1, nullptr, &gws_ln, nullptr, 0, nullptr,
                           KernelProfiler::event_for("te.ln1"));

    // Save residual for FFN
    set_arg(k_copy, 0, sizeof(cl_mem), &x);
    set_arg(k_copy, 1, sizeof(cl_mem), &residual);
    set_arg(k_copy, 2, sizeof(int), &iN);
    clEnqueueNDRangeKernel(queue, k_copy, 1, nullptr, &n_th, nullptr, 0, nullptr, nullptr);

    // FFN: transpose [T,H]→[H,T], conv1d k=3 (H→FFN), ReLU, conv1d k=3 (FFN→H), transpose back
    int rows_th = T, cols_th = H;
    set_arg(k_transpose, 0, sizeof(cl_mem), &x);
    set_arg(k_transpose, 1, sizeof(cl_mem), &x_cf);
    set_arg(k_transpose, 2, sizeof(int), &rows_th);
    set_arg(k_transpose, 3, sizeof(int), &cols_th);
    size_t gws_tr[2] = {(size_t)T, (size_t)H};
    clEnqueueNDRangeKernel(queue, k_transpose, 2, nullptr, gws_tr, nullptr, 0, nullptr,
                           KernelProfiler::event_for("te.transpose1"));

    // FFN conv1d_1: [H, T] → [FFN, T], kernel=3, pad=1
    cl_mem w1 = weights.get_buffer(lkey(li, "feed_forward.conv_1.weight"));
    cl_mem b1 = weights.get_buffer(lkey(li, "feed_forward.conv_1.bias"));
    cl_mem ffn1 = conv1d_gpu(cl_ctx, queue, x_cf, w1, b1, H, FFN, T, 3, 1, 1, 1, true, "te.ffn1");
    if (!ffn1) { NNOPT_ERROR("te: ffn conv1 failed"); return -9; }

    // ReLU in-place
    int relu_n = FFN * T;
    set_arg(k_relu, 0, sizeof(cl_mem), &ffn1);
    set_arg(k_relu, 1, sizeof(int), &relu_n);
    size_t gws_relu = (size_t)relu_n;
    clEnqueueNDRangeKernel(queue, k_relu, 1, nullptr, &gws_relu, nullptr, 0, nullptr,
                           KernelProfiler::event_for("te.relu"));

    // FFN conv1d_2: [FFN, T] → [H, T], kernel=3, pad=1
    cl_mem w2 = weights.get_buffer(lkey(li, "feed_forward.conv_2.weight"));
    cl_mem b2 = weights.get_buffer(lkey(li, "feed_forward.conv_2.bias"));
    cl_mem ffn2 = conv1d_gpu(cl_ctx, queue, ffn1, w2, b2, FFN, H, T, 3, 1, 1, 1, true, "te.ffn2");
    clReleaseMemObject(ffn1);
    if (!ffn2) { NNOPT_ERROR("te: ffn conv2 failed"); return -10; }

    // Transpose [H, T] → [T, H] back into x
    int rows_ht = H, cols_ht = T;
    set_arg(k_transpose, 0, sizeof(cl_mem), &ffn2);
    set_arg(k_transpose, 1, sizeof(cl_mem), &x);
    set_arg(k_transpose, 2, sizeof(int), &rows_ht);
    set_arg(k_transpose, 3, sizeof(int), &cols_ht);
    size_t gws_tr2[2] = {(size_t)H, (size_t)T};
    clEnqueueNDRangeKernel(queue, k_transpose, 2, nullptr, gws_tr2, nullptr, 0, nullptr,
                           KernelProfiler::event_for("te.transpose2"));
    clReleaseMemObject(ffn2);

    // LayerNorm(x + residual)
    cl_mem fln_g = weights.get_buffer(lkey(li, "final_layer_norm.weight"));
    cl_mem fln_b = weights.get_buffer(lkey(li, "final_layer_norm.bias"));
    set_arg(k_ln_res, 0, sizeof(cl_mem), &x);
    set_arg(k_ln_res, 1, sizeof(cl_mem), &residual);
    set_arg(k_ln_res, 2, sizeof(cl_mem), &fln_g);
    set_arg(k_ln_res, 3, sizeof(cl_mem), &fln_b);
    set_arg(k_ln_res, 4, sizeof(int), &iT);
    set_arg(k_ln_res, 5, sizeof(int), &iH);
    set_arg(k_ln_res, 6, sizeof(float), &LN_EPS);
    clEnqueueNDRangeKernel(queue, k_ln_res, 1, nullptr, &gws_ln, nullptr, 0, nullptr,
                           KernelProfiler::event_for("te.ln2"));
  }

  // 3) Project: linear [T, H] → [T, 2H] (conv1d k=1 = linear)
  cl_mem proj_w = weights.get_buffer("text_encoder.project.weight");
  cl_mem proj_b = weights.get_buffer("text_encoder.project.bias");
  cl_mem stats = linear_gpu(cl_ctx, queue, x, proj_w, proj_b, T, H, 2*H, "te.project");
  if (!stats) { NNOPT_ERROR("te: project failed"); return -11; }

  // Clean up scratch
  clReleaseMemObject(residual);
  clReleaseMemObject(attn_ctx);
  clReleaseMemObject(x_cf);
  clReleaseMemObject(ffn_cf);

  if (out_padding_mask) *out_padding_mask = nullptr;
  *out_hidden_states = x;
  *out_stats = stats;

  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  NNOPT_CHECKPOINT_FMT("text_encoder: T=%d layers=%d emb_scale=%.3f (%.0f ms GPU)",
                       T, NL, (double)emb_scale, ms);
  return 0;
}
