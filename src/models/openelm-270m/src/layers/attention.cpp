// Reference: model_info/modeling_openelm.py:189-356 (OpenELMMultiHeadCausalAttention.forward)
//
// OpenELM has VARIABLE per-layer head counts:
//   QH  = MODEL_CONFIG::NUM_QUERY_HEADS[layer_idx_]
//   KVH = MODEL_CONFIG::NUM_KV_HEADS[layer_idx_]
// Head dim is scalar in config (HEAD_DIM).

#include "layers/attention.h"

#include "debug_utils.h"
#include "layers/layer_norm.h"
#include "model_config.h"
#include "opencl_context.h"
#include "utils.h"
#include "weights.h"

#include <cmath>
#include <string>
#include <vector>

static inline bool set_arg_checked(cl_kernel k, cl_uint idx, size_t sz, const void* val, const char* what) {
  const cl_int err = clSetKernelArg(k, idx, sz, val);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("clSetKernelArg(%u) %s failed: %d", idx, what, (int)err);
    return false;
  }
  return true;
}

Attention::Attention(OpenCLContext& cl_ctx, Weights& weights, int layer_idx)
    : cl_ctx_(cl_ctx), weights_(weights), layer_idx_(layer_idx) {}

Attention::~Attention() {
  if (cos_) clReleaseMemObject(cos_);
  if (sin_) clReleaseMemObject(sin_);
  if (k_cache_) clReleaseMemObject(k_cache_);
  if (v_cache_) clReleaseMemObject(v_cache_);

  release_activation_buffers_();

  if (rope_kernel_) clReleaseKernel(rope_kernel_);
  if (rope_program_) clReleaseProgram(rope_program_);

  if (scores_kernel_) clReleaseKernel(scores_kernel_);
  if (softmax_kernel_) clReleaseKernel(softmax_kernel_);
  if (out_kernel_) clReleaseKernel(out_kernel_);
  if (attn_program_) clReleaseProgram(attn_program_);
}

void Attention::release_activation_buffers_() {
  if (buf_qkv_)     clReleaseMemObject(buf_qkv_);     buf_qkv_     = nullptr; buf_qkv_cap_     = 0;
  if (buf_q_)       clReleaseMemObject(buf_q_);       buf_q_       = nullptr; buf_q_cap_       = 0;
  if (buf_k_)       clReleaseMemObject(buf_k_);       buf_k_       = nullptr; buf_k_cap_       = 0;
  if (buf_v_)       clReleaseMemObject(buf_v_);       buf_v_       = nullptr; buf_v_cap_       = 0;
  if (buf_qn_)      clReleaseMemObject(buf_qn_);      buf_qn_      = nullptr; buf_qn_cap_      = 0;
  if (buf_kn_)      clReleaseMemObject(buf_kn_);      buf_kn_      = nullptr; buf_kn_cap_      = 0;
  if (buf_scores_)  clReleaseMemObject(buf_scores_);  buf_scores_  = nullptr; buf_scores_cap_  = 0;
  if (buf_ctx_out_) clReleaseMemObject(buf_ctx_out_); buf_ctx_out_ = nullptr; buf_ctx_out_cap_ = 0;
  if (buf_proj_)    clReleaseMemObject(buf_proj_);    buf_proj_    = nullptr; buf_proj_cap_    = 0;
}

namespace {
// Grow a buffer to required_bytes, releasing the old one if smaller. Capacity
// stays sticky so successive prefill+decode calls don't churn.
inline bool grow_buffer(cl_context ctx, cl_mem* buf, size_t* cap, size_t required_bytes, const char* label, int layer_idx) {
  if (*cap >= required_bytes && *buf != nullptr) return true;
  if (*buf) clReleaseMemObject(*buf);
  cl_int err = CL_SUCCESS;
  *buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, required_bytes, nullptr, &err);
  if (err != CL_SUCCESS || !*buf) {
    NNOPT_ERROR_FMT("Attention[%d]: grow %s to %zu B failed (%d)", layer_idx, label, required_bytes, (int)err);
    *buf = nullptr;
    *cap = 0;
    return false;
  }
  *cap = required_bytes;
  return true;
}
}  // namespace

bool Attention::ensure_activation_buffers_(int seq_q, int seq_k) {
  const int H   = MODEL_CONFIG::HIDDEN_SIZE;
  const int D   = MODEL_CONFIG::HEAD_DIM;
  const int QH  = MODEL_CONFIG::NUM_QUERY_HEADS[layer_idx_];
  const int KVH = MODEL_CONFIG::NUM_KV_HEADS[layer_idx_];

  const int Q_DIM   = QH * D;
  const int KV_DIM  = KVH * D;
  const int QKV_DIM = Q_DIM + 2 * KV_DIM;

  cl_context ctx = cl_ctx_.context();
  const size_t fp = sizeof(nnopt_storage_t);

  if (!grow_buffer(ctx, &buf_qkv_,     &buf_qkv_cap_,     (size_t)seq_q * (size_t)QKV_DIM * fp,           "qkv",     layer_idx_)) return false;
  if (!grow_buffer(ctx, &buf_q_,       &buf_q_cap_,       (size_t)seq_q * (size_t)Q_DIM   * fp,           "q",       layer_idx_)) return false;
  if (!grow_buffer(ctx, &buf_k_,       &buf_k_cap_,       (size_t)seq_q * (size_t)KV_DIM  * fp,           "k",       layer_idx_)) return false;
  if (!grow_buffer(ctx, &buf_v_,       &buf_v_cap_,       (size_t)seq_q * (size_t)KV_DIM  * fp,           "v",       layer_idx_)) return false;
  if (MODEL_CONFIG::NORMALIZE_QK_PROJECTIONS) {
    if (!grow_buffer(ctx, &buf_qn_,    &buf_qn_cap_,      (size_t)seq_q * (size_t)Q_DIM   * fp,           "qn",      layer_idx_)) return false;
    if (!grow_buffer(ctx, &buf_kn_,    &buf_kn_cap_,      (size_t)seq_q * (size_t)KV_DIM  * fp,           "kn",      layer_idx_)) return false;
  }
  if (!grow_buffer(ctx, &buf_scores_,  &buf_scores_cap_,  (size_t)QH * (size_t)seq_q * (size_t)seq_k * fp, "scores",  layer_idx_)) return false;
  if (!grow_buffer(ctx, &buf_ctx_out_, &buf_ctx_out_cap_, (size_t)seq_q * (size_t)Q_DIM   * fp,           "ctx_out", layer_idx_)) return false;
  if (!grow_buffer(ctx, &buf_proj_,    &buf_proj_cap_,    (size_t)seq_q * (size_t)H       * fp,           "proj",    layer_idx_)) return false;
  return true;
}

bool Attention::set_weights() {
  const std::string prefix = std::string("transformer.layers.") + std::to_string(layer_idx_) + ".attn.";

  // OpenELM uses fused qkv_proj and out_proj (both bias-free)
  wq_ = weights_.get_buffer(prefix + "qkv_proj.weight");
  wo_ = weights_.get_buffer(prefix + "out_proj.weight");

  if (!wq_ || !wo_) {
    NNOPT_ERROR_FMT("Attention[%d]: missing qkv_proj/out_proj weight(s): wq=%p wo=%p", layer_idx_, (void*)wq_, (void*)wo_);
    return false;
  }
  return true;
}

bool Attention::ensure_kv_cache() {
  if (k_cache_ && v_cache_) return true;

  const int KVH = MODEL_CONFIG::NUM_KV_HEADS[layer_idx_];
  const int KV_DIM = KVH * MODEL_CONFIG::HEAD_DIM;
  const size_t bytes = (size_t)MODEL_CONFIG::MAX_CONTEXT_LENGTH * (size_t)KV_DIM * sizeof(nnopt_storage_t);

  cl_int err = CL_SUCCESS;
  k_cache_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
  if (err != CL_SUCCESS || !k_cache_) {
    NNOPT_ERROR_FMT("alloc k_cache_ layer %d failed: %d", layer_idx_, (int)err);
    return false;
  }
  v_cache_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
  if (err != CL_SUCCESS || !v_cache_) {
    NNOPT_ERROR_FMT("alloc v_cache_ layer %d failed: %d", layer_idx_, (int)err);
    clReleaseMemObject(k_cache_);
    k_cache_ = nullptr;
    return false;
  }

  return true;
}

bool Attention::ensure_rope_tables(int seq_len) {
  if (seq_len <= 0) return false;
  if (cos_ && sin_ && rope_seq_len_ >= seq_len) return true;

  if (cos_) {
    clReleaseMemObject(cos_);
    cos_ = nullptr;
  }
  if (sin_) {
    clReleaseMemObject(sin_);
    sin_ = nullptr;
  }
  rope_seq_len_ = 0;

  const int D = MODEL_CONFIG::HEAD_DIM;
  const int half_dim = D / 2;

  // OpenELM uses rope_freq_constant as the base frequency constant.
  const float theta = (float)MODEL_CONFIG::ROPE_FREQ_CONSTANT;

  std::vector<float> cos_f((size_t)seq_len * (size_t)D);
  std::vector<float> sin_f((size_t)seq_len * (size_t)D);

  for (int t = 0; t < seq_len; t++) {
    for (int i = 0; i < half_dim; i++) {
      const float exponent = (2.0f * (float)i) / (float)D;
      const float inv_freq = std::pow(theta, -exponent);
      const float angle = (float)t * inv_freq;
      const float c = std::cos(angle);
      const float s = std::sin(angle);

      cos_f[(size_t)t * (size_t)D + (size_t)i] = c;
      sin_f[(size_t)t * (size_t)D + (size_t)i] = s;
      // HF-style RoPE duplicates cos/sin across both halves.
      cos_f[(size_t)t * (size_t)D + (size_t)i + (size_t)half_dim] = c;
      sin_f[(size_t)t * (size_t)D + (size_t)i + (size_t)half_dim] = s;
    }
  }

  std::vector<nnopt_storage_t> cos_h(cos_f.size());
  std::vector<nnopt_storage_t> sin_h(sin_f.size());
  for (size_t i = 0; i < cos_f.size(); i++) {
#ifdef NNOPT_USE_FP16
    cos_h[i] = nnopt_f32_to_f16(cos_f[i]);
    sin_h[i] = nnopt_f32_to_f16(sin_f[i]);
#else
    cos_h[i] = cos_f[i];
    sin_h[i] = sin_f[i];
#endif
  }

  cl_int err = CL_SUCCESS;
  cos_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, cos_h.size() * sizeof(nnopt_storage_t),
                        cos_h.data(), &err);
  if (err != CL_SUCCESS || !cos_) {
    NNOPT_ERROR_FMT("alloc cos_ failed: %d", (int)err);
    return false;
  }
  sin_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sin_h.size() * sizeof(nnopt_storage_t),
                        sin_h.data(), &err);
  if (err != CL_SUCCESS || !sin_) {
    NNOPT_ERROR_FMT("alloc sin_ failed: %d", (int)err);
    clReleaseMemObject(cos_);
    cos_ = nullptr;
    return false;
  }

  rope_seq_len_ = seq_len;
  return true;
}

bool Attention::initialize() {
  rope_program_ = cl_ctx_.build_program_from_file(
      "kernels/rope.cl",
#ifdef NNOPT_USE_FP16
      "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
      ""
#endif
  );
  if (!rope_program_) {
    NNOPT_ERROR("Failed to build kernels/rope.cl");
    return false;
  }

  attn_program_ = cl_ctx_.build_program_from_file(
      "kernels/attention.cl",
#ifdef NNOPT_USE_FP16
      "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
      ""
#endif
  );
  if (!attn_program_) {
    NNOPT_ERROR("Failed to build kernels/attention.cl");
    return false;
  }

  cl_int err = CL_SUCCESS;
  rope_kernel_ = clCreateKernel(rope_program_, "rope_apply_qk", &err);
  if (err != CL_SUCCESS || !rope_kernel_) {
    NNOPT_ERROR_FMT("clCreateKernel rope_apply_qk failed: %d", (int)err);
    return false;
  }

  scores_kernel_ = clCreateKernel(attn_program_, "gqa_attn_scores", &err);
  if (err != CL_SUCCESS || !scores_kernel_) {
    NNOPT_ERROR_FMT("clCreateKernel gqa_attn_scores failed: %d", (int)err);
    return false;
  }
  softmax_kernel_ = clCreateKernel(attn_program_, "gqa_softmax", &err);
  if (err != CL_SUCCESS || !softmax_kernel_) {
    NNOPT_ERROR_FMT("clCreateKernel gqa_softmax failed: %d", (int)err);
    return false;
  }
  out_kernel_ = clCreateKernel(attn_program_, "gqa_attn_out", &err);
  if (err != CL_SUCCESS || !out_kernel_) {
    NNOPT_ERROR_FMT("clCreateKernel gqa_attn_out failed: %d", (int)err);
    return false;
  }

  if (!set_weights()) return false;
  if (!ensure_kv_cache()) return false;

  return true;
}

cl_mem Attention::forward(cl_command_queue queue, cl_mem input, int seq_q, int start_pos) {
  return forward(queue, input, /*cos=*/nullptr, /*sin=*/nullptr, seq_q, start_pos);
}

cl_mem Attention::forward(cl_command_queue queue, cl_mem input, cl_mem cos, cl_mem sin, int seq_q, int start_pos) {
  const int H = MODEL_CONFIG::HIDDEN_SIZE;
  const int D = MODEL_CONFIG::HEAD_DIM;
  const int QH = MODEL_CONFIG::NUM_QUERY_HEADS[layer_idx_];
  const int KVH = MODEL_CONFIG::NUM_KV_HEADS[layer_idx_];

  const int Q_DIM = QH * D;
  const int KV_DIM = KVH * D;
  const int QKV_DIM = Q_DIM + 2 * KV_DIM;

  const int seq_k = start_pos + seq_q;
  if (seq_k > MODEL_CONFIG::MAX_CONTEXT_LENGTH) {
    NNOPT_ERROR_FMT("seq_k %d exceeds MAX_CONTEXT_LENGTH %d at layer %d", seq_k, MODEL_CONFIG::MAX_CONTEXT_LENGTH, layer_idx_);
    return nullptr;
  }

  if (!ensure_kv_cache()) return nullptr;
  if (!ensure_activation_buffers_(seq_q, seq_k)) return nullptr;

  cl_int err = CL_SUCCESS;

  // qkv = qkv_proj(hidden_states)
  if (!pytorch_linear(queue, seq_q, QKV_DIM, H, input, wq_, buf_qkv_)) {
    NNOPT_ERROR_FMT("qkv_proj gemm failed at layer %d", layer_idx_);
    return nullptr;
  }

  // Split qkv -> q,k,v (contiguous) via per-row copies into persistent buffers.
  const size_t row_bytes = (size_t)QKV_DIM * sizeof(nnopt_storage_t);
  const size_t q_bytes = (size_t)Q_DIM * sizeof(nnopt_storage_t);
  const size_t kv_bytes = (size_t)KV_DIM * sizeof(nnopt_storage_t);

  for (int r = 0; r < seq_q; r++) {
    const size_t src_off = (size_t)r * row_bytes;
    const size_t q_off = (size_t)r * q_bytes;
    const size_t k_off = (size_t)r * kv_bytes;
    const size_t v_off = (size_t)r * kv_bytes;

    err = clEnqueueCopyBuffer(queue, buf_qkv_, buf_q_, src_off, q_off, q_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("split q copy failed: %d", (int)err); return nullptr; }
    err = clEnqueueCopyBuffer(queue, buf_qkv_, buf_k_, src_off + q_bytes, k_off, kv_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("split k copy failed: %d", (int)err); return nullptr; }
    err = clEnqueueCopyBuffer(queue, buf_qkv_, buf_v_, src_off + q_bytes + kv_bytes, v_off, kv_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("split v copy failed: %d", (int)err); return nullptr; }
  }

  // q_norm/k_norm: rmsnorm_forward returns a NEW buffer on each call (it's a
  // free helper, not aware of our persistent buffers). We can't cleanly own
  // its outputs without restructuring rmsnorm_forward, so leave this as a per-
  // call alloc — it's the only remaining alloc per attention forward, and
  // post-norm q/k are then assigned back to the persistent buf_q_/buf_k_ via
  // copies before the rope path uses them.
  cl_mem q_for_rope = buf_q_;
  cl_mem k_for_rope = buf_k_;
  if (MODEL_CONFIG::NORMALIZE_QK_PROJECTIONS) {
    const std::string p = std::string("transformer.layers.") + std::to_string(layer_idx_) + ".attn.";
    cl_mem q_w = weights_.get_buffer(p + "q_norm.weight");
    cl_mem k_w = weights_.get_buffer(p + "k_norm.weight");
    if (!q_w || !k_w) {
      NNOPT_ERROR_FMT("Attention[%d]: normalize_qk_projections but missing q_norm/k_norm weights", layer_idx_);
      return nullptr;
    }
    cl_mem qn = rmsnorm_forward(cl_ctx_, queue, buf_q_, q_w, /*rows=*/seq_q * QH, /*cols=*/D, MODEL_CONFIG::NORM_EPS);
    if (!qn) { NNOPT_ERROR_FMT("Attention[%d]: q rmsnorm_forward failed", layer_idx_); return nullptr; }
    cl_mem kn = rmsnorm_forward(cl_ctx_, queue, buf_k_, k_w, /*rows=*/seq_q * KVH, /*cols=*/D, MODEL_CONFIG::NORM_EPS);
    if (!kn) { NNOPT_ERROR_FMT("Attention[%d]: k rmsnorm_forward failed", layer_idx_); clReleaseMemObject(qn); return nullptr; }
    // Copy into the persistent _qn/_kn buffers, release the rmsnorm allocations
    err = clEnqueueCopyBuffer(queue, qn, buf_qn_, 0, 0, (size_t)seq_q * Q_DIM * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
    if (err == CL_SUCCESS)
      err = clEnqueueCopyBuffer(queue, kn, buf_kn_, 0, 0, (size_t)seq_q * KV_DIM * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
    NNOPT_DEBUG_SYNC(queue);
    clReleaseMemObject(qn);
    clReleaseMemObject(kn);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Attention[%d]: copy qn/kn into persistent bufs failed (%d)", layer_idx_, (int)err); return nullptr; }
    q_for_rope = buf_qn_;
    k_for_rope = buf_kn_;
    if (layer_idx_ == 0) {
      NNOPT_LAYER_CHECK("block0_sub_attn_q_norm_out", queue, q_for_rope, (size_t)seq_q * (size_t)Q_DIM);
      NNOPT_LAYER_CHECK("block0_sub_attn_k_norm_out", queue, k_for_rope, (size_t)seq_q * (size_t)KV_DIM);
    }
  }

  // RoPE tables
  cl_mem cos_use = cos;
  cl_mem sin_use = sin;
  if (!cos_use || !sin_use) {
    if (!ensure_rope_tables(seq_k)) {
      NNOPT_ERROR_FMT("ensure_rope_tables failed at layer %d", layer_idx_);
      return nullptr;
    }
    cos_use = cos_;
    sin_use = sin_;
  }

  // Apply RoPE to q,k (in-place on the rope-input buffers).
  if (!set_arg_checked(rope_kernel_, 0, sizeof(cl_mem), &q_for_rope, "q") ||
      !set_arg_checked(rope_kernel_, 1, sizeof(cl_mem), &k_for_rope, "k") ||
      !set_arg_checked(rope_kernel_, 2, sizeof(cl_mem), &cos_use, "cos") ||
      !set_arg_checked(rope_kernel_, 3, sizeof(cl_mem), &sin_use, "sin") ||
      !set_arg_checked(rope_kernel_, 4, sizeof(int), &seq_q, "seq_q") ||
      !set_arg_checked(rope_kernel_, 5, sizeof(int), &QH, "q_heads") ||
      !set_arg_checked(rope_kernel_, 6, sizeof(int), &KVH, "kv_heads") ||
      !set_arg_checked(rope_kernel_, 7, sizeof(int), &D, "head_dim") ||
      !set_arg_checked(rope_kernel_, 8, sizeof(int), &start_pos, "start_pos")) {
    return nullptr;
  }

  {
    const int half_dim = D / 2;
    const size_t gws = (size_t)(seq_q * QH * half_dim + seq_q * KVH * half_dim);
    err = clEnqueueNDRangeKernel(queue, rope_kernel_, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("rope dispatch failed: %d", (int)err); return nullptr; }
    NNOPT_DEBUG_SYNC(queue);
  }

  if (layer_idx_ == 0) {
    NNOPT_LAYER_CHECK("block0_sub_attn_rope_q_out", queue, q_for_rope, (size_t)seq_q * (size_t)Q_DIM);
    NNOPT_LAYER_CHECK("block0_sub_attn_rope_k_out", queue, k_for_rope, (size_t)seq_q * (size_t)KV_DIM);
  }

  // KV cache write at start_pos (post-RoPE k, raw v).
  {
    const size_t dst_off_bytes = (size_t)start_pos * (size_t)KV_DIM * sizeof(nnopt_storage_t);
    const size_t copy_bytes = (size_t)seq_q * (size_t)KV_DIM * sizeof(nnopt_storage_t);
    err = clEnqueueCopyBuffer(queue, k_for_rope, k_cache_, 0, dst_off_bytes, copy_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("copy k->cache failed: %d", (int)err); return nullptr; }
    err = clEnqueueCopyBuffer(queue, buf_v_, v_cache_, 0, dst_off_bytes, copy_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("copy v->cache failed: %d", (int)err); return nullptr; }
  }

  // scores = scaled dot-product + causal mask is inside kernel
  const float scale = 1.0f / std::sqrt((float)D);
  if (!set_arg_checked(scores_kernel_, 0, sizeof(cl_mem), &q_for_rope, "q") ||
      !set_arg_checked(scores_kernel_, 1, sizeof(cl_mem), &k_cache_, "k_cache") ||
      !set_arg_checked(scores_kernel_, 2, sizeof(cl_mem), &buf_scores_, "scores") ||
      !set_arg_checked(scores_kernel_, 3, sizeof(int), &seq_q, "seq_q") ||
      !set_arg_checked(scores_kernel_, 4, sizeof(int), &seq_k, "seq_k") ||
      !set_arg_checked(scores_kernel_, 5, sizeof(int), &QH, "num_q_heads") ||
      !set_arg_checked(scores_kernel_, 6, sizeof(int), &KVH, "num_kv_heads") ||
      !set_arg_checked(scores_kernel_, 7, sizeof(int), &D, "head_dim") ||
      !set_arg_checked(scores_kernel_, 8, sizeof(float), &scale, "scale")) {
    return nullptr;
  }
  {
    size_t gws_s[3] = {(size_t)QH, (size_t)seq_q, (size_t)seq_k};
    err = clEnqueueNDRangeKernel(queue, scores_kernel_, 3, nullptr, gws_s, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("scores dispatch failed: %d", (int)err); return nullptr; }
    NNOPT_DEBUG_SYNC(queue);
  }

  // softmax(scores) along seq_k
  {
    const int total_rows = QH * seq_q;
    if (!set_arg_checked(softmax_kernel_, 0, sizeof(cl_mem), &buf_scores_, "scores") ||
        !set_arg_checked(softmax_kernel_, 1, sizeof(int), &seq_q, "seq_q") ||
        !set_arg_checked(softmax_kernel_, 2, sizeof(int), &seq_k, "seq_k") ||
        !set_arg_checked(softmax_kernel_, 3, sizeof(int), &total_rows, "total_rows")) {
      return nullptr;
    }
    size_t gws = (size_t)total_rows;
    err = clEnqueueNDRangeKernel(queue, softmax_kernel_, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("softmax dispatch failed: %d", (int)err); return nullptr; }
    NNOPT_DEBUG_SYNC(queue);
  }

  if (layer_idx_ == 0) {
    NNOPT_LAYER_CHECK("block0_sub_attn_softmax_out", queue, buf_scores_, (size_t)QH * (size_t)seq_q * (size_t)seq_k);
  }

  // out = scores @ v_cache
  if (!set_arg_checked(out_kernel_, 0, sizeof(cl_mem), &buf_scores_, "scores") ||
      !set_arg_checked(out_kernel_, 1, sizeof(cl_mem), &v_cache_, "v_cache") ||
      !set_arg_checked(out_kernel_, 2, sizeof(cl_mem), &buf_ctx_out_, "out") ||
      !set_arg_checked(out_kernel_, 3, sizeof(int), &seq_q, "seq_q") ||
      !set_arg_checked(out_kernel_, 4, sizeof(int), &seq_k, "seq_k") ||
      !set_arg_checked(out_kernel_, 5, sizeof(int), &QH, "num_q_heads") ||
      !set_arg_checked(out_kernel_, 6, sizeof(int), &KVH, "num_kv_heads") ||
      !set_arg_checked(out_kernel_, 7, sizeof(int), &D, "head_dim")) {
    return nullptr;
  }
  {
    size_t gws_o[3] = {(size_t)QH, (size_t)seq_q, (size_t)(D / 4)};
    err = clEnqueueNDRangeKernel(queue, out_kernel_, 3, nullptr, gws_o, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("attn_out dispatch failed: %d", (int)err); return nullptr; }
    NNOPT_DEBUG_SYNC(queue);
  }

  if (layer_idx_ == 0) {
    NNOPT_LAYER_CHECK("block0_sub_attn_out_proj_in", queue, buf_ctx_out_, (size_t)seq_q * (size_t)Q_DIM);
  }

  // out_proj(ctx_out) — writes directly into the borrowed-handle output buffer.
  if (!pytorch_linear(queue, seq_q, H, Q_DIM, buf_ctx_out_, wo_, buf_proj_)) {
    NNOPT_ERROR_FMT("out_proj gemm failed at layer %d", layer_idx_);
    return nullptr;
  }

  if (layer_idx_ == 0) {
    NNOPT_LAYER_CHECK("block0_sub_attn_out_proj_out", queue, buf_proj_, (size_t)seq_q * (size_t)H);
  }

  // BORROWED handle — caller must NOT release. Owned by *this until next call.
  return buf_proj_;
}
