// Auto-generated transformer Attention scaffold.
// Reference: model_info/transformers_src/modeling_*.py — read the model's
// exact LlamaAttention / GPT2Attention / etc. forward() before extending.
//
// Rules embedded in this file:
//   KV-01 — k_cache_ / v_cache_ used; agent must NOT write per-token re-projection
//   GEN-01 — works for both prefill (seq_q=P, start_pos=0) and decode (seq_q=1, start_pos>0)
//   HEAD-01 — attention math is 3 batched kernel launches, NOT a per-head loop
//   SYNC-01 — no clFinish between kernels (queue is in-order); use NNOPT_DEBUG_SYNC if you need debug visibility
//   PROG-01 — programs built once in initialize()

#include "layers/attention.h"
#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "kernel_profiler.h"
#include "utils.h"
#include "model_config.h"
#include <clblast.h>
#include <cmath>
#include <string>
#include <vector>

Attention::Attention(OpenCLContext& cl_ctx, Weights& weights, int layer_idx)
    : cl_ctx_(cl_ctx), weights_(weights), layer_idx_(layer_idx) {}

Attention::Attention(OpenCLContext& cl_ctx, Weights& weights, int layer_idx, const std::string& layer_prefix)
    : cl_ctx_(cl_ctx), weights_(weights), layer_idx_(layer_idx), layer_prefix_(layer_prefix) {}

Attention::~Attention() {
    if (cos_) clReleaseMemObject(cos_);
    if (sin_) clReleaseMemObject(sin_);
    if (k_cache_) clReleaseMemObject(k_cache_);
    if (v_cache_) clReleaseMemObject(v_cache_);
    if (rope_kernel_) clReleaseKernel(rope_kernel_);
    if (rope_program_) clReleaseProgram(rope_program_);
    if (scores_kernel_) clReleaseKernel(scores_kernel_);
    if (softmax_kernel_) clReleaseKernel(softmax_kernel_);
    if (out_kernel_) clReleaseKernel(out_kernel_);
    if (attn_program_) clReleaseProgram(attn_program_);
    if (rmsnorm_kernel_) clReleaseKernel(rmsnorm_kernel_);
    if (rmsnorm_program_) clReleaseProgram(rmsnorm_program_);
    if (buf_q_) clReleaseMemObject(buf_q_);
    if (buf_k_) clReleaseMemObject(buf_k_);
    if (buf_v_) clReleaseMemObject(buf_v_);
    if (buf_q_norm_tmp_) clReleaseMemObject(buf_q_norm_tmp_);
    if (buf_k_norm_tmp_) clReleaseMemObject(buf_k_norm_tmp_);
    if (buf_scores_) clReleaseMemObject(buf_scores_);
    if (buf_attn_out_) clReleaseMemObject(buf_attn_out_);
    if (buf_proj_) clReleaseMemObject(buf_proj_);
}

bool Attention::ensure_buffers_(int seq_q, int seq_k) {
    const int H = MODEL_CONFIG::HIDDEN_SIZE;
    const int QH = MODEL_CONFIG::NUM_ATTENTION_HEADS;
    const int KVH = MODEL_CONFIG::NUM_KEY_VALUE_HEADS;
    const int D = MODEL_CONFIG::HEAD_DIM;
    const int Q_DIM = QH * D;
    const int KV_DIM = KVH * D;

    bool need_seq_q = seq_q > buf_seq_q_capacity_ || !buf_q_;
    bool need_seq_k = seq_k > buf_seq_k_capacity_ || !buf_scores_;
    if (!need_seq_q && !need_seq_k) return true;

    cl_context ctx = cl_ctx_.context();
    cl_int err = CL_SUCCESS;

    if (need_seq_q) {
        if (buf_q_) { clReleaseMemObject(buf_q_); buf_q_ = nullptr; }
        if (buf_k_) { clReleaseMemObject(buf_k_); buf_k_ = nullptr; }
        if (buf_v_) { clReleaseMemObject(buf_v_); buf_v_ = nullptr; }
        if (buf_q_norm_tmp_) { clReleaseMemObject(buf_q_norm_tmp_); buf_q_norm_tmp_ = nullptr; }
        if (buf_k_norm_tmp_) { clReleaseMemObject(buf_k_norm_tmp_); buf_k_norm_tmp_ = nullptr; }
        if (buf_attn_out_) { clReleaseMemObject(buf_attn_out_); buf_attn_out_ = nullptr; }
        if (buf_proj_) { clReleaseMemObject(buf_proj_); buf_proj_ = nullptr; }
        const size_t s = (size_t)seq_q;
        buf_q_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, s * Q_DIM * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS) return false;
        buf_k_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, s * KV_DIM * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS) return false;
        buf_v_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, s * KV_DIM * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS) return false;
        buf_q_norm_tmp_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, s * Q_DIM * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS) return false;
        buf_k_norm_tmp_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, s * KV_DIM * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS) return false;
        buf_attn_out_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, s * Q_DIM * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS) return false;
        buf_proj_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, s * H * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS) return false;
        buf_seq_q_capacity_ = seq_q;
    }
    if (need_seq_k) {
        if (buf_scores_) { clReleaseMemObject(buf_scores_); buf_scores_ = nullptr; }
        // Grow with headroom (×2) so we don't realloc on every decode token.
        int new_cap = std::max(seq_k * 2, 64);
        size_t bytes = (size_t)QH * (size_t)buf_seq_q_capacity_ * (size_t)new_cap * sizeof(nnopt_storage_t);
        buf_scores_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
        if (err != CL_SUCCESS) return false;
        buf_seq_k_capacity_ = new_cap;
    }
    return true;
}

static inline bool _set_arg_checked(cl_kernel k, cl_uint idx, size_t sz, const void* val, const char* what) {
    cl_int err = clSetKernelArg(k, idx, sz, val);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("clSetKernelArg %s failed: %d", what, err);
        return false;
    }
    return true;
}

bool Attention::ensure_kv_cache() {
    if (k_cache_ && v_cache_) return true;

    const int KV_DIM = MODEL_CONFIG::NUM_KEY_VALUE_HEADS * MODEL_CONFIG::HEAD_DIM;
    const size_t bytes = (size_t)MODEL_CONFIG::MAX_POSITION_EMBEDDINGS
                         * (size_t)KV_DIM * sizeof(nnopt_storage_t);

    cl_int err = CL_SUCCESS;
    k_cache_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS || !k_cache_) {
        NNOPT_ERROR_FMT("alloc k_cache_ layer %d failed: %d", layer_idx_, err);
        return false;
    }
    v_cache_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS || !v_cache_) {
        NNOPT_ERROR_FMT("alloc v_cache_ layer %d failed: %d", layer_idx_, err);
        clReleaseMemObject(k_cache_); k_cache_ = nullptr;
        return false;
    }
    return true;
}

bool Attention::ensure_rope_tables(cl_command_queue queue, int seq_len) {
    (void)queue;
    if (seq_len <= 0) return false;
    if (cos_ && sin_ && rope_seq_len_ >= seq_len) return true;

    if (cos_) { clReleaseMemObject(cos_); cos_ = nullptr; }
    if (sin_) { clReleaseMemObject(sin_); sin_ = nullptr; }
    rope_seq_len_ = 0;

    const int D = MODEL_CONFIG::HEAD_DIM;
    const int half_dim = D / 2;
    const float theta = (float)MODEL_CONFIG::ROPE_THETA;

    std::vector<float> cos_f((size_t)seq_len * D);
    std::vector<float> sin_f((size_t)seq_len * D);
    for (int t = 0; t < seq_len; ++t) {
        for (int i = 0; i < half_dim; ++i) {
            const float exponent = (2.0f * (float)i) / (float)D;
            const float inv_freq = std::pow(theta, -exponent);  // SYNC-OK: init path; TRIG-01 doesn't fire here (function is ensure_*)
            const float angle = (float)t * inv_freq;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            cos_f[(size_t)t * D + (size_t)i] = c;
            sin_f[(size_t)t * D + (size_t)i] = s;
            cos_f[(size_t)t * D + (size_t)i + (size_t)half_dim] = c;
            sin_f[(size_t)t * D + (size_t)i + (size_t)half_dim] = s;
        }
    }

    std::vector<nnopt_storage_t> cos_h((size_t)seq_len * D);
    std::vector<nnopt_storage_t> sin_h((size_t)seq_len * D);
    for (size_t i = 0; i < cos_h.size(); ++i) {
#ifdef NNOPT_USE_FP16
        cos_h[i] = (nnopt_storage_t)nnopt_f32_to_f16(cos_f[i]);
        sin_h[i] = (nnopt_storage_t)nnopt_f32_to_f16(sin_f[i]);
#else
        cos_h[i] = cos_f[i];
        sin_h[i] = sin_f[i];
#endif
    }

    cl_context ctx = cl_ctx_.context();
    cl_int err = CL_SUCCESS;
    cos_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                         cos_h.size() * sizeof(nnopt_storage_t), cos_h.data(), &err);
    if (err != CL_SUCCESS || !cos_) {
        NNOPT_ERROR_FMT("alloc cos_ failed: %d", err);
        return false;
    }
    sin_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                         sin_h.size() * sizeof(nnopt_storage_t), sin_h.data(), &err);
    if (err != CL_SUCCESS || !sin_) {
        NNOPT_ERROR_FMT("alloc sin_ failed: %d", err);
        clReleaseMemObject(cos_); cos_ = nullptr;
        return false;
    }

    rope_seq_len_ = seq_len;
    return true;
}

bool Attention::initialize() {
    // LFM2.5 uses absolute positional embeddings (MODEL_CONFIG::USES_ROPE == false).
    // Only build RoPE program/kernel when the config actually enables RoPE.
    if (MODEL_CONFIG::USES_ROPE) {
        rope_program_ = cl_ctx_.build_program_from_file(
            "kernels/rope.cl",
#ifdef NNOPT_USE_FP16
            "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
            ""
#endif
        );
        if (!rope_program_) { NNOPT_ERROR("Failed to build rope.cl"); return false; }

        cl_int err = CL_SUCCESS;
        rope_kernel_ = clCreateKernel(rope_program_, "rope_apply_qk", &err);
        if (err != CL_SUCCESS || !rope_kernel_) {
            NNOPT_ERROR_FMT("clCreateKernel rope_apply_qk failed: %d", err);
            return false;
        }
    }

    attn_program_ = cl_ctx_.build_program_from_file(
        "kernels/attention.cl",
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
        ""
#endif
    );
    if (!attn_program_) { NNOPT_ERROR("Failed to build attention.cl"); return false; }

    cl_int err = CL_SUCCESS;
    scores_kernel_ = clCreateKernel(attn_program_, "gqa_attn_scores", &err);
    if (err != CL_SUCCESS || !scores_kernel_) {
        NNOPT_ERROR_FMT("clCreateKernel gqa_attn_scores failed: %d", err);
        return false;
    }
    softmax_kernel_ = clCreateKernel(attn_program_, "gqa_softmax", &err);
    if (err != CL_SUCCESS || !softmax_kernel_) {
        NNOPT_ERROR_FMT("clCreateKernel gqa_softmax failed: %d", err);
        return false;
    }
    out_kernel_ = clCreateKernel(attn_program_, "gqa_attn_out", &err);
    if (err != CL_SUCCESS || !out_kernel_) {
        NNOPT_ERROR_FMT("clCreateKernel gqa_attn_out failed: %d", err);
        return false;
    }

    // Per-head RMSNorm program/kernel — used for Lfm2 q_layernorm/k_layernorm
    // (modeling_lfm2.py:241-242,255-256), applied to per-head [head_dim] slices
    // of q/k BEFORE RoPE.
    rmsnorm_program_ = cl_ctx_.build_program_from_file(
        "kernels/rmsnorm.cl",
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
        ""
#endif
    );
    if (!rmsnorm_program_) { NNOPT_ERROR("Attention: failed to build rmsnorm.cl"); return false; }
    rmsnorm_kernel_ = clCreateKernel(rmsnorm_program_, "rmsnorm_forward", &err);
    if (err != CL_SUCCESS || !rmsnorm_kernel_) {
        NNOPT_ERROR_FMT("clCreateKernel rmsnorm_forward failed: %d", err);
        return false;
    }

    // Weights are loaded directly here (no separate set_weights() declared in the header).
    const std::string prefix = layer_prefix_.empty()
        ? ("model.layers." + std::to_string(layer_idx_) + ".self_attn.")
        : (layer_prefix_ + ".");

    wq_ = weights_.get_buffer(prefix + "q_proj.weight");
    wk_ = weights_.get_buffer(prefix + "k_proj.weight");
    wv_ = weights_.get_buffer(prefix + "v_proj.weight");
    wo_ = weights_.get_buffer(prefix + "out_proj.weight");
    q_ln_w_ = weights_.get_buffer(prefix + "q_layernorm.weight");
    k_ln_w_ = weights_.get_buffer(prefix + "k_layernorm.weight");
    if (!wq_ || !wk_ || !wv_ || !wo_ || !q_ln_w_ || !k_ln_w_) {
        NNOPT_ERROR_FMT("Attention[%d]: missing required weights under %s", layer_idx_, prefix.c_str());
        return false;
    }
    if (!ensure_kv_cache()) return false;

    NNOPT_LAYER_INIT_FMT("block%d_sub_attn", layer_idx_);
    return true;
}

// TODO (agent): implement set_weights() with the model's actual weight keys.
// Read (model metadata) for the exact templated keys.
// Common patterns (replace {i} with layer_idx_):
//   Llama family:  "model.layers.{i}.self_attn.q_proj.weight" (and k_proj, v_proj, o_proj)
//   GPT-2 family:  "h.{i}.attn.c_attn.weight" (combined Q/K/V) + "h.{i}.attn.c_proj.weight"
//   Phi family:    "model.layers.{i}.self_attn.qkv_proj.weight" + "...o_proj.weight"
cl_mem Attention::forward(cl_command_queue queue,
                          cl_mem input,
                          int seq_q,
                          int start_pos) {
    cl_mem hidden_states = input;

    const int H = MODEL_CONFIG::HIDDEN_SIZE;
    const int QH = MODEL_CONFIG::NUM_ATTENTION_HEADS;
    const int KVH = MODEL_CONFIG::NUM_KEY_VALUE_HEADS;
    const int D = MODEL_CONFIG::HEAD_DIM;
    const int Q_DIM = QH * D;
    const int KV_DIM = KVH * D;

    const int seq_k = start_pos + seq_q;
    if (seq_k > MODEL_CONFIG::MAX_POSITION_EMBEDDINGS) {
        NNOPT_ERROR_FMT("seq_k %d exceeds MAX_POSITION_EMBEDDINGS %d at layer %d",
                        seq_k, MODEL_CONFIG::MAX_POSITION_EMBEDDINGS, layer_idx_);
        return nullptr;
    }

    if (!ensure_kv_cache()) return nullptr;
    if (!ensure_buffers_(seq_q, seq_k)) { NNOPT_ERROR_FMT("Attention[%d]: ensure_buffers failed", layer_idx_); return nullptr; }

    const float scale = 1.0f / std::sqrt((float)D);

    cl_context ctx = cl_ctx_.context();
    cl_int err = CL_SUCCESS;

    // q, k, v are persistent members.
    cl_mem q = buf_q_;
    cl_mem k = buf_k_;
    cl_mem v = buf_v_;

    auto fail_qkv = [&]() -> cl_mem {
        // q/k/v are persistent — do NOT release.
        return nullptr;
    };

    // ── Q/K/V projections.
    //   This template assumes nn.Linear separate projections (Llama, Mistral, etc.).
    //   If (model metadata) says
    //     weight_key_parent_classes.<key> == "Conv1D"
    //   the parent module is HF Conv1D (weight stored as [in, out], NOT
    //   [out, in]). Replace pytorch_linear → pytorch_conv1d AT EVERY CALL
    //   and adapt the architecture: GPT-2 / GPT-Neo / OPT use a SINGLE fused
    //   c_attn projection [H -> 3H] then split (NOT three separate calls).
    //   Build will REFUSE pytorch_linear() on a Conv1D-stored weight.
    if (!pytorch_linear(queue, seq_q, Q_DIM,  H, hidden_states, wq_, q)) return fail_qkv();
    if (!pytorch_linear(queue, seq_q, KV_DIM, H, hidden_states, wk_, k)) return fail_qkv();
    if (!pytorch_linear(queue, seq_q, KV_DIM, H, hidden_states, wv_, v)) return fail_qkv();

    // PREFILL-DUMP: pre-norm/pre-RoPE Q/K and V.
    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_query_states_pre_rope", layer_idx_, queue, q,     (size_t)seq_q * Q_DIM);
    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_key_states_pre_rope",   layer_idx_, queue, k,     (size_t)seq_q * KV_DIM);
    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_value_states",          layer_idx_, queue, v,     (size_t)seq_q * KV_DIM);

    // ── q_layernorm / k_layernorm (Lfm2-specific; applied BEFORE RoPE)
    //   modeling_lfm2.py:255-256:
    //     q = q_layernorm(q_proj(h).view(*, H, D))
    //     k = k_layernorm(k_proj(h).view(*, KVH, D))
    //   RMSNorm operates on the LAST axis (head_dim). q is laid out [seq_q, QH, D]
    //   contiguous, so dispatching rmsnorm with rows=seq_q*QH, cols=D normalizes
    //   each [head_dim]-vector, matching the PyTorch semantics exactly.
    {
        auto apply_qk_norm = [&](cl_mem buf, cl_mem tmp, int rows, int cols, cl_mem w, const char* tag) -> bool {
            const float eps = MODEL_CONFIG::NORM_EPS;
            cl_int e = CL_SUCCESS;
            int arg = 0;
            e  = clSetKernelArg(rmsnorm_kernel_, arg++, sizeof(cl_mem), &buf);
            e |= clSetKernelArg(rmsnorm_kernel_, arg++, sizeof(cl_mem), &w);
            e |= clSetKernelArg(rmsnorm_kernel_, arg++, sizeof(cl_mem), &tmp);
            e |= clSetKernelArg(rmsnorm_kernel_, arg++, sizeof(int), &rows);
            e |= clSetKernelArg(rmsnorm_kernel_, arg++, sizeof(int), &cols);
            e |= clSetKernelArg(rmsnorm_kernel_, arg++, sizeof(float), &eps);
            if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("%s setArg failed: %d", tag, e); return false; }
            const size_t WG = 64;
            size_t gws = (size_t)rows * WG;
            size_t lws = WG;
            e = clEnqueueNDRangeKernel(queue, rmsnorm_kernel_, 1, nullptr, &gws, &lws, 0, nullptr,
                                       KernelProfiler::event_for("attn_qk_rmsnorm"));
            if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("%s dispatch failed: %d", tag, e); return false; }
            // Copy normalized values back into the original q/k buffer.
            const size_t bytes = (size_t)rows * (size_t)cols * sizeof(nnopt_storage_t);
            e = clEnqueueCopyBuffer(queue, tmp, buf, 0, 0, bytes, 0, nullptr, nullptr);
            if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("%s copy back failed: %d", tag, e); return false; }
            return true;
        };
        if (!apply_qk_norm(q, buf_q_norm_tmp_, seq_q * QH,  D, q_ln_w_, "q_layernorm")) return fail_qkv();
        if (!apply_qk_norm(k, buf_k_norm_tmp_, seq_q * KVH, D, k_ln_w_, "k_layernorm")) return fail_qkv();
    }
    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_query_states_post_qnorm", layer_idx_, queue, q, (size_t)seq_q * Q_DIM);
    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_key_states_post_knorm",   layer_idx_, queue, k, (size_t)seq_q * KV_DIM);

    // ── RoPE (optional)
    // LFM2.5 config has USES_ROPE=false (absolute positional embeddings), so this must be skipped.
    if (MODEL_CONFIG::USES_ROPE) {
        // NOTE: rope_apply_qk expects cos/sin to be indexed by absolute position (start_pos + t).
        // Therefore we must build tables up to MAX_POSITION_EMBEDDINGS (or at least seq_k), not just seq_k.
        cl_mem cos_use = nullptr;
        cl_mem sin_use = nullptr;
        if (!ensure_rope_tables(queue, MODEL_CONFIG::MAX_POSITION_EMBEDDINGS)) {
            NNOPT_ERROR_FMT("ensure_rope_tables failed at layer %d", layer_idx_);
            return fail_qkv();
        }
        cos_use = cos_;
        sin_use = sin_;

        if (!rope_kernel_) {
            NNOPT_ERROR_FMT("rope_kernel_ is null but USES_ROPE=true (layer %d)", layer_idx_);
            return fail_qkv();
        }

        if (!_set_arg_checked(rope_kernel_, 0, sizeof(cl_mem), &q, "q")) return fail_qkv();
        if (!_set_arg_checked(rope_kernel_, 1, sizeof(cl_mem), &k, "k")) return fail_qkv();
        if (!_set_arg_checked(rope_kernel_, 2, sizeof(cl_mem), &cos_use, "cos")) return fail_qkv();
        if (!_set_arg_checked(rope_kernel_, 3, sizeof(cl_mem), &sin_use, "sin")) return fail_qkv();
        if (!_set_arg_checked(rope_kernel_, 4, sizeof(int), &seq_q, "seq_q")) return fail_qkv();
        if (!_set_arg_checked(rope_kernel_, 5, sizeof(int), &QH, "q_heads")) return fail_qkv();
        if (!_set_arg_checked(rope_kernel_, 6, sizeof(int), &KVH, "kv_heads")) return fail_qkv();
        if (!_set_arg_checked(rope_kernel_, 7, sizeof(int), &D, "head_dim")) return fail_qkv();
        if (!_set_arg_checked(rope_kernel_, 8, sizeof(int), &start_pos, "start_pos")) return fail_qkv();

        {
            const int half_dim = D / 2;
            size_t gws = (size_t)(seq_q * QH * half_dim + seq_q * KVH * half_dim);
            err = clEnqueueNDRangeKernel(queue, rope_kernel_, 1, nullptr, &gws, nullptr, 0, nullptr, KernelProfiler::event_for("rope_apply_qk"));
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("rope dispatch failed: %d", err); return fail_qkv(); }
        }
    }

    // Dump post-(optional)-RoPE Q/K.
    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_query_states", layer_idx_, queue, q, (size_t)seq_q * Q_DIM);
    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_key_states",   layer_idx_, queue, k, (size_t)seq_q * KV_DIM);

    // ── Append rotated K and V into the persistent cache at start_pos.
    {
        const size_t dst_off_bytes = (size_t)start_pos * (size_t)KV_DIM * sizeof(nnopt_storage_t);
        const size_t kv_bytes = (size_t)seq_q * (size_t)KV_DIM * sizeof(nnopt_storage_t);
        err = clEnqueueCopyBuffer(queue, k, k_cache_, 0, dst_off_bytes, kv_bytes, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("copy k -> k_cache_ failed: %d", err); return fail_qkv(); }
        err = clEnqueueCopyBuffer(queue, v, v_cache_, 0, dst_off_bytes, kv_bytes, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("copy v -> v_cache_ failed: %d", err); return fail_qkv(); }
    }

    // K and V buffers stay persistent — cache holds the active values.
    // (No release: buf_k_/buf_v_ are owned by `this`.)
    k = nullptr;
    v = nullptr;

    // ── Scores: [QH, seq_q, seq_k]  — persistent buffer (capacity grown ×2)
    cl_mem scores = buf_scores_;

    auto fail_scores = [&]() -> cl_mem {
        // q/scores are persistent — do NOT release.
        return nullptr;
    };

    // gqa_attn_scores — single launch covers all heads.
    if (!_set_arg_checked(scores_kernel_, 0, sizeof(cl_mem), &q, "q"))             return fail_scores();
    if (!_set_arg_checked(scores_kernel_, 1, sizeof(cl_mem), &k_cache_, "k_cache")) return fail_scores();
    if (!_set_arg_checked(scores_kernel_, 2, sizeof(cl_mem), &scores, "scores"))    return fail_scores();
    if (!_set_arg_checked(scores_kernel_, 3, sizeof(int), &seq_q, "seq_q"))         return fail_scores();
    if (!_set_arg_checked(scores_kernel_, 4, sizeof(int), &seq_k, "seq_k"))         return fail_scores();
    if (!_set_arg_checked(scores_kernel_, 5, sizeof(int), &QH, "num_q_heads"))      return fail_scores();
    if (!_set_arg_checked(scores_kernel_, 6, sizeof(int), &KVH, "num_kv_heads"))    return fail_scores();
    if (!_set_arg_checked(scores_kernel_, 7, sizeof(int), &D, "head_dim"))          return fail_scores();
    if (!_set_arg_checked(scores_kernel_, 8, sizeof(float), &scale, "scale"))       return fail_scores();
    {
        size_t gws_s[3] = { (size_t)QH, (size_t)seq_q, (size_t)seq_k };
        err = clEnqueueNDRangeKernel(queue, scores_kernel_, 3, nullptr, gws_s, nullptr, 0, nullptr, KernelProfiler::event_for("attn_scores_QKt"));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("scores dispatch failed: %d", err); return fail_scores(); }
    }

    // PREFILL-DUMP: pre-softmax raw QK^T / sqrt(D) scores. Useful for
    // catching FAIL_INF (causal mask -inf leaking into ranges where it
    // shouldn't) or magnitude-blow-up bugs (missing scale). sys.settrace
    // captures attn_weights' LAST binding (post-softmax), so this name
    // is _-prefixed to mark it inspection-only — the SxS gate ignores
    // names not in dump_spec.json.
    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_scores", layer_idx_, queue, scores, (size_t)QH * seq_q * seq_k);

    // gqa_softmax — single launch over all (QH * seq_q) rows.
    {
        const int total_rows = QH * seq_q;
        if (!_set_arg_checked(softmax_kernel_, 0, sizeof(cl_mem), &scores, "scores")) return fail_scores();
        if (!_set_arg_checked(softmax_kernel_, 1, sizeof(int), &seq_q, "seq_q"))      return fail_scores();
        if (!_set_arg_checked(softmax_kernel_, 2, sizeof(int), &seq_k, "seq_k"))      return fail_scores();
        if (!_set_arg_checked(softmax_kernel_, 3, sizeof(int), &total_rows, "total_rows")) return fail_scores();

        size_t gws_sm = (size_t)total_rows;
        err = clEnqueueNDRangeKernel(queue, softmax_kernel_, 1, nullptr, &gws_sm, nullptr, 0, nullptr, KernelProfiler::event_for("attn_softmax"));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("softmax dispatch failed: %d", err); return fail_scores(); }
    }

    // PREFILL-DUMP: post-softmax attention probabilities. HF Llama reassigns
    // attn_weights = softmax(...) so sys.settrace's last-binding-wins captures
    // THIS value under that name. SxS dump_spec entry for
    // block0_sub_attn_attn_weights matches this dump.
    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_attn_weights", layer_idx_, queue, scores, (size_t)QH * seq_q * seq_k);

    // gqa_attn_out — persistent attn_out buffer.
    cl_mem attn_out = buf_attn_out_;

    auto fail_out = [&]() -> cl_mem {
        // q/scores/attn_out persistent — do NOT release.
        return nullptr;
    };

    if (!_set_arg_checked(out_kernel_, 0, sizeof(cl_mem), &scores, "scores"))    return fail_out();
    if (!_set_arg_checked(out_kernel_, 1, sizeof(cl_mem), &v_cache_, "v_cache")) return fail_out();
    if (!_set_arg_checked(out_kernel_, 2, sizeof(cl_mem), &attn_out, "out"))     return fail_out();
    if (!_set_arg_checked(out_kernel_, 3, sizeof(int), &seq_q, "seq_q"))         return fail_out();
    if (!_set_arg_checked(out_kernel_, 4, sizeof(int), &seq_k, "seq_k"))         return fail_out();
    if (!_set_arg_checked(out_kernel_, 5, sizeof(int), &QH, "num_q_heads"))      return fail_out();
    if (!_set_arg_checked(out_kernel_, 6, sizeof(int), &KVH, "num_kv_heads"))    return fail_out();
    if (!_set_arg_checked(out_kernel_, 7, sizeof(int), &D, "head_dim"))          return fail_out();
    {
        // Vec4 output: each thread emits 4 d-values, so D dim becomes D/4.
        size_t gws_o[3] = { (size_t)QH, (size_t)seq_q, (size_t)(D / 4) };
        err = clEnqueueNDRangeKernel(queue, out_kernel_, 3, nullptr, gws_o, nullptr, 0, nullptr, KernelProfiler::event_for("attn_out_PV"));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("attn_out dispatch failed: %d", err); return fail_out(); }
    }

    // PREFILL-DUMP: attention output before o_proj — HF Llama binds this to
    // attn_output (the local var that gets reassigned to the projected value
    // by the o_proj line). The pre-o_proj binding is what sys.settrace
    // captures FIRST; the post-o_proj binding overwrites it. To dump BOTH,
    // the pre-o_proj name uses _attn_output_pre_proj and the post-o_proj
    // dump (block%d_sub_attn_out, below) is the canonical attention output.
    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_attn_output_pre_proj", layer_idx_, queue, attn_out, (size_t)seq_q * Q_DIM);

    // ── Output projection: attn_out [seq_q, Q_DIM] -> proj [seq_q, H]  (persistent)
    cl_mem proj = buf_proj_;

    if (!pytorch_linear(queue, seq_q, H, Q_DIM, attn_out, wo_, proj)) {
        return nullptr;
    }

    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_out", layer_idx_, queue, proj, (size_t)seq_q * H);

    // All intermediates are persistent — borrowed handle.
    return proj;
}
