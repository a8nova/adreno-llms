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
#include "utils.h"
#include "model_config.h"
#include "prof.h"
#include <clblast.h>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

Attention::Attention(OpenCLContext& cl_ctx, Weights& weights, int layer_idx)
    : cl_ctx_(cl_ctx), weights_(weights), layer_idx_(layer_idx) {}

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
    if (decode_q_buf_)        clReleaseMemObject(decode_q_buf_);
    if (decode_k_buf_)        clReleaseMemObject(decode_k_buf_);
    if (decode_v_buf_)        clReleaseMemObject(decode_v_buf_);
    if (decode_scores_buf_)   clReleaseMemObject(decode_scores_buf_);
    if (decode_attn_out_buf_) clReleaseMemObject(decode_attn_out_buf_);
    if (fused_qkv_m1_)            clReleaseKernel(fused_qkv_m1_);
    if (fused_rope_kvwrite_m1_)   clReleaseKernel(fused_rope_kvwrite_m1_);
    if (fused_decode_attn_m1_)    clReleaseKernel(fused_decode_attn_m1_);
    if (fused_oproj_res_m1_)      clReleaseKernel(fused_oproj_res_m1_);
    if (fused_qkv_no4_img_)       clReleaseKernel(fused_qkv_no4_img_);
    if (gemv_k576_no4_img_)       clReleaseKernel(gemv_k576_no4_img_);
    if (fused_oproj_no4_img_)     clReleaseKernel(fused_oproj_no4_img_);
    if (wq_img_)                  clReleaseMemObject(wq_img_);
    if (wk_img_)                  clReleaseMemObject(wk_img_);
    if (wv_img_)                  clReleaseMemObject(wv_img_);
    if (wo_img_)                  clReleaseMemObject(wo_img_);
    if (block_fused_prog_)        clReleaseProgram(block_fused_prog_);
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
    rope_program_ = cl_ctx_.build_program_from_file(
        "kernels/rope.cl",
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
        ""
#endif
    );
    if (!rope_program_) { NNOPT_ERROR("Failed to build rope.cl"); return false; }

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
    rope_kernel_ = clCreateKernel(rope_program_, "rope_apply_qk", &err);
    if (err != CL_SUCCESS || !rope_kernel_) {
        NNOPT_ERROR_FMT("clCreateKernel rope_apply_qk failed: %d", err);
        return false;
    }

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

    if (!set_weights()) return false;
    if (!ensure_kv_cache()) return false;

    // ── Decode fast-path kernels (block_fused.cl)
    block_fused_prog_ = cl_ctx_.build_program_from_file(
        "kernels/block_fused.cl",
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
        ""
#endif
    );
    if (!block_fused_prog_) { NNOPT_ERROR("Failed to build block_fused.cl (attn)"); return false; }

    fused_qkv_m1_ = clCreateKernel(block_fused_prog_, "fused_qkv_gemv_m1", &err);
    if (err != CL_SUCCESS || !fused_qkv_m1_) {
        NNOPT_ERROR_FMT("clCreateKernel fused_qkv_gemv_m1 failed: %d", err);
        return false;
    }
    fused_rope_kvwrite_m1_ = clCreateKernel(block_fused_prog_, "fused_rope_kvwrite_m1", &err);
    if (err != CL_SUCCESS || !fused_rope_kvwrite_m1_) {
        NNOPT_ERROR_FMT("clCreateKernel fused_rope_kvwrite_m1 failed: %d", err);
        return false;
    }
    fused_oproj_res_m1_ = clCreateKernel(block_fused_prog_, "fused_oproj_residual_m1", &err);
    if (err != CL_SUCCESS || !fused_oproj_res_m1_) {
        NNOPT_ERROR_FMT("clCreateKernel fused_oproj_residual_m1 failed: %d", err);
        return false;
    }

    fused_decode_attn_m1_ = clCreateKernel(block_fused_prog_, "fused_decode_attn_m1", &err);
    if (err != CL_SUCCESS || !fused_decode_attn_m1_) {
        NNOPT_ERROR_FMT("clCreateKernel fused_decode_attn_m1 failed: %d", err);
        return false;
    }

    // ── Persistent decode buffers — allocated once, reused every decode step.
    const int Q_DIM_CONST  = MODEL_CONFIG::NUM_ATTENTION_HEADS   * MODEL_CONFIG::HEAD_DIM;
    const int KV_DIM_CONST = MODEL_CONFIG::NUM_KEY_VALUE_HEADS   * MODEL_CONFIG::HEAD_DIM;
    cl_context bctx = cl_ctx_.context();
    cl_int berr = CL_SUCCESS;
    decode_q_buf_        = clCreateBuffer(bctx, CL_MEM_READ_WRITE, (size_t)Q_DIM_CONST  * sizeof(nnopt_storage_t), nullptr, &berr);
    if (berr || !decode_q_buf_) { NNOPT_ERROR_FMT("alloc decode_q_buf_ failed: %d", berr); return false; }
    decode_k_buf_        = clCreateBuffer(bctx, CL_MEM_READ_WRITE, (size_t)KV_DIM_CONST * sizeof(nnopt_storage_t), nullptr, &berr);
    if (berr || !decode_k_buf_) { NNOPT_ERROR_FMT("alloc decode_k_buf_ failed: %d", berr); return false; }
    decode_v_buf_        = clCreateBuffer(bctx, CL_MEM_READ_WRITE, (size_t)KV_DIM_CONST * sizeof(nnopt_storage_t), nullptr, &berr);
    if (berr || !decode_v_buf_) { NNOPT_ERROR_FMT("alloc decode_v_buf_ failed: %d", berr); return false; }
    decode_attn_out_buf_ = clCreateBuffer(bctx, CL_MEM_READ_WRITE, (size_t)Q_DIM_CONST  * sizeof(nnopt_storage_t), nullptr, &berr);
    if (berr || !decode_attn_out_buf_) { NNOPT_ERROR_FMT("alloc decode_attn_out_buf_ failed: %d", berr); return false; }
    // decode_scores_buf_ is unused — scores live in kernel local memory.

#ifdef NNOPT_USE_FP16
    // Image2d_t-backed weight views (Adreno texture cache).
    // K=576 (HIDDEN_SIZE) for all four. Wq is [576,576], Wk/Wv are [192,576],
    // Wo is [576,576]. All well below CL_DEVICE_IMAGE2D_MAX_HEIGHT=16384.
    fused_qkv_no4_img_   = nullptr;  // deprecated — fused-3-image variant regressed (branchy code).
    gemv_k576_no4_img_   = clCreateKernel(block_fused_prog_, "gemv_m1_k576_no4_img", &err);
    if (err != CL_SUCCESS) gemv_k576_no4_img_ = nullptr;
    fused_oproj_no4_img_ = clCreateKernel(block_fused_prog_, "fused_oproj_residual_m1_no4_img", &err);
    if (err != CL_SUCCESS) fused_oproj_no4_img_ = nullptr;

    if (gemv_k576_no4_img_ || fused_oproj_no4_img_) {
        const int H = MODEL_CONFIG::HIDDEN_SIZE;
        const int K_PIX = H / 4;  // 144 vec4 pixels per row

        cl_image_format fmt;
        fmt.image_channel_order     = CL_RGBA;
        fmt.image_channel_data_type = CL_HALF_FLOAT;

        auto wrap = [&](cl_mem buf, int N) -> cl_mem {
            cl_image_desc desc;
            std::memset(&desc, 0, sizeof(desc));
            desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
            desc.image_width     = (size_t)K_PIX;
            desc.image_height    = (size_t)N;
            desc.image_row_pitch = 0;
            desc.buffer          = buf;
            cl_int e = CL_SUCCESS;
            cl_mem img = clCreateImage(bctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &e);
            return (e == CL_SUCCESS) ? img : nullptr;
        };

        if (gemv_k576_no4_img_) {
            wq_img_ = wrap(wq_, Q_DIM_CONST);
            wk_img_ = wrap(wk_, KV_DIM_CONST);
            wv_img_ = wrap(wv_, KV_DIM_CONST);
            qkv_img_ready_ = (wq_img_ && wk_img_ && wv_img_);
        }
        if (fused_oproj_no4_img_) {
            wo_img_ = wrap(wo_, Q_DIM_CONST);
            oproj_img_ready_ = (wo_img_ != nullptr);
        }
    }
#endif

    NNOPT_LAYER_INIT_FMT("block%d_sub_attn", layer_idx_);
    return true;
}

// TODO (agent): implement set_weights() with the model's actual weight keys.
// Read (model metadata) for the exact templated keys.
// Common patterns (replace {i} with layer_idx_):
//   Llama family:  "model.layers.{i}.self_attn.q_proj.weight" (and k_proj, v_proj, o_proj)
//   GPT-2 family:  "h.{i}.attn.c_attn.weight" (combined Q/K/V) + "h.{i}.attn.c_proj.weight"
//   Phi family:    "model.layers.{i}.self_attn.qkv_proj.weight" + "...o_proj.weight"
bool Attention::set_weights() {
    // Reference: (model metadata)::weight_keys
    const std::string prefix = "model.layers." + std::to_string(layer_idx_) + ".self_attn.";
    wq_ = weights_.get_buffer(prefix + "q_proj.weight");
    wk_ = weights_.get_buffer(prefix + "k_proj.weight");
    wv_ = weights_.get_buffer(prefix + "v_proj.weight");
    wo_ = weights_.get_buffer(prefix + "o_proj.weight");

    if (!wq_ || !wk_ || !wv_ || !wo_) {
        NNOPT_ERROR_FMT("Attention[%d]: missing weight buffer(s)", layer_idx_);
        return false;
    }
    return true;
}

cl_mem Attention::forward(cl_command_queue queue,
                          cl_mem input,
                          cl_mem cos,
                          cl_mem sin,
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

    const float scale = 1.0f / std::sqrt((float)D);

    cl_context ctx = cl_ctx_.context();
    cl_int err = CL_SUCCESS;

    cl_mem q = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)seq_q * Q_DIM * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !q) { NNOPT_ERROR_FMT("alloc q failed: %d", err); return nullptr; }
    cl_mem k = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)seq_q * KV_DIM * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !k) { NNOPT_ERROR_FMT("alloc k failed: %d", err); clReleaseMemObject(q); return nullptr; }
    cl_mem v = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)seq_q * KV_DIM * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !v) {
        NNOPT_ERROR_FMT("alloc v failed: %d", err);
        clReleaseMemObject(q); clReleaseMemObject(k);
        return nullptr;
    }

    auto fail_qkv = [&]() -> cl_mem {
        if (q) clReleaseMemObject(q);
        if (k) clReleaseMemObject(k);
        if (v) clReleaseMemObject(v);
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

    // PREFILL-DUMP: pre-RoPE Q/K and V are useful for SxS bisection.
    // Names match HF Llama's local-var names (query_states / key_states /
    // value_states) so sys.settrace last-binding-wins captures align — but
    // RoPE reassigns query/key, so for HF-style refs the LAST binding will
    // be POST-rope. We emit the post-RoPE dump below; this pre-RoPE pair is
    // for fp16-port debugging only (the SxS gate ignores names not in
    // dump_spec.json). _-prefix marks them as inspection-only so Build's
    // unknown-dump-name advisory does not flag them.
    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_query_states_pre_rope", layer_idx_, queue, q,     (size_t)seq_q * Q_DIM);
    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_key_states_pre_rope",   layer_idx_, queue, k,     (size_t)seq_q * KV_DIM);
    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_value_states",          layer_idx_, queue, v,     (size_t)seq_q * KV_DIM);

    // ── RoPE: prefer caller-provided cos/sin, otherwise lazy-init the layer's own.
    cl_mem cos_use = cos;
    cl_mem sin_use = sin;
    if (!cos_use || !sin_use) {
        if (!ensure_rope_tables(queue, seq_k)) {
            NNOPT_ERROR_FMT("ensure_rope_tables failed at layer %d", layer_idx_);
            return fail_qkv();
        }
        cos_use = cos_;
        sin_use = sin_;
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
        err = nnopt_prof::enqueue(queue, rope_kernel_, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("rope dispatch failed: %d", err); return fail_qkv(); }
    }

    // PREFILL-DUMP: post-RoPE Q/K — these are the LAST bindings of
    // query_states / key_states in HF Llama's forward() (RoPE reassigns
    // them). sys.settrace captures THESE values under those names, so the
    // SxS dump_spec entry for block0_sub_attn_query_states / key_states
    // matches against this dump (not the pre-RoPE one above).
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

    // K and V buffers are no longer needed — cache holds the data.
    clReleaseMemObject(k); k = nullptr;
    clReleaseMemObject(v); v = nullptr;

    // ── Scores: [QH, seq_q, seq_k]
    cl_mem scores = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                   (size_t)QH * seq_q * seq_k * sizeof(nnopt_storage_t),
                                   nullptr, &err);
    if (err != CL_SUCCESS || !scores) {
        NNOPT_ERROR_FMT("alloc scores failed: %d", err);
        clReleaseMemObject(q);
        return nullptr;
    }

    auto fail_scores = [&]() -> cl_mem {
        if (q) clReleaseMemObject(q);
        if (scores) clReleaseMemObject(scores);
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
        err = nnopt_prof::enqueue(queue, scores_kernel_, 3, nullptr, gws_s, nullptr, 0, nullptr, nullptr);
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
        err = nnopt_prof::enqueue(queue, softmax_kernel_, 1, nullptr, &gws_sm, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("softmax dispatch failed: %d", err); return fail_scores(); }
    }

    // PREFILL-DUMP: post-softmax attention probabilities. HF Llama reassigns
    // attn_weights = softmax(...) so sys.settrace's last-binding-wins captures
    // THIS value under that name. SxS dump_spec entry for
    // block0_sub_attn_attn_weights matches this dump.
    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_attn_weights", layer_idx_, queue, scores, (size_t)QH * seq_q * seq_k);

    // gqa_attn_out — single launch covering all heads.
    cl_mem attn_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                     (size_t)seq_q * Q_DIM * sizeof(nnopt_storage_t),
                                     nullptr, &err);
    if (err != CL_SUCCESS || !attn_out) {
        NNOPT_ERROR_FMT("alloc attn_out failed: %d", err);
        return fail_scores();
    }

    auto fail_out = [&]() -> cl_mem {
        if (q) clReleaseMemObject(q);
        if (scores) clReleaseMemObject(scores);
        if (attn_out) clReleaseMemObject(attn_out);
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
        err = nnopt_prof::enqueue(queue, out_kernel_, 3, nullptr, gws_o, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("attn_out dispatch failed: %d", err); return fail_out(); }
    }

    // PREFILL-DUMP: attention output before o_proj — HF Llama binds this to
    // attn_output (the local var that gets reassigned to the projected value
    // by the o_proj line). The pre-o_proj binding is what sys.settrace
    // captures FIRST; the post-o_proj binding overwrites it. To dump BOTH,
    // the pre-o_proj name uses _attn_output_pre_proj and the post-o_proj
    // dump (block%d_sub_attn_out, below) is the canonical attention output.
    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_attn_output_pre_proj", layer_idx_, queue, attn_out, (size_t)seq_q * Q_DIM);

    // ── Output projection: attn_out [seq_q, Q_DIM] -> proj [seq_q, H]
    cl_mem proj = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 (size_t)seq_q * H * sizeof(nnopt_storage_t),
                                 nullptr, &err);
    if (err != CL_SUCCESS || !proj) {
        NNOPT_ERROR_FMT("alloc out_proj failed: %d", err);
        return fail_out();
    }

    // Output projection. If Attention.json says
    //   weight_key_parent_classes.c_proj == "Conv1D"
    // (GPT-2 / GPT-Neo / OPT), swap pytorch_linear → pytorch_conv1d here.
    // Build refuses pytorch_linear() on a Conv1D-stored weight.
    if (!pytorch_linear(queue, seq_q, H, Q_DIM, attn_out, wo_, proj)) {
        if (q) clReleaseMemObject(q);
        clReleaseMemObject(scores);
        clReleaseMemObject(attn_out);
        clReleaseMemObject(proj);
        return nullptr;
    }

    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_out", layer_idx_, queue, proj, (size_t)seq_q * H);

    // Cleanup intermediates. proj is returned and owned by caller.
    if (q) clReleaseMemObject(q);
    clReleaseMemObject(scores);
    clReleaseMemObject(attn_out);

    return proj;
}

// ── Decode fast path (M=1) ───────────────────────────────────────────────────
// Replaces 3 CLBlast QKV GEMMs + rope + KV copy + CLBlast o_proj + element_add
// with 5 custom GEMV kernel dispatches. Updates residual in-place.
bool Attention::forward_decode_into_residual(cl_command_queue queue,
                                             cl_mem x, int start_pos,
                                             cl_mem residual) {
    const int H     = MODEL_CONFIG::HIDDEN_SIZE;
    const int QH    = MODEL_CONFIG::NUM_ATTENTION_HEADS;
    const int KVH   = MODEL_CONFIG::NUM_KEY_VALUE_HEADS;
    const int D     = MODEL_CONFIG::HEAD_DIM;
    const int Q_DIM = QH * D;
    const int KV_DIM = KVH * D;
    const int seq_k  = start_pos + 1;
    const float scale = 1.0f / std::sqrt((float)D);

    if (!ensure_kv_cache()) return false;
    if (!ensure_rope_tables(queue, seq_k)) return false;

    cl_int err = CL_SUCCESS;

    // Use persistent decode buffers — no allocation per step.
    cl_mem q = decode_q_buf_;
    cl_mem k = decode_k_buf_;
    cl_mem v = decode_v_buf_;

    // 1. QKV projection: prefer image2d_t no4 path (3 separate dispatches share x).
    //    Fused-3-images variant regressed in early measurement — branchy code in
    //    a single kernel (per-segment image select) hurt texture-cache flow.
    //    Three back-to-back dispatches of the proven gemv_m1_k576_no4_img total
    //    340ms vs 672ms for the fused variant.
    if (qkv_img_ready_ && gemv_k576_no4_img_) {
        const size_t WG = 64;
        size_t lws = WG;
        // Q
        if (!_set_arg_checked(gemv_k576_no4_img_, 0, sizeof(cl_mem), &x,        "x"))        return false;
        if (!_set_arg_checked(gemv_k576_no4_img_, 1, sizeof(cl_mem), &wq_img_,  "Wq_img"))   return false;
        if (!_set_arg_checked(gemv_k576_no4_img_, 2, sizeof(cl_mem), &q,        "q_out"))    return false;
        if (!_set_arg_checked(gemv_k576_no4_img_, 3, sizeof(int),    &Q_DIM,    "N=Q_DIM"))  return false;
        size_t gws_q = (size_t)(Q_DIM / 4) * WG;
        err = nnopt_prof::enqueue(queue, gemv_k576_no4_img_, 1, nullptr, &gws_q, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("gemv_m1_k576_no4_img(Q) failed: %d", err); return false; }
        // K
        if (!_set_arg_checked(gemv_k576_no4_img_, 0, sizeof(cl_mem), &x,        "x"))        return false;
        if (!_set_arg_checked(gemv_k576_no4_img_, 1, sizeof(cl_mem), &wk_img_,  "Wk_img"))   return false;
        if (!_set_arg_checked(gemv_k576_no4_img_, 2, sizeof(cl_mem), &k,        "k_out"))    return false;
        if (!_set_arg_checked(gemv_k576_no4_img_, 3, sizeof(int),    &KV_DIM,   "N=KV_DIM")) return false;
        size_t gws_k = (size_t)(KV_DIM / 4) * WG;
        err = nnopt_prof::enqueue(queue, gemv_k576_no4_img_, 1, nullptr, &gws_k, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("gemv_m1_k576_no4_img(K) failed: %d", err); return false; }
        // V
        if (!_set_arg_checked(gemv_k576_no4_img_, 0, sizeof(cl_mem), &x,        "x"))        return false;
        if (!_set_arg_checked(gemv_k576_no4_img_, 1, sizeof(cl_mem), &wv_img_,  "Wv_img"))   return false;
        if (!_set_arg_checked(gemv_k576_no4_img_, 2, sizeof(cl_mem), &v,        "v_out"))    return false;
        if (!_set_arg_checked(gemv_k576_no4_img_, 3, sizeof(int),    &KV_DIM,   "N=KV_DIM")) return false;
        err = nnopt_prof::enqueue(queue, gemv_k576_no4_img_, 1, nullptr, &gws_k, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("gemv_m1_k576_no4_img(V) failed: %d", err); return false; }
    } else {
        if (!_set_arg_checked(fused_qkv_m1_, 0, sizeof(cl_mem), &x,      "x"))      return false;
        if (!_set_arg_checked(fused_qkv_m1_, 1, sizeof(cl_mem), &wq_,    "w_q"))    return false;
        if (!_set_arg_checked(fused_qkv_m1_, 2, sizeof(cl_mem), &wk_,    "w_k"))    return false;
        if (!_set_arg_checked(fused_qkv_m1_, 3, sizeof(cl_mem), &wv_,    "w_v"))    return false;
        if (!_set_arg_checked(fused_qkv_m1_, 4, sizeof(cl_mem), &q,      "q_out"))  return false;
        if (!_set_arg_checked(fused_qkv_m1_, 5, sizeof(cl_mem), &k,      "k_out"))  return false;
        if (!_set_arg_checked(fused_qkv_m1_, 6, sizeof(cl_mem), &v,      "v_out"))  return false;
        if (!_set_arg_checked(fused_qkv_m1_, 7, sizeof(int),    &H,      "H"))      return false;
        if (!_set_arg_checked(fused_qkv_m1_, 8, sizeof(int),    &Q_DIM,  "Q_DIM"))  return false;
        if (!_set_arg_checked(fused_qkv_m1_, 9, sizeof(int),    &KV_DIM, "KV_DIM")) return false;
        const size_t WG = 64;
        size_t gws = (size_t)(Q_DIM + 2 * KV_DIM) * WG;
        size_t lws = WG;
        err = nnopt_prof::enqueue(queue, fused_qkv_m1_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("fused_qkv_gemv_m1 failed: %d", err); return false; }
    }

    // 2. fused_rope_kvwrite_m1: rotate Q in-place, write K+V into cache.
    const int half_dim = D / 2;
    const int q_pairs  = QH  * half_dim;
    const int k_pairs  = KVH * half_dim;
    if (!_set_arg_checked(fused_rope_kvwrite_m1_, 0, sizeof(cl_mem), &q,        "q"))         return false;
    if (!_set_arg_checked(fused_rope_kvwrite_m1_, 1, sizeof(cl_mem), &k,        "k_in"))      return false;
    if (!_set_arg_checked(fused_rope_kvwrite_m1_, 2, sizeof(cl_mem), &v,        "v_in"))      return false;
    if (!_set_arg_checked(fused_rope_kvwrite_m1_, 3, sizeof(cl_mem), &k_cache_, "k_cache"))   return false;
    if (!_set_arg_checked(fused_rope_kvwrite_m1_, 4, sizeof(cl_mem), &v_cache_, "v_cache"))   return false;
    if (!_set_arg_checked(fused_rope_kvwrite_m1_, 5, sizeof(cl_mem), &cos_,     "cos_tab"))   return false;
    if (!_set_arg_checked(fused_rope_kvwrite_m1_, 6, sizeof(cl_mem), &sin_,     "sin_tab"))   return false;
    if (!_set_arg_checked(fused_rope_kvwrite_m1_, 7, sizeof(int),    &QH,       "q_heads"))   return false;
    if (!_set_arg_checked(fused_rope_kvwrite_m1_, 8, sizeof(int),    &KVH,      "kv_heads"))  return false;
    if (!_set_arg_checked(fused_rope_kvwrite_m1_, 9, sizeof(int),    &D,        "head_dim"))  return false;
    if (!_set_arg_checked(fused_rope_kvwrite_m1_,10, sizeof(int),    &KV_DIM,   "kv_dim"))    return false;
    if (!_set_arg_checked(fused_rope_kvwrite_m1_,11, sizeof(int),    &start_pos,"start_pos")) return false;
    {
        size_t gws = (size_t)(q_pairs + k_pairs + KV_DIM);
        err = nnopt_prof::enqueue(queue, fused_rope_kvwrite_m1_, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("fused_rope_kvwrite_m1 failed: %d", err); return false; }
    }

    // 3. fused_decode_attn_m1: QK^T + softmax + weighted-V in one dispatch.
    //    One WG per Q head; dynamic local mem = (seq_k + 64) * sizeof(float).
    const int GRP = QH / KVH;
    const size_t local_mem_bytes = (size_t)(seq_k + 64) * sizeof(float);
    if (!_set_arg_checked(fused_decode_attn_m1_, 0, sizeof(cl_mem), &q,              "q"))       return false;
    if (!_set_arg_checked(fused_decode_attn_m1_, 1, sizeof(cl_mem), &k_cache_,       "k_cache")) return false;
    if (!_set_arg_checked(fused_decode_attn_m1_, 2, sizeof(cl_mem), &v_cache_,       "v_cache")) return false;
    if (!_set_arg_checked(fused_decode_attn_m1_, 3, sizeof(cl_mem), &decode_attn_out_buf_, "attn_out")) return false;
    if (clSetKernelArg(fused_decode_attn_m1_, 4, local_mem_bytes, nullptr) != CL_SUCCESS) {
        NNOPT_ERROR("fused_decode_attn_m1 arg ls failed"); return false;
    }
    if (!_set_arg_checked(fused_decode_attn_m1_, 5, sizeof(int),   &KV_DIM, "KV_DIM")) return false;
    if (!_set_arg_checked(fused_decode_attn_m1_, 6, sizeof(int),   &D,      "D"))      return false;
    if (!_set_arg_checked(fused_decode_attn_m1_, 7, sizeof(int),   &KVH,    "KVH"))    return false;
    if (!_set_arg_checked(fused_decode_attn_m1_, 8, sizeof(int),   &GRP,    "GRP"))    return false;
    if (!_set_arg_checked(fused_decode_attn_m1_, 9, sizeof(int),   &seq_k,  "seq_k"))  return false;
    if (!_set_arg_checked(fused_decode_attn_m1_,10, sizeof(float), &scale,  "scale"))  return false;
    {
        const size_t WG = 64;
        size_t gws = (size_t)QH * WG;
        size_t lws = WG;
        err = nnopt_prof::enqueue(queue, fused_decode_attn_m1_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("fused_decode_attn_m1 failed: %d", err); return false; }
    }

    // 4. o_proj + residual add: prefer image2d_t no4 path; fall back to buffer kernel.
    cl_mem attn_out = decode_attn_out_buf_;
    if (oproj_img_ready_ && fused_oproj_no4_img_) {
        if (!_set_arg_checked(fused_oproj_no4_img_, 0, sizeof(cl_mem), &attn_out, "attn_out")) return false;
        if (!_set_arg_checked(fused_oproj_no4_img_, 1, sizeof(cl_mem), &wo_img_,  "Wo_img"))   return false;
        if (!_set_arg_checked(fused_oproj_no4_img_, 2, sizeof(cl_mem), &residual, "residual")) return false;
        if (!_set_arg_checked(fused_oproj_no4_img_, 3, sizeof(int),    &H,        "H"))        return false;
        const size_t WG = 64;
        size_t gws = (size_t)(H / 4) * WG;
        size_t lws = WG;
        err = nnopt_prof::enqueue(queue, fused_oproj_no4_img_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("fused_oproj_residual_m1_no4_img failed: %d", err); return false; }
    } else {
        if (!_set_arg_checked(fused_oproj_res_m1_, 0, sizeof(cl_mem), &attn_out, "attn_out")) return false;
        if (!_set_arg_checked(fused_oproj_res_m1_, 1, sizeof(cl_mem), &wo_,      "w_o"))      return false;
        if (!_set_arg_checked(fused_oproj_res_m1_, 2, sizeof(cl_mem), &residual, "residual")) return false;
        if (!_set_arg_checked(fused_oproj_res_m1_, 3, sizeof(int),    &H,        "H"))        return false;
        if (!_set_arg_checked(fused_oproj_res_m1_, 4, sizeof(int),    &Q_DIM,    "Q_DIM"))    return false;
        const size_t WG = 64;
        size_t gws = (size_t)H * WG;
        size_t lws = WG;
        err = nnopt_prof::enqueue(queue, fused_oproj_res_m1_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("fused_oproj_residual_m1 failed: %d", err); return false; }
    }

    return true;
}
