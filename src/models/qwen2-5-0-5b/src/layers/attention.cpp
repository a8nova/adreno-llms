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
#include "prof.h"
#include "model_config.h"
#include <clblast.h>
#include <cmath>
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
    if (bias_kernel_) clReleaseKernel(bias_kernel_);
    if (bias_program_) clReleaseProgram(bias_program_);
    if (buf_q_)       clReleaseMemObject(buf_q_);
    if (buf_k_)       clReleaseMemObject(buf_k_);
    if (buf_v_)       clReleaseMemObject(buf_v_);
    if (buf_scores_)  clReleaseMemObject(buf_scores_);
    if (buf_attn_o_)  clReleaseMemObject(buf_attn_o_);
    if (buf_proj_)    clReleaseMemObject(buf_proj_);
    // Step 18 fused-QKV resources.
    if (sub_q_qkv_)   clReleaseMemObject(sub_q_qkv_);
    if (sub_k_qkv_)   clReleaseMemObject(sub_k_qkv_);
    if (sub_v_qkv_)   clReleaseMemObject(sub_v_qkv_);
    if (buf_qkv_)     clReleaseMemObject(buf_qkv_);
    if (wqkv_)        clReleaseMemObject(wqkv_);
    if (bqkv_)        clReleaseMemObject(bqkv_);
}

bool Attention::ensure_activation_buffers_(int seq_q, int seq_k) {
    const int H      = MODEL_CONFIG::HIDDEN_SIZE;
    const int Q_DIM  = MODEL_CONFIG::NUM_ATTENTION_HEADS  * MODEL_CONFIG::HEAD_DIM;
    const int KV_DIM = MODEL_CONFIG::NUM_KEY_VALUE_HEADS  * MODEL_CONFIG::HEAD_DIM;
    const int QH     = MODEL_CONFIG::NUM_ATTENTION_HEADS;

    const bool need_resize_q = !buf_q_ || (seq_q > buf_capacity_seq_q_);
    const bool need_resize_s = !buf_scores_ ||
                               (seq_q > buf_capacity_seq_q_) ||
                               (seq_k > buf_capacity_seq_k_);
    if (!need_resize_q && !need_resize_s &&
        buf_q_ && buf_k_ && buf_v_ && buf_scores_ && buf_attn_o_ && buf_proj_) {
        return true;
    }

    cl_int err = CL_SUCCESS;
    cl_context ctx = cl_ctx_.context();
    auto realloc_buf = [&](cl_mem* dst, size_t bytes, const char* name) -> bool {
        if (*dst) { clReleaseMemObject(*dst); *dst = nullptr; }
        *dst = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
        if (err != CL_SUCCESS || !*dst) {
            NNOPT_ERROR_FMT("Attention[%d]: alloc %s (%zu bytes) failed: %d",
                            layer_idx_, name, bytes, (int)err);
            return false;
        }
        return true;
    };

    const int new_seq_q = (seq_q > buf_capacity_seq_q_) ? seq_q : buf_capacity_seq_q_;
    const int new_seq_k = (seq_k > buf_capacity_seq_k_) ? seq_k : buf_capacity_seq_k_;

    if (need_resize_q) {
        if (!realloc_buf(&buf_q_,      (size_t)new_seq_q * Q_DIM  * sizeof(nnopt_storage_t), "q"))      return false;
        if (!realloc_buf(&buf_k_,      (size_t)new_seq_q * KV_DIM * sizeof(nnopt_storage_t), "k"))      return false;
        if (!realloc_buf(&buf_v_,      (size_t)new_seq_q * KV_DIM * sizeof(nnopt_storage_t), "v"))      return false;
        if (!realloc_buf(&buf_attn_o_, (size_t)new_seq_q * Q_DIM  * sizeof(nnopt_storage_t), "attn_o")) return false;
        if (!realloc_buf(&buf_proj_,   (size_t)new_seq_q * H      * sizeof(nnopt_storage_t), "proj"))   return false;
    }
    if (need_resize_s) {
        if (!realloc_buf(&buf_scores_, (size_t)QH * new_seq_q * new_seq_k * sizeof(nnopt_storage_t), "scores")) return false;
    }
    buf_capacity_seq_q_ = new_seq_q;
    buf_capacity_seq_k_ = new_seq_k;
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

    // Qwen2Attention q/k/v projections have bias (see layer contract Attention.json).
    bias_program_ = cl_ctx_.build_program_from_file(
        "kernels/bias_add.cl",
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
        ""
#endif
    );
    if (!bias_program_) { NNOPT_ERROR("Failed to build bias_add.cl"); return false; }

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

    bias_kernel_ = clCreateKernel(bias_program_, "bias_add_rowmajor", &err);
    if (err != CL_SUCCESS || !bias_kernel_) {
        NNOPT_ERROR_FMT("clCreateKernel bias_add_rowmajor failed: %d", err);
        return false;
    }

    if (!set_weights()) return false;
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
bool Attention::set_weights() {
    // Reference: (model metadata)
    const std::string p = "model.layers." + std::to_string(layer_idx_) + ".self_attn.";

    // Qwen2Attention uses Linear projections WITH bias for q/k/v, and o_proj without bias.
    wq_ = weights_.get_buffer(p + "q_proj.weight");
    wk_ = weights_.get_buffer(p + "k_proj.weight");
    wv_ = weights_.get_buffer(p + "v_proj.weight");
    wo_ = weights_.get_buffer(p + "o_proj.weight");

    bq_ = weights_.get_buffer(p + "q_proj.bias");
    bk_ = weights_.get_buffer(p + "k_proj.bias");
    bv_ = weights_.get_buffer(p + "v_proj.bias");

    if (!wq_ || !wk_ || !wv_ || !wo_ || !bq_ || !bk_ || !bv_) {
        NNOPT_ERROR_FMT("Attention[%d]: missing q/k/v/o weights or q/k/v bias buffers", layer_idx_);
        return false;
    }

    // Step 18: build the stacked QKV weight + bias for the fused decode path.
    // Layout: rows [0..Q_DIM) = q_proj, [Q_DIM..Q_DIM+KV_DIM) = k_proj,
    // [Q_DIM+KV_DIM..Q_DIM+2*KV_DIM) = v_proj. Same fp16 storage as the
    // individual buffers — three clEnqueueCopyBuffer's at init.
    {
        const int H      = MODEL_CONFIG::HIDDEN_SIZE;
        const int Q_DIM  = MODEL_CONFIG::NUM_ATTENTION_HEADS * MODEL_CONFIG::HEAD_DIM;
        const int KV_DIM = MODEL_CONFIG::NUM_KEY_VALUE_HEADS * MODEL_CONFIG::HEAD_DIM;
        const size_t kElem  = sizeof(nnopt_storage_t);
        const size_t kQwBytes = (size_t)Q_DIM  * (size_t)H * kElem;
        const size_t kKwBytes = (size_t)KV_DIM * (size_t)H * kElem;
        const size_t kVwBytes = kKwBytes;
        const size_t kQbBytes = (size_t)Q_DIM  * kElem;
        const size_t kKbBytes = (size_t)KV_DIM * kElem;
        const size_t kVbBytes = kKbBytes;

        cl_int werr = CL_SUCCESS;
        wqkv_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_ONLY,
                               kQwBytes + kKwBytes + kVwBytes, nullptr, &werr);
        bqkv_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_ONLY,
                               kQbBytes + kKbBytes + kVbBytes, nullptr, &werr);
        if (!wqkv_ || !bqkv_) {
            NNOPT_ERROR_FMT("Attention[%d]: alloc wqkv_/bqkv_ failed", layer_idx_);
            qkv_fused_ok_ = false;
        } else {
            cl_command_queue q = cl_ctx_.queue();
            cl_int e1 = clEnqueueCopyBuffer(q, wq_, wqkv_, 0, 0, kQwBytes, 0, nullptr, nullptr);
            cl_int e2 = clEnqueueCopyBuffer(q, wk_, wqkv_, 0, kQwBytes, kKwBytes, 0, nullptr, nullptr);
            cl_int e3 = clEnqueueCopyBuffer(q, wv_, wqkv_, 0, kQwBytes + kKwBytes, kVwBytes, 0, nullptr, nullptr);
            cl_int e4 = clEnqueueCopyBuffer(q, bq_, bqkv_, 0, 0, kQbBytes, 0, nullptr, nullptr);
            cl_int e5 = clEnqueueCopyBuffer(q, bk_, bqkv_, 0, kQbBytes, kKbBytes, 0, nullptr, nullptr);
            cl_int e6 = clEnqueueCopyBuffer(q, bv_, bqkv_, 0, kQbBytes + kKbBytes, kVbBytes, 0, nullptr, nullptr);
            if (e1 == CL_SUCCESS && e2 == CL_SUCCESS && e3 == CL_SUCCESS &&
                e4 == CL_SUCCESS && e5 == CL_SUCCESS && e6 == CL_SUCCESS) {
                qkv_fused_ok_ = true;
            } else {
                NNOPT_ERROR_FMT("Attention[%d]: stacked QKV copy failed (%d %d %d %d %d %d)",
                                layer_idx_, e1, e2, e3, e4, e5, e6);
                qkv_fused_ok_ = false;
            }
        }
    }

    return true;
}

void Attention::preallocate_decode_buffers_max() {
    ensure_activation_buffers_(1, MODEL_CONFIG::MAX_POSITION_EMBEDDINGS);
}

cl_mem Attention::forward(cl_command_queue queue,
                          cl_mem input,
                          cl_mem cos,
                          cl_mem sin,
                          int seq_q,
                          int start_pos,
                          cl_mem counter,
                          cl_mem residual_dest) {
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
    if (!ensure_activation_buffers_(seq_q, seq_k)) return nullptr;

    const float scale = 1.0f / std::sqrt((float)D);

    cl_context ctx = cl_ctx_.context();
    (void)ctx;  // kept for downstream; persistent buffers replace per-call clCreateBuffer
    cl_int err = CL_SUCCESS;

    // Reuse persistent buffers — saves 6 alloc/free per layer per token at decode.
    cl_mem q = buf_q_;
    cl_mem k = buf_k_;
    cl_mem v = buf_v_;

    auto fail_qkv = [&]() -> cl_mem {
        return nullptr;
    };

    // Step 18: fused QKV decode path. Allocate buf_qkv_ + sub-buffers lazily
    // on first decode forward, then dispatch a single GEMV (N=1152) plus a
    // single bias_add (cols=1152). Falls through to unfused below if any
    // step fails (kept buf_q_/k_/v_ writes intact in the unfused branch).
    bool qkv_done = false;
    if (qkv_fused_ok_ && wqkv_ && bqkv_ && seq_q == 1) {
        const int N_FUSED = Q_DIM + 2 * KV_DIM;  // 1152 for Qwen2.5-0.5B
        if (!buf_qkv_) {
            cl_int eb = CL_SUCCESS;
            buf_qkv_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                       (size_t)N_FUSED * sizeof(nnopt_storage_t),
                                       nullptr, &eb);
            if (eb != CL_SUCCESS || !buf_qkv_) {
                NNOPT_ERROR_FMT("Attention[%d]: alloc buf_qkv_ failed: %d", layer_idx_, eb);
                qkv_fused_ok_ = false;
            } else {
                cl_buffer_region rq{0, (size_t)Q_DIM  * sizeof(nnopt_storage_t)};
                cl_buffer_region rk{(size_t)Q_DIM         * sizeof(nnopt_storage_t),
                                    (size_t)KV_DIM * sizeof(nnopt_storage_t)};
                cl_buffer_region rv{(size_t)(Q_DIM + KV_DIM) * sizeof(nnopt_storage_t),
                                    (size_t)KV_DIM * sizeof(nnopt_storage_t)};
                cl_int es = CL_SUCCESS;
                sub_q_qkv_ = clCreateSubBuffer(buf_qkv_, CL_MEM_READ_WRITE,
                                               CL_BUFFER_CREATE_TYPE_REGION, &rq, &es);
                if (es == CL_SUCCESS) sub_k_qkv_ = clCreateSubBuffer(buf_qkv_, CL_MEM_READ_WRITE,
                                                   CL_BUFFER_CREATE_TYPE_REGION, &rk, &es);
                if (es == CL_SUCCESS) sub_v_qkv_ = clCreateSubBuffer(buf_qkv_, CL_MEM_READ_WRITE,
                                                   CL_BUFFER_CREATE_TYPE_REGION, &rv, &es);
                if (es != CL_SUCCESS) {
                    NNOPT_ERROR_FMT("Attention[%d]: sub-buffer creation failed: %d", layer_idx_, es);
                    qkv_fused_ok_ = false;
                }
            }
        }
        if (qkv_fused_ok_ && sub_q_qkv_ && sub_k_qkv_ && sub_v_qkv_) {
            // Single GEMV: out[N=1152] = hidden_states @ wqkv_[1152, H]^T
            if (!pytorch_linear(queue, seq_q, N_FUSED, H, hidden_states, wqkv_, buf_qkv_)) {
                NNOPT_ERROR_FMT("Attention[%d]: fused QKV pytorch_linear failed", layer_idx_);
                qkv_fused_ok_ = false;
            } else {
                // Single bias_add over the full [1152].
                int rows = seq_q;
                int cols = N_FUSED;
                if (!_set_arg_checked(bias_kernel_, 0, sizeof(cl_mem), &buf_qkv_, "out")) return fail_qkv();
                if (!_set_arg_checked(bias_kernel_, 1, sizeof(cl_mem), &bqkv_,    "bias")) return fail_qkv();
                if (!_set_arg_checked(bias_kernel_, 2, sizeof(int),    &rows,     "rows")) return fail_qkv();
                if (!_set_arg_checked(bias_kernel_, 3, sizeof(int),    &cols,     "cols")) return fail_qkv();
                size_t gws = ((size_t)rows * (size_t)cols) >> 2;
                cl_int eb = nnopt_prof::enqueue(queue, bias_kernel_, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
                if (eb != CL_SUCCESS) {
                    NNOPT_ERROR_FMT("Attention[%d]: fused QKV bias_add failed: %d", layer_idx_, eb);
                    return fail_qkv();
                }
                // q/k/v point at sub-buffers of buf_qkv_; downstream RoPE,
                // KV-write, scores, attn_out treat them as before.
                q = sub_q_qkv_;
                k = sub_k_qkv_;
                v = sub_v_qkv_;
                qkv_done = true;
            }
        }
    }
    if (!qkv_done) { /* fall through to unfused 3-call path below */ }

    // ── Q/K/V projections.
    //   This template assumes nn.Linear separate projections (Llama, Mistral, etc.).
    //   If (model metadata) says
    //     weight_key_parent_classes.<key> == "Conv1D"
    //   the parent module is HF Conv1D (weight stored as [in, out], NOT
    //   [out, in]). Replace pytorch_linear → pytorch_conv1d AT EVERY CALL
    //   and adapt the architecture: GPT-2 / GPT-Neo / OPT use a SINGLE fused
    //   c_attn projection [H -> 3H] then split (NOT three separate calls).
    //   Build will REFUSE pytorch_linear() on a Conv1D-stored weight.
    if (!qkv_done) {
        if (!pytorch_linear(queue, seq_q, Q_DIM,  H, hidden_states, wq_, q)) return fail_qkv();
        if (!pytorch_linear(queue, seq_q, KV_DIM, H, hidden_states, wk_, k)) return fail_qkv();
        if (!pytorch_linear(queue, seq_q, KV_DIM, H, hidden_states, wv_, v)) return fail_qkv();
    }

    // Bias add (Qwen2Attention has bias on q/k/v).
    auto bias_add = [&](cl_mem out, cl_mem bias, int rows, int cols) -> bool {
        if (!_set_arg_checked(bias_kernel_, 0, sizeof(cl_mem), &out, "out")) return false;
        if (!_set_arg_checked(bias_kernel_, 1, sizeof(cl_mem), &bias, "bias")) return false;
        if (!_set_arg_checked(bias_kernel_, 2, sizeof(int), &rows, "rows")) return false;
        if (!_set_arg_checked(bias_kernel_, 3, sizeof(int), &cols, "cols")) return false;
        // Vec4 dispatch — kernel emits 4 fp16 per thread. Q_DIM=896, KV_DIM=128
        // both divisible by 4 (config-derived; trivially true for any model
        // where head_dim is a multiple of 4, which is every common transformer).
        size_t gws = ((size_t)rows * (size_t)cols) >> 2;
        cl_int e = nnopt_prof::enqueue(queue, bias_kernel_, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) {
            NNOPT_ERROR_FMT("bias_add_rowmajor dispatch failed: %d", e);
            return false;
        }
        return true;
    };

    if (!qkv_done) {
        if (!bias_add(q, bq_, seq_q, Q_DIM)) return fail_qkv();
        if (!bias_add(k, bk_, seq_q, KV_DIM)) return fail_qkv();
        if (!bias_add(v, bv_, seq_q, KV_DIM)) return fail_qkv();
    }

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
    if (!_set_arg_checked(rope_kernel_, 8, sizeof(cl_mem), &counter, "counter")) return fail_qkv();

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
    //
    // Decode hot path: dispatch a kv_write kernel (recordable). At prefill
    // (seq_q > 1) we still use clEnqueueCopyBuffer because the kernel is
    // single-row-per-launch and would need seq_q dispatches — slower
    // than one big copy for prefill. For decode (seq_q == 1) the kernel
    // path is identical-cost AND is recordable for the recordable_queues
    // integration. Only NDRangeKernel can be inside a cl_qcom_recordable
    // recording.
    if (seq_q == 1) {
        if (!kv_write_kernel(queue, k, k_cache_, counter, KV_DIM)) {
            NNOPT_ERROR_FMT("kv_write k -> k_cache_ kernel failed (layer %d)", layer_idx_);
            return fail_qkv();
        }
        if (!kv_write_kernel(queue, v, v_cache_, counter, KV_DIM)) {
            NNOPT_ERROR_FMT("kv_write v -> v_cache_ kernel failed (layer %d)", layer_idx_);
            return fail_qkv();
        }
    } else {
        const size_t dst_off_bytes = (size_t)start_pos * (size_t)KV_DIM * sizeof(nnopt_storage_t);
        const size_t kv_bytes = (size_t)seq_q * (size_t)KV_DIM * sizeof(nnopt_storage_t);
        err = clEnqueueCopyBuffer(queue, k, k_cache_, 0, dst_off_bytes, kv_bytes, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("copy k -> k_cache_ failed: %d", err); return fail_qkv(); }
        err = clEnqueueCopyBuffer(queue, v, v_cache_, 0, dst_off_bytes, kv_bytes, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("copy v -> v_cache_ failed: %d", err); return fail_qkv(); }
    }

    // K and V buffers are persistent class members — DON'T release them.
    // The cache copies above are byte-blits that complete before any
    // downstream kernel reads from k_cache_/v_cache_ (in-order queue).
    k = nullptr;
    v = nullptr;

    // Reuse persistent buf_scores_ (sized for current seq_q, seq_k).
    cl_mem scores = buf_scores_;

    auto fail_scores = [&]() -> cl_mem {
        return nullptr;
    };

    // gqa_attn_scores — single launch covers all heads.
    if (!_set_arg_checked(scores_kernel_, 0, sizeof(cl_mem), &q, "q"))              return fail_scores();
    if (!_set_arg_checked(scores_kernel_, 1, sizeof(cl_mem), &k_cache_, "k_cache")) return fail_scores();
    if (!_set_arg_checked(scores_kernel_, 2, sizeof(cl_mem), &scores, "scores"))    return fail_scores();
    if (!_set_arg_checked(scores_kernel_, 3, sizeof(int), &seq_q, "seq_q"))         return fail_scores();
    if (!_set_arg_checked(scores_kernel_, 4, sizeof(cl_mem), &counter, "counter"))  return fail_scores();
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
        if (!_set_arg_checked(softmax_kernel_, 0, sizeof(cl_mem), &scores, "scores"))   return fail_scores();
        if (!_set_arg_checked(softmax_kernel_, 1, sizeof(int), &seq_q, "seq_q"))        return fail_scores();
        if (!_set_arg_checked(softmax_kernel_, 2, sizeof(cl_mem), &counter, "counter")) return fail_scores();
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

    // gqa_attn_out — reuse persistent buf_attn_o_.
    cl_mem attn_out = buf_attn_o_;

    auto fail_out = [&]() -> cl_mem {
        return nullptr;
    };

    if (!_set_arg_checked(out_kernel_, 0, sizeof(cl_mem), &scores, "scores"))     return fail_out();
    if (!_set_arg_checked(out_kernel_, 1, sizeof(cl_mem), &v_cache_, "v_cache")) return fail_out();
    if (!_set_arg_checked(out_kernel_, 2, sizeof(cl_mem), &attn_out, "out"))     return fail_out();
    if (!_set_arg_checked(out_kernel_, 3, sizeof(int), &seq_q, "seq_q"))         return fail_out();
    if (!_set_arg_checked(out_kernel_, 4, sizeof(cl_mem), &counter, "counter"))  return fail_out();
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
    //
    // Decode fast path: when seq_q==1 and the caller passed a residual_dest
    // (= hidden state buffer), fuse o_proj + residual_add in one launch via
    // pytorch_linear_add. Saves the element_add_inplace launch in
    // Model::forward. Race-free under the in-order queue: input_layernorm
    // already finished reading hidden, and o_proj's epilogue reads/writes
    // the same locations sequentially per WG.
    if (seq_q == 1 && residual_dest != nullptr &&
        pytorch_linear_add(queue, seq_q, H, Q_DIM, attn_out, wo_, residual_dest)) {
        NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_out_fused", layer_idx_, queue, residual_dest, (size_t)seq_q * H);
        clRetainMemObject(residual_dest);
        return residual_dest;
    }

    cl_mem proj = buf_proj_;

    // Output projection. If Attention.json says
    //   weight_key_parent_classes.c_proj == "Conv1D"
    // (GPT-2 / GPT-Neo / OPT), swap pytorch_linear → pytorch_conv1d here.
    // Build refuses pytorch_linear() on a Conv1D-stored weight.
    if (!pytorch_linear(queue, seq_q, H, Q_DIM, attn_out, wo_, proj)) {
        return nullptr;
    }

    NNOPT_LAYER_CHECK_FMT("block%d_sub_attn_out", layer_idx_, queue, proj, (size_t)seq_q * H);

    // Caller owns the returned cl_mem (will clReleaseMemObject it). Since
    // proj is our persistent buf_proj_, retain before returning so the
    // caller's release just decrements back to our owned reference.
    clRetainMemObject(proj);
    return proj;
}
