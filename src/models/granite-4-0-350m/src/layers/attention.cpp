// Reference: model_info/transformers_src/modeling_granitemoehybrid.py:121-191 GraniteMoeHybridAttention.forward

#include "layers/attention.h"
#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "model_config.h"
#include "kernel_profiler.h"

#include <clblast.h>
#include <cmath>
#include <cstdio>
#include <string>

namespace {

std::string fmt_key(const std::string& templ, int i) {
    std::string out = templ;
    size_t pos = out.find("{i}");
    if (pos != std::string::npos) out.replace(pos, 3, std::to_string(i));
    return out;
}

bool set_arg_checked(cl_kernel k, cl_uint idx, size_t sz, const void* val, const char* name) {
    cl_int err = clSetKernelArg(k, idx, sz, val);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("clSetKernelArg(%s) idx=%u failed (%d)", name, (unsigned)idx, err);
        return false;
    }
    return true;
}

}  // namespace

Attention::Attention(OpenCLContext& cl_ctx, Weights& weights, int layer_idx)
    : cl_ctx_(cl_ctx), weights_(weights), layer_idx_(layer_idx) {}

Attention::~Attention() {
    if (rope_kernel_) clReleaseKernel(rope_kernel_);
    if (gqa_scores_kernel_) clReleaseKernel(gqa_scores_kernel_);
    if (gqa_softmax_kernel_) clReleaseKernel(gqa_softmax_kernel_);
    if (gqa_out_kernel_) clReleaseKernel(gqa_out_kernel_);
    if (program_) clReleaseProgram(program_);
    if (k_cache_) clReleaseMemObject(k_cache_);
    if (v_cache_) clReleaseMemObject(v_cache_);
    if (attn_scores_) clReleaseMemObject(attn_scores_);
    if (attn_ctx_)    clReleaseMemObject(attn_ctx_);
}

// Geometric-growth scratch buffer ensure: realloc only if the requested size
// exceeds current capacity, then double. Avoids the per-call clCreateBuffer
// in attention's hot path.
static bool ensure_scratch_buffer(cl_context ctx, cl_mem& buf, size_t& cap_bytes,
                                  size_t needed_bytes, const char* name, int layer_idx) {
    if (buf && cap_bytes >= needed_bytes) return true;
    if (buf) {
        clReleaseMemObject(buf);
        buf = nullptr;
    }
    size_t new_cap = (cap_bytes == 0) ? needed_bytes : (cap_bytes * 2);
    while (new_cap < needed_bytes) new_cap *= 2;
    cl_int err = CL_SUCCESS;
    buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, new_cap, nullptr, &err);
    if (err != CL_SUCCESS || !buf) {
        NNOPT_ERROR_FMT("Attention[%d]: scratch '%s' clCreateBuffer(%zu B) failed (%d)",
                        layer_idx, name, new_cap, err);
        cap_bytes = 0;
        buf = nullptr;
        return false;
    }
    cap_bytes = new_cap;
    return true;
}

bool Attention::initialize() {
    cl_int err = CL_SUCCESS;

    // Weights
    const std::string kq = fmt_key("model.layers.{i}.self_attn.q_proj.weight", layer_idx_);
    const std::string kk = fmt_key("model.layers.{i}.self_attn.k_proj.weight", layer_idx_);
    const std::string kv = fmt_key("model.layers.{i}.self_attn.v_proj.weight", layer_idx_);
    const std::string ko = fmt_key("model.layers.{i}.self_attn.o_proj.weight", layer_idx_);

    q_w_ = weights_.get_buffer(kq);
    k_w_ = weights_.get_buffer(kk);
    v_w_ = weights_.get_buffer(kv);
    o_w_ = weights_.get_buffer(ko);
    if (!q_w_ || !k_w_ || !v_w_ || !o_w_) {
        NNOPT_ERROR_FMT("Attention[%d]: missing weights", layer_idx_);
        return false;
    }

    program_ = cl_ctx_.build_program_from_file("kernels/attention.cl");
    if (!program_) return false;

    rope_kernel_ = clCreateKernel(program_, "rope_apply_qk", &err);
    if (err != CL_SUCCESS || !rope_kernel_) {
        NNOPT_ERROR_FMT("clCreateKernel(rope_apply_qk) failed (%d)", err);
        return false;
    }

    gqa_scores_kernel_ = clCreateKernel(program_, "gqa_attn_scores", &err);
    if (err != CL_SUCCESS || !gqa_scores_kernel_) {
        NNOPT_ERROR_FMT("clCreateKernel(gqa_attn_scores) failed (%d)", err);
        return false;
    }
    gqa_softmax_kernel_ = clCreateKernel(program_, "gqa_softmax", &err);
    if (err != CL_SUCCESS || !gqa_softmax_kernel_) {
        NNOPT_ERROR_FMT("clCreateKernel(gqa_softmax) failed (%d)", err);
        return false;
    }
    gqa_out_kernel_ = clCreateKernel(program_, "gqa_attn_out", &err);
    if (err != CL_SUCCESS || !gqa_out_kernel_) {
        NNOPT_ERROR_FMT("clCreateKernel(gqa_attn_out) failed (%d)", err);
        return false;
    }

    // KV cache
    const size_t kv_elems =
        (size_t)MODEL_CONFIG::MAX_SEQ_LEN * (size_t)MODEL_CONFIG::NUM_KV_HEADS * (size_t)MODEL_CONFIG::HEAD_DIM;
    k_cache_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, kv_elems * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !k_cache_) {
        NNOPT_ERROR_FMT("clCreateBuffer(k_cache) failed (%d)", err);
        return false;
    }
    v_cache_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, kv_elems * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !v_cache_) {
        NNOPT_ERROR_FMT("clCreateBuffer(v_cache) failed (%d)", err);
        return false;
    }

    return true;
}

cl_mem Attention::forward(cl_command_queue queue, cl_mem input, int seq_len, int start_pos, cl_mem cos, cl_mem sin) {
    NNOPT_LAYER_CHECK_INPUT("attn", queue, input, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

    const int q_heads = MODEL_CONFIG::NUM_HEADS;
    const int kv_heads = MODEL_CONFIG::NUM_KV_HEADS;
    const int head_dim = MODEL_CONFIG::HEAD_DIM;
    const int q_dim = q_heads * head_dim;
    const int kv_dim = kv_heads * head_dim;

    cl_int err = CL_SUCCESS;

    cl_mem q = nullptr, k = nullptr, v = nullptr;
    cl_mem out = nullptr;

    auto fail = [&]() -> cl_mem {
        if (q) clReleaseMemObject(q);
        if (k) clReleaseMemObject(k);
        if (v) clReleaseMemObject(v);
        // attn_scores_, attn_ctx_ are persistent — do NOT release on failure.
        if (out) clReleaseMemObject(out);
        return nullptr;
    };

    q = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                       (size_t)seq_len * (size_t)q_dim * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !q) {
        NNOPT_ERROR_FMT("Attention[%d]: create q failed (%d)", layer_idx_, err);
        return fail();
    }
    k = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                       (size_t)seq_len * (size_t)kv_dim * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !k) {
        NNOPT_ERROR_FMT("Attention[%d]: create k failed (%d)", layer_idx_, err);
        return fail();
    }
    v = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                       (size_t)seq_len * (size_t)kv_dim * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !v) {
        NNOPT_ERROR_FMT("Attention[%d]: create v failed (%d)", layer_idx_, err);
        return fail();
    }

    // Step #4: fused QKV at M=1, K=1024 (single dispatch instead of 3).
    // Falls back to per-projection pytorch_linear for prefill (M>1).
    if (seq_len == 1 && MODEL_CONFIG::HIDDEN_SIZE == 1024 &&
        try_fused_qkv_gemv_m1_k1024(queue, input, q_w_, k_w_, v_w_, q, k, v, q_dim, kv_dim)) {
        // fused QKV dispatched in one go
    } else {
        if (!pytorch_linear(queue, seq_len, q_dim, MODEL_CONFIG::HIDDEN_SIZE, input, q_w_, q)) return fail();
        if (!pytorch_linear(queue, seq_len, kv_dim, MODEL_CONFIG::HIDDEN_SIZE, input, k_w_, k)) return fail();
        if (!pytorch_linear(queue, seq_len, kv_dim, MODEL_CONFIG::HIDDEN_SIZE, input, v_w_, v)) return fail();
    }

    // Sub-op instrumentation for block0 (matches reference dumps captured from HF GraniteMoeHybridAttention.forward).
    if (layer_idx_ == 0) {
        NNOPT_LAYER_CHECK("block0_sub_attn_query_states", queue, q, (size_t)seq_len * (size_t)q_dim);
        NNOPT_LAYER_CHECK("block0_sub_attn_key_states", queue, k, (size_t)seq_len * (size_t)kv_dim);
        NNOPT_LAYER_CHECK("block0_sub_attn_value_states", queue, v, (size_t)seq_len * (size_t)kv_dim);
    }

    // Apply RoPE (in-place) then write K/V into cache at [start_pos, start_pos+seq_len)
    // cos/sin are provided by the Embedding layer.
    if (!cos || !sin) {
        NNOPT_ERROR_FMT("Attention[%d]: cos/sin buffers are null", layer_idx_);
        return fail();
    }

    // Layout expected by rope_apply_qk: q=[seq_q,q_heads,head_dim], k=[seq_q,kv_heads,head_dim]
    // Our projections are [seq_q, q_dim] and [seq_q, kv_dim], which is the same contiguous layout.
    // Each work-item handles one (t, hq, d) — no local memory or cross-WI cooperation,
    // so we let the driver pick lws (passing nullptr is correct for non-cooperative kernels).
    {
        const int seq_q = seq_len;
        const int start = start_pos;

        if (!set_arg_checked(rope_kernel_, 0, sizeof(cl_mem), &q, "q")) return fail();
        if (!set_arg_checked(rope_kernel_, 1, sizeof(cl_mem), &k, "k")) return fail();
        if (!set_arg_checked(rope_kernel_, 2, sizeof(cl_mem), &cos, "cos")) return fail();
        if (!set_arg_checked(rope_kernel_, 3, sizeof(cl_mem), &sin, "sin")) return fail();
        if (!set_arg_checked(rope_kernel_, 4, sizeof(int), &seq_q, "seq_q")) return fail();
        if (!set_arg_checked(rope_kernel_, 5, sizeof(int), &start, "start_pos")) return fail();
        if (!set_arg_checked(rope_kernel_, 6, sizeof(int), &q_heads, "num_q_heads")) return fail();
        if (!set_arg_checked(rope_kernel_, 7, sizeof(int), &kv_heads, "num_kv_heads")) return fail();
        if (!set_arg_checked(rope_kernel_, 8, sizeof(int), &head_dim, "head_dim")) return fail();

        size_t gws_rope = (size_t)seq_q * (size_t)q_heads * (size_t)head_dim;
        err = clEnqueueNDRangeKernel(queue, rope_kernel_, 1, nullptr, &gws_rope, nullptr, 0, nullptr, KernelProfiler::event_for("rope_apply_qk"));
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("Attention[%d]: rope_apply_qk dispatch failed (%d)", layer_idx_, err);
            return fail();
        }
    }

    if (layer_idx_ == 0) {
        // Same tensors as above, but after in-place RoPE.
        NNOPT_LAYER_CHECK("block0_sub_attn_query_states_rope", queue, q, (size_t)seq_len * (size_t)q_dim);
        NNOPT_LAYER_CHECK("block0_sub_attn_key_states_rope", queue, k, (size_t)seq_len * (size_t)kv_dim);
    }

    const size_t kv_row_elems = (size_t)kv_heads * (size_t)head_dim;
    const size_t kv_row_bytes = kv_row_elems * sizeof(nnopt_storage_t);
    const size_t kv_off_bytes = (size_t)start_pos * kv_row_bytes;

    err = clEnqueueCopyBuffer(queue, k, k_cache_, 0, kv_off_bytes, (size_t)seq_len * kv_row_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Attention[%d]: copy k->k_cache failed (%d)", layer_idx_, err);
        return fail();
    }
    err = clEnqueueCopyBuffer(queue, v, v_cache_, 0, kv_off_bytes, (size_t)seq_len * kv_row_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Attention[%d]: copy v->v_cache failed (%d)", layer_idx_, err);
        return fail();
    }

    const int seq_k = start_pos + seq_len;

    // scores/context in storage_t (attention.cl uses storage_t).
    // Persistent across forward calls — geometric-growth allocator avoids the
    // per-call clCreateBuffer that was eating ~2.3 ms each on Adreno.
    const size_t scores_bytes = (size_t)seq_len * (size_t)q_heads * (size_t)seq_k * sizeof(nnopt_storage_t);
    const size_t ctx_bytes    = (size_t)seq_len * (size_t)q_heads * (size_t)head_dim * sizeof(nnopt_storage_t);
    if (!ensure_scratch_buffer(cl_ctx_.context(), attn_scores_, attn_scores_cap_bytes_, scores_bytes, "attn_scores", layer_idx_)) return fail();
    if (!ensure_scratch_buffer(cl_ctx_.context(), attn_ctx_,    attn_ctx_cap_bytes_,    ctx_bytes,    "attn_ctx",    layer_idx_)) return fail();
    cl_mem attn_scores = attn_scores_;
    cl_mem attn_ctx    = attn_ctx_;

    // GraniteMoeHybridAttention sets self.scaling = config.attention_multiplier (line 130).
    // HF eager_attention_forward applies it as: attn_weights = (Q @ K^T) * scaling.
    // It does NOT also multiply by 1/sqrt(head_dim) — attention_multiplier IS the scale.
    // For Granite 4.0 350m: attention_multiplier = 0.015625 (= 1/64 = 1/HEAD_DIM, which
    // is muP scaling, not the standard 1/sqrt(d_k) = 1/8).
    const float scale = MODEL_CONFIG::ATTENTION_MULTIPLIER;

    if (!set_arg_checked(gqa_scores_kernel_, 0, sizeof(cl_mem), &q, "q")) return fail();
    if (!set_arg_checked(gqa_scores_kernel_, 1, sizeof(cl_mem), &k_cache_, "k_cache")) return fail();
    if (!set_arg_checked(gqa_scores_kernel_, 2, sizeof(cl_mem), &attn_scores, "scores")) return fail();
    if (!set_arg_checked(gqa_scores_kernel_, 3, sizeof(int), &seq_len, "seq_q")) return fail();
    if (!set_arg_checked(gqa_scores_kernel_, 4, sizeof(int), &seq_k, "seq_k")) return fail();
    if (!set_arg_checked(gqa_scores_kernel_, 5, sizeof(int), &q_heads, "q_heads")) return fail();
    if (!set_arg_checked(gqa_scores_kernel_, 6, sizeof(int), &kv_heads, "kv_heads")) return fail();
    if (!set_arg_checked(gqa_scores_kernel_, 7, sizeof(int), &head_dim, "head_dim")) return fail();
    if (!set_arg_checked(gqa_scores_kernel_, 8, sizeof(float), &scale, "scale")) return fail();

    size_t gws_scores[3] = {(size_t)seq_k, (size_t)q_heads, (size_t)seq_len};
    err = clEnqueueNDRangeKernel(queue, gqa_scores_kernel_, 3, nullptr, gws_scores, nullptr, 0, nullptr, KernelProfiler::event_for("gqa_attn_scores"));
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Attention[%d]: gqa_attn_scores dispatch failed (%d)", layer_idx_, err);
        return fail();
    }

    // Softmax in-place over seq_k
    if (!set_arg_checked(gqa_softmax_kernel_, 0, sizeof(cl_mem), &attn_scores, "scores")) return fail();
    if (!set_arg_checked(gqa_softmax_kernel_, 1, sizeof(int), &seq_len, "seq_q")) return fail();
    if (!set_arg_checked(gqa_softmax_kernel_, 2, sizeof(int), &seq_k, "seq_k")) return fail();
    if (!set_arg_checked(gqa_softmax_kernel_, 3, sizeof(int), &q_heads, "q_heads")) return fail();

    size_t gws_sm = (size_t)seq_len * (size_t)q_heads;
    err = clEnqueueNDRangeKernel(queue, gqa_softmax_kernel_, 1, nullptr, &gws_sm, nullptr, 0, nullptr, KernelProfiler::event_for("gqa_softmax"));
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Attention[%d]: gqa_softmax dispatch failed (%d)", layer_idx_, err);
        return fail();
    }

    // Output context
    if (!set_arg_checked(gqa_out_kernel_, 0, sizeof(cl_mem), &attn_scores, "probs")) return fail();
    if (!set_arg_checked(gqa_out_kernel_, 1, sizeof(cl_mem), &v_cache_, "v_cache")) return fail();
    if (!set_arg_checked(gqa_out_kernel_, 2, sizeof(cl_mem), &attn_ctx, "ctx")) return fail();
    if (!set_arg_checked(gqa_out_kernel_, 3, sizeof(int), &seq_len, "seq_q")) return fail();
    if (!set_arg_checked(gqa_out_kernel_, 4, sizeof(int), &seq_k, "seq_k")) return fail();
    if (!set_arg_checked(gqa_out_kernel_, 5, sizeof(int), &q_heads, "q_heads")) return fail();
    if (!set_arg_checked(gqa_out_kernel_, 6, sizeof(int), &kv_heads, "kv_heads")) return fail();
    if (!set_arg_checked(gqa_out_kernel_, 7, sizeof(int), &head_dim, "head_dim")) return fail();

    size_t gws_out[3] = {(size_t)head_dim, (size_t)q_heads, (size_t)seq_len};
    err = clEnqueueNDRangeKernel(queue, gqa_out_kernel_, 3, nullptr, gws_out, nullptr, 0, nullptr, KernelProfiler::event_for("gqa_attn_out"));
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Attention[%d]: gqa_attn_out dispatch failed (%d)", layer_idx_, err);
        return fail();
    }

    // Project back: reshape [seq, q_heads, head_dim] -> [seq, q_dim] (already contiguous)
    out = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                         (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out) return fail();

    if (layer_idx_ == 0) {
        NNOPT_LAYER_CHECK("block0_sub_attn_attn_output", queue, attn_ctx, (size_t)seq_len * (size_t)q_dim);
    }

    if (!pytorch_linear(queue, seq_len, MODEL_CONFIG::HIDDEN_SIZE, q_dim, attn_ctx, o_w_, out)) return fail();

    // cleanup intermediates, keep out
    clReleaseMemObject(q);
    clReleaseMemObject(k);
    clReleaseMemObject(v);
    // attn_scores / attn_ctx are persistent (member); do NOT release here.

    NNOPT_LAYER_CHECK_FMT("attn_%d", layer_idx_, queue, out, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
    return out;
}
