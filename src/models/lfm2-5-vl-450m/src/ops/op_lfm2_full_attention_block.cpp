// Reference: transformers/models/lfm2/modeling_lfm2.py Lfm2DecoderLayer.forward + Lfm2Attention.forward
// Lfm2DecoderLayer.forward:
//   residual = hidden_states
//   hidden_states, _ = self.self_attn(hidden_states=self.operator_norm(hidden_states), ...)
//   hidden_states = hidden_states + residual
//   hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
// Lfm2Attention.forward:
//   q = q_layernorm(q_proj(hidden).view(..., heads, head_dim)).transpose(1, 2)
//   k = k_layernorm(k_proj(hidden).view(..., kv_heads, head_dim)).transpose(1, 2)
//   v = v_proj(hidden).view(..., kv_heads, head_dim).transpose(1, 2)
//   q, k = apply_rotary_pos_emb(q, k, cos, sin)
//   attn_output = softmax(q @ repeat_kv(k).T * scale + causal_mask) @ repeat_kv(v)
//   output = out_proj(attn_output.reshape(..., hidden_size))

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "forward_dispatch.h"
#include "model.h"
#include "ops/lfm2_common.h"
#include "utils.h"
#include <CL/cl.h>
#include <algorithm>
#include <cmath>
#include <string>

extern Model* g_active_model_for_vlm_splice;

namespace {
static bool set_arg_local(cl_kernel k, cl_uint idx, size_t sz, const void* v, const char* name) {
    cl_int err = clSetKernelArg(k, idx, sz, v);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("lfm2_attn: clSetKernelArg(%u,%s) failed (%d)", (unsigned)idx, name, (int)err);
        return false;
    }
    return true;
}

static cl_kernel kernel(OpenCLContext& cl_ctx, const char* name) {
    cl_program p = lfm2_program(cl_ctx);
    if (!p) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(p, name, &err);
    if (err != CL_SUCCESS || !k) {
        NNOPT_ERROR_FMT("lfm2_attn: clCreateKernel(%s) failed (%d)", name, (int)err);
        return nullptr;
    }
    return k;
}

static cl_mem head_rms(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                       cl_mem input_heads, int rows, int heads, int head_dim,
                       const std::string& weight_key) {
    cl_mem w = weights.get_buffer(weight_key);
    if (!w) {
        NNOPT_ERROR_FMT("lfm2_attn head_rms: missing %s", weight_key.c_str());
        return nullptr;
    }
    cl_mem out = lfm2_alloc(cl_ctx, (size_t)rows * (size_t)heads * (size_t)head_dim, "head_rms");
    if (!out) return nullptr;
    static cl_kernel k = nullptr;
    if (!k) k = kernel(cl_ctx, "rms_norm_heads");
    if (!k) { clReleaseMemObject(out); return nullptr; }
    const float eps = MODEL_CONFIG::NORM_EPS;
    if (!set_arg_local(k,0,sizeof(cl_mem),&input_heads,"input") ||
        !set_arg_local(k,1,sizeof(cl_mem),&w,"weight") ||
        !set_arg_local(k,2,sizeof(cl_mem),&out,"out") ||
        !set_arg_local(k,3,sizeof(int),&rows,"rows") ||
        !set_arg_local(k,4,sizeof(int),&heads,"heads") ||
        !set_arg_local(k,5,sizeof(int),&head_dim,"head_dim") ||
        !set_arg_local(k,6,sizeof(float),&eps,"eps") ||
        !lfm2_kernel1_lws(queue, k, (size_t)rows * (size_t)heads * 32u, 32u, "lfm2_head_rms")) {
        clReleaseMemObject(out);
        return nullptr;
    }
    return out;
}

static bool seq_to_head_major(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem src, cl_mem dst,
                              int rows, int heads, int head_dim) {
    static cl_kernel k = nullptr;
    if (!k) k = kernel(cl_ctx, "seq_to_heads");
    if (!k) return false;
    return set_arg_local(k,0,sizeof(cl_mem),&src,"src") &&
           set_arg_local(k,1,sizeof(cl_mem),&dst,"dst") &&
           set_arg_local(k,2,sizeof(int),&rows,"rows") &&
           set_arg_local(k,3,sizeof(int),&heads,"heads") &&
           set_arg_local(k,4,sizeof(int),&head_dim,"head_dim") &&
           lfm2_kernel1(queue, k, (size_t)rows * (size_t)heads * (size_t)head_dim, "lfm2_seq_to_heads");
}

static bool head_major_to_seq(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem src, cl_mem dst,
                              int rows, int heads, int head_dim) {
    static cl_kernel k = nullptr;
    if (!k) k = kernel(cl_ctx, "heads_to_seq");
    if (!k) return false;
    return set_arg_local(k,0,sizeof(cl_mem),&src,"src") &&
           set_arg_local(k,1,sizeof(cl_mem),&dst,"dst") &&
           set_arg_local(k,2,sizeof(int),&rows,"rows") &&
           set_arg_local(k,3,sizeof(int),&heads,"heads") &&
           set_arg_local(k,4,sizeof(int),&head_dim,"head_dim") &&
           lfm2_kernel1(queue, k, (size_t)rows * (size_t)heads * (size_t)head_dim, "lfm2_heads_to_seq");
}
} // namespace

extern "C" cl_mem op_lfm2_full_attention_block(OpenCLContext& cl_ctx,
                                               Weights& weights,
                                               cl_command_queue queue,
                                               cl_mem hidden_in,
                                               int seq_len,
                                               int hidden_size,
                                               int layer_idx,
                                               int start_pos) {
    const std::string prefix = "model.language_model.layers." + std::to_string(layer_idx);
    const int q_heads = MODEL_CONFIG::NUM_ATTENTION_HEADS;
    const int kv_heads = MODEL_CONFIG::NUM_KEY_VALUE_HEADS;
    const int head_dim = hidden_size / q_heads;
    const int q_dim = q_heads * head_dim;
    const int kv_dim = kv_heads * head_dim;
    const int n = seq_len * hidden_size;

    cl_mem norm1 = lfm2_rms_norm(cl_ctx, weights, queue, hidden_in, seq_len, hidden_size, prefix + ".operator_norm.weight");
    if (!norm1) return nullptr;

    cl_mem qw = weights.get_buffer(prefix + ".self_attn.q_proj.weight");
    cl_mem kw = weights.get_buffer(prefix + ".self_attn.k_proj.weight");
    cl_mem vw = weights.get_buffer(prefix + ".self_attn.v_proj.weight");
    cl_mem ow = weights.get_buffer(prefix + ".self_attn.out_proj.weight");
    if (!qw || !kw || !vw || !ow) {
        clReleaseMemObject(norm1);
        NNOPT_ERROR_FMT("lfm2_attn: missing projection weights for layer %d", layer_idx);
        return nullptr;
    }

    cl_mem q_seq = lfm2_alloc(cl_ctx, (size_t)seq_len * q_dim, "q_seq");
    cl_mem k_seq = lfm2_alloc(cl_ctx, (size_t)seq_len * kv_dim, "k_seq");
    cl_mem v_seq = lfm2_alloc(cl_ctx, (size_t)seq_len * kv_dim, "v_seq");
    cl_mem q_heads_buf = lfm2_alloc(cl_ctx, (size_t)seq_len * q_dim, "q_heads");
    cl_mem k_heads_buf = lfm2_alloc(cl_ctx, (size_t)seq_len * kv_dim, "k_heads");
    cl_mem v_heads_buf = lfm2_alloc(cl_ctx, (size_t)seq_len * kv_dim, "v_heads");
    if (!q_seq || !k_seq || !v_seq || !q_heads_buf || !k_heads_buf || !v_heads_buf) {
        if (norm1) clReleaseMemObject(norm1); if (q_seq) clReleaseMemObject(q_seq); if (k_seq) clReleaseMemObject(k_seq); if (v_seq) clReleaseMemObject(v_seq); if (q_heads_buf) clReleaseMemObject(q_heads_buf); if (k_heads_buf) clReleaseMemObject(k_heads_buf); if (v_heads_buf) clReleaseMemObject(v_heads_buf);
        return nullptr;
    }
    if (!pytorch_linear(queue, seq_len, q_dim, hidden_size, norm1, qw, q_seq) ||
        !pytorch_linear(queue, seq_len, kv_dim, hidden_size, norm1, kw, k_seq) ||
        !pytorch_linear(queue, seq_len, kv_dim, hidden_size, norm1, vw, v_seq)) {
        clReleaseMemObject(norm1); clReleaseMemObject(q_seq); clReleaseMemObject(k_seq); clReleaseMemObject(v_seq); clReleaseMemObject(q_heads_buf); clReleaseMemObject(k_heads_buf); clReleaseMemObject(v_heads_buf);
        return nullptr;
    }
    if (layer_idx == 2 && start_pos == 0) {
        NNOPT_LAYER_CHECK("block0_sub_self_attn_q_proj_out", queue, q_seq, (size_t)seq_len * q_dim);
        NNOPT_LAYER_CHECK("block0_sub_self_attn_k_proj_out", queue, k_seq, (size_t)seq_len * kv_dim);
        NNOPT_LAYER_CHECK("block0_sub_self_attn_v_proj_out", queue, v_seq, (size_t)seq_len * kv_dim);
    }
    if (!seq_to_head_major(cl_ctx, queue, q_seq, q_heads_buf, seq_len, q_heads, head_dim) ||
        !seq_to_head_major(cl_ctx, queue, k_seq, k_heads_buf, seq_len, kv_heads, head_dim) ||
        !seq_to_head_major(cl_ctx, queue, v_seq, v_heads_buf, seq_len, kv_heads, head_dim)) {
        clReleaseMemObject(norm1); clReleaseMemObject(q_seq); clReleaseMemObject(k_seq); clReleaseMemObject(v_seq); clReleaseMemObject(q_heads_buf); clReleaseMemObject(k_heads_buf); clReleaseMemObject(v_heads_buf);
        return nullptr;
    }
    clReleaseMemObject(q_seq);
    clReleaseMemObject(k_seq);
    clReleaseMemObject(v_seq);

    cl_mem q_norm = head_rms(cl_ctx, weights, queue, q_heads_buf, seq_len, q_heads, head_dim, prefix + ".self_attn.q_layernorm.weight");
    cl_mem k_norm = head_rms(cl_ctx, weights, queue, k_heads_buf, seq_len, kv_heads, head_dim, prefix + ".self_attn.k_layernorm.weight");
    clReleaseMemObject(q_heads_buf);
    clReleaseMemObject(k_heads_buf);
    if (!q_norm || !k_norm) { if (q_norm) clReleaseMemObject(q_norm); if (k_norm) clReleaseMemObject(k_norm); clReleaseMemObject(norm1); clReleaseMemObject(v_heads_buf); return nullptr; }

    static cl_kernel rope_k = nullptr, scores_k = nullptr, softmax_k = nullptr, apply_k = nullptr, kv_write_k = nullptr, flash_k = nullptr, flash_prefill_k = nullptr;
    if (!rope_k) rope_k = kernel(cl_ctx, "apply_rope_lfm2");
    if (!scores_k) scores_k = kernel(cl_ctx, "lfm2_attn_scores");
    if (!softmax_k) softmax_k = kernel(cl_ctx, "lfm2_softmax_rows");
    if (!apply_k) apply_k = kernel(cl_ctx, "lfm2_attn_apply");
    if (!kv_write_k) kv_write_k = kernel(cl_ctx, "kv_cache_write_counter");
    if (!flash_k) flash_k = kernel(cl_ctx, "lfm2_flash_attn_decode");
    if (!flash_prefill_k) flash_prefill_k = kernel(cl_ctx, "lfm2_flash_attn_prefill");
    if (!rope_k || !scores_k || !softmax_k || !apply_k || !kv_write_k || !flash_k || !flash_prefill_k) {
        clReleaseMemObject(norm1); clReleaseMemObject(q_norm); clReleaseMemObject(k_norm); clReleaseMemObject(v_heads_buf); return nullptr;
    }
    float theta = 1000000.0f;
    Model* model_for_counter = g_active_model_for_vlm_splice;
    cl_mem counter_buf = model_for_counter ? model_for_counter->counter_buf() : nullptr;
    if (!counter_buf) {
        NNOPT_ERROR("lfm2_attn: counter_buf not allocated (Model::ensure_caches must be called first)");
        clReleaseMemObject(norm1); clReleaseMemObject(q_norm); clReleaseMemObject(k_norm); clReleaseMemObject(v_heads_buf); return nullptr;
    }
    if (!set_arg_local(rope_k,0,sizeof(cl_mem),&q_norm,"q") ||
        !set_arg_local(rope_k,1,sizeof(cl_mem),&k_norm,"k") ||
        !set_arg_local(rope_k,2,sizeof(int),&seq_len,"rows") ||
        !set_arg_local(rope_k,3,sizeof(int),&q_heads,"q_heads") ||
        !set_arg_local(rope_k,4,sizeof(int),&kv_heads,"kv_heads") ||
        !set_arg_local(rope_k,5,sizeof(int),&head_dim,"head_dim") ||
        !set_arg_local(rope_k,6,sizeof(cl_mem),&counter_buf,"counter") ||
        !set_arg_local(rope_k,7,sizeof(float),&theta,"theta") ||
        !lfm2_kernel1(queue, rope_k, (size_t)std::max(q_heads, kv_heads) * (size_t)seq_len * (size_t)(head_dim / 2), "lfm2_rope")) {
        clReleaseMemObject(norm1); clReleaseMemObject(q_norm); clReleaseMemObject(k_norm); clReleaseMemObject(v_heads_buf); return nullptr;
    }

    // Write new K and V into the persistent KV cache at offset start_pos.
    // K/V cache buffers are [kv_heads, max_kv_seq, head_dim] storage_t.
    Model* model = g_active_model_for_vlm_splice;
    if (!model || !model->caches_ready()) {
        NNOPT_ERROR("lfm2_attn: KV caches not initialized (Model::ensure_caches not called)");
        clReleaseMemObject(norm1); clReleaseMemObject(q_norm); clReleaseMemObject(k_norm); clReleaseMemObject(v_heads_buf); return nullptr;
    }
    const int max_kv = model->max_kv_seq();
    cl_mem k_cache = model->kv_K_cache(layer_idx);
    cl_mem v_cache = model->kv_V_cache(layer_idx);
    if (!k_cache || !v_cache) {
        NNOPT_ERROR_FMT("lfm2_attn: KV cache buffers missing for layer %d", layer_idx);
        clReleaseMemObject(norm1); clReleaseMemObject(q_norm); clReleaseMemObject(k_norm); clReleaseMemObject(v_heads_buf); return nullptr;
    }
    if (start_pos + seq_len > max_kv) {
        NNOPT_ERROR_FMT("lfm2_attn: prompt grew past KV cache capacity (start_pos=%d + seq_len=%d > max=%d)",
                        start_pos, seq_len, max_kv);
        clReleaseMemObject(norm1); clReleaseMemObject(q_norm); clReleaseMemObject(k_norm); clReleaseMemObject(v_heads_buf); return nullptr;
    }
    // K: write k_norm[kv_heads, seq_len, head_dim] → k_cache[kv_heads, max_kv, head_dim] @ counter[0]
    if (!set_arg_local(kv_write_k,0,sizeof(cl_mem),&k_norm,"src") ||
        !set_arg_local(kv_write_k,1,sizeof(cl_mem),&k_cache,"cache") ||
        !set_arg_local(kv_write_k,2,sizeof(int),&seq_len,"new_rows") ||
        !set_arg_local(kv_write_k,3,sizeof(int),&kv_heads,"heads") ||
        !set_arg_local(kv_write_k,4,sizeof(int),&head_dim,"head_dim") ||
        !set_arg_local(kv_write_k,5,sizeof(int),&max_kv,"stride") ||
        !set_arg_local(kv_write_k,6,sizeof(cl_mem),&counter_buf,"counter") ||
        !lfm2_kernel1(queue, kv_write_k, (size_t)kv_heads * (size_t)seq_len * (size_t)head_dim, "kv_cache_write_K")) {
        clReleaseMemObject(norm1); clReleaseMemObject(q_norm); clReleaseMemObject(k_norm); clReleaseMemObject(v_heads_buf); return nullptr;
    }
    // V: write v_heads_buf[kv_heads, seq_len, head_dim] → v_cache @ counter[0]
    if (!set_arg_local(kv_write_k,0,sizeof(cl_mem),&v_heads_buf,"src") ||
        !set_arg_local(kv_write_k,1,sizeof(cl_mem),&v_cache,"cache") ||
        !set_arg_local(kv_write_k,2,sizeof(int),&seq_len,"new_rows") ||
        !set_arg_local(kv_write_k,3,sizeof(int),&kv_heads,"heads") ||
        !set_arg_local(kv_write_k,4,sizeof(int),&head_dim,"head_dim") ||
        !set_arg_local(kv_write_k,5,sizeof(int),&max_kv,"stride") ||
        !set_arg_local(kv_write_k,6,sizeof(cl_mem),&counter_buf,"counter") ||
        !lfm2_kernel1(queue, kv_write_k, (size_t)kv_heads * (size_t)seq_len * (size_t)head_dim, "kv_cache_write_V")) {
        clReleaseMemObject(norm1); clReleaseMemObject(q_norm); clReleaseMemObject(k_norm); clReleaseMemObject(v_heads_buf); return nullptr;
    }
    // k_norm/v_heads_buf are now redundant with k_cache/v_cache for this layer; release.
    clReleaseMemObject(k_norm);
    clReleaseMemObject(v_heads_buf);

    const int q_rows = seq_len;
    const int k_rows = start_pos + seq_len;
    cl_mem out_heads = lfm2_alloc(cl_ctx, (size_t)q_rows * q_dim, "attn_out_heads");
    cl_mem attn_seq = lfm2_alloc(cl_ctx, (size_t)q_rows * q_dim, "attn_seq");
    cl_mem proj = lfm2_alloc(cl_ctx, (size_t)q_rows * hidden_size, "attn_proj");
    if (!out_heads || !attn_seq || !proj) {
        if (out_heads) clReleaseMemObject(out_heads); if (attn_seq) clReleaseMemObject(attn_seq); if (proj) clReleaseMemObject(proj);
        clReleaseMemObject(norm1); clReleaseMemObject(q_norm); return nullptr;
    }
    float scale = 1.0f / std::sqrt((float)head_dim);
    cl_mem scores = nullptr;
    if (q_rows == 1) {
        // DECODE FAST PATH — flash attention (online softmax): one workgroup per
        // q_head, LWS=head_dim=64. No intermediate scores buffer; fuses
        // scores+softmax+apply in one kernel.
        if (!set_arg_local(flash_k,0,sizeof(cl_mem),&q_norm,"q") ||
            !set_arg_local(flash_k,1,sizeof(cl_mem),&k_cache,"k") ||
            !set_arg_local(flash_k,2,sizeof(cl_mem),&v_cache,"v") ||
            !set_arg_local(flash_k,3,sizeof(cl_mem),&out_heads,"out_heads") ||
            !set_arg_local(flash_k,4,sizeof(cl_mem),&counter_buf,"counter") ||
            !set_arg_local(flash_k,5,sizeof(int),&max_kv,"k_stride") ||
            !set_arg_local(flash_k,6,sizeof(int),&q_heads,"q_heads") ||
            !set_arg_local(flash_k,7,sizeof(int),&kv_heads,"kv_heads") ||
            !set_arg_local(flash_k,8,sizeof(int),&head_dim,"head_dim") ||
            !set_arg_local(flash_k,9,sizeof(float),&scale,"scale") ||
            !lfm2_kernel1_lws(queue, flash_k, (size_t)q_heads * (size_t)head_dim, (size_t)head_dim, "lfm2_flash_attn_decode")) {
            clReleaseMemObject(norm1); clReleaseMemObject(q_norm); clReleaseMemObject(out_heads); clReleaseMemObject(attn_seq); clReleaseMemObject(proj); return nullptr;
        }
    } else {
        // PREFILL PATH — tiled flash attention with online softmax (BQ=4).
        // Fuses scores + softmax + apply into one kernel, eliminates the
        // materialized scores buffer (~105 MB for 1806 tokens).
        const int fa_bq = 4;
        const int num_q_tiles = (q_rows + fa_bq - 1) / fa_bq;
        size_t fa_global = (size_t)q_heads * (size_t)num_q_tiles * 64u;
        size_t fa_local = 64u;
        if (!set_arg_local(flash_prefill_k,0,sizeof(cl_mem),&q_norm,"q") ||
            !set_arg_local(flash_prefill_k,1,sizeof(cl_mem),&k_cache,"k") ||
            !set_arg_local(flash_prefill_k,2,sizeof(cl_mem),&v_cache,"v") ||
            !set_arg_local(flash_prefill_k,3,sizeof(cl_mem),&out_heads,"out") ||
            !set_arg_local(flash_prefill_k,4,sizeof(int),&q_rows,"q_rows") ||
            !set_arg_local(flash_prefill_k,5,sizeof(int),&k_rows,"k_rows") ||
            !set_arg_local(flash_prefill_k,6,sizeof(int),&max_kv,"k_stride") ||
            !set_arg_local(flash_prefill_k,7,sizeof(int),&q_heads,"q_heads") ||
            !set_arg_local(flash_prefill_k,8,sizeof(int),&kv_heads,"kv_heads") ||
            !set_arg_local(flash_prefill_k,9,sizeof(int),&head_dim,"head_dim") ||
            !set_arg_local(flash_prefill_k,10,sizeof(int),&start_pos,"start_pos") ||
            !set_arg_local(flash_prefill_k,11,sizeof(float),&scale,"scale") ||
            !lfm2_kernel1_lws(queue, flash_prefill_k, fa_global, fa_local, "lfm2_flash_attn_prefill")) {
            clReleaseMemObject(norm1); clReleaseMemObject(q_norm); clReleaseMemObject(out_heads); clReleaseMemObject(attn_seq); clReleaseMemObject(proj); return nullptr;
        }
    }
    if (!head_major_to_seq(cl_ctx, queue, out_heads, attn_seq, q_rows, q_heads, head_dim)) {
        clReleaseMemObject(norm1); clReleaseMemObject(q_norm); if (scores) clReleaseMemObject(scores); clReleaseMemObject(out_heads); clReleaseMemObject(attn_seq); clReleaseMemObject(proj); return nullptr;
    }
    if (layer_idx == 2 && start_pos == 0) NNOPT_LAYER_CHECK("block0_sub_self_attn_out_proj_in", queue, attn_seq, (size_t)q_rows * q_dim);
    if (!pytorch_linear(queue, q_rows, hidden_size, q_dim, attn_seq, ow, proj)) {
        clReleaseMemObject(norm1); clReleaseMemObject(q_norm); if (scores) clReleaseMemObject(scores); clReleaseMemObject(out_heads); clReleaseMemObject(attn_seq); clReleaseMemObject(proj); return nullptr;
    }
    if (layer_idx == 2 && start_pos == 0) NNOPT_LAYER_CHECK("block0_sub_self_attn_out_proj_out", queue, proj, (size_t)q_rows * (size_t)hidden_size);
    clReleaseMemObject(norm1); clReleaseMemObject(q_norm); if (scores) clReleaseMemObject(scores); clReleaseMemObject(out_heads); clReleaseMemObject(attn_seq);

    // Fused: after_op = hidden_in + proj ; norm2 = rms_norm(after_op) * ffn_norm_weight.
    // Saves one kernel dispatch per layer × 16 layers = 16 saves per decode token.
    cl_mem after_op = nullptr, norm2 = nullptr;
    bool fused_ok = lfm2_rms_norm_add(cl_ctx, weights, queue, hidden_in, proj,
                                       seq_len, hidden_size, prefix + ".ffn_norm.weight",
                                       &after_op, &norm2);
    clReleaseMemObject(proj);
    if (!fused_ok) return nullptr;
    NNOPT_LAYER_CHECK_FMT("lfm2_operator_%d", layer_idx, queue, after_op, (size_t)n);
    cl_mem mlp = lfm2_mlp(cl_ctx, weights, queue, norm2, seq_len, hidden_size, MODEL_CONFIG::INTERMEDIATE_SIZE, prefix, layer_idx);
    clReleaseMemObject(norm2);
    if (!mlp) { clReleaseMemObject(after_op); return nullptr; }
    if (!element_add_inplace(queue, lfm2_program(cl_ctx), after_op, mlp, (size_t)n)) {
        clReleaseMemObject(after_op); clReleaseMemObject(mlp); return nullptr;
    }
    clReleaseMemObject(mlp);
    NNOPT_LAYER_CHECK_FMT("lfm2_layer_%d", layer_idx, queue, after_op, (size_t)n);
    return after_op;
}
