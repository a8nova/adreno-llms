// Auto-generated graph-mode backbone for LiquidAI/LFM2.5-VL-450M.
//
// THIS IS A STARTING SKELETON. The agent's job is to replace the zero-logits
// stub below with real op_<class>(...) calls from src/forward_dispatch.h.
//
// Roadmap (execution order from .nnport/graph.json):

//
// Per-class instance counts:

//
// Canonical weight-key prefix: (none) — graph.json keys match weights/model.meta.json verbatim.
//
// The function below COMPILES AND LINKS as-is (so Build → Deploy → Infer
// succeeds end-to-end on a fresh scaffold). It returns zero-filled logits,
// which means the sampler will pick token 0 every step until the agent
// replaces the body with real ops. That is the SIGNAL that backbone is
// the next thing to port — not a build-system bug.
//
// WORKFLOW:
//   1. PortNode(project_dir) returns one node at a time with its class,
//      dump_name, weight_keys, AND canonical_weight_keys (the strings to
//      actually pass into weights.get_buffer()).
//   2. After porting each primitive op, append the op_<class>() call to
//      the block-loop below in the EXACT execution order shown above.
//   3. Wrap each call with NNOPT_LAYER_CHECK("<dump_name>", queue, out, n)
//      using the dump_name from PortNode (NOT the node id — they differ).
//   4. The lm_head call at the bottom is structured for tied-embedding ports
//      (BLOOM, LLaMA-family with tie_word_embeddings=true). For untied
//      heads, swap word_embeddings_weight for lm_head_weight.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "forward_dispatch.h"
#include "model.h"
#include "ops/lfm2_common.h"
#include "utils.h"

#include <CL/cl.h>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

extern Model* g_active_model_for_vlm_splice;

namespace {

static cl_mem upload_input_ids(OpenCLContext& cl_ctx,
                               cl_command_queue queue,
                               const std::vector<int32_t>& ids) {
    if (ids.empty()) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_context ctx = cl_ctx.context();
    cl_mem buf = clCreateBuffer(ctx,
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                ids.size() * sizeof(int32_t),
                                const_cast<int32_t*>(ids.data()),
                                &err);
    if (err != CL_SUCCESS || !buf) {
        NNOPT_ERROR_FMT("upload_input_ids: clCreateBuffer failed (%d)", (int)err);
        return nullptr;
    }
    (void)queue;
    return buf;
}

static bool apply_vlm_masked_scatter(cl_command_queue queue,
                                     cl_mem text_embeds,
                                     int seq_len,
                                     int hidden,
                                     const std::vector<int32_t>& input_ids) {
    // Reference: modeling_lfm2_vl.py Lfm2VlModel.forward:
    //   special_image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    //   inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
    // Semantics: for each position i where input_ids[i] == IMAGE_TOKEN_ID (396),
    // copy image_features[k++] -> inputs_embeds[i, :]. k is a flat counter.
    // The total count of <image> tokens MUST equal image_features rows.
    Model* model = g_active_model_for_vlm_splice;
    if (!model || !model->has_image() || !text_embeds) return true;

    const cl_mem image = model->image_features_buf();
    const int image_N = model->image_features_N();
    if (!image || image_N <= 0) return true;

    // Count <image> placeholders in input_ids
    int n_placeholders = 0;
    for (int i = 0; i < seq_len; ++i) {
        if (input_ids[(size_t)i] == MODEL_CONFIG::IMAGE_TOKEN_ID) n_placeholders++;
    }
    if (n_placeholders == 0) return true;

    if (n_placeholders != image_N) {
        NNOPT_ERROR_FMT("VLM masked_scatter: %d <image> placeholders != %d image_features rows",
                        n_placeholders, image_N);
        return false;
    }

    const size_t row_bytes = (size_t)hidden * sizeof(nnopt_storage_t);
    int k = 0;
    for (int i = 0; i < seq_len; ++i) {
        if (input_ids[(size_t)i] != MODEL_CONFIG::IMAGE_TOKEN_ID) continue;
        cl_int err = clEnqueueCopyBuffer(queue, image, text_embeds,
                                         (size_t)k * row_bytes,
                                         (size_t)i * row_bytes,
                                         row_bytes, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("VLM masked_scatter: copy feature %d -> token %d failed (%d)",
                            k, i, (int)err);
            return false;
        }
        k++;
    }
    NNOPT_DEBUG_SYNC(queue);
    return true;
}

}  // namespace

std::vector<float> model_forward_graph(OpenCLContext& cl_ctx,
                                       Weights& weights,
                                       const std::vector<int32_t>& input_ids,
                                       int start_pos) {
    NNOPT_CHECKPOINT("model_forward_graph entry (graph-mode backbone)");
    const bool is_decode = (start_pos > 0);
    if (is_decode) { lfm2_pool_reclaim(); lfm2_pool_set_active(true); }
    else           { lfm2_pool_set_active(false); }

    cl_command_queue queue = cl_ctx.queue();
    if (!queue) {
        NNOPT_ERROR("model_forward_graph: cl_ctx.queue() returned null");
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    }

    const int seq_len = (int)input_ids.size();
    const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
    const int vocab = MODEL_CONFIG::VOCAB_SIZE;
    (void)hidden;
    (void)vocab;

    cl_mem ids_buf = upload_input_ids(cl_ctx, queue, input_ids);
    if (!ids_buf) {
        NNOPT_ERROR("model_forward_graph: ids_buf null");
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    }

    // Lazily allocate KV + conv caches sized for the first call's full prompt
    // plus headroom for the remaining decode budget. Subsequent decode calls
    // are no-ops here (ensure_caches returns early when already sized).
    {
        Model* m = g_active_model_for_vlm_splice;
        if (m) {
            const int requested = (start_pos == 0)
                ? (seq_len + 256)                // prefill: reserve 256 decode slots
                : (start_pos + seq_len);          // decode: must already fit
            if (!m->ensure_caches(requested)) {
                clReleaseMemObject(ids_buf);
                NNOPT_ERROR("model_forward_graph: Model::ensure_caches failed");
                return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
            }
            // Track 5 — write {start_pos, k_rows} into the persistent counter
            // buffer that the rope / kv_write / flash_attn_decode /
            // depthwise_conv3 kernels now read from. CL_FALSE — in-order queue
            // serializes against the kernels below.
            const int k_rows = start_pos + seq_len;
            if (!m->update_counter(queue, start_pos, k_rows)) {
                clReleaseMemObject(ids_buf);
                NNOPT_ERROR("model_forward_graph: Model::update_counter failed");
                return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
            }
        }
    }

    // ── Token embedding (auto-emitted; agent may edit weight key if PortNode shows otherwise) ──
    // Contract weight_prefix: "model.language_model."  →  guessed key: "model.language_model.embed_tokens.weight"
    cl_mem x = op_Embedding(cl_ctx, weights, queue,
                            ids_buf,
                            /*num_tokens=*/seq_len,
                            /*hidden_size=*/hidden,
                            "model.language_model.embed_tokens.weight");
    if (!x) {
        clReleaseMemObject(ids_buf);
        NNOPT_ERROR("model_forward_graph: op_Embedding returned null (check weight key: model.language_model.embed_tokens.weight)");
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    }
    NNOPT_LAYER_CHECK("token_embeddings", queue, x, (size_t)seq_len * (size_t)hidden);

    // Diagnostic override: if PyTorch-computed merged inputs_embeds were
    // loaded (reference/inputs_embeds.bin), splat them into the first N rows
    // of x and SKIP the on-device vision pipeline's masked_scatter. The
    // remaining positions [N, seq_len) keep their op_Embedding output, which
    // is correct for the text tokens generated during decode (those positions
    // never contained image placeholders in the reference).
    Model* active_model = g_active_model_for_vlm_splice;
    const bool use_ref_embeds = (start_pos == 0)
        && active_model
        && active_model->has_reference_inputs_embeds();
    if (use_ref_embeds) {
        const int ref_N = active_model->reference_inputs_embeds_N();
        const int n_override = (ref_N < seq_len) ? ref_N : seq_len;
        const size_t row_bytes = (size_t)hidden * sizeof(nnopt_storage_t);
        cl_int copy_err = clEnqueueCopyBuffer(queue,
                                              active_model->reference_inputs_embeds_buf(),
                                              x,
                                              0, 0,
                                              (size_t)n_override * row_bytes,
                                              0, nullptr, nullptr);
        if (copy_err != CL_SUCCESS) {
            clReleaseMemObject(ids_buf);
            clReleaseMemObject(x);
            NNOPT_ERROR_FMT("model_forward_graph: reference inputs_embeds copy failed (%d)", (int)copy_err);
            return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
        }
        NNOPT_CHECKPOINT_FMT("model_forward_graph: overrode first %d rows of inputs_embeds from reference/inputs_embeds.bin (vision pipeline + masked_scatter bypassed for prefill)",
                             n_override);
    } else if (start_pos == 0 && !apply_vlm_masked_scatter(queue, x, seq_len, hidden, input_ids)) {
        clReleaseMemObject(ids_buf);
        clReleaseMemObject(x);
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    }
    NNOPT_LAYER_CHECK("vlm_inputs_embeds", queue, x, (size_t)seq_len * (size_t)hidden);

    // ── Per-layer decoder dispatch (AUTO-EMITTED from contract.submodels[lm].layer_kinds) ──
    // Hybrid model: 16 layers, kinds present = conv, full_attention
    // // 0=conv  // 1=full_attention
    //
    // Each op_<kind>_block(...) is declared in src/forward_dispatch.h and must
    // be implemented in src/ops/op_lfm2_conv_block.cpp
    // (and 1 sibling file(s) for other kinds). The scaffold emitted empty stubs
    // that return nullptr with a "TODO: implement me" message — fill in each
    // body from model_info/transformers_src/modeling_lfm2*.py.
    //
    // DO NOT replace this loop with an identity passthrough — the contract
    // declares 16 decoder layers and the on-device transcript
    // will be empty if these block-fns are not implemented.
    static const int LAYER_KINDS[16] = {
        0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0,
        1, 0, 1, 0
    };
    for (int i = 0; i < 16; ++i) {
        // Yield the GPU per layer during the heavy multi-second prefill (seq_len>1)
        // so the UI thread doesn't starve → ANR. Skip it on single-token decode
        // steps, which are short and where the per-layer sleep would just add latency.
        if (seq_len > 1) cl_ctx.yield_for_compositor();
        cl_mem next = nullptr;
        switch (LAYER_KINDS[i]) {
            case 0:  // conv
                next = op_lfm2_conv_block(cl_ctx, weights, queue, x,
                                    seq_len, hidden, i, start_pos);
                break;
            case 1:  // full_attention
                next = op_lfm2_full_attention_block(cl_ctx, weights, queue, x,
                                    seq_len, hidden, i, start_pos);
                break;
            default:
                NNOPT_ERROR_FMT("model_forward_graph: layer %d has unknown kind %d", i, LAYER_KINDS[i]);
                break;
        }
        if (!next) {
            clReleaseMemObject(ids_buf);
            clReleaseMemObject(x);
            NNOPT_ERROR_FMT("model_forward_graph: decoder layer %d (kind=%d) returned null — implement the corresponding op_<model_type>_<kind>_block in src/ops/", i, LAYER_KINDS[i]);
            return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
        }
        if (next != x) {
            clReleaseMemObject(x);
            x = next;
        }
    }
    NNOPT_LAYER_CHECK("decoder_final_hidden", queue, x, (size_t)seq_len * (size_t)hidden);

    // ── Final RMSNorm and tied LM head ─────────────────────────────────────────
    // PyTorch Lfm2Model.forward applies self.embedding_norm after all decoder layers;
    // Lfm2VlForConditionalGeneration ties lm_head.weight to model.language_model.embed_tokens.weight.
    cl_mem normed = lfm2_rms_norm(cl_ctx, weights, queue, x, seq_len, hidden,
                                  "model.language_model.embedding_norm.weight");
    if (!normed) {
        clReleaseMemObject(ids_buf);
        clReleaseMemObject(x);
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    }
    NNOPT_LAYER_CHECK("embedding_norm", queue, normed, (size_t)seq_len * (size_t)hidden);

    cl_mem lm_w = weights.get_buffer("model.language_model.embed_tokens.weight");
    if (!lm_w) {
        clReleaseMemObject(ids_buf);
        clReleaseMemObject(normed);
        clReleaseMemObject(x);
        NNOPT_ERROR("model_forward_graph: missing tied lm_head weight model.language_model.embed_tokens.weight");
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    }
    // Reference: model_info/transformers_src/modeling_lfm2_vl.py:385-387
    // logits = self.lm_head(hidden_states[:, slice_indices, :]).  The HF
    // generation path keeps only the final-token logits.  Full [T,V] logits are
    // still available for explicit dump/SxS runs, but normal device inference
    // avoids repeatedly allocating ~236MB for an 1800-token VLM prompt, which
    // was destabilizing the Adreno driver after several decode steps.
    const bool full_lm_head = (std::getenv("NNOPT_DUMP_LAYERS") != nullptr) ||
                              (std::getenv("NNOPT_FORCE_FULL_LM_HEAD") != nullptr);
    cl_mem logits = nullptr;
    cl_mem last_hidden = nullptr;
    if (full_lm_head) {
        logits = lfm2_alloc(cl_ctx, (size_t)seq_len * (size_t)vocab, "lm_logits_full");
        if (!logits) {
            clReleaseMemObject(ids_buf);
            clReleaseMemObject(normed);
            clReleaseMemObject(x);
            return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
        }
        if (!pytorch_linear(queue, seq_len, vocab, hidden, normed, lm_w, logits)) {
            clReleaseMemObject(ids_buf);
            clReleaseMemObject(logits);
            clReleaseMemObject(normed);
            clReleaseMemObject(x);
            return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
        }
        NNOPT_LAYER_CHECK("lm_head", queue, logits, (size_t)seq_len * (size_t)vocab);
    } else {
        last_hidden = lfm2_alloc(cl_ctx, (size_t)hidden, "lm_head_last_hidden");
        logits = lfm2_alloc(cl_ctx, (size_t)vocab, "lm_logits_last");
        if (!last_hidden || !logits) {
            if (last_hidden) clReleaseMemObject(last_hidden);
            if (logits) clReleaseMemObject(logits);
            clReleaseMemObject(ids_buf);
            clReleaseMemObject(normed);
            clReleaseMemObject(x);
            return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
        }
        cl_int copy_err = clEnqueueCopyBuffer(queue, normed, last_hidden,
                                              (size_t)(seq_len - 1) * (size_t)hidden * sizeof(nnopt_storage_t),
                                              0,
                                              (size_t)hidden * sizeof(nnopt_storage_t),
                                              0, nullptr, nullptr);
        if (copy_err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("model_forward_graph: copy last hidden failed (%d)", (int)copy_err);
            clReleaseMemObject(last_hidden);
            clReleaseMemObject(ids_buf);
            clReleaseMemObject(logits);
            clReleaseMemObject(normed);
            clReleaseMemObject(x);
            return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
        }
        if (!pytorch_linear(queue, 1, vocab, hidden, last_hidden, lm_w, logits)) {
            clReleaseMemObject(last_hidden);
            clReleaseMemObject(ids_buf);
            clReleaseMemObject(logits);
            clReleaseMemObject(normed);
            clReleaseMemObject(x);
            return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
        }
        NNOPT_LAYER_CHECK("lm_head", queue, logits, (size_t)vocab);
    }

    std::vector<nnopt_storage_t> last_storage((size_t)vocab);
    const size_t off = full_lm_head ? (size_t)(seq_len - 1) * (size_t)vocab * sizeof(nnopt_storage_t) : 0u;
    cl_int err = clEnqueueReadBuffer(queue, logits, CL_TRUE, off,
                                     (size_t)vocab * sizeof(nnopt_storage_t),
                                     last_storage.data(), 0, nullptr, nullptr);
    std::vector<float> out((size_t)vocab, 0.0f);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("model_forward_graph: logits readback failed (%d)", (int)err);
    } else {
        for (int i = 0; i < vocab; ++i) {
#ifdef NNOPT_USE_FP16
            out[(size_t)i] = nnopt_f16_to_f32((uint16_t)last_storage[(size_t)i]);
#else
            out[(size_t)i] = last_storage[(size_t)i];
#endif
        }
    }
    if (last_hidden) clReleaseMemObject(last_hidden);
    clReleaseMemObject(ids_buf);
    clReleaseMemObject(logits);
    clReleaseMemObject(normed);
    clReleaseMemObject(x);
    return out;
}

// GPU-side variant: returns the fp16 logits cl_mem for GPU argmax (no host readback).
// Caller owns the returned buffer and must clReleaseMemObject it.
cl_mem model_forward_graph_logits_buf(OpenCLContext& cl_ctx,
                                      Weights& weights,
                                      const std::vector<int32_t>& input_ids,
                                      int start_pos) {
    const bool is_decode_g = (start_pos > 0);
    if (is_decode_g) { lfm2_pool_reclaim(); lfm2_pool_set_active(true); }
    else             { lfm2_pool_set_active(false); }
    cl_command_queue queue = cl_ctx.queue();
    if (!queue) return nullptr;

    const int seq_len = (int)input_ids.size();
    const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
    const int vocab = MODEL_CONFIG::VOCAB_SIZE;

    cl_mem ids_buf = upload_input_ids(cl_ctx, queue, input_ids);
    if (!ids_buf) return nullptr;

    {
        Model* m = g_active_model_for_vlm_splice;
        if (m) {
            const int requested = (start_pos == 0) ? (seq_len + 256) : (start_pos + seq_len);
            if (!m->ensure_caches(requested)) { clReleaseMemObject(ids_buf); return nullptr; }
            const int k_rows = start_pos + seq_len;
            if (!m->update_counter(queue, start_pos, k_rows)) { clReleaseMemObject(ids_buf); return nullptr; }
        }
    }

    cl_mem x = op_Embedding(cl_ctx, weights, queue, ids_buf, seq_len, hidden,
                            "model.language_model.embed_tokens.weight");
    if (!x) { clReleaseMemObject(ids_buf); return nullptr; }

    Model* active_model = g_active_model_for_vlm_splice;
    const bool use_ref_embeds = (start_pos == 0) && active_model && active_model->has_reference_inputs_embeds();
    if (use_ref_embeds) {
        const int ref_N = active_model->reference_inputs_embeds_N();
        const int n_override = (ref_N < seq_len) ? ref_N : seq_len;
        clEnqueueCopyBuffer(queue, active_model->reference_inputs_embeds_buf(), x,
                            0, 0, (size_t)n_override * (size_t)hidden * sizeof(nnopt_storage_t),
                            0, nullptr, nullptr);
    } else if (start_pos == 0 && !apply_vlm_masked_scatter(queue, x, seq_len, hidden, input_ids)) {
        clReleaseMemObject(ids_buf); clReleaseMemObject(x); return nullptr;
    }

    static const int LAYER_KINDS[16] = {0,0,1,0,0,1,0,0,1,0,1,0,1,0,1,0};
    for (int i = 0; i < 16; ++i) {
        // Yield the GPU per layer during the heavy multi-second prefill (seq_len>1)
        // so the UI thread doesn't starve → ANR. Skip it on single-token decode
        // steps, which are short and where the per-layer sleep would just add latency.
        if (seq_len > 1) cl_ctx.yield_for_compositor();
        cl_mem next = nullptr;
        switch (LAYER_KINDS[i]) {
            case 0: next = op_lfm2_conv_block(cl_ctx, weights, queue, x, seq_len, hidden, i, start_pos); break;
            case 1: next = op_lfm2_full_attention_block(cl_ctx, weights, queue, x, seq_len, hidden, i, start_pos); break;
        }
        if (!next) { clReleaseMemObject(ids_buf); clReleaseMemObject(x); return nullptr; }
        if (next != x) { clReleaseMemObject(x); x = next; }
    }

    cl_mem normed = lfm2_rms_norm(cl_ctx, weights, queue, x, seq_len, hidden,
                                  "model.language_model.embedding_norm.weight");
    if (!normed) { clReleaseMemObject(ids_buf); clReleaseMemObject(x); return nullptr; }

    cl_mem lm_w = weights.get_buffer("model.language_model.embed_tokens.weight");
    if (!lm_w) { clReleaseMemObject(ids_buf); clReleaseMemObject(normed); clReleaseMemObject(x); return nullptr; }

    cl_mem last_hidden = lfm2_alloc(cl_ctx, (size_t)hidden, "lm_head_last_hidden_greedy");
    cl_mem logits = lfm2_alloc(cl_ctx, (size_t)vocab, "lm_logits_greedy");
    if (!last_hidden || !logits) {
        if (last_hidden) clReleaseMemObject(last_hidden);
        if (logits) clReleaseMemObject(logits);
        clReleaseMemObject(ids_buf); clReleaseMemObject(normed); clReleaseMemObject(x);
        return nullptr;
    }
    clEnqueueCopyBuffer(queue, normed, last_hidden,
                        (size_t)(seq_len - 1) * (size_t)hidden * sizeof(nnopt_storage_t),
                        0, (size_t)hidden * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
    if (!pytorch_linear(queue, 1, vocab, hidden, last_hidden, lm_w, logits)) {
        clReleaseMemObject(last_hidden); clReleaseMemObject(logits);
        clReleaseMemObject(ids_buf); clReleaseMemObject(normed); clReleaseMemObject(x);
        return nullptr;
    }
    clReleaseMemObject(last_hidden);
    clReleaseMemObject(ids_buf);
    clReleaseMemObject(normed);
    clReleaseMemObject(x);
    return logits;  // caller owns
}
