// Auto-generated graph-mode backbone for HuggingFaceTB/SmolVLM-256M-Instruct.
//
// THIS IS A STARTING SKELETON. The agent's job is to replace the zero-logits
// stub below with real op_<class>(...) calls from src/forward_dispatch.h.
//
// Roadmap (execution order from .nnport/graph.json):
//   [  0] Embedding            dump=model_text_model_embed_tokens  weight_keys=model.text_model.embed_tokens.weight
//   [  1] Conv2d               dump=model_vision_model_embeddings_patch_embedding  weight_keys=model.vision_model.embeddings.patch_embedding.weight, model.vision_model.embeddings.patch_embedding.bias
//   [  2] Embedding            dump=model_vision_model_embeddings_position_embedding  weight_keys=model.vision_model.embeddings.position_embedding.weight
//   [  3] Idefics3VisionEmbeddings dump=embedding  weight_keys=model.vision_model.embeddings.patch_embedding.weight, model.vision_model.embeddings.patch_embedding.bias…
//   [  4] LayerNorm            dump=block0_sub_layer_norm1  weight_keys=model.vision_model.encoder.layers.0.layer_norm1.weight, model.vision_model.encoder.layers.0.layer_norm1.bias
//   [  5] Linear               dump=block0_sub_self_attn  weight_keys=model.vision_model.encoder.layers.0.self_attn.q_proj.weight, model.vision_model.encoder.layers.0.self_attn.q_proj.bias
//   [  6] Linear               dump=block0_sub_self_attn  weight_keys=model.vision_model.encoder.layers.0.self_attn.k_proj.weight, model.vision_model.encoder.layers.0.self_attn.k_proj.bias
//   [  7] Linear               dump=block0_sub_self_attn  weight_keys=model.vision_model.encoder.layers.0.self_attn.v_proj.weight, model.vision_model.encoder.layers.0.self_attn.v_proj.bias
//   [  8] Linear               dump=block0_sub_self_attn  weight_keys=model.vision_model.encoder.layers.0.self_attn.out_proj.weight, model.vision_model.encoder.layers.0.self_attn.out_proj.bias
//   [  9] Idefics3VisionAttention dump=block0_sub_self_attn  weight_keys=model.vision_model.encoder.layers.0.self_attn.k_proj.weight, model.vision_model.encoder.layers.0.self_attn.k_proj.bias…
//   [ 10] LayerNorm            dump=block0_sub_layer_norm2  weight_keys=model.vision_model.encoder.layers.0.layer_norm2.weight, model.vision_model.encoder.layers.0.layer_norm2.bias
//   [ 11] Linear               dump=block0_sub_mlp  weight_keys=model.vision_model.encoder.layers.0.mlp.fc1.weight, model.vision_model.encoder.layers.0.mlp.fc1.bias
//   [ 12] PytorchGELUTanh      dump=block0_sub_mlp
//   [ 13] Linear               dump=block0_sub_mlp  weight_keys=model.vision_model.encoder.layers.0.mlp.fc2.weight, model.vision_model.encoder.layers.0.mlp.fc2.bias
//   [ 14] Idefics3VisionMLP    dump=block0_sub_mlp  weight_keys=model.vision_model.encoder.layers.0.mlp.fc1.weight, model.vision_model.encoder.layers.0.mlp.fc1.bias…
//   [ 15] Idefics3EncoderLayer dump=model_vision_model_encoder_layers_0  weight_keys=model.vision_model.encoder.layers.0.self_attn.k_proj.weight, model.vision_model.encoder.layers.0.self_attn.k_proj.bias…
//   [ 16] LayerNorm            dump=block1_sub_layer_norm1  weight_keys=model.vision_model.encoder.layers.1.layer_norm1.weight, model.vision_model.encoder.layers.1.layer_norm1.bias
//   [ 17] Linear               dump=block1_sub_self_attn  weight_keys=model.vision_model.encoder.layers.1.self_attn.q_proj.weight, model.vision_model.encoder.layers.1.self_attn.q_proj.bias
//   [ 18] Linear               dump=block1_sub_self_attn  weight_keys=model.vision_model.encoder.layers.1.self_attn.k_proj.weight, model.vision_model.encoder.layers.1.self_attn.k_proj.bias
//   [ 19] Linear               dump=block1_sub_self_attn  weight_keys=model.vision_model.encoder.layers.1.self_attn.v_proj.weight, model.vision_model.encoder.layers.1.self_attn.v_proj.bias
//   [ 20] Linear               dump=block1_sub_self_attn  weight_keys=model.vision_model.encoder.layers.1.self_attn.out_proj.weight, model.vision_model.encoder.layers.1.self_attn.out_proj.bias
//   [ 21] Idefics3VisionAttention dump=block1_sub_self_attn  weight_keys=model.vision_model.encoder.layers.1.self_attn.k_proj.weight, model.vision_model.encoder.layers.1.self_attn.k_proj.bias…
//   [ 22] LayerNorm            dump=block1_sub_layer_norm2  weight_keys=model.vision_model.encoder.layers.1.layer_norm2.weight, model.vision_model.encoder.layers.1.layer_norm2.bias
//   [ 23] Linear               dump=block1_sub_mlp  weight_keys=model.vision_model.encoder.layers.1.mlp.fc1.weight, model.vision_model.encoder.layers.1.mlp.fc1.bias
//   [ 24] PytorchGELUTanh      dump=block1_sub_mlp
//   [ 25] Linear               dump=block1_sub_mlp  weight_keys=model.vision_model.encoder.layers.1.mlp.fc2.weight, model.vision_model.encoder.layers.1.mlp.fc2.bias
//   [ 26] Idefics3VisionMLP    dump=block1_sub_mlp  weight_keys=model.vision_model.encoder.layers.1.mlp.fc1.weight, model.vision_model.encoder.layers.1.mlp.fc1.bias…
//   [ 27] Idefics3EncoderLayer dump=model_vision_model_encoder_layers_1  weight_keys=model.vision_model.encoder.layers.1.self_attn.k_proj.weight, model.vision_model.encoder.layers.1.self_attn.k_proj.bias…
//   [ 28] LayerNorm            dump=block2_sub_layer_norm1  weight_keys=model.vision_model.encoder.layers.2.layer_norm1.weight, model.vision_model.encoder.layers.2.layer_norm1.bias
//   [ 29] Linear               dump=block2_sub_self_attn  weight_keys=model.vision_model.encoder.layers.2.self_attn.q_proj.weight, model.vision_model.encoder.layers.2.self_attn.q_proj.bias
//   [ 30] Linear               dump=block2_sub_self_attn  weight_keys=model.vision_model.encoder.layers.2.self_attn.k_proj.weight, model.vision_model.encoder.layers.2.self_attn.k_proj.bias
//   [ 31] Linear               dump=block2_sub_self_attn  weight_keys=model.vision_model.encoder.layers.2.self_attn.v_proj.weight, model.vision_model.encoder.layers.2.self_attn.v_proj.bias
//   [ 32] Linear               dump=block2_sub_self_attn  weight_keys=model.vision_model.encoder.layers.2.self_attn.out_proj.weight, model.vision_model.encoder.layers.2.self_attn.out_proj.bias
//   [ 33] Idefics3VisionAttention dump=block2_sub_self_attn  weight_keys=model.vision_model.encoder.layers.2.self_attn.k_proj.weight, model.vision_model.encoder.layers.2.self_attn.k_proj.bias…
//   [ 34] LayerNorm            dump=block2_sub_layer_norm2  weight_keys=model.vision_model.encoder.layers.2.layer_norm2.weight, model.vision_model.encoder.layers.2.layer_norm2.bias
//   [ 35] Linear               dump=block2_sub_mlp  weight_keys=model.vision_model.encoder.layers.2.mlp.fc1.weight, model.vision_model.encoder.layers.2.mlp.fc1.bias
//   [ 36] PytorchGELUTanh      dump=block2_sub_mlp
//   [ 37] Linear               dump=block2_sub_mlp  weight_keys=model.vision_model.encoder.layers.2.mlp.fc2.weight, model.vision_model.encoder.layers.2.mlp.fc2.bias
//   [ 38] Idefics3VisionMLP    dump=block2_sub_mlp  weight_keys=model.vision_model.encoder.layers.2.mlp.fc1.weight, model.vision_model.encoder.layers.2.mlp.fc1.bias…
//   [ 39] Idefics3EncoderLayer dump=model_vision_model_encoder_layers_2  weight_keys=model.vision_model.encoder.layers.2.self_attn.k_proj.weight, model.vision_model.encoder.layers.2.self_attn.k_proj.bias…
//   [ 40] LayerNorm            dump=block3_sub_layer_norm1  weight_keys=model.vision_model.encoder.layers.3.layer_norm1.weight, model.vision_model.encoder.layers.3.layer_norm1.bias
//   [ 41] Linear               dump=block3_sub_self_attn  weight_keys=model.vision_model.encoder.layers.3.self_attn.q_proj.weight, model.vision_model.encoder.layers.3.self_attn.q_proj.bias
//   [ 42] Linear               dump=block3_sub_self_attn  weight_keys=model.vision_model.encoder.layers.3.self_attn.k_proj.weight, model.vision_model.encoder.layers.3.self_attn.k_proj.bias
//   [ 43] Linear               dump=block3_sub_self_attn  weight_keys=model.vision_model.encoder.layers.3.self_attn.v_proj.weight, model.vision_model.encoder.layers.3.self_attn.v_proj.bias
//   [ 44] Linear               dump=block3_sub_self_attn  weight_keys=model.vision_model.encoder.layers.3.self_attn.out_proj.weight, model.vision_model.encoder.layers.3.self_attn.out_proj.bias
//   [ 45] Idefics3VisionAttention dump=block3_sub_self_attn  weight_keys=model.vision_model.encoder.layers.3.self_attn.k_proj.weight, model.vision_model.encoder.layers.3.self_attn.k_proj.bias…
//   [ 46] LayerNorm            dump=block3_sub_layer_norm2  weight_keys=model.vision_model.encoder.layers.3.layer_norm2.weight, model.vision_model.encoder.layers.3.layer_norm2.bias
//   [ 47] Linear               dump=block3_sub_mlp  weight_keys=model.vision_model.encoder.layers.3.mlp.fc1.weight, model.vision_model.encoder.layers.3.mlp.fc1.bias
//   [ 48] PytorchGELUTanh      dump=block3_sub_mlp
//   [ 49] Linear               dump=block3_sub_mlp  weight_keys=model.vision_model.encoder.layers.3.mlp.fc2.weight, model.vision_model.encoder.layers.3.mlp.fc2.bias
//   [ 50] Idefics3VisionMLP    dump=block3_sub_mlp  weight_keys=model.vision_model.encoder.layers.3.mlp.fc1.weight, model.vision_model.encoder.layers.3.mlp.fc1.bias…
//   [ 51] Idefics3EncoderLayer dump=model_vision_model_encoder_layers_3  weight_keys=model.vision_model.encoder.layers.3.self_attn.k_proj.weight, model.vision_model.encoder.layers.3.self_attn.k_proj.bias…
//   [ 52] LayerNorm            dump=block4_sub_layer_norm1  weight_keys=model.vision_model.encoder.layers.4.layer_norm1.weight, model.vision_model.encoder.layers.4.layer_norm1.bias
//   [ 53] Linear               dump=block4_sub_self_attn  weight_keys=model.vision_model.encoder.layers.4.self_attn.q_proj.weight, model.vision_model.encoder.layers.4.self_attn.q_proj.bias
//   [ 54] Linear               dump=block4_sub_self_attn  weight_keys=model.vision_model.encoder.layers.4.self_attn.k_proj.weight, model.vision_model.encoder.layers.4.self_attn.k_proj.bias
//   [ 55] Linear               dump=block4_sub_self_attn  weight_keys=model.vision_model.encoder.layers.4.self_attn.v_proj.weight, model.vision_model.encoder.layers.4.self_attn.v_proj.bias
//   [ 56] Linear               dump=block4_sub_self_attn  weight_keys=model.vision_model.encoder.layers.4.self_attn.out_proj.weight, model.vision_model.encoder.layers.4.self_attn.out_proj.bias
//   [ 57] Idefics3VisionAttention dump=block4_sub_self_attn  weight_keys=model.vision_model.encoder.layers.4.self_attn.k_proj.weight, model.vision_model.encoder.layers.4.self_attn.k_proj.bias…
//   [ 58] LayerNorm            dump=block4_sub_layer_norm2  weight_keys=model.vision_model.encoder.layers.4.layer_norm2.weight, model.vision_model.encoder.layers.4.layer_norm2.bias
//   [ 59] Linear               dump=block4_sub_mlp  weight_keys=model.vision_model.encoder.layers.4.mlp.fc1.weight, model.vision_model.encoder.layers.4.mlp.fc1.bias
//   [ 60] PytorchGELUTanh      dump=block4_sub_mlp
//   [ 61] Linear               dump=block4_sub_mlp  weight_keys=model.vision_model.encoder.layers.4.mlp.fc2.weight, model.vision_model.encoder.layers.4.mlp.fc2.bias
//   [ 62] Idefics3VisionMLP    dump=block4_sub_mlp  weight_keys=model.vision_model.encoder.layers.4.mlp.fc1.weight, model.vision_model.encoder.layers.4.mlp.fc1.bias…
//   [ 63] Idefics3EncoderLayer dump=model_vision_model_encoder_layers_4  weight_keys=model.vision_model.encoder.layers.4.self_attn.k_proj.weight, model.vision_model.encoder.layers.4.self_attn.k_proj.bias…
//   [ 64] LayerNorm            dump=block5_sub_layer_norm1  weight_keys=model.vision_model.encoder.layers.5.layer_norm1.weight, model.vision_model.encoder.layers.5.layer_norm1.bias
//   [ 65] Linear               dump=block5_sub_self_attn  weight_keys=model.vision_model.encoder.layers.5.self_attn.q_proj.weight, model.vision_model.encoder.layers.5.self_attn.q_proj.bias
//   [ 66] Linear               dump=block5_sub_self_attn  weight_keys=model.vision_model.encoder.layers.5.self_attn.k_proj.weight, model.vision_model.encoder.layers.5.self_attn.k_proj.bias
//   [ 67] Linear               dump=block5_sub_self_attn  weight_keys=model.vision_model.encoder.layers.5.self_attn.v_proj.weight, model.vision_model.encoder.layers.5.self_attn.v_proj.bias
//   [ 68] Linear               dump=block5_sub_self_attn  weight_keys=model.vision_model.encoder.layers.5.self_attn.out_proj.weight, model.vision_model.encoder.layers.5.self_attn.out_proj.bias
//   [ 69] Idefics3VisionAttention dump=block5_sub_self_attn  weight_keys=model.vision_model.encoder.layers.5.self_attn.k_proj.weight, model.vision_model.encoder.layers.5.self_attn.k_proj.bias…
//   [ 70] LayerNorm            dump=block5_sub_layer_norm2  weight_keys=model.vision_model.encoder.layers.5.layer_norm2.weight, model.vision_model.encoder.layers.5.layer_norm2.bias
//   [ 71] Linear               dump=block5_sub_mlp  weight_keys=model.vision_model.encoder.layers.5.mlp.fc1.weight, model.vision_model.encoder.layers.5.mlp.fc1.bias
//   [ 72] PytorchGELUTanh      dump=block5_sub_mlp
//   [ 73] Linear               dump=block5_sub_mlp  weight_keys=model.vision_model.encoder.layers.5.mlp.fc2.weight, model.vision_model.encoder.layers.5.mlp.fc2.bias
//   [ 74] Idefics3VisionMLP    dump=block5_sub_mlp  weight_keys=model.vision_model.encoder.layers.5.mlp.fc1.weight, model.vision_model.encoder.layers.5.mlp.fc1.bias…
//   [ 75] Idefics3EncoderLayer dump=model_vision_model_encoder_layers_5  weight_keys=model.vision_model.encoder.layers.5.self_attn.k_proj.weight, model.vision_model.encoder.layers.5.self_attn.k_proj.bias…
//   [ 76] LayerNorm            dump=block6_sub_layer_norm1  weight_keys=model.vision_model.encoder.layers.6.layer_norm1.weight, model.vision_model.encoder.layers.6.layer_norm1.bias
//   [ 77] Linear               dump=block6_sub_self_attn  weight_keys=model.vision_model.encoder.layers.6.self_attn.q_proj.weight, model.vision_model.encoder.layers.6.self_attn.q_proj.bias
//   [ 78] Linear               dump=block6_sub_self_attn  weight_keys=model.vision_model.encoder.layers.6.self_attn.k_proj.weight, model.vision_model.encoder.layers.6.self_attn.k_proj.bias
//   [ 79] Linear               dump=block6_sub_self_attn  weight_keys=model.vision_model.encoder.layers.6.self_attn.v_proj.weight, model.vision_model.encoder.layers.6.self_attn.v_proj.bias
//   …(+470 more nodes — see .nnport/graph.json)
//
// Per-class instance counts:
//   - Linear: 284
//   - LlamaRMSNorm: 61
//   - LlamaSdpaAttention: 30
//   - SiLU: 30
//   - LlamaMLP: 30
//   - LlamaDecoderLayer: 30
//   - LayerNorm: 25
//   - Idefics3VisionAttention: 12
//   - PytorchGELUTanh: 12
//   - Idefics3VisionMLP: 12
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
#include "utils.h"  // nnopt_storage_t, pytorch_linear, nnopt_f16_to_f32
#include "profile.h"

#include <CL/cl.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// VLM splice: image features cache populated by Model::set_image (model.cpp).
// Returns false when no image has been set (text-only path).
extern "C" bool nnopt_get_image_features(const float** data_out,
                                         int* N_out,
                                         int* D_out);

// VLM splice: in-place multi-position replace at placeholder rows.
//   text_embeds[positions[i], :] = image_features[i, :], i in [0, N).
// Defined in src/ops/splice.cpp.
bool splice_image_tokens(
    OpenCLContext& cl_ctx,
    cl_mem text_embeds,
    cl_mem image_features,
    cl_mem positions,
    int N,
    int D);

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

}  // namespace

std::vector<float> model_forward_graph(OpenCLContext& cl_ctx,
                                       Weights& weights,
                                       const std::vector<int32_t>& input_ids,
                                       int start_pos) {
    NNOPT_CHECKPOINT("model_forward_graph entry (graph-mode backbone)");

    cl_command_queue queue = cl_ctx.queue();
    if (!queue) {
        NNOPT_ERROR("model_forward_graph: cl_ctx.queue() returned null");
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    }

    int seq_len = (int)input_ids.size();

    // IMPORTANT: num_tokens MUST match the actual length of input_ids.
    // Faking seq_len to match a reference dump shape is invalid because the token-id
    // buffer only has `input_ids.size()` elements. Using a larger `num_tokens`
    // makes the embedding kernel read token_ids[t] out-of-bounds (undefined values),
    // which then indexes wte out-of-bounds and poisons the entire graph.
    const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
    const int vocab = MODEL_CONFIG::VOCAB_SIZE;

    if (seq_len <= 0) {
        NNOPT_ERROR("model_forward_graph: empty input_ids");
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    }

    cl_mem ids_buf = upload_input_ids(cl_ctx, queue, input_ids);
    if (!ids_buf) {
        NNOPT_ERROR("model_forward_graph: ids_buf null");
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    }

    // Track allocations for cleanup on failure.
    cl_mem x = nullptr;
    cl_mem tmp = nullptr;
    cl_mem normed_final = nullptr;
    cl_mem logits_all = nullptr;

    // Persistent decode scratch (seq_len==1): logits_all is the big one (96 KB
    // for vocab=49280). Declared before `fail` so the lambda can preserve it.
    static cl_mem s_logits_all_decode = nullptr;
    bool logits_all_is_persistent = false;

    auto fail = [&]() -> std::vector<float> {
        if (logits_all && !logits_all_is_persistent) { clReleaseMemObject(logits_all); }
        logits_all = nullptr;
        if (normed_final) { clReleaseMemObject(normed_final); normed_final = nullptr; }
        if (tmp) { clReleaseMemObject(tmp); tmp = nullptr; }
        if (x) { clReleaseMemObject(x); x = nullptr; }
        if (ids_buf) { clReleaseMemObject(ids_buf); ids_buf = nullptr; }
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    };

    NNOPT_PROFILE_BEGIN(queue, "00_embedding");
    // ── Text token embedding ──
    // Reference key (contract): model.text_model.embed_tokens.weight
    x = op_Embedding(cl_ctx, weights, queue, ids_buf, /*num_tokens=*/seq_len,
                     /*hidden_size=*/hidden,
                     "model.text_model.embed_tokens.weight");
    NNOPT_PROFILE_END(queue, "00_embedding");
    if (!x) {
        NNOPT_ERROR("model_forward_graph: op_Embedding failed");
        return fail();
    }
    // Dump name must match dump_spec.json (primary): model_text_model_embed_tokens
    NNOPT_LAYER_CHECK("model_text_model_embed_tokens", queue, x, (size_t)seq_len * (size_t)hidden);

    // ── VLM splice: replace image-placeholder embeddings with image features ──
    // Generic across HF VLMs that pre-expand the placeholder in input_ids
    // (Idefics3/SmolVLM, LLaVA-NeXT, Qwen-VL, PaliGemma). We scan input_ids
    // for IMAGE_TOKEN_ID at runtime; the projector output (cached in model.cpp
    // via nnopt_get_image_features) must have exactly num_placeholders rows.
    //
    // SKIPPED transparently when (a) no image was set, or (b) input_ids has
    // no IMAGE_TOKEN_ID — so text-only ports compile against this code path.
    {
      const float* img_feats_host = nullptr;
      int img_N = 0;
      int img_D = 0;
      const bool have_image = nnopt_get_image_features(&img_feats_host, &img_N, &img_D);

      if (have_image && img_N > 0 && img_D == hidden) {
        // Scan input_ids for image_token_id placeholder positions in THIS forward.
        std::vector<int32_t> positions;
        positions.reserve((size_t)img_N);
        for (int i = 0; i < (int)input_ids.size(); ++i) {
          if (input_ids[(size_t)i] == MODEL_CONFIG::IMAGE_TOKEN_ID) {
            positions.push_back(i);
          }
        }
        // Decode steps feed a single new token (no image placeholders): splice
        // is a prefill-only op since image features were already written into
        // the KV cache during prefill. Skip silently when there's nothing to
        // splice in THIS forward.
        if (positions.empty()) {
          // fall through to decoder loop, no splice needed
        } else if ((int)positions.size() != img_N) {
          NNOPT_ERROR_FMT(
              "splice: placeholder count mismatch — input_ids has %d image_token_id=%d positions, "
              "but projector emitted %d feature rows. Check chat template / processor.",
              (int)positions.size(), MODEL_CONFIG::IMAGE_TOKEN_ID, img_N);
          return fail();
        } else {

        // Upload positions (int32 [N]).
        cl_int err = CL_SUCCESS;
        cl_mem pos_buf = clCreateBuffer(cl_ctx.context(),
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        positions.size() * sizeof(int32_t),
                                        positions.data(), &err);
        if (err != CL_SUCCESS || !pos_buf) {
          NNOPT_ERROR_FMT("splice: clCreateBuffer(positions) failed (%d)", (int)err);
          return fail();
        }

        // Upload image features (storage_t [N, D]). Convert host fp32 → fp16 if needed.
        cl_mem feat_buf = nullptr;
#ifdef NNOPT_USE_FP16
        std::vector<nnopt_storage_t> feats_half((size_t)img_N * (size_t)img_D);
        for (size_t k = 0; k < feats_half.size(); ++k) {
          feats_half[k] = (nnopt_storage_t)nnopt_f32_to_f16(img_feats_host[k]);
        }
        feat_buf = clCreateBuffer(cl_ctx.context(),
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  feats_half.size() * sizeof(nnopt_storage_t),
                                  feats_half.data(), &err);
#else
        feat_buf = clCreateBuffer(cl_ctx.context(),
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  (size_t)img_N * (size_t)img_D * sizeof(nnopt_storage_t),
                                  const_cast<float*>(img_feats_host), &err);
#endif
        if (err != CL_SUCCESS || !feat_buf) {
          NNOPT_ERROR_FMT("splice: clCreateBuffer(image_features) failed (%d)", (int)err);
          clReleaseMemObject(pos_buf);
          return fail();
        }

        const bool ok = splice_image_tokens(cl_ctx, x, feat_buf, pos_buf, img_N, hidden);
        clReleaseMemObject(feat_buf);
        clReleaseMemObject(pos_buf);
        if (!ok) {
          NNOPT_ERROR("splice: splice_image_tokens failed");
          return fail();
        }
        NNOPT_LAYER_CHECK("inputs_embeds_post_splice", queue, x, (size_t)seq_len * (size_t)hidden);
        }  // end else (count matches → splice work)
      } else if (have_image && img_D != hidden) {
        NNOPT_ERROR_FMT(
            "splice: image feature D=%d != LM hidden=%d — connector output is the wrong shape",
            img_D, hidden);
        return fail();
      }
      // else: text-only path, fall through to decoder layers.
    }

    // ── Decoder layers (Llama-style) ──
    const int num_layers = MODEL_CONFIG::NUM_HIDDEN_LAYERS;
    const int intermediate = MODEL_CONFIG::INTERMEDIATE_SIZE;
    const int num_q = MODEL_CONFIG::NUM_ATTENTION_HEADS;
    const int num_kv = MODEL_CONFIG::NUM_KEY_VALUE_HEADS;
    const int head_dim = MODEL_CONFIG::HEAD_DIM;
    const float rms_eps = (float)MODEL_CONFIG::RMS_NORM_EPS;

    for (int i = 0; i < num_layers; ++i) {
        // dump_spec expects these composite module boundaries:
        //   model_text_model_layers_<i>_input_layernorm
        //   model_text_model_layers_<i>_post_attention_layernorm
        // Our LlamaDecoderLayer op internally applies both norms; here we surface
        // the inputs to each via explicit RMSNorm calls so SxS can bisect.

        // --- input_layernorm ---
        char in_ln_name[128];
        std::snprintf(in_ln_name, sizeof(in_ln_name), "model_text_model_layers_%d_input_layernorm", i);
        NNOPT_LAYER_CHECK_INPUT(in_ln_name, queue, x, (size_t)seq_len * (size_t)hidden);

        char in_norm_w[128];
        std::snprintf(in_norm_w, sizeof(in_norm_w), "model.text_model.layers.%d.input_layernorm.weight", i);
        // Visibility-only RMSNorm: skip entirely unless debug layer checks are on.
        // Without this gate the dump-only norm cost ~47% of decode time (benchmark.md).
        cl_mem x_norm = nullptr;
#ifdef NNOPT_DEBUG
        if (nnopt_debug_layers_enabled()) {
            NNOPT_PROFILE_BEGIN(queue, "10_visibility_norms");
            x_norm = op_LlamaRMSNorm(cl_ctx, weights, queue,
                                     x, /*rows=*/seq_len, /*cols=*/hidden,
                                     /*eps=*/rms_eps, in_norm_w);
            NNOPT_PROFILE_END(queue, "10_visibility_norms");
            if (!x_norm) {
                NNOPT_ERROR_FMT("model_forward_graph: op_LlamaRMSNorm(input_layernorm) failed at layer %d", i);
                return fail();
            }
            NNOPT_LAYER_CHECK(in_ln_name, queue, x_norm, (size_t)seq_len * (size_t)hidden);
        }
#endif

        // --- decoder layer ---
        char layer_name[96];
        std::snprintf(layer_name, sizeof(layer_name), "model_text_model_layers_%d", i);

        char post_norm_w[128];
        char q_w[128];
        char k_w[128];
        char v_w[128];
        char o_w[128];
        char gate_w[128];
        char up_w[128];
        char down_w[128];

        std::snprintf(post_norm_w, sizeof(post_norm_w), "model.text_model.layers.%d.post_attention_layernorm.weight", i);

        std::snprintf(q_w, sizeof(q_w), "model.text_model.layers.%d.self_attn.q_proj.weight", i);
        std::snprintf(k_w, sizeof(k_w), "model.text_model.layers.%d.self_attn.k_proj.weight", i);
        std::snprintf(v_w, sizeof(v_w), "model.text_model.layers.%d.self_attn.v_proj.weight", i);
        std::snprintf(o_w, sizeof(o_w), "model.text_model.layers.%d.self_attn.o_proj.weight", i);

        std::snprintf(gate_w, sizeof(gate_w), "model.text_model.layers.%d.mlp.gate_proj.weight", i);
        std::snprintf(up_w, sizeof(up_w), "model.text_model.layers.%d.mlp.up_proj.weight", i);
        std::snprintf(down_w, sizeof(down_w), "model.text_model.layers.%d.mlp.down_proj.weight", i);

        // IMPORTANT: LlamaDecoderLayer expects the *raw* hidden_states and it
        // computes its own norms internally. We call it on x (not x_norm) for
        // correct forward, and only use x_norm for dump visibility.
        if (x_norm) {
            clReleaseMemObject(x_norm);
            x_norm = nullptr;
        }

        NNOPT_PROFILE_BEGIN(queue, "20_decoder_layer");
        tmp = op_LlamaDecoderLayer(cl_ctx,
                                  weights,
                                  queue,
                                  x,
                                  /*rows=*/seq_len,
                                  /*hidden_size=*/hidden,
                                  /*intermediate_size=*/intermediate,
                                  /*num_q_heads=*/num_q,
                                  /*num_kv_heads=*/num_kv,
                                  /*head_dim=*/head_dim,
                                  /*start_pos=*/start_pos,
                                  /*in_norm_w=*/in_norm_w,
                                  /*post_norm_w=*/post_norm_w,
                                  /*rms_eps=*/rms_eps,
                                  /*q_w=*/q_w,
                                  /*k_w=*/k_w,
                                  /*v_w=*/v_w,
                                  /*o_w=*/o_w,
                                  /*gate_w=*/gate_w,
                                  /*up_w=*/up_w,
                                  /*down_w=*/down_w);
        NNOPT_PROFILE_END(queue, "20_decoder_layer");
        if (!tmp) {
            NNOPT_ERROR_FMT("model_forward_graph: op_LlamaDecoderLayer failed at layer %d", i);
            return fail();
        }
        clReleaseMemObject(x);
        x = tmp;
        tmp = nullptr;

        NNOPT_LAYER_CHECK(layer_name, queue, x, (size_t)seq_len * (size_t)hidden);

        // --- post_attention_layernorm (visibility only) ---
        // Skip entirely unless debug layer checks are on; same rationale as input_layernorm above.
#ifdef NNOPT_DEBUG
        if (nnopt_debug_layers_enabled()) {
            char post_ln_name[128];
            std::snprintf(post_ln_name, sizeof(post_ln_name), "model_text_model_layers_%d_post_attention_layernorm", i);
            NNOPT_PROFILE_BEGIN(queue, "10_visibility_norms");
            cl_mem post_ln_out = op_LlamaRMSNorm(cl_ctx, weights, queue,
                                                 x, /*rows=*/seq_len, /*cols=*/hidden,
                                                 /*eps=*/rms_eps, post_norm_w);
            NNOPT_PROFILE_END(queue, "10_visibility_norms");
            if (post_ln_out) {
                NNOPT_LAYER_CHECK(post_ln_name, queue, post_ln_out, (size_t)seq_len * (size_t)hidden);
                clReleaseMemObject(post_ln_out);
                post_ln_out = nullptr;
            }
        }
#endif
    }

    // ── Final RMSNorm ──
    NNOPT_PROFILE_BEGIN(queue, "30_final_norm");
    normed_final = op_LlamaRMSNorm(cl_ctx,
                                  weights,
                                  queue,
                                  x,
                                  /*rows=*/seq_len,
                                  /*cols=*/hidden,
                                  /*eps=*/rms_eps,
                                  "model.text_model.norm.weight");
    NNOPT_PROFILE_END(queue, "30_final_norm");
    if (!normed_final) {
        NNOPT_ERROR("model_forward_graph: final op_LlamaRMSNorm failed");
        return fail();
    }
    NNOPT_LAYER_CHECK("model_text_model_norm", queue, normed_final, (size_t)seq_len * (size_t)hidden);

    // ── lm_head logits over full sequence ──
    {
        // NOTE: weight key is top-level (no "model." prefix) in weights/model.meta.json.
        cl_mem W = weights.get_buffer("lm_head.weight");
        if (!W) {
            NNOPT_ERROR("model_forward_graph: missing lm_head.weight");
            return fail();
        }

        cl_int err = CL_SUCCESS;
        const size_t logits_elems = (size_t)seq_len * (size_t)vocab;
        // Note: logits_all persistence was tried and slightly regressed
        // (-5%) on Adreno 620 — likely cache-locality. Keep per-call alloc.
        (void)s_logits_all_decode;
        logits_all = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                    logits_elems * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS || !logits_all) {
            NNOPT_ERROR_FMT("model_forward_graph: clCreateBuffer(logits_all) failed (%d)", (int)err);
            return fail();
        }

        NNOPT_PROFILE_BEGIN(queue, "40_lm_head");
        if (!pytorch_linear(queue, /*M=*/seq_len, /*N=*/vocab, /*K=*/hidden, normed_final, W, logits_all)) {
            NNOPT_ERROR("model_forward_graph: pytorch_linear lm_head failed");
            return fail();
        }
        NNOPT_PROFILE_END(queue, "40_lm_head");
        NNOPT_LAYER_CHECK("lm_head", queue, logits_all, (size_t)seq_len * (size_t)vocab);
    }

    // ── Read last-row logits to host as float32 ──
    std::vector<float> host_logits((size_t)vocab, 0.0f);
#ifdef NNOPT_USE_FP16
    // Decode greedy fast path: run argmax on GPU, read only 1 int (vs ~98 KB
    // for the full fp16 logits row). Skip when validation needs the full row.
    if (seq_len == 1
#ifdef NNOPT_DEBUG
        && !nnopt_debug_layers_enabled()
#endif
       ) {
      const size_t row_off_bytes = (size_t)(seq_len - 1) * (size_t)vocab * sizeof(nnopt_storage_t);
      cl_int e0 = CL_SUCCESS;
      cl_mem idx_buf = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(int32_t), nullptr, &e0);
      if (e0 == CL_SUCCESS && idx_buf) {
        cl_buffer_region region = {row_off_bytes, (size_t)vocab * sizeof(nnopt_storage_t)};
        cl_mem last_row = clCreateSubBuffer(logits_all, CL_MEM_READ_ONLY,
                                            CL_BUFFER_CREATE_TYPE_REGION, &region, &e0);
        if (e0 == CL_SUCCESS && last_row && argmax_fp16_dispatch(queue, vocab, last_row, idx_buf)) {
          NNOPT_PROFILE_BEGIN(queue, "50_logits_d2h");
          int32_t argmax_idx = 0;
          if (clEnqueueReadBuffer(queue, idx_buf, CL_TRUE, 0, sizeof(int32_t),
                                  &argmax_idx, 0, nullptr, nullptr) == CL_SUCCESS) {
            // Synthesize a one-hot logits row so the host-side greedy sampler
            // picks the same id without further math.
            if (argmax_idx >= 0 && argmax_idx < vocab) {
              host_logits[(size_t)argmax_idx] = 1.0e30f;
            }
            clReleaseMemObject(last_row);
            clReleaseMemObject(idx_buf);
            NNOPT_PROFILE_END(queue, "50_logits_d2h");
            clReleaseMemObject(ids_buf);
            ids_buf = nullptr;
            clReleaseMemObject(x);
            x = nullptr;
            clReleaseMemObject(normed_final);
            normed_final = nullptr;
            if (!logits_all_is_persistent) clReleaseMemObject(logits_all);
            logits_all = nullptr;
            return host_logits;
          }
          NNOPT_PROFILE_END(queue, "50_logits_d2h");
        }
        if (last_row) clReleaseMemObject(last_row);
        clReleaseMemObject(idx_buf);
      }
      // Fall through to full readback on any error path.
    }
#endif
    NNOPT_PROFILE_BEGIN(queue, "50_logits_d2h");
    {
        const size_t row_off_elems = (size_t)(seq_len - 1) * (size_t)vocab;
        const size_t row_off_bytes = row_off_elems * sizeof(nnopt_storage_t);

#ifdef NNOPT_USE_FP16
        std::vector<nnopt_storage_t> tmp_half((size_t)vocab);
        cl_int err = clEnqueueReadBuffer(queue, logits_all, CL_TRUE,
                                         row_off_bytes,
                                         (size_t)vocab * sizeof(nnopt_storage_t),
                                         tmp_half.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("model_forward_graph: read logits failed (%d)", (int)err);
            return fail();
        }
        for (int i = 0; i < vocab; ++i) {
            host_logits[(size_t)i] = nnopt_f16_to_f32((uint16_t)tmp_half[(size_t)i]);
        }
#else
        cl_int err = clEnqueueReadBuffer(queue, logits_all, CL_TRUE,
                                         row_off_bytes,
                                         (size_t)vocab * sizeof(float),
                                         host_logits.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("model_forward_graph: read logits failed (%d)", (int)err);
            return fail();
        }
#endif
    }
    NNOPT_PROFILE_END(queue, "50_logits_d2h");

    // Cleanup GPU buffers we own.
    clReleaseMemObject(ids_buf);
    ids_buf = nullptr;
    clReleaseMemObject(x);
    x = nullptr;
    clReleaseMemObject(normed_final);
    normed_final = nullptr;
    if (!logits_all_is_persistent) clReleaseMemObject(logits_all);
    logits_all = nullptr;

    return host_logits;
}
