// Reference: nn.Linear (bias=False); lm_head tied to model.embed_tokens.weight
// Linear.cpp — shared implementation for ALL 127 Linear node(s).
//
// You write the Linear forward() math ONCE here. backbone.cpp calls
// this function once per traced node of class Linear, passing the
// appropriate weight_prefix string for each call — your implementation
// uses that prefix to load weights, so the same C++ math correctly
// handles every sibling instance.
//
// Sibling nodes (order  dump_name  weight_prefix):
//      5  model_layers_0_self_attn_q_proj           weight_prefix=model.layers.0.self_attn.q_proj
//      6  model_layers_0_self_attn_k_proj           weight_prefix=model.layers.0.self_attn.k_proj
//      7  model_layers_0_self_attn_v_proj           weight_prefix=model.layers.0.self_attn.v_proj
//     10  model_layers_0_self_attn_o_proj           weight_prefix=model.layers.0.self_attn.o_proj
//     14  model_layers_0_mlp_gate_proj              weight_prefix=model.layers.0.mlp.gate_proj
//     16  model_layers_0_mlp_up_proj                weight_prefix=model.layers.0.mlp.up_proj
//     17  model_layers_0_mlp_down_proj              weight_prefix=model.layers.0.mlp.down_proj
//     21  model_layers_1_self_attn_q_proj           weight_prefix=model.layers.1.self_attn.q_proj
//     22  model_layers_1_self_attn_k_proj           weight_prefix=model.layers.1.self_attn.k_proj
//     23  model_layers_1_self_attn_v_proj           weight_prefix=model.layers.1.self_attn.v_proj
//     26  model_layers_1_self_attn_o_proj           weight_prefix=model.layers.1.self_attn.o_proj
//     30  model_layers_1_mlp_gate_proj              weight_prefix=model.layers.1.mlp.gate_proj
//   … (+115 more siblings — see .nnport/scaffold_manifest.json)
//
// Representative shape (from sibling 0):
//   input:  [1, 7, 640]
//   output: [1, 7, 1024]
//
// Primary reference dump for cosine validation:
//   reference/layers/model_layers_0_self_attn_q_proj_output.bin
//   (per-node cosine validation runs against EVERY sibling's dump too —
//    if your math is correct for one weight_prefix, it's correct for all.)
//
// ─── SIGNATURE ──────────────────────────────────────────────────────────
//
// Hardened universal signature. EVERY <Class>_forward in this scaffold
// uses these 10 params, so backbone.cpp can call any op uniformly:
//
//   cl_ctx          — OpenCLContext& (queue, device, context)
//   weights         — Weights& (use `weights.get_buffer(wp + ".<param>")`)
//   queue           — cl_command_queue for kernel dispatch
//   input           — cl_mem of the input tensor (int32 for Embedding;
//                     nnopt_storage_t for everything else)
//   seq_len         — T dimension of input
//   layer_idx       — 0..NUM_LAYERS-1 for per-layer ops; -1 for global
//                     ops (Embedding, final norm, lm_head). Use this for
//                     model_config arrays like NUM_QUERY_HEADS[layer_idx].
//   start_pos       — generation step offset for decode (0 during prefill;
//                     prompt_len + step during decode). Needed by ops with
//                     persistent state (attention KV cache, embedding wpe).
//   k_cache_inout   — pointer to layer's K-cache cl_mem (attention only;
//                     other ops ignore). Caller owns the buffer.
//   v_cache_inout   — pointer to layer's V-cache cl_mem (attention only).
//   encoder_hidden_states — cl_mem of the encoder's output for cross-attention
//                     (encoder-decoder models like Whisper, T5, SeamlessM4T).
//                     PASSED for decoder.encoder_attn calls; nullptr everywhere
//                     else. When non-null, the attention op uses
//                     encoder_hidden_states as the K/V source (Q from input).
//   weight_prefix   — state_dict prefix (use `std::string(wp) + ".<p>"`).
//
// Returns: cl_mem (newly-allocated) holding this op's output. Caller owns
// and releases. Return `nullptr` on any internal error (after calling
// NNOPT_ERROR_FMT for the log).
//
// ─── IMPLEMENTATION CHECKLIST ──────────────────────────────────────────
//
// 1. Add a citation comment in the first 40 lines:
//    `// Reference: model_info/transformers_src/<file>.py:<lines> Linear.forward`
//    (Build refuses to compile without it.)
// 2. Load every weight the PyTorch forward() touches via
//    `weights.get_buffer(std::string(weight_prefix) + ".<param>")`.
//    Use `weights.get_shape(wp + ".weight")` for dimensions when needed.
// 3. Allocate output buffer:
//    `cl_mem out = nnopt_pool_alloc(cl_ctx.context(), //                                  numel * sizeof(nnopt_storage_t),
//                                  nullptr, &err);`
// 4. Dispatch your kernel(s); `clFinish(queue)` before returning.
// 5. NEVER return a passthrough or zeros — implement the real math or
//    keep the NNOPT_ERROR sentinel (per AUTONOMOUS_PORTING.md §0a).
//
// ─── ARGS YOU PROBABLY DON'T NEED ──────────────────────────────────────
//
// Mark unused args with `(void)arg;` to silence warnings. Common per-class:
//   - Embedding/Linear/Norm/Activation: ignore layer_idx, start_pos, k/v_cache, encoder_hidden_states
//   - RotaryEmbedding: ignore k/v_cache, encoder_hidden_states
//   - Self-attention: USE input, k_cache_inout, v_cache_inout; ignore encoder_hidden_states
//   - Cross-attention (decoder.encoder_attn): USE input (Q source), encoder_hidden_states (K/V source); ignore k/v_cache
//   - WhisperAttention etc. (dual-mode): branch on `encoder_hidden_states != nullptr` to pick self vs cross
//   - MLP: ignore layer_idx, start_pos, k/v_cache, encoder_hidden_states
//   - DecoderLayer (encoder-decoder): USE layer_idx, start_pos, k/v_cache, encoder_hidden_states

// Reference: nn.Linear (bias=False) — out[M,N] = x[M,K] @ W[N,K]^T (pytorch_linear).
// Used standalone only for lm_head; per-layer projections go through the
// composite Gemma3DecoderLayer. lm_head is TIED to embed_tokens.weight
// (no lm_head.weight tensor in safetensors).

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include <string>
#include <vector>

extern "C" {
cl_mem Linear_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem input,
    int seq_len,
    int layer_idx,
    int start_pos,
    cl_mem* k_cache_inout,
    cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states,
    const char* weight_prefix)
{
    (void)layer_idx; (void)start_pos; (void)k_cache_inout; (void)v_cache_inout;
    (void)encoder_hidden_states;
    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();

    // Resolve weight: lm_head is tied to the embedding table.
    std::string wkey = wp + ".weight";
    cl_mem W = weights.get_buffer(wkey, /*optional=*/true);
    if (!W) {
        wkey = "model.embed_tokens.weight";
        W = weights.get_buffer(wkey);
    }
    if (!W) { NNOPT_ERROR_FMT("Linear: missing weight for %s", wp.c_str()); return nullptr; }

    std::vector<int> shp = weights.get_shape(wkey);   // [out_features, in_features]
    if (shp.size() != 2) { NNOPT_ERROR_FMT("Linear: bad weight shape for %s", wkey.c_str()); return nullptr; }
    int N = shp[0];   // out_features
    int K = shp[1];   // in_features
    int M = seq_len;

    cl_int err = CL_SUCCESS;
    cl_mem out = nnopt_pool_alloc(cl_ctx.context(),
                                (size_t)M * N * sizeof(nnopt_storage_t), &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("Linear: alloc out %d", err); return nullptr; }

    if (!pytorch_linear(queue, M, N, K, input, W, out)) {
        NNOPT_ERROR_FMT("Linear: pytorch_linear failed for %s (M=%d N=%d K=%d)", wkey.c_str(), M, N, K);
        nnopt_pool_free(out);
        return nullptr;
    }
    return out;
}
}
