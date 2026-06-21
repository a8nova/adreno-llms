// Embedding.cpp — shared implementation for ALL 6 Embedding node(s).
//
// You write the Embedding forward() math ONCE here. backbone.cpp calls
// this function once per traced node of class Embedding, passing the
// appropriate weight_prefix string for each call — your implementation
// uses that prefix to load weights, so the same C++ math correctly
// handles every sibling instance.
//
// Sibling nodes (order  dump_name  weight_prefix):
//      0  embedding_wte                             weight_prefix=text_encoder.shared
//     14  decoder_model_decoder_embed_tokens_frontend_out  weight_prefix=decoder.model.decoder.embed_tokens
//     15  embed_tokens_layer_0                      weight_prefix=decoder.model.decoder.embed_tokens.0
//     16  embed_tokens_layer_1                      weight_prefix=decoder.model.decoder.embed_tokens.1
//     17  embed_tokens_layer_2                      weight_prefix=decoder.model.decoder.embed_tokens.2
//     18  embed_tokens_layer_3                      weight_prefix=decoder.model.decoder.embed_tokens.3
//
// Representative shape (from sibling 0):
//   input:  [1, 11]
//   output: [1, 11, 768]
//
// Primary reference dump for cosine validation:
//   reference/layers/embedding_wte_output.bin
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
//    `// Reference: model_info/transformers_src/<file>.py:<lines> Embedding.forward`
//    (Build refuses to compile without it.)
// 2. Load every weight the PyTorch forward() touches via
//    `weights.get_buffer(std::string(weight_prefix) + ".<param>")`.
//    Use `weights.get_shape(wp + ".weight")` for dimensions when needed.
// 3. Allocate output buffer:
//    `cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
//                                  numel * sizeof(nnopt_storage_t),
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

// Reference: model_info/transformers_src/modeling_musicgen.py (nn.Embedding usage)

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"
#include <string>

extern "C" {
cl_mem Embedding_forward(
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
    (void)k_cache_inout;
    (void)v_cache_inout;
    (void)encoder_hidden_states;

    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();

    // HF MusicGen uses plain nn.Embedding weights. Keys are EXACT in the state_dict.
    // Expected examples:
    //   - text_encoder.shared.weight
    //   - decoder.model.decoder.embed_tokens.weight                 (module list wrapper)
    //   - decoder.model.decoder.embed_tokens.0.weight
    //   - decoder.model.decoder.embed_tokens.1.weight
    //   - decoder.model.decoder.embed_tokens.2.weight
    //   - decoder.model.decoder.embed_tokens.3.weight
    //
    // CRITICAL: never "helpfully" rewrite prefixes except for the known wrapper node
    // that ends with ".embed_tokens" (no codebook index). In that case the wrapper
    // corresponds to codebook 0 in the reference graph.
    std::string w_key;
    cl_mem wte = nullptr;

    if (!wp.empty() && wp.size() >= 11 && wp.rfind(".embed_tokens") == (wp.size() - 11)) {
        // Wrapper: decoder.model.decoder.embed_tokens → use the ModuleList's own .weight.
        // In MusicGen, decoder.model.decoder.embed_tokens is an nn.ModuleList-like wrapper
        // that (in our traced graph) resolves to a real weight tensor key.
        w_key = wp + ".weight";
    } else {
        w_key = wp + ".weight";
    }

    // IMPORTANT: weights are keyed exactly like safetensors. These keys are present for musicgen-small:
    //   decoder.model.decoder.embed_tokens.{0,1,2,3}.weight
    //   text_encoder.shared.weight
    wte = weights.get_buffer(w_key, /*optional=*/false);

    if (!wte) {
        NNOPT_ERROR_FMT("Embedding_forward: missing weight buffer for key '%s' (wp='%s')", w_key.c_str(), wp.c_str());
        return nullptr;
    }

    // NOTE: weights/model.meta.json stores tensors under meta["tensors"].
    // Weights::get_shape expects the canonical key string (same as safetensors).
    const std::vector<int> w_shape = weights.get_shape(w_key);
    if (w_shape.size() != 2) {
        NNOPT_ERROR_FMT("Embedding_forward: expected 2D weight %s, got rank=%zu", w_key.c_str(), w_shape.size());
        return nullptr;
    }
    const int vocab_size = w_shape[0];
    const int hidden_size = w_shape[1];

    // input is int32 ids.
    // The scaffold passes seq_len as the TIME dimension (tgt_len), *not*
    // the flattened element count. Embedding sees a flat ids buffer of
    // length = batch * seq_len.
    //
    // In our current MusicGen harness, all embedding calls are effectively
    // batch=1, so num_tokens == seq_len.
    const int num_tokens = seq_len;

    // Input contract debug (int32 ids): layer checks expect this name for
    // decoder input_ids.
    // NOTE: NNOPT_LAYER_CHECK_INPUT_INT casts int->float for SxS.
    NNOPT_LAYER_CHECK_INPUT_INT("block0_sub_musicgendecoder_input_ids", queue, input, (size_t)num_tokens);

    // SxS alignment rule: if a reference dump expects decode-shaped tensors
    // (M==1) for some nodes, *that is handled by the NNOPT_LAYER_CHECK call site*
    // (caller passes 1 element count). Embedding_forward always computes the
    // full [seq_len, hidden] embedding.

    // IMPORTANT: MusicGen-small has 4 separate embedding tables (one per codebook)
    // at decoder.model.decoder.embed_tokens.{0,1,2,3}. Each table has vocab_size=2049.
    // There is NO single concatenated vocab table, so we must NOT offset token IDs by
    // codebook index here. The caller selects the correct table via weight_prefix.
    (void)layer_idx;
    const size_t out_elems = (size_t)num_tokens * (size_t)hidden_size;

    cl_int err = CL_SUCCESS;
    cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                out_elems * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding_forward: clCreateBuffer(out) failed: %d", err);
        return nullptr;
    }

    // Program must be built/cached by the OpenCLContext.
    // IMPORTANT: Do NOT release the program object here; it's owned by OpenCLContext's cache.
    static cl_program prog = nullptr;
    if (!prog) {
        prog = cl_ctx.build_program_from_file("kernels/embedding.cl");
    }
    if (!prog) {
        NNOPT_ERROR("Embedding_forward: failed to build kernels/embedding.cl");
        clReleaseMemObject(out);
        return nullptr;
    }

    cl_kernel kernel = clCreateKernel(prog, "embedding_lookup", &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding_forward: clCreateKernel(embedding_lookup) failed: %d", err);
        clReleaseMemObject(out);
        return nullptr;
    }

    if (!set_arg_checked(kernel, 0, sizeof(cl_mem), &input, "ids")) { clReleaseKernel(kernel); clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(kernel, 1, sizeof(cl_mem), &wte,   "wte")) { clReleaseKernel(kernel); clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(kernel, 2, sizeof(cl_mem), &out,   "out")) { clReleaseKernel(kernel); clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(kernel, 3, sizeof(int), &num_tokens, "num_tokens")) { clReleaseKernel(kernel); clReleaseMemObject(out); return nullptr; }

    if (!set_arg_checked(kernel, 4, sizeof(int), &hidden_size, "hidden_size")) { clReleaseKernel(kernel); clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(kernel, 5, sizeof(int), &vocab_size, "vocab_size")) { clReleaseKernel(kernel); clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(kernel, 6, sizeof(int), &start_pos, "start_pos")) { clReleaseKernel(kernel); clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(kernel, 7, sizeof(int), &layer_idx, "layer_idx")) { clReleaseKernel(kernel); clReleaseMemObject(out); return nullptr; }

    const size_t gws[1] = { out_elems };
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, gws, nullptr, 0, nullptr,
                                 KernelProfiler::event_for((wp + "/embedding_lookup").c_str()));
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding_forward: dispatch failed: %d", err);
        clReleaseKernel(kernel);
        clReleaseMemObject(out);
        return nullptr;
    }

    NNOPT_DEBUG_SYNC(queue);

    clReleaseKernel(kernel);

    // Debug: if ids are out of vocab, output is undefined. We can't read ids here cheaply.
    (void)vocab_size;

    return out;
}
}
