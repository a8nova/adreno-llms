// Reference: model_info/transformers_src/modeling_musicgen.py:260-343 MusicgenDecoderLayer.forward (fc1/fc2)
// Linear.cpp — shared implementation for ALL 53 Linear node(s).
//
// Implements a plain nn.Linear projection: out = x @ W^T (+ b).
// MusicGen's decoder uses bias=False for fc1/fc2, but we support optional bias
// for generality because the graph may contain other Linear nodes.
//
// Sibling nodes (order  dump_name  weight_prefix):
//     25  fc1_0                                     weight_prefix=decoder.model.decoder.layers.0.fc1
//     27  fc2_0                                     weight_prefix=decoder.model.decoder.layers.0.fc2
//     34  fc1_1                                     weight_prefix=decoder.model.decoder.layers.1.fc1
//     36  fc2_1                                     weight_prefix=decoder.model.decoder.layers.1.fc2
//     43  fc1_2                                     weight_prefix=decoder.model.decoder.layers.2.fc1
//     45  fc2_2                                     weight_prefix=decoder.model.decoder.layers.2.fc2
//     52  fc1_3                                     weight_prefix=decoder.model.decoder.layers.3.fc1
//     54  fc2_3                                     weight_prefix=decoder.model.decoder.layers.3.fc2
//     61  fc1_4                                     weight_prefix=decoder.model.decoder.layers.4.fc1
//     63  fc2_4                                     weight_prefix=decoder.model.decoder.layers.4.fc2
//     70  fc1_5                                     weight_prefix=decoder.model.decoder.layers.5.fc1
//     72  fc2_5                                     weight_prefix=decoder.model.decoder.layers.5.fc2
//   … (+41 more siblings — see .nnport/scaffold_manifest.json)
//
// Representative shape (from sibling 0):
//   input:  [1, 1, 1024]
//   output: [1, 1, 4096]
//
// Primary reference dump for cosine validation:
//   reference/layers/fc1_0_output.bin
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

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"

#include <CL/cl.h>
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
    (void)layer_idx;
    (void)start_pos;
    (void)k_cache_inout;
    (void)v_cache_inout;
    (void)encoder_hidden_states;

    if (!input) {
        NNOPT_ERROR("Linear_forward: input is null");
        return nullptr;
    }

    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();

    cl_mem W = nullptr;
    cl_mem b = nullptr;

    std::string wkey;
    std::string bkey;

    // Prefer standard nn.Linear naming.
    if (weights.has_tensor(wp + ".weight")) { W = weights.get_buffer(wp + ".weight"); wkey = wp + ".weight"; }
    else if (weights.has_tensor(wp + "_weight")) { W = weights.get_buffer(wp + "_weight"); wkey = wp + "_weight"; }

    if (weights.has_tensor(wp + ".bias")) { b = weights.get_buffer(wp + ".bias"); bkey = wp + ".bias"; }
    else if (weights.has_tensor(wp + "_bias")) { b = weights.get_buffer(wp + "_bias"); bkey = wp + "_bias"; }

    if (!W) {
        NNOPT_ERROR_FMT("Linear_forward: missing weight for wp=%s", wp.c_str());
        return nullptr;
    }

    const std::vector<int> wshape = weights.get_shape(wkey);
    int N = 0;
    int K = 0;
    if (wshape.size() != 2) {
        NNOPT_ERROR_FMT("Linear_forward: expected 2D weight for key=%s", wkey.c_str());
        return nullptr;
    }
    N = wshape[0];
    K = wshape[1];

    const int M = seq_len;
    const size_t out_numel = (size_t)M * (size_t)N;

    cl_int err = CL_SUCCESS;
    cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                out_numel * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Linear_forward: clCreateBuffer(out) failed: %d", (int)err);
        return nullptr;
    }

    if (!pytorch_linear(queue, M, N, K, input, W, out)) {
        NNOPT_ERROR_FMT("Linear_forward: pytorch_linear failed (wp=%s)", wp.c_str());
        clReleaseMemObject(out);
        return nullptr;
    }

    // Optional bias add (rare in this model but kept for generality).
    if (b) {
        static cl_program utils_prog = nullptr;
        static cl_kernel bias_kern = nullptr;
        // This model's Linear ops are biasless; if a future node has bias we need
        // a dedicated kernel, but kernels/utils.cl doesn't provide one today.
        NNOPT_ERROR_FMT("Linear_forward: bias not supported yet (wp=%s)", wp.c_str());
        clReleaseMemObject(out);
        return nullptr;
    }

    NNOPT_DEBUG_SYNC(queue);
    return out;
}
}
