// Reference: model_info/transformers_src/activations.py (PytorchGELUTanh) + model_info/transformers_src/modeling_musicgen.py:258-344 MusicgenDecoderLayer.forward
// GELUActivation.cpp — shared implementation for ALL 24 GELUActivation node(s).
//
// Implements the activation_fn used in MusicgenDecoderLayer: GELU (tanh approximation).
// This wraps kernels/gelu.cl::gelu_tanh.
//
// You write the GELUActivation forward() math ONCE here. backbone.cpp calls
// this function once per traced node of class GELUActivation, passing the
// appropriate weight_prefix string for each call — your implementation
// ignores weight_prefix (no weights).
//
// Sibling nodes (order  dump_name  weight_prefix):
//     26  activation_fn_0                           weight_prefix=<none>
//     35  activation_fn_1                           weight_prefix=<none>
//     44  activation_fn_2                           weight_prefix=<none>
//     53  activation_fn_3                           weight_prefix=<none>
//     62  activation_fn_4                           weight_prefix=<none>
//     71  activation_fn_5                           weight_prefix=<none>
//     80  activation_fn_6                           weight_prefix=<none>
//     89  activation_fn_7                           weight_prefix=<none>
//     98  activation_fn_8                           weight_prefix=<none>
//    107  activation_fn_9                           weight_prefix=<none>
//    116  activation_fn_10                          weight_prefix=<none>
//    125  activation_fn_11                          weight_prefix=<none>
//   … (+12 more siblings — see .nnport/scaffold_manifest.json)
//
// Representative shape (from sibling 0):
//   input:  [1, 1, 4096]
//   output: [1, 1, 4096]
//
// Primary reference dump for cosine validation:
//   reference/layers/activation_fn_0_output.bin
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
//    `// Reference: model_info/transformers_src/<file>.py:<lines> GELUActivation.forward`
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

#include <string>

extern "C" {
cl_mem GELUActivation_forward(
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
    (void)weights;
    (void)layer_idx;
    (void)start_pos;
    (void)k_cache_inout;
    (void)v_cache_inout;
    (void)encoder_hidden_states;
    (void)weight_prefix;

    cl_int err = CL_SUCCESS;

    cl_program prog = cl_ctx.build_program_from_file("kernels/gelu.cl"); // PROGRAM-INIT-OK: OpenCLContext caches programs by path
    if (!prog) {
        NNOPT_ERROR("GELUActivation_forward: failed to build kernels/gelu.cl");
        return nullptr;
    }

    cl_kernel k = clCreateKernel(prog, "gelu_tanh", &err);
    if (err != CL_SUCCESS || !k) {
        NNOPT_ERROR_FMT("GELUActivation_forward: clCreateKernel(gelu_tanh) failed (%d)", (int)err);
        return nullptr;
    }

    const size_t n = (size_t)seq_len * (size_t)MODEL_CONFIG::FFN_DIM;

    cl_mem out = nullptr;
    auto cleanup = [&]() -> cl_mem {
        if (k) { clReleaseKernel(k); k = nullptr; }
        if (out) { clReleaseMemObject(out); out = nullptr; }
        return nullptr;
    };

    out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, n * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out) {
        NNOPT_ERROR_FMT("GELUActivation_forward: clCreateBuffer(out) failed (%d)", (int)err);
        return cleanup();
    }

    const int n_i = (int)n;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &input, "in")) return cleanup();
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &out, "out")) return cleanup();
    if (!set_arg_checked(k, 2, sizeof(int), &n_i, "n")) return cleanup();

    const size_t gws[1] = { ((n + 255) / 256) * 256 };
    const size_t lws[1] = { 256 };
    err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, lws, 0, nullptr,
                                 KernelProfiler::event_for("gelu_tanh"));
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("GELUActivation_forward: dispatch gelu_tanh failed (%d)", (int)err);
        return cleanup();
    }

    NNOPT_DEBUG_SYNC(queue);

    // Success: caller owns out.
    if (k) { clReleaseKernel(k); k = nullptr; }
    return out;
}
}
