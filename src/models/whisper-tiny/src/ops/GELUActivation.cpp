// GELUActivation.cpp — shared implementation for ALL 8 GELUActivation node(s).
//
// You write the GELUActivation forward() math ONCE here. backbone.cpp calls
// this function once per traced node of class GELUActivation, passing the
// appropriate weight_prefix string for each call — your implementation
// uses that prefix to load weights, so the same C++ math correctly
// handles every sibling instance.
//
// Sibling nodes (order  dump_name  weight_prefix):
//      5  activation_fn_0                           weight_prefix=<none>
//     12  activation_fn_1                           weight_prefix=<none>
//     19  activation_fn_2                           weight_prefix=<none>
//     26  activation_fn_3                           weight_prefix=<none>
//     37  decoder_activation_fn_0                   weight_prefix=<none>
//     45  decoder_activation_fn_1                   weight_prefix=<none>
//     54  decoder_activation_fn_2                   weight_prefix=<none>
//     63  decoder_activation_fn_3                   weight_prefix=<none>
//
// Representative shape (from sibling 0):
//   input:  [1, 1500, 1536]
//   output: [1, 1500, 1536]
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

// Reference: model_info/transformers_src/modeling_whisper.py:250-340 WhisperEncoderLayer.forward / WhisperDecoderLayer.forward (activation_fn = GELU)

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"

#include <CL/cl.h>
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
    (void)weights; (void)layer_idx; (void)start_pos;
    (void)k_cache_inout; (void)v_cache_inout;
    (void)encoder_hidden_states; (void)weight_prefix;

    if (!input) {
        NNOPT_ERROR("GELUActivation_forward: input is null");
        return nullptr;
    }

    // Determine columns from which node called us.
    // Encoder MLP uses ENCODER_FFN_DIM; decoder MLP uses DECODER_FFN_DIM.
    // We detect decoder by whether this op is invoked from inside a decoder layer.
    // Reference: model_info/transformers_src/modeling_whisper.py (WhisperDecoderLayer.forward / WhisperEncoderLayer.forward)
    const int rows = seq_len;
    const bool is_decoder = (weight_prefix && std::string(weight_prefix).find("decoder") != std::string::npos);
    const int cols = is_decoder ? (int)MODEL_CONFIG::DECODER_FFN_DIM
                                : (int)MODEL_CONFIG::ENCODER_FFN_DIM;
    const int n = rows * cols;

    cl_int err = CL_SUCCESS;
    cl_mem out = nullptr;
    cl_program prog = nullptr;
    cl_kernel k = nullptr;

    auto cleanup = [&]() -> cl_mem {
        if (k) { clReleaseKernel(k); k = nullptr; }
        if (prog) { clReleaseProgram(prog); prog = nullptr; }
        if (out) { clReleaseMemObject(out); out = nullptr; }
        return nullptr;
    };

    static cl_program s_prog = nullptr;
    static cl_kernel s_k = nullptr;
    if (!s_prog) {
        // PROGRAM-INIT-OK: function-local static cache
        s_prog = cl_ctx.build_program_from_file("kernels/gelu.cl");
        if (!s_prog) {
            NNOPT_ERROR("GELUActivation_forward: failed to build kernels/gelu.cl");
            return nullptr;
        }
        cl_int kerr = CL_SUCCESS;
        s_k = clCreateKernel(s_prog, "gelu_forward", &kerr);
        if (kerr != CL_SUCCESS || !s_k) {
            NNOPT_ERROR_FMT("GELUActivation_forward: clCreateKernel gelu_forward failed %d", (int)kerr);
            clReleaseProgram(s_prog); s_prog = nullptr;
            return nullptr;
        }
    }
    prog = s_prog; k = s_k;
    clRetainProgram(prog);
    clRetainKernel(k);

    out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                         (size_t)n * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out) {
        NNOPT_ERROR_FMT("GELUActivation_forward: clCreateBuffer(out) %d", (int)err);
        return cleanup();
    }

    if (!set_arg_checked(k, 0, sizeof(cl_mem), &input, "x")) return cleanup();
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &out, "y")) return cleanup();
    if (!set_arg_checked(k, 2, sizeof(int), &n, "n")) return cleanup();

    size_t gws = (size_t)n;
    err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("gelu_forward"));
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("GELUActivation_forward: dispatch failed %d", (int)err);
        return cleanup();
    }

    // SYNC-01: in-order queue — the next enqueued kernel consumes `out` and is
    // automatically serialized after this GELU. No clFinish needed. (Was a
    // per-token pipeline stall in decode; removed — see benchmark.md opt #8.)

    if (k) { clReleaseKernel(k); k = nullptr; }
    if (prog) { clReleaseProgram(prog); prog = nullptr; }
    return out;
}
}
