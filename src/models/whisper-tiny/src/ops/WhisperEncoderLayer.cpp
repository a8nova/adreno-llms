// WhisperEncoderLayer.cpp — shared implementation for ALL 5 WhisperEncoderLayer node(s).
//
// You write the WhisperEncoderLayer forward() math ONCE here. backbone.cpp calls
// this function once per traced node of class WhisperEncoderLayer, passing the
// appropriate weight_prefix string for each call — your implementation
// uses that prefix to load weights, so the same C++ math correctly
// handles every sibling instance.
//
// Sibling nodes (order  dump_name  weight_prefix):
//      0  model_encoder_frontend_out                weight_prefix=<none>
//      7  layer_0                                   weight_prefix=<none>
//     14  layer_1                                   weight_prefix=<none>
//     21  layer_2                                   weight_prefix=<none>
//     28  layer_3                                   weight_prefix=<none>
//
// Representative shape (from sibling 0):
//   input:  [1, 1500, 384]
//   output: [1, 1500, 384]
//
// Primary reference dump for cosine validation:
//   reference/layers/model_encoder_frontend_out_output.bin
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
//    `// Reference: model_info/transformers_src/<file>.py:<lines> WhisperEncoderLayer.forward`
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

// Reference: model_info/transformers_src/modeling_whisper.py:342-399 WhisperEncoderLayer.forward

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include <CL/cl.h>
#include <cstdio>
#include <string>

extern "C" {

extern cl_mem LayerNorm_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);

extern cl_mem WhisperSdpaAttention_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);

extern cl_mem Linear_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);

extern cl_mem GELUActivation_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);

cl_mem WhisperEncoderLayer_forward(
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
    (void)start_pos;
    (void)k_cache_inout;
    (void)v_cache_inout;
    (void)encoder_hidden_states;
    (void)weight_prefix;

    if (layer_idx < 0 || layer_idx >= MODEL_CONFIG::ENCODER_LAYERS) {
        NNOPT_ERROR_FMT("WhisperEncoderLayer_forward: invalid layer_idx=%d", layer_idx);
        return nullptr;
    }

    const size_t n = (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE;
    // No _FMT variant exists for INPUT; build a formatted name.
    {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "layer_%d", layer_idx);
        NNOPT_LAYER_CHECK_INPUT(nm, queue, input, n);
    }

    cl_mem normed1 = nullptr;
    cl_mem attn_out = nullptr;
    cl_mem res1 = nullptr;
    cl_mem normed2 = nullptr;
    cl_mem fc1 = nullptr;
    cl_mem act = nullptr;
    cl_mem fc2 = nullptr;
    cl_mem out = nullptr;

    auto cleanup = [&]() -> cl_mem {
        if (normed1) { clReleaseMemObject(normed1); normed1 = nullptr; }
        if (attn_out) { clReleaseMemObject(attn_out); attn_out = nullptr; }
        if (res1) { clReleaseMemObject(res1); res1 = nullptr; }
        if (normed2) { clReleaseMemObject(normed2); normed2 = nullptr; }
        if (fc1) { clReleaseMemObject(fc1); fc1 = nullptr; }
        if (act) { clReleaseMemObject(act); act = nullptr; }
        if (fc2) { clReleaseMemObject(fc2); fc2 = nullptr; }
        if (out) { clReleaseMemObject(out); out = nullptr; }
        return nullptr;
    };

    // self_attn_layer_norm
    {
        char wp_ln[128];
        std::snprintf(wp_ln, sizeof(wp_ln), "model.encoder.layers.%d.self_attn_layer_norm", layer_idx);
        normed1 = LayerNorm_forward(
            cl_ctx, weights, queue,
            input, seq_len, layer_idx, /*start_pos=*/0,
            nullptr, nullptr,
            nullptr,
            wp_ln);
        if (!normed1) return cleanup();
    }

    // self_attn
    // self-attn weight prefix must match safetensors keys; use encoder layer's self_attn module.
    {
        char wp_attn[128];
        std::snprintf(wp_attn, sizeof(wp_attn), "model.encoder.layers.%d.self_attn", layer_idx);
        attn_out = WhisperSdpaAttention_forward(
            cl_ctx, weights, queue,
            normed1, seq_len, layer_idx, /*start_pos=*/0,
            nullptr, nullptr,
            nullptr,
            wp_attn);
    }
    if (!attn_out) return cleanup();

    // residual add 1
    // PROGRAM-INIT-OK: function-local static cache
    static cl_program s_utils_prog = nullptr;
    if (!s_utils_prog) {
        s_utils_prog = cl_ctx.build_program_from_file("kernels/utils.cl");
        if (!s_utils_prog) {
            NNOPT_ERROR("WhisperEncoderLayer_forward: failed to build kernels/utils.cl");
            return cleanup();
        }
    }

    res1 = element_add(queue, s_utils_prog, input, attn_out, n);
    if (!res1) return cleanup();
    if (layer_idx == 0) { NNOPT_LAYER_CHECK("enc0_res1_dbg", queue, res1, n); }

    // final_layer_norm
    {
        char wp_ln[128];
        std::snprintf(wp_ln, sizeof(wp_ln), "model.encoder.layers.%d.final_layer_norm", layer_idx);
        normed2 = LayerNorm_forward(
            cl_ctx, weights, queue,
            res1, seq_len, layer_idx, /*start_pos=*/0,
            nullptr, nullptr,
            nullptr,
            wp_ln);
        if (!normed2) return cleanup();
    }

    // fc1 -> gelu -> fc2
    {
        char wp_fc1[128];
        std::snprintf(wp_fc1, sizeof(wp_fc1), "model.encoder.layers.%d.fc1", layer_idx);
        fc1 = Linear_forward(cl_ctx, weights, queue, normed2, seq_len, layer_idx, 0, nullptr, nullptr, nullptr, wp_fc1);
        if (!fc1) return cleanup();
    }

    act = GELUActivation_forward(cl_ctx, weights, queue, fc1, seq_len, layer_idx, 0, nullptr, nullptr, nullptr, "");
    if (!act) return cleanup();

    {
        char wp_fc2[128];
        std::snprintf(wp_fc2, sizeof(wp_fc2), "model.encoder.layers.%d.fc2", layer_idx);
        fc2 = Linear_forward(cl_ctx, weights, queue, act, seq_len, layer_idx, 0, nullptr, nullptr, nullptr, wp_fc2);
        if (!fc2) return cleanup();
    }

    // residual add 2
    out = element_add(queue, s_utils_prog, res1, fc2, n);
    if (!out) return cleanup();

    // Success: release intermediates but keep `out` for caller.
    if (normed1) { clReleaseMemObject(normed1); normed1 = nullptr; }
    if (attn_out) { clReleaseMemObject(attn_out); attn_out = nullptr; }
    if (res1) { clReleaseMemObject(res1); res1 = nullptr; }
    if (normed2) { clReleaseMemObject(normed2); normed2 = nullptr; }
    if (fc1) { clReleaseMemObject(fc1); fc1 = nullptr; }
    if (act) { clReleaseMemObject(act); act = nullptr; }
    if (fc2) { clReleaseMemObject(fc2); fc2 = nullptr; }

    NNOPT_LAYER_CHECK_FMT("layer_%d", layer_idx, queue, out, n);
    return out;
}

}
