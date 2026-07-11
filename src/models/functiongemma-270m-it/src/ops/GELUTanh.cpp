// Reference: model_info/transformers_src/modeling_gemma3.py ACT2FN["gelu_pytorch_tanh"]
// GELUTanh.cpp — shared implementation for ALL 18 GELUTanh node(s).
//
// You write the GELUTanh forward() math ONCE here. backbone.cpp calls
// this function once per traced node of class GELUTanh, passing the
// appropriate weight_prefix string for each call — your implementation
// uses that prefix to load weights, so the same C++ math correctly
// handles every sibling instance.
//
// Sibling nodes (order  dump_name  weight_prefix):
//     15  model_layers_0_mlp_act_fn                 weight_prefix=<none>
//     31  model_layers_1_mlp_act_fn                 weight_prefix=<none>
//     48  model_layers_2_mlp_act_fn                 weight_prefix=<none>
//     65  model_layers_3_mlp_act_fn                 weight_prefix=<none>
//     82  model_layers_4_mlp_act_fn                 weight_prefix=<none>
//     99  model_layers_5_mlp_act_fn                 weight_prefix=<none>
//    116  model_layers_6_mlp_act_fn                 weight_prefix=<none>
//    133  model_layers_7_mlp_act_fn                 weight_prefix=<none>
//    150  model_layers_8_mlp_act_fn                 weight_prefix=<none>
//    167  model_layers_9_mlp_act_fn                 weight_prefix=<none>
//    184  model_layers_10_mlp_act_fn                weight_prefix=<none>
//    201  model_layers_11_mlp_act_fn                weight_prefix=<none>
//   … (+6 more siblings — see .nnport/scaffold_manifest.json)
//
// Representative shape (from sibling 0):
//   input:  [1, 7, 2048]
//   output: [1, 7, 2048]
//
// Primary reference dump for cosine validation:
//   reference/layers/model_layers_0_mlp_act_fn_output.bin
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
//    `// Reference: model_info/transformers_src/<file>.py:<lines> GELUTanh.forward`
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

// Reference: model_info/transformers_src/modeling_gemma3.py ACT2FN["gelu_pytorch_tanh"]
//   gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"
#include <string>

namespace {
cl_program g_prog = nullptr;
cl_kernel  g_kern = nullptr;
bool ensure_kernel(OpenCLContext& cl_ctx) {
    if (g_kern) return true;
    g_prog = cl_ctx.build_program_from_file("kernels/gemma3_ops.cl");  // PROGRAM-INIT-OK
    if (!g_prog) { NNOPT_ERROR("GELU: build gemma3_ops.cl failed"); return false; }
    cl_int err = CL_SUCCESS;
    g_kern = clCreateKernel(g_prog, "gemma3_gelu_tanh", &err);
    if (err != CL_SUCCESS || !g_kern) { NNOPT_ERROR_FMT("GELU: clCreateKernel %d", err); return false; }
    return true;
}
} // namespace

// Reusable: applies gelu-tanh to n elements, returns new buffer.
cl_mem gemma3_gelu_run(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem input, int n) {
    if (!ensure_kernel(cl_ctx)) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_mem out = nnopt_pool_alloc(cl_ctx.context(),
                                (size_t)n * sizeof(nnopt_storage_t), &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("GELU: alloc out %d", err); return nullptr; }
    auto cleanup = [&]() -> cl_mem { if (out) { nnopt_pool_free(out); out = nullptr; } return nullptr; };
    if (!set_arg_checked(g_kern, 0, sizeof(cl_mem), &input, "x")) return cleanup();
    if (!set_arg_checked(g_kern, 1, sizeof(cl_mem), &out, "out")) return cleanup();
    if (!set_arg_checked(g_kern, 2, sizeof(int), &n, "n")) return cleanup();
    size_t gws = (size_t)n;
    err = clEnqueueNDRangeKernel(queue, g_kern, 1, nullptr, &gws, nullptr,
                                 0, nullptr, KernelProfiler::event_for("op_gemma3_gelu"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("GELU: dispatch %d", err); return cleanup(); }
    return out;
}

extern "C" {
cl_mem GELUTanh_forward(
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
    (void)k_cache_inout; (void)v_cache_inout; (void)encoder_hidden_states; (void)weight_prefix;
    int n = seq_len * MODEL_CONFIG::INTERMEDIATE_SIZE;
    return gemma3_gelu_run(cl_ctx, queue, input, n);
}
}
