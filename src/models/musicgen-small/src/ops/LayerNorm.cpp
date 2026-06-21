// LayerNorm.cpp — shared implementation for ALL 72 LayerNorm node(s).
//
// You write the LayerNorm forward() math ONCE here. backbone.cpp calls
// this function once per traced node of class LayerNorm, passing the
// appropriate weight_prefix string for each call — your implementation
// uses that prefix to load weights, so the same C++ math correctly
// handles every sibling instance.
//
// Sibling nodes (order  dump_name  weight_prefix):
//     20  self_attn_layer_norm_0                    weight_prefix=decoder.model.decoder.layers.0.self_attn_layer_norm
//     22  encoder_attn_layer_norm_0                 weight_prefix=decoder.model.decoder.layers.0.encoder_attn_layer_norm
//     24  final_layer_norm_0                        weight_prefix=decoder.model.decoder.layers.0.final_layer_norm
//     29  self_attn_layer_norm_1                    weight_prefix=decoder.model.decoder.layers.1.self_attn_layer_norm
//     31  encoder_attn_layer_norm_1                 weight_prefix=decoder.model.decoder.layers.1.encoder_attn_layer_norm
//     33  final_layer_norm_1                        weight_prefix=decoder.model.decoder.layers.1.final_layer_norm
//     38  self_attn_layer_norm_2                    weight_prefix=decoder.model.decoder.layers.2.self_attn_layer_norm
//     40  encoder_attn_layer_norm_2                 weight_prefix=decoder.model.decoder.layers.2.encoder_attn_layer_norm
//     42  final_layer_norm_2                        weight_prefix=decoder.model.decoder.layers.2.final_layer_norm
//     47  self_attn_layer_norm_3                    weight_prefix=decoder.model.decoder.layers.3.self_attn_layer_norm
//     49  encoder_attn_layer_norm_3                 weight_prefix=decoder.model.decoder.layers.3.encoder_attn_layer_norm
//     51  final_layer_norm_3                        weight_prefix=decoder.model.decoder.layers.3.final_layer_norm
//   … (+60 more siblings — see .nnport/scaffold_manifest.json)
//
// Representative shape (from sibling 0):
//   input:  [1, 1, 1024]
//   output: [1, 1, 1024]
//
// Primary reference dump for cosine validation:
//   reference/layers/self_attn_layer_norm_0_output.bin
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
//    `// Reference: model_info/transformers_src/<file>.py:<lines> LayerNorm.forward`
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

// Reference: model_info/transformers_src/modeling_musicgen.py (nn.LayerNorm usage)

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"

#include <CL/cl.h>
#include <string>

// Internal worker. out_opt != nullptr → write into the CALLER-OWNED buffer
// (no allocation, never released here) and return it. out_opt == nullptr →
// allocate a fresh output (legacy behavior; caller releases). The out_opt path
// exists for the recordable-replay decode loop, which needs every per-step
// cl_mem to be persistent so the recorded dispatch handles stay valid.
static cl_mem layernorm_dispatch_impl(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem input,
    int seq_len,
    const char* weight_prefix,
    cl_mem out_opt)
{
    const bool owned = (out_opt == nullptr);
    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();

    // LayerNorm params are ALWAYS stored as "<prefix>.weight" and "<prefix>.bias" in this scaffold
    // (see .nnport/layer_contracts/Attention.json: self_attn_layer_norm_weight/bias, and
    // .nnport/layer_contracts/EncoderAttnLayerNorm.json, FinalLayerNorm.json, Model.json).
    // The previous "_weight" / "_bias" fallback silently loaded the WRONG tensors (or nullptr)
    // for decoder LN, corrupting self_attn_layer_norm_* immediately.
    cl_mem gamma = weights.get_buffer(wp + ".weight");
    cl_mem beta  = weights.get_buffer(wp + ".bias", /*optional=*/true);

    if (!gamma) {
        NNOPT_ERROR_FMT("LayerNorm_forward: missing gamma tensor for %s.weight", wp.c_str());
        return nullptr;
    }

    // LayerNorm in MusicGen decoder operates on decoder hidden size (1024).
    // Using MODEL_CONFIG::HIDDEN_SIZE is ambiguous across hybrid configs.
    const int cols = MODEL_CONFIG::DECODER_HIDDEN_SIZE;
    const int rows = seq_len;
    const size_t numel = (size_t)rows * (size_t)cols;

    cl_int err = CL_SUCCESS;
    cl_mem out = out_opt;
    if (!out) {
        out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                             numel * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("LayerNorm_forward: clCreateBuffer(out) failed: %d", (int)err);
            return nullptr;
        }
    }
    (void)numel;

    // Build program once per process.
    static cl_program prog = nullptr;
    static cl_kernel kern = nullptr;
    if (!prog) {
        prog = cl_ctx.build_program_from_file("kernels/layernorm_simple.cl");
        if (!prog) {
            NNOPT_ERROR("LayerNorm_forward: build_program_from_file(layernorm_simple.cl) failed");
            clReleaseMemObject(out);
            return nullptr;
        }
        kern = clCreateKernel(prog, "layernorm_simple", &err);
        if (err != CL_SUCCESS || !kern) {
            NNOPT_ERROR_FMT("LayerNorm_forward: clCreateKernel(layernorm_simple) failed: %d", (int)err);
            clReleaseMemObject(out);
            return nullptr;
        }
    }

    const float eps = MODEL_CONFIG::LAYER_NORM_EPS;

    // LayerNorm in HF is ALWAYS affine when the module has a weight (gamma).
    // Some of our traced graphs omit the bias tensor (beta). In that case,
    // PyTorch effectively uses beta=0.
    const int has_gamma = (gamma != nullptr) ? 1 : 0;
    const int has_beta  = (beta  != nullptr) ? 1 : 0;

    // Never pass nullptr to kernel args.
    // If beta missing, pass any non-null buffer (unused when has_beta==0).
    cl_mem gamma_arg = gamma ? gamma : weights.get_buffer("text_encoder.shared.weight", /*optional=*/true);
    cl_mem beta_arg  = beta  ? beta  : gamma_arg;

    int arg = 0;
    err = clSetKernelArg(kern, arg++, sizeof(cl_mem), &input);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm_forward: arg in %d", (int)err); if (owned) clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(kern, arg++, sizeof(cl_mem), &out);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm_forward: arg out %d", (int)err); if (owned) clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(kern, arg++, sizeof(int), &rows);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm_forward: arg rows %d", (int)err); if (owned) clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(kern, arg++, sizeof(int), &cols);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm_forward: arg cols %d", (int)err); if (owned) clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(kern, arg++, sizeof(float), &eps);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm_forward: arg eps %d", (int)err); if (owned) clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(kern, arg++, sizeof(cl_mem), &gamma_arg);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm_forward: arg gamma %d", (int)err); if (owned) clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(kern, arg++, sizeof(cl_mem), &beta_arg);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm_forward: arg beta %d", (int)err); if (owned) clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(kern, arg++, sizeof(int), &has_gamma);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm_forward: arg has_gamma %d", (int)err); if (owned) clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(kern, arg++, sizeof(int), &has_beta);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm_forward: arg has_beta %d", (int)err); if (owned) clReleaseMemObject(out); return nullptr; }

    // One workgroup (LN_WG=256, matching layernorm_simple.cl) per row: the
    // kernel does a local reduction over `cols` instead of one serial thread
    // walking all 1024 columns (was 1.24ms/call — 95% of GPU time per profile).
    constexpr size_t kLnWg = 256;
    size_t lws[1] = {kLnWg};
    size_t gws[1] = {(size_t)rows * kLnWg};
    err = clEnqueueNDRangeKernel(queue, kern, 1, nullptr, gws, lws, 0, nullptr,
                                 KernelProfiler::event_for(("layernorm_simple:" + wp).c_str()));
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("LayerNorm_forward: dispatch failed: %d", (int)err);
        if (owned) clReleaseMemObject(out);
        return nullptr;
    }

    NNOPT_DEBUG_SYNC(queue);
    return out;
}

extern "C" {
cl_mem LayerNorm_forward(
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
    return layernorm_dispatch_impl(cl_ctx, weights, queue, input, seq_len,
                                   weight_prefix, /*out_opt=*/nullptr);
}

// Recordable-replay variant: LayerNorm into a persistent caller-owned buffer
// (sized >= seq_len*DECODER_HIDDEN_SIZE storage_t). Returns false on failure.
bool LayerNorm_forward_into(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem input,
    int seq_len,
    const char* weight_prefix,
    cl_mem out)
{
    return layernorm_dispatch_impl(cl_ctx, weights, queue, input, seq_len,
                                   weight_prefix, out) != nullptr;
}
}
