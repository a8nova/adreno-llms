// Reference: model_info/transformers_src/modeling_gemma3.py:134-165 Gemma3DecoderLayer.forward
// Gemma3DecoderLayer.cpp — shared implementation for ALL 18 Gemma3DecoderLayer node(s).
//
// You write the Gemma3DecoderLayer forward() math ONCE here. backbone.cpp calls
// this function once per traced node of class Gemma3DecoderLayer, passing the
// appropriate weight_prefix string for each call — your implementation
// uses that prefix to load weights, so the same C++ math correctly
// handles every sibling instance.
//
// Sibling nodes (order  dump_name  weight_prefix):
//      3  embedding                                 weight_prefix=<none>
//     36  layer_1                                   weight_prefix=<none>
//     53  layer_2                                   weight_prefix=<none>
//     70  layer_3                                   weight_prefix=<none>
//     87  layer_4                                   weight_prefix=<none>
//    104  layer_5                                   weight_prefix=<none>
//    121  layer_6                                   weight_prefix=<none>
//    138  layer_7                                   weight_prefix=<none>
//    155  layer_8                                   weight_prefix=<none>
//    172  layer_9                                   weight_prefix=<none>
//    189  layer_10                                  weight_prefix=<none>
//    206  layer_11                                  weight_prefix=<none>
//   … (+6 more siblings — see .nnport/scaffold_manifest.json)
//
// Representative shape (from sibling 0):
//   input:  [1, 7, 640]
//   output: [1, 7, 640]
//
// Primary reference dump for cosine validation:
//   reference/layers/embedding_output.bin
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
//    `// Reference: model_info/transformers_src/<file>.py:<lines> Gemma3DecoderLayer.forward`
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

// Reference: model_info/transformers_src/modeling_gemma3.py:134-165 Gemma3DecoderLayer.forward
//   r = x; x = input_layernorm(x); x = self_attn(x); x = post_attention_layernorm(x); x = r + x
//   r = x; x = pre_feedforward_layernorm(x); x = mlp(x); x = post_feedforward_layernorm(x); x = r + x

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include <string>
#include <cstdio>

// Composite primitives (defined in sibling op files).
cl_mem gemma3_rmsnorm_run(OpenCLContext& cl_ctx, cl_command_queue queue,
                          cl_mem input, int rows, int cols, float eps,
                          cl_mem weight_buf);
// fp32-residual variant: reads fp32 input residual, writes fp16 normed output.
cl_mem gemma3_rmsnorm_run_f32in(OpenCLContext& cl_ctx, cl_command_queue queue,
                                cl_mem input_f32, int rows, int cols, float eps,
                                cl_mem weight_buf);
extern "C" cl_mem Gemma3Attention_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem Gemma3MLP_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);

extern "C" {
cl_mem Gemma3DecoderLayer_forward(
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
    (void)encoder_hidden_states;
    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();
    const int M = seq_len;
    const int H = MODEL_CONFIG::HIDDEN_SIZE;
    const float eps = MODEL_CONFIG::RMS_NORM_EPS;
    const size_t n = (size_t)M * H;
    static cl_program utils_prog = nullptr;
    if (!utils_prog) utils_prog = cl_ctx.build_program_from_file("kernels/utils.cl");  // PROGRAM-INIT-OK
    if (!utils_prog) { NNOPT_ERROR("DecoderLayer: build utils.cl failed"); return nullptr; }

    cl_mem in_norm_w   = weights.get_buffer(wp + ".input_layernorm.weight");
    cl_mem post_attn_w = weights.get_buffer(wp + ".post_attention_layernorm.weight");
    cl_mem pre_ff_w    = weights.get_buffer(wp + ".pre_feedforward_layernorm.weight");
    cl_mem post_ff_w   = weights.get_buffer(wp + ".post_feedforward_layernorm.weight");
    if (!in_norm_w || !post_attn_w || !pre_ff_w || !post_ff_w) {
        NNOPT_ERROR_FMT("DecoderLayer: missing norm weights for %s", wp.c_str());
        return nullptr;
    }

    // NOTE: Gemma3 residual stream grows past fp16 max (~65504) by mid-stack
    // (reference layer-7 output max ~67487). It MUST be kept in fp32. `input`
    // and the returned `out` are RAW fp32 buffers (float, not nnopt_storage_t).
    // Sublayer activations (normed/attn/mlp) stay fp16 — they are small.
    cl_mem normed = nullptr, attn = nullptr, attn_n = nullptr, res1 = nullptr;
    cl_mem ff_normed = nullptr, mlp = nullptr, mlp_n = nullptr, out = nullptr;
    auto cleanup = [&]() -> cl_mem {
        if (normed)    { nnopt_pool_free(normed);    normed    = nullptr; }
        if (attn)      { nnopt_pool_free(attn);      attn      = nullptr; }
        if (attn_n)    { nnopt_pool_free(attn_n);    attn_n    = nullptr; }
        if (res1)      { nnopt_pool_free(res1);      res1      = nullptr; }
        if (ff_normed) { nnopt_pool_free(ff_normed); ff_normed = nullptr; }
        if (mlp)       { nnopt_pool_free(mlp);       mlp       = nullptr; }
        if (mlp_n)     { nnopt_pool_free(mlp_n);     mlp_n     = nullptr; }
        if (out)       { nnopt_pool_free(out);       out       = nullptr; }
        return nullptr;
    };

    // fp32 residual-add: out_f32 = a_f32 + b_f16. Builds the gemma3_add_f32
    // kernel lazily (cached static). Returns a new fp32 buffer (caller owns).
    static cl_program g_add_prog = nullptr;
    static cl_kernel  g_add_kern = nullptr;
    if (!g_add_kern) {
        g_add_prog = cl_ctx.build_program_from_file("kernels/gemma3_ops.cl");  // PROGRAM-INIT-OK
        if (!g_add_prog) { NNOPT_ERROR("DecoderLayer: build gemma3_ops.cl failed"); return cleanup(); }
        cl_int aerr = CL_SUCCESS;
        g_add_kern = clCreateKernel(g_add_prog, "gemma3_add_f32", &aerr);
        if (aerr != CL_SUCCESS || !g_add_kern) { NNOPT_ERROR_FMT("DecoderLayer: kernel gemma3_add_f32 %d", aerr); return cleanup(); }
    }
    auto add_f32 = [&](cl_mem a_f32, cl_mem b_f16) -> cl_mem {
        cl_int e = CL_SUCCESS;
        cl_mem r = nnopt_pool_alloc(cl_ctx.context(), n * sizeof(float), &e);
        if (e != CL_SUCCESS || !r) { NNOPT_ERROR_FMT("DecoderLayer: alloc add_f32 %d", e); return nullptr; }
        int ni = (int)n;
        if (!set_arg_checked(g_add_kern, 0, sizeof(cl_mem), &a_f32, "a"))  { nnopt_pool_free(r); return nullptr; }
        if (!set_arg_checked(g_add_kern, 1, sizeof(cl_mem), &b_f16, "b"))  { nnopt_pool_free(r); return nullptr; }
        if (!set_arg_checked(g_add_kern, 2, sizeof(cl_mem), &r, "out"))    { nnopt_pool_free(r); return nullptr; }
        if (!set_arg_checked(g_add_kern, 3, sizeof(int), &ni, "n"))        { nnopt_pool_free(r); return nullptr; }
        size_t gws = n;
        e = clEnqueueNDRangeKernel(queue, g_add_kern, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("DecoderLayer: add_f32 dispatch %d", e); nnopt_pool_free(r); return nullptr; }
        return r;
    };

    const std::string attn_wp = wp + ".self_attn";
    const std::string mlp_wp  = wp + ".mlp";

    // ── Attention block ── (input is fp32 residual)
    normed = gemma3_rmsnorm_run_f32in(cl_ctx, queue, input, M, H, eps, in_norm_w);
    if (!normed) { NNOPT_ERROR("DecoderLayer: input_layernorm failed"); return cleanup(); }
    char dn[96]; snprintf(dn, sizeof(dn), "model_layers_%d_input_layernorm", layer_idx);
    NNOPT_LAYER_CHECK(dn, queue, normed, n);

    attn = Gemma3Attention_forward(cl_ctx, weights, queue, normed, M, layer_idx, start_pos,
                                   k_cache_inout, v_cache_inout, nullptr, attn_wp.c_str());
    if (!attn) { NNOPT_ERROR("DecoderLayer: self_attn failed"); return cleanup(); }
    snprintf(dn, sizeof(dn), "model_layers_%d_attn", layer_idx);
    NNOPT_LAYER_CHECK(dn, queue, attn, n);

    attn_n = gemma3_rmsnorm_run(cl_ctx, queue, attn, M, H, eps, post_attn_w);
    if (!attn_n) { NNOPT_ERROR("DecoderLayer: post_attention_layernorm failed"); return cleanup(); }

    // res1 = input(fp32 residual) + attn_n(fp16). Stays fp32.
    res1 = add_f32(input, attn_n);
    if (!res1) { NNOPT_ERROR("DecoderLayer: residual1 add failed"); return cleanup(); }
    nnopt_pool_free(normed); normed = nullptr;
    nnopt_pool_free(attn);   attn   = nullptr;
    nnopt_pool_free(attn_n); attn_n = nullptr;

    // ── Feedforward block ── (res1 is fp32 residual)
    ff_normed = gemma3_rmsnorm_run_f32in(cl_ctx, queue, res1, M, H, eps, pre_ff_w);
    if (!ff_normed) { NNOPT_ERROR("DecoderLayer: pre_feedforward_layernorm failed"); return cleanup(); }

    mlp = Gemma3MLP_forward(cl_ctx, weights, queue, ff_normed, M, layer_idx, start_pos,
                            nullptr, nullptr, nullptr, mlp_wp.c_str());
    if (!mlp) { NNOPT_ERROR("DecoderLayer: mlp failed"); return cleanup(); }
    snprintf(dn, sizeof(dn), "model_layers_%d_mlp", layer_idx);
    NNOPT_LAYER_CHECK(dn, queue, mlp, n);

    mlp_n = gemma3_rmsnorm_run(cl_ctx, queue, mlp, M, H, eps, post_ff_w);
    if (!mlp_n) { NNOPT_ERROR("DecoderLayer: post_feedforward_layernorm failed"); return cleanup(); }

    // out = res1(fp32 residual) + mlp_n(fp16). Stays fp32 → next layer / final norm.
    out = add_f32(res1, mlp_n);
    if (!out) { NNOPT_ERROR("DecoderLayer: residual2 add failed"); return cleanup(); }

    nnopt_pool_free(res1);      res1      = nullptr;
    nnopt_pool_free(ff_normed); ff_normed = nullptr;
    nnopt_pool_free(mlp);       mlp       = nullptr;
    nnopt_pool_free(mlp_n);     mlp_n     = nullptr;
    return out;
}
}
