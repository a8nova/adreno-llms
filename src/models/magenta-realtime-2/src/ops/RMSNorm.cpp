// RMSNorm.cpp — shared implementation for ALL 80 RMSNorm node(s).
//
// You write the RMSNorm forward() math ONCE here. backbone.cpp calls
// this function once per traced node of class RMSNorm, passing the
// appropriate weight_prefix string for each call — your implementation
// uses that prefix to load weights, so the same C++ math correctly
// handles every sibling instance.
//
// Sibling nodes (order  dump_name  weight_prefix):
//      4  sampler.depthformer.decoder.temporal_body.layers.0.layers.0.layers.0.body.layers.0._rms_norm  weight_prefix=sampler.depthformer.decoder.temporal_body.layers.0.layers.0.layers.0.body.layers.0._rms_norm
//      5  sampler.depthformer.decoder.temporal_body.layers.0.layers.0.layers.0.body.layers.4._rms_norm  weight_prefix=sampler.depthformer.decoder.temporal_body.layers.0.layers.0.layers.0.body.layers.4._rms_norm
//      6  sampler.depthformer.decoder.temporal_body.layers.0.layers.0.layers.1.body.layers.0._rms_norm  weight_prefix=sampler.depthformer.decoder.temporal_body.layers.0.layers.0.layers.1.body.layers.0._rms_norm
//      7  sampler.depthformer.decoder.temporal_body.layers.0.layers.0.layers.1.body.layers.4._rms_norm  weight_prefix=sampler.depthformer.decoder.temporal_body.layers.0.layers.0.layers.1.body.layers.4._rms_norm
//      8  sampler.depthformer.decoder.temporal_body.layers.0.layers.0.layers.2.body.layers.0._rms_norm  weight_prefix=sampler.depthformer.decoder.temporal_body.layers.0.layers.0.layers.2.body.layers.0._rms_norm
//     11  sampler.depthformer.decoder.temporal_body.layers.0.layers.0.layers.2.body.layers.5._rms_norm  weight_prefix=sampler.depthformer.decoder.temporal_body.layers.0.layers.0.layers.2.body.layers.5._rms_norm
//     12  sampler.depthformer.decoder.temporal_body.layers.0.layers.1.layers.0.body.layers.0._rms_norm  weight_prefix=sampler.depthformer.decoder.temporal_body.layers.0.layers.1.layers.0.body.layers.0._rms_norm
//     13  sampler.depthformer.decoder.temporal_body.layers.0.layers.1.layers.0.body.layers.4._rms_norm  weight_prefix=sampler.depthformer.decoder.temporal_body.layers.0.layers.1.layers.0.body.layers.4._rms_norm
//     14  sampler.depthformer.decoder.temporal_body.layers.0.layers.1.layers.1.body.layers.0._rms_norm  weight_prefix=sampler.depthformer.decoder.temporal_body.layers.0.layers.1.layers.1.body.layers.0._rms_norm
//     15  sampler.depthformer.decoder.temporal_body.layers.0.layers.1.layers.1.body.layers.4._rms_norm  weight_prefix=sampler.depthformer.decoder.temporal_body.layers.0.layers.1.layers.1.body.layers.4._rms_norm
//     16  sampler.depthformer.decoder.temporal_body.layers.0.layers.1.layers.2.body.layers.0._rms_norm  weight_prefix=sampler.depthformer.decoder.temporal_body.layers.0.layers.1.layers.2.body.layers.0._rms_norm
//     19  sampler.depthformer.decoder.temporal_body.layers.0.layers.1.layers.2.body.layers.5._rms_norm  weight_prefix=sampler.depthformer.decoder.temporal_body.layers.0.layers.1.layers.2.body.layers.5._rms_norm
//   … (+68 more siblings — see .nnport/scaffold_manifest.json)
//
// Representative shape (from sibling 0):
//   input:  []
//   output: []
//
// Primary reference dump for cosine validation:
//   reference/layers/sampler.depthformer.decoder.temporal_body.layers.0.layers.0.layers.0.body.layers.0._rms_norm_output.bin
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
//    `// Reference: model_info/transformers_src/<file>.py:<lines> RMSNorm.forward`
//    (Build refuses to compile without it.)
// 2. Load every weight the PyTorch forward() touches via
//    `weights.get_buffer(std::string(weight_prefix) + ".<param>")`.
//    Use `weights.get_shape(wp + ".weight")` for dimensions when needed.
// 3. Allocate output buffer:
//    `cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE,
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
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" {
cl_mem RMSNorm_forward(
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
    (void)layer_idx; (void)start_pos;
    (void)k_cache_inout; (void)v_cache_inout; (void)encoder_hidden_states;
    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();
    if (wp.empty() || !input) { NNOPT_ERROR("RMSNorm: null wp/input"); return nullptr; }

    // scale weight (fp16), shape [dim]
    const std::string w_key = wp + ".weight";
    cl_mem scale = weights.get_buffer(w_key);
    if (!scale) { NNOPT_ERROR_FMT("RMSNorm: missing %s", w_key.c_str()); return nullptr; }
    const std::vector<int> sh = weights.get_shape(w_key);
    const int dim = sh.empty() ? 0 : sh.back();
    if (dim <= 0) { NNOPT_ERROR_FMT("RMSNorm: bad dim for %s", w_key.c_str()); return nullptr; }
    const int rows = seq_len;

    // Workgroup-per-row vs the original one-work-item-per-row. This is the pre-norm at the head of
    // every attention and MLP block — 84 per frame — and running a whole row on one lane wastes
    // eleven of twelve compute units. NNOPT_RMS=serial restores the old kernel for A/B (same switch
    // as the fused post-norm: it is the same bug in two places).
    static int par = -1, par_epoch = -1;
    if (par_epoch != nnopt_toggle_epoch()) {
        const char* r = std::getenv("NNOPT_RMS");
        par = (r && std::strcmp(r, "serial") == 0) ? 0 : 1;
        par_epoch = nnopt_toggle_epoch();
    }
    static cl_kernel kern_ser = nullptr, kern_par = nullptr;
    cl_kernel& kslot = par ? kern_par : kern_ser;
    if (!kslot) {
        const char* file = par ? "kernels/rms_norm_wg_f32.cl" : "kernels/rms_norm_f32.cl";
        cl_program prog = cl_ctx.build_program_from_file(file);
        if (!prog) { NNOPT_ERROR_FMT("RMSNorm: build %s failed", file); return nullptr; }
        cl_int e; kslot = clCreateKernel(prog, par ? "rms_norm_wg_f32" : "rms_norm_f32", &e);
        if (e != CL_SUCCESS || !kslot) { NNOPT_ERROR_FMT("RMSNorm: clCreateKernel (err=%d)", e); return nullptr; }
    }
    // One kernel object per call site: a recordable-queue capture references the kernel and NOT its
    // args, so a kernel shared across layers replays every dispatch with the last layer's buffers.
    cl_kernel kern = nnopt_kernel_instance(kslot, wp);
    cl_int err = CL_SUCCESS;
    cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE,
                                (size_t)rows * dim * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("RMSNorm: clCreateBuffer (err=%d)", err); return nullptr; }
    const float eps = 1e-6f;
    clSetKernelArg(kern, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kern, 1, sizeof(cl_mem), &scale);
    clSetKernelArg(kern, 2, sizeof(cl_mem), &out);
    clSetKernelArg(kern, 3, sizeof(int), &dim);
    clSetKernelArg(kern, 4, sizeof(float), &eps);
    // Enqueued through profEnqueue so it SHOWS UP. A raw clEnqueueNDRangeKernel is invisible to the
    // profiler, which is how 8400 serial dispatches per chunk stayed hidden through several rounds
    // of "where is the time going".
    if (par) {
        const size_t gws = (size_t)rows * 128, lws = 128;
        err = cl_ctx.profEnqueue(kern, 1, &gws, &lws, "rms");
    } else {
        const size_t gws = (size_t)rows;
        err = cl_ctx.profEnqueue(kern, 1, &gws, nullptr, "rms");
    }
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("RMSNorm: enqueue (err=%d)", err); pool_free(out); return nullptr; }
    
    return out;
}
}
