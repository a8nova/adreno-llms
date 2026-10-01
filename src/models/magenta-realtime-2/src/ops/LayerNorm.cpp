// LayerNorm.cpp — shared implementation for ALL 2 LayerNorm node(s).
//
// You write the LayerNorm forward() math ONCE here. backbone.cpp calls
// this function once per traced node of class LayerNorm, passing the
// appropriate weight_prefix string for each call — your implementation
// uses that prefix to load weights, so the same C++ math correctly
// handles every sibling instance.
//
// Sibling nodes (order  dump_name  weight_prefix):
//      3  sampler.depthformer.encoder.body.layers.3._layer_norm  weight_prefix=sampler.depthformer.encoder.body.layers.3._layer_norm
//    113  sampler.depthformer.decoder.depth_body.layers.2._layer_norm  weight_prefix=sampler.depthformer.decoder.depth_body.layers.2._layer_norm
//
// Representative shape (from sibling 0):
//   input:  [1, 1, 256]
//   output: [1, 1, 256]
//
// Primary reference dump for cosine validation:
//   reference/layers/sampler.depthformer.encoder.body.layers.3._layer_norm_output.bin
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
    (void)layer_idx; (void)start_pos;
    (void)k_cache_inout; (void)v_cache_inout; (void)encoder_hidden_states;
    // Reference: sequence_layers/mlx/normalization.py LayerNormalization.
    // out = (x-mean)/sqrt(var+eps)*weight + bias  (weight/bias key prefix "<wp>", e.g. "..._layer_norm")
    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();
    if (wp.empty() || !input) { NNOPT_ERROR("LayerNorm: null wp/input"); return nullptr; }
    cl_mem w = weights.get_buffer(wp + ".weight");
    cl_mem b = weights.get_buffer(wp + ".bias");
    if (!w || !b) { NNOPT_ERROR_FMT("LayerNorm: missing %s .weight/.bias", wp.c_str()); return nullptr; }
    const std::vector<int> sh = weights.get_shape(wp + ".weight");
    const int dim = sh.empty() ? 0 : sh.back();
    if (dim <= 0) { NNOPT_ERROR_FMT("LayerNorm: bad dim %s", wp.c_str()); return nullptr; }
    const int rows = seq_len;
    // Workgroup-per-row vs one work item per row (the same serial-kernel bug as RMSNorm and the
    // attentions). This runs at the tail of the depth stack, 12x per frame. NNOPT_RMS=serial keeps
    // every norm on the old kernels for A/B.
    cl_int err = CL_SUCCESS;
    static int par = -1, par_epoch = -1;
    if (par_epoch != nnopt_toggle_epoch()) {
        const char* r = std::getenv("NNOPT_RMS");
        par = (r && std::strcmp(r, "serial") == 0) ? 0 : 1;
        par_epoch = nnopt_toggle_epoch();
    }
    static cl_kernel kern_ser = nullptr, kern_par = nullptr;
    cl_kernel& kslot = par ? kern_par : kern_ser;
    if (!kslot) {
        const char* file = par ? "kernels/layer_norm_wg_f32.cl" : "kernels/layer_norm_f32.cl";
        cl_program p = cl_ctx.build_program_from_file(file);
        if (!p) { NNOPT_ERROR_FMT("LayerNorm: build %s failed", file); return nullptr; }
        kslot = clCreateKernel(p, par ? "layer_norm_wg_f32" : "layer_norm_f32", &err);
        if (err != CL_SUCCESS || !kslot) { NNOPT_ERROR_FMT("LayerNorm: clCreateKernel (err=%d)", err); return nullptr; }
    }
    // Per-call-site kernel object — see the note in RMSNorm.cpp.
    cl_kernel kern = nnopt_kernel_instance(kslot, wp);
    cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)rows*dim*sizeof(float), nullptr, &err);
    if (!out) { NNOPT_ERROR("LayerNorm: alloc"); return nullptr; }
    const float eps = 1e-6f;
    clSetKernelArg(kern,0,sizeof(cl_mem),&input); clSetKernelArg(kern,1,sizeof(cl_mem),&w);
    clSetKernelArg(kern,2,sizeof(cl_mem),&b); clSetKernelArg(kern,3,sizeof(cl_mem),&out);
    clSetKernelArg(kern,4,sizeof(int),&dim); clSetKernelArg(kern,5,sizeof(float),&eps);
    // profEnqueue, not a raw enqueue: a kernel that skips the profiler is one the profile cannot
    // show is slow.
    if (par) { const size_t g=(size_t)rows*128, l=128; err = cl_ctx.profEnqueue(kern,1,&g,&l,"ln"); }
    else     { const size_t g=(size_t)rows;           err = cl_ctx.profEnqueue(kern,1,&g,nullptr,"ln"); }
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm: enqueue (err=%d)", err); pool_free(out); return nullptr; }
    
    return out;
}
}
