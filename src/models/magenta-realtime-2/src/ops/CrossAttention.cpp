// CrossAttention.cpp — StreamingDotProductAttention core (cross-attn to conditioning source).
// Reference: sequence_layers/mlx/attention.py StreamingDotProductAttention.
//
// Q from the (normed) decoder hidden state; K/V from an external source (the encoder
// conditioning, dim 256). Same learnable sink (sink_key/value_embeddings) + per_dim_scale
// as SinkAttention.
//
// ── conditioning ring depth (kMaxSrc) ──────────────────────────────────────────────────────────
// Upstream is use_streaming_cross_attention=True with max_past_horizon=41 for mrt2_small
// (magenta_rt/mlx/model.py:415,495): it re-runs the conditioning encoder EVERY frame
// (mlx/depthformer.py:1085-1088) and keeps a 41-frame KV ring, so a step attends over
// [sink, source x min(pos+1, 42)]. This port encodes the block once per chunk, so it had
// kv = [sink, source] — one key against upstream's 42.
//
// That is not a small difference. The sink is the learned "attend to nothing" escape hatch, and
// shrinking the denominator from 42 sources to 1 hands it far more of the softmax mass. Measured
// against the real mrt2_small weights (2026-09-01, two prompts, 1464 cross-attn calls): sink mass
// 0.57-0.64 upstream vs 0.88 here, i.e. this port was delivering 29-34% of the reference's
// conditioning, with 0.67-0.75 mean relative L2 error on the context vector in all 12 layers.
// Layers 2,3,5,7,8,9,11 sat at 0.93-0.99 sink — effectively no conditioning at all.
//
// The fix is exact rather than an approximation: within a chunk the conditioning is constant, so
// all 42 source keys are bit-identical (measured logit spread exactly 0.0), and attending over N
// identical keys equals attending over one whose logit is raised by ln(N). So the ring costs one
// scalar. Verified against upstream to 2e-06 (float32 noise).
//
// `start_pos` carries the absolute frame index (0-based, persistent across serve-mode chunks,
// reset only with the session) — it was previously unused on this path. NNOPT_XATTNDEPTH overrides
// the depth for A/B: 1 restores the pre-fix behaviour, no rebuild needed.
//
//   q  = hidden @ q_proj.weight.T        [H*D]            (q_proj: in_dim -> H*D)
//   kv = source @ kv_proj.weight.T       [2*H*D]          (kv_proj: src_dim -> 2*H*D)
//   k = kv[:H*D], v = kv[H*D:]
//   attn = sink_attention(q,k,v, sink_k, sink_v, per_dim_scale)   (reuses sink core)
// Weights (weight_prefix = "...body.layers.1.inner"):
//   <wp>.q_proj.weight  [H*D, in_dim] ; <wp>.kv_proj.weight [2*H*D, src_dim]
//   <wp>._per_dim_scale [D] ; <wp>.sink_key_embeddings/.sink_value_embeddings [1,H,D]
// `encoder_hidden_states` carries the source buffer (fp32, [src_dim]).

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>

extern "C" {
cl_mem CrossAttention_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout, cl_mem encoder_hidden_states,
    const char* weight_prefix, const char* prenorm_prefix)
{
    (void)layer_idx;(void)k_cache_inout;(void)v_cache_inout;
    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();
    if (wp.empty() || !input) { NNOPT_ERROR("CrossAttention: null wp/input"); return nullptr; }
    if (!encoder_hidden_states) { NNOPT_ERROR("CrossAttention: null source"); return nullptr; }
    if (seq_len != 1) { NNOPT_ERROR_FMT("CrossAttention: seq_len=%d unsupported", seq_len); return nullptr; }

    cl_mem pds = weights.get_buffer(wp + "._per_dim_scale");
    cl_mem sink_k = weights.get_buffer(wp + ".sink_key_embeddings");
    cl_mem sink_v = weights.get_buffer(wp + ".sink_value_embeddings");
    if (!pds || !sink_k || !sink_v) { NNOPT_ERROR_FMT("CrossAttention: missing weights %s", wp.c_str()); return nullptr; }
    const std::vector<int> sksh = weights.get_shape(wp + ".sink_key_embeddings"); // [1,H,D]
    if (sksh.size()!=3) { NNOPT_ERROR("CrossAttention: bad ranks"); return nullptr; }
    int hd = 0, in_dim = 0, kv_out = 0, src_dim = 0;   // logical dims — Q4 declares [N, K/2]
    if (!nnopt_weight_dims(weights, wp + ".q_proj.weight",  &hd,     &in_dim))  return nullptr;
    if (!nnopt_weight_dims(weights, wp + ".kv_proj.weight", &kv_out, &src_dim)) return nullptr;
    const int H = sksh[1], D = sksh[2];
    if (hd != H*D || kv_out != 2*H*D) { NNOPT_ERROR("CrossAttention: dim mismatch"); return nullptr; }
    const float inv_sqrt_d = 1.0f / std::sqrt((float)D);

    // Ring depth: 41 past frames + the current one. Read once per toggle epoch, like the other
    // NNOPT_ switches, so the per-frame path stays free of getenv.
    static const int kMaxSrc = 42;
    static int max_src = -1, max_src_epoch = -1;
    if (max_src_epoch != nnopt_toggle_epoch()) {
        const char* e = std::getenv("NNOPT_XATTNDEPTH");
        max_src = (e && *e) ? std::atoi(e) : kMaxSrc;
        if (max_src < 1) max_src = 1;
        max_src_epoch = nnopt_toggle_epoch();
    }
    // start_pos < 0 means "caller has no frame counter" (the non-streaming op-test path), which is
    // frame 0 — one source, ln(1) = 0, byte-identical to the old behaviour.
    const int n_src = (start_pos < 0) ? 1
                    : (start_pos + 1 < max_src ? start_pos + 1 : max_src);
    const float log_n = std::log((float)n_src);

    cl_int err = CL_SUCCESS;
    // One workgroup per head instead of one work ITEM per head — the same serial-launch bug as the
    // norms and the cached self-attention. 12 calls per frame. NNOPT_ATTN=serial restores the old
    // kernel (shared switch with CachedSelfAttention: same bug, same toggle).
    static int par = -1, par_epoch = -1;
    if (par_epoch != nnopt_toggle_epoch()) {
        const char* a = std::getenv("NNOPT_ATTN");
        par = (a && std::strcmp(a, "serial") == 0) ? 0 : 1;
        par_epoch = nnopt_toggle_epoch();
    }
    static cl_kernel sk_ser = nullptr, sk_par = nullptr;
    cl_kernel& sk_slot = par ? sk_par : sk_ser;
    if (!sk_slot) {
        const char* file = par ? "kernels/sink_attention_wg_f32.cl" : "kernels/sink_attention_f32.cl";
        cl_program p = cl_ctx.build_program_from_file(file);
        if(!p){NNOPT_ERROR("CrossAttention: build sink");return nullptr;}
        sk_slot=clCreateKernel(p, par ? "sink_attention_wg_f32" : "sink_attention_f32",&err);
    }
    // One kernel object per call site — a recording references the kernel, not its args, so the 12
    // cross-attentions in a frame would otherwise all replay with the last one's buffers.
    cl_kernel sk = nnopt_kernel_instance(sk_slot, wp, ".xattn");
    if (!sk) { NNOPT_ERROR("CrossAttention: kernel create"); return nullptr; }

    // Both projections go through nnopt_gemv, so fp16 and Q4 bundles share this path.
    // Folded into q_proj ONLY: kv_proj reads the encoder states, which the block's pre-norm does
    // not touch.
    const std::string rms_key = (nnopt_fuse_level() >= 3 && prenorm_prefix)
                                ? std::string(prenorm_prefix) + ".weight" : std::string();
    cl_mem q  = rms_key.empty()
      ? nnopt_gemv(cl_ctx, weights, queue, input, wp + ".q_proj.weight",
                   1, in_dim, hd, nullptr)                                 // [H*D]
      : nnopt_gemv_fused(cl_ctx, weights, queue, input, wp + ".q_proj.weight",
                         1, in_dim, hd, nullptr, rms_key, false);
    cl_mem kv = nnopt_gemv(cl_ctx, weights, queue, encoder_hidden_states, wp + ".kv_proj.weight",
                           1, src_dim, kv_out, nullptr);                   // [2*H*D]
    if (!q || !kv) { NNOPT_ERROR("CrossAttention: proj failed"); if(q)pool_free(q); if(kv)pool_free(kv); return nullptr; }
    // pack [q | k | v] into one [3*H*D] buffer for the sink kernel
    cl_mem qkv = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)3*hd*sizeof(float), nullptr, &err);
    // Kernel copies, not clEnqueueCopyBuffer: a recording captures only NDRange dispatches, so a
    // copy here would not replay and every replayed frame would pack a stale q/kv.
    nnopt_copy_buffer(cl_ctx, q,  qkv, 0, 0,  hd,     wp, ".xattn.packq");
    nnopt_copy_buffer(cl_ctx, kv, qkv, 0, hd, kv_out, wp, ".xattn.packkv");
    pool_free(q); pool_free(kv);

    cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)hd*sizeof(float), nullptr, &err);
    clSetKernelArg(sk,0,sizeof(cl_mem),&qkv); clSetKernelArg(sk,1,sizeof(cl_mem),&sink_k);
    clSetKernelArg(sk,2,sizeof(cl_mem),&sink_v); clSetKernelArg(sk,3,sizeof(cl_mem),&pds);
    clSetKernelArg(sk,4,sizeof(cl_mem),&out); clSetKernelArg(sk,5,sizeof(int),&H);
    clSetKernelArg(sk,6,sizeof(int),&D); clSetKernelArg(sk,7,sizeof(float),&inv_sqrt_d);
    clSetKernelArg(sk,8,sizeof(float),&log_n);
    // Through profEnqueue so cross-attention finally appears in the profile at all.
    cl_int xerr;
    if (par) { const size_t sg=(size_t)H*64, sl=64; xerr = cl_ctx.profEnqueue(sk,1,&sg,&sl,"xattn"); }
    else     { const size_t sg=(size_t)H;           xerr = cl_ctx.profEnqueue(sk,1,&sg,nullptr,"xattn"); }
    if(xerr!=CL_SUCCESS){
        NNOPT_ERROR("CrossAttention: sink enqueue"); pool_free(qkv); pool_free(out); return nullptr; }
    
    pool_free(qkv);
    return out;
}
}
