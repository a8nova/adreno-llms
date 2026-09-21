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
// ── two implementations, and when each is right ────────────────────────────────────────────────
// ln(N) is exact only while the conditioning is CONSTANT, which is the single-prompt case. Drag a
// blend and the ring's 42 vectors stop being identical: upstream then cross-fades the change over
// 42 frames (1.68 s) because the ring still holds the old conditioning, whereas one source
// switches instantly. That is the staircase heard on 2-3 prompt blends.
//
// So when the caller hands down a ring (k_cache_inout/v_cache_inout — previously unused here, like
// start_pos), this keeps the real thing: PROJECTED k/v per slot, written round-robin at pos % S.
// Projection stays one GEMV per layer per frame exactly as before — only the attention widens,
// from 2 keys to n_valid+1. Callers with no ring (the op-test path) keep the ln(N) form.
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
    (void)layer_idx;
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
    // log_n is no longer computed here: it is derived on device from the xpos buffer inside the
    // sink kernels. N = min(xpos+1, max_src) grows every frame until it saturates, so a host
    // float here was a per-frame value baked into a kernel argument, and a recording pins
    // argument values at capture — every replayed frame reused the captured frame's ln(2).
    // n_src stays: the RING path still passes it, and that path already vetoes recording.
    // OFF BY DEFAULT (NNOPT_XATTNRING=1 to arm). The ring is the reference-exact form and its math
    // is verified against the ln(N) path offline, but it has never run on an Adreno: it ships a new
    // kernel, and the one build that went out with it on by default produced a degenerate render.
    // Until that is reproduced and understood, the default path stays the one that has been heard.
    static int ring_on = -1, ring_epoch = -1;
    if (ring_epoch != nnopt_toggle_epoch()) {
        const char* e = std::getenv("NNOPT_XATTNRING");
        ring_on = (e && e[0] && e[0] != '0') ? 1 : 0;
        ring_epoch = nnopt_toggle_epoch();
    }
    const bool use_ring = ring_on && k_cache_inout && v_cache_inout && max_src > 1 && start_pos >= 0;

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

    if (use_ring) {
        // Ring slots hold PROJECTED k/v, not raw sources: kv_proj is per-layer, so projecting on
        // read would cost 42 GEMVs per layer per frame instead of one. This is the same trade the
        // self-attention KV cache already makes.
        if (!*k_cache_inout) {
            *k_cache_inout = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE,
                                        (size_t)max_src*hd*sizeof(float), nullptr, &err);
            *v_cache_inout = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE,
                                        (size_t)max_src*hd*sizeof(float), nullptr, &err);
            if (!*k_cache_inout || !*v_cache_inout) {
                NNOPT_ERROR("CrossAttention: ring alloc"); pool_free(q); pool_free(kv); return nullptr; }
        }
        // The slot index is a host value baked into a dispatch argument and it advances every
        // frame, so a captured frame would replay writing slot N forever. Veto recording rather
        // than silently freeze the ring.
        nnopt_record_mark_unsafe("cross-attention ring slot advances per frame");
        const int slot = start_pos % max_src;
        nnopt_copy_buffer(cl_ctx, kv, *k_cache_inout, 0,  slot*hd, hd, wp, ".xattn.ringk");
        nnopt_copy_buffer(cl_ctx, kv, *v_cache_inout, hd, slot*hd, hd, wp, ".xattn.ringv");
        pool_free(kv);

        static cl_kernel rk_slot = nullptr;
        if (!rk_slot) {
            cl_program p = cl_ctx.build_program_from_file("kernels/sink_attention_ring_f32.cl");
            if (!p) { NNOPT_ERROR("CrossAttention: build ring"); pool_free(q); return nullptr; }
            rk_slot = clCreateKernel(p, "sink_attention_ring_f32", &err);
        }
        cl_kernel rk = nnopt_kernel_instance(rk_slot, wp, ".xattnring");
        if (!rk) { NNOPT_ERROR("CrossAttention: ring kernel"); pool_free(q); return nullptr; }
        cl_mem rout = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)hd*sizeof(float), nullptr, &err);
        if (!rout) { NNOPT_ERROR("CrossAttention: ring out alloc"); pool_free(q); return nullptr; }
        clSetKernelArg(rk,0,sizeof(cl_mem),&q);
        clSetKernelArg(rk,1,sizeof(cl_mem),k_cache_inout);
        clSetKernelArg(rk,2,sizeof(cl_mem),v_cache_inout);
        clSetKernelArg(rk,3,sizeof(cl_mem),&sink_k);
        clSetKernelArg(rk,4,sizeof(cl_mem),&sink_v);
        clSetKernelArg(rk,5,sizeof(cl_mem),&pds);
        clSetKernelArg(rk,6,sizeof(cl_mem),&rout);
        clSetKernelArg(rk,7,sizeof(int),&H);
        clSetKernelArg(rk,8,sizeof(int),&D);
        clSetKernelArg(rk,9,sizeof(float),&inv_sqrt_d);
        clSetKernelArg(rk,10,sizeof(int),&n_src);
        clSetKernelArg(rk,11,sizeof(int),&max_src);
        const size_t rg=(size_t)H*64, rl=64;
        const cl_int rerr = cl_ctx.profEnqueue(rk,1,&rg,&rl,"xattn_ring");
        pool_free(q);
        if (rerr != CL_SUCCESS) { NNOPT_ERROR("CrossAttention: ring enqueue"); pool_free(rout); return nullptr; }
        return rout;
    }
    // NO PACKING. The sink kernel now takes q and kv as the two buffers qkv_proj already produced.
    // Packing them into one [3*H*D] block cost two copy dispatches per call (24/frame) to move
    // 0.05 ms of data -- ~500:1 launch cost against work -- and the unpacked form is the same values
    // at a different offset, so this removes the launches and adds nothing. (The copies existed as
    // KERNELS rather than clEnqueueCopyBuffer because a recording captures only NDRange dispatches;
    // removing them entirely keeps the frame recordable for the same reason, with one less thing
    // that can go stale on replay.)
    cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)hd*sizeof(float), nullptr, &err);
    clSetKernelArg(sk,0,sizeof(cl_mem),&q);  clSetKernelArg(sk,1,sizeof(cl_mem),&kv);
    clSetKernelArg(sk,2,sizeof(cl_mem),&sink_k);
    clSetKernelArg(sk,3,sizeof(cl_mem),&sink_v); clSetKernelArg(sk,4,sizeof(cl_mem),&pds);
    clSetKernelArg(sk,5,sizeof(cl_mem),&out); clSetKernelArg(sk,6,sizeof(int),&H);
    clSetKernelArg(sk,7,sizeof(int),&D); clSetKernelArg(sk,8,sizeof(float),&inv_sqrt_d);
    // Slot 1 is the cross-attention's own position (see nnopt_pos_buffer): xpos, not pos.
    cl_mem xposb = nnopt_pos_buffer(cl_ctx, queue, start_pos, /*constant=*/false, /*slot=*/1);
    if (!xposb) { NNOPT_ERROR("CrossAttention: xpos buffer"); pool_free(q); pool_free(kv);
                  pool_free(out); return nullptr; }
    clSetKernelArg(sk,9,sizeof(cl_mem),&xposb);
    clSetKernelArg(sk,10,sizeof(int),&max_src);
    // Through profEnqueue so cross-attention finally appears in the profile at all.
    cl_int xerr;
    if (par) { const size_t sg=(size_t)H*64, sl=64; xerr = cl_ctx.profEnqueue(sk,1,&sg,&sl,"xattn"); }
    else     { const size_t sg=(size_t)H;           xerr = cl_ctx.profEnqueue(sk,1,&sg,nullptr,"xattn"); }
    if(xerr!=CL_SUCCESS){
        NNOPT_ERROR("CrossAttention: sink enqueue");
        pool_free(q); pool_free(kv); pool_free(out); return nullptr; }

    // q/kv are freed HERE, not before the enqueue: the kernel reads them directly now, so they must
    // outlive the dispatch. (They used to be freed right after being copied into the packed buffer.)
    pool_free(q); pool_free(kv);
    return out;
}
}
