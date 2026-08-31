// CachedSelfAttention.cpp — self-attention with a persistent K/V cache (AR generation).
// Reference: sequence_layers/mlx attention .step (streaming). Subsumes SinkAttention:
//   - temporal self-attn: has_sink=true, cache over frames (pos = frame index, ≤ max_past_horizon)
//   - depth self-attn:     has_sink=false, cache over RVQ codebooks (pos = codebook index)
// Caller owns kcache/vcache buffers [maxpos*H*D] and passes the current position via start_pos.
//   qkv = input @ qkv_proj.weight.T ; current k,v written to cache[pos]; attend [sink?, 0..pos].

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>

extern "C" cl_mem CachedSelfAttention_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout, cl_mem encoder_hidden_states,
    const char* weight_prefix, const char* prenorm_prefix)
{
    (void)layer_idx; (void)seq_len; (void)encoder_hidden_states;
    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();
    if (wp.empty() || !input || !k_cache_inout || !v_cache_inout) { NNOPT_ERROR("CachedSelfAttn: null arg"); return nullptr; }
    const int pos = start_pos;
    cl_mem pds   = weights.get_buffer(wp + "._per_dim_scale");
    if (!pds) { NNOPT_ERROR_FMT("CachedSelfAttn: missing %s._per_dim_scale", wp.c_str()); return nullptr; }
    const bool has_sink = weights.has_tensor(wp + ".sink_key_embeddings");
    cl_mem sink_k = has_sink ? weights.get_buffer(wp + ".sink_key_embeddings") : pds; // dummy bind when unused
    cl_mem sink_v = has_sink ? weights.get_buffer(wp + ".sink_value_embeddings") : pds;
    int qkv_out = 0, in_dim = 0;
    if (!nnopt_weight_dims(weights, wp + ".qkv_proj.weight", &qkv_out, &in_dim)) return nullptr;
    const std::vector<int> psh = weights.get_shape(wp + "._per_dim_scale");
    const int D = psh.back(), H = (qkv_out/3)/D, HD = H*D;
    const float inv_sqrt_d = 1.0f/std::sqrt((float)D);

    cl_int err = CL_SUCCESS;
    // One workgroup per head instead of one work ITEM per head. The serial kernel dispatched H (=12)
    // fibers for the entire GPU — Adreno cannot split a workgroup across compute units (§3.2.5), so
    // twelve items is twelve lanes of one CU — and it recomputed the per-dim softplus scale once per
    // (position, dim) pair instead of once per dim. NNOPT_ATTN=serial restores it for A/B.
    constexpr int kWgAttnMaxD = 256, kWgAttnMaxPos = 160;
    static int par = -1, par_epoch = -1;
    if (par_epoch != nnopt_toggle_epoch()) {
        const char* a = std::getenv("NNOPT_ATTN");
        par = (a && std::strcmp(a, "serial") == 0) ? 0 : 1;
        par_epoch = nnopt_toggle_epoch();
    }
    static cl_kernel ak_ser=nullptr, ak_par=nullptr;
    cl_kernel& ak_slot = par ? ak_par : ak_ser;
    if (!ak_slot) {
        const char* file = par ? "kernels/self_attn_cached_wg_f32.cl" : "kernels/self_attn_cached_f32.cl";
        cl_program p=cl_ctx.build_program_from_file(file);
        if(!p){NNOPT_ERROR("CachedSelfAttn: build cached");return nullptr;}
        ak_slot=clCreateKernel(p, par ? "self_attn_cached_wg_f32" : "self_attn_cached_f32",&err);
    }
    // One kernel object per call site: a recordable-queue capture references the kernel and NOT its
    // arguments, so a kernel shared across the 12 temporal layers (and across the 12 RVQ levels of
    // the depth body) would replay every dispatch with the last one's buffers.
    cl_kernel ak = nnopt_kernel_instance(ak_slot, wp, ".attn");
    if (!ak){ NNOPT_ERROR("CachedSelfAttn: kernel create"); return nullptr; }

    // The block's pre-norm folds into THIS GEMV rather than costing its own dispatch. Exact: the
    // normalisation constant is uniform over the row, so it factors straight out of the dot
    // product (see kernels/linear_bias_fused_f32.cl). 36 dispatches a frame at ~56 us of host cost
    // each, on a path measured 94% host-bound.
    const std::string rms_key = (nnopt_fuse_level() >= 3 && prenorm_prefix)
                                ? std::string(prenorm_prefix) + ".weight" : std::string();
    cl_mem qkv = rms_key.empty()
        ? nnopt_gemv(cl_ctx, weights, queue, input, wp + ".qkv_proj.weight",
                            1, in_dim, qkv_out, nullptr)
        : nnopt_gemv_fused(cl_ctx, weights, queue, input, wp + ".qkv_proj.weight",
                           1, in_dim, qkv_out, nullptr, rms_key, false);
    if (!qkv) { NNOPT_ERROR("CachedSelfAttn: qkv"); return nullptr; }

    // Position lives in a device buffer. Two reasons, both required for record/replay: a recording
    // can only capture NDRange dispatches (so the k/v append must be a kernel, not a copy), and it
    // pins argument VALUES at capture time (so a scalar pos would freeze at frame 0).
    // has_sink distinguishes the two bodies: the temporal step has sink embeddings and its position
    // is the frame index (varies per replay); the depth body has none and its position is the RVQ
    // level (identical every frame, so it takes an immutable buffer the recording can pin).
    cl_mem posb = nnopt_pos_buffer(cl_ctx, queue, pos, /*constant=*/!has_sink);
    if (!posb) { NNOPT_ERROR("CachedSelfAttn: pos buffer"); pool_free(qkv); return nullptr; }
    {
        static cl_kernel kv = nullptr;
        if (!kv) {
            cl_program p = cl_ctx.build_program_from_file("kernels/kv_append_f32.cl");
            if (!p) { NNOPT_ERROR("CachedSelfAttn: build kv_append"); pool_free(qkv); return nullptr; }
            kv = clCreateKernel(p, "kv_append_f32", &err);
            if (!kv) { NNOPT_ERROR("CachedSelfAttn: kv_append kernel"); pool_free(qkv); return nullptr; }
        }
        cl_kernel kvi = nnopt_kernel_instance(kv, wp, ".kv_append");
        clSetKernelArg(kvi,0,sizeof(cl_mem),&qkv);
        clSetKernelArg(kvi,1,sizeof(cl_mem),k_cache_inout);
        clSetKernelArg(kvi,2,sizeof(cl_mem),v_cache_inout);
        clSetKernelArg(kvi,3,sizeof(cl_mem),&posb);
        clSetKernelArg(kvi,4,sizeof(int),&HD);
        const size_t kg = (size_t)HD;
        if (cl_ctx.profEnqueue(kvi,1,&kg,nullptr,"kv_append") != CL_SUCCESS) {
            NNOPT_ERROR("CachedSelfAttn: kv_append enqueue"); pool_free(qkv); return nullptr;
        }
    }

    cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)HD*sizeof(float), nullptr, &err);
    const int hs = has_sink?1:0;
    // temporal self-attn (has sink) windows to max_past_horizon=41; depth self-attn (no sink) is unbounded.
    // -1 tells the kernel "unbounded, use pos+1" — computing it here would bake a frame-0 value
    // into a recording. The host copy below is only for the span/limit check.
    const int max_past = has_sink ? 41 : -1;
    const int max_past_host = has_sink ? 41 : (pos+1);
    clSetKernelArg(ak,0,sizeof(cl_mem),&qkv); clSetKernelArg(ak,1,sizeof(cl_mem),k_cache_inout); clSetKernelArg(ak,2,sizeof(cl_mem),v_cache_inout);
    clSetKernelArg(ak,3,sizeof(cl_mem),&sink_k); clSetKernelArg(ak,4,sizeof(cl_mem),&sink_v); clSetKernelArg(ak,5,sizeof(cl_mem),&pds);
    clSetKernelArg(ak,6,sizeof(cl_mem),&out); clSetKernelArg(ak,7,sizeof(int),&H); clSetKernelArg(ak,8,sizeof(int),&D);
    clSetKernelArg(ak,9,sizeof(cl_mem),&posb); clSetKernelArg(ak,10,sizeof(int),&hs); clSetKernelArg(ak,11,sizeof(float),&inv_sqrt_d);
    clSetKernelArg(ak,12,sizeof(int),&max_past);
    // The parallel kernel sizes two local arrays from D and the attended span; outside those bounds
    // it would read past them, so fall back rather than produce quiet garbage.
    const int span = (has_sink ? 1 : 0) + (pos - std::max(0, pos - max_past_host) + 1);
    const bool wg_ok = par && D <= kWgAttnMaxD && span <= kWgAttnMaxPos;
    if (wg_ok) {
        const size_t ag=(size_t)H*64, al=(size_t)64;   // one 64-wide workgroup per head
        if(cl_ctx.profEnqueue(ak,1,&ag,&al,"attn")!=CL_SUCCESS){NNOPT_ERROR("CachedSelfAttn: attn");pool_free(qkv);pool_free(out);return nullptr;}
    } else {
        if (par) {
            // Only worth one line, once: a silent fallback is how a "verified" optimisation turns
            // out never to have run.
            static bool warned = false;
            if (!warned) { warned = true;
                std::fprintf(stderr, "ATTN_NOTE wg fallback D=%d span=%d (limits %d/%d)\n",
                             D, span, kWgAttnMaxD, kWgAttnMaxPos); }
            if (!ak_ser) {
                cl_program p=cl_ctx.build_program_from_file("kernels/self_attn_cached_f32.cl");
                if(!p){NNOPT_ERROR("CachedSelfAttn: build serial fallback");pool_free(qkv);pool_free(out);return nullptr;}
                ak_ser=clCreateKernel(p,"self_attn_cached_f32",&err);
            }
            // This path passes `pos` and `max_past_host` as HOST SCALARS, which a capture would
            // freeze at the recorded frame — every replay would then attend the wrong span. It is
            // an exception path (span<=160 and D<=256 normally hold), but "normally" is not a
            // guarantee, so taking it vetoes recording instead of silently corrupting a replay.
            nnopt_record_mark_unsafe("CachedSelfAttn serial fallback bakes host pos into args");
            ak = nnopt_kernel_instance(ak_ser, wp, ".attn_ser");
            clSetKernelArg(ak,0,sizeof(cl_mem),&qkv); clSetKernelArg(ak,1,sizeof(cl_mem),k_cache_inout); clSetKernelArg(ak,2,sizeof(cl_mem),v_cache_inout);
            clSetKernelArg(ak,3,sizeof(cl_mem),&sink_k); clSetKernelArg(ak,4,sizeof(cl_mem),&sink_v); clSetKernelArg(ak,5,sizeof(cl_mem),&pds);
            clSetKernelArg(ak,6,sizeof(cl_mem),&out); clSetKernelArg(ak,7,sizeof(int),&H); clSetKernelArg(ak,8,sizeof(int),&D);
            clSetKernelArg(ak,9,sizeof(int),&pos); clSetKernelArg(ak,10,sizeof(int),&hs); clSetKernelArg(ak,11,sizeof(float),&inv_sqrt_d);
            clSetKernelArg(ak,12,sizeof(int),&max_past_host);
        }
        size_t ag=(size_t)H;
        if(cl_ctx.profEnqueue(ak,1,&ag,nullptr,"attn")!=CL_SUCCESS){NNOPT_ERROR("CachedSelfAttn: attn");pool_free(qkv);pool_free(out);return nullptr;}
    }
    
    pool_free(qkv);
    return out;
}
