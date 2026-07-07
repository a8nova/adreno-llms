// MoonshineAttention.cpp — shared implementation for ALL 18 MoonshineAttention nodes.
// Reference: model_info/transformers_src/modeling_moonshine.py MoonshineAttention.forward
//   (:252-373), apply_rotary_pos_emb.
//
// One op, three modes, routed by (layer_idx, weight_prefix) through the
// standard <Class>_forward signature — mirrors WhisperSdpaAttention.cpp:
//   layer_idx == -1                          → encoder self-attn (uncached,
//                                              RoPE on Q/K, non-causal)
//   layer_idx >= 0, prefix *.self_attn       → decoder self-attn over the
//                                              persistent per-layer KV ring
//                                              (OPT-1; start_pos = past)
//   layer_idx >= 0, prefix *.encoder_attn    → decoder cross-attn over the
//                                              per-waveform cached encoder K/V
//                                              (no RoPE, non-causal)
//
// This file also OWNS the decoder KV state (self rings + cross K/V + enc_T):
// backbone.cpp calls MoonshineAttention_build_decoder_caches() once per
// waveform and passes `start_pos` (the cache cursor) per forward.
//
// Kernels used: attention (kernels/attn.cl, the OPT-2 64-lane-WG kernel),
// rope_apply / split_heads / merge_heads (kernels/pack.cl).

#include "../opencl_context.h"
#include "../weights.h"
#include "../nnopt_error.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"
#include <cmath>
#include <string>

cl_kernel moonshine_kernel(OpenCLContext& cl_ctx, const char* name);
bool moonshine_rope_tables(OpenCLContext& cl_ctx, int max_pos, int head_dim,
                           float theta_base, cl_mem& cos_out, cl_mem& sin_out);
extern "C" cl_mem Linear_forward(OpenCLContext&, Weights&, cl_command_queue,
    cl_mem, int, int, int, cl_mem*, cl_mem*, cl_mem, const char*);

// ── config-derived attention constants ──────────────────────────────────
static constexpr int HID      = MODEL_CONFIG::HIDDEN_SIZE;                    // 288
static constexpr int N_HEADS  = MODEL_CONFIG::ENCODER_NUM_ATTENTION_HEADS;    // 8
static constexpr int HEAD_DIM = HID / N_HEADS;                                // 36
static constexpr int KV_CAP   = MODEL_CONFIG::MAX_SEQUENCE_LENGTH;            // 194

// ── decoder KV state (owned here; see file header) ──────────────────────
namespace {
    cl_mem g_self_k[MODEL_CONFIG::DECODER_NUM_HIDDEN_LAYERS] = {};
    cl_mem g_self_v[MODEL_CONFIG::DECODER_NUM_HIDDEN_LAYERS] = {};
    cl_mem g_cross_k[MODEL_CONFIG::DECODER_NUM_HIDDEN_LAYERS] = {};
    cl_mem g_cross_v[MODEL_CONFIG::DECODER_NUM_HIDDEN_LAYERS] = {};
    int    g_cross_T = 0;   // enc_T the cross caches were built for
}

// ── head packing ────────────────────────────────────────────────────────

// split [seq, H*D] → [H, seq, D]
cl_mem moonshine_split_heads(OpenCLContext& cl_ctx, cl_command_queue q,
                             cl_mem in, int seq) {
    cl_int err = CL_SUCCESS;
    cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                (size_t)seq * HID * sizeof(nnopt_storage_t), nullptr, &err);
    if (!out) { NNOPT_ERROR("split_heads alloc"); return nullptr; }
    cl_kernel k = moonshine_kernel(cl_ctx, "split_heads");
    if (!k) { NNOPT_ERROR("split_heads kernel"); clReleaseMemObject(out); return nullptr; }
    int H = N_HEADS, D = HEAD_DIM;
    set_arg_checked(k, 0, sizeof(cl_mem), &in, "in");
    set_arg_checked(k, 1, sizeof(cl_mem), &out, "out");
    set_arg_checked(k, 2, sizeof(int), &seq, "seq");
    set_arg_checked(k, 3, sizeof(int), &H, "H");
    set_arg_checked(k, 4, sizeof(int), &D, "D");
    size_t gws = (size_t)seq * HID;
    err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("split_heads"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("split_heads dispatch %d", err); clReleaseMemObject(out); return nullptr; }
    return out;
}

// merge [H, seq, D] → [seq, H*D]
static cl_mem merge_heads(OpenCLContext& cl_ctx, cl_command_queue q,
                          cl_mem in, int seq) {
    cl_int err = CL_SUCCESS;
    cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                (size_t)seq * HID * sizeof(nnopt_storage_t), nullptr, &err);
    if (!out) { NNOPT_ERROR("merge_heads alloc"); return nullptr; }
    cl_kernel k = moonshine_kernel(cl_ctx, "merge_heads");
    if (!k) { NNOPT_ERROR("merge_heads kernel"); clReleaseMemObject(out); return nullptr; }
    int H = N_HEADS, D = HEAD_DIM;
    set_arg_checked(k, 0, sizeof(cl_mem), &in, "in");
    set_arg_checked(k, 1, sizeof(cl_mem), &out, "out");
    set_arg_checked(k, 2, sizeof(int), &seq, "seq");
    set_arg_checked(k, 3, sizeof(int), &H, "H");
    set_arg_checked(k, 4, sizeof(int), &D, "D");
    size_t gws = (size_t)seq * HID;
    err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("merge_heads"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("merge_heads dispatch %d", err); clReleaseMemObject(out); return nullptr; }
    return out;
}

// ── RoPE (partial rotary, interleaved tables from moonshine_common) ─────
// x [H, seq, D] roped in place at absolute positions pos_offset..pos_offset+seq.
static bool rope_apply(OpenCLContext& cl_ctx, cl_command_queue q,
                       cl_mem x, int seq, int pos_offset) {
    // partial_rotary_factor=0.9 → rotary_dim = int(36*0.9)=32 (even).
    static const int ROTARY_DIM =
        ((int)((float)HEAD_DIM * MODEL_CONFIG::PARTIAL_ROTARY_FACTOR)) & ~1;
    cl_mem cos_t = nullptr, sin_t = nullptr;
    int max_pos = MODEL_CONFIG::MAX_SEQUENCE_LENGTH > 512 ? MODEL_CONFIG::MAX_SEQUENCE_LENGTH : 512;
    if (seq + pos_offset > max_pos) max_pos = seq + pos_offset;
    if (!moonshine_rope_tables(cl_ctx, max_pos, HEAD_DIM,
                               (float)MODEL_CONFIG::ROPE_THETA, cos_t, sin_t)) {
        NNOPT_ERROR("rope tables build failed"); return false;
    }
    cl_int err = CL_SUCCESS;
    cl_kernel k = moonshine_kernel(cl_ctx, "rope_apply");
    if (!k) { NNOPT_ERROR("rope_apply kernel"); return false; }
    int H = N_HEADS, D = HEAD_DIM, rd = ROTARY_DIM;
    set_arg_checked(k, 0, sizeof(cl_mem), &x, "x");
    set_arg_checked(k, 1, sizeof(cl_mem), &cos_t, "cos");
    set_arg_checked(k, 2, sizeof(cl_mem), &sin_t, "sin");
    set_arg_checked(k, 3, sizeof(int), &H, "H");
    set_arg_checked(k, 4, sizeof(int), &seq, "seq");
    set_arg_checked(k, 5, sizeof(int), &D, "D");
    set_arg_checked(k, 6, sizeof(int), &rd, "rotary_dim");
    set_arg_checked(k, 7, sizeof(int), &pos_offset, "pos_offset");
    size_t gws = (size_t)H * seq;
    err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("rope_apply"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("rope_apply dispatch %d", err); return false; }
    return true;
}

// ── attention core (OPT-2 kernel) ───────────────────────────────────────
// Q [H,Tq,D] × K/V [H,·,D] (kv_stride rows apart per head) → out [H,Tq,D].
static cl_mem attention_core(OpenCLContext& cl_ctx, cl_command_queue q,
                             cl_mem Q, cl_mem K, cl_mem V,
                             int Tq, int Tk, int causal, int q_pos_offset,
                             int kv_stride) {
    cl_int err = CL_SUCCESS;
    cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                (size_t)N_HEADS * Tq * HEAD_DIM * sizeof(nnopt_storage_t), nullptr, &err);
    if (!out) { NNOPT_ERROR("attention alloc"); return nullptr; }
    cl_kernel k = moonshine_kernel(cl_ctx, "attention");
    if (!k) { NNOPT_ERROR("attention kernel"); clReleaseMemObject(out); return nullptr; }
    int H = N_HEADS, D = HEAD_DIM;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    set_arg_checked(k, 0, sizeof(cl_mem), &Q, "Q");
    set_arg_checked(k, 1, sizeof(cl_mem), &K, "K");
    set_arg_checked(k, 2, sizeof(cl_mem), &V, "V");
    set_arg_checked(k, 3, sizeof(cl_mem), &out, "out");
    set_arg_checked(k, 4, sizeof(int), &H, "H");
    set_arg_checked(k, 5, sizeof(int), &Tq, "Tq");
    set_arg_checked(k, 6, sizeof(int), &Tk, "Tk");
    set_arg_checked(k, 7, sizeof(int), &D, "D");
    set_arg_checked(k, 8, sizeof(float), &scale, "scale");
    set_arg_checked(k, 9, sizeof(int), &causal, "causal");
    set_arg_checked(k, 10, sizeof(int), &q_pos_offset, "q_pos_offset");
    set_arg_checked(k, 11, sizeof(int), &kv_stride, "kv_stride");
    // OPT-2: one 64-lane workgroup per (h, i) query row (kernel stages scores
    // in local memory; ATTN_MAX_TK=1024 rows ≈ 25s of audio for cross-attn).
    if (Tk > 1024) { NNOPT_ERROR_FMT("attention Tk=%d exceeds ATTN_MAX_TK=1024 (~25s audio) — refusing (online-softmax kernel needed for longer clips)", Tk); clReleaseMemObject(out); return nullptr; }
    size_t lws = 64;
    size_t gws = (size_t)H * Tq * lws;
    err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, &lws, 0, nullptr,
                                 KernelProfiler::event_for("attention"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("attention dispatch %d", err); clReleaseMemObject(out); return nullptr; }
    return out;
}

// ── KV-cache append (OPT-1) ─────────────────────────────────────────────
// Append new [H, T_new, D] head-split rows into a persistent [H, KV_CAP, D]
// cache at position `pos`, per head, via one clEnqueueCopyBufferRect.
static bool kv_append(cl_command_queue q, cl_mem cache, cl_mem newkv,
                      int pos, int t_new, int kv_cap) {
    const size_t es = sizeof(nnopt_storage_t);
    const size_t row_bytes = (size_t)t_new * HEAD_DIM * es;   // contiguous [T_new, D] block per head
    size_t src_origin[3] = {0, 0, 0};
    size_t dst_origin[3] = {(size_t)pos * HEAD_DIM * es, 0, 0};
    size_t region[3]     = {row_bytes, (size_t)N_HEADS, 1};
    cl_int err = clEnqueueCopyBufferRect(
        q, newkv, cache, src_origin, dst_origin, region,
        /*src_row_pitch=*/row_bytes, /*src_slice_pitch=*/0,
        /*dst_row_pitch=*/(size_t)kv_cap * HEAD_DIM * es, /*dst_slice_pitch=*/0,
        0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("kv_append rect copy %d", err); return false; }
    return true;
}

// ── mode bodies ─────────────────────────────────────────────────────────

// Encoder self-attention (uncached): RoPE on Q/K, non-causal, K/V from input.
static cl_mem attention_uncached(OpenCLContext& cl_ctx, Weights& w, cl_command_queue q,
                                 cl_mem input, int seq, const std::string& wp) {
    cl_mem qp=nullptr,kp=nullptr,vp=nullptr,qh=nullptr,kh=nullptr,vh=nullptr,ao=nullptr,merged=nullptr,outp=nullptr;
    auto cleanup=[&]()->cl_mem{
        for (cl_mem* p:{&qp,&kp,&vp,&qh,&kh,&vh,&ao,&merged}) if(*p){clReleaseMemObject(*p);*p=nullptr;}
        return nullptr;
    };
    qp = Linear_forward(cl_ctx,w,q,input,seq,-1,0,nullptr,nullptr,nullptr,(wp+".q_proj").c_str());
    kp = Linear_forward(cl_ctx,w,q,input,seq,-1,0,nullptr,nullptr,nullptr,(wp+".k_proj").c_str());
    vp = Linear_forward(cl_ctx,w,q,input,seq,-1,0,nullptr,nullptr,nullptr,(wp+".v_proj").c_str());
    if(!qp||!kp||!vp){NNOPT_ERROR_FMT("attn proj failed %s",wp.c_str());return cleanup();}
    qh = moonshine_split_heads(cl_ctx,q,qp,seq);
    kh = moonshine_split_heads(cl_ctx,q,kp,seq);
    vh = moonshine_split_heads(cl_ctx,q,vp,seq);
    if(!qh||!kh||!vh) return cleanup();
    if(!rope_apply(cl_ctx,q,qh,seq,0)) return cleanup();
    if(!rope_apply(cl_ctx,q,kh,seq,0)) return cleanup();
    ao = attention_core(cl_ctx,q,qh,kh,vh,seq,seq,/*causal=*/0,0,/*kv_stride=*/seq);
    if(!ao) return cleanup();
    merged = merge_heads(cl_ctx,q,ao,seq);
    if(!merged) return cleanup();
    outp = Linear_forward(cl_ctx,w,q,merged,seq,-1,0,nullptr,nullptr,nullptr,(wp+".o_proj").c_str());
    cleanup();
    if(!outp){NNOPT_ERROR_FMT("o_proj failed %s",wp.c_str());return nullptr;}
    return outp;
}

// Decoder SELF-attention with persistent KV cache. Processes T_new query rows
// whose absolute positions start at `past` (prefill = past 0). Appends the new
// K/V into the layer cache, attends over all past+T_new rows.
static cl_mem self_attention_cached(OpenCLContext& cl_ctx, Weights& w, cl_command_queue q,
                                    cl_mem input, int t_new, int past,
                                    cl_mem k_cache, cl_mem v_cache,
                                    const std::string& wp) {
    cl_mem qp=nullptr,kp=nullptr,vp=nullptr,qh=nullptr,kh=nullptr,vh=nullptr,ao=nullptr,merged=nullptr,outp=nullptr;
    auto cleanup=[&]()->cl_mem{
        for (cl_mem* p:{&qp,&kp,&vp,&qh,&kh,&vh,&ao,&merged}) if(*p){clReleaseMemObject(*p);*p=nullptr;}
        return nullptr;
    };
    qp = Linear_forward(cl_ctx,w,q,input,t_new,-1,0,nullptr,nullptr,nullptr,(wp+".q_proj").c_str());
    kp = Linear_forward(cl_ctx,w,q,input,t_new,-1,0,nullptr,nullptr,nullptr,(wp+".k_proj").c_str());
    vp = Linear_forward(cl_ctx,w,q,input,t_new,-1,0,nullptr,nullptr,nullptr,(wp+".v_proj").c_str());
    if(!qp||!kp||!vp){NNOPT_ERROR_FMT("self-attn proj failed %s",wp.c_str());return cleanup();}
    qh = moonshine_split_heads(cl_ctx,q,qp,t_new);
    kh = moonshine_split_heads(cl_ctx,q,kp,t_new);
    vh = moonshine_split_heads(cl_ctx,q,vp,t_new);
    if(!qh||!kh||!vh) return cleanup();
    // RoPE at ABSOLUTE positions (past..past+t_new) — identical math to the
    // uncached path, which roped the full sequence from position 0.
    if(!rope_apply(cl_ctx,q,qh,t_new,past)) return cleanup();
    if(!rope_apply(cl_ctx,q,kh,t_new,past)) return cleanup();
    if(!kv_append(q,k_cache,kh,past,t_new,KV_CAP)) return cleanup();
    if(!kv_append(q,v_cache,vh,past,t_new,KV_CAP)) return cleanup();
    ao = attention_core(cl_ctx,q,qh,k_cache,v_cache,t_new,past+t_new,/*causal=*/1,
                        /*q_pos_offset=*/past,/*kv_stride=*/KV_CAP);
    if(!ao) return cleanup();
    merged = merge_heads(cl_ctx,q,ao,t_new);
    if(!merged) return cleanup();
    outp = Linear_forward(cl_ctx,w,q,merged,t_new,-1,0,nullptr,nullptr,nullptr,(wp+".o_proj").c_str());
    cleanup();
    if(!outp){NNOPT_ERROR_FMT("self o_proj failed %s",wp.c_str());return nullptr;}
    return outp;
}

// Decoder CROSS-attention with the per-waveform cached encoder K/V.
static cl_mem cross_attention_cached(OpenCLContext& cl_ctx, Weights& w, cl_command_queue q,
                                     cl_mem input, int t_new,
                                     cl_mem k_cache, cl_mem v_cache, int enc_T,
                                     const std::string& wp) {
    cl_mem qp=nullptr,qh=nullptr,ao=nullptr,merged=nullptr,outp=nullptr;
    auto cleanup=[&]()->cl_mem{
        for (cl_mem* p:{&qp,&qh,&ao,&merged}) if(*p){clReleaseMemObject(*p);*p=nullptr;}
        return nullptr;
    };
    qp = Linear_forward(cl_ctx,w,q,input,t_new,-1,0,nullptr,nullptr,nullptr,(wp+".q_proj").c_str());
    if(!qp){NNOPT_ERROR_FMT("cross q_proj failed %s",wp.c_str());return cleanup();}
    qh = moonshine_split_heads(cl_ctx,q,qp,t_new);
    if(!qh) return cleanup();
    // No RoPE on cross-attention (Moonshine), K/V precomputed per waveform.
    ao = attention_core(cl_ctx,q,qh,k_cache,v_cache,t_new,enc_T,/*causal=*/0,
                        /*q_pos_offset=*/0,/*kv_stride=*/enc_T);
    if(!ao) return cleanup();
    merged = merge_heads(cl_ctx,q,ao,t_new);
    if(!merged) return cleanup();
    outp = Linear_forward(cl_ctx,w,q,merged,t_new,-1,0,nullptr,nullptr,nullptr,(wp+".o_proj").c_str());
    cleanup();
    if(!outp){NNOPT_ERROR_FMT("cross o_proj failed %s",wp.c_str());return nullptr;}
    return outp;
}

// ── cache lifecycle (called by backbone.cpp once per waveform) ──────────
// Allocates the self K/V rings and computes the cross K/V from the encoder
// output. Returns false on failure.
extern "C" bool MoonshineAttention_build_decoder_caches(
    OpenCLContext& cl_ctx, Weights& w, cl_command_queue q, cl_mem enc, int enc_T) {
    const size_t es = sizeof(nnopt_storage_t);
    cl_int err = CL_SUCCESS;
    for (int i = 0; i < MODEL_CONFIG::DECODER_NUM_HIDDEN_LAYERS; ++i) {
        for (cl_mem* p : {&g_self_k[i], &g_self_v[i], &g_cross_k[i], &g_cross_v[i]})
            if (*p) { clReleaseMemObject(*p); *p = nullptr; }
        g_self_k[i] = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                     (size_t)N_HEADS * KV_CAP * HEAD_DIM * es, nullptr, &err);
        g_self_v[i] = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                     (size_t)N_HEADS * KV_CAP * HEAD_DIM * es, nullptr, &err);
        if (!g_self_k[i] || !g_self_v[i]) { NNOPT_ERROR("self KV alloc"); return false; }
        std::string base = "model.decoder.layers." + std::to_string(i) + ".encoder_attn";
        cl_mem kp = Linear_forward(cl_ctx,w,q,enc,enc_T,-1,0,nullptr,nullptr,nullptr,(base+".k_proj").c_str());
        cl_mem vp = Linear_forward(cl_ctx,w,q,enc,enc_T,-1,0,nullptr,nullptr,nullptr,(base+".v_proj").c_str());
        if(!kp||!vp){NNOPT_ERROR("cross KV proj");if(kp)clReleaseMemObject(kp);if(vp)clReleaseMemObject(vp);return false;}
        g_cross_k[i] = moonshine_split_heads(cl_ctx,q,kp,enc_T);
        g_cross_v[i] = moonshine_split_heads(cl_ctx,q,vp,enc_T);
        clReleaseMemObject(kp); clReleaseMemObject(vp);
        if (!g_cross_k[i] || !g_cross_v[i]) { NNOPT_ERROR("cross KV split"); return false; }
    }
    g_cross_T = enc_T;
    return true;
}

// ── standard op entry point ─────────────────────────────────────────────
// Routing (see file header): layer_idx == -1 → encoder uncached;
// layer_idx >= 0 → decoder, ".self_attn" vs ".encoder_attn" by prefix.
// seq_len = T (encoder) or t_new (decoder); start_pos = past (decoder self).
extern "C" cl_mem MoonshineAttention_forward(
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
    (void)k_cache_inout; (void)v_cache_inout; (void)encoder_hidden_states;
    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();
    if (wp.empty()) { NNOPT_ERROR("MoonshineAttention: empty weight_prefix"); return nullptr; }
    if (layer_idx < 0)
        return attention_uncached(cl_ctx, weights, queue, input, seq_len, wp);
    if (layer_idx >= MODEL_CONFIG::DECODER_NUM_HIDDEN_LAYERS) {
        NNOPT_ERROR_FMT("MoonshineAttention: layer_idx %d out of range", layer_idx);
        return nullptr;
    }
    if (wp.find(".encoder_attn") != std::string::npos)
        return cross_attention_cached(cl_ctx, weights, queue, input, seq_len,
                                      g_cross_k[layer_idx], g_cross_v[layer_idx],
                                      g_cross_T, wp);
    return self_attention_cached(cl_ctx, weights, queue, input, seq_len, start_pos,
                                 g_self_k[layer_idx], g_self_v[layer_idx], wp);
}
