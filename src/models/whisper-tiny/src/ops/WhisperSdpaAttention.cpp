// WhisperSdpaAttention.cpp — scaled dot-product attention for Whisper (self + cross).
// Reference: model_info/transformers_src/modeling_whisper.py:96-175 eager_attention_forward
//            model_info/transformers_src/modeling_whisper.py:241-359 WhisperAttention.forward
//
// Implements (eager, correctness-first):
//   Q = q_proj(hidden_states)
//   K,V = proj(current_states) where current_states = hidden_states (self) or encoder_hidden_states (cross)
//   query_states = (Q * scaling).view([B,T,-1,head_dim]).transpose(1,2)
//   key_states/value_states = K/V.view([B,-1,num_heads,head_dim]).transpose(1,2)
//   scores = query_states @ key_states^T
//   + causal mask for self-attn when is_causal
//   probs = softmax(scores)
//   ctx = probs @ value_states
//   ctx = ctx.transpose(1,2).reshape([B,T,hidden])
//   out = out_proj(ctx)
//
// Notes:
// - No cache yet; computed fresh each call.
// - Uses simple packing kernels to convert [T,hidden] row-major into [H,T,D] for the OpenCL attention kernels.

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"

#include <CL/cl.h>
#include <clblast.h>
#include <cmath>
#include <map>
#include <string>
#include <vector>

// Scratch-buffer pool (opt #9). The encoder calls attention 4× with identical
// 1500-seq shapes, and decode calls it many times with small shapes — yet every
// call used to clCreateBuffer/clReleaseMemObject its large internal buffers (the
// score buffer alone is [6,1500,1500]×2B ≈ 27 MB), and that per-op alloc/free of
// big buffers is the dominant ENCODER host overhead (TTFT ~6s with only ~1.5s GPU).
// These buffers are pure scratch (consumed within the call, never returned), and
// the in-order queue fully drains one call before the next reuses the slot, so a
// persistent grow-only buffer reused across all calls is safe. Returns nullptr on
// allocation failure. Buffers are intentionally never freed (process lifetime).
static cl_mem nnopt_attn_scratch(OpenCLContext& cl_ctx, cl_mem& slot, size_t& cap, size_t need_bytes) {
    if (slot && cap >= need_bytes) return slot;
    if (slot) clReleaseMemObject(slot);
    cl_int e = CL_SUCCESS;
    slot = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, need_bytes, nullptr, &e);
    if (e != CL_SUCCESS || !slot) { slot = nullptr; cap = 0; return nullptr; }
    cap = need_bytes;
    return slot;
}

// Per-clip KV caches (opt #7 cross-attn, opt #12 self-attn), at file scope so they
// can be reset between clips when the SAME process transcribes many clips (batch
// mode). Both are keyed by weight_prefix, so a new clip's (different) audio would
// otherwise read the previous clip's cross-attn K/V. WhisperSdpaAttention_reset_caches()
// is called from the backbone whenever the encoder re-runs (i.e. a new clip).
static std::map<std::string, std::pair<cl_mem, cl_mem>> s_cross_kv_cache;
static std::map<std::string, std::pair<cl_mem, cl_mem>> s_self_kv_cache;

extern "C" void WhisperSdpaAttention_reset_caches() {
    for (auto& kv : s_cross_kv_cache) {
        if (kv.second.first)  clReleaseMemObject(kv.second.first);
        if (kv.second.second) clReleaseMemObject(kv.second.second);
    }
    s_cross_kv_cache.clear();
    for (auto& kv : s_self_kv_cache) {
        if (kv.second.first)  clReleaseMemObject(kv.second.first);
        if (kv.second.second) clReleaseMemObject(kv.second.second);
    }
    s_self_kv_cache.clear();
}

extern "C" {
cl_mem WhisperSdpaAttention_forward(
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

    if (!input) {
        NNOPT_ERROR("WhisperSdpaAttention_forward: input is null");
        return nullptr;
    }
    if (!weight_prefix) {
        NNOPT_ERROR("WhisperSdpaAttention_forward: weight_prefix is null");
        return nullptr;
    }

    const bool is_cross = (encoder_hidden_states != nullptr);
    const bool is_self = !is_cross;

    // Whisper has NO causal masking in the encoder (bidirectional self-attn).
    // Only decoder self-attn is causal.
    // Reference: model_info/transformers_src/modeling_whisper.py WhisperDecoder.forward creates causal_mask and passes
    // it only to decoder layers; WhisperEncoder.forward passes attention_mask=None.
    const bool use_causal_mask = (std::string(weight_prefix).find("model.decoder.layers.") != std::string::npos) && is_self;

    // Heads are determined by WHICH MODULE we are in (encoder self-attn vs decoder self/cross),
    // NOT by whether this is cross-attention.
    // In Whisper, both decoder self-attn and decoder cross-attn use decoder_attention_heads.
    const bool is_decoder_module = (std::string(weight_prefix).find("decoder") != std::string::npos);
    const int H = is_decoder_module ? (int)MODEL_CONFIG::DECODER_ATTENTION_HEADS
                                    : (int)MODEL_CONFIG::ENCODER_ATTENTION_HEADS;
    const int hidden = (int)MODEL_CONFIG::HIDDEN_SIZE;         // 384
    const int D = hidden / H;                                  // 64
    const int Tq = seq_len;
    // For self-attention, K/V come from the same input sequence.
    // For cross-attention, K/V come from the ENCODER sequence length (typically 1500 for Whisper).
    const int Tk = is_cross ? (int)MODEL_CONFIG::MAX_SOURCE_POSITIONS : Tq;

    // HF WhisperAttention.forward pre-scales Q by head_dim**-0.5 then calls attention with scaling=1.0.
    // Do the same (Q *= 1/sqrt(D)); do NOT also scale scores.
    const float q_scaling = 1.0f / std::sqrt((float)D);

    const std::string wp(weight_prefix);

    // Load weights.
    cl_mem q_w = weights.get_buffer(wp + ".q_proj.weight");
    cl_mem q_b = weights.get_buffer(wp + ".q_proj.bias");
    cl_mem k_w = weights.get_buffer(wp + ".k_proj.weight");
    cl_mem v_w = weights.get_buffer(wp + ".v_proj.weight");
    cl_mem v_b = weights.get_buffer(wp + ".v_proj.bias");
    cl_mem o_w = weights.get_buffer(wp + ".out_proj.weight");
    cl_mem o_b = weights.get_buffer(wp + ".out_proj.bias");

    // k_proj has bias=False in Whisper.
    if (!q_w || !q_b || !k_w || !v_w || !v_b || !o_w || !o_b) {
        NNOPT_ERROR_FMT("WhisperSdpaAttention_forward: missing weights for %s", wp.c_str());
        return nullptr;
    }

    cl_int err = CL_SUCCESS;

    cl_mem q_th = nullptr;        // [Tq, hidden]
    cl_mem k_th = nullptr;        // [Tk, hidden]
    cl_mem v_th = nullptr;        // [Tk, hidden]
    cl_mem q_htd = nullptr;       // [H, Tq, D]
    cl_mem k_htd = nullptr;       // [H, Tk, D]
    cl_mem v_htd = nullptr;       // [H, Tk, D]
    cl_mem scores = nullptr;      // float [H,Tq,Tk]
    cl_mem ctx_htd = nullptr;     // [H,Tq,D]
    cl_mem ctx_thd = nullptr;     // [Tq,H,D]
    cl_mem out = nullptr;         // [Tq,hidden]

    // NOTE: q_th/k_th/v_th/q_htd/scores/ctx_htd/ctx_thd are POOLED scratch
    // (nnopt_attn_scratch) — persistent and reused, never released here. Only the
    // per-call / cache-borrowed / returned buffers are released: k_htd, v_htd, out.
    auto cleanup = [&]() -> cl_mem {
        if (k_htd) { clReleaseMemObject(k_htd); k_htd = nullptr; }
        if (v_htd) { clReleaseMemObject(v_htd); v_htd = nullptr; }
        if (out) { clReleaseMemObject(out); out = nullptr; }
        return nullptr;
    };

    // Static program caches (Rule PROG-01).
    static cl_program s_attn_prog = nullptr;
    static cl_kernel s_k_scores = nullptr;
    static cl_kernel s_k_softmax = nullptr;
    static cl_kernel s_k_wsum = nullptr;
    static cl_kernel s_k_trans = nullptr;

    static cl_program s_pack_prog = nullptr;
    static cl_kernel s_k_pack = nullptr;
    static cl_kernel s_k_pack_cache = nullptr;   // opt #12: pack K/V straight into the KV cache

    static cl_program s_bias_prog = nullptr;
    static cl_kernel s_k_bias = nullptr;

    if (!s_attn_prog) {
        // PROGRAM-INIT-OK: function-local static cache
        s_attn_prog = cl_ctx.build_program_from_file("kernels/attn.cl");
        if (!s_attn_prog) { NNOPT_ERROR("WhisperSdpaAttention_forward: build kernels/attn.cl failed"); return nullptr; }
        cl_int kerr = CL_SUCCESS;
        s_k_scores = clCreateKernel(s_attn_prog, "attn_scores", &kerr);
        if (kerr != CL_SUCCESS || !s_k_scores) { NNOPT_ERROR_FMT("attn_scores kernel create failed %d", (int)kerr); return nullptr; }
        s_k_softmax = clCreateKernel(s_attn_prog, "attn_softmax", &kerr);
        if (kerr != CL_SUCCESS || !s_k_softmax) { NNOPT_ERROR_FMT("attn_softmax kernel create failed %d", (int)kerr); return nullptr; }
        s_k_wsum = clCreateKernel(s_attn_prog, "attn_weighted_sum", &kerr);
        if (kerr != CL_SUCCESS || !s_k_wsum) { NNOPT_ERROR_FMT("attn_weighted_sum kernel create failed %d", (int)kerr); return nullptr; }
        s_k_trans = clCreateKernel(s_attn_prog, "attn_transpose_htd_to_thd", &kerr);
        if (kerr != CL_SUCCESS || !s_k_trans) { NNOPT_ERROR_FMT("attn_transpose kernel create failed %d", (int)kerr); return nullptr; }
    }

    if (!s_pack_prog) {
        // PROGRAM-INIT-OK: function-local static cache
        s_pack_prog = cl_ctx.build_program_from_file("kernels/pack.cl");
        if (!s_pack_prog) { NNOPT_ERROR("WhisperSdpaAttention_forward: build kernels/pack.cl failed"); return nullptr; }
        cl_int kerr = CL_SUCCESS;
        s_k_pack = clCreateKernel(s_pack_prog, "pack_th_to_htd", &kerr);
        if (kerr != CL_SUCCESS || !s_k_pack) { NNOPT_ERROR_FMT("pack_th_to_htd kernel create failed %d", (int)kerr); return nullptr; }
        s_k_pack_cache = clCreateKernel(s_pack_prog, "pack_th_to_htd_cache", &kerr);
        if (kerr != CL_SUCCESS || !s_k_pack_cache) { NNOPT_ERROR_FMT("pack_th_to_htd_cache kernel create failed %d", (int)kerr); return nullptr; }
    }

    if (!s_bias_prog) {
        // PROGRAM-INIT-OK: function-local static cache
        s_bias_prog = cl_ctx.build_program_from_file("kernels/bias_add.cl");
        if (!s_bias_prog) { NNOPT_ERROR("WhisperSdpaAttention_forward: build kernels/bias_add.cl failed"); return nullptr; }
        cl_int kerr = CL_SUCCESS;
        s_k_bias = clCreateKernel(s_bias_prog, "bias_add_rows", &kerr);
        if (kerr != CL_SUCCESS || !s_k_bias) { NNOPT_ERROR_FMT("bias_add_rows kernel create failed %d", (int)kerr); return nullptr; }
    }

    // ── Cross-attention K/V cache (opt #7) ──────────────────────────────────
    // Cross-attn K/V are projected+packed from the FIXED encoder output, so they
    // are identical across every decode step. Project once, then reuse the packed
    // [H,Tk,D] buffers — skipping k/v GEMM (Tk=1500), v-bias, and pack_k/pack_v on
    // every subsequent step. Keyed by weight_prefix (e.g. "...layers.2.encoder_attn"),
    // which uniquely names the layer. Process-lifetime cache = per-clip (each clip
    // is a fresh binary run), and the encoder output is constant within a run.
    // k_htd/v_htd are never mutated after packing (only q_htd is scaled), so this
    // is safe. The cache holds its own retained reference; the function borrows one
    // (retained on hit, retained-into-cache on miss) and releases it at the end.
    // s_cross_kv_cache / s_self_kv_cache are file-scope (resettable between clips).
    bool cross_kv_hit = false;
    if (is_cross) {
        auto it = s_cross_kv_cache.find(wp);
        if (it != s_cross_kv_cache.end()) {
            k_htd = it->second.first;  clRetainMemObject(k_htd);
            v_htd = it->second.second; clRetainMemObject(v_htd);
            cross_kv_hit = true;
        }
    }

    // ── Decoder self-attention KV cache (opt #12) ───────────────────────────
    // In incremental decode every call processes ONE new token (Tq==1), so the
    // single query attends to all PAST positions — no causal mask needed. We keep
    // a persistent per-layer K/V cache [H, CAP, D] (CAP = max target positions),
    // append this token's packed K/V at column `start_pos`, and run the score /
    // context GEMMs over the first start_pos+1 columns (stride CAP). This turns
    // decode from O(N²) (reprocess the whole prefix each step) into O(N).
    // Gate: decoder self-attn (use_causal_mask) AND single-token (Tq==1). The
    // rare multi-token decoder prefill (e.g. --token-ids eval) keeps the naive
    // causal path below. Cache is process-lifetime = per-clip (fresh binary run).
    const int KV_CAP = (int)MODEL_CONFIG::MAX_TARGET_POSITIONS;
    const bool is_dec_self = use_causal_mask;         // decoder self-attn (uses the KV cache)
    const bool self_decode = is_dec_self && (Tq == 1); // single-token step → GEMM against cache
    const int Tk_attn = self_decode ? (start_pos + Tq) : Tk;
    cl_mem self_kcache = nullptr, self_vcache = nullptr;
    if (is_dec_self) {
        if (start_pos + Tq > KV_CAP) {
            NNOPT_ERROR_FMT("WhisperSdpaAttention_forward: self-attn KV cache overflow (start_pos=%d Tq=%d > CAP=%d)",
                            start_pos, Tq, KV_CAP);
            return cleanup();
        }
        auto it = s_self_kv_cache.find(wp);
        if (it == s_self_kv_cache.end()) {
            cl_int ce = CL_SUCCESS;
            cl_mem kc = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                       (size_t)H * (size_t)KV_CAP * (size_t)D * sizeof(nnopt_storage_t), nullptr, &ce);
            if (ce != CL_SUCCESS || !kc) { NNOPT_ERROR_FMT("clCreateBuffer(self kcache) %d", (int)ce); return cleanup(); }
            cl_mem vc = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                       (size_t)H * (size_t)KV_CAP * (size_t)D * sizeof(nnopt_storage_t), nullptr, &ce);
            if (ce != CL_SUCCESS || !vc) { NNOPT_ERROR_FMT("clCreateBuffer(self vcache) %d", (int)ce); clReleaseMemObject(kc); return cleanup(); }
            it = s_self_kv_cache.emplace(wp, std::make_pair(kc, vc)).first;
        }
        self_kcache = it->second.first;
        self_vcache = it->second.second;
    }

    // Pooled scratch slots (opt #9) — persistent, grow-only, reused across calls.
    static cl_mem s_q_th = nullptr;    static size_t s_q_th_cap = 0;
    static cl_mem s_k_th = nullptr;    static size_t s_k_th_cap = 0;
    static cl_mem s_v_th = nullptr;    static size_t s_v_th_cap = 0;
    static cl_mem s_q_htd = nullptr;   static size_t s_q_htd_cap = 0;
    static cl_mem s_scores = nullptr;  static size_t s_scores_cap = 0;
    static cl_mem s_ctx_htd = nullptr; static size_t s_ctx_htd_cap = 0;
    static cl_mem s_ctx_thd = nullptr; static size_t s_ctx_thd_cap = 0;

    // Allocate projections. Q is always recomputed (depends on the decoder token).
    q_th = nnopt_attn_scratch(cl_ctx, s_q_th, s_q_th_cap, (size_t)Tq * (size_t)hidden * sizeof(nnopt_storage_t));
    if (!q_th) { NNOPT_ERROR("scratch(q_th) alloc failed"); return cleanup(); }

    if (!cross_kv_hit) {
        k_th = nnopt_attn_scratch(cl_ctx, s_k_th, s_k_th_cap, (size_t)Tk * (size_t)hidden * sizeof(nnopt_storage_t));
        if (!k_th) { NNOPT_ERROR("scratch(k_th) alloc failed"); return cleanup(); }
        v_th = nnopt_attn_scratch(cl_ctx, s_v_th, s_v_th_cap, (size_t)Tk * (size_t)hidden * sizeof(nnopt_storage_t));
        if (!v_th) { NNOPT_ERROR("scratch(v_th) alloc failed"); return cleanup(); }
    }

    // Q = input @ q_w^T + q_b
    if (!pytorch_linear(queue, /*M=*/Tq, /*N=*/hidden, /*K=*/hidden, input, q_w, q_th,
                        is_cross ? "gemm_crossattn_q" : "gemm_selfattn_q")) {
        NNOPT_ERROR("WhisperSdpaAttention_forward: q_proj gemm failed");
        return cleanup();
    }
    {
        cl_kernel bk = s_k_bias;
        clRetainKernel(bk);
        if (!set_arg_checked(bk, 0, sizeof(cl_mem), &q_th, "y")) { clReleaseKernel(bk); return cleanup(); }
        if (!set_arg_checked(bk, 1, sizeof(cl_mem), &q_b, "bias")) { clReleaseKernel(bk); return cleanup(); }
        if (!set_arg_checked(bk, 2, sizeof(int), &Tq, "rows")) { clReleaseKernel(bk); return cleanup(); }
        if (!set_arg_checked(bk, 3, sizeof(int), &hidden, "cols")) { clReleaseKernel(bk); return cleanup(); }
        size_t gws = (size_t)Tq * (size_t)hidden;
        err = clEnqueueNDRangeKernel(queue, bk, 1, nullptr, &gws, nullptr, 0, nullptr,
                                     KernelProfiler::event_for("bias_add_q"));
        clReleaseKernel(bk);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("bias_add q dispatch %d", (int)err); return cleanup(); }
    }

    // K,V source + projection — skipped entirely on a cross-attn cache hit.
    if (!cross_kv_hit) {
        cl_mem kv_src = is_cross ? encoder_hidden_states : input;
        if (!kv_src) { NNOPT_ERROR("WhisperSdpaAttention_forward: kv_src null"); return cleanup(); }

        if (!pytorch_linear(queue, /*M=*/Tk, /*N=*/hidden, /*K=*/hidden, kv_src, k_w, k_th,
                            is_cross ? "gemm_crossattn_k" : "gemm_selfattn_k")) {
            NNOPT_ERROR("WhisperSdpaAttention_forward: k_proj gemm failed");
            return cleanup();
        }

        if (!pytorch_linear(queue, /*M=*/Tk, /*N=*/hidden, /*K=*/hidden, kv_src, v_w, v_th,
                            is_cross ? "gemm_crossattn_v" : "gemm_selfattn_v")) {
            NNOPT_ERROR("WhisperSdpaAttention_forward: v_proj gemm failed");
            return cleanup();
        }
        {
            cl_kernel bk = s_k_bias;
            clRetainKernel(bk);
            if (!set_arg_checked(bk, 0, sizeof(cl_mem), &v_th, "y")) { clReleaseKernel(bk); return cleanup(); }
            if (!set_arg_checked(bk, 1, sizeof(cl_mem), &v_b, "bias")) { clReleaseKernel(bk); return cleanup(); }
            if (!set_arg_checked(bk, 2, sizeof(int), &Tk, "rows")) { clReleaseKernel(bk); return cleanup(); }
            if (!set_arg_checked(bk, 3, sizeof(int), &hidden, "cols")) { clReleaseKernel(bk); return cleanup(); }
            size_t gws = (size_t)Tk * (size_t)hidden;
            err = clEnqueueNDRangeKernel(queue, bk, 1, nullptr, &gws, nullptr, 0, nullptr,
                                         KernelProfiler::event_for("bias_add_v"));
            clReleaseKernel(bk);
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("bias_add v dispatch %d", (int)err); return cleanup(); }
        }
    }

    // Pack Q/K/V from [T,hidden] into [H,T,D]. Q is always packed; K/V are packed
    // only on a cache miss (on a hit, k_htd/v_htd already hold the cached values).
    q_htd = nnopt_attn_scratch(cl_ctx, s_q_htd, s_q_htd_cap, (size_t)H * (size_t)Tq * (size_t)D * sizeof(nnopt_storage_t));
    if (!q_htd) { NNOPT_ERROR("scratch(q_htd) alloc failed"); return cleanup(); }

    // Contiguous [H,Tk,D] K/V buffers — not needed when self_decode packs straight
    // into the persistent cache, nor on a cross-attn cache hit.
    if (!cross_kv_hit && !self_decode) {
        k_htd = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                               (size_t)H * (size_t)Tk * (size_t)D * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS || !k_htd) { NNOPT_ERROR_FMT("clCreateBuffer(k_htd) %d", (int)err); return cleanup(); }

        v_htd = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                               (size_t)H * (size_t)Tk * (size_t)D * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS || !v_htd) { NNOPT_ERROR_FMT("clCreateBuffer(v_htd) %d", (int)err); return cleanup(); }
    }

    {
        cl_kernel pk = s_k_pack;
        clRetainKernel(pk);
        if (!set_arg_checked(pk, 0, sizeof(cl_mem), &q_th, "in_th")) { clReleaseKernel(pk); return cleanup(); }
        if (!set_arg_checked(pk, 1, sizeof(cl_mem), &q_htd, "out_htd")) { clReleaseKernel(pk); return cleanup(); }
        if (!set_arg_checked(pk, 2, sizeof(int), &Tq, "T")) { clReleaseKernel(pk); return cleanup(); }
        if (!set_arg_checked(pk, 3, sizeof(int), &H, "H")) { clReleaseKernel(pk); return cleanup(); }
        if (!set_arg_checked(pk, 4, sizeof(int), &D, "D")) { clReleaseKernel(pk); return cleanup(); }
        size_t gws = (size_t)H * (size_t)Tq * (size_t)D;
        err = clEnqueueNDRangeKernel(queue, pk, 1, nullptr, &gws, nullptr, 0, nullptr,
                                     KernelProfiler::event_for("pack_q"));
        clReleaseKernel(pk);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("pack_q dispatch %d", (int)err); return cleanup(); }
    }
    if (self_decode) {
        // Pack this token's K/V straight into the persistent cache at column
        // start_pos (one launch each). After this the cache holds positions
        // 0..start_pos and the attention GEMMs read it with per-head stride CAP.
        cl_mem packs[2][2] = { { k_th, self_kcache }, { v_th, self_vcache } };
        const char* plabels[2] = { "pack_k_cache", "pack_v_cache" };
        for (int s = 0; s < 2; ++s) {
            cl_kernel pk = s_k_pack_cache;
            clRetainKernel(pk);
            if (!set_arg_checked(pk, 0, sizeof(cl_mem), &packs[s][0], "in_th")) { clReleaseKernel(pk); return cleanup(); }
            if (!set_arg_checked(pk, 1, sizeof(cl_mem), &packs[s][1], "out_cache")) { clReleaseKernel(pk); return cleanup(); }
            if (!set_arg_checked(pk, 2, sizeof(int), &Tq, "T")) { clReleaseKernel(pk); return cleanup(); }
            if (!set_arg_checked(pk, 3, sizeof(int), &H, "H")) { clReleaseKernel(pk); return cleanup(); }
            if (!set_arg_checked(pk, 4, sizeof(int), &D, "D")) { clReleaseKernel(pk); return cleanup(); }
            if (!set_arg_checked(pk, 5, sizeof(int), &KV_CAP, "CAP")) { clReleaseKernel(pk); return cleanup(); }
            if (!set_arg_checked(pk, 6, sizeof(int), &start_pos, "col0")) { clReleaseKernel(pk); return cleanup(); }
            size_t gws = (size_t)H * (size_t)Tq * (size_t)D;
            err = clEnqueueNDRangeKernel(queue, pk, 1, nullptr, &gws, nullptr, 0, nullptr,
                                         KernelProfiler::event_for(plabels[s]));
            clReleaseKernel(pk);
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("%s dispatch %d", plabels[s], (int)err); return cleanup(); }
        }
    } else if (!cross_kv_hit) {
        {
            cl_kernel pk = s_k_pack;
            clRetainKernel(pk);
            if (!set_arg_checked(pk, 0, sizeof(cl_mem), &k_th, "in_th")) { clReleaseKernel(pk); return cleanup(); }
            if (!set_arg_checked(pk, 1, sizeof(cl_mem), &k_htd, "out_htd")) { clReleaseKernel(pk); return cleanup(); }
            if (!set_arg_checked(pk, 2, sizeof(int), &Tk, "T")) { clReleaseKernel(pk); return cleanup(); }
            if (!set_arg_checked(pk, 3, sizeof(int), &H, "H")) { clReleaseKernel(pk); return cleanup(); }
            if (!set_arg_checked(pk, 4, sizeof(int), &D, "D")) { clReleaseKernel(pk); return cleanup(); }
            size_t gws = (size_t)H * (size_t)Tk * (size_t)D;
            err = clEnqueueNDRangeKernel(queue, pk, 1, nullptr, &gws, nullptr, 0, nullptr,
                                         KernelProfiler::event_for("pack_k"));
            clReleaseKernel(pk);
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("pack_k dispatch %d", (int)err); return cleanup(); }
        }
        {
            cl_kernel pk = s_k_pack;
            clRetainKernel(pk);
            if (!set_arg_checked(pk, 0, sizeof(cl_mem), &v_th, "in_th")) { clReleaseKernel(pk); return cleanup(); }
            if (!set_arg_checked(pk, 1, sizeof(cl_mem), &v_htd, "out_htd")) { clReleaseKernel(pk); return cleanup(); }
            if (!set_arg_checked(pk, 2, sizeof(int), &Tk, "T")) { clReleaseKernel(pk); return cleanup(); }
            if (!set_arg_checked(pk, 3, sizeof(int), &H, "H")) { clReleaseKernel(pk); return cleanup(); }
            if (!set_arg_checked(pk, 4, sizeof(int), &D, "D")) { clReleaseKernel(pk); return cleanup(); }
            size_t gws = (size_t)H * (size_t)Tk * (size_t)D;
            err = clEnqueueNDRangeKernel(queue, pk, 1, nullptr, &gws, nullptr, 0, nullptr,
                                         KernelProfiler::event_for("pack_v"));
            clReleaseKernel(pk);
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("pack_v dispatch %d", (int)err); return cleanup(); }
        }

        // Populate the cross-attn cache on first computation. Retain so the cache
        // owns an independent reference; the function's own reference is released
        // by the normal success/cleanup path below. (opt #7)
        if (is_cross) {
            clRetainMemObject(k_htd);
            clRetainMemObject(v_htd);
            s_cross_kv_cache[wp] = std::make_pair(k_htd, v_htd);
        }

        // Multi-token decoder self-attn prefill (e.g. --token-ids eval): this call
        // uses the naive causal path above, but ALSO seed the KV cache so later
        // single-token decode steps have the prefix. (opt #12)
        if (is_dec_self) {
            cl_mem packs[2][2] = { { k_th, self_kcache }, { v_th, self_vcache } };
            const char* plabels[2] = { "pack_k_cache", "pack_v_cache" };
            for (int s = 0; s < 2; ++s) {
                cl_kernel pk = s_k_pack_cache;
                clRetainKernel(pk);
                if (!set_arg_checked(pk, 0, sizeof(cl_mem), &packs[s][0], "in_th")) { clReleaseKernel(pk); return cleanup(); }
                if (!set_arg_checked(pk, 1, sizeof(cl_mem), &packs[s][1], "out_cache")) { clReleaseKernel(pk); return cleanup(); }
                if (!set_arg_checked(pk, 2, sizeof(int), &Tk, "T")) { clReleaseKernel(pk); return cleanup(); }
                if (!set_arg_checked(pk, 3, sizeof(int), &H, "H")) { clReleaseKernel(pk); return cleanup(); }
                if (!set_arg_checked(pk, 4, sizeof(int), &D, "D")) { clReleaseKernel(pk); return cleanup(); }
                if (!set_arg_checked(pk, 5, sizeof(int), &KV_CAP, "CAP")) { clReleaseKernel(pk); return cleanup(); }
                if (!set_arg_checked(pk, 6, sizeof(int), &start_pos, "col0")) { clReleaseKernel(pk); return cleanup(); }
                size_t gws = (size_t)H * (size_t)Tk * (size_t)D;
                err = clEnqueueNDRangeKernel(queue, pk, 1, nullptr, &gws, nullptr, 0, nullptr,
                                             KernelProfiler::event_for(plabels[s]));
                clReleaseKernel(pk);
                if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("%s (prefill seed) dispatch %d", plabels[s], (int)err); return cleanup(); }
            }
        }
    }

    // scores/probs buffer — storage_t (fp16 under USE_FP16) so it can be both the
    // output of the QKᵀ GEMM and the input of the probs@V GEMM (opt #1/#2).
    // Tk_attn = number of keys attended (= start_pos+1 for the cached self-decode
    // path, else Tk).
    scores = nnopt_attn_scratch(cl_ctx, s_scores, s_scores_cap, (size_t)H * (size_t)Tq * (size_t)Tk_attn * sizeof(nnopt_storage_t));
    if (!scores) { NNOPT_ERROR("scratch(scores) alloc failed"); return cleanup(); }

    // scores = (Q * 1/sqrt(D)) @ Kᵀ.
    // Non-causal path (encoder self-attn + decoder cross-attn — the 93% hotspot):
    //   one CLBlast GemmStridedBatched over all heads, with the query scaling
    //   folded into alpha (opt #1/#4). No separate scale kernel, no naive scores.
    // Causal path (decoder self-attn): the GEMM can't express the triangular mask,
    //   so keep the naive kernel — first scale Q in place, then attn_scores applies
    //   the mask. This path is tiny (Tq=Tk=seq_len, a few dozen) so it's not a
    //   bottleneck.
    if (self_decode) {
        // Cached single-token decode: one GEMM over the first Tk_attn cache
        // columns (per-head stride KV_CAP). No mask — the lone query is at the
        // latest position and attends to every cached past key. (opt #12)
        if (!attn_scores_batched(queue, H, Tq, Tk_attn, D, q_scaling, q_htd, self_kcache, scores,
                                 "attn_scores_gemm", KV_CAP)) {
            NNOPT_ERROR("WhisperSdpaAttention_forward: attn_scores_batched (self-decode) failed");
            return cleanup();
        }
    } else if (!use_causal_mask) {
        if (!attn_scores_batched(queue, H, Tq, Tk, D, q_scaling, q_htd, k_htd, scores)) {
            NNOPT_ERROR("WhisperSdpaAttention_forward: attn_scores_batched failed");
            return cleanup();
        }
    } else {
        // Scale Q in place (HF pre-scales the query, then uses scaling=1.0).
        {
            static cl_program s_scale_prog = nullptr;
            static cl_kernel s_scale_k = nullptr;
            if (!s_scale_prog) {
                // PROGRAM-INIT-OK
                s_scale_prog = cl_ctx.build_program_from_file("kernels/scale_float.cl");
                if (!s_scale_prog) { NNOPT_ERROR("WhisperSdpaAttention_forward: build kernels/scale_float.cl failed"); return cleanup(); }
                cl_int kerr = CL_SUCCESS;
                s_scale_k = clCreateKernel(s_scale_prog, "scale_storage_inplace", &kerr);
                if (kerr != CL_SUCCESS || !s_scale_k) { NNOPT_ERROR_FMT("scale_storage_inplace kernel create failed %d", (int)kerr); return cleanup(); }
            }
            cl_kernel sk = s_scale_k;
            clRetainKernel(sk);
            const int n = H * Tq * D;
            if (!set_arg_checked(sk, 0, sizeof(cl_mem), &q_htd, "x")) { clReleaseKernel(sk); return cleanup(); }
            if (!set_arg_checked(sk, 1, sizeof(float), &q_scaling, "scale")) { clReleaseKernel(sk); return cleanup(); }
            if (!set_arg_checked(sk, 2, sizeof(int), &n, "n")) { clReleaseKernel(sk); return cleanup(); }
            size_t gws = (size_t)n;
            err = clEnqueueNDRangeKernel(queue, sk, 1, nullptr, &gws, nullptr, 0, nullptr,
                                         KernelProfiler::event_for("attn_scale_q"));
            clReleaseKernel(sk);
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("attn_scale_q dispatch %d", (int)err); return cleanup(); }
        }
        cl_kernel ks = s_k_scores;
        clRetainKernel(ks);
        int causal = 1;
        if (!set_arg_checked(ks, 0, sizeof(cl_mem), &q_htd, "q")) { clReleaseKernel(ks); return cleanup(); }
        if (!set_arg_checked(ks, 1, sizeof(cl_mem), &k_htd, "k")) { clReleaseKernel(ks); return cleanup(); }
        if (!set_arg_checked(ks, 2, sizeof(cl_mem), &scores, "scores")) { clReleaseKernel(ks); return cleanup(); }
        if (!set_arg_checked(ks, 3, sizeof(int), &Tq, "Tq")) { clReleaseKernel(ks); return cleanup(); }
        if (!set_arg_checked(ks, 4, sizeof(int), &Tk, "Tk")) { clReleaseKernel(ks); return cleanup(); }
        if (!set_arg_checked(ks, 5, sizeof(int), &H, "H")) { clReleaseKernel(ks); return cleanup(); }
        if (!set_arg_checked(ks, 6, sizeof(int), &D, "D")) { clReleaseKernel(ks); return cleanup(); }
        if (!set_arg_checked(ks, 7, sizeof(int), &causal, "causal")) { clReleaseKernel(ks); return cleanup(); }
        size_t gws = (size_t)H * (size_t)Tq * (size_t)Tk;
        err = clEnqueueNDRangeKernel(queue, ks, 1, nullptr, &gws, nullptr, 0, nullptr,
                                     KernelProfiler::event_for("attn_scores"));
        clReleaseKernel(ks);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("attn_scores dispatch %d", (int)err); return cleanup(); }
    }

    // softmax in-place on scores (becomes probs). One workgroup per (h,tq) row;
    // SOFTMAX_WG must match the kernel's reqd_work_group_size in kernels/attn.cl.
    {
        const size_t SOFTMAX_WG = 128;
        cl_kernel km = s_k_softmax;
        clRetainKernel(km);
        if (!set_arg_checked(km, 0, sizeof(cl_mem), &scores, "scores")) { clReleaseKernel(km); return cleanup(); }
        if (!set_arg_checked(km, 1, sizeof(int), &Tq, "Tq")) { clReleaseKernel(km); return cleanup(); }
        if (!set_arg_checked(km, 2, sizeof(int), &Tk_attn, "Tk")) { clReleaseKernel(km); return cleanup(); }
        if (!set_arg_checked(km, 3, sizeof(int), &H, "H")) { clReleaseKernel(km); return cleanup(); }
        size_t lws = SOFTMAX_WG;
        size_t gws = (size_t)H * (size_t)Tq * SOFTMAX_WG;
        err = clEnqueueNDRangeKernel(queue, km, 1, nullptr, &gws, &lws, 0, nullptr,
                                     KernelProfiler::event_for("attn_softmax"));
        clReleaseKernel(km);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("attn_softmax dispatch %d", (int)err); return cleanup(); }
    }

    // ctx_htd = probs @ v
    ctx_htd = nnopt_attn_scratch(cl_ctx, s_ctx_htd, s_ctx_htd_cap, (size_t)H * (size_t)Tq * (size_t)D * sizeof(nnopt_storage_t));
    if (!ctx_htd) { NNOPT_ERROR("scratch(ctx_htd) alloc failed"); return cleanup(); }

    // ctx = probs @ V — one CLBlast GemmStridedBatched over all heads (opt #2).
    // Works for both causal and non-causal: the mask is already baked into probs
    // by the softmax. For the cached self-decode path, V is the cache (per-head
    // stride KV_CAP) over Tk_attn columns. Replaces the naive attn_wsum.
    {
        cl_mem v_for_ctx = self_decode ? self_vcache : v_htd;
        int v_stride = self_decode ? KV_CAP : -1;
        if (!attn_context_batched(queue, H, Tq, Tk_attn, D, scores, v_for_ctx, ctx_htd,
                                  "attn_wsum_gemm", v_stride)) {
            NNOPT_ERROR("WhisperSdpaAttention_forward: attn_context_batched failed");
            return cleanup();
        }
    }
    (void)s_k_wsum;  // naive weighted-sum kernel retained as fallback; unused on the live path

    // ctx_thd = transpose [H,Tq,D] -> [Tq,H,D]
    ctx_thd = nnopt_attn_scratch(cl_ctx, s_ctx_thd, s_ctx_thd_cap, (size_t)Tq * (size_t)hidden * sizeof(nnopt_storage_t));
    if (!ctx_thd) { NNOPT_ERROR("scratch(ctx_thd) alloc failed"); return cleanup(); }

    {
        cl_kernel kt = s_k_trans;
        clRetainKernel(kt);
        if (!set_arg_checked(kt, 0, sizeof(cl_mem), &ctx_htd, "x_htd")) { clReleaseKernel(kt); return cleanup(); }
        if (!set_arg_checked(kt, 1, sizeof(cl_mem), &ctx_thd, "y_thd")) { clReleaseKernel(kt); return cleanup(); }
        if (!set_arg_checked(kt, 2, sizeof(int), &Tq, "Tq")) { clReleaseKernel(kt); return cleanup(); }
        if (!set_arg_checked(kt, 3, sizeof(int), &H, "H")) { clReleaseKernel(kt); return cleanup(); }
        if (!set_arg_checked(kt, 4, sizeof(int), &D, "D")) { clReleaseKernel(kt); return cleanup(); }
        size_t gws = (size_t)Tq * (size_t)H * (size_t)D;
        err = clEnqueueNDRangeKernel(queue, kt, 1, nullptr, &gws, nullptr, 0, nullptr,
                                     KernelProfiler::event_for("attn_transpose"));
        clReleaseKernel(kt);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("attn_transpose dispatch %d", (int)err); return cleanup(); }
    }

    // out = out_proj(ctx_thd) + o_b
    out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                         (size_t)Tq * (size_t)hidden * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("clCreateBuffer(out) %d", (int)err); return cleanup(); }

    if (!pytorch_linear(queue, /*M=*/Tq, /*N=*/hidden, /*K=*/hidden, ctx_thd, o_w, out,
                        is_cross ? "gemm_crossattn_o" : "gemm_selfattn_o")) {
        NNOPT_ERROR("WhisperSdpaAttention_forward: out_proj gemm failed");
        return cleanup();
    }

    {
        cl_kernel bk = s_k_bias;
        clRetainKernel(bk);
        if (!set_arg_checked(bk, 0, sizeof(cl_mem), &out, "y")) { clReleaseKernel(bk); return cleanup(); }
        if (!set_arg_checked(bk, 1, sizeof(cl_mem), &o_b, "bias")) { clReleaseKernel(bk); return cleanup(); }
        if (!set_arg_checked(bk, 2, sizeof(int), &Tq, "rows")) { clReleaseKernel(bk); return cleanup(); }
        if (!set_arg_checked(bk, 3, sizeof(int), &hidden, "cols")) { clReleaseKernel(bk); return cleanup(); }
        size_t gws = (size_t)Tq * (size_t)hidden;
        err = clEnqueueNDRangeKernel(queue, bk, 1, nullptr, &gws, nullptr, 0, nullptr,
                                     KernelProfiler::event_for("bias_add_o"));
        clReleaseKernel(bk);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("bias_add o dispatch %d", (int)err); return cleanup(); }
    }

    // Success: release only the per-call / cache-borrowed buffers; the pooled
    // scratch (q_th/k_th/v_th/q_htd/scores/ctx_htd/ctx_thd) persists for reuse (opt #9).
    if (k_htd) { clReleaseMemObject(k_htd); k_htd = nullptr; }
    if (v_htd) { clReleaseMemObject(v_htd); v_htd = nullptr; }

    return out;
}
} // extern "C"
