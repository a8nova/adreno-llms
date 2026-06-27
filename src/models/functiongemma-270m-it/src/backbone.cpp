// Auto-generated backbone for functiongemma-270m-it.
//
// This file is intentionally small: the scaffold emits only TOP-LEVEL ops
// here. Composite ops (the per-layer wrapper) call their primitives
// internally, so backbone.cpp doesn't grow with NUM_LAYERS.
//
// Per-class trace counts (informational):
//    127× Linear
//    109× Gemma3RMSNorm
//     18× Gemma3DecoderLayer
//     18× Gemma3Attention
//     18× GELUTanh
//     18× Gemma3MLP
//      2× Gemma3RotaryEmbedding
//      1× Gemma3TextScaledWordEmbedding
//
// ─── UNIVERSAL <Class>_forward SIGNATURE ───────────────────────────────
// Every <Class>_forward in src/ops/<Class>.cpp uses these 11 args:
//   (OpenCLContext&, Weights&, cl_command_queue,
//    cl_mem input, int seq_len, int layer_idx, int start_pos,
//    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
//    cl_mem encoder_hidden_states,
//    const char* weight_prefix) → returns cl_mem output
//
// encoder_hidden_states is non-null only for cross-attention calls in
// encoder-decoder models (Whisper, T5, SeamlessM4T). Pass nullptr for
// every other call (causal LM ops, encoder self-attention, primitives).
//
// ─── THE AGENT'S JOB IN THIS FILE ──────────────────────────────────────
// 1. Add per-layer K/V caches to Model (std::vector<cl_mem> with
//    NUM_HIDDEN_LAYERS entries). Pass &k_caches[layer_idx],
//    &v_caches[layer_idx] into the per-layer call below.
// 2. The per-layer LAYER_CHECK name needs the per-iteration dump_name —
//    sprintf "layer_%d" (or whatever the trace uses; see
//    .nnport/scaffold_manifest.json::nodes for the canonical strings).
// 3. The final logits readout below reads the LAST token's row at
//    sizeof(nnopt_storage_t) × VOCAB_SIZE bytes from x. If the trace's
//    final node has a different output shape, fix the readout.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"
#include "profiler.h"
#include <cstdio>

extern "C" cl_mem Gemma3DecoderLayer_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem Gemma3TextScaledWordEmbedding_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem Gemma3RotaryEmbedding_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem Gemma3RMSNorm_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem Linear_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);

// fp32-residual helper (defined in src/ops/Gemma3RMSNorm.cpp). Converts the
// fp16 embedding output into a fresh fp32 buffer so the residual stream — which
// in Gemma3 exceeds fp16 max (~65504) by mid-stack — never overflows.
cl_mem gemma3_f16_to_f32_run(OpenCLContext& cl_ctx, cl_command_queue queue,
                             cl_mem in_f16, int n);

// ── Greedy GPU-argmax fast-path ────────────────────────────────────────────
// When enabled (set by main for temperature<=0), the forward computes argmax on
// the GPU and reads back only the token id, skipping the 262144-element logits
// readback + host f16→f32 convert + host argmax (all run with the GPU idle).
static bool g_greedy_argmax = false;
static int  g_argmax_token  = -1;
extern "C" void nnopt_set_greedy_argmax(int on) { g_greedy_argmax = (on != 0); }
extern "C" int  nnopt_get_argmax_token() { return g_argmax_token; }

// Returns the argmax index of logits[0:n] computed on the GPU, or -1 on failure.
static int nnopt_gpu_argmax(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem logits, int n) {
    static cl_program s_prog = nullptr;
    static cl_kernel  s_k = nullptr;
    if (!s_k) {
        s_prog = cl_ctx.build_program_from_file("kernels/gemma3_ops.cl");
        if (!s_prog) { NNOPT_ERROR("argmax: build failed"); return -1; }
        cl_int e = CL_SUCCESS;
        s_k = clCreateKernel(s_prog, "gemma3_argmax", &e);
        if (e != CL_SUCCESS || !s_k) { NNOPT_ERROR_FMT("argmax: clCreateKernel %d", (int)e); s_k = nullptr; return -1; }
    }
    cl_int err = CL_SUCCESS;
    cl_mem out = nnopt_pool_alloc(cl_ctx.context(), sizeof(int), &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("argmax: alloc %d", (int)err); return -1; }
    if (!set_arg_checked(s_k, 0, sizeof(cl_mem), &logits, "logits") ||
        !set_arg_checked(s_k, 1, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(s_k, 2, sizeof(int), &n, "n")) { nnopt_pool_free(out); return -1; }
    size_t lws = 256, gws = 256;   // single workgroup, must match ARGMAX_WG
    err = clEnqueueNDRangeKernel(queue, s_k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("argmax: dispatch %d", (int)err); nnopt_pool_free(out); return -1; }
    int idx = -1;
    err = clEnqueueReadBuffer(queue, out, CL_TRUE, 0, sizeof(int), &idx, 0, nullptr, nullptr);
    nnopt_pool_free(out);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("argmax: readback %d", (int)err); return -1; }
    return idx;
}

// ── Per-layer KV cache (persists across forward() calls) ───────────────────
// Holds fp32 K and V (post-norm, post-RoPE) for every absolute position, so
// decode processes only the new token (M=1) instead of reprocessing the whole
// growing sequence. Re-allocated at prefill (start_pos==0). Capacity = prompt
// length + KV_CACHE_DECODE_MARGIN generated tokens.
static std::vector<cl_mem> g_k_cache;
static std::vector<cl_mem> g_v_cache;
static int g_kv_cache_cap = 0;
static const int KV_CACHE_DECODE_MARGIN = 2048;

static bool ensure_kv_cache(OpenCLContext& cl_ctx, int needed_cap) {
    const int L = MODEL_CONFIG::NUM_HIDDEN_LAYERS;
    const int kv_per_pos = MODEL_CONFIG::NUM_KEY_VALUE_HEADS * MODEL_CONFIG::HEAD_DIM;
    // free any existing buffers
    for (cl_mem b : g_k_cache) if (b) nnopt_pool_free(b);
    for (cl_mem b : g_v_cache) if (b) nnopt_pool_free(b);
    g_k_cache.assign(L, nullptr);
    g_v_cache.assign(L, nullptr);
    g_kv_cache_cap = needed_cap;
    const size_t bytes = (size_t)needed_cap * (size_t)kv_per_pos * sizeof(float);
    cl_int err = CL_SUCCESS;
    for (int i = 0; i < L; ++i) {
        g_k_cache[i] = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
        if (err != CL_SUCCESS || !g_k_cache[i]) { NNOPT_ERROR_FMT("KV cache: alloc k[%d] %d", i, err); return false; }
        g_v_cache[i] = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
        if (err != CL_SUCCESS || !g_v_cache[i]) { NNOPT_ERROR_FMT("KV cache: alloc v[%d] %d", i, err); return false; }
    }
    return true;
}

std::vector<float> model_forward_graph(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    int start_pos)
{
    NNOPT_CHECKPOINT("model_forward_graph entry");
    KernelProfiler::HostTimer _ht_fwd("host_forward_total");
    cl_command_queue queue = cl_ctx.queue();
    int seq_len = (int)input_ids.size();

    // (Re)allocate the KV cache at prefill (start_pos==0). Subsequent decode
    // steps reuse it, appending the new token's K/V at `start_pos`.
    if (start_pos == 0) {
        if (!ensure_kv_cache(cl_ctx, seq_len + KV_CACHE_DECODE_MARGIN)) {
            return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
        }
    }
    if (start_pos + seq_len > g_kv_cache_cap) {
        // Grow to fit (rare: generation longer than the initial margin). This
        // discards cached positions, so only the freshly-reprocessed window is
        // valid — callers should size the margin to the expected generation.
        NNOPT_ERROR_FMT("KV cache overflow (start_pos=%d seq_len=%d cap=%d) — growing",
                        start_pos, seq_len, g_kv_cache_cap);
        if (!ensure_kv_cache(cl_ctx, start_pos + seq_len + KV_CACHE_DECODE_MARGIN)) {
            return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
        }
    }

    cl_int err = CL_SUCCESS;
    cl_mem ids_buf = clCreateBuffer(cl_ctx.context(),
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        seq_len * sizeof(int32_t), (void*)input_ids.data(), &err);
    if (err != CL_SUCCESS || !ids_buf) {
        NNOPT_ERROR_FMT("failed to upload input_ids (err=%d)", err);
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    }
    cl_mem x = ids_buf;


    // ── Pre-loop top-level ops ─────────────────────────────────────────
    // [0] Gemma3TextScaledWordEmbedding — module_path=model.embed_tokens, dump=embedding_wte
    {
        cl_mem _out = Gemma3TextScaledWordEmbedding_forward(
            cl_ctx, weights, queue,
            x, seq_len, /*layer_idx=*/-1, start_pos,
            /*k_cache_inout=*/nullptr, /*v_cache_inout=*/nullptr,
            /*encoder_hidden_states=*/nullptr,
            "model.embed_tokens");
        if (!_out) {
        fflush(stdout);
        fprintf(stderr, "\nNNOPT_FORWARD_HALTED node=%s order=%d : op not implemented yet (returned null). "
                        "The forward ran and dumped layer_dumps/ for every implemented op before it. "
                        "Implement this op next, in graph order, then rebuild.\n", "embedding_wte", (int)(0));
        fflush(stderr);
        if (x) nnopt_pool_free(x);
        exit(0);
        }
        NNOPT_LAYER_CHECK("embedding", queue, _out, /*num_elems=*/seq_len * MODEL_CONFIG::HIDDEN_SIZE);
        if (_out != x) { if (x) nnopt_pool_free(x); x = _out; }
    }

    // [1] Gemma3RotaryEmbedding — module_path=model.rotary_emb, dump=model_rotary_emb
    {
        cl_mem _out = Gemma3RotaryEmbedding_forward(
            cl_ctx, weights, queue,
            x, seq_len, /*layer_idx=*/-1, start_pos,
            /*k_cache_inout=*/nullptr, /*v_cache_inout=*/nullptr,
            /*encoder_hidden_states=*/nullptr,
            "");
        if (!_out) {
        fflush(stdout);
        fprintf(stderr, "\nNNOPT_FORWARD_HALTED node=%s order=%d : op not implemented yet (returned null). "
                        "The forward ran and dumped layer_dumps/ for every implemented op before it. "
                        "Implement this op next, in graph order, then rebuild.\n", "model_rotary_emb", (int)(1));
        fflush(stderr);
        if (x) nnopt_pool_free(x);
        exit(0);
        }
        NNOPT_LAYER_CHECK("model_rotary_emb", queue, _out, /*num_elems=*/seq_len * MODEL_CONFIG::HIDDEN_SIZE);
        if (_out != x) { if (x) nnopt_pool_free(x); x = _out; }
    }

    // [2] Gemma3RotaryEmbedding — module_path=model.rotary_emb_local, dump=model_rotary_emb_local
    {
        cl_mem _out = Gemma3RotaryEmbedding_forward(
            cl_ctx, weights, queue,
            x, seq_len, /*layer_idx=*/-1, start_pos,
            /*k_cache_inout=*/nullptr, /*v_cache_inout=*/nullptr,
            /*encoder_hidden_states=*/nullptr,
            "");
        if (!_out) {
        fflush(stdout);
        fprintf(stderr, "\nNNOPT_FORWARD_HALTED node=%s order=%d : op not implemented yet (returned null). "
                        "The forward ran and dumped layer_dumps/ for every implemented op before it. "
                        "Implement this op next, in graph order, then rebuild.\n", "model_rotary_emb_local", (int)(2));
        fflush(stderr);
        if (x) nnopt_pool_free(x);
        exit(0);
        }
        NNOPT_LAYER_CHECK("model_rotary_emb_local", queue, _out, /*num_elems=*/seq_len * MODEL_CONFIG::HIDDEN_SIZE);
        if (_out != x) { if (x) nnopt_pool_free(x); x = _out; }
    }

    // ── Seed the fp32 residual stream ──────────────────────────────────
    // From here on `x` is a RAW fp32 buffer (the residual). Gemma3's residual
    // grows past fp16 max mid-stack; keeping it fp32 avoids inf→NaN. Each
    // Gemma3DecoderLayer reads fp32 input and returns fp32 output.
    {
        cl_mem _x32 = gemma3_f16_to_f32_run(cl_ctx, queue, x, seq_len * MODEL_CONFIG::HIDDEN_SIZE);
        if (!_x32) {
            NNOPT_ERROR("failed to seed fp32 residual from embedding");
            if (x) nnopt_pool_free(x);
            return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
        }
        if (x) nnopt_pool_free(x);
        x = _x32;
    }

    // ── Per-layer loop (18 × Gemma3DecoderLayer, container=model.layers, single-stack) ──
    // Gemma3DecoderLayer_forward CALLS THE PRIMITIVES INTERNALLY (RMSNorm, Attention,
    // MLP, etc.). Those primitive calls live inside src/ops/Gemma3DecoderLayer.cpp, not here.
    KernelProfiler::HostTimer* _ht_layers = new KernelProfiler::HostTimer("host_decoder_loop");
    for (int layer_idx = 0; layer_idx < MODEL_CONFIG::NUM_HIDDEN_LAYERS; layer_idx++) {
        char _wp[96];
        snprintf(_wp, sizeof(_wp), "model.layers.%d", layer_idx);
        cl_mem _out = Gemma3DecoderLayer_forward(
            cl_ctx, weights, queue,
            x, seq_len, layer_idx, start_pos,
            /*k_cache_inout=*/&g_k_cache[layer_idx],
            /*v_cache_inout=*/&g_v_cache[layer_idx],
            /*encoder_hidden_states=*/nullptr,
            _wp);
        if (!_out) {
            char _hn[96]; snprintf(_hn, sizeof(_hn), "embedding%d", layer_idx);
            fflush(stdout);
            fprintf(stderr, "\nNNOPT_FORWARD_HALTED node=%s order=%d : op not implemented yet (returned null). "
                            "The forward ran and dumped layer_dumps/ for every implemented op before it. "
                            "Implement this op next, in graph order, then rebuild.\n", _hn, (int)(layer_idx));
            fflush(stderr);
            if (x) nnopt_pool_free(x);
            exit(0);
        }
        if (_out != x) { if (x) nnopt_pool_free(x); x = _out; }
        // Per-layer dump (Fix C): the residual `x` is fp32 here, so dump it with
        // the fp32 helper (NNOPT_LAYER_CHECK assumes storage_t / fp16). We make a
        // temporary fp16 copy ONLY for the debug check; the live residual stays
        // fp32. The reference dump for `embedding%d` is fp32; the fp16 copy may
        // saturate at large magnitudes but that is a debug-only artifact and
        // does not affect the live computation.
        { char _dn[96]; snprintf(_dn, sizeof(_dn), "embedding%d", layer_idx);
          NNOPT_LAYER_CHECK_F32(_dn, queue, x, (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE); }
    }
    delete _ht_layers;  // stop the decoder-loop host timer

    // ── Post-loop top-level ops (final norm, lm_head, …) ───────────────
    // [309] Gemma3RMSNorm — module_path=model.norm, dump=final_norm
    {
        cl_mem _out = Gemma3RMSNorm_forward(
            cl_ctx, weights, queue,
            x, seq_len, /*layer_idx=*/-1, start_pos,
            /*k_cache_inout=*/nullptr, /*v_cache_inout=*/nullptr,
            /*encoder_hidden_states=*/nullptr,
            "model.norm");
        if (!_out) {
        fflush(stdout);
        fprintf(stderr, "\nNNOPT_FORWARD_HALTED node=%s order=%d : op not implemented yet (returned null). "
                        "The forward ran and dumped layer_dumps/ for every implemented op before it. "
                        "Implement this op next, in graph order, then rebuild.\n", "final_norm", (int)(309));
        fflush(stderr);
        if (x) nnopt_pool_free(x);
        exit(0);
        }
        NNOPT_LAYER_CHECK("final_norm", queue, _out, /*num_elems=*/seq_len * MODEL_CONFIG::HIDDEN_SIZE);
        if (_out != x) { if (x) nnopt_pool_free(x); x = _out; }
    }

    // ── Slice final-norm output to the LAST token row before lm_head ────────
    // Only the last position's logits are needed (next-token prediction), so
    // run lm_head with M=1. At decode seq_len is already 1; at prefill this
    // avoids a wasted [seq, VOCAB] GEMM AND keeps the lm_head shape fixed at
    // M=1 across prefill+decode so CLBlast compiles that routine just once.
    if (x && seq_len > 1) {
        cl_int serr = CL_SUCCESS;
        const size_t row_bytes = (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t);
        cl_mem last = nnopt_pool_alloc(cl_ctx.context(), row_bytes, &serr);
        if (serr != CL_SUCCESS || !last) {
            NNOPT_ERROR_FMT("lm_head slice: alloc %d", serr);
            nnopt_pool_free(x);
            return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
        }
        const size_t off = (size_t)(seq_len - 1) * row_bytes;
        serr = clEnqueueCopyBuffer(queue, x, last, off, 0, row_bytes, 0, nullptr, nullptr);
        if (serr != CL_SUCCESS) {
            NNOPT_ERROR_FMT("lm_head slice: copy %d", serr);
            nnopt_pool_free(last);
            nnopt_pool_free(x);
            return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
        }
        nnopt_pool_free(x);
        x = last;
    }

    // [310] Linear — module_path=lm_head, dump=lm_head
    {
        cl_mem _out = Linear_forward(
            cl_ctx, weights, queue,
            x, /*seq_len=*/1, /*layer_idx=*/-1, start_pos,
            /*k_cache_inout=*/nullptr, /*v_cache_inout=*/nullptr,
            /*encoder_hidden_states=*/nullptr,
            "lm_head");
        if (!_out) {
        fflush(stdout);
        fprintf(stderr, "\nNNOPT_FORWARD_HALTED node=%s order=%d : op not implemented yet (returned null). "
                        "The forward ran and dumped layer_dumps/ for every implemented op before it. "
                        "Implement this op next, in graph order, then rebuild.\n", "lm_head", (int)(310));
        fflush(stderr);
        if (x) nnopt_pool_free(x);
        exit(0);
        }
        NNOPT_LAYER_CHECK("lm_head", queue, _out, /*num_elems=*/seq_len * MODEL_CONFIG::HIDDEN_SIZE);
        if (_out != x) { if (x) nnopt_pool_free(x); x = _out; }
    }

    // ── Logits readout — agent: tune if final op's output isn't [T, V] ──
    const int vocab = MODEL_CONFIG::VOCAB_SIZE;
    // Greedy fast-path: argmax on GPU, read back only the token id. Skips the
    // full 262144 readback + host convert + host argmax (GPU-idle host work).
    if (g_greedy_argmax && x) {
        KernelProfiler::HostTimer _ht_rb("host_final_readback");
        g_argmax_token = nnopt_gpu_argmax(cl_ctx, queue, x, vocab);
        // NNOPT_ARGMAX_VERIFY=1: cross-check the GPU argmax against a full host
        // readback + host argmax and log any mismatch (proves the GPU kernel is
        // correct vs the old path; leave OFF in production — it negates the win).
        if (const char* vv = std::getenv("NNOPT_ARGMAX_VERIFY"); vv && vv[0] == '1') {
            std::vector<nnopt_storage_t> hs(vocab);
            clEnqueueReadBuffer(queue, x, CL_TRUE, 0, vocab * sizeof(nnopt_storage_t), hs.data(), 0, nullptr, nullptr);
            int hi = 0; float hbest = -1e30f;
            for (int i = 0; i < vocab; i++) {
#ifdef NNOPT_USE_FP16
                float fv = nnopt_f16_to_f32(static_cast<uint16_t>(hs[i]));
#else
                float fv = static_cast<float>(hs[i]);
#endif
                if (fv > hbest) { hbest = fv; hi = i; }
            }
            if (hi != g_argmax_token)
                fprintf(stderr, "ARGMAX_VERIFY MISMATCH: gpu=%d host=%d (host_logit=%g)\n", g_argmax_token, hi, hbest);
            else
                fprintf(stderr, "ARGMAX_VERIFY ok: %d\n", hi);
        }
        nnopt_pool_free(x);
        if (g_argmax_token >= 0) return std::vector<float>();   // sentinel: use g_argmax_token
        // On failure, fall through is not possible (x freed); return empty and
        // let the caller's sampler path handle the -1 (it won't, but this only
        // triggers on a GPU error already logged above).
        return std::vector<float>();
    }
    std::vector<float> logits(vocab, 0.0f);
    if (x) {
        KernelProfiler::HostTimer _ht_rb("host_final_readback");
        std::vector<nnopt_storage_t> host_storage(vocab);
        // lm_head was sliced to a single row (M=1), so the next-token logits
        // are at offset 0 regardless of the original sequence length.
        size_t row_off = 0;
        clEnqueueReadBuffer(queue, x, CL_TRUE, row_off,
            vocab * sizeof(nnopt_storage_t),
            host_storage.data(), 0, nullptr, nullptr);
        // Decode storage → float. Under fp16, nnopt_storage_t is cl_half (raw
        // uint16 bits): a bare (float) cast reinterprets the BIT PATTERN as an
        // integer, not the half-float value. Always decode via the IEEE-754 codec.
        for (int i = 0; i < vocab; i++)
#ifdef NNOPT_USE_FP16
            logits[i] = nnopt_f16_to_f32(static_cast<uint16_t>(host_storage[i]));
#else
            logits[i] = static_cast<float>(host_storage[i]);
#endif
        nnopt_pool_free(x);
    }

    return logits;
}
