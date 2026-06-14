// Reference: model_info/transformers_src/modeling_musicgen.py:179-290 MusicgenAttention.forward
// MusicgenSdpaAttention.cpp — shared implementation for ALL 48 MusicgenSdpaAttention node(s).
//
// Implements the eager_attention_forward path used by MusicGen when
// config._attn_implementation selects eager (we treat all as eager for now):
//   q = q_proj(x) -> [B, T, H] reshape [B, nH, T, d]
//   k,v from (x or encoder_hidden_states) similarly
//   scores = (q @ k^T) * scale (+mask) ; softmax ; out = scores @ v
//   out = out.transpose(1,2).reshape(B,T,H) ; out_proj(out)
//
// NOTE: This implementation ignores caching/past_key_values; the scaffold's
// generate loop is currently blocked earlier by missing music pipeline pieces.
//
// Sibling nodes (order  dump_name  weight_prefix):
//     21  self_attn_0                               weight_prefix=<none>
//     23  encoder_attn_0                            weight_prefix=<none>
//     30  self_attn_1                               weight_prefix=<none>
//     32  encoder_attn_1                            weight_prefix=<none>
//     39  self_attn_2                               weight_prefix=<none>
//     41  encoder_attn_2                            weight_prefix=<none>
//     48  self_attn_3                               weight_prefix=<none>
//     50  encoder_attn_3                            weight_prefix=<none>
//     57  self_attn_4                               weight_prefix=<none>
//     59  encoder_attn_4                            weight_prefix=<none>
//     66  self_attn_5                               weight_prefix=<none>
//     68  encoder_attn_5                            weight_prefix=<none>
//   … (+36 more siblings — see .nnport/scaffold_manifest.json)
//
// Representative shape (from sibling 0):
//   input:  [1, 1, 1024]
//   output: [1, 1, 1024]
//
// Primary reference dump for cosine validation:
//   reference/layers/self_attn_0_output.bin
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
//    `// Reference: model_info/transformers_src/<file>.py:<lines> MusicgenSdpaAttention.forward`
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

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"

#include <CL/cl.h>
#include <string>
#include <vector>

extern "C" {
cl_mem MusicgenSdpaAttention_forward(
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

    if (!input) {
        NNOPT_ERROR("MusicgenSdpaAttention_forward: input is null");
        return nullptr;
    }

    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();

    const bool is_cross = (encoder_hidden_states != nullptr);
    cl_mem kv_src = is_cross ? encoder_hidden_states : input;

    // Weight keys (MusicgenAttention)
    const std::string k_w = wp + ".k_proj.weight";
    const std::string q_w = wp + ".q_proj.weight";
    const std::string v_w = wp + ".v_proj.weight";
    const std::string o_w = wp + ".out_proj.weight";

    if (!weights.has_tensor(q_w) || !weights.has_tensor(k_w) || !weights.has_tensor(v_w) || !weights.has_tensor(o_w)) {
        NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: missing proj weight(s) at wp=%s", wp.c_str());
        return nullptr;
    }

    // MusicGen decoder attention dims come from the decoder config, not the T5 text encoder.
    // Using MODEL_CONFIG::HIDDEN_SIZE here is wrong (it refers to the text encoder embedding dim=768).
    // IMPORTANT: MusicGen is an ENCODER-DECODER. The decoder runs with batch size
    // B*num_codebooks, where num_codebooks=4. Our graph currently calls attention
    // with `seq_len` representing the flattened token axis (B*codebooks*Time).
    // The attention math must run over the real time axis only.
    //
    // Reference: modeling_musicgen.py shows input_ids shape (batch*num_codebooks, seq)
    // and attention operates over `Time`.
    const int hidden = MODEL_CONFIG::DECODER_HIDDEN_SIZE;
    const int num_heads = MODEL_CONFIG::NUM_ATTENTION_HEADS;
    const int head_dim = hidden / num_heads;
    const float scale = 1.0f / std::sqrt((float)head_dim);

    if (num_heads <= 0 || head_dim <= 0 || (num_heads * head_dim) != hidden) {
        NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: bad config hidden=%d num_heads=%d head_dim=%d", hidden, num_heads, head_dim);
        return nullptr;
    }

    // Self-attention must support both prefill (seq_len>1) and decode (seq_len==1).
    // The previous hard check caused runtime abort when prefill ran with a real prompt.
    // Reference: modeling_musicgen.py:179-290 MusicgenAttention.forward (query_states uses input_shape=hidden_states.shape[:-1]).

    const int Mq = seq_len;
    // Cross-attention K/V come from the T5 encoder states; their sequence
    // length is the PROMPT length, not the decoder seq_len. backbone.cpp
    // publishes it after Model::encode_text (g_musicgen_enc_len).
    extern int g_musicgen_enc_len;
    const int Mk = is_cross ? g_musicgen_enc_len : seq_len;
    if (is_cross && Mk <= 0) {
        NNOPT_ERROR("MusicgenSdpaAttention_forward: cross-attn requested but encoder length is 0 (encode_text not run?)");
        return nullptr;
    }

    cl_int err = CL_SUCCESS;
    // ── Scratch-buffer pool (opt #2, BENCHMARKS.md) ─────────────────────
    // This op runs 48×/step (24 layers × self+cross) × 2 CFG branches; the
    // original code clCreateBuffer'd 7 buffers per call (~7200 allocs per
    // 150-token clip). Adreno buffer creation is ms-scale → pool by slot,
    // grow-only. Single-threaded dispatch; lifetime = process.
    struct Pool { cl_mem buf = nullptr; size_t cap = 0; };
    static Pool pool[6];
    auto pooled = [&](int slot, size_t bytes) -> cl_mem {
        if (pool[slot].cap < bytes) {
            if (pool[slot].buf) clReleaseMemObject(pool[slot].buf);
            cl_int perr = CL_SUCCESS;
            pool[slot].buf = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &perr);
            pool[slot].cap = (perr == CL_SUCCESS && pool[slot].buf) ? bytes : 0;
            if (perr != CL_SUCCESS) pool[slot].buf = nullptr;
        }
        return pool[slot].buf;
    };
    cl_mem q_tok = pooled(0, (size_t)Mq * hidden * sizeof(nnopt_storage_t));
    cl_mem k_tok = pooled(1, (size_t)Mk * hidden * sizeof(nnopt_storage_t));
    cl_mem v_tok = pooled(2, (size_t)Mk * hidden * sizeof(nnopt_storage_t));
    cl_mem scores = nullptr;
    cl_mem out_heads = nullptr;
    cl_mem out_tok = nullptr;
    cl_mem out = nullptr;

    auto cleanup = [&]() -> cl_mem {
        // pooled buffers (q/k/v, scores, out_heads, out_tok) persist; only the
        // returned `out` is caller-owned.
        if (out) { clReleaseMemObject(out); out = nullptr; }
        return nullptr;
    };

    if (!q_tok || !k_tok || !v_tok) { NNOPT_ERROR("MusicgenSdpaAttention_forward: pool alloc failed (q/k/v)"); return cleanup(); }

    // Projections
    if (!pytorch_linear(queue, Mq, hidden, hidden, input, weights.get_buffer(q_w), q_tok)) {
        NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: q_proj failed wp=%s", wp.c_str());
        return cleanup();
    }
    if (!pytorch_linear(queue, Mk, hidden, hidden, kv_src, weights.get_buffer(k_w), k_tok)) {
        NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: k_proj failed wp=%s", wp.c_str());
        return cleanup();
    }
    if (!pytorch_linear(queue, Mk, hidden, hidden, kv_src, weights.get_buffer(v_w), v_tok)) {
        NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: v_proj failed wp=%s", wp.c_str());
        return cleanup();
    }

    // ── Persistent KV cache (self-attention decode) ────────────────────
    // modeling_musicgen.py MusicgenAttention.forward: past_key_value carries
    // K/V for positions [0, start_pos); this step appends rows at start_pos
    // and attends over ALL cached positions. Without this, every decode step
    // attends only to itself → outputs barely change per step → the sampler
    // collapses to one token (the port's longest-lived failure signature).
    // Cache layout matches k_tok/v_tok: token-major [pos, hidden], fp16/fp32
    // per nnopt_storage_t. Caller (backbone.cpp) owns one cache pair per layer.
    cl_mem k_attn = k_tok;
    cl_mem v_attn = v_tok;
    int Mk_eff = Mk;
    if (!is_cross && k_cache_inout && v_cache_inout) {
        constexpr int kMaxCachePos = 256;   // generation cap; 256*1024*2B = 512KB/buffer
        if (start_pos + seq_len > kMaxCachePos) {
            NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: start_pos+seq_len %d exceeds KV cache cap %d",
                            start_pos + seq_len, kMaxCachePos);
            return cleanup();
        }
        const size_t row_bytes = (size_t)hidden * sizeof(nnopt_storage_t);
        if (!*k_cache_inout) {
            *k_cache_inout = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                            (size_t)kMaxCachePos * row_bytes, nullptr, &err);
            if (err != CL_SUCCESS || !*k_cache_inout) { NNOPT_ERROR_FMT("k_cache alloc %d", (int)err); return cleanup(); }
        }
        if (!*v_cache_inout) {
            *v_cache_inout = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                            (size_t)kMaxCachePos * row_bytes, nullptr, &err);
            if (err != CL_SUCCESS || !*v_cache_inout) { NNOPT_ERROR_FMT("v_cache alloc %d", (int)err); return cleanup(); }
        }
        err = clEnqueueCopyBuffer(queue, k_tok, *k_cache_inout, 0,
                                  (size_t)start_pos * row_bytes, (size_t)seq_len * row_bytes,
                                  0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("k_cache append %d", (int)err); return cleanup(); }
        err = clEnqueueCopyBuffer(queue, v_tok, *v_cache_inout, 0,
                                  (size_t)start_pos * row_bytes, (size_t)seq_len * row_bytes,
                                  0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("v_cache append %d", (int)err); return cleanup(); }
        k_attn = *k_cache_inout;
        v_attn = *v_cache_inout;
        Mk_eff = start_pos + seq_len;
    }

    // scores buffer in fp32 (pooled slot 3)
    scores = pooled(3, (size_t)num_heads * (size_t)Mq * (size_t)Mk_eff * sizeof(float));
    if (!scores) { NNOPT_ERROR("MusicgenSdpaAttention_forward: pool alloc failed (scores)"); return cleanup(); }

    static cl_program attn_prog = nullptr;
    static cl_kernel k_scores = nullptr;
    static cl_kernel k_softmax = nullptr;
    static cl_kernel k_out = nullptr;
    static cl_kernel k_h2t = nullptr;
    static cl_kernel k_fused = nullptr;
    if (!attn_prog) {
        attn_prog = cl_ctx.build_program_from_file("kernels/attention.cl"); // PROGRAM-INIT-OK
        if (!attn_prog) { NNOPT_ERROR("MusicgenSdpaAttention_forward: build_program_from_file(attention.cl) failed"); return cleanup(); }
        k_scores = clCreateKernel(attn_prog, "gqa_attn_scores", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: clCreateKernel(gqa_attn_scores) %d", (int)err); return cleanup(); }
        k_softmax = clCreateKernel(attn_prog, "gqa_softmax", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: clCreateKernel(gqa_softmax) %d", (int)err); return cleanup(); }
        k_out = clCreateKernel(attn_prog, "gqa_attn_out", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: clCreateKernel(gqa_attn_out) %d", (int)err); return cleanup(); }
        k_h2t = clCreateKernel(attn_prog, "heads_to_tokens", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: clCreateKernel(heads_to_tokens) %d", (int)err); return cleanup(); }
        k_fused = clCreateKernel(attn_prog, "fused_decode_attention", &err);  // seq_q==1 fast path
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: clCreateKernel(fused_decode_attention) %d", (int)err); k_fused = nullptr; }
    }

    const int num_kv_heads = num_heads; // MusicGen uses MHA (no GQA)
    const int causal = is_cross ? 0 : 1;

    // ── Fused single-query path (decode, Mq==1) ────────────────────────────
    // Collapses scores→softmax→out→heads_to_tokens (4 dispatches) into ONE.
    // Mathematically identical (fp32 accum, same scale). For the single query
    // the causal mask is a no-op (the query is the latest position and attends
    // all cached keys), so it's correct for both self-attn (Mk_eff=start_pos+1)
    // and cross-attn (Mk_eff=enc_len). head_dim must be 64 (matches kernel WG).
    out_tok = pooled(5, (size_t)Mq * (size_t)hidden * sizeof(nnopt_storage_t));
    if (!out_tok) { NNOPT_ERROR("MusicgenSdpaAttention_forward: pool alloc failed (out_tok)"); return cleanup(); }
    if (Mq == 1 && k_fused && head_dim == 64 && Mk_eff <= 256) {
        int arg = 0;
        if (!set_arg_checked(k_fused, arg++, sizeof(cl_mem), &q_tok, "q")) return cleanup();
        if (!set_arg_checked(k_fused, arg++, sizeof(cl_mem), &k_attn, "k")) return cleanup();
        if (!set_arg_checked(k_fused, arg++, sizeof(cl_mem), &v_attn, "v")) return cleanup();
        if (!set_arg_checked(k_fused, arg++, sizeof(cl_mem), &out_tok, "out")) return cleanup();
        if (!set_arg_checked(k_fused, arg++, sizeof(int), &Mk_eff, "seq_k")) return cleanup();
        if (!set_arg_checked(k_fused, arg++, sizeof(int), &num_heads, "num_heads")) return cleanup();
        if (!set_arg_checked(k_fused, arg++, sizeof(int), &head_dim, "head_dim")) return cleanup();
        if (!set_arg_checked(k_fused, arg++, sizeof(float), &scale, "scale")) return cleanup();
        const size_t gws[1] = {(size_t)num_heads * 64};
        const size_t lws[1] = {64};
        err = clEnqueueNDRangeKernel(queue, k_fused, 1, nullptr, gws, lws, 0, nullptr,
                                     KernelProfiler::event_for(("attn_fused:" + wp).c_str()));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: fused dispatch %d", (int)err); return cleanup(); }
        // Skip the unfused chain — out_tok is ready for out_proj.
        goto do_out_proj;
    }

    // gqa_attn_scores
    {
        int arg = 0;
        if (!set_arg_checked(k_scores, arg++, sizeof(cl_mem), &q_tok, "q")) return cleanup();
        if (!set_arg_checked(k_scores, arg++, sizeof(cl_mem), &k_attn, "k")) return cleanup();
        if (!set_arg_checked(k_scores, arg++, sizeof(cl_mem), &scores, "scores")) return cleanup();
        if (!set_arg_checked(k_scores, arg++, sizeof(int), &Mq, "seq_q")) return cleanup();
        if (!set_arg_checked(k_scores, arg++, sizeof(int), &Mk_eff, "seq_k")) return cleanup();
        if (!set_arg_checked(k_scores, arg++, sizeof(int), &num_heads, "num_q_heads")) return cleanup();
        if (!set_arg_checked(k_scores, arg++, sizeof(int), &num_kv_heads, "num_kv_heads")) return cleanup();
        if (!set_arg_checked(k_scores, arg++, sizeof(int), &head_dim, "head_dim")) return cleanup();
        if (!set_arg_checked(k_scores, arg++, sizeof(float), &scale, "scale")) return cleanup();
        if (!set_arg_checked(k_scores, arg++, sizeof(int), &causal, "causal")) return cleanup();

        const size_t gws[3] = {(size_t)Mk_eff, (size_t)Mq, (size_t)num_heads};
        err = clEnqueueNDRangeKernel(queue, k_scores, 3, nullptr, gws, nullptr, 0, nullptr,
                                     KernelProfiler::event_for(("attn_scores:" + wp).c_str()));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: attn_scores dispatch %d", (int)err); return cleanup(); }
    }

    // softmax
    {
        int arg = 0;
        if (!set_arg_checked(k_softmax, arg++, sizeof(cl_mem), &scores, "scores")) return cleanup();
        if (!set_arg_checked(k_softmax, arg++, sizeof(int), &Mq, "seq_q")) return cleanup();
        if (!set_arg_checked(k_softmax, arg++, sizeof(int), &Mk_eff, "seq_k")) return cleanup();
        if (!set_arg_checked(k_softmax, arg++, sizeof(int), &num_heads, "num_q_heads")) return cleanup();

        const size_t gws[2] = {(size_t)Mq, (size_t)num_heads};
        err = clEnqueueNDRangeKernel(queue, k_softmax, 2, nullptr, gws, nullptr, 0, nullptr,
                                     KernelProfiler::event_for(("attn_softmax:" + wp).c_str()));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: softmax dispatch %d", (int)err); return cleanup(); }
    }

    // out_heads (pooled slot 4)
    out_heads = pooled(4, (size_t)num_heads * (size_t)Mq * (size_t)head_dim * sizeof(nnopt_storage_t));
    if (!out_heads) { NNOPT_ERROR("MusicgenSdpaAttention_forward: pool alloc failed (out_heads)"); return cleanup(); }

    {
        int arg = 0;
        if (!set_arg_checked(k_out, arg++, sizeof(cl_mem), &scores, "scores")) return cleanup();
        if (!set_arg_checked(k_out, arg++, sizeof(cl_mem), &v_attn, "v")) return cleanup();
        if (!set_arg_checked(k_out, arg++, sizeof(cl_mem), &out_heads, "out_heads")) return cleanup();
        if (!set_arg_checked(k_out, arg++, sizeof(int), &Mq, "seq_q")) return cleanup();
        if (!set_arg_checked(k_out, arg++, sizeof(int), &Mk_eff, "seq_k")) return cleanup();
        if (!set_arg_checked(k_out, arg++, sizeof(int), &num_heads, "num_q_heads")) return cleanup();
        if (!set_arg_checked(k_out, arg++, sizeof(int), &num_kv_heads, "num_kv_heads")) return cleanup();
        if (!set_arg_checked(k_out, arg++, sizeof(int), &head_dim, "head_dim")) return cleanup();

        const size_t gws[3] = {(size_t)head_dim, (size_t)Mq, (size_t)num_heads};
        err = clEnqueueNDRangeKernel(queue, k_out, 3, nullptr, gws, nullptr, 0, nullptr,
                                     KernelProfiler::event_for(("attn_out:" + wp).c_str()));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: attn_out dispatch %d", (int)err); return cleanup(); }
    }

    // heads_to_tokens (pooled slot 5)
    out_tok = pooled(5, (size_t)Mq * (size_t)hidden * sizeof(nnopt_storage_t));
    if (!out_tok) { NNOPT_ERROR("MusicgenSdpaAttention_forward: pool alloc failed (out_tok)"); return cleanup(); }

    {
        int arg = 0;
        if (!set_arg_checked(k_h2t, arg++, sizeof(cl_mem), &out_heads, "in_heads")) return cleanup();
        if (!set_arg_checked(k_h2t, arg++, sizeof(cl_mem), &out_tok, "out_tokens")) return cleanup();
        if (!set_arg_checked(k_h2t, arg++, sizeof(int), &Mq, "seq_len")) return cleanup();
        if (!set_arg_checked(k_h2t, arg++, sizeof(int), &num_heads, "num_heads")) return cleanup();
        if (!set_arg_checked(k_h2t, arg++, sizeof(int), &head_dim, "head_dim")) return cleanup();

        const size_t gws[1] = {(size_t)Mq * (size_t)hidden};
        err = clEnqueueNDRangeKernel(queue, k_h2t, 1, nullptr, gws, nullptr, 0, nullptr,
                                     KernelProfiler::event_for(("attn_h2t:" + wp).c_str()));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: heads_to_tokens dispatch %d", (int)err); return cleanup(); }
    }

    // out_proj
do_out_proj:
    out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)Mq * (size_t)hidden * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: clCreateBuffer(out) %d", (int)err); return cleanup(); }
    if (!pytorch_linear(queue, Mq, hidden, hidden, out_tok, weights.get_buffer(o_w), out)) {
        NNOPT_ERROR_FMT("MusicgenSdpaAttention_forward: out_proj failed wp=%s", wp.c_str());
        return cleanup();
    }

    // intermediates are pooled — nothing to free (out is caller-owned)

    NNOPT_DEBUG_SYNC(queue);
    return out;
}
}
