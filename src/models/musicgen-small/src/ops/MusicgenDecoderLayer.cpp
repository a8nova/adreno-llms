// Reference: model_info/transformers_src/modeling_musicgen.py:250-344 MusicgenDecoderLayer.forward
// MusicgenDecoderLayer.cpp — shared implementation for ALL 25 MusicgenDecoderLayer node(s).
//
// You write the MusicgenDecoderLayer forward() math ONCE here. backbone.cpp calls
// this function once per traced node of class MusicgenDecoderLayer, passing the
// appropriate weight_prefix string for each call — your implementation
// uses that prefix to load weights, so the same C++ math correctly
// handles every sibling instance.
//
// Sibling nodes (order  dump_name  weight_prefix):
//     19  decoder_model_decoder_frontend_out        weight_prefix=<none>
//     28  layer_0                                   weight_prefix=<none>
//     37  layer_1                                   weight_prefix=<none>
//     46  layer_2                                   weight_prefix=<none>
//     55  layer_3                                   weight_prefix=<none>
//     64  layer_4                                   weight_prefix=<none>
//     73  layer_5                                   weight_prefix=<none>
//     82  layer_6                                   weight_prefix=<none>
//     91  layer_7                                   weight_prefix=<none>
//    100  layer_8                                   weight_prefix=<none>
//    109  layer_9                                   weight_prefix=<none>
//    118  layer_10                                  weight_prefix=<none>
//   … (+13 more siblings — see .nnport/scaffold_manifest.json)
//
// Representative shape (from sibling 0):
//   input:  [1, 1, 1024]
//   output: [1, 1, 1024]
//
// Primary reference dump for cosine validation:
//   reference/layers/decoder_model_decoder_frontend_out_output.bin
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
//    `// Reference: model_info/transformers_src/<file>.py:<lines> MusicgenDecoderLayer.forward`
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

#include <string>
#include <cstdio>

extern "C" {
cl_mem LayerNorm_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);

cl_mem MusicgenSdpaAttention_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);

cl_mem Linear_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);

cl_mem GELUActivation_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);

static inline void wp_child(char* dst, size_t dst_sz, const char* parent, const char* child) {
    if (!parent || parent[0] == 0) {
        std::snprintf(dst, dst_sz, "%s", child);
    } else {
        std::snprintf(dst, dst_sz, "%s.%s", parent, child);
    }
}

cl_mem MusicgenDecoderLayer_forward(
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
    if (!input) {
        NNOPT_ERROR("MusicgenDecoderLayer_forward: input is null");
        return nullptr;
    }

    const char* wp = (weight_prefix ? weight_prefix : "");

    // Reference: MusicgenDecoderLayer.forward
    // residual = hidden_states
    // hidden_states = self_attn_layer_norm(hidden_states)
    // hidden_states = self_attn(hidden_states, past_key_values, attention_mask)
    // hidden_states = residual + dropout(hidden_states)
    // if encoder_hidden_states: cross-attn block
    // final FFN: ln -> fc1 -> act -> fc2 -> residual

    cl_mem x = input;
    cl_mem residual = input;

    // self_attn_layer_norm
    // Reference: modeling_musicgen.py MusicgenDecoderLayer.forward:
    //   residual = hidden_states
    //   hidden_states = self_attn_layer_norm(hidden_states)

    // Input dump for LN (lets us disambiguate wiring vs LN math).
    char dump_self_ln_in[64];
    std::snprintf(dump_self_ln_in, sizeof(dump_self_ln_in), "self_attn_layer_norm_%d", layer_idx);
    NNOPT_LAYER_CHECK_INPUT(dump_self_ln_in, queue, x,
                            (size_t)seq_len * (size_t)MODEL_CONFIG::DECODER_HIDDEN_SIZE);

    char wp_self_ln[256];
    wp_child(wp_self_ln, sizeof(wp_self_ln), wp, "self_attn_layer_norm");
    cl_mem normed = LayerNorm_forward(
        cl_ctx, weights, queue,
        x, seq_len, layer_idx, start_pos,
        nullptr, nullptr,
        nullptr,
        wp_self_ln);

    // Dump at the decoder-layer boundary so SxS can localize divergence.
    // Name must match dump_spec.json (e.g. self_attn_layer_norm_0).
    char dump_self_ln[64];
    std::snprintf(dump_self_ln, sizeof(dump_self_ln), "self_attn_layer_norm_%d", layer_idx);
    // Contract shape: [seq_len, DECODER_HIDDEN_SIZE]
    NNOPT_LAYER_CHECK(dump_self_ln, queue, normed,
                      (size_t)seq_len * (size_t)MODEL_CONFIG::DECODER_HIDDEN_SIZE);
    if (!normed) {
        NNOPT_ERROR("MusicgenDecoderLayer_forward: self_attn_layer_norm returned null");
        return nullptr;
    }

    // self_attn
    char wp_self_attn[256];
    wp_child(wp_self_attn, sizeof(wp_self_attn), wp, "self_attn");
    cl_mem attn_out = MusicgenSdpaAttention_forward(
        cl_ctx, weights, queue,
        normed, seq_len, layer_idx, start_pos,
        k_cache_inout, v_cache_inout,
        /*encoder_hidden_states=*/nullptr,
        wp_self_attn);

    // Dump attention output (before residual add) with the canonical name.
    char dump_self_attn[64];
    std::snprintf(dump_self_attn, sizeof(dump_self_attn), "self_attn_%d", layer_idx);
    // Contract shape: [seq_len, DECODER_HIDDEN_SIZE]
    NNOPT_LAYER_CHECK(dump_self_attn, queue, attn_out,
                      (size_t)seq_len * (size_t)MODEL_CONFIG::DECODER_HIDDEN_SIZE);
    clReleaseMemObject(normed);
    if (!attn_out) {
        NNOPT_ERROR("MusicgenDecoderLayer_forward: self_attn returned null");
        return nullptr;
    }

    // residual add (in-place): residual += attn_out
    static cl_program utils_program = nullptr;
    if (!utils_program) {
        utils_program = cl_ctx.build_program_from_file("kernels/utils.cl"); // PROGRAM-INIT-OK
        if (!utils_program) {
            NNOPT_ERROR("MusicgenDecoderLayer_forward: failed to build utils program");
            clReleaseMemObject(attn_out);
            return nullptr;
        }
    }
    if (!element_add_inplace(queue, utils_program, residual, attn_out,
                             (size_t)seq_len * (size_t)MODEL_CONFIG::DECODER_HIDDEN_SIZE)) {
        clReleaseMemObject(attn_out);
        NNOPT_ERROR("MusicgenDecoderLayer_forward: residual add after self_attn failed");
        return nullptr;
    }
    clReleaseMemObject(attn_out);
    clRetainMemObject(residual);
    x = residual;

    // Cross-attention block (optional)
    if (encoder_hidden_states) {
        residual = x;

        char wp_enc_ln[256];
        wp_child(wp_enc_ln, sizeof(wp_enc_ln), wp, "encoder_attn_layer_norm");
        cl_mem enc_normed = LayerNorm_forward(
            cl_ctx, weights, queue,
            x, seq_len, layer_idx, start_pos,
            nullptr, nullptr,
            nullptr,
            wp_enc_ln);
        char dump_enc_ln[64];
        std::snprintf(dump_enc_ln, sizeof(dump_enc_ln), "encoder_attn_layer_norm_%d", layer_idx);
        // Contract shape: [seq_len, DECODER_HIDDEN_SIZE]
        NNOPT_LAYER_CHECK(dump_enc_ln, queue, enc_normed,
                          (size_t)seq_len * (size_t)MODEL_CONFIG::DECODER_HIDDEN_SIZE);
        if (!enc_normed) {
            clReleaseMemObject(x);
            NNOPT_ERROR("MusicgenDecoderLayer_forward: encoder_attn_layer_norm returned null");
            return nullptr;
        }

        char wp_enc_attn[256];
        wp_child(wp_enc_attn, sizeof(wp_enc_attn), wp, "encoder_attn");
        cl_mem enc_attn_out = MusicgenSdpaAttention_forward(
            cl_ctx, weights, queue,
            enc_normed, seq_len, layer_idx, start_pos,
            k_cache_inout, v_cache_inout,
            /*encoder_hidden_states=*/encoder_hidden_states,
            wp_enc_attn);
        char dump_enc_attn[64];
        std::snprintf(dump_enc_attn, sizeof(dump_enc_attn), "encoder_attn_%d", layer_idx);
        // Contract shape: [seq_len, DECODER_HIDDEN_SIZE]
        NNOPT_LAYER_CHECK(dump_enc_attn, queue, enc_attn_out,
                          (size_t)seq_len * (size_t)MODEL_CONFIG::DECODER_HIDDEN_SIZE);
        clReleaseMemObject(enc_normed);
        if (!enc_attn_out) {
            clReleaseMemObject(x);
            NNOPT_ERROR("MusicgenDecoderLayer_forward: encoder_attn returned null");
            return nullptr;
        }

        // residual add (in-place): residual += enc_attn_out
        if (!element_add_inplace(queue, utils_program, residual, enc_attn_out,
                                 (size_t)seq_len * (size_t)MODEL_CONFIG::DECODER_HIDDEN_SIZE)) {
            clReleaseMemObject(enc_attn_out);
            clReleaseMemObject(x);
            NNOPT_ERROR("MusicgenDecoderLayer_forward: residual add after encoder_attn failed");
            return nullptr;
        }
        clReleaseMemObject(enc_attn_out);
        clReleaseMemObject(x);
        clRetainMemObject(residual);
        x = residual;
    }

    // FFN
    residual = x;
    char wp_final_ln[256];
    wp_child(wp_final_ln, sizeof(wp_final_ln), wp, "final_layer_norm");
    cl_mem ffn_in = LayerNorm_forward(
        cl_ctx, weights, queue,
        x, seq_len, layer_idx, start_pos,
        nullptr, nullptr,
        nullptr,
        wp_final_ln);
    clReleaseMemObject(x);
    if (!ffn_in) {
        NNOPT_ERROR("MusicgenDecoderLayer_forward: final_layer_norm returned null");
        return nullptr;
    }

    char wp_fc1[256];
    wp_child(wp_fc1, sizeof(wp_fc1), wp, "fc1");
    cl_mem fc1_out = Linear_forward(
        cl_ctx, weights, queue,
        ffn_in, seq_len, layer_idx, start_pos,
        nullptr, nullptr,
        nullptr,
        wp_fc1);
    char dump_fc1[64];
    std::snprintf(dump_fc1, sizeof(dump_fc1), "fc1_%d", layer_idx);
    // Contract shape: [seq_len, INTERMEDIATE_SIZE]
    NNOPT_LAYER_CHECK(dump_fc1, queue, fc1_out,
                      (size_t)seq_len * (size_t)MODEL_CONFIG::INTERMEDIATE_SIZE);
    clReleaseMemObject(ffn_in);
    if (!fc1_out) {
        NNOPT_ERROR("MusicgenDecoderLayer_forward: fc1 returned null");
        return nullptr;
    }

    cl_mem act_out = GELUActivation_forward(
        cl_ctx, weights, queue,
        fc1_out, seq_len, layer_idx, start_pos,
        nullptr, nullptr,
        nullptr,
        "");
    clReleaseMemObject(fc1_out);
    if (!act_out) {
        NNOPT_ERROR("MusicgenDecoderLayer_forward: gelu returned null");
        return nullptr;
    }

    char wp_fc2[256];
    wp_child(wp_fc2, sizeof(wp_fc2), wp, "fc2");
    cl_mem fc2_out = Linear_forward(
        cl_ctx, weights, queue,
        act_out, seq_len, layer_idx, start_pos,
        nullptr, nullptr,
        nullptr,
        wp_fc2);
    char dump_fc2[64];
    std::snprintf(dump_fc2, sizeof(dump_fc2), "fc2_%d", layer_idx);
    // Contract shape: [seq_len, DECODER_HIDDEN_SIZE]
    NNOPT_LAYER_CHECK(dump_fc2, queue, fc2_out,
                      (size_t)seq_len * (size_t)MODEL_CONFIG::DECODER_HIDDEN_SIZE);
    clReleaseMemObject(act_out);
    if (!fc2_out) {
        NNOPT_ERROR("MusicgenDecoderLayer_forward: fc2 returned null");
        return nullptr;
    }

    // residual add (in-place): residual += fc2_out
    if (!element_add_inplace(queue, utils_program, residual, fc2_out,
                             (size_t)seq_len * (size_t)MODEL_CONFIG::DECODER_HIDDEN_SIZE)) {
        clReleaseMemObject(fc2_out);
        NNOPT_ERROR("MusicgenDecoderLayer_forward: residual add after FFN failed");
        return nullptr;
    }
    clReleaseMemObject(fc2_out);
    clRetainMemObject(residual);
    return residual;
}

// ─── M=2 CFG-batched decoder layer (BENCHMARKS Stage 1) ──────────────────
// Processes BOTH CFG rows (row0 = conditioned, row1 = unconditional) in ONE
// pass for every row-wise op (LayerNorm, fc1, GELU, fc2, residual adds) so the
// host pays HALF the dispatch overhead for the non-attention work. Attention
// is intrinsically per-row (each row has its own KV bank + its own encoder
// states: cond=g_enc_states/bank0, uncond=g_enc_zero/bank32), so it runs once
// per row — the result of each per-row attention is copied back into the
// [2,hidden] activation buffer at that row's offset, then a single M=2
// residual add folds it in.
//
// Math is BYTE-IDENTICAL to running MusicgenDecoderLayer_forward twice (once
// per row): same LN kernel (row-parallel), same attention kernels/KV banks,
// same FFN GEMMs (CLBlast M=2 == two M=1 rows stacked). The ONLY change is
// dispatch BATCHING of the row-wise ops, which is the entire point.
//
// enc[2] / k_cache[2] / v_cache[2] supply per-row encoder states and KV cache
// pointers. input/output are [2, DECODER_HIDDEN_SIZE].
cl_mem MusicgenDecoderLayer_forward_m2(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem input,               // [2, hidden]
    int layer_idx,
    int start_pos,
    cl_mem* k_cache_row0, cl_mem* v_cache_row0,
    cl_mem* k_cache_row1, cl_mem* v_cache_row1,
    cl_mem enc_row0, cl_mem enc_row1,
    const char* weight_prefix)
{
    if (!input) { NNOPT_ERROR("MusicgenDecoderLayer_forward_m2: input is null"); return nullptr; }
    const char* wp = (weight_prefix ? weight_prefix : "");
    const int hidden = MODEL_CONFIG::DECODER_HIDDEN_SIZE;
    const size_t row_bytes = (size_t)hidden * sizeof(nnopt_storage_t);
    cl_int err = CL_SUCCESS;

    static cl_program utils_program_m2 = nullptr;
    if (!utils_program_m2) {
        utils_program_m2 = cl_ctx.build_program_from_file("kernels/utils.cl"); // PROGRAM-INIT-OK
        if (!utils_program_m2) { NNOPT_ERROR("m2: utils program build failed"); return nullptr; }
    }

    // Per-row scratch for attention I/O ([1,hidden]). Pooled (process lifetime).
    static cl_mem s_row_in = nullptr;   // attention input slice
    auto row_scratch = [&](cl_mem& slot) -> cl_mem {
        if (!slot) { cl_int e=CL_SUCCESS; slot = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, row_bytes, nullptr, &e); if (e!=CL_SUCCESS) slot=nullptr; }
        return slot;
    };

    cl_mem residual = input;

    // Helper: run attention for one row, writing its output into dst[row offset].
    // `src` is the [2,hidden] LayerNorm output; extracts row r, runs attention,
    // copies the [1,hidden] result into dst at r*hidden.
    auto attn_row = [&](cl_mem src, cl_mem dst, int r, const char* wp_attn,
                        cl_mem* kc, cl_mem* vc, cl_mem enc) -> bool {
        cl_mem rin = row_scratch(s_row_in);
        if (!rin) { NNOPT_ERROR("m2 attn_row: scratch alloc failed"); return false; }
        if (clEnqueueCopyBuffer(queue, src, rin, (size_t)r*row_bytes, 0, row_bytes, 0, nullptr, nullptr) != CL_SUCCESS) {
            NNOPT_ERROR("m2 attn_row: row extract copy failed"); return false; }
        cl_mem ao = MusicgenSdpaAttention_forward(
            cl_ctx, weights, queue, rin, /*seq_len=*/1, layer_idx, start_pos,
            kc, vc, enc, wp_attn);
        if (!ao) { NNOPT_ERROR("m2 attn_row: attention returned null"); return false; }
        cl_int ce = clEnqueueCopyBuffer(queue, ao, dst, 0, (size_t)r*row_bytes, row_bytes, 0, nullptr, nullptr);
        clReleaseMemObject(ao);
        if (ce != CL_SUCCESS) { NNOPT_ERROR("m2 attn_row: writeback copy failed"); return false; }
        return true;
    };

    // ── Self-attention block ──────────────────────────────────────────────
    char wp_self_ln[256]; wp_child(wp_self_ln, sizeof(wp_self_ln), wp, "self_attn_layer_norm");
    cl_mem normed = LayerNorm_forward(cl_ctx, weights, queue, residual, /*seq_len=*/2,
                                      layer_idx, start_pos, nullptr, nullptr, nullptr, wp_self_ln);
    if (!normed) { NNOPT_ERROR("m2: self_attn_layer_norm null"); return nullptr; }

    cl_mem attn_out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 2*row_bytes, nullptr, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(normed); NNOPT_ERROR("m2: attn_out alloc"); return nullptr; }
    char wp_self_attn[256]; wp_child(wp_self_attn, sizeof(wp_self_attn), wp, "self_attn");
    if (!attn_row(normed, attn_out, 0, wp_self_attn, k_cache_row0, v_cache_row0, nullptr) ||
        !attn_row(normed, attn_out, 1, wp_self_attn, k_cache_row1, v_cache_row1, nullptr)) {
        clReleaseMemObject(normed); clReleaseMemObject(attn_out); return nullptr; }
    clReleaseMemObject(normed);
    if (!element_add_inplace(queue, utils_program_m2, residual, attn_out, (size_t)2*hidden)) {
        clReleaseMemObject(attn_out); NNOPT_ERROR("m2: residual add after self_attn"); return nullptr; }
    clReleaseMemObject(attn_out);
    cl_mem x = residual;   // alias (caller owns input)

    // ── Cross-attention block (per-row encoder states) ────────────────────
    if (enc_row0 || enc_row1) {
        char wp_enc_ln[256]; wp_child(wp_enc_ln, sizeof(wp_enc_ln), wp, "encoder_attn_layer_norm");
        cl_mem enc_normed = LayerNorm_forward(cl_ctx, weights, queue, x, /*seq_len=*/2,
                                              layer_idx, start_pos, nullptr, nullptr, nullptr, wp_enc_ln);
        if (!enc_normed) { NNOPT_ERROR("m2: encoder_attn_layer_norm null"); return nullptr; }
        cl_mem enc_attn_out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 2*row_bytes, nullptr, &err);
        if (err != CL_SUCCESS) { clReleaseMemObject(enc_normed); NNOPT_ERROR("m2: enc_attn_out alloc"); return nullptr; }
        char wp_enc_attn[256]; wp_child(wp_enc_attn, sizeof(wp_enc_attn), wp, "encoder_attn");
        // Cross-attn ignores k/v cache; pass row-0 cache slots (unused).
        if (!attn_row(enc_normed, enc_attn_out, 0, wp_enc_attn, k_cache_row0, v_cache_row0, enc_row0) ||
            !attn_row(enc_normed, enc_attn_out, 1, wp_enc_attn, k_cache_row1, v_cache_row1, enc_row1)) {
            clReleaseMemObject(enc_normed); clReleaseMemObject(enc_attn_out); return nullptr; }
        clReleaseMemObject(enc_normed);
        if (!element_add_inplace(queue, utils_program_m2, x, enc_attn_out, (size_t)2*hidden)) {
            clReleaseMemObject(enc_attn_out); NNOPT_ERROR("m2: residual add after encoder_attn"); return nullptr; }
        clReleaseMemObject(enc_attn_out);
    }

    // ── FFN block (fully M=2) ─────────────────────────────────────────────
    char wp_final_ln[256]; wp_child(wp_final_ln, sizeof(wp_final_ln), wp, "final_layer_norm");
    cl_mem ffn_in = LayerNorm_forward(cl_ctx, weights, queue, x, /*seq_len=*/2,
                                      layer_idx, start_pos, nullptr, nullptr, nullptr, wp_final_ln);
    if (!ffn_in) { NNOPT_ERROR("m2: final_layer_norm null"); return nullptr; }
    char wp_fc1[256]; wp_child(wp_fc1, sizeof(wp_fc1), wp, "fc1");
    cl_mem fc1_out = Linear_forward(cl_ctx, weights, queue, ffn_in, /*seq_len=*/2,
                                    layer_idx, start_pos, nullptr, nullptr, nullptr, wp_fc1);
    clReleaseMemObject(ffn_in);
    if (!fc1_out) { NNOPT_ERROR("m2: fc1 null"); return nullptr; }
    cl_mem act_out = GELUActivation_forward(cl_ctx, weights, queue, fc1_out, /*seq_len=*/2,
                                            layer_idx, start_pos, nullptr, nullptr, nullptr, "");
    clReleaseMemObject(fc1_out);
    if (!act_out) { NNOPT_ERROR("m2: gelu null"); return nullptr; }
    char wp_fc2[256]; wp_child(wp_fc2, sizeof(wp_fc2), wp, "fc2");
    cl_mem fc2_out = Linear_forward(cl_ctx, weights, queue, act_out, /*seq_len=*/2,
                                    layer_idx, start_pos, nullptr, nullptr, nullptr, wp_fc2);
    clReleaseMemObject(act_out);
    if (!fc2_out) { NNOPT_ERROR("m2: fc2 null"); return nullptr; }
    if (!element_add_inplace(queue, utils_program_m2, x, fc2_out, (size_t)2*hidden)) {
        clReleaseMemObject(fc2_out); NNOPT_ERROR("m2: residual add after FFN"); return nullptr; }
    clReleaseMemObject(fc2_out);
    clRetainMemObject(x);
    return x;
}
}
