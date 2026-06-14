// Reference: model_info/transformers_src/modeling_musicgen.py:~250-420 MusicgenDecoder.forward
// Auto-generated backbone for facebook/musicgen-small.
//
// NOTE (debug harness): This workspace is still wiring the true end-to-end
// MusicGen pipeline (text_encoder -> encoder_hidden_states -> decoder ->
// codebooks -> vocoder). For now, this backbone focuses on matching the
// traced *decoder* subgraph shapes so layer dumps align with reference.
//
// Key reference behavior:
// - MusicgenDecoder.forward reshapes decoder_input_ids from (bsz*num_codebooks, tgt_len)
//   to (bsz, num_codebooks, tgt_len) and computes:
//     inputs_embeds = sum(embed_tokens[c](input[:, c]) for c in range(num_codebooks))
//     positions = embed_positions(input, past_key_values_length)
//     hidden_states = inputs_embeds + positions
//   Reference: modeling_musicgen.py: MusicgenDecoder.forward

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"
#include "profiler.h"

#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>

extern "C" cl_mem MusicgenDecoderLayer_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem Embedding_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem LayerNorm_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem MusicgenDecoderLayer_forward_m2(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int layer_idx, int start_pos,
    cl_mem* k_cache_row0, cl_mem* v_cache_row0,
    cl_mem* k_cache_row1, cl_mem* v_cache_row1,
    cl_mem enc_row0, cl_mem enc_row1, const char* weight_prefix);

// ── Per-decoder-layer MEGAKERNEL (src/ops/MegaDecoderLayer.cpp) ──────────
extern "C" void mega_reset_cross_kv();
extern "C" void mega_invalidate_cross_kv();
extern "C" bool mega_precompute_cross_kv(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem enc_row0, cl_mem enc_row1, int enc_len);
extern "C" bool mega_decoder_layer_m2(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem x_in0, cl_mem x_out0, cl_mem x_in1, cl_mem x_out1,
    int layer_idx, int start_pos,
    cl_mem* k_cache_r0, cl_mem* v_cache_r0,
    cl_mem* k_cache_r1, cl_mem* v_cache_r1);
extern "C" bool mega_lmheads_dispatch(OpenCLContext&, cl_command_queue,
                                      cl_mem, cl_mem, cl_mem, int, int);
extern "C" bool mega_decoder_layer_m2_n(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem x_in0, cl_mem x_out0, cl_mem x_in1, cl_mem x_out1,
    int layer_idx, int start_pos,
    cl_mem* k_cache_r0, cl_mem* v_cache_r0,
    cl_mem* k_cache_r1, cl_mem* v_cache_r1,
    int num_wg, int single_row);

// ── Generation-scope decoder state ──────────────────────────────────────
// modeling_musicgen.py: encoder_hidden_states is computed ONCE per generation
// (T5 + enc_to_dec_proj) and reused by every decoder layer's cross-attention;
// past_key_values persist per layer across decode steps. These globals are
// the C++ counterpart, owned here and reset between generations.
int g_musicgen_enc_len = 0;                       // read by MusicgenSdpaAttention (cross K/V length)
static cl_mem g_enc_states = nullptr;             // [enc_len, 1024] nnopt_storage_t
static cl_mem g_enc_zero = nullptr;               // zeros twin — CFG unconditional branch
static int    g_cfg_uncond = 0;                   // 0 = conditioned, 1 = unconditional
static cl_mem g_k_cache[64] = {};                 // per-layer K cache (bank0: cond, bank1 @+32: uncond)
static cl_mem g_v_cache[64] = {};                 // per-layer V cache (dual-bank, same layout)

// CFG (modeling_musicgen.py:1734): the unconditional branch runs the SAME
// decoder with encoder_hidden_states = zeros (attention projections are
// bias-free, so cross-attn contributes exactly 0) and its OWN KV caches.
extern "C" void model_set_cfg_branch(int uncond) { g_cfg_uncond = uncond ? 1 : 0; }

extern "C" bool model_set_encoder_states(OpenCLContext& cl_ctx, const float* states, int T, int dim) {
    if (!states || T <= 0 || dim != MODEL_CONFIG::DECODER_HIDDEN_SIZE) {
        NNOPT_ERROR_FMT("model_set_encoder_states: bad args (T=%d dim=%d)", T, dim);
        return false;
    }
    if (g_enc_states) { clReleaseMemObject(g_enc_states); g_enc_states = nullptr; }
    std::vector<nnopt_storage_t> host((size_t)T * dim);
#ifdef NNOPT_USE_FP16
    for (size_t i = 0; i < host.size(); ++i) host[i] = nnopt_f32_to_f16(states[i]);
#else
    for (size_t i = 0; i < host.size(); ++i) host[i] = states[i];
#endif
    cl_int err = CL_SUCCESS;
    g_enc_states = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  host.size() * sizeof(nnopt_storage_t), host.data(), &err);
    if (err != CL_SUCCESS || !g_enc_states) {
        NNOPT_ERROR_FMT("model_set_encoder_states: upload failed (err=%d)", (int)err);
        g_enc_states = nullptr;
        return false;
    }
    if (g_enc_zero) { clReleaseMemObject(g_enc_zero); g_enc_zero = nullptr; }
    std::vector<nnopt_storage_t> zero_host(host.size());
#ifdef NNOPT_USE_FP16
    for (size_t i = 0; i < zero_host.size(); ++i) zero_host[i] = nnopt_f32_to_f16(0.0f);
#else
    for (size_t i = 0; i < zero_host.size(); ++i) zero_host[i] = 0.0f;
#endif
    g_enc_zero = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                zero_host.size() * sizeof(nnopt_storage_t), zero_host.data(), &err);
    if (err != CL_SUCCESS || !g_enc_zero) {
        NNOPT_ERROR_FMT("model_set_encoder_states: zero-twin upload failed (err=%d)", (int)err);
        g_enc_zero = nullptr;
        return false;
    }
    g_musicgen_enc_len = T;
    NNOPT_CHECKPOINT("encoder_hidden_states uploaded (T5 conditioning active)");
    return true;
}

extern "C" void model_reset_decode_state() {
    // KV caches are KEPT across generations (fixed [kMaxK, hidden] capacity):
    // position p is always written by step p before any step ≥p reads it, so
    // stale contents are unreachable. Releasing+lazily-reallocating the 128
    // one-MB buffers here was the dominant chunk of the measured ~4.8 s
    // first-step allocation stall (driver pool refill at ~layer 10-13).
    //
    // Cross-attn K/V are VALUE-invalidated only — mega_precompute_cross_kv
    // reuses the buffers/views/kernel clones when enc_len is unchanged.
    mega_invalidate_cross_kv();
}

static bool read_i32_at_index(
    OpenCLContext& cl_ctx,
    cl_mem parent,
    size_t index,
    int32_t* out_value)
{
    // Adreno often rejects tiny sub-buffers (CL_MISALIGNED_SUB_BUFFER_OFFSET = -13).
    // Read the single int32 to host instead, then re-upload a 1-int buffer.
    if (!out_value) return false;

    cl_command_queue queue = cl_ctx.queue();
    cl_int err = CL_SUCCESS;
    int32_t tmp = 0;
    const size_t byte_off = index * sizeof(int32_t);
    err = clEnqueueReadBuffer(queue, parent, CL_TRUE, byte_off, sizeof(int32_t), &tmp, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("clEnqueueReadBuffer(int32 idx=%zu) failed (err=%d)", index, (int)err);
        return false;
    }
    *out_value = tmp;
    return true;
}

// Internal core. When logits_dev_out != nullptr we write the [num_codebooks*vocab]
// logits into a NEWLY-ALLOCATED device buffer (returned via the out-param) and
// SKIP the blocking host read — the caller is responsible for reading or further
// processing on-device (used by the interleaved CFG path so cond+uncond stream
// into the in-order queue with a SINGLE drain at the end instead of one drain
// per pass). When logits_dev_out == nullptr, behaves exactly as before: reads
// logits to host and returns them.
static std::vector<float> model_forward_graph_core(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    int start_pos,
    cl_mem* logits_dev_out)
{
    NNOPT_CHECKPOINT("model_forward_graph entry");
    cl_command_queue queue = cl_ctx.queue();

    // The decoder reference harness expects the *decoder_input_ids* to be the
    // MusicGen decoder start token (pad token tensor) = 2048 for EACH codebook.
    // See reference/layers/*embed_tokens_layer_{0..3}_input.bin: mean=min=max=2048.
    //
    // Feeding text token IDs here is invalid and immediately corrupts the
    // per-codebook embeddings (first divergence observed at embed_tokens_layer_1).
    const int num_codebooks = MODEL_CONFIG::NUM_CODEBOOKS;
    const int tgt_len = 1;

    // IMPORTANT: for this workspace we want the decoder (music) BOS/pad id (=2048).
    // DECODER_START_TOKEN_ID can be polluted by the text-encoder's decoder_start_token_id in some scaffolds.
    // Use BOS_TOKEN_ID, which is already verified correct for the music decoder.
    //
    // input_ids carries the CURRENT per-codebook tokens for this decode step
    // (the delay-pattern grid column built in main.cpp). At step 0 the grid
    // holds BOS for every codebook, matching the reference prefill. Falling
    // back to BOS when fewer than 4 ids arrive keeps SxS harness runs valid.
    const int32_t decoder_start_id = (int32_t)MODEL_CONFIG::BOS_TOKEN_ID;
    std::vector<int32_t> host_cb_ids(num_codebooks, decoder_start_id);
    if ((int)input_ids.size() >= num_codebooks) {
        for (int c = 0; c < num_codebooks; ++c) host_cb_ids[(size_t)c] = input_ids[(size_t)c];
    }

    // Dump the discrete decoder_input_ids tensor (shape [1,4,1] flattened => 4 elems)
    // so SxS can confirm we match the reference harness.
    // Note: NNOPT_LAYER_CHECK_INPUT_INT treats the buffer as int32.
    //
    // IMPORTANT: the reference dump for this name is ..._input.bin (NOT _output.bin)
    // because it is an input-side local. So we must use NNOPT_LAYER_CHECK_INPUT_INT
    // (which appends _input) and we must NOT create a plain NNOPT_LAYER_CHECK dump
    // under the same stem.

    cl_int err = CL_SUCCESS;
    // Keep token IDs in a host array for correctness + dumpability.
    // We still upload to device for the embedding kernels.
    const int32_t v0 = host_cb_ids[0];
    const int32_t v1 = host_cb_ids[1];
    const int32_t v2 = host_cb_ids[2];
    const int32_t v3 = host_cb_ids[3];

    // Dump decoder_input_ids as int32.
    // IMPORTANT: this is an *INT* buffer, so use the INT-specific macro.
    // Also: we must dump 4 ints (one per codebook). The previous approach
    // created a temporary cl_mem and some runs ended up dumping only 1 int,
    // causing SxS size mismatches.
    {
        int32_t ids_i[4] = {v0, v1, v2, v3};

        // Use a dedicated device buffer for the dump to avoid any aliasing.
        cl_int dump_err = CL_SUCCESS;
        cl_mem ids_dump = clCreateBuffer(
            cl_ctx.context(),
            CL_MEM_READ_WRITE,
            (size_t)num_codebooks * sizeof(int32_t),
            nullptr,
            &dump_err);
        if (dump_err == CL_SUCCESS && ids_dump) {
            dump_err = clEnqueueWriteBuffer(queue, ids_dump, CL_TRUE, 0,
                                           (size_t)num_codebooks * sizeof(int32_t),
                                           ids_i, 0, nullptr, nullptr);
            if (dump_err == CL_SUCCESS) {
                NNOPT_LAYER_CHECK_INPUT_INT("block0_sub_musicgendecoder_input_ids", queue, ids_dump, (size_t)num_codebooks);
            }
            clReleaseMemObject(ids_dump);
        }
    }

    // ── Embeddings: sum over codebooks ────────────────────────────────
    // Each codebook embedding consumes a single token id (tgt_len=1).
    cl_mem x = nullptr;
    {

        cl_mem id0 = nullptr;
        cl_mem id1 = nullptr;
        cl_mem id2 = nullptr;
        cl_mem id3 = nullptr;
        cl_mem e0 = nullptr;
        cl_mem e1 = nullptr;
        cl_mem e2 = nullptr;
        cl_mem e3 = nullptr;

        auto cleanup = [&]() -> std::vector<float> {
            if (id0) clReleaseMemObject(id0);
            if (id1) clReleaseMemObject(id1);
            if (id2) clReleaseMemObject(id2);
            if (id3) clReleaseMemObject(id3);
            if (e0) clReleaseMemObject(e0);
            if (e1) clReleaseMemObject(e1);
            if (e2) clReleaseMemObject(e2);
            if (e3) clReleaseMemObject(e3);
            return std::vector<float>();
        };

        // UPLOAD-OK: 1 int32 per codebook token id.
        cl_int up_err = CL_SUCCESS;
        id0 = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int32_t), (void*)&v0, &up_err);
        if (up_err != CL_SUCCESS || !id0) { NNOPT_ERROR_FMT("id0 upload failed (err=%d)", (int)up_err); return cleanup(); }
        id1 = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int32_t), (void*)&v1, &up_err);
        if (up_err != CL_SUCCESS || !id1) { NNOPT_ERROR_FMT("id1 upload failed (err=%d)", (int)up_err); return cleanup(); }
        id2 = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int32_t), (void*)&v2, &up_err);
        if (up_err != CL_SUCCESS || !id2) { NNOPT_ERROR_FMT("id2 upload failed (err=%d)", (int)up_err); return cleanup(); }
        id3 = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int32_t), (void*)&v3, &up_err);
        if (up_err != CL_SUCCESS || !id3) { NNOPT_ERROR_FMT("id3 upload failed (err=%d)", (int)up_err); return cleanup(); }

        e0 = Embedding_forward(cl_ctx, weights, queue, id0, tgt_len, -1, start_pos,
                              nullptr, nullptr, nullptr,
                              "decoder.model.decoder.embed_tokens.0");
        e1 = Embedding_forward(cl_ctx, weights, queue, id1, tgt_len, -1, start_pos,
                              nullptr, nullptr, nullptr,
                              "decoder.model.decoder.embed_tokens.1");
        e2 = Embedding_forward(cl_ctx, weights, queue, id2, tgt_len, -1, start_pos,
                              nullptr, nullptr, nullptr,
                              "decoder.model.decoder.embed_tokens.2");
        e3 = Embedding_forward(cl_ctx, weights, queue, id3, tgt_len, -1, start_pos,
                              nullptr, nullptr, nullptr,
                              "decoder.model.decoder.embed_tokens.3");

        if (!e0 || !e1 || !e2 || !e3) {
            NNOPT_ERROR("Embedding_forward returned null for one of the codebooks");
            return cleanup();
        }

        if (!e0 || !e1 || !e2 || !e3) {
            NNOPT_ERROR("Embedding_forward returned null for one of the codebooks");
            if (e0) clReleaseMemObject(e0);
            if (e1) clReleaseMemObject(e1);
            if (e2) clReleaseMemObject(e2);
            if (e3) clReleaseMemObject(e3);
            return std::vector<float>();
        }

        const size_t hidden = (size_t)MODEL_CONFIG::DECODER_HIDDEN_SIZE;
        // Dump contract expects *element counts*, not byte counts.
        // embed_tokens_layer_{0..3} dumps are activations of shape [tgt_len, hidden]. Here tgt_len=1.
        NNOPT_LAYER_CHECK("embed_tokens_layer_0", queue, e0, hidden);
        NNOPT_LAYER_CHECK("embed_tokens_layer_1", queue, e1, hidden);
        NNOPT_LAYER_CHECK("embed_tokens_layer_2", queue, e2, hidden);
        NNOPT_LAYER_CHECK("embed_tokens_layer_3", queue, e3, hidden);

        // NOTE: Avoid emitting per-pass suffixed dumps for embed_tokens_layer_*.
        // The reference names are unsuffixed (embed_tokens_layer_0_output.bin etc), and
        // SxS currently fuzzy-matches these suffixed names to unrelated layer_* outputs.

        cl_program utils_prog = cl_ctx.get_utils_program();
        if (!utils_prog) {
            NNOPT_ERROR("failed to get utils_program for element_add");
            clReleaseMemObject(e0);
            clReleaseMemObject(e1);
            clReleaseMemObject(e2);
            clReleaseMemObject(e3);
            return std::vector<float>();
        }

        const size_t n = hidden;
        cl_mem s01 = element_add(queue, utils_prog, e0, e1, n);
        cl_mem s23 = element_add(queue, utils_prog, e2, e3, n);
        if (!s01 || !s23) {
            NNOPT_ERROR("failed to sum codebook embeddings (stage1)");
            if (s01) clReleaseMemObject(s01);
            if (s23) clReleaseMemObject(s23);
            clReleaseMemObject(e0);
            clReleaseMemObject(e1);
            clReleaseMemObject(e2);
            clReleaseMemObject(e3);
            return std::vector<float>();
        }
        x = element_add(queue, utils_prog, s01, s23, n);

        clReleaseMemObject(e0);
        clReleaseMemObject(e1);
        clReleaseMemObject(e2);
        clReleaseMemObject(e3);
        clReleaseMemObject(s01);
        clReleaseMemObject(s23);

        if (!x) {
            NNOPT_ERROR("failed to sum codebook embeddings (stage2)");
            return std::vector<float>();
        }

        // forward_graph.json expects a *pre-hook* dump for decoder.model.decoder.embed_tokens.
        // In the PyTorch trace, this is the ModuleList itself and the hook captures an internal
        // handle-like object (not a real activation tensor). The reference dump for
        // decoder_model_decoder_embed_tokens_frontend_out therefore has output_shape [1,1]
        // and is effectively non-numeric.
        //
        // IMPORTANT: Do NOT emit a real tensor under this dump name; doing so causes SxS
        // alignment failures (ref size=1, dump size=hidden).
        (void)hidden;
    }

    // ── Positional embeddings add (sinusoidal table) ───────────────────
    // positions = embed_positions(input, past_key_values_length)
    // In reference, positions is [tgt_len, hidden] (tgt_len=1) taken at abs pos = start_pos.
    {
        cl_program utils_prog = cl_ctx.get_utils_program();
        if (!utils_prog) {
            NNOPT_ERROR("failed to get utils_program for positional embedding add");
            if (x) clReleaseMemObject(x);
            return std::vector<float>();
        }
        cl_mem pos_w = weights.get_buffer("decoder.model.decoder.embed_positions.weights");
        if (!pos_w) {
            NNOPT_ERROR("missing decoder.model.decoder.embed_positions.weights");
            if (x) clReleaseMemObject(x);
            return std::vector<float>();
        }

            const size_t hidden = (size_t)MODEL_CONFIG::DECODER_HIDDEN_SIZE;

            // Reference alignment: our decoder harness always uses tgt_len=1.
        // MusicgenSinusoidalPositionalEmbedding.forward(past_key_values_length)
        // returns positions for absolute indices:
        //   [past_key_values_length, past_key_values_length + tgt_len)
        // In our harness, past_key_values_length corresponds to start_pos.
        // Using abs_pos=0 makes every decode step use position 0 and breaks decode.
        const size_t abs_pos = (size_t)start_pos;

        // BENCHMARKS A1/A6: the position row lives in the embed_positions weight
        // buffer, which is already resident on-device for the whole run. Add it
        // straight into x[] with a device-side offset add — NO clEnqueueReadBuffer
        // (CL_TRUE) host round-trip and NO per-step scratch upload, both of which
        // previously stalled the in-order queue every decode step.
        if (!element_add_offset(queue, utils_prog, x, pos_w, abs_pos * hidden, hidden)) {
            NNOPT_ERROR("failed to add positional embeddings (offset add)");
            if (x) clReleaseMemObject(x);
            return std::vector<float>();
        }
        // NOTE: no reference dump exists for this intermediate; keep for runtime sanity only.
        // Do not suffix by pass to avoid fuzzy-match collisions.
        NNOPT_LAYER_CHECK("decoder_embed_positions_added", queue, x, hidden);
    }

    // ── Decoder layers (24) ────────────────────────────────────────────
    // For dump alignment against the current reference harness, run each layer with seq_len=tgt_len=1.
    // NUM_HIDDEN_LAYERS (= decoder.num_hidden_layers = 24) is the MUSIC decoder
    // depth. NUM_DECODER_LAYERS was polluted by text_encoder.num_decoder_layers
    // (T5's own decoder, 12) via the same flattener leak as the BOS id — the
    // binary silently ran HALF the decoder (layers 0-11 validated perfectly,
    // 12-23 never executed; logits cos 0.81, noise output, 2026-06-04).
    for (int layer_idx = 0; layer_idx < MODEL_CONFIG::NUM_HIDDEN_LAYERS; ++layer_idx) {
        char wp[96];
        snprintf(wp, sizeof(wp), "decoder.model.decoder.layers.%d", layer_idx);

        // These names must match reference/layers/*.bin exactly.
        // (They are also enumerated in .nnport/dump_name_map.json.)
        char ln_name[64];
        char attn_name[64];
        snprintf(ln_name, sizeof(ln_name), "self_attn_layer_norm_%d", layer_idx);
        snprintf(attn_name, sizeof(attn_name), "self_attn_%d", layer_idx);

        // Per-layer persistent KV cache + the shared T5 conditioning states.
        // g_enc_states may legitimately be null in SxS harness runs that skip
        // encode_text — the layer then skips its cross-attn block, matching
        // the old (unconditioned) behavior instead of crashing.
        const int bank = g_cfg_uncond ? 32 : 0;     // dual KV banks for CFG
        cl_mem enc = g_cfg_uncond ? g_enc_zero : g_enc_states;
        cl_mem out = MusicgenDecoderLayer_forward(
            cl_ctx, weights, queue,
            x, tgt_len, layer_idx, start_pos,
            &g_k_cache[bank + layer_idx],
            &g_v_cache[bank + layer_idx],
            enc,
            wp);
        if (!out) {
            NNOPT_ERROR_FMT("MusicgenDecoderLayer_forward returned null at layer %d", layer_idx);
            if (x) clReleaseMemObject(x);
            return std::vector<float>();
        }
        if (out != x) {
            clReleaseMemObject(x);
            x = out;
        }

        // NOTE: Do NOT emit fake sub-op dumps here.
        // self_attn_layer_norm_i and self_attn_i are emitted inside MusicgenDecoderLayer_forward
        // from the true intermediate tensors. Duplicating them here aliases unrelated buffers
        // and destroys the SxS boundary signal.

        char dn[64];
        snprintf(dn, sizeof(dn), "layer_%d", layer_idx);
        NNOPT_LAYER_CHECK(dn, queue, x, (size_t)MODEL_CONFIG::DECODER_HIDDEN_SIZE);
    }

    // ── Logits readout ─────────────────────────────────────────────────
    // MusicGen predicts 4 parallel codebooks. In the reference decoder, lm_head is shared
    // across codebooks (weights: model.lm_head.weight) and applied to each codebook stream.
    // Our debug harness only runs ONE stream (sum of codebook embeddings) at tgt_len=1, so
    // we compute a single-row logits [1, vocab] from the final hidden state.
    //
    // Reference: model_info/transformers_src/modeling_musicgen.py (MusicgenForConditionalGeneration.forward)
    //            + MusicgenModel.forward (decoder output -> lm_head).

    // ── Final decoder layer norm ────────────────────────────────────────
    // modeling_musicgen.py MusicgenDecoder.forward: hidden_states =
    // self.layer_norm(hidden_states) AFTER the layer stack, BEFORE lm_heads.
    // Weight: decoder.model.decoder.layer_norm.{weight,bias}. Omitting this
    // norm skews every logit row (was missing in the harness readout).
    {
        cl_mem normed = LayerNorm_forward(
            cl_ctx, weights, queue,
            x, tgt_len, -1, start_pos,
            nullptr, nullptr, nullptr,
            "decoder.model.decoder.layer_norm");
        if (!normed) {
            NNOPT_ERROR("final decoder layer_norm failed");
            if (x) clReleaseMemObject(x);
            return std::vector<float>();
        }
        if (normed != x) { clReleaseMemObject(x); x = normed; }
        NNOPT_LAYER_CHECK("decoder_final_layer_norm", queue, x,
                          (size_t)MODEL_CONFIG::DECODER_HIDDEN_SIZE);
    }

    // ── Logits readout: ALL FOUR codebook heads ─────────────────────────
    // modeling_musicgen.py MusicgenForCausalLM.forward: lm_logits = stack(
    // [head(hidden) for head in self.lm_heads]) — one [vocab] row per
    // codebook. Returns flattened [4 * vocab]; main.cpp samples per codebook.
    const int vocab = MODEL_CONFIG::VOCAB_SIZE;
    const int hidden = MODEL_CONFIG::DECODER_HIDDEN_SIZE;
    std::vector<float> logits((size_t)num_codebooks * vocab, 0.0f);

    // Opt #2 (BENCHMARKS.md): the 4 lm_heads share the input row — concat their
    // weights ONCE into [4*vocab, hidden] and run ONE GEMM + ONE readback per
    // step instead of 4+4 (8 dispatch/sync points per step, x2 with CFG).
    static cl_mem heads_cat = nullptr;
    if (!heads_cat) {
        std::vector<nnopt_storage_t> cat((size_t)num_codebooks * vocab * hidden);
        for (int cb = 0; cb < num_codebooks; ++cb) {
            char hk[64]; snprintf(hk, sizeof(hk), "decoder.lm_heads.%d.weight", cb);
            const std::vector<float> hw = weights.get_host_vec(hk);
            if ((int)hw.size() != vocab * hidden) { NNOPT_ERROR_FMT("lm_heads concat: bad %s", hk); heads_cat = nullptr; break; }
            for (size_t i = 0; i < hw.size(); ++i)
#ifdef NNOPT_USE_FP16
                cat[(size_t)cb * vocab * hidden + i] = nnopt_f32_to_f16(hw[i]);
#else
                cat[(size_t)cb * vocab * hidden + i] = hw[i];
#endif
        }
        cl_int cerr = CL_SUCCESS;
        heads_cat = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   cat.size() * sizeof(nnopt_storage_t), cat.data(), &cerr);
        if (cerr != CL_SUCCESS) { NNOPT_ERROR_FMT("lm_heads concat upload %d", (int)cerr); heads_cat = nullptr; }
    }
    if (heads_cat) {
        const int N = num_codebooks * vocab;
        cl_int e2 = CL_SUCCESS;
        // Interleaved-CFG path needs a DISTINCT logits buffer per pass (so cond
        // and uncond don't clobber each other before the combine). Allocate a
        // fresh buffer when logits_dev_out is requested; otherwise reuse the
        // process-static scratch (single-pass / non-CFG behavior, unchanged).
        cl_mem lg_dev = nullptr;
        if (logits_dev_out) {
            lg_dev = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)N * sizeof(nnopt_storage_t), nullptr, &e2);
        } else {
            static cl_mem lg_static = nullptr;
            if (!lg_static) lg_static = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)N * sizeof(nnopt_storage_t), nullptr, &e2);
            lg_dev = lg_static;
        }
        if (lg_dev && pytorch_linear(queue, 1, N, hidden, x, heads_cat, lg_dev)) {
            NNOPT_LAYER_CHECK("lm_head", queue, lg_dev, (size_t)vocab);
            if (x) clReleaseMemObject(x);
            if (logits_dev_out) {
                // No host read — caller (interleaved CFG) drains the queue once
                // after BOTH passes are enqueued, then combines on-device.
                *logits_dev_out = lg_dev;
                return std::vector<float>();
            }
#ifdef NNOPT_USE_FP16
            std::vector<uint16_t> tmp((size_t)N);
            if (clEnqueueReadBuffer(queue, lg_dev, CL_TRUE, 0, (size_t)N * sizeof(uint16_t), tmp.data(), 0, nullptr, nullptr) == CL_SUCCESS)
                for (int i = 0; i < N; ++i) logits[(size_t)i] = nnopt_f16_to_f32(tmp[(size_t)i]);
#else
            clEnqueueReadBuffer(queue, lg_dev, CL_TRUE, 0, (size_t)N * sizeof(float), logits.data(), 0, nullptr, nullptr);
#endif
            return logits;
        }
        if (logits_dev_out && lg_dev) clReleaseMemObject(lg_dev);
        NNOPT_ERROR("lm_heads concat GEMM failed — falling back to per-head path");
    }

    // Per-head fallback does not support the device-out (interleaved CFG) path;
    // signal the caller to use the host-logits two-pass route instead.
    if (logits_dev_out) { *logits_dev_out = nullptr; if (x) clReleaseMemObject(x); return std::vector<float>(); }

    for (int cb = 0; cb < num_codebooks; ++cb) {
        char head_key[64];
        snprintf(head_key, sizeof(head_key), "decoder.lm_heads.%d.weight", cb);
        cl_mem lm_w = weights.get_buffer(head_key);
        if (!lm_w) {
            NNOPT_ERROR_FMT("missing %s", head_key);
            if (x) clReleaseMemObject(x);
            return std::vector<float>();
        }
        cl_int err_logits = CL_SUCCESS;
        cl_mem logits_dev = clCreateBuffer(
            cl_ctx.context(), CL_MEM_READ_WRITE,
            (size_t)vocab * sizeof(nnopt_storage_t), nullptr, &err_logits);
        if (err_logits != CL_SUCCESS || !logits_dev) {
            NNOPT_ERROR_FMT("failed to allocate logits_dev cb=%d (err=%d)", cb, (int)err_logits);
            if (x) clReleaseMemObject(x);
            return std::vector<float>();
        }
        if (!pytorch_linear(queue, /*M=*/1, /*N=*/vocab, /*K=*/hidden, x, lm_w, logits_dev)) {
            NNOPT_ERROR_FMT("pytorch_linear(lm_heads.%d) failed", cb);
            clReleaseMemObject(logits_dev);
            if (x) clReleaseMemObject(x);
            return std::vector<float>();
        }
        if (cb == 0) NNOPT_LAYER_CHECK("lm_head", queue, logits_dev, (size_t)vocab);

        float* dst = logits.data() + (size_t)cb * vocab;
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> tmp((size_t)vocab);
        cl_int rd = clEnqueueReadBuffer(queue, logits_dev, CL_TRUE, 0,
                                        (size_t)vocab * sizeof(uint16_t), tmp.data(), 0, nullptr, nullptr);
        if (rd != CL_SUCCESS) {
            NNOPT_ERROR_FMT("read logits_dev cb=%d failed (err=%d)", cb, (int)rd);
        } else {
            for (int i = 0; i < vocab; ++i) dst[i] = nnopt_f16_to_f32(tmp[(size_t)i]);
        }
#else
        cl_int rd = clEnqueueReadBuffer(queue, logits_dev, CL_TRUE, 0,
                                        (size_t)vocab * sizeof(float), dst, 0, nullptr, nullptr);
        if (rd != CL_SUCCESS) {
            NNOPT_ERROR_FMT("read logits_dev cb=%d failed (err=%d)", cb, (int)rd);
        }
#endif
        clReleaseMemObject(logits_dev);
    }

    if (x) clReleaseMemObject(x);
    return logits;   // [num_codebooks * vocab]
}

// Public single-pass entry (unchanged behavior): runs the decoder + reads
// logits to host.
std::vector<float> model_forward_graph(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    int start_pos)
{
    return model_forward_graph_core(cl_ctx, weights, input_ids, start_pos, nullptr);
}

// Interleaved-CFG entry: runs the decoder and leaves logits in a freshly
// allocated DEVICE buffer (returned via the pointer) WITHOUT a host read, so
// the caller can enqueue the second CFG pass before any queue drain. Returns
// nullptr in *out (and logs) only if the concat-GEMM readout path is unavailable.
extern "C" void model_forward_graph_logits_dev(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    int start_pos,
    cl_mem* logits_dev_out)
{
    if (logits_dev_out) *logits_dev_out = nullptr;
    (void)model_forward_graph_core(cl_ctx, weights, input_ids, start_pos, logits_dev_out);
}

// ─── Full M=2 CFG-batched forward (BENCHMARKS Stage 1) ────────────────────
// Runs the decoder ONCE over a [2,hidden] activation buffer: row0 = conditioned
// (g_enc_states + KV bank 0), row1 = unconditional (g_enc_zero + KV bank 32).
// Every row-wise op (embeddings/pos via duplication, LayerNorm, fc1/GELU/fc2,
// residuals, lm_heads GEMM) runs M=2 → HALF the dispatch overhead of the two
// interleaved M=1 passes. Attention is per-row inside MusicgenDecoderLayer_
// forward_m2 (own KV bank + own encoder states). lm_heads run M=2 → [2,N];
// cfg_combine blends row0/row1 on-device; ONE host read of the blended row.
//
// input_ids are the SAME for both rows (the grid column is fed to both CFG
// branches), so embeddings + positional add are computed once at M=1 and
// duplicated into both rows — byte-identical to running them per-pass.
//
// Returns the blended logits [num_codebooks*vocab] = uncond + g*(cond-uncond).
// Falls back to {} on any failure (caller then uses the interleaved route).
// Stage-4 GPU-resident grid state (set by the host before the decode loop via
// mega_set_decode_grid). When g_grid_dev != null: embeddings read ids from
// grid[:,step] (embed_prologue) and sample_grid writes grid[:,step+1] — no
// per-step host id upload, one whole-grid readback at end-of-decode.
static cl_mem g_grid_dev = nullptr;   // [num_codebooks, steps1] int32, or null
static int    g_grid_steps1 = 0;
static int    g_grid_bos = 0;

// Stage-3/4 kernels+scratch — declared BEFORE m2_impl because the recordable-
// replay fast path overrides their per-step scalar args by handle (definitions/
// lazy init stay in their dispatch helpers below).
static cl_program s_sg_prog = nullptr;
static cl_kernel  s_sg_kernel = nullptr;
static cl_mem     s_sg_ids = nullptr;
static cl_program s_emb_prog = nullptr;
static cl_kernel  s_emb_kernel = nullptr;
static cl_mem     s_emb_tables = nullptr;
static cl_mem     s_emb_out = nullptr;
static cl_mem     s_emb_host_ids = nullptr;

// ── Recordable-queue decode (cl_qcom_recordable_queues) ─────────────────────
// The 250-step pipeline enqueues ~292 dispatches per step; at ~30 µs driver
// cost each that is ~9 ms/step of pure dispatch overhead (measured: decode
// wall 101.8 ms/step vs 92.8 ms/step kernel-sum, 2026-06-05 profile). The
// QCOM extension records one steady-state step ONCE per geometry (CFG-2 /
// single-row) and replays it with 4 scalar arg overrides (step, start_pos,
// self-attn seq_len, sampler step) — the replay's per-dispatch cost measured
// 4.04× cheaper (NNOPT_RECORD_PROBE). NNOPT_RECORD=0 kills the path; any
// build/replay failure disables it permanently and decode continues live.
typedef void* nnopt_recording_t;
static cl_command_queue  g_rec_queue = nullptr;
static nnopt_recording_t g_recording[2] = {nullptr, nullptr};   // [0]=CFG-2 [1]=single-row
static bool              g_rec_disabled = false;
static cl_command_queue  g_queue_override = nullptr;   // set during the record pass
static int               g_geo_live_steps[2] = {0, 0};
static OpenCLContext*    g_rec_ctx = nullptr;

extern "C" int  mega_record_supported();
// Step-params buffer (owned by MegaDecoderLayer.cpp): sp[0]=start_pos for the
// CURRENT step, FillBuffer'd once per step on the in-order queue. Kernels read
// it instead of literal scalar args, so recorded replays need NO arg overrides
// (this driver rejects them with -59).
extern "C" bool   mega_write_step_params(OpenCLContext&, cl_command_queue, int start_pos);
extern "C" cl_mem mega_step_params_buf();
extern "C" bool LayerNorm_forward_into(OpenCLContext&, Weights&, cl_command_queue,
                                       cl_mem input, int seq_len,
                                       const char* weight_prefix, cl_mem out);

// Drop captured recordings (new generation: grid / cross-KV cl_mems change, so
// the baked dispatch args go stale). Safe to call with none captured.
static void record_invalidate() {
    if (g_rec_ctx) {
        for (int i = 0; i < 2; ++i)
            if (g_recording[i]) { g_rec_ctx->release_recording(g_recording[i]); g_recording[i] = nullptr; }
    }
    g_geo_live_steps[0] = g_geo_live_steps[1] = 0;
}

// Forward-declared Stage-3 helper (defined below this function). out_grid (and
// grid_steps1) are set by Stage-4 GPU-resident-grid mode to write the sampled
// ids straight into the device grid; nullptr → CPU grid (read 4 ids back).
static bool sample_grid_dispatch(
    OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem logits2,
    int num_codebooks, int vocab, float guidance, float temperature, int top_k,
    uint32_t seed, int step, int force_argmax, int32_t* out_ids, int single);
// Stage 4+5 fused embed prologue (defined below). host_ids non-null overrides
// the GPU grid (used for prefill/step-0 if needed); nullptr → read GPU grid.
// out_buf non-null → write the embedding into rows 0..out_rows-1 of THAT
// buffer (persistent x — recordable-replay path; the returned handle is
// out_buf, NOT retained). out_buf null → legacy: retained s_emb_out, 1 row.
static cl_mem embed_prologue_dispatch(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    int num_codebooks, int vocab, int hidden, int step, const int32_t* host_ids,
    cl_mem out_buf, int out_rows);

std::vector<float> model_forward_graph_cfg_m2_impl(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    int start_pos,
    float guidance,
    int32_t* out_sampled_ids,      // non-null → Stage-3 fused-readout sampling
    float sp_temperature, int sp_top_k, uint32_t sp_seed, int sp_force_argmax)
{
    NNOPT_CHECKPOINT("model_forward_graph_cfg_m2 entry");
    // Record pass: the recording-build re-issue redirects every dispatch to the
    // RECORDABLE queue (captured, not executed).
    cl_command_queue queue = g_queue_override ? g_queue_override : cl_ctx.queue();
    cl_int err = CL_SUCCESS;

    // CFG-EARLY single-row mode: guidance <= 1 runs ONLY the cond row through
    // the same fast machinery (num_wg=1, M=1 lm_heads). KV bank 0 continues
    // seamlessly from the CFG steps; bank 32 (uncond) simply stops growing —
    // valid because guidance never returns within a generation.
    const bool single = (guidance <= 1.0f);
    const int num_rows = single ? 1 : 2;

    const int num_codebooks = MODEL_CONFIG::NUM_CODEBOOKS;
    const int hidden = MODEL_CONFIG::DECODER_HIDDEN_SIZE;
    const int vocab  = MODEL_CONFIG::VOCAB_SIZE;
    const size_t row_bytes = (size_t)hidden * sizeof(nnopt_storage_t);

    // ── Recordable-replay candidacy ─────────────────────────────────────────
    // (mega_n / mega_perrow moved up here from the layer loop — the replay path
    // needs them before any per-step work.)
    static const int mega_n = [](){ const char* e = std::getenv("NNOPT_MEGA_LAYERS"); return e ? std::atoi(e) : MODEL_CONFIG::NUM_HIDDEN_LAYERS; }();
    static const int s_mega_dump_layer = [](){ const char* e = std::getenv("NNOPT_MEGA_DUMP_LAYER"); return e ? std::atoi(e) : -1; }();
    static const bool mega_perrow = [](){ const char* e = std::getenv("NNOPT_MEGA_PERROW"); return e && e[0]=='1'; }();
    const bool gpu_grid = (g_grid_dev != nullptr);
    // All per-step cl_mems persistent (required for replay; also kills ~5
    // alloc/free + 2 copies per step on the normal live path).
    const bool persist_x = gpu_grid && (mega_n >= MODEL_CONFIG::NUM_HIDDEN_LAYERS) && !mega_perrow;
    static const bool rec_env_on = [](){ const char* e = std::getenv("NNOPT_RECORD"); return !(e && e[0]=='0'); }();
    static const bool rec_dbg_off = [](){
        // Any layer-dump / debug mode changes the dispatch stream — never record.
        if (std::getenv("NNOPT_MEGA_DUMP_LAYER")) return false;
        const char* d = std::getenv("NNOPT_DEBUG_LAYERS");
        return !(d && d[0] == '1');
    }();
    const int geo = single ? 1 : 0;
    static const bool rec_pipeline_on = [](){
        const char* e = std::getenv("NNOPT_PIPELINE"); return !(e && e[0] == '0');
    }();
    const bool rec_candidate = rec_env_on && rec_dbg_off && persist_x
        && rec_pipeline_on && (out_sampled_ids != nullptr) && !g_rec_disabled
        && cl_ctx.has_recordable_queues();

    // ── Per-step params write (skipped during the record pass: FillBuffer is
    // not an NDRange and must not land in the recording; the value is already
    // correct from the live step being re-issued) ───────────────────────────
    if (!g_queue_override) {
        if (!mega_write_step_params(cl_ctx, queue, start_pos)) {
            NNOPT_ERROR("m2: step-params write failed"); return {};
        }
    }

    // ── Replay fast path: the whole step is TWO driver calls (params fill +
    // recording enqueue, ZERO arg overrides — driver-proven combination) ────
    if (rec_candidate && g_recording[geo] && !g_queue_override) {
        cl_int re = cl_ctx.enqueue_recording(queue, g_recording[geo], 0, nullptr);
        if (re == CL_SUCCESS) {
            clFlush(queue);
            for (int c = 0; c < num_codebooks; ++c) out_sampled_ids[c] = -1;  // pipeline sentinel
            return std::vector<float>{0.0f};
        }
        NNOPT_ERROR_FMT("record: replay failed (%d) — disabling, live dispatch", (int)re);
        g_rec_disabled = true;
    }

    // Both CFG rows share the same input ids — build the codebook id list once.
    const int32_t decoder_start_id = (int32_t)MODEL_CONFIG::BOS_TOKEN_ID;
    std::vector<int32_t> ids(num_codebooks, decoder_start_id);
    if ((int)input_ids.size() >= num_codebooks)
        for (int c = 0; c < num_codebooks; ++c) ids[(size_t)c] = input_ids[(size_t)c];

    cl_program utils_prog = cl_ctx.get_utils_program();
    if (!utils_prog) { NNOPT_ERROR("m2: utils_program null"); return {}; }

    // ── Stage 4+5: GPU-resident grid + fused embed prologue ────────────────
    // When the device grid is active (g_grid_dev set by mega_set_decode_grid),
    // ONE embed_prologue dispatch reads the 4 ids from grid[:,start_pos], sums
    // their embedding rows, and adds the positional row — replacing 4 embed
    // kernels + 4 id uploads + 3 element_add + 1 pos-add. No per-step host id
    // upload; ids come straight from the GPU grid.
    // ── Persistent [2,hidden] activation x (persist_x path) ────────────────
    // embed_prologue writes BOTH rows of x directly (out_rows=num_rows) — no
    // emb scratch, no emb→x CopyBuffers (which a recording cannot capture).
    static cl_mem s_x2 = nullptr;
    if (persist_x) {
        if (!s_x2) {
            s_x2 = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 2 * row_bytes, nullptr, &err);
            if (err != CL_SUCCESS || !s_x2) { NNOPT_ERROR("m2: persistent x alloc"); s_x2 = nullptr; return {}; }
        }
        if (!embed_prologue_dispatch(cl_ctx, weights, queue, num_codebooks, vocab,
                                     hidden, start_pos, /*host_ids=*/nullptr,
                                     /*out_buf=*/s_x2, /*out_rows=*/num_rows)) {
            NNOPT_ERROR("m2: embed_prologue (persist) failed"); return {};
        }
    }

    cl_mem emb = nullptr;
    if (persist_x) {
        // handled above — x is s_x2.
    } else if (gpu_grid) {
        emb = embed_prologue_dispatch(cl_ctx, weights, queue, num_codebooks, vocab,
                                      hidden, start_pos, /*host_ids=*/nullptr,
                                      /*out_buf=*/nullptr, /*out_rows=*/1);
        if (!emb) { NNOPT_ERROR("m2: embed_prologue failed"); return {}; }
    } else
    // ── Embeddings (sum over codebooks), computed once at M=1 ─────────────
    {
        cl_mem ev[4] = {nullptr,nullptr,nullptr,nullptr};
        cl_mem idbuf[4] = {nullptr,nullptr,nullptr,nullptr};
        bool ok = true;
        for (int c = 0; c < num_codebooks && ok; ++c) {
            idbuf[c] = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(int32_t), (void*)&ids[(size_t)c], &err);
            if (err != CL_SUCCESS || !idbuf[c]) { NNOPT_ERROR_FMT("m2: id%d upload %d", c, (int)err); ok = false; break; }
            char ek[64]; snprintf(ek, sizeof(ek), "decoder.model.decoder.embed_tokens.%d", c);
            ev[c] = Embedding_forward(cl_ctx, weights, queue, idbuf[c], 1, -1, start_pos, nullptr, nullptr, nullptr, ek);
            if (!ev[c]) { NNOPT_ERROR_FMT("m2: embed %d null", c); ok = false; }
        }
        if (ok) {
            cl_mem s01 = element_add(queue, utils_prog, ev[0], ev[1], (size_t)hidden);
            cl_mem s23 = element_add(queue, utils_prog, ev[2], ev[3], (size_t)hidden);
            if (s01 && s23) emb = element_add(queue, utils_prog, s01, s23, (size_t)hidden);
            if (s01) clReleaseMemObject(s01);
            if (s23) clReleaseMemObject(s23);
        }
        for (int c = 0; c < num_codebooks; ++c) { if (ev[c]) clReleaseMemObject(ev[c]); if (idbuf[c]) clReleaseMemObject(idbuf[c]); }
        if (!emb) { NNOPT_ERROR("m2: embedding sum failed"); return {}; }
    }

    // ── Positional add (sinusoidal table, abs pos = start_pos) ────────────
    // Skipped in gpu_grid mode — embed_prologue already folded the pos row in.
    if (!gpu_grid) {
        cl_mem pos_w = weights.get_buffer("decoder.model.decoder.embed_positions.weights");
        if (!pos_w) { clReleaseMemObject(emb); NNOPT_ERROR("m2: missing embed_positions"); return {}; }
        if (!element_add_offset(queue, utils_prog, emb, pos_w, (size_t)start_pos * hidden, (size_t)hidden)) {
            clReleaseMemObject(emb); NNOPT_ERROR("m2: pos add failed"); return {}; }
    }

    // ── Build [2,hidden] activation buffer: row0=row1=emb (identical input) ─
    // persist_x: x IS the persistent s_x2 (embed already wrote both rows).
    cl_mem x = nullptr;
    if (persist_x) {
        x = s_x2;
    } else {
        x = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 2*row_bytes, nullptr, &err);
        if (err != CL_SUCCESS) { clReleaseMemObject(emb); NNOPT_ERROR("m2: x alloc"); return {}; }
        err  = clEnqueueCopyBuffer(queue, emb, x, 0, 0,          row_bytes, 0, nullptr, nullptr);
        err |= clEnqueueCopyBuffer(queue, emb, x, 0, row_bytes,  row_bytes, 0, nullptr, nullptr);
        clReleaseMemObject(emb);
        if (err != CL_SUCCESS) { clReleaseMemObject(x); NNOPT_ERROR("m2: x dup copy"); return {}; }
    }

    // ── MEGAKERNEL gating ─────────────────────────────────────────────────
    // NNOPT_MEGA_LAYERS=n routes layers [0,n) through the per-layer megakernel
    // (one dispatch/layer/row, fp32 accum), the rest stay on the M=2 path.
    // Precompute cross-attn K/V once per generation (kills per-step recompute).
    // Default = ALL layers (the validated shipped state; 2026-06-04 Step-0
    // found the old default 0 left the binary on the 0.31 tok/s M=2 path —
    // every "shipped" optimization was env-gated off). NNOPT_MEGA_LAYERS=0
    // reverts to the M=2 baseline for A/B.
    // (mega_n / s_mega_dump_layer / mega_perrow are defined at the top of this
    // function — the recordable-replay candidacy needs them before any work.)
    // NNOPT_MEGA_PERROW=1 routes mega layers through the per-row dispatch
    // (extract row → mega(1 WG) → writeback, ×2 rows) for controlled same-binary
    // A/B of the Stage-1 M=2-native megakernel vs the prior per-row megakernel
    // under identical thermal state (BENCHMARKS A/B methodology). The per-row
    // path dispatches the SAME M=2 kernel restricted to ONE workgroup with both
    // row slots bound to the single-row scratch buffers (selecting one CFG row's
    // KV bank + cross-K/V), so the kernel body is byte-identical between modes.
    static cl_mem mega_row_in[2]  = {nullptr, nullptr};
    static cl_mem mega_row_out[2] = {nullptr, nullptr};
    auto mega_scratch = [&](cl_mem& slot) -> cl_mem {
        if (!slot) { cl_int e=CL_SUCCESS; slot = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, row_bytes, nullptr, &e); if (e!=CL_SUCCESS) slot=nullptr; }
        return slot;
    };
    if (mega_n > 0) {
        if (!mega_precompute_cross_kv(cl_ctx, weights, queue, g_enc_states, g_enc_zero, g_musicgen_enc_len)) {
            if (!persist_x) clReleaseMemObject(x); NNOPT_ERROR("m2: mega cross-kv precompute failed"); return {}; }
    }
    // ── Stage 1: M=2-native megakernel sub-buffer views into x[2,hidden] ──
    // The megakernel reads/writes the two rows IN PLACE (no host extract/
    // writeback copies). Each row is a sub-buffer view of x at offset r*row_bytes.
    // Sub-buffers are host objects (no GPU dispatch). persist_x: x is the
    // process-lifetime s_x2, so the views are created ONCE and reused every
    // step (the recordable replay bakes these handles). Legacy path: x is
    // freshly allocated each step → views recreated per call. They alias x,
    // so in-place (x_in==x_out) is safe: the kernel loads the full residual
    // into local memory before any store (see decoder_layer_mega:
    // `for c: residual[c]=LOAD(x_in,c)` then a barrier precedes every write).
    static cl_mem s_x2_row[2] = {nullptr, nullptr};
    cl_mem x_row0 = nullptr, x_row1 = nullptr;
    if (mega_n > 0 && !mega_perrow) {
        if (persist_x && s_x2_row[0]) {
            x_row0 = s_x2_row[0]; x_row1 = s_x2_row[1];
        } else {
            cl_buffer_region r0 = {0, row_bytes};
            cl_buffer_region r1 = {row_bytes, row_bytes};
            x_row0 = clCreateSubBuffer(x, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r0, &err);
            if (err == CL_SUCCESS)
                x_row1 = clCreateSubBuffer(x, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r1, &err);
            if (err != CL_SUCCESS || !x_row0 || !x_row1) {
                if (x_row0) clReleaseMemObject(x_row0);
                if (x_row1) clReleaseMemObject(x_row1);
                if (!persist_x) clReleaseMemObject(x);
                NNOPT_ERROR_FMT("mega: x sub-buffer %d", (int)err); return {};
            }
            if (persist_x) { s_x2_row[0] = x_row0; s_x2_row[1] = x_row1; }
        }
    }

    // ── Decoder layers (24) ───────────────────────────────────────────────
    for (int layer_idx = 0; layer_idx < MODEL_CONFIG::NUM_HIDDEN_LAYERS; ++layer_idx) {
        char wp[96]; snprintf(wp, sizeof(wp), "decoder.model.decoder.layers.%d", layer_idx);

        if (layer_idx < mega_n) {
            if (mega_perrow) {
                // A/B baseline: per-row dispatch (extract row r → mega(1 WG) →
                // writeback). Uses the SAME M=2 kernel with 1 WG; binding both
                // row slots to the single-row scratch + the row's KV/cross bank.
                bool ok = true;
                for (int r = 0; r < 2 && ok; ++r) {
                    cl_mem rin = mega_scratch(mega_row_in[r]);
                    cl_mem rout = mega_scratch(mega_row_out[r]);
                    if (!rin || !rout) { ok = false; break; }
                    if (clEnqueueCopyBuffer(queue, x, rin, (size_t)r*row_bytes, 0, row_bytes, 0, nullptr, nullptr) != CL_SUCCESS) { ok = false; break; }
                    cl_mem* kc = (r == 0) ? &g_k_cache[0 + layer_idx] : &g_k_cache[32 + layer_idx];
                    cl_mem* vc = (r == 0) ? &g_v_cache[0 + layer_idx] : &g_v_cache[32 + layer_idx];
                    if (!mega_decoder_layer_m2_n(cl_ctx, weights, queue, rin, rout, rin, rout,
                            layer_idx, start_pos, kc, vc, kc, vc, /*num_wg=*/1, /*single_row=*/r)) { ok = false; break; }
                    if (clEnqueueCopyBuffer(queue, rout, x, 0, (size_t)r*row_bytes, row_bytes, 0, nullptr, nullptr) != CL_SUCCESS) { ok = false; break; }
                }
                if (!ok) { if (x_row0) clReleaseMemObject(x_row0); if (x_row1) clReleaseMemObject(x_row1);
                           clReleaseMemObject(x); NNOPT_ERROR_FMT("mega: perrow layer %d failed", layer_idx); return {}; }
                continue;
            }
            // M=2-native megakernel: BOTH CFG rows in ONE dispatch, in place.
            // (single-row CFG-early mode: num_wg=1 — only the cond row runs.)
            if (!mega_decoder_layer_m2_n(cl_ctx, weights, queue,
                    x_row0, x_row0, x_row1, x_row1, layer_idx, start_pos,
                    &g_k_cache[0  + layer_idx], &g_v_cache[0  + layer_idx],   // row0 cond  (bank 0)
                    &g_k_cache[32 + layer_idx], &g_v_cache[32 + layer_idx],   // row1 uncond (bank 32)
                    /*num_wg=*/num_rows, /*single_row=*/0)) {
                if (!persist_x) {
                    clReleaseMemObject(x_row0); clReleaseMemObject(x_row1);
                    clReleaseMemObject(x);
                }
                NNOPT_ERROR_FMT("mega: m2 layer %d failed", layer_idx); return {};
            }
            continue;
        }

        cl_mem out = MusicgenDecoderLayer_forward_m2(
            cl_ctx, weights, queue, x, layer_idx, start_pos,
            &g_k_cache[0  + layer_idx], &g_v_cache[0  + layer_idx],   // row0 = cond  (bank 0)
            &g_k_cache[32 + layer_idx], &g_v_cache[32 + layer_idx],   // row1 = uncond (bank 32)
            g_enc_states, g_enc_zero, wp);
        if (!out) { clReleaseMemObject(x); NNOPT_ERROR_FMT("m2: layer %d null", layer_idx); return {}; }
        if (out != x) { clReleaseMemObject(x); x = out; }

        // ── Incremental-validation dump: row0 of x AFTER this layer (env-gated) ─
        // NNOPT_MEGA_DUMP_LAYER=N dumps row0 right after decoder layer N. Run
        // baseline (NNOPT_MEGA_LAYERS=0) vs mega (NNOPT_MEGA_LAYERS=N+1) at
        // step 0 and cosine-compare layer_dumps/mega_probe_layerN.bin (≥0.999).
        if (layer_idx == s_mega_dump_layer && start_pos == 0) {
            std::vector<nnopt_storage_t> hostbuf((size_t)hidden);
            if (clEnqueueReadBuffer(queue, x, CL_TRUE, 0, (size_t)hidden*sizeof(nnopt_storage_t), hostbuf.data(), 0, nullptr, nullptr) == CL_SUCCESS) {
                char path[128]; snprintf(path, sizeof(path), "layer_dumps/mega_probe_layer%d.bin", layer_idx);
                if (FILE* f = std::fopen(path, "wb")) {
                    std::vector<float> fbuf((size_t)hidden);
                    for (int i = 0; i < hidden; ++i)
#ifdef NNOPT_USE_FP16
                        fbuf[(size_t)i] = nnopt_f16_to_f32((uint16_t)hostbuf[(size_t)i]);
#else
                        fbuf[(size_t)i] = hostbuf[(size_t)i];
#endif
                    std::fwrite(fbuf.data(), sizeof(float), (size_t)hidden, f);
                    std::fclose(f);
                    fprintf(stderr, "MEGA_PROBE_DUMP %s [%d]\n", path, hidden);
                }
            }
        }
    }

    // Release the M=2-native row sub-buffer views (parent x outlives them).
    // persist_x: the views are process-lifetime (cached in s_x2_row) — keep.
    if (!persist_x) {
        if (x_row0) clReleaseMemObject(x_row0);
        if (x_row1) clReleaseMemObject(x_row1);
    }

    // ── Final decoder layer norm (M=2) ────────────────────────────────────
    if (persist_x) {
        // Persistent LN output (the replay path bakes this handle; x = s_x2
        // stays untouched as next step's embed destination).
        static cl_mem s_xn = nullptr;
        if (!s_xn) {
            s_xn = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 2 * row_bytes, nullptr, &err);
            if (err != CL_SUCCESS || !s_xn) { NNOPT_ERROR("m2: persistent ln alloc"); s_xn = nullptr; return {}; }
        }
        if (!LayerNorm_forward_into(cl_ctx, weights, queue, x, /*seq_len=*/num_rows,
                                    "decoder.model.decoder.layer_norm", s_xn)) {
            NNOPT_ERROR("m2: final layer_norm (persist) failed"); return {};
        }
        x = s_xn;
    } else {
        cl_mem normed = LayerNorm_forward(cl_ctx, weights, queue, x, /*seq_len=*/num_rows, -1, start_pos,
                                          nullptr, nullptr, nullptr, "decoder.model.decoder.layer_norm");
        if (!normed) { clReleaseMemObject(x); NNOPT_ERROR("m2: final layer_norm null"); return {}; }
        if (normed != x) { clReleaseMemObject(x); x = normed; }
    }

    // ── lm_heads concat GEMM, M=2 → [2, num_codebooks*vocab] ───────────────
    static cl_mem heads_cat = nullptr;
    if (!heads_cat) {
        std::vector<nnopt_storage_t> cat((size_t)num_codebooks * vocab * hidden);
        bool ok = true;
        for (int cb = 0; cb < num_codebooks && ok; ++cb) {
            char hk[64]; snprintf(hk, sizeof(hk), "decoder.lm_heads.%d.weight", cb);
            const std::vector<float> hw = weights.get_host_vec(hk);
            if ((int)hw.size() != vocab * hidden) { NNOPT_ERROR_FMT("m2: lm_heads bad %s", hk); ok = false; break; }
            for (size_t i = 0; i < hw.size(); ++i)
#ifdef NNOPT_USE_FP16
                cat[(size_t)cb * vocab * hidden + i] = nnopt_f32_to_f16(hw[i]);
#else
                cat[(size_t)cb * vocab * hidden + i] = hw[i];
#endif
        }
        if (ok) {
            cl_int ce = CL_SUCCESS;
            heads_cat = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       cat.size() * sizeof(nnopt_storage_t), cat.data(), &ce);
            if (ce != CL_SUCCESS) { NNOPT_ERROR_FMT("m2: lm_heads upload %d", (int)ce); heads_cat = nullptr; }
        }
    }
    if (!heads_cat) { if (!persist_x) clReleaseMemObject(x); return {}; }

    const int N = num_codebooks * vocab;
    // logits2 = [2, N]: row0 = cond logits, row1 = uncond logits.
    static cl_mem logits2 = nullptr;
    if (!logits2) { logits2 = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)2*N*sizeof(nnopt_storage_t), nullptr, &err); }
    if (!logits2) { if (!persist_x) clReleaseMemObject(x); NNOPT_ERROR("m2: logits2 alloc"); return {}; }
    // lm_heads: own texture GEMV by default (CLBlast GEMM was unprofiled,
    // suspected of internal waits, and compiled outside the binary cache).
    // NNOPT_LMHEADS=clblast reverts. Falls back to CLBlast on dispatch failure.
    static const bool own_heads = [](){
        const char* e = std::getenv("NNOPT_LMHEADS");
        return !(e && e[0] == 'c');
    }();
    bool heads_ok = own_heads &&
        mega_lmheads_dispatch(cl_ctx, queue, heads_cat, x, logits2, num_rows, N);
    if (!heads_ok && !pytorch_linear(queue, /*M=*/num_rows, /*N=*/N, /*K=*/hidden, x, heads_cat, logits2)) {
        if (!persist_x) clReleaseMemObject(x); NNOPT_ERROR("m2: lm_heads GEMM failed"); return {}; }
    if (!persist_x) clReleaseMemObject(x);
    if (single && !out_sampled_ids) {
        // HOST-LOGITS path only: duplicate cond logits into the uncond row so
        // the unchanged cfg_combine_rows computes u + g(c-u) = c EXACTLY.
        // The fused path passes single=1 to sample_grid instead (the kernel
        // reads the cond row directly — exact, and replaces this CopyBuffer,
        // which the recordable replay could not capture).
        err = clEnqueueCopyBuffer(queue, logits2, logits2, 0, (size_t)N * sizeof(nnopt_storage_t),
                                  (size_t)N * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("m2: single-row logits dup %d", (int)err); return {}; }
    }

    // ── Stage 3 (hybrid): on-GPU CFG blend + sample + grid write ───────────
    // logits2 [2,N] is the fast CLBlast lm_heads output. When sampling (and/or
    // GPU-resident grid) is requested, dispatch sample_grid: it blends per
    // codebook, samples (or argmax under force), and writes 4 ids to out_ids
    // (+ the GPU grid with delay forcing). No 8192-fp16 readback; the host reads
    // only 4 int32s (and, in Stage-4 GPU-grid mode, nothing per step).
    if (out_sampled_ids) {
        bool sok = sample_grid_dispatch(cl_ctx, queue, logits2, num_codebooks, vocab,
                                        guidance, sp_temperature, sp_top_k, sp_seed,
                                        start_pos, sp_force_argmax, out_sampled_ids,
                                        single ? 1 : 0);
        if (!sok) { NNOPT_ERROR("m2: sample_grid failed"); return {}; }

        // ── Recordable-queue capture ────────────────────────────────────────
        // After the SECOND+ live step of this geometry (all lazy first-use init
        // — KV caches, weight images, scratch, persistent x — is done and every
        // cl_mem handle is now stable), re-issue this exact step to the
        // recordable queue (captured, NOT executed) and keep the recording.
        // Subsequent steps of the same geometry take the replay fast path.
        if (rec_candidate && !g_recording[geo] && !g_queue_override
            && g_geo_live_steps[geo] >= 1 && mega_record_supported()) {
            g_rec_ctx = &cl_ctx;
            if (!g_rec_queue) g_rec_queue = cl_ctx.create_recordable_queue();
            if (!g_rec_queue) {
                g_rec_disabled = true;
            } else if (nnopt_recording_t rec = cl_ctx.new_recording(g_rec_queue)) {
                KernelProfiler::suppress(true);   // recorded dispatches never execute
                g_queue_override = g_rec_queue;
                int32_t dummy_ids[8] = {};
                std::vector<float> rr = model_forward_graph_cfg_m2_impl(
                    cl_ctx, weights, input_ids, start_pos, guidance,
                    dummy_ids, sp_temperature, sp_top_k, sp_seed, sp_force_argmax);
                g_queue_override = nullptr;
                KernelProfiler::suppress(false);
                cl_int ee = cl_ctx.end_recording(rec);
                if (rr.empty() || ee != CL_SUCCESS) {
                    NNOPT_ERROR_FMT("record: capture failed (geo=%d end=%d) — staying live", geo, (int)ee);
                    cl_ctx.release_recording(rec);
                    g_rec_disabled = true;
                } else {
                    g_recording[geo] = rec;
                    fprintf(stderr, "RECORD: %s geometry captured at step %d — replaying from next step\n",
                            geo ? "single-row" : "CFG-2", start_pos);
                }
            } else {
                g_rec_disabled = true;
            }
        }
        if (!g_queue_override) g_geo_live_steps[geo]++;
        return std::vector<float>{0.0f};   // sentinel: ids in out_sampled_ids (+ GPU grid)
    }

    // ── CFG combine on-device: out[i] = uncond + g*(cond-uncond) ───────────
    // cond = logits2[0..N), uncond = logits2[N..2N). Reuse cfg_combine with
    // a byte-offset on the uncond pointer via a tiny dedicated kernel arg path:
    // run cfg_combine over the two halves using sub-region copies is overkill;
    // instead use the offset-aware combine by passing the same buffer twice
    // with element offsets baked into a small wrapper kernel. Simpler + proven:
    // copy uncond half down, then cfg_combine(cond, uncond, out).
    static cl_mem combined = nullptr;
    if (!combined) { combined = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)N*sizeof(nnopt_storage_t), nullptr, &err); }
    if (!combined) { NNOPT_ERROR("m2: combined alloc"); return {}; }

    static cl_kernel kc2 = nullptr;
    if (!kc2) { kc2 = clCreateKernel(utils_prog, "cfg_combine_rows", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("m2: cfg_combine_rows kernel %d", (int)err); kc2 = nullptr; } }
    if (!kc2) { return {}; }
    {
        int arg = 0;
        clSetKernelArg(kc2, arg++, sizeof(cl_mem), &logits2);   // [2,N]: row0 cond, row1 uncond
        clSetKernelArg(kc2, arg++, sizeof(cl_mem), &combined);
        clSetKernelArg(kc2, arg++, sizeof(float), &guidance);
        clSetKernelArg(kc2, arg++, sizeof(int), &N);
        size_t gws = (size_t)N;
        err = clEnqueueNDRangeKernel(queue, kc2, 1, nullptr, &gws, nullptr, 0, nullptr,
                                     KernelProfiler::event_for("cfg_combine_rows"));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("m2: cfg_combine_rows dispatch %d", (int)err); return {}; }
    }

    std::vector<float> out((size_t)N, 0.0f);
#ifdef NNOPT_USE_FP16
    std::vector<uint16_t> tmp((size_t)N);
    if (clEnqueueReadBuffer(queue, combined, CL_TRUE, 0, (size_t)N*sizeof(uint16_t), tmp.data(), 0, nullptr, nullptr) == CL_SUCCESS)
        for (int i = 0; i < N; ++i) out[(size_t)i] = nnopt_f16_to_f32(tmp[(size_t)i]);
#else
    clEnqueueReadBuffer(queue, combined, CL_TRUE, 0, (size_t)N*sizeof(float), out.data(), 0, nullptr, nullptr);
#endif
    return out;
}

// Public logits entry (unchanged behavior): runs the decoder + lm_heads + blend
// and returns blended logits. Delegates to the impl with no fused sampling.
std::vector<float> model_forward_graph_cfg_m2(
    OpenCLContext& cl_ctx, Weights& weights,
    const std::vector<int32_t>& input_ids, int start_pos, float guidance)
{
    return model_forward_graph_cfg_m2_impl(cl_ctx, weights, input_ids, start_pos, guidance,
                                           /*out_sampled_ids=*/nullptr, 1.0f, 0, 0u, 0);
}

// Stage-3 entry: runs the decoder then the fused readout, writing 4 sampled
// int32 ids to out_ids[num_codebooks]. Returns true on success.
extern "C" bool model_forward_graph_cfg_m2_sampled(
    OpenCLContext& cl_ctx, Weights& weights,
    const std::vector<int32_t>& input_ids, int start_pos, float guidance,
    float temperature, int top_k, uint32_t seed, int force_argmax,
    int32_t* out_ids)
{
    std::vector<float> r = model_forward_graph_cfg_m2_impl(
        cl_ctx, weights, input_ids, start_pos, guidance,
        out_ids, temperature, top_k, seed, force_argmax);
    return !r.empty();   // impl returns {0.0f} sentinel on success, {} on failure
}

// ── Stage 3 (hybrid) sample-grid helper ─────────────────────────────────────
// Dispatches sample_grid on the [2,N] CLBlast lm_heads logits: on-GPU CFG blend
// + temperature/top-k sample (or argmax under force), writing 4 ids to a GPU
// buffer. In Stage-3 mode (g_grid_dev==null) the host reads the 4 ids back.
// In Stage-4 mode (g_grid_dev set via mega_set_decode_grid) the kernel ALSO
// writes them into the device grid with delay forcing and the per-step readback
// is skipped (the grid is read once at end-of-decode).
extern "C" void mega_set_decode_grid(cl_mem grid_dev, int steps1, int bos) {
    g_grid_dev = grid_dev; g_grid_steps1 = steps1; g_grid_bos = bos;
    // New generation (fresh grid buffer): recorded dispatches baked the OLD
    // grid handle — invalidate so the next steps re-record against this one.
    record_invalidate();
}

// Stage-4 grid lifecycle (C API for the host decode loop).
// Allocates the device grid [num_codebooks, steps1] int32, fills it with the
// host grid (col 0 = BOS, all delay-window cells = BOS), registers it, and
// returns the cl_mem (host owns release via mega_free_decode_grid).
extern "C" cl_mem mega_alloc_decode_grid(
    OpenCLContext& cl_ctx, const int32_t* host_grid, int num_codebooks, int steps1, int bos) {
    cl_int err = CL_SUCCESS;
    cl_mem g = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              (size_t)num_codebooks * steps1 * sizeof(int32_t),
                              (void*)host_grid, &err);
    if (err != CL_SUCCESS || !g) { NNOPT_ERROR_FMT("grid alloc %d", (int)err); return nullptr; }
    mega_set_decode_grid(g, steps1, bos);
    return g;
}

// Read the whole device grid back into host_grid [num_codebooks, steps1].
extern "C" bool mega_read_decode_grid(
    OpenCLContext& cl_ctx, cl_mem grid_dev, int32_t* host_grid, int num_codebooks, int steps1) {
    if (!grid_dev) return false;
    return clEnqueueReadBuffer(cl_ctx.queue(), grid_dev, CL_TRUE, 0,
                               (size_t)num_codebooks * steps1 * sizeof(int32_t),
                               host_grid, 0, nullptr, nullptr) == CL_SUCCESS;
}

// Streaming-EnCodec support: NON-blocking read of grid columns [col0, col1)
// for all codebook rows. In-order queue → the read executes after every
// kernel enqueued so far (i.e., after step col1-2's sample), while the GPU
// continues with later, already-enqueued steps. The event (from the last of
// the 4 row-segment reads) fires when the data is in host_grid. Caller owns
// the event (clReleaseEvent).
extern "C" bool mega_read_decode_grid_cols_async(
    OpenCLContext& cl_ctx, cl_mem grid_dev, int32_t* host_grid,
    int num_codebooks, int steps1, int col0, int col1, cl_event* evt_out) {
    if (!grid_dev || col1 <= col0) return false;
    cl_command_queue q = cl_ctx.queue();
    for (int k = 0; k < num_codebooks; ++k) {
        const size_t off = ((size_t)k * steps1 + col0) * sizeof(int32_t);
        const size_t len = (size_t)(col1 - col0) * sizeof(int32_t);
        cl_event* ev = (k == num_codebooks - 1) ? evt_out : nullptr;
        if (clEnqueueReadBuffer(q, grid_dev, CL_FALSE, off, len,
                                host_grid + (size_t)k * steps1 + col0,
                                0, nullptr, ev) != CL_SUCCESS) {
            NNOPT_ERROR_FMT("grid cols async read failed (k=%d)", k);
            return false;
        }
    }
    clFlush(q);
    return true;
}

extern "C" void mega_free_decode_grid(cl_mem grid_dev) {
    if (g_grid_dev == grid_dev) mega_set_decode_grid(nullptr, 0, 0);
    if (grid_dev) clReleaseMemObject(grid_dev);
}

static bool sample_grid_dispatch(
    OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem logits2,
    int num_codebooks, int vocab, float guidance, float temperature, int top_k,
    uint32_t seed, int step, int force_argmax, int32_t* out_ids, int single)
{
    cl_int err = CL_SUCCESS;
    if (!s_sg_prog) {
        s_sg_prog = cl_ctx.build_program_from_file("kernels/sample_grid.cl"); // PROGRAM-INIT-OK
        if (!s_sg_prog) { NNOPT_ERROR("sample_grid: build failed"); return false; }
        s_sg_kernel = clCreateKernel(s_sg_prog, "sample_grid", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("sample_grid: clCreateKernel %d", (int)err); return false; }
    }
    if (!s_sg_ids) {
        s_sg_ids = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                  (size_t)num_codebooks * sizeof(int32_t), nullptr, &err);
        if (err != CL_SUCCESS || !s_sg_ids) { NNOPT_ERROR_FMT("sample_grid: ids alloc %d", (int)err); s_sg_ids = nullptr; return false; }
    }

    const int write_grid = (g_grid_dev != nullptr) ? 1 : 0;
    cl_mem grid = g_grid_dev ? g_grid_dev : s_sg_ids;   // arg must be a valid cl_mem
    const int steps1 = g_grid_steps1;
    const int bos = g_grid_bos;

    cl_kernel kk = s_sg_kernel;
    int a = 0;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &logits2, "logits2")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &s_sg_ids, "out_ids")) return false;
    if (!set_arg_checked(kk, a++, sizeof(float), &guidance, "guidance")) return false;
    if (!set_arg_checked(kk, a++, sizeof(float), &temperature, "temperature")) return false;
    if (!set_arg_checked(kk, a++, sizeof(int), &top_k, "top_k")) return false;
    if (!set_arg_checked(kk, a++, sizeof(uint32_t), &seed, "seed")) return false;
    cl_mem sp_buf = mega_step_params_buf();   // sp[0]=step (written this step)
    if (!sp_buf) { NNOPT_ERROR("sample_grid: step-params buffer missing"); return false; }
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &sp_buf, "sp")) return false;
    (void)step;   // value travels via the step-params buffer
    if (!set_arg_checked(kk, a++, sizeof(int), &force_argmax, "force_argmax")) return false;
    if (!set_arg_checked(kk, a++, sizeof(int), &vocab, "vocab")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &grid, "grid")) return false;
    if (!set_arg_checked(kk, a++, sizeof(int), &write_grid, "write_grid")) return false;
    if (!set_arg_checked(kk, a++, sizeof(int), &steps1, "steps1")) return false;
    if (!set_arg_checked(kk, a++, sizeof(int), &bos, "bos")) return false;
    if (!set_arg_checked(kk, a++, sizeof(int), &single, "single")) return false;

    const size_t gws[1] = {(size_t)num_codebooks * 256};
    const size_t lws[1] = {256};
    err = clEnqueueNDRangeKernel(queue, kk, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("sample_grid: dispatch %d", (int)err); return false; }

    // PIPELINE MODE (NNOPT_PIPELINE=1 + device grid): SKIP the blocking 16 B
    // ids read — this read was THE per-step bubble (GPU drains → host wakes →
    // re-enqueues → GPU restarts ≈ tens of ms/step measured as the wall-vs-
    // kernel-time gap). With the grid GPU-resident the host needs NOTHING per
    // step; it enqueues the whole decode ahead and reads the grid ONCE at the
    // end. clFlush keeps the GPU fed without draining.
    static const bool pipeline = [](){
        const char* e = std::getenv("NNOPT_PIPELINE");
        return !(e && e[0] == '0');
    }();
    if (pipeline && g_grid_dev) {
        // Record pass: queue is the RECORDABLE queue — no flush (nothing to
        // submit; the dispatches are being captured, not executed).
        if (queue != g_rec_queue) clFlush(queue);
        for (int c = 0; c < num_codebooks; ++c) out_ids[c] = -1;   // sentinel: not read
        return true;
    }
    // Stage 3 (no device grid): read the 4 ids back. Stage 4 (device grid):
    // still read them so the host can log STEP_ARGMAX + advance generated_ids.
    if (clEnqueueReadBuffer(queue, s_sg_ids, CL_TRUE, 0,
                            (size_t)num_codebooks * sizeof(int32_t), out_ids, 0, nullptr, nullptr) != CL_SUCCESS) {
        NNOPT_ERROR("sample_grid: ids readback failed"); return false;
    }
    return true;
}

// ── Stage 4+5 fused embed prologue ──────────────────────────────────────────
// Builds (once) the concatenated embedding table [num_codebooks*vocab, hidden]
// (cb-major) + a process-lifetime [hidden] output buffer, then dispatches the
// embed_prologue kernel: reads 4 ids (from the GPU grid at column `step`, or
// host_ids), sums their embedding rows, adds the sinusoidal pos row at `step`.
// Returns a retained [hidden] cl_mem (caller releases). nullptr on failure.
// (Program/kernel/scratch statics are declared above m2_impl — the replay path
// overrides this kernel's `step` arg by handle.)
static cl_mem embed_prologue_dispatch(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    int num_codebooks, int vocab, int hidden, int step, const int32_t* host_ids,
    cl_mem out_buf, int out_rows)
{
    cl_int err = CL_SUCCESS;
    if (!s_emb_prog) {
        s_emb_prog = cl_ctx.build_program_from_file("kernels/embed_prologue.cl"); // PROGRAM-INIT-OK
        if (!s_emb_prog) { NNOPT_ERROR("embed_prologue: build failed"); return nullptr; }
        s_emb_kernel = clCreateKernel(s_emb_prog, "embed_prologue", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("embed_prologue: clCreateKernel %d", (int)err); return nullptr; }
    }
    // MusicGen embed tables are [embed_rows, hidden] with embed_rows = vocab+1
    // (the extra row is the BOS/pad token id = vocab). The id can equal BOS, so
    // the concatenated table MUST carry all embed_rows per codebook.
    static int s_embed_rows = 0;
    if (!s_emb_tables) {
        const std::vector<int> sh0 = weights.get_shape("decoder.model.decoder.embed_tokens.0.weight");
        if (sh0.size() != 2 || sh0[1] != hidden) { NNOPT_ERROR_FMT("embed_prologue: bad embed shape rank=%zu", sh0.size()); return nullptr; }
        s_embed_rows = sh0[0];
        std::vector<nnopt_storage_t> cat((size_t)num_codebooks * s_embed_rows * hidden);
        for (int cb = 0; cb < num_codebooks; ++cb) {
            char ek[64]; snprintf(ek, sizeof(ek), "decoder.model.decoder.embed_tokens.%d.weight", cb);
            const std::vector<float> ew = weights.get_host_vec(ek);
            if ((int)ew.size() != s_embed_rows * hidden) { NNOPT_ERROR_FMT("embed_prologue: bad %s (%zu)", ek, ew.size()); return nullptr; }
            for (size_t i = 0; i < ew.size(); ++i)
#ifdef NNOPT_USE_FP16
                cat[(size_t)cb * s_embed_rows * hidden + i] = nnopt_f32_to_f16(ew[i]);
#else
                cat[(size_t)cb * s_embed_rows * hidden + i] = ew[i];
#endif
        }
        s_emb_tables = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      cat.size() * sizeof(nnopt_storage_t), cat.data(), &err);
        if (err != CL_SUCCESS || !s_emb_tables) { NNOPT_ERROR_FMT("embed_prologue: tables upload %d", (int)err); s_emb_tables = nullptr; return nullptr; }
    }
    if (!s_emb_out) {
        s_emb_out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                   (size_t)hidden * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS || !s_emb_out) { NNOPT_ERROR_FMT("embed_prologue: out alloc %d", (int)err); s_emb_out = nullptr; return nullptr; }
    }

    cl_mem pos_w = weights.get_buffer("decoder.model.decoder.embed_positions.weights");
    if (!pos_w) { NNOPT_ERROR("embed_prologue: missing embed_positions"); return nullptr; }

    const int use_grid = host_ids ? 0 : 1;
    cl_mem grid = g_grid_dev;
    if (use_grid && !grid) { NNOPT_ERROR("embed_prologue: grid mode but no device grid"); return nullptr; }
    // host_ids fallback buffer (rarely used).
    if (host_ids) {
        if (!s_emb_host_ids)
            s_emb_host_ids = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)num_codebooks*sizeof(int32_t), nullptr, &err);
        clEnqueueWriteBuffer(queue, s_emb_host_ids, CL_FALSE, 0, (size_t)num_codebooks*sizeof(int32_t), host_ids, 0, nullptr, nullptr);
    }
    cl_mem grid_arg = grid ? grid : s_emb_out;            // valid cl_mem placeholder
    cl_mem hid_arg  = s_emb_host_ids ? s_emb_host_ids : s_emb_out;
    const int steps1 = g_grid_steps1;

    cl_mem dst = out_buf ? out_buf : s_emb_out;
    const int rows = out_buf ? out_rows : 1;

    cl_kernel kk = s_emb_kernel;
    int a = 0;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &s_emb_tables, "emb_tables")) return nullptr;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &pos_w, "pos_w")) return nullptr;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &dst, "out")) return nullptr;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &grid_arg, "grid")) return nullptr;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &hid_arg, "host_ids")) return nullptr;
    if (!set_arg_checked(kk, a++, sizeof(int), &use_grid, "use_grid")) return nullptr;
    if (!set_arg_checked(kk, a++, sizeof(int), &steps1, "steps1")) return nullptr;
    cl_mem sp_buf = mega_step_params_buf();   // sp[0]=step (written this step)
    if (!sp_buf) { NNOPT_ERROR("embed_prologue: step-params buffer missing"); return nullptr; }
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &sp_buf, "sp")) return nullptr;
    (void)step;   // value travels via the step-params buffer
    if (!set_arg_checked(kk, a++, sizeof(int), &num_codebooks, "num_codebooks")) return nullptr;
    if (!set_arg_checked(kk, a++, sizeof(int), &s_embed_rows, "embed_rows")) return nullptr;
    if (!set_arg_checked(kk, a++, sizeof(int), &hidden, "hidden")) return nullptr;
    if (!set_arg_checked(kk, a++, sizeof(int), &rows, "out_rows")) return nullptr;

    const size_t gws[1] = {256};
    const size_t lws[1] = {256};
    err = clEnqueueNDRangeKernel(queue, kk, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("embed_prologue: dispatch %d", (int)err); return nullptr; }
    if (out_buf) return out_buf;    // caller-owned persistent x; NOT retained
    clRetainMemObject(s_emb_out);   // caller releases (matches the non-grid emb path lifetime)
    return s_emb_out;
}
