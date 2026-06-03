// Auto-generated backbone for openai/whisper-tiny.
// Reference: docs/MODALITY_ASR.md (enc-dec split; encoder runs once, decoder runs per step)
//
// NOTE: This file is edited for ASR bring-up. It provides BOTH signatures:
//   - model_forward_graph(..., input_features, start_pos)  (ASR graph)
//   - model_forward_graph(..., start_pos)                  (wrapper to satisfy forward_dispatch.h)
//
// Correctness-first: encoder is run every call currently (no encode/decode split
// in Model yet). This is slower but keeps the model runnable while ops converge.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"

#include "forward_dispatch_audio.h"  // ForwardDispatch::get_input_features

#include <CL/cl.h>
#include <cstdio>
#include <vector>

extern "C" cl_mem WhisperEncoderLayer_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem WhisperDecoderLayer_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem WhisperEncoderFrontend_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem LayerNorm_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem Linear_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem GELUActivation_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem WhisperSdpaAttention_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" void WhisperSdpaAttention_reset_caches();

// Encoder-output cache (keyed by the input_features buffer pointer). File-scope
// so streaming mode can force a recompute between sliding windows: it reuses one
// feats buffer pointer across windows, which would otherwise false-hit this
// cache and return a stale encoder output. Invalidating before each window's
// prefill guarantees a MISS (which also re-runs WhisperSdpaAttention_reset_caches
// via the miss path), so each window encodes its own audio. The normal
// --audio/--audio-list path never calls the invalidator and is unaffected.
static cl_mem s_enc_cache = nullptr;
static cl_mem s_enc_key   = nullptr;
extern "C" void WhisperBackbone_invalidate_encoder_cache() {
    if (s_enc_cache) { clReleaseMemObject(s_enc_cache); s_enc_cache = nullptr; }
    s_enc_key = nullptr;
}

extern "C" cl_mem Embedding_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);

static std::vector<float> _asr_forward_graph(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    cl_mem input_features,
    int start_pos)
{
    NNOPT_CHECKPOINT("model_forward_graph(asr) entry");
    cl_command_queue queue = cl_ctx.queue();
    const int seq_len = (int)input_ids.size();

    if (!input_features) {
        NNOPT_ERROR("model_forward_graph(asr): input_features is null (call ForwardDispatch::set_input_features)");
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    }

    cl_int err = CL_SUCCESS;
    cl_mem ids_buf = clCreateBuffer(cl_ctx.context(),
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        (size_t)seq_len * sizeof(int32_t), (void*)input_ids.data(), &err);
    if (err != CL_SUCCESS || !ids_buf) {
        NNOPT_ERROR_FMT("failed to upload input_ids (err=%d)", (int)err);
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    }

    // Encoder-output cache: the encoder output depends ONLY on input_features
    // (the mel spectrogram), which is identical across every decode step of a
    // clip. Recomputing the full encoder per step was the RTF killer (TTFT ≈ one
    // full encode, then repeated for every generated token). Cache it keyed on
    // the input_features buffer pointer — main.cpp allocates that buffer once per
    // clip, so a matching pointer means "same audio, reuse the encoder output".
    // On a hit we skip the entire frontend + encoder layers + final norm below.
    cl_mem x = nullptr;
    cl_mem encoder_hidden_states = nullptr;
    // s_enc_cache / s_enc_key are file-scope (see WhisperBackbone_invalidate_encoder_cache).
    if (s_enc_cache && s_enc_key == input_features) {
        encoder_hidden_states = s_enc_cache;
        clRetainMemObject(encoder_hidden_states);  // balance this call's release
    } else {
    // Encoder cache MISS = a new clip's audio. The per-layer cross/self attention
    // KV caches (keyed by weight_prefix) belong to the PREVIOUS clip and are now
    // stale — clear them so this clip recomputes from its own encoder output.
    // No-op on the first clip. Enables one process to transcribe many clips
    // (amortizing the per-process JIT compile across them). (lever #1)
    WhisperSdpaAttention_reset_caches();
    // Encoder sequence length, derived from the input_features buffer so the
    // encoder runs on the ACTUAL audio length (streaming short windows) instead of
    // the padded 30s. Batch/eval feeds a 3000-frame mel → enc_T = 1500 → every
    // size/seq below is byte-identical to before. Mirrors the frontend's formula.
    int enc_T = MODEL_CONFIG::MAX_SOURCE_POSITIONS;  // 1500 (full 30s) by default
    {
        const int srcCapFrames = MODEL_CONFIG::MAX_SOURCE_POSITIONS * 2;  // 3000
        size_t _fb = 0;
        if (clGetMemObjectInfo(input_features, CL_MEM_SIZE, sizeof(_fb), &_fb, nullptr) == CL_SUCCESS && _fb > 0) {
            int frames = (int)(_fb / ((size_t)MODEL_CONFIG::NUM_MEL_BINS * sizeof(nnopt_storage_t)));
            if (frames > 0 && frames <= srcCapFrames) enc_T = (frames & ~1) / 2;
        }
    }
    // Encoder frontend (Conv1d×2 + GELU + permute + pos add) → [enc_T, 384]
    x = WhisperEncoderFrontend_forward(
        cl_ctx, weights, queue,
        input_features,
        /*seq_len=*/enc_T * 2,
        /*layer_idx=*/-1, /*start_pos=*/0,
        nullptr, nullptr,
        nullptr,
        "model.encoder");
    if (!x) {
        NNOPT_ERROR("WhisperEncoderFrontend_forward failed");
        clReleaseMemObject(ids_buf);
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    }

    // Residual-add support: a Whisper encoder layer has TWO residual adds
    // (post-self-attn and post-MLP). The inline ops below were missing them.
    static cl_program s_enc_utils_prog = nullptr;
    if (!s_enc_utils_prog) {
        s_enc_utils_prog = cl_ctx.build_program_from_file("kernels/utils.cl");
        if (!s_enc_utils_prog) {
            NNOPT_ERROR("backbone: failed to build kernels/utils.cl for encoder residuals");
            clReleaseMemObject(x); clReleaseMemObject(ids_buf);
            return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
        }
    }
    const size_t enc_n = (size_t)enc_T * (size_t)MODEL_CONFIG::HIDDEN_SIZE;

    // Encoder layers — emit per-node dumps matching reference/forward_graph.json dump_name.
    // This is required for SxSDebug alignment.
    for (int layer_idx = 0; layer_idx < MODEL_CONFIG::ENCODER_LAYERS; layer_idx++) {
        // residual saved BEFORE self_attn_layer_norm (Whisper pre-norm residual).
        cl_mem residual = x;
        clRetainMemObject(residual);  // survive step-1's release of x

        // 1) self_attn_layer_norm_{i}
        {
            char wp_ln[96];
            std::snprintf(wp_ln, sizeof(wp_ln), "model.encoder.layers.%d.self_attn_layer_norm", layer_idx);
            cl_mem out = LayerNorm_forward(
                cl_ctx, weights, queue,
                x, (int)enc_T, layer_idx, /*start_pos=*/0,
                nullptr, nullptr,
                nullptr,
                wp_ln);
            if (!out) {
                NNOPT_ERROR_FMT("LayerNorm_forward(self_attn_layer_norm_%d) failed", layer_idx);
                clReleaseMemObject(x);
                clReleaseMemObject(ids_buf);
                return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
            }
            if (out != x) { clReleaseMemObject(x); x = out; }
            NNOPT_LAYER_CHECK_FMT("self_attn_layer_norm_%d", layer_idx, queue, x,
                                  (size_t)enc_T * (size_t)MODEL_CONFIG::HIDDEN_SIZE);
        }

        // 2) self_attn_{i}
        {
            // Weight prefix for attention module
            char wp_attn[96];
            std::snprintf(wp_attn, sizeof(wp_attn), "model.encoder.layers.%d.self_attn", layer_idx);
            cl_mem out = WhisperSdpaAttention_forward(
                cl_ctx, weights, queue,
                x, (int)enc_T, layer_idx, /*start_pos=*/0,
                nullptr, nullptr,
                nullptr,
                wp_attn);
            if (!out) {
                NNOPT_ERROR_FMT("WhisperSdpaAttention_forward(self_attn_%d) failed", layer_idx);
                clReleaseMemObject(x);
                clReleaseMemObject(ids_buf);
                return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
            }
            if (out != x) { clReleaseMemObject(x); x = out; }
            NNOPT_LAYER_CHECK_FMT("self_attn_%d", layer_idx, queue, x,
                                  (size_t)enc_T * (size_t)MODEL_CONFIG::HIDDEN_SIZE);
        }

        // residual add 1: x = residual + self_attn_out  (this is final_layer_norm's input)
        {
            cl_mem r = element_add(queue, s_enc_utils_prog, residual, x, enc_n);
            clReleaseMemObject(residual);
            if (!r) {
                clReleaseMemObject(x); clReleaseMemObject(ids_buf);
                return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
            }
            if (r != x) clReleaseMemObject(x);
            x = r;
        }
        // residual saved BEFORE final_layer_norm (second pre-norm residual).
        residual = x;
        clRetainMemObject(residual);

        // 3) final_layer_norm_{i}
        {
            char wp_ln[96];
            std::snprintf(wp_ln, sizeof(wp_ln), "model.encoder.layers.%d.final_layer_norm", layer_idx);
            cl_mem out = LayerNorm_forward(
                cl_ctx, weights, queue,
                x, (int)enc_T, layer_idx, /*start_pos=*/0,
                nullptr, nullptr,
                nullptr,
                wp_ln);
            if (!out) {
                NNOPT_ERROR_FMT("LayerNorm_forward(final_layer_norm_%d) failed", layer_idx);
                clReleaseMemObject(x);
                clReleaseMemObject(ids_buf);
                return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
            }
            if (out != x) { clReleaseMemObject(x); x = out; }
            NNOPT_LAYER_CHECK_FMT("final_layer_norm_%d", layer_idx, queue, x,
                                  (size_t)enc_T * (size_t)MODEL_CONFIG::HIDDEN_SIZE);
        }

        // 4) fc1_{i}
        {
            char wp_fc1[96];
            std::snprintf(wp_fc1, sizeof(wp_fc1), "model.encoder.layers.%d.fc1", layer_idx);
            cl_mem out = Linear_forward(
                cl_ctx, weights, queue,
                x, (int)enc_T, layer_idx, /*start_pos=*/0,
                nullptr, nullptr,
                nullptr,
                wp_fc1);
            if (!out) {
                NNOPT_ERROR_FMT("Linear_forward(fc1_%d) failed", layer_idx);
                clReleaseMemObject(x);
                clReleaseMemObject(ids_buf);
                return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
            }
            if (out != x) { clReleaseMemObject(x); x = out; }
            NNOPT_LAYER_CHECK_FMT("fc1_%d", layer_idx, queue, x,
                                  (size_t)enc_T * (size_t)MODEL_CONFIG::ENCODER_FFN_DIM);
        }

        // 5) activation_fn_{i}
        {
            cl_mem out = GELUActivation_forward(
                cl_ctx, weights, queue,
                x, (int)enc_T, layer_idx, /*start_pos=*/0,
                nullptr, nullptr,
                nullptr,
                "");
            if (!out) {
                NNOPT_ERROR_FMT("GELUActivation_forward(activation_fn_%d) failed", layer_idx);
                clReleaseMemObject(x);
                clReleaseMemObject(ids_buf);
                return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
            }
            if (out != x) { clReleaseMemObject(x); x = out; }
            NNOPT_LAYER_CHECK_FMT("activation_fn_%d", layer_idx, queue, x,
                                  (size_t)enc_T * (size_t)MODEL_CONFIG::ENCODER_FFN_DIM);
        }

        // 6) fc2_{i}
        {
            char wp_fc2[96];
            std::snprintf(wp_fc2, sizeof(wp_fc2), "model.encoder.layers.%d.fc2", layer_idx);
            cl_mem out = Linear_forward(
                cl_ctx, weights, queue,
                x, (int)enc_T, layer_idx, /*start_pos=*/0,
                nullptr, nullptr,
                nullptr,
                wp_fc2);
            if (!out) {
                NNOPT_ERROR_FMT("Linear_forward(fc2_%d) failed", layer_idx);
                clReleaseMemObject(x);
                clReleaseMemObject(ids_buf);
                return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
            }
            if (out != x) { clReleaseMemObject(x); x = out; }
            NNOPT_LAYER_CHECK_FMT("fc2_%d", layer_idx, queue, x,
                                  (size_t)enc_T * (size_t)MODEL_CONFIG::HIDDEN_SIZE);
        }

        // residual add 2: x = residual + fc2_out  → encoder layer output.
        // (Replaces the broken step-7 that re-ran the WHOLE layer on fc2 output.)
        {
            cl_mem r = element_add(queue, s_enc_utils_prog, residual, x, enc_n);
            clReleaseMemObject(residual);
            if (!r) {
                clReleaseMemObject(x); clReleaseMemObject(ids_buf);
                return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
            }
            if (r != x) clReleaseMemObject(x);
            x = r;
        }
        NNOPT_LAYER_CHECK_FMT("layer_%d", layer_idx, queue, x,
                              (size_t)enc_T * (size_t)MODEL_CONFIG::HIDDEN_SIZE);
    }

    // Final encoder LayerNorm (model.encoder.layer_norm) — applied AFTER all
    // encoder layers, before the output feeds decoder cross-attention. Whisper's
    // WhisperEncoder.forward ends with self.layer_norm(hidden_states); without it
    // the decoder cross-attn attends to un-normalized encoder states (cos~0.05).
    {
        cl_mem out = LayerNorm_forward(
            cl_ctx, weights, queue,
            x, (int)enc_T, /*layer_idx=*/-1, /*start_pos=*/0,
            nullptr, nullptr, nullptr,
            "model.encoder.layer_norm");
        if (!out) {
            NNOPT_ERROR("LayerNorm_forward(model.encoder.layer_norm) failed");
            clReleaseMemObject(x); clReleaseMemObject(ids_buf);
            return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
        }
        if (out != x) { clReleaseMemObject(x); x = out; }
    }

    encoder_hidden_states = x;
    x = nullptr;
    // Populate the encoder-output cache for subsequent decode steps of this clip.
    if (s_enc_cache) clReleaseMemObject(s_enc_cache);
    s_enc_cache = encoder_hidden_states;
    clRetainMemObject(s_enc_cache);   // cache holds its own ref
    s_enc_key = input_features;
    }  // end encoder compute (cache miss)

    // Decoder embedding (token + position)
    x = Embedding_forward(
        cl_ctx, weights, queue,
        ids_buf, seq_len, /*layer_idx=*/-1, start_pos,
        nullptr, nullptr,
        nullptr,
        "model.decoder.embed_tokens");
    clReleaseMemObject(ids_buf);
    ids_buf = nullptr;
    if (!x) {
        NNOPT_ERROR("Embedding_forward failed");
        clReleaseMemObject(encoder_hidden_states);
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    }
    NNOPT_LAYER_CHECK("embedding", queue, x, (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE);

    // Decoder layers — emit per-node dumps matching reference/forward_graph.json.
    for (int layer_idx = 0; layer_idx < MODEL_CONFIG::DECODER_LAYERS; layer_idx++) {
        // The decoder layer is computed ENTIRELY by WhisperDecoderLayer_forward
        // below (self-attn + cross-attn + MLP, all 3 residual adds, single KV
        // update). The previous inline self_attn_layer_norm + self_attn here were
        // broken: they applied NO residual and double-updated the KV cache before
        // the composite ran on their (wrong) output. Removed — feed the composite
        // the real layer input.
        // layer composite
        {
            char wp_layer[96];
            std::snprintf(wp_layer, sizeof(wp_layer), "model.decoder.layers.%d", layer_idx);
            cl_mem out = WhisperDecoderLayer_forward(
                cl_ctx, weights, queue,
                x, seq_len, layer_idx, start_pos,
                nullptr, nullptr,
                encoder_hidden_states,
                wp_layer);
            if (!out) {
                NNOPT_ERROR_FMT("WhisperDecoderLayer_forward failed at layer %d", layer_idx);
                clReleaseMemObject(x);
                clReleaseMemObject(encoder_hidden_states);
                return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
            }
            if (out != x) { clReleaseMemObject(x); x = out; }
            NNOPT_LAYER_CHECK_FMT("decoder_layer_%d", layer_idx, queue, x,
                                  (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE);
        }
    }

    // Final decoder norm
    {
        cl_mem out = LayerNorm_forward(
            cl_ctx, weights, queue,
            x, seq_len, /*layer_idx=*/-1, start_pos,
            nullptr, nullptr,
            nullptr,
            "model.decoder.layer_norm");
        if (!out) {
            NNOPT_ERROR("decoder final LayerNorm failed");
            clReleaseMemObject(x);
            clReleaseMemObject(encoder_hidden_states);
            return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
        }
        if (out != x) { clReleaseMemObject(x); x = out; }
    }
    NNOPT_LAYER_CHECK("final_norm", queue, x, (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE);

    // lm_head (tied to embed_tokens.weight)
    const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
    const int vocab = MODEL_CONFIG::VOCAB_SIZE;
    cl_mem lm_w = weights.get_buffer("model.decoder.embed_tokens.weight");
    if (!lm_w) {
        NNOPT_ERROR("missing model.decoder.embed_tokens.weight");
        clReleaseMemObject(x);
        clReleaseMemObject(encoder_hidden_states);
        return std::vector<float>(vocab, 0.0f);
    }

    cl_mem logits_buf = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                       (size_t)seq_len * (size_t)vocab * sizeof(nnopt_storage_t),
                                       nullptr, &err);
    if (err != CL_SUCCESS || !logits_buf) {
        NNOPT_ERROR_FMT("clCreateBuffer(logits) failed %d", (int)err);
        clReleaseMemObject(x);
        clReleaseMemObject(encoder_hidden_states);
        return std::vector<float>(vocab, 0.0f);
    }
    if (!pytorch_linear(queue, /*M=*/seq_len, /*N=*/vocab, /*K=*/hidden, x, lm_w, logits_buf,
                        "gemm_lm_head")) {
        NNOPT_ERROR("pytorch_linear lm_head failed");
        clReleaseMemObject(logits_buf);
        clReleaseMemObject(x);
        clReleaseMemObject(encoder_hidden_states);
        return std::vector<float>(vocab, 0.0f);
    }

    std::vector<float> logits(vocab, 0.0f);
    {
        std::vector<nnopt_storage_t> host_storage(vocab);
        size_t row_off = (size_t)(seq_len - 1) * (size_t)vocab * sizeof(nnopt_storage_t);
        clEnqueueReadBuffer(queue, logits_buf, CL_TRUE, row_off,
                            (size_t)vocab * sizeof(nnopt_storage_t),
                            host_storage.data(), 0, nullptr, nullptr);
        // Decode storage → float. Under fp16, nnopt_storage_t is cl_half (raw
        // uint16 bits): a bare (float) cast reinterprets the bit pattern as an
        // integer (0..65535), NOT the fp16 value — which silently turned every
        // logit into its raw half-float bits. Argmax then picked the most
        // *negative* logit (sign bit set → bits > 0x8000) instead of the
        // largest. Use the IEEE-754 codec so logits are real values.
        for (int i = 0; i < vocab; i++) {
#ifdef NNOPT_USE_FP16
            logits[i] = nnopt_f16_to_f32(static_cast<uint16_t>(host_storage[i]));
#else
            logits[i] = static_cast<float>(host_storage[i]);
#endif
        }
    }

    clReleaseMemObject(logits_buf);
    clReleaseMemObject(x);
    clReleaseMemObject(encoder_hidden_states);
    return logits;
}

// ASR signature (used by future Model::encode/decode split).
std::vector<float> model_forward_graph(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    cl_mem input_features,
    int start_pos)
{
    return _asr_forward_graph(cl_ctx, weights, input_ids, input_features, start_pos);
}

// Wrapper signature required by forward_dispatch.h / model.cpp.
std::vector<float> model_forward_graph(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    int start_pos)
{
    cl_mem feats = ForwardDispatch::get_input_features();
    return _asr_forward_graph(cl_ctx, weights, input_ids, feats, start_pos);
}
