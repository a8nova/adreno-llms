#pragma once
// Auto-generated graph-mode model interface for facebook/musicgen-small.
// Backbone class: musicgen
//
// In graph mode the Model class is intentionally thin. The per-op code that
// implements the forward() data flow lives in src/ops/*.cpp, written by the
// agent. This file defines only the public interface main.cpp uses.

#include "opencl_context.h"
#include "weights.h"
#include "sampler.h"
#include "tokenizer.h"
#include <vector>
#include <cstdint>

class Model {
public:
    Model(OpenCLContext& cl_ctx, Weights& weights);
    ~Model();

    // Initialize per-op state (cached programs, KV buffers, RoPE tables…).
    // Optional — graph-mode ops can lazy-init on first call. Returns true.
    bool initialize();

    // Stage [1] of the MusicGen pipeline: host-side T5 encoder +
    // enc_to_dec_proj over the TEXT prompt ids. Uploads encoder_hidden_states
    // for every decoder layer's cross-attention and resets the per-layer KV
    // caches (new generation). Must be called once before the decode loop.
    bool encode_text(const std::vector<int32_t>& text_ids);
    // TTFT prewarm: upload ZERO encoder states of the right shape, precompute
    // cross-KV on them, and run ONE discarded CFG-2 step at start_pos=0 — this
    // absorbs every first-pass GPU resource creation (weight buffers, texture
    // views, kernel clones, scratch, KV allocs — the measured 4.8 s layers-10-13
    // driver stall) while host T5 encodes on a worker thread. All value-state
    // the dummy touches is overwritten: apply_encoder_states() re-uploads real
    // states + re-precomputes cross-KV (model_reset_decode_state inside), and
    // the real step 0 rewrites KV position 0 / grid col 1.
    bool prewarm_decoder(int enc_len);
    // Second half of encode_text() for the prewarm-overlap path: takes the
    // t5_encode_host() output computed on a worker thread (host-only, no CL).
    bool apply_encoder_states(const std::vector<float>& states);

    // Stage [2] decode step. input_ids = the CURRENT per-codebook tokens
    // (delay-pattern grid column, 4 ids). Returns flattened logits
    // [NUM_CODEBOOKS * VOCAB_SIZE] — one row per codebook head.
    // start_pos is the absolute KV-cache position (0, 1, 2, ...).
    std::vector<float> forward(const std::vector<int32_t>& input_ids, int start_pos);

    // CFG decode step (modeling_musicgen.py ClassifierFreeGuidanceLogitsProcessor):
    // runs conditioned + unconditional branches (independent KV caches) and
    // returns uncond + guidance * (cond - uncond), flattened [4 * VOCAB].
    // guidance <= 1 → single conditioned pass (plain forward).
    std::vector<float> forward_cfg(const std::vector<int32_t>& input_ids, int start_pos, float guidance);

    // Stage-3 FUSED-READOUT decode step: runs the decoder + the single
    // fused_readout dispatch (final LN + lm_head GEMV + CFG blend + sample) and
    // writes 4 sampled token ids to out_ids[NUM_CODEBOOKS]. No logits cross the
    // host bus. force_argmax!=0 makes the sampler pure argmax (greedy guard).
    // Returns true on success. Only valid for guidance>1 (CFG path).
    bool forward_cfg_sampled(const std::vector<int32_t>& input_ids, int start_pos,
                             float guidance, float temperature, int top_k,
                             uint32_t seed, int force_argmax, int32_t* out_ids);

    // Stage-4 GPU-resident decode grid. alloc uploads the host grid (BOS-filled)
    // to the device and registers it so embeddings read ids from grid[:,step]
    // and the sampler writes grid[:,step+1] — no per-step id upload. read pulls
    // the whole grid back once (end of decode). free releases + deregisters.
    void* gpu_grid_alloc(const int32_t* host_grid, int num_codebooks, int steps1, int bos);
    bool  gpu_grid_read(void* grid, int32_t* host_grid, int num_codebooks, int steps1);
    // Non-blocking partial read for the streaming-EnCodec worker (see backbone).
    bool  gpu_grid_read_cols_async(void* grid, int32_t* host_grid, int num_codebooks,
                                   int steps1, int col0, int col1, void** evt_out);
    void  gpu_grid_free(void* grid);

    // Convenience wrapper used by main.cpp's generate loop.
    std::vector<float> forward(const std::vector<int32_t>& input_ids) {
        return forward(input_ids, 0);
    }

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
};
