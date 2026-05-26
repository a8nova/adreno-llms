#pragma once
// Auto-generated graph-mode model interface for facebook/mms-tts-eng.
// Backbone class: vits
//
// In graph mode the Model class is intentionally thin. The per-op code that
// implements the forward() data flow lives in src/ops/*.cpp, written by the
// agent one node at a time via PortNode. This file defines only the public
// interface main.cpp uses (constructor, generate-style forward).

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

    // NOTE: TTS (VITS) ports do NOT use an autoregressive logits forward.
    // The entry point is forward_graph(...) which consumes deterministic
    // fixtures (duration_noise, prior_noise) and produces PCM audio.

    // TTS forward. Runs the full VITS pipeline once and writes int16 PCM.
    int forward_graph(const std::vector<int32_t>& input_ids,
                      const std::vector<float>& duration_noise,
                      const std::vector<float>& prior_noise,
                      std::vector<int16_t>& out_pcm_int16);

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
};
