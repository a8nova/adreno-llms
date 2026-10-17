#pragma once
// Auto-generated graph-mode model interface for KittenML/kitten-tts-nano-0.1.
// Backbone class: unknown
//
// In graph mode the Model class is intentionally thin. The per-op code that
// implements the forward() data flow lives in src/ops/*.cpp, written by the
// agent. This file defines only the public interface main.cpp uses.

#include "opencl_context.h"
#include "weights.h"
// sampler.h intentionally NOT included — this modality has no token-sampling
// output, so the scaffold skips sampler generation entirely (depth-anything
// 2026-07-04: the unconditional include was a guaranteed first-build error
// for every audio/image/composite port).
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

    // Backbone forward. Returns logits[VOCAB_SIZE] for the LAST token of
    // input_ids (matches PyTorch convention). start_pos is the absolute
    // KV-cache offset; 0 for prefill.
    std::vector<float> forward(const std::vector<int32_t>& input_ids, int start_pos);

    // Convenience wrapper used by main.cpp's generate loop.
    std::vector<float> forward(const std::vector<int32_t>& input_ids) {
        return forward(input_ids, 0);
    }

    // TTS single-shot entry point used by main.cpp. Threads the aux fixtures
    // (speaker style vector + captured RNG noise) alongside the phoneme ids
    // into the ONNX submodel graph and writes the produced waveform to `pcm`
    // as 16-bit signed samples. Returns 0 on success, non-zero on failure.
    int forward_graph(
        const std::vector<int32_t>& input_ids,
        const std::vector<float>& style,
        const std::vector<float>& rng_uniform,
        const std::vector<float>& rng_normal,
        std::vector<int16_t>& pcm);

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
};
