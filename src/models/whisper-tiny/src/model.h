#pragma once
// Auto-generated graph-mode model interface for openai/whisper-tiny.
// Backbone class: whisper
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

    // Backbone forward. Returns logits[VOCAB_SIZE] for the LAST token of
    // input_ids (matches PyTorch convention). start_pos is the absolute
    // KV-cache offset; 0 for prefill.
    std::vector<float> forward(const std::vector<int32_t>& input_ids, int start_pos);

    // Convenience wrapper used by main.cpp's generate loop.
    std::vector<float> forward(const std::vector<int32_t>& input_ids) {
        return forward(input_ids, 0);
    }

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
};
