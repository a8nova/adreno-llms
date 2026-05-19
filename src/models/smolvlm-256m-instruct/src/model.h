#pragma once
// Auto-generated graph-mode model interface for HuggingFaceTB/SmolVLM-256M-Instruct.
// Backbone class: Idefics3ForConditionalGeneration
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

    // VLM: set image bytes (RGB HWC uint8) for subsequent forward() calls.
    // The implementation runs the vision pipeline and caches projected
    // features inside the model.
    bool set_image(const std::vector<uint8_t>& rgb_u8, int width, int height);

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

    // Cached projected image features (host-side fp32) produced by
    // vision_pipeline_forward. Empty when no image is set.
    std::vector<float> image_features_;
    int image_features_N_ = 0;
    bool has_image_ = false;
};
