#pragma once
// Minimal sampler — depth-anything is a dense image->depth model with no
// autoregressive decoding. main.cpp's generic loop still references these
// symbols, so provide a compile-complete greedy argmax implementation.
#include <vector>
#include <cstdint>

struct SamplerConfig {
    float    temperature        = 0.0f;
    int      top_k              = 1;
    float    top_p              = 1.0f;
    float    repetition_penalty = 1.0f;
    uint32_t seed               = 42u;
    int      eos_token_id       = -1;
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& cfg) : cfg_(cfg) {}

    // Greedy argmax over logits. generated_ids unused (no penalties for depth).
    int sample(std::vector<float>& logits,
               const std::vector<int32_t>& /*generated_ids*/) const {
        if (logits.empty()) return cfg_.eos_token_id;
        int best = 0;
        float best_v = logits[0];
        for (size_t i = 1; i < logits.size(); ++i) {
            if (logits[i] > best_v) { best_v = logits[i]; best = (int)i; }
        }
        return best;
    }

private:
    SamplerConfig cfg_;
};
