#pragma once
// Token sampler for autoregressive generation.
// Reference: standard HF GenerationConfig semantics — greedy (temperature=0
// or top_k=1) is argmax over logits, matching the PyTorch reference used by
// GenerateReference for deterministic evaluation.

#include <cstdint>
#include <vector>

struct SamplerConfig {
    float    temperature        = 0.0f;   // 0 => greedy (argmax)
    int      top_k              = 1;      // 1 => greedy
    float    top_p              = 1.0f;   // nucleus threshold
    float    repetition_penalty = 1.0f;   // 1.0 => disabled
    uint32_t seed               = 42u;
    int      eos_token_id       = -1;     // -1 => none
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& cfg);

    // Return the next token id given the full logits[VOCAB_SIZE] vector and
    // the tokens generated so far (for repetition penalty). const because it
    // mutates only the internal RNG (which is mutable).
    int sample(std::vector<float>& logits,
               const std::vector<int32_t>& generated_ids) const;

private:
    SamplerConfig    cfg_;
    mutable uint32_t rng_state_;
};
