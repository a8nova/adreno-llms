#pragma once
#include <vector>
#include <cstdint>

struct SamplerConfig {
    float temperature = 1.0f;
    int top_k = 50;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
    int eos_token_id = -1;
    uint32_t seed = 42;
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& config);

    int sample(std::vector<float>& logits,
               const std::vector<int32_t>& generated_ids) const;

private:
    SamplerConfig config_;
};
