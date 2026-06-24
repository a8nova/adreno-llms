#pragma once
// Reference: /Users/alazarshenkute/.nnopt/repos/OpenVoice/openvoice/models.py:325-331 SynthesizerTrn.voice_conversion
// Sampler shim for graph-mode harness; audio ports still use the same greedy API.
#include <cstdint>
#include <vector>

struct SamplerConfig {
    float temperature = 0.0f;
    int top_k = 1;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
    uint32_t seed = 42u;
    int eos_token_id = -1;
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& cfg) : cfg_(cfg) {}
    int sample(std::vector<float>& logits, const std::vector<int32_t>& generated_ids) const;
private:
    SamplerConfig cfg_;
};
