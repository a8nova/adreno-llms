#pragma once
// Minimal sampler interface to satisfy the scaffolded main.cpp.
//
// This port targets an MLX audio model; main.cpp is scaffolded for text
// generation, but we keep the build green by providing the expected symbols.
//
// NOTE: These are placeholders — the actual runtime path should be reworked to
// run the model's forward graph (likely producing waveform samples directly)
// instead of token sampling.

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

  int sample(const std::vector<float>& logits,
             const std::vector<int32_t>& /*generated_ids*/) const {
    // Greedy argmax. Temperature/top-k/top-p ignored for now.
    int best = 0;
    float best_v = (logits.empty() ? 0.0f : logits[0]);
    for (int i = 1; i < (int)logits.size(); ++i) {
      if (logits[i] > best_v) {
        best_v = logits[i];
        best = i;
      }
    }
    return best;
  }

 private:
  SamplerConfig cfg_;
};
