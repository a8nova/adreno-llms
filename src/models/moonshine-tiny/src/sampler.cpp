// Token sampler implementation.
// Reference: standard HF GenerationConfig sampling — greedy / temperature /
// top-k / top-p / repetition penalty. Greedy (temperature==0 or top_k==1)
// is a plain argmax, matching the PyTorch reference for deterministic eval.

#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <limits>

Sampler::Sampler(const SamplerConfig& cfg)
    : cfg_(cfg), rng_state_(cfg.seed ? cfg.seed : 1u) {}

// xorshift32 — deterministic, seedable, no <random> dependency.
static inline uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

int Sampler::sample(std::vector<float>& logits,
                    const std::vector<int32_t>& generated_ids) const {
    const int vocab = static_cast<int>(logits.size());
    if (vocab <= 0) return 0;

    // Repetition penalty (HF convention): divide positive logits, multiply
    // negative logits by the penalty for already-generated tokens.
    if (cfg_.repetition_penalty != 1.0f) {
        for (int32_t id : generated_ids) {
            if (id < 0 || id >= vocab) continue;
            float v = logits[id];
            logits[id] = (v > 0.0f) ? v / cfg_.repetition_penalty
                                    : v * cfg_.repetition_penalty;
        }
    }

    // Greedy path: temperature 0 or top_k 1 → argmax.
    if (cfg_.temperature <= 0.0f || cfg_.top_k == 1) {
        int best = 0;
        float best_v = logits[0];
        for (int i = 1; i < vocab; ++i) {
            if (logits[i] > best_v) { best_v = logits[i]; best = i; }
        }
        return best;
    }

    // Temperature scaling.
    const float inv_t = 1.0f / cfg_.temperature;
    std::vector<int> idx(vocab);
    for (int i = 0; i < vocab; ++i) idx[i] = i;

    // top-k restriction.
    int k = (cfg_.top_k > 0 && cfg_.top_k < vocab) ? cfg_.top_k : vocab;
    if (k < vocab) {
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        idx.resize(k);
    } else {
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return logits[a] > logits[b]; });
    }

    // Softmax over the (sorted) candidate set with temperature.
    float max_logit = logits[idx[0]];
    std::vector<float> probs(idx.size());
    float sum = 0.0f;
    for (size_t i = 0; i < idx.size(); ++i) {
        float p = std::exp((logits[idx[i]] - max_logit) * inv_t);
        probs[i] = p;
        sum += p;
    }
    for (float& p : probs) p /= sum;

    // top-p (nucleus): keep the smallest prefix whose cumulative prob >= top_p.
    if (cfg_.top_p < 1.0f) {
        float cum = 0.0f;
        size_t keep = probs.size();
        for (size_t i = 0; i < probs.size(); ++i) {
            cum += probs[i];
            if (cum >= cfg_.top_p) { keep = i + 1; break; }
        }
        probs.resize(keep);
        idx.resize(keep);
        // Renormalize.
        float s = 0.0f;
        for (float p : probs) s += p;
        if (s > 0.0f) for (float& p : probs) p /= s;
    }

    // Sample from the categorical distribution.
    float r = (xorshift32(rng_state_) & 0xFFFFFF) / static_cast<float>(0x1000000);
    float cum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (r <= cum) return idx[i];
    }
    return idx.back();
}
