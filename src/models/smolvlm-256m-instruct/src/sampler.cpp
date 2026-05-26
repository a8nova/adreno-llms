#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

Sampler::Sampler(const SamplerConfig& config) : config_(config) {}

int Sampler::sample(std::vector<float>& logits,
                    const std::vector<int32_t>& generated_ids) const {
    // Debug: show top logit before any processing. Gated on NNOPT_DEBUG to keep
    // Release runs clean (was firing every decode step, polluting stdout).
#ifdef NNOPT_DEBUG
    if (!logits.empty()) {
        auto max_it = std::max_element(logits.begin(), logits.end());
        int max_idx = static_cast<int>(max_it - logits.begin());
        bool all_zero = std::all_of(logits.begin(), logits.end(), [](float v) { return v == 0.0f; });
        fprintf(stderr, "Sampler: max_logit=%.4f at id=%d, logits_size=%zu%s\n",
                *max_it, max_idx, logits.size(), all_zero ? " [ALL ZEROS - BROKEN!]" : "");
    }
#endif
    // 1. Apply repetition penalty
    if (config_.repetition_penalty != 1.0f && !generated_ids.empty()) {
        for (int id : generated_ids) {
            if (id >= 0 && id < static_cast<int>(logits.size())) {
                if (logits[id] > 0.0f) {
                    logits[id] /= config_.repetition_penalty;
                } else {
                    logits[id] *= config_.repetition_penalty;
                }
            }
        }
    }

    // 2. Greedy: return argmax when temperature <= 0
    if (config_.temperature <= 0.0f) {
        return static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    // 3. Apply temperature
    for (auto& l : logits) l /= config_.temperature;

    // 4. Build sorted index for top-k / top-p filtering
    std::vector<int> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return logits[a] > logits[b]; });

    // 5. Top-k: keep only the k highest logits
    int keep = static_cast<int>(logits.size());
    if (config_.top_k > 0 && config_.top_k < keep) {
        keep = config_.top_k;
    }

    // 6. Softmax over kept logits
    float max_val = logits[indices[0]];
    std::vector<float> probs(keep);
    float sum = 0.0f;
    for (int i = 0; i < keep; ++i) {
        probs[i] = std::exp(logits[indices[i]] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < keep; ++i) probs[i] /= sum;

    // 7. Top-p (nucleus): find cutoff where cumulative prob >= top_p
    if (config_.top_p < 1.0f) {
        float cumulative = 0.0f;
        int cutoff = keep;
        for (int i = 0; i < keep; ++i) {
            cumulative += probs[i];
            if (cumulative >= config_.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        if (cutoff < keep) {
            keep = cutoff;
            // Re-normalize
            sum = 0.0f;
            for (int i = 0; i < keep; ++i) sum += probs[i];
            for (int i = 0; i < keep; ++i) probs[i] /= sum;
        }
    }

    // 8. Sample from the filtered distribution
    static thread_local std::mt19937 rng(42);
    if (config_.seed != 42) {
        rng.seed(config_.seed);
    }
    std::discrete_distribution<int> dist(probs.begin(), probs.begin() + keep);
    return indices[dist(rng)];
}
