// Reference: /Users/alazarshenkute/.nnopt/repos/OpenVoice/openvoice/models.py:325-331 SynthesizerTrn.voice_conversion
#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>

int Sampler::sample(std::vector<float>& logits, const std::vector<int32_t>& generated_ids) const {
    (void)generated_ids;
    if (logits.empty()) return cfg_.eos_token_id >= 0 ? cfg_.eos_token_id : 0;
    int best = 0;
    float best_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < logits.size(); ++i) {
        float v = logits[i];
        if (!std::isfinite(v)) v = -std::numeric_limits<float>::infinity();
        if (cfg_.eos_token_id >= 0 && (int)i == cfg_.eos_token_id && logits.size() > 1) continue;
        if (v > best_val) { best_val = v; best = (int)i; }
    }
    return best;
}
