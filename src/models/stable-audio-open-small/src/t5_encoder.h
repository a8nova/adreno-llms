#pragma once
// On-device T5-base encoder for prompt conditioning — CPU fp32.
//
// Runs ONCE per prompt (~seconds), so it lives on the CPU in fp32:
//   - T5 activations are a known fp16-overflow class (the magenta codec
//     lesson); fp32 end-to-end removes the risk outright.
//   - Keeps GPU memory headroom for the DiT/VAE (the CL heap on Adreno 620
//     fragments past ~128 MB single allocations).
// The DiT + VAE remain on the GPU; this stage is off the critical path.
//
// Weight pack: weights/t5_encoder.fp16.bin + .meta.json (same schema as the
// main model pack, loaded via the existing Weights class with a null CL
// context). Reference parity contract (verified host-side vs the PyTorch
// dumps in reference/t5_layers/):
//   - RMS layer norm (no mean subtract, no bias), eps 1e-6
//   - UNSCALED attention scores (T5 has no 1/sqrt(dk))
//   - one relative-position-bias table (block 0), 32 buckets, shared by all
//     12 blocks; padded KEY positions masked with -1e9 (queries run as-is)
//   - plain ReLU FFN (wi/wo), no biases on any linear
//   - conditioner wrapper: proj_out=Identity, PAD ROWS ZEROED at the end,
//     seconds embedding appended as token 65 == global_embed.

#include <string>
#include <vector>
#include <cstdint>

class T5CondEncoder {
public:
    // All paths relative to the run dir (weights/...). Returns false loudly
    // if any file is missing — caller falls back to asset-file conditioning.
    bool load(const std::string& weights_bin, const std::string& weights_meta,
              const std::string& tokenizer_bin, const std::string& seconds_table_bin);

    // prompt + seconds -> cross_attn_cond [65*768] and global_embed [768].
    bool compute(const std::string& prompt, int seconds_total,
                 std::vector<float>& cross_out, std::vector<float>& global_out,
                 int* n_real_tokens);

    static constexpr int MAX_TOKENS = 64;
    static constexpr int DIM = 768;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};
