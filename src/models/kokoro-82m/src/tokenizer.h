#pragma once
// Reference: /Users/alazarshenkute/Projects/nnopt/ports/kokoro-android-opencl-motorola-razr-2020/.nnport/tokenizer.json
// NOTE: Kokoro TTS graph-mode port currently bypasses text tokenization and
// consumes precomputed phoneme/token IDs fixtures. This tokenizer is a minimal
// compile-time stub so the scaffold builds; it is NOT used by forward_graph().

#include <cstdint>
#include <string>
#include <vector>

class Tokenizer {
public:
    Tokenizer() noexcept = default;

    // Minimal API expected by scaffold.
    bool load(const std::string& /*vocab_path*/) { return true; }

    std::vector<int32_t> encode(const std::string& /*text*/) const { return {}; }

    std::string decode(const std::vector<int32_t>& /*ids*/) const { return ""; }

    int eos_token_id() const { return 0; }
};
