#pragma once
// Minimal tokenizer stub — depth-anything is a dense image->depth model with
// no text tokenization. main.cpp's generic loop references these symbols; the
// deterministic-eval path uses --token-ids so encode()/decode() are unused.
#include <string>
#include <vector>
#include <cstdint>

class Tokenizer {
public:
    bool load(const std::string& /*vocab_path*/) { return false; }
    std::vector<int> encode(const std::string& /*text*/) { return {}; }
    std::string decode(const std::vector<int32_t>& /*ids*/) { return ""; }
    int eos_token_id() const { return -1; }
};
