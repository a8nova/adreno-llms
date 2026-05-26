#pragma once
// Reference: .nnport/tokenizer_contract.json (TTS ports load token IDs from assets/test_input_ids.bin)
// NOTE: MMS-TTS runtime does not require a tokenizer for deterministic evaluation;
// main.cpp only uses Tokenizer when --token-ids is not provided. Scaffold expects
// this header to exist.

#include <string>
#include <vector>
#include <cstdint>

class Tokenizer {
public:
    bool load(const std::string& vocab_path);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int32_t>& ids);
    int eos_token_id() const;
};
