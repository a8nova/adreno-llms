// Qwen2-family byte-level BPE from the vocab/merges the converter extracted
// out of the GGUF metadata (model/tokenizer.json).
//
// Pre-tokenizer: hand-rolled scanner implementing the qwen2 regex
//   (?:'[sSdDmM]|'[rR][eE]|'[vV][eE]|'[lL][lL])
//   | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+ | \s+(?!\S) | \s+
// Unicode classes are exact for ASCII; codepoints >= 0x80 use a compact
// approximation (letters unless known space/punct) — exact for the golden
// prompts and normal English chat; CJK edge cases logged in BACKTRACK.md.
//
// Class DECLARATION only; method bodies are defined in tokenizer.cpp.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class Tokenizer {
  public:
    void load(const std::string& tokenizer_json_path);

    std::vector<int> encode(const std::string& text) const;

    std::string decode(const std::vector<int>& ids) const;

  private:
    // ---- GPT-2 byte encoder ----
    void init_byte_maps();
    static void decode_utf8(const std::string& s, size_t i, uint32_t* cp, int* len);
    static void append_utf8(std::string& s, uint32_t cp);

    // ---- pre-tokenizer scanner over raw text codepoints ----
    static bool is_space_cp(uint32_t c);
    static bool is_digit_cp(uint32_t c);
    static bool is_letter_cp(uint32_t c);

    void encode_plain(const std::string& text, std::vector<int>& out) const;

    void bpe_piece(const std::string& raw, std::vector<int>& out) const;

    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> tok2id_;
    std::unordered_map<std::string, int> merge_rank_;
    std::vector<std::pair<std::string, int>> specials_;
    std::unordered_map<int, uint32_t> byte2cp_;
    std::unordered_map<uint32_t, int> cp2byte_;
};
