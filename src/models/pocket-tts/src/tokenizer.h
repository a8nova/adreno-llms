// SentencePiece-unigram tokenizer for pocket-tts (text → token ids), no protobuf.
// Reads the flat tokenizer_vocab.bin produced by scripts/convert_tokenizer.py and
// runs the unigram Viterbi segmentation (Metaspace + byte fallback). Validated
// byte-for-byte against the Python `sentencepiece` reference on English text.
#pragma once
#include <string>
#include <vector>
#include <unordered_map>

class Tokenizer {
public:
    bool load(const std::string& vocab_path);
    std::vector<int> encode(const std::string& text) const;   // text → ids (no bos/eos)
    int bos() const { return bos_; }
    int eos() const { return eos_; }
    int vocab_size() const { return (int)score_.size(); }

private:
    std::unordered_map<std::string, int> piece2id_;   // UTF-8 piece → id
    std::vector<float> score_;                         // id → unigram log-prob
    int byte_fallback_[256];                           // byte → id of "<0xHH>" (-1 if absent)
    int unk_ = 0, bos_ = 1, eos_ = 2, pad_ = 3;
    int max_piece_cp_ = 1;                             // longest piece, in codepoints
};
