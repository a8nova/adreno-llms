#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// SentencePiece UNIGRAM tokenizer for MusicCoCa's text tower.
//
// This is what makes the prompt path actually end-to-end: without it the app would have to
// tokenize elsewhere and hand the engine ids, which is the same "runs on device except for the
// part that doesn't" problem the MusicCoCa port itself was fixing.
//
// The model (spm.model, 16k pieces, byte fallback) is flattened by scripts/extract_spm_vocab.py
// into weights/musiccoca_spm.bin so nothing here needs protobuf.
//
// Encoding follows SentencePiece exactly for the text this path sees:
//   lowercase → collapse whitespace runs → strip → prepend a dummy space → map ' ' to U+2581
//   → Viterbi over the piece lattice (maximising the summed log-probs)
//   → any character with no piece becomes its UTF-8 bytes as <0xNN> pieces
//
// LIMITATION, stated rather than hidden: the reference normalizes with nmt_nfkc via a 238 KB
// precompiled charsmap. This implements the ASCII-exact subset of it (NFKC is the identity on
// ASCII), so ASCII prompts match the reference exactly — verified against the Python tokenizer.
// Non-ASCII input is passed through and byte-fallback still yields valid ids, but a prompt
// containing characters NFKC would fold (full-width forms, ligatures, exotic spaces) can tokenize
// differently from the reference. encode() reports that case via `had_non_ascii`.
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class SpmTokenizer {
public:
    bool load(const std::string& path);

    // text → ids, with the leading BOS the text encoder expects. Truncates to max_ids pieces
    // (BOS included) the same way the reference does.
    bool encode(const std::string& text, int max_ids, std::vector<int32_t>& ids_out,
                bool* had_non_ascii = nullptr) const;

    int vocab_size() const { return (int)pieces_.size(); }
    int bos_id() const { return bos_id_; }

private:
    struct Piece {
        std::string text;
        float score = 0.0f;
        uint8_t type = 1;
    };
    std::vector<Piece> pieces_;
    std::unordered_map<std::string, int> index_;   // piece text → id (NORMAL/USER_DEFINED only)
    int byte_id_[256];                             // byte value → <0xNN> id, -1 if absent
    size_t max_piece_bytes_ = 1;
    int unk_id_ = 0, bos_id_ = 1, eos_id_ = 2;
    float min_score_ = 0.0f;
};
