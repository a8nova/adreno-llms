#pragma once
// Reference: .nnport/tokenizer.json:model  (type=Synthetic-VITS-Bypass)
// Reference: .nnport/tokenizer_contract.json  (kind=phoneme_pinned_ids)
//
// KittenML/kitten-tts-nano-0.1 is a VITS/StyleTTS2-family phoneme TTS model.
// Its tokenizer is a phoneme symbol table (178 IPA symbols, index == token id)
// plus a G2P front-end (espeak IPA, en-us, with_stress). The full text->phoneme
// step (espeak) is the P2 on-device wiring; convergence + eval consume the
// pinned phoneme ids published by GenerateReference in assets/test_input_ids.bin.
//
// This class loads the real symbol<->id vocab (assets/phoneme_vocab.tsv) so
// encode() maps IPA phoneme strings to ids with the model's framing, and
// eos_token_id() returns the trailing framing symbol. Raw-text->IPA (espeak)
// is not linked in this build; the deterministic path feeds pinned ids.

#include <string>
#include <vector>
#include <unordered_map>

class Tokenizer {
public:
    Tokenizer() = default;

    // Loads the phoneme symbol vocab. `vocab_path` is main.cpp's
    // "weights/tokenizer_vocab.bin" hint; for this phoneme model the real
    // symbol table lives at assets/phoneme_vocab.tsv, which we load directly.
    // Returns true when the symbol table is available.
    bool load(const std::string& vocab_path);

    // Maps an already-phonemized IPA string to token ids (per-character symbol
    // lookup) with the model's framing prepended/appended. Text that has NOT
    // been run through espeak G2P will not match the reference ids — the
    // deterministic eval path uses assets/test_input_ids.bin instead.
    std::vector<int> encode(const std::string& text);

    // Decodes ids back to their symbol strings concatenated.
    std::string decode(const std::vector<int32_t>& ids);

    // The trailing framing symbol id (end-of-sequence marker for this model).
    int eos_token_id() const { return eos_id_; }

private:
    // symbol -> id, and id -> symbol (symbols may be multi-byte UTF-8 IPA).
    std::unordered_map<std::string, int> sym_to_id_;
    std::vector<std::string> id_to_sym_;
    int bos_id_ = 0;   // framing[0]
    int eos_id_ = 0;   // framing[2] (or framing[1] if the middle slot is unused)
    int pad_id_ = 0;   // framing[1]
    bool loaded_ = false;
};
