// Reference: .nnport/tokenizer.json:model  (type=Synthetic-VITS-Bypass)
// Reference: .nnport/tokenizer_contract.json  (kind=phoneme_pinned_ids)
// Reference: .nnport/phoneme_vocab.json  (symbols[], framing[0,10,0], phonemizer=espeak_ipa)
//
// Phoneme symbol tokenizer for KittenML/kitten-tts-nano-0.1. Loads the real
// symbol<->id table from assets/phoneme_vocab.tsv. encode() performs greedy
// longest-match over the symbol table on an IPA phoneme string and applies the
// model framing (bos prepended, eos appended). Raw-text->IPA (espeak G2P) is
// not linked here; the deterministic eval path consumes assets/test_input_ids.bin.

#include "tokenizer.h"
#include "debug_utils.h"

#include <fstream>
#include <sstream>
#include <algorithm>

// Load the phoneme symbol vocab. The TSV is "<symbol>\t<id>" per line, where
// the id is the row index (0..177). We build both directions of the map.
// framing = [bos, pad, eos] = [0, 10, 0] per .nnport/phoneme_vocab.json; the
// pinned reference ids in assets/test_input_ids.bin begin with 0 ('$') and end
// with 10 ('…'), confirming bos=0 / trailing=10.
bool Tokenizer::load(const std::string& /*vocab_path*/) {
    const char* tsv_path = "assets/phoneme_vocab.tsv";
    std::ifstream f(tsv_path);
    if (!f.is_open()) {
        NNOPT_ERROR_FMT("tokenizer: cannot open %s", tsv_path);
        return false;
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        // Split on the last tab so a symbol that is itself whitespace/tab-like
        // still parses (id is always the trailing integer field).
        size_t tab = line.rfind('\t');
        if (tab == std::string::npos) continue;
        std::string sym = line.substr(0, tab);
        int id = 0;
        try {
            id = std::stoi(line.substr(tab + 1));
        } catch (...) {
            continue;
        }
        if (id < 0) continue;
        if ((int)id_to_sym_.size() <= id) id_to_sym_.resize(id + 1);
        id_to_sym_[id] = sym;
        sym_to_id_[sym] = id;
    }

    if (id_to_sym_.empty()) {
        NNOPT_ERROR("tokenizer: phoneme vocab loaded 0 symbols");
        return false;
    }

    // Framing from .nnport/phoneme_vocab.json = [0, 10, 0].
    bos_id_ = 0;
    pad_id_ = 10;
    eos_id_ = 10;   // trailing framing symbol ('…') — matches pinned-id tail.

    loaded_ = true;
    NNOPT_CHECKPOINT_FMT("tokenizer: loaded %zu phoneme symbols", id_to_sym_.size());
    return true;
}

// Greedy longest-match tokenization over UTF-8 IPA symbols. IPA phonemes can be
// multi-byte (e.g. 'ˈ', 'ɪ'), so we try the longest byte prefix present in the
// symbol table at each position. Applies framing: bos prepended, eos appended.
std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> ids;
    if (!loaded_) return ids;

    ids.push_back(bos_id_);

    // Determine the longest symbol byte-length so the greedy match is bounded.
    size_t max_sym_len = 1;
    for (const auto& kv : sym_to_id_) max_sym_len = std::max(max_sym_len, kv.first.size());

    size_t i = 0;
    while (i < text.size()) {
        bool matched = false;
        size_t max_len = std::min(max_sym_len, text.size() - i);
        for (size_t len = max_len; len >= 1; --len) {
            std::string cand = text.substr(i, len);
            auto it = sym_to_id_.find(cand);
            if (it != sym_to_id_.end()) {
                ids.push_back(it->second);
                i += len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            // Unknown byte — skip it rather than emit a spurious id.
            i += 1;
        }
    }

    ids.push_back(eos_id_);
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) {
    std::string out;
    for (int32_t id : ids) {
        if (id >= 0 && id < (int)id_to_sym_.size()) out += id_to_sym_[id];
    }
    return out;
}
