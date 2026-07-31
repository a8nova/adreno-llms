// On-device G2P phonemizer + streaming text chunker for KittenTTS --stream.
//
// Replaces the host-side scripts/phonemize.py (espeak-ng CLI) so streaming runs
// entirely in C++ on-device — no Python, no network. Two responsibilities:
//   1. Phonemizer — wraps espeak-ng's text->phoneme C API (IPA UTF-8) and maps
//      the phoneme string onto KittenTTS's 178-symbol vocab (assets/
//      phoneme_vocab.tsv), then wraps [bos] + ids + [eos] EXACTLY as
//      scripts/phonemize.py::build() does: bos=0 ("$"), eos=10 ("…"). Greedy
//      longest-match handles multi-codepoint IPA (diphthongs, length marks).
//      Unknown symbols are dropped (never mis-mapped), matching to_ids().
//   2. chunk_text — clause/word chunking (first chunk kept tiny for low
//      time-to-first-audio; later chunks grow) so streamed prosody is natural
//      and the playback buffer outruns synthesis at RTF < 1.
//
// Compiled only when espeak-ng is vendored + linked (CMake defines
// NNOPT_TTS_STREAMING). Without it this header's TU is empty and the existing
// single-shot build is unaffected.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace nnopt_tts {

// Split free text into synthesis chunks:
//   - first chunk kept tiny (<= first_max_words, or earlier at punctuation) so
//     time-to-first-audio is small;
//   - remaining text split on clause/sentence punctuation, merged up to a
//     graduated word cap (4 + 4*chunk_index, capped at max_words);
//   - a punctuation-free span > max_words is hard-split at word boundaries
//     (one oversized tail chunk is an underrun bomb).
std::vector<std::string> chunk_text(const std::string& text,
                                    int first_max_words = 5,
                                    int max_words = 14);

class Phonemizer {
public:
    Phonemizer() = default;
    ~Phonemizer();

    // espeak_data_parent: directory that CONTAINS espeak-ng-data/ (espeak's
    //   `path` arg wants the PARENT). Deploy pushes assets/* to the run dir, so
    //   espeak-ng-data/ lives at assets/espeak-ng-data/ → pass "assets".
    // vocab_tsv: "<phoneme>\t<id>\n" UTF-8 table (assets/phoneme_vocab.tsv).
    // voice: espeak voice, e.g. "en-us" (KittenTTS reference uses en-us).
    // Returns false + logs on any failure — caller MUST abort (a silent
    // phonemizer yields silent audio that every cosine gate misses).
    bool init(const std::string& espeak_data_parent,
              const std::string& vocab_tsv,
              const std::string& voice = "en-us");

    // text -> KittenTTS vocab ids, wrapped [0] + ids + [10] (matches
    // scripts/phonemize.py::build()). Phonemes absent from the vocab are
    // dropped (logged once per unique symbol at NNOPT_DEBUG_LAYERS=1).
    std::vector<int32_t> phonemize(const std::string& text) const;

    bool ready() const { return ready_; }

private:
    bool ready_ = false;
    // PHONEME (UTF-8, possibly multi-codepoint e.g. "eɪ") -> vocab id.
    // Built once in init(); sorted longest-key-first so multi-char IPA symbols
    // bind before their prefixes in phonemize().
    std::vector<std::pair<std::string, int32_t>> vocab_;
    // Framing ids, hardcoded to match scripts/phonemize.py: bos=0 ("$" in the
    // vocab), eos=10 ("…"). The tsv's own values, pinned here so a vocab reorder
    // can't silently break the wrapping the reference ids were validated with.
    int32_t bos_id_ = 0;
    int32_t eos_id_ = 10;
};

}  // namespace nnopt_tts
