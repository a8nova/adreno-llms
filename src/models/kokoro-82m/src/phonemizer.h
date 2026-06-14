// On-device G2P phonemizer + streaming text chunker for Kokoro --stream.
//
// Replaces the host-side Python misaki/espeak step so streaming runs entirely
// in C++ on-device. Two responsibilities:
//   1. Phonemizer — wraps espeak-ng's text->phoneme C API and maps the IPA
//      phoneme string onto Kokoro's own vocab ids (model_info/config.json
//      ::vocab, distilled host-side into assets/phoneme_vocab.tsv). espeak-ng
//      is Kokoro's documented fallback phonemizer; its inventory is a close
//      (not identical) match to misaki — unknown symbols are dropped, never
//      mis-mapped to 0.
//   2. chunk_text — clause/word chunking identical to the retired stream.py so
//      streamed prosody matches the validated host-driven path.
//
// Compiled only when espeak-ng is vendored + linked (CMake defines
// NNOPT_TTS_STREAMING). NO Python, NO network.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace nnopt_tts {

// Split free text into synthesis chunks (mirrors stream.py chunk_text):
//   - first chunk kept tiny (<= first_max_words, or earlier at punctuation) so
//     time-to-first-audio is small;
//   - remaining text split on clause/sentence punctuation, merged up to a
//     graduated word cap (4 + 4*chunk_index, capped at max_words);
//   - a punctuation-free span > max_words is hard-split at word boundaries.
std::vector<std::string> chunk_text(const std::string& text,
                                    int first_max_words = 5,
                                    int max_words = 14);

class Phonemizer {
public:
    Phonemizer() = default;
    ~Phonemizer();

    // espeak_data_parent: directory that CONTAINS espeak-ng-data/ (espeak's
    //   `path` arg wants the PARENT). e.g. "." when espeak-ng-data/ sits in the
    //   device run dir.
    // vocab_tsv: "<phoneme>\t<id>\n" UTF-8 table distilled from config.json
    //   ::vocab (no on-device JSON parsing). Empty key = pad/BOS id.
    // voice: espeak voice, e.g. "en-us" (Kokoro 'a'), "en-gb" ('b').
    // Returns false + logs on any failure — caller MUST abort (a silent
    // phonemizer yields silent audio every cosine gate misses).
    bool init(const std::string& espeak_data_parent,
              const std::string& vocab_tsv,
              const std::string& voice = "en-us");

    // text -> Kokoro vocab ids, wrapped [pad] + ids + [pad] (matches stream.py).
    std::vector<int32_t> phonemize(const std::string& text) const;

    bool ready() const { return ready_; }

private:
    bool ready_ = false;
    std::vector<std::pair<std::string, int32_t>> vocab_;  // sorted by key length desc
    int32_t pad_id_ = 0;
};

}  // namespace nnopt_tts
