// On-device G2P phonemizer + chunker. See phonemizer.h. NO Python, NO network.

#include "phonemizer.h"

// Compiled only when espeak-ng is vendored + linked (CMake defines
// NNOPT_TTS_STREAMING). Without it this TU is empty so the existing build is
// unaffected until espeak is vendored.
#ifdef NNOPT_TTS_STREAMING

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

#include <espeak-ng/speak_lib.h>

namespace nnopt_tts {

// ---------------------------------------------------------------------------
// chunk_text — faithful port of the retired stream.py chunk_text().
// ---------------------------------------------------------------------------

namespace {

std::vector<std::string> split_ws(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string w;
    while (iss >> w) out.push_back(w);
    return out;
}

std::string join_ws(const std::vector<std::string>& v, size_t from, size_t to) {
    std::string out;
    for (size_t i = from; i < to && i < v.size(); ++i) {
        if (!out.empty()) out += ' ';
        out += v[i];
    }
    return out;
}

bool ends_with_any(const std::string& s, const char* chars) {
    if (s.empty()) return false;
    return std::strchr(chars, s.back()) != nullptr;
}

size_t word_count(const std::string& s) { return split_ws(s).size(); }

// Equivalent of re.split(r'(?<=[.!?;:,])\s+', rest): cut on a whitespace run
// whose immediately-preceding non-space char is clause punctuation.
std::vector<std::string> split_on_clause_punct(const std::string& rest) {
    std::vector<std::string> parts;
    std::string cur;
    for (size_t i = 0; i < rest.size(); ++i) {
        char c = rest[i];
        if (std::isspace((unsigned char)c)) {
            char prev = cur.empty() ? '\0' : cur.back();
            if (prev && std::strchr(".!?;:,", prev)) {
                while (i + 1 < rest.size() && std::isspace((unsigned char)rest[i + 1])) ++i;
                parts.push_back(cur);
                cur.clear();
                continue;
            }
        }
        cur += c;
    }
    if (!cur.empty()) parts.push_back(cur);
    return parts;
}

}  // namespace

std::vector<std::string> chunk_text(const std::string& text,
                                    int first_max_words, int max_words) {
    std::vector<std::string> words = split_ws(text);
    if (words.empty()) return {};

    size_t first_n = words.size();
    bool broke = false;
    for (size_t i = 0; i < (size_t)first_max_words && i < words.size(); ++i) {
        if (ends_with_any(words[i], ",.!?;:")) { first_n = i + 1; broke = true; break; }
    }
    if (!broke) first_n = std::min((size_t)first_max_words, words.size());

    std::vector<std::string> chunks;
    chunks.push_back(join_ws(words, 0, first_n));
    std::string rest = join_ws(words, first_n, words.size());
    if (rest.empty()) return chunks;

    std::vector<std::string> parts;
    for (std::string& p : split_on_clause_punct(rest)) {
        size_t b = p.find_first_not_of(" \t");
        size_t e = p.find_last_not_of(" \t");
        if (b == std::string::npos) continue;
        p = p.substr(b, e - b + 1);
        std::vector<std::string> pw = split_ws(p);
        if ((int)pw.size() > max_words) {
            int n_pieces = ((int)pw.size() + max_words - 1) / max_words;   // ceil
            int per = ((int)pw.size() + n_pieces - 1) / n_pieces;          // balanced
            for (size_t i = 0; i < pw.size(); i += per)
                parts.push_back(join_ws(pw, i, i + per));
        } else {
            parts.push_back(p);
        }
    }

    std::string cur;
    for (const std::string& p : parts) {
        int cap = std::min(max_words, 4 + 4 * (int)chunks.size());
        std::string cand = cur.empty() ? p : cur + " " + p;
        if (!cur.empty() && (int)word_count(cand) > cap) {
            chunks.push_back(cur);
            cur = p;
        } else {
            cur = cand;
        }
        if (ends_with_any(cur, ".!?") && (int)word_count(cur) >= 4) {
            chunks.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) chunks.push_back(cur);
    return chunks;
}

// ---------------------------------------------------------------------------
// Phonemizer
// ---------------------------------------------------------------------------

namespace {

int utf8_len(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead >> 5) == 0x06) return 2;
    if ((lead >> 4) == 0x0E) return 3;
    if ((lead >> 3) == 0x1E) return 4;
    return 1;
}

}  // namespace

Phonemizer::~Phonemizer() {
    if (ready_) espeak_Terminate();
}

bool Phonemizer::init(const std::string& espeak_data_parent,
                      const std::string& vocab_tsv,
                      const std::string& voice) {
    int sr = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, /*buflength=*/0,
                               espeak_data_parent.c_str(), /*options=*/0);
    if (sr < 0) {
        std::fprintf(stderr,
                     "FATAL Phonemizer: espeak_Initialize failed (path=%s). Is "
                     "espeak-ng-data present on device?\n",
                     espeak_data_parent.c_str());
        return false;
    }
    if (espeak_SetVoiceByName(voice.c_str()) != EE_OK) {
        std::fprintf(stderr, "FATAL Phonemizer: espeak voice '%s' not found.\n",
                     voice.c_str());
        espeak_Terminate();
        return false;
    }

    std::ifstream in(vocab_tsv);
    if (!in) {
        std::fprintf(stderr, "FATAL Phonemizer: cannot open vocab tsv %s\n",
                     vocab_tsv.c_str());
        espeak_Terminate();
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string ph = line.substr(0, tab);
        int32_t id = (int32_t)std::strtol(line.c_str() + tab + 1, nullptr, 10);
        if (ph.empty()) { pad_id_ = id; continue; }
        vocab_.emplace_back(ph, id);
    }
    if (vocab_.empty()) {
        std::fprintf(stderr, "FATAL Phonemizer: vocab tsv %s yielded 0 entries\n",
                     vocab_tsv.c_str());
        espeak_Terminate();
        return false;
    }
    std::sort(vocab_.begin(), vocab_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    ready_ = true;
    std::fprintf(stderr, "Phonemizer ready: espeak sr=%d, %zu phonemes, voice=%s\n",
                 sr, vocab_.size(), voice.c_str());
    return true;
}

std::vector<int32_t> Phonemizer::phonemize(const std::string& text) const {
    std::vector<int32_t> ids;
    if (!ready_ || text.empty()) return ids;

    // phonememode bit 1 (0x02) = IPA UTF-8 output; espeak still emits a space at
    // word boundaries (matched against the vocab space token, id 16).
    std::string phonemes;
    const char* text_c = text.c_str();
    const void* inptr = static_cast<const void*>(text_c);
    while (inptr != nullptr) {
        const char* clause =
            espeak_TextToPhonemes(&inptr, espeakCHARS_UTF8, /*phonememode=*/0x02);
        if (clause && *clause) {
            if (!phonemes.empty()) phonemes += ' ';
            phonemes += clause;
        }
    }

    static std::set<std::string> warned;
    ids.push_back(pad_id_);
    size_t i = 0;
    while (i < phonemes.size()) {
        bool matched = false;
        for (const auto& kv : vocab_) {  // sorted longest-first → longest match
            const std::string& key = kv.first;
            if (key.size() <= phonemes.size() - i &&
                std::memcmp(phonemes.data() + i, key.data(), key.size()) == 0) {
                ids.push_back(kv.second);
                i += key.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            int adv = utf8_len((unsigned char)phonemes[i]);
            if (std::getenv("NNOPT_DEBUG_LAYERS")) {
                std::string sym = phonemes.substr(i, adv);
                if (warned.insert(sym).second)
                    std::fprintf(stderr, "Phonemizer: skipping unknown phoneme '%s'\n",
                                 sym.c_str());
            }
            i += adv;
        }
    }
    ids.push_back(pad_id_);
    return ids;
}

}  // namespace nnopt_tts

#endif  // NNOPT_TTS_STREAMING
