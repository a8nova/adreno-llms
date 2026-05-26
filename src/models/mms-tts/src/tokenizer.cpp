// Reference: ports/adreno-llm-multilang/scripts/convert_mms.py (VTMV format)
//
// MMS-TTS character-level tokenizer. Loads weights/<lang>/tokenizer_vocab.bin
// produced by convert_mms.py (VTMV format) and produces int32 ids from a
// UTF-8 prompt. For languages whose vocab is Latin/romanized (amh, kor, …),
// runs uroman first per the `is_uroman` header flag.
//
// VITS interleaves a blank token (id from vocab header) between every
// emitted id when add_blank=true (the default for MMS-TTS).

#include "tokenizer.h"
#include "uroman.h"
#include "debug_utils.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace {

constexpr uint32_t VOCAB_MAGIC   = 0x564D5456u;   // 'VTMV'
constexpr uint32_t VOCAB_VERSION = 1u;

// Module-local state. The Tokenizer class instance has no fields per the
// scaffold's .h (it only exposes the four methods); store decoded vocab in
// a file-scope struct keyed on the loaded path. One instance per process
// is the documented usage so this is fine.
struct LoadedVocab {
    std::unordered_map<std::string, int32_t> token_to_id;
    int32_t pad_id   = -1;
    int32_t unk_id   = -1;
    int32_t blank_id = -1;
    bool    add_blank = true;
    bool    is_uroman = false;
    bool    loaded    = false;
};
LoadedVocab g_vocab;

bool read_u32(std::istream& s, uint32_t& v) { return static_cast<bool>(s.read(reinterpret_cast<char*>(&v), 4)); }
bool read_i32(std::istream& s, int32_t&  v) { return static_cast<bool>(s.read(reinterpret_cast<char*>(&v), 4)); }
bool read_u16(std::istream& s, uint16_t& v) { return static_cast<bool>(s.read(reinterpret_cast<char*>(&v), 2)); }
bool read_u8 (std::istream& s, uint8_t&  v) { return static_cast<bool>(s.read(reinterpret_cast<char*>(&v), 1)); }

// Length of a UTF-8 codepoint starting at byte b0. Returns 0 for invalid
// lead bytes — caller advances by 1 to recover.
int utf8_len(uint8_t b0) {
    if (b0 < 0x80)               return 1;
    if ((b0 & 0xE0) == 0xC0)     return 2;
    if ((b0 & 0xF0) == 0xE0)     return 3;
    if ((b0 & 0xF8) == 0xF0)     return 4;
    return 0;
}

}  // namespace

bool Tokenizer::load(const std::string& vocab_path) {
    g_vocab = LoadedVocab{};
    std::ifstream f(vocab_path, std::ios::binary);
    if (!f) {
        NNOPT_ERROR_FMT("Tokenizer::load failed to open %s", vocab_path.c_str());
        return false;
    }

    uint32_t magic, version, n;
    int32_t  pad_id, unk_id, blank_id;
    uint8_t  add_blank, is_uroman;
    uint16_t reserved;

    if (!read_u32(f, magic)   || magic   != VOCAB_MAGIC)   { NNOPT_ERROR("vocab magic mismatch");   return false; }
    if (!read_u32(f, version) || version != VOCAB_VERSION) { NNOPT_ERROR("vocab version mismatch"); return false; }
    if (!read_u32(f, n))                                   { NNOPT_ERROR("vocab truncated header"); return false; }
    if (!read_i32(f, pad_id) || !read_i32(f, unk_id) || !read_i32(f, blank_id)) {
        NNOPT_ERROR("vocab truncated special-ids"); return false;
    }
    if (!read_u8(f, add_blank) || !read_u8(f, is_uroman) || !read_u16(f, reserved)) {
        NNOPT_ERROR("vocab truncated flags"); return false;
    }

    g_vocab.pad_id    = pad_id;
    g_vocab.unk_id    = unk_id;
    g_vocab.blank_id  = blank_id;
    g_vocab.add_blank = add_blank != 0;
    g_vocab.is_uroman = is_uroman != 0;
    g_vocab.token_to_id.reserve(n);

    for (uint32_t i = 0; i < n; ++i) {
        int32_t  id;
        uint32_t utf8_n;
        if (!read_i32(f, id) || !read_u32(f, utf8_n)) {
            NNOPT_ERROR_FMT("vocab truncated at entry %u", i); return false;
        }
        if (utf8_n > 64) { NNOPT_ERROR("vocab entry oversize"); return false; }
        std::string tok(utf8_n, '\0');
        if (utf8_n > 0 && !f.read(tok.data(), utf8_n)) {
            NNOPT_ERROR_FMT("vocab body truncated at entry %u", i); return false;
        }
        g_vocab.token_to_id.emplace(std::move(tok), id);
    }
    g_vocab.loaded = true;
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text_in) {
    std::vector<int> ids;
    if (!g_vocab.loaded) {
        NNOPT_ERROR("Tokenizer::encode called before successful load()");
        return ids;
    }

    // Step 1: uroman (Latin-vocab langs only).
    std::string text = g_vocab.is_uroman
                         ? adreno_llm::uroman::romanize(text_in)
                         : text_in;

    // Step 2: char-level lookup. Drop unknowns (matches HF behaviour for
    // tokenization_vits.py — it filters then tokenizes).
    ids.reserve(text.size() * (g_vocab.add_blank ? 2 : 1) + 2);

    // HF's tokenization_vits.py interleaves the PAD token id between every
    // char (and one at the start and end), producing [pad, c0, pad, c1, …,
    // pad, cN-1, pad] — 2N+1 ids for N chars. Naming gotcha: the token named
    // "_" in some vocabs is NOT what HF interleaves; HF uses `pad_token_id`,
    // which is e.g. id 0 → "c" for facebook/mms-tts-amh.
    const int32_t interleave_id =
        (g_vocab.pad_id >= 0) ? g_vocab.pad_id :
        (g_vocab.blank_id >= 0 ? g_vocab.blank_id : -1);
    auto emit = [&](int32_t id) {
        if (g_vocab.add_blank && interleave_id >= 0) {
            ids.push_back(interleave_id);
        }
        ids.push_back(id);
    };

    size_t i = 0;
    while (i < text.size()) {
        const int n = utf8_len(static_cast<uint8_t>(text[i]));
        if (n <= 0) { ++i; continue; }
        if (i + static_cast<size_t>(n) > text.size()) break;
        std::string key(text.substr(i, static_cast<size_t>(n)));
        auto it = g_vocab.token_to_id.find(key);
        if (it != g_vocab.token_to_id.end()) {
            emit(it->second);
        }
        // else: drop. HF's tokenizer filters before tokenizing for VITS.
        i += static_cast<size_t>(n);
    }

    // Trailing pad-blank (matches HF's add_blank behaviour).
    if (g_vocab.add_blank && interleave_id >= 0 && !ids.empty()) {
        ids.push_back(interleave_id);
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& /*ids*/) {
    // VITS doesn't decode ids back to text — output is raw audio.
    return "";
}

int Tokenizer::eos_token_id() const {
    // VITS has no EOS. The scaffold uses this for sampler config which is
    // unused for non-autoregressive TTS, so -1 is safe.
    return -1;
}
