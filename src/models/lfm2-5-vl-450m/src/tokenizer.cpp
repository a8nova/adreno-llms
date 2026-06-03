// Reference: .nnport/tokenizer.json:model.vocab
// Reference: .nnport/tokenizer.json:decoder
//
// Starting-point tokenizer implementation emitted by PortTokenizer. Iterates
// per the contract; agent extends as needed. The decoder_chain below is baked
// from .nnport/tokenizer_contract.json::decoder_chain at scaffold-emit time
// and covers ByteLevel, Metaspace, Replace, Strip, ByteFallback, and Fuse —
// the most common HuggingFace decoder steps. Add cases below if the contract
// emits a step type not handled here.

#include "tokenizer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

// ── Step record baked from contract.decoder_chain ─────────────────────────
struct DecoderStep {
    std::string type;
    std::string from;
    std::string to;
    std::string replacement;
};

static const std::vector<DecoderStep> kDecoderChain = {
    DecoderStep{ /*type=*/"ByteLevel", /*from=*/"", /*to=*/"", /*replacement=*/"" },
};

static const std::unordered_set<int32_t> kSkipOnDecodeIds = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 64394, 64395, 64396, 64397, 64398, 64399 };

// ── Stream-helper: read a little-endian uint32 ────────────────────────────
static bool read_u32_le(std::ifstream& in, uint32_t& v) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) return false;
    v = static_cast<uint32_t>(b[0])
      | (static_cast<uint32_t>(b[1]) << 8)
      | (static_cast<uint32_t>(b[2]) << 16)
      | (static_cast<uint32_t>(b[3]) << 24);
    return true;
}

// ── ByteLevel decoder map (HuggingFace bytes_to_unicode inverted) ─────────
// HF maps each byte 0..255 to a unique Unicode code point so that all bytes
// can round-trip through string handling. To decode, we reverse: each Unicode
// codepoint in the chain output maps back to a byte. This is the standard
// GPT-2 / RoBERTa / Llama-BPE table.
static const std::unordered_map<uint32_t, uint8_t>& bytelevel_unicode_to_byte() {
    static const std::unordered_map<uint32_t, uint8_t> kMap = []() {
        std::unordered_map<uint32_t, uint8_t> m;
        // Bytes that are printable ASCII in HF's table map to themselves.
        std::vector<int> bs;
        for (int b = 33; b <= 126; b++) bs.push_back(b);
        for (int b = 161; b <= 172; b++) bs.push_back(b);
        for (int b = 174; b <= 255; b++) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }
        for (size_t i = 0; i < bs.size(); i++) {
            m[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
        }
        return m;
    }();
    return kMap;
}

// Decode a UTF-8 string into a sequence of Unicode codepoints.
static std::vector<uint32_t> utf8_to_codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp = 0;
        size_t len = 1;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else { i++; continue; }
        if (i + len > s.size()) break;
        for (size_t k = 1; k < len; k++) {
            unsigned char cc = static_cast<unsigned char>(s[i + k]);
            cp = (cp << 6) | (cc & 0x3F);
        }
        out.push_back(cp);
        i += len;
    }
    return out;
}

// Apply HuggingFace ByteLevel decode: codepoint sequence → byte sequence string.
static std::string apply_bytelevel(const std::string& concatenated) {
    const auto& m = bytelevel_unicode_to_byte();
    std::vector<uint32_t> cps = utf8_to_codepoints(concatenated);
    std::string out;
    out.reserve(cps.size());
    for (uint32_t cp : cps) {
        auto it = m.find(cp);
        if (it != m.end()) out.push_back(static_cast<char>(it->second));
        // Otherwise drop the codepoint silently — keeps output ASCII-clean
        // when the chain is misconfigured. Agent should diagnose if needed.
    }
    return out;
}

// Apply Metaspace decode: replace each occurrence of the replacement token
// (typically "▁", U+2581) with a space. Emitted by SentencePiece-family
// tokenizers (LLaMA, T5, etc.).
static std::string apply_metaspace(const std::string& s, const std::string& replacement) {
    if (replacement.empty()) return s;
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s.compare(i, replacement.size(), replacement) == 0) {
            out.push_back(' ');
            i += replacement.size();
        } else {
            out.push_back(s[i]);
            i++;
        }
    }
    return out;
}

// Apply Replace decode: simple string replacement.
static std::string apply_replace(const std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s.compare(i, from.size(), from) == 0) {
            out.append(to);
            i += from.size();
        } else {
            out.push_back(s[i]);
            i++;
        }
    }
    return out;
}

// Apply Strip decode: trim whitespace from one or both sides.
static std::string apply_strip(const std::string& s, const std::string& direction) {
    auto is_ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    size_t start = 0, end = s.size();
    if (direction != "right") {
        while (start < end && is_ws(static_cast<unsigned char>(s[start]))) start++;
    }
    if (direction != "left") {
        while (end > start && is_ws(static_cast<unsigned char>(s[end - 1]))) end--;
    }
    return s.substr(start, end - start);
}

// Apply ByteFallback decode: replace each occurrence of "<0xHH>" with the byte HH.
static std::string apply_byte_fallback(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (i + 6 <= s.size() && s[i] == '<' && s[i + 1] == '0' && s[i + 2] == 'x' && s[i + 5] == '>') {
            char h1 = s[i + 3], h2 = s[i + 4];
            auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
                if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
                return -1;
            };
            int v1 = hex(h1), v2 = hex(h2);
            if (v1 >= 0 && v2 >= 0) {
                out.push_back(static_cast<char>((v1 << 4) | v2));
                i += 6;
                continue;
            }
        }
        out.push_back(s[i]);
        i++;
    }
    return out;
}

static std::string apply_decoder_chain(const std::string& concatenated) {
    std::string s = concatenated;
    for (const auto& step : kDecoderChain) {
        if (step.type == "ByteLevel") {
            s = apply_bytelevel(s);
        } else if (step.type == "Metaspace") {
            s = apply_metaspace(s, step.replacement);
        } else if (step.type == "Replace") {
            s = apply_replace(s, step.from, step.to);
        } else if (step.type == "Strip") {
            // step.from used as direction by some HF dumps; fallback to "both".
            s = apply_strip(s, step.from);
        } else if (step.type == "ByteFallback") {
            s = apply_byte_fallback(s);
        } else if (step.type == "Fuse") {
            // No-op for plain string concatenation.
        }
        // Unknown step types fall through. Add a case above if your contract
        // emits a step type not handled here.
    }
    return s;
}

}  // namespace

bool Tokenizer::load(const std::string& vocab_path) {
    std::ifstream in(vocab_path, std::ios::binary);
    if (!in) return false;

    uint32_t vocab_count = 0, merges_count = 0;
    if (!read_u32_le(in, vocab_count) || !read_u32_le(in, merges_count)) return false;

    id_to_token_.assign(vocab_count, std::string());
    token_to_id_.clear();
    token_to_id_.reserve(vocab_count);

    for (uint32_t k = 0; k < vocab_count; k++) {
        uint32_t id = 0, byte_len = 0;
        if (!read_u32_le(in, id)) return false;
        if (!read_u32_le(in, byte_len)) return false;
        std::string tok(byte_len, '\0');
        if (byte_len > 0) {
            in.read(&tok[0], byte_len);
            if (!in) return false;
        }
        if (id >= id_to_token_.size()) {
            id_to_token_.resize(id + 1, std::string());
        }
        id_to_token_[id] = tok;
        token_to_id_[tok] = static_cast<int32_t>(id);
    }

    merges_.clear();
    merges_.reserve(merges_count);
    for (uint32_t k = 0; k < merges_count; k++) {
        uint32_t a_len = 0, b_len = 0;
        if (!read_u32_le(in, a_len)) return false;
        std::string a(a_len, '\0');
        if (a_len > 0) { in.read(&a[0], a_len); if (!in) return false; }
        if (!read_u32_le(in, b_len)) return false;
        std::string b(b_len, '\0');
        if (b_len > 0) { in.read(&b[0], b_len); if (!in) return false; }
        merges_.emplace_back(std::move(a), std::move(b));
    }

    bos_id_ = 2;
    eos_id_ = 2;
    pad_id_ = 2;
    return true;
}

// Greedy longest-prefix tokenization. This is intentionally simple — it works
// well enough for the convergence loop's text-prompt mode in many models, but
// will diverge from HuggingFace's BPE / SentencePiece / WordPiece tokenizers
// on edge cases. Iterate on this method as the model family requires; the
// reference signal is reference/reference_tokens.json::input_ids.
// Detect whether the baked decoder chain treats "▁" (U+2581) as a space
// marker. If so, the matching ENCODE side must pre-process the input by
// prepending "▁" and converting interior spaces to "▁" — otherwise vocab
// lookups for tokens like "▁The", "▁teacher" will miss and the prompt
// round-trips back as "Theteacherworkedatthe" with no spaces. This is the
// standard SentencePiece convention shared by Llama / Mistral / OpenELM /
// Gemma / Phi.
static bool sentencepiece_metaspace_active() {
    for (const auto& step : kDecoderChain) {
        if ((step.type == "Metaspace") || (step.type == "Replace" && step.from == "â")) {
            return true;
        }
    }
    return false;
}

static std::string sentencepiece_preprocess(const std::string& text) {
    // Convention: prepend "▁" to the start of the string, and replace every
    // ASCII space with "▁". Internal whitespace runs collapse to one "▁"
    // each. The token vocab carries entries like "▁The"; this transform
    // makes greedy-longest-match find them.
    static const char kSpaceMarker[] = "â"; // UTF-8 for U+2581
    std::string out;
    out.reserve(text.size() + 4);
    out.append(kSpaceMarker);
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == ' ') {
            out.append(kSpaceMarker);
        } else {
            out.push_back(text[i]);
        }
    }
    return out;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> out;
    out.reserve(text.size());
    // Apply SentencePiece-family preprocessing when the decoder chain
    // expects "▁" markers — without this, prompt encode → decode loses
    // every space because the vocab stores "▁word", not "word".
    const std::string staged = sentencepiece_metaspace_active()
        ? sentencepiece_preprocess(text)
        : text;
    size_t i = 0;
    while (i < staged.size()) {
        // Greedy longest match against vocab.
        size_t best_len = 0;
        int32_t best_id = -1;
        for (size_t span = std::min<size_t>(staged.size() - i, 32); span >= 1; span--) {
            auto it = token_to_id_.find(staged.substr(i, span));
            if (it != token_to_id_.end()) {
                best_len = span;
                best_id = it->second;
                break;
            }
        }
        if (best_id < 0) {
            // Unknown byte — emit no token and advance one byte. Agent should
            // extend this with the family-appropriate fallback (ByteFallback,
            // <unk>, byte-pair merges, etc.) for production correctness.
            i++;
            continue;
        }
        out.push_back(best_id);
        i += best_len;
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& token_ids) const {
    std::string concatenated;
    concatenated.reserve(token_ids.size() * 4);
    for (int32_t id : token_ids) {
        if (kSkipOnDecodeIds.find(id) != kSkipOnDecodeIds.end()) continue;
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) continue;
        concatenated.append(id_to_token_[static_cast<size_t>(id)]);
    }
    return apply_decoder_chain(concatenated);
}

int32_t Tokenizer::token_to_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it == token_to_id_.end()) return -1;
    return it->second;
}

namespace {

// HuggingFace ByteLevel ENCODE side: map each input byte 0x00..0xFF to a
// Unicode codepoint (same table used in decode, but inverted). The output is
// a UTF-8 string. This is required to match HF BPE tokenizers: the vocab
// stores entries like "Ġthis" (U+0120 + "this") for " this", and "Ċ"
// (U+010A) for "\n", so we must apply this transform before vocab lookup.
static const std::array<std::string, 256>& bytelevel_byte_to_utf8() {
    static const std::array<std::string, 256> kTable = []() {
        std::array<std::string, 256> t;
        std::vector<int> bs;
        for (int b = 33; b <= 126; b++) bs.push_back(b);
        for (int b = 161; b <= 172; b++) bs.push_back(b);
        for (int b = 174; b <= 255; b++) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }
        auto cp_to_utf8 = [](uint32_t cp) -> std::string {
            std::string s;
            if (cp < 0x80) {
                s.push_back(static_cast<char>(cp));
            } else if (cp < 0x800) {
                s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else if (cp < 0x10000) {
                s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else {
                s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            return s;
        };
        for (size_t i = 0; i < bs.size(); i++) {
            t[static_cast<size_t>(bs[i])] = cp_to_utf8(static_cast<uint32_t>(cs[i]));
        }
        return t;
    }();
    return kTable;
}

// Apply HuggingFace ByteLevel encode: each byte → its Unicode codepoint as
// UTF-8. Used by encode_with_image for sub-string tokenization where the
// chat-template machinery needs exact HF-compatible token ids.
static std::string apply_bytelevel_encode(const std::string& s) {
    const auto& tbl = bytelevel_byte_to_utf8();
    std::string out;
    out.reserve(s.size() * 2);
    for (unsigned char c : s) {
        out.append(tbl[c]);
    }
    return out;
}

}  // namespace

// Greedy-longest-match encode over the ByteLevel-encoded form of `text`.
// This is the HF-compatible path used by encode_with_image; it differs from
// the public encode() by applying the standard GPT-2/Llama-BPE ByteLevel
// transform first (space → "Ġ", "\n" → "Ċ", etc.) so vocab lookups for
// tokens like "Ġthis", "Ġimage", "Ċ" hit.
//
// Note: this does NOT run the full BPE-merge algorithm. The on-disk
// tokenizer_vocab.bin has merges_count=0 — the vocab is already the
// merged-expanded final set. Greedy longest match over the ByteLevel-encoded
// form produces byte-identical output to HF for typical English prompts;
// extend with merge-rank BPE if a future prompt exposes a divergence.
static std::vector<int32_t> bytelevel_greedy_encode(
    const std::string& text,
    const std::unordered_map<std::string, int32_t>& token_to_id) {
    std::vector<int32_t> out;
    if (text.empty()) return out;
    const std::string staged = apply_bytelevel_encode(text);
    size_t i = 0;
    while (i < staged.size()) {
        size_t best_len = 0;
        int32_t best_id = -1;
        size_t max_span = std::min<size_t>(staged.size() - i, 64);
        for (size_t span = max_span; span >= 1; span--) {
            auto it = token_to_id.find(staged.substr(i, span));
            if (it != token_to_id.end()) {
                best_len = span;
                best_id = it->second;
                break;
            }
        }
        if (best_id < 0) {
            // No vocab hit even for a single byte's ByteLevel codepoint —
            // this shouldn't happen with a well-formed BPE vocab, but skip
            // forward to avoid an infinite loop and let the agent diagnose.
            i++;
            continue;
        }
        out.push_back(best_id);
        i += best_len;
    }
    return out;
}

std::vector<int32_t> Tokenizer::encode_with_image(
    const std::string& prompt,
    int grid_h, int grid_w,
    int thumbnail_spatial_h, int thumbnail_spatial_w,
    int tile_tokens,
    bool use_thumbnail,
    bool add_generation_prompt) const {
    // Resolve special-token ids by name. These IDs are stable for the
    // LFM2-VL tokenizer (see model_info/tokenizer_config.json + vocab dump):
    //   <|startoftext|>=1, <|im_start|>=6, <|im_end|>=7, <image>=396
    // We still look them up by string so a future tokenizer revision that
    // shifts the IDs doesn't silently miscompare.
    const int32_t bos       = token_to_id("<|startoftext|>");
    const int32_t im_start  = token_to_id("<|im_start|>");
    const int32_t im_end    = token_to_id("<|im_end|>");
    const int32_t img_token = token_to_id("<image>");

    // Number of <image> tokens per the reference processor:
    //   per_tile = tile_tokens (= 256 for tile_size=512, patch=16, downsample=2)
    //   thumbnail = (h/2)*(w/2) where (h,w) is the thumbnail patch grid
    const int n_tiles = grid_h * grid_w;
    int n_image_tokens = n_tiles * tile_tokens;
    if (use_thumbnail && thumbnail_spatial_h > 0 && thumbnail_spatial_w > 0) {
        n_image_tokens += (thumbnail_spatial_h / 2) * (thumbnail_spatial_w / 2);
    }

    // Tokenize sub-strings via the ByteLevel-aware path so " this" → "Ġthis"
    // (id 1033), "\n" → "Ċ" (id 708), etc.
    const std::vector<int32_t> user_ids      = bytelevel_greedy_encode("user", token_to_id_);
    const std::vector<int32_t> newline_ids   = bytelevel_greedy_encode("\n", token_to_id_);
    const std::vector<int32_t> text_ids      = bytelevel_greedy_encode(prompt, token_to_id_);
    const std::vector<int32_t> assistant_ids = bytelevel_greedy_encode("assistant", token_to_id_);

    std::vector<int32_t> out;
    out.reserve(8 + user_ids.size() + newline_ids.size() * 3
                + n_image_tokens + text_ids.size() + assistant_ids.size());

    // <|startoftext|><|im_start|>user\n
    out.push_back(bos);
    out.push_back(im_start);
    out.insert(out.end(), user_ids.begin(), user_ids.end());
    out.insert(out.end(), newline_ids.begin(), newline_ids.end());

    // <image>*N
    for (int k = 0; k < n_image_tokens; ++k) out.push_back(img_token);

    // {prompt}<|im_end|>\n
    out.insert(out.end(), text_ids.begin(), text_ids.end());
    out.push_back(im_end);
    out.insert(out.end(), newline_ids.begin(), newline_ids.end());

    // <|im_start|>assistant\n
    if (add_generation_prompt) {
        out.push_back(im_start);
        out.insert(out.end(), assistant_ids.begin(), assistant_ids.end());
        out.insert(out.end(), newline_ids.begin(), newline_ids.end());
    }

    return out;
}
