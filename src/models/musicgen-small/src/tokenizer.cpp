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
    DecoderStep{ /*type=*/"Metaspace", /*from=*/"", /*to=*/"", /*replacement=*/"▁" },
};

static const std::unordered_set<int32_t> kSkipOnDecodeIds = { 0, 1, 2, 32000, 32001, 32002, 32003, 32004, 32005, 32006, 32007, 32008, 32009, 32010, 32011, 32012, 32013, 32014, 32015, 32016, 32017, 32018, 32019, 32020, 32021, 32022, 32023, 32024, 32025, 32026, 32027, 32028, 32029, 32030, 32031, 32032, 32033, 32034, 32035, 32036, 32037, 32038, 32039, 32040, 32041, 32042, 32043, 32044, 32045, 32046, 32047, 32048, 32049, 32050, 32051, 32052, 32053, 32054, 32055, 32056, 32057, 32058, 32059, 32060, 32061, 32062, 32063, 32064, 32065, 32066, 32067, 32068, 32069, 32070, 32071, 32072, 32073, 32074, 32075, 32076, 32077, 32078, 32079, 32080, 32081, 32082, 32083, 32084, 32085, 32086, 32087, 32088, 32089, 32090, 32091, 32092, 32093, 32094, 32095, 32096, 32097, 32098, 32099 };

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

    // Optional trailing section: 'SCOR' magic + float32 piece log-prob per id
    // (SentencePiece unigram scores; exported from tokenizer.json:model.vocab).
    {
        char magic[4] = {0};
        in.read(magic, 4);
        if (in && std::memcmp(magic, "SCOR", 4) == 0) {
            scores_.assign(id_to_token_.size(), -1e9f);
            in.read(reinterpret_cast<char*>(scores_.data()),
                    (std::streamsize)(scores_.size() * sizeof(float)));
            if (!in) scores_.clear();   // truncated → fall back to greedy
        }
    }

    bos_id_ = 0;
    eos_id_ = 1;
    pad_id_ = 0;
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
    static const char kSpaceMarker[] = "▁"; // UTF-8 E2 96 81 (the scaffold emitted this DOUBLE-encoded â - mojibake; fixed 2026-06-05) for U+2581
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
    if (!scores_.empty()) {
        // ── SentencePiece UNIGRAM (Viterbi) ────────────────────────────────
        // best[i] = max total log-prob segmenting staged[0..i); back[i] =
        // (start, id) of the piece ending at i. Matches HuggingFace's T5
        // tokenization (greedy longest-match diverges on real prompts — the
        // old path also never worked: the vocab file format was mismatched).
        const size_t n = staged.size();
        const size_t kMaxPiece = 64;   // bytes; T5 pieces are far shorter
        std::vector<float> best(n + 1, -1e30f);
        std::vector<int32_t> back_id(n + 1, -1);
        std::vector<size_t> back_start(n + 1, 0);
        best[0] = 0.0f;
        for (size_t i = 0; i < n; i++) {
            if (best[i] <= -1e29f) continue;
            const size_t maxspan = std::min(n - i, kMaxPiece);
            bool matched = false;
            for (size_t span = 1; span <= maxspan; span++) {
                auto it = token_to_id_.find(staged.substr(i, span));
                if (it == token_to_id_.end()) continue;
                matched = true;
                const float sc = best[i] + scores_[(size_t)it->second];
                if (sc > best[i + span]) {
                    best[i + span] = sc;
                    back_id[i + span] = it->second;
                    back_start[i + span] = i;
                }
            }
            if (!matched) {
                // No piece starts here: consume one byte as <unk> (id 2) with
                // a heavy penalty so real segmentations always win.
                const float sc = best[i] - 100.0f;
                if (sc > best[i + 1]) {
                    best[i + 1] = sc;
                    back_id[i + 1] = 2;
                    back_start[i + 1] = i;
                }
            }
        }
        std::vector<int32_t> rev;
        for (size_t i = n; i > 0; i = back_start[i]) {
            if (back_id[i] < 0) { rev.clear(); break; }   // no path — give up
            rev.push_back(back_id[i]);
            if (back_start[i] == 0) break;
        }
        for (size_t k = rev.size(); k > 0; k--) out.push_back(rev[k - 1]);
        if (!out.empty()) out.push_back(eos_id_);   // T5 appends </s>
        return out;
    }
    size_t i = 0;
    while (i < staged.size()) {
        // Greedy longest match against vocab (fallback when no scores section).
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
            i++;
            continue;
        }
        out.push_back(best_id);
        i += best_len;
    }
    if (!out.empty()) out.push_back(eos_id_);   // T5 appends </s>
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
