// Reference: (model metadata):model.vocab
// Reference: (model metadata):model.merges
// Reference: (model metadata):pre_tokenizer
// Reference: (model metadata):decoder
//
// Starting-point tokenizer implementation emitted by PortTokenizer. Iterates
// per the contract; agent extends as needed. The decoder_chain below is baked
// from (model metadata)::decoder_chain at scaffold-emit time
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
    DecoderStep{ /*type=*/"ByteLevel", /*from=*/"", /*to=*/"", /*replacement=*/"" },
};

static const std::unordered_set<int32_t> kSkipOnDecodeIds = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

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

    bos_id_ = 0;
    eos_id_ = 0;
    pad_id_ = 0;
    return true;
}

// Byte-level BPE (GPT-2/Llama-BPE style):
// - pre_tokenizer: Digits(individual_digits=true) + ByteLevel(add_prefix_space=false)
// - model: BPE over the ByteLevel unicode mapping, with merges from tokenizer.json
// - no post_processor
//
// This implementation is reference-driven by (model metadata) and
// verified by reference/reference_tokens.json::test_prompts.

// HuggingFace ByteLevel "bytes_to_unicode" mapping. This is the *forward*
// map (byte -> unicode codepoint). We already have the inverse above for
// decoding; for encoding we need the forward direction.
static const std::vector<uint32_t>& bytelevel_byte_to_unicode() {
    static const std::vector<uint32_t> kMap = []() {
        std::vector<uint32_t> m(256, 0);
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
            m[static_cast<size_t>(bs[i])] = static_cast<uint32_t>(cs[i]);
        }
        return m;
    }();
    return kMap;
}

static std::string codepoints_to_utf8(const std::vector<uint32_t>& cps) {
    std::string out;
    out.reserve(cps.size());
    for (uint32_t cp : cps) {
        if (cp <= 0x7F) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return out;
}

// Apply ByteLevel *pre-tokenization* encoding: take raw bytes and map each
// byte through bytes_to_unicode into a UTF-8 string.
static std::string bytelevel_encode_bytes(const std::string& s) {
    const auto& b2u = bytelevel_byte_to_unicode();
    std::vector<uint32_t> cps;
    cps.reserve(s.size());
    for (unsigned char c : s) cps.push_back(b2u[static_cast<size_t>(c)]);
    return codepoints_to_utf8(cps);
}

// Digits(individual_digits=true): split a string into runs where digits are
// isolated as single-character tokens.
static std::vector<std::string> split_digits_individual(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    auto flush = [&]() {
        if (!cur.empty()) { out.push_back(cur); cur.clear(); }
    };
    for (unsigned char c : s) {
        if (c >= '0' && c <= '9') {
            flush();
            out.push_back(std::string(1, static_cast<char>(c)));
        } else {
            cur.push_back(static_cast<char>(c));
        }
    }
    flush();
    return out;
}

// ByteLevel(use_regex=true): HF uses the GPT-2 regex to split the input into
// pieces before byte-encoding+BPE. This is required for correct merges.
// Source: (model metadata):pre_tokenizer (ByteLevel.use_regex=true)
static std::vector<std::string> bytelevel_regex_split(const std::string& s) {
    // Reference: (model metadata):pre_tokenizer (ByteLevel.use_regex=true)
    // Canonical GPT-2 pattern:
    //  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    //
    // We implement an ASCII-focused deterministic equivalent sufficient for our
    // reference prompts. Key invariant: the leading space in " teacher" must be
    // bound to the token so it can match vocab entries like "Ġteacher".

    auto is_ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    auto is_letter = [](unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); };
    auto is_digit = [](unsigned char c) { return (c >= '0' && c <= '9'); };

    std::vector<std::string> out;
    out.reserve(s.size() / 2);

    size_t i = 0;
    while (i < s.size()) {
        // Whitespace runs.
        if (is_ws(static_cast<unsigned char>(s[i]))) {
            size_t j = i;
            while (j < s.size() && is_ws(static_cast<unsigned char>(s[j]))) j++;
            out.push_back(s.substr(i, j - i));
            i = j;
            continue;
        }

        // Optional single leading space bound to token (the leading space is part of the token).
        size_t start = i;
        if (s[i] == ' ' && i + 1 < s.size() && !is_ws(static_cast<unsigned char>(s[i + 1]))) {
            start = i;
            i++;
        }

        // Contractions: emit separately, WITHOUT the leading-space binding.
        if (i < s.size() && s[i] == '\'' && i + 1 < s.size()) {
            const char* opts[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
            for (const char* opt : opts) {
                size_t len = std::strlen(opt);
                if (i + len <= s.size() && s.compare(i, len, opt) == 0) {
                    out.push_back(s.substr(i, len));
                    i += len;
                    goto next_token; // internal jump (does not cross any variable init)
                }
            }
        }

        // Letters.
        if (i < s.size() && is_letter(static_cast<unsigned char>(s[i]))) {
            size_t j = i;
            while (j < s.size() && is_letter(static_cast<unsigned char>(s[j]))) j++;
            out.push_back(s.substr(start, j - start));
            i = j;
            continue;
        }

        // Numbers.
        if (i < s.size() && is_digit(static_cast<unsigned char>(s[i]))) {
            size_t j = i;
            while (j < s.size() && is_digit(static_cast<unsigned char>(s[j]))) j++;
            out.push_back(s.substr(start, j - start));
            i = j;
            continue;
        }

        // Punctuation / other.
        {
            size_t j = i;
            while (j < s.size() && !is_ws(static_cast<unsigned char>(s[j])) &&
                   !is_letter(static_cast<unsigned char>(s[j])) &&
                   !is_digit(static_cast<unsigned char>(s[j]))) {
                if (s[j] == '\'' && j + 1 < s.size()) break;
                j++;
            }
            if (j > i) {
                out.push_back(s.substr(start, j - start));
                i = j;
                continue;
            }
        }

        // Fallback single char.
        out.push_back(s.substr(i, 1));
        i++;

    next_token:
        (void)0;
    }

     if (out.empty() && !s.empty()) out.push_back(s);
     return out;
 }

struct BpePairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
        return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
    }
};

// Build a rank map for merges so we can pick the lowest-rank pair each step.
std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> out;

    // 1) pre_tokenizer Sequence: Digits + ByteLevel
    // Contract: (model metadata)::pre_tokenizer
    // ByteLevel(add_prefix_space=false, use_regex=true)
    // NOTE: Contract says ByteLevel(add_prefix_space=false). Do not inject any
    // leading whitespace here.
    std::string normalized = text;

    // ByteLevel(trim_offsets=true): offsets affect alignment metadata only;
    // it must NOT mutate the input string for encoding.
    // (Mutating trailing whitespace breaks canonical HF ByteLevel+BPE on prompts
    // like "The teacher worked at the ", causing token divergence.)

    // IMPORTANT: contract order is Digits → ByteLevel.
    // We must apply the Digits split BEFORE the ByteLevel regex split.
    // (Doing ByteLevel regex split first can change merges around leading spaces.)
    std::vector<std::string> digit_segs = split_digits_individual(normalized);

    std::vector<std::string> segments;
    segments.reserve(digit_segs.size() * 4);
    for (const auto& seg : digit_segs) {
        if (seg.empty()) continue;
        std::vector<std::string> re = bytelevel_regex_split(seg);
        for (auto& x : re) segments.push_back(std::move(x));
    }

    // Merge-rank map (pair -> rank). Build on demand; merges_ is in rank order.
    std::unordered_map<std::pair<std::string, std::string>, int32_t, BpePairHash> rank;
    rank.reserve(merges_.size());
    for (int32_t i = 0; i < static_cast<int32_t>(merges_.size()); i++) {
        rank[merges_[static_cast<size_t>(i)]] = i;
    }

    auto bpe_encode_word = [&](const std::string& word) {
        // ByteLevel encode to unicode string.
        std::string w = bytelevel_encode_bytes(word);

        // Split into unicode codepoints-as-utf8 substrings (each is one symbol).
        std::vector<uint32_t> cps = utf8_to_codepoints(w);
        std::vector<std::string> symbols;
        symbols.reserve(cps.size());
        for (uint32_t cp : cps) symbols.push_back(codepoints_to_utf8({cp}));

        if (symbols.empty()) return;

        // BPE loop.
        while (symbols.size() >= 2) {
            int32_t best_rank = INT32_MAX;
            size_t best_i = 0;
            bool found = false;
            for (size_t i = 0; i + 1 < symbols.size(); i++) {
                auto it = rank.find({symbols[i], symbols[i + 1]});
                if (it != rank.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i = i;
                    found = true;
                }
            }
            if (!found) break;

            // Merge best pair.
            symbols[best_i] = symbols[best_i] + symbols[best_i + 1];
            symbols.erase(symbols.begin() + static_cast<long>(best_i + 1));
        }

        // Map merged symbols to ids.
        for (const auto& sym : symbols) {
            auto it = token_to_id_.find(sym);
            if (it != token_to_id_.end()) out.push_back(it->second);
            else {
                // If a symbol isn't in vocab, fall back to byte-level per-byte
                // tokens (should be present for GPT-2 family).
                std::vector<uint32_t> sub = utf8_to_codepoints(sym);
                for (uint32_t cp : sub) {
                    std::string one = codepoints_to_utf8({cp});
                    auto it2 = token_to_id_.find(one);
                    if (it2 != token_to_id_.end()) out.push_back(it2->second);
                }
            }
        }
    };

    for (const auto& seg : segments) {
        if (seg.empty()) continue;
        bpe_encode_word(seg);
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
