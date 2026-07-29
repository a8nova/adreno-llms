// Tokenizer method bodies (moved verbatim out of tokenizer.h).
#include "tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "json_mini.h"

void Tokenizer::load(const std::string& tokenizer_json_path) {
    std::ifstream f(tokenizer_json_path);
    std::stringstream ss;
    ss << f.rdbuf();
    auto root = jmini::parse(ss.str());
    const auto& toks = root->at("tokenizer.ggml.tokens").arr;
    vocab_.reserve(toks.size());
    for (size_t i = 0; i < toks.size(); ++i) {
        vocab_.push_back(toks[i]->s());
        tok2id_[toks[i]->s()] = (int)i;
    }
    const auto& types = root->at("tokenizer.ggml.token_type").arr;
    for (size_t i = 0; i < types.size(); ++i) {
        // Match BOTH literally, as llama.cpp does. CONTROL(3) is the obvious one
        // (<|im_start|>, <|im_end|>); USER_DEFINED(4) is easy to miss and matters just as much —
        // on this model <think>/</think> are type 4, and collecting only type 3 meant a prompt
        // containing "<think>" got BPE-split into '<','th','ink','>' instead of resolving to the
        // single token 248068. The model then sees text no training example ever contained and
        // answers with an immediate <|im_end|>, i.e. an empty reply with a healthy-looking
        // prefill/decode benchmark. Nothing errors — it just silently stops working.
        const long t = types[i]->i();
        if (t == 3 || t == 4)
            specials_.push_back({vocab_[i], (int)i});
    }
    const auto& merges = root->at("tokenizer.ggml.merges").arr;
    for (size_t r = 0; r < merges.size(); ++r) {
        const std::string& m = merges[r]->s();
        size_t sp = m.find(' ');
        merge_rank_[m.substr(0, sp) + "\x01" + m.substr(sp + 1)] = (int)r;
    }
    init_byte_maps();
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> out;
    // split on special-token literals first (longest match wins)
    size_t i = 0;
    std::string plain;
    auto flush = [&]() {
        if (!plain.empty()) { encode_plain(plain, out); plain.clear(); }
    };
    while (i < text.size()) {
        int sid = -1;
        size_t slen = 0;
        for (const auto& [s, id] : specials_)
            if (s.size() > slen && text.compare(i, s.size(), s) == 0) {
                sid = id; slen = s.size();
            }
        if (sid >= 0) { flush(); out.push_back(sid); i += slen; }
        else plain += text[i++];
    }
    flush();
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string bytes;
    for (int id : ids) {
        const std::string& t = vocab_[id];
        // token string is in byte-encoder space: map codepoints back
        size_t i = 0;
        while (i < t.size()) {
            uint32_t cp; int len;
            decode_utf8(t, i, &cp, &len);
            auto it = cp2byte_.find(cp);
            if (it != cp2byte_.end()) bytes += (char)it->second;
            i += len;
        }
    }
    return bytes;
}

// ---- GPT-2 byte encoder ----
void Tokenizer::init_byte_maps() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::vector<bool> in_bs(256, false);
    for (int b : bs) in_bs[b] = true;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = in_bs[b] ? (uint32_t)b : (uint32_t)(256 + n++);
        byte2cp_[b] = cp;
        cp2byte_[cp] = b;
    }
}

void Tokenizer::decode_utf8(const std::string& s, size_t i, uint32_t* cp, int* len) {
    const uint8_t c = s[i];
    if (c < 0x80) { *cp = c; *len = 1; }
    else if ((c >> 5) == 6) { *cp = ((c & 31) << 6) | (s[i+1] & 63); *len = 2; }
    else if ((c >> 4) == 14) {
        *cp = ((c & 15) << 12) | ((s[i+1] & 63) << 6) | (s[i+2] & 63); *len = 3;
    } else {
        *cp = ((c & 7) << 18) | ((s[i+1] & 63) << 12) | ((s[i+2] & 63) << 6) | (s[i+3] & 63);
        *len = 4;
    }
}

void Tokenizer::append_utf8(std::string& s, uint32_t cp) {
    if (cp < 0x80) s += (char)cp;
    else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 63)); }
    else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 63));
        s += (char)(0x80 | (cp & 63));
    } else {
        s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 63));
        s += (char)(0x80 | ((cp >> 6) & 63)); s += (char)(0x80 | (cp & 63));
    }
}

// ---- pre-tokenizer scanner over raw text codepoints ----
bool Tokenizer::is_space_cp(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0B ||
           c == 0x0C || c == 0xA0 || c == 0x2028 || c == 0x2029 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x3000;
}
bool Tokenizer::is_digit_cp(uint32_t c) { return c >= '0' && c <= '9'; }
bool Tokenizer::is_letter_cp(uint32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    if (is_space_cp(c)) return false;
    // approximation: general punctuation/symbol blocks are not letters
    if ((c >= 0x2000 && c <= 0x206F) || (c >= 0x20A0 && c <= 0x2BFF) ||
        (c >= 0x3000 && c <= 0x303F) || (c >= 0xFE30 && c <= 0xFE4F) ||
        (c >= 0xFF01 && c <= 0xFF20) || (c >= 0xA1 && c <= 0xBF) ||
        c == 0xD7 || c == 0xF7)
        return false;
    return true;
}

void Tokenizer::encode_plain(const std::string& text, std::vector<int>& out) const {
    // decode to codepoints once
    std::vector<uint32_t> cps;
    std::vector<size_t> offs;
    size_t i = 0;
    while (i < text.size()) {
        uint32_t cp; int len;
        decode_utf8(text, i, &cp, &len);
        cps.push_back(cp);
        offs.push_back(i);
        i += len;
    }
    offs.push_back(text.size());
    const size_t N = cps.size();
    size_t p = 0;
    auto piece = [&](size_t a, size_t b) {  // [a,b) codepoints
        bpe_piece(text.substr(offs[a], offs[b] - offs[a]), out);
    };
    while (p < N) {
        // 1. contractions
        if (cps[p] == '\'' && p + 1 < N) {
            uint32_t c1 = cps[p+1] | 32;
            if (c1 == 's' || c1 == 't' || c1 == 'd' || c1 == 'm') {
                piece(p, p + 2); p += 2; continue;
            }
            if (p + 2 < N) {
                uint32_t c2 = cps[p+2] | 32;
                if ((c1=='r'&&c2=='e')||(c1=='v'&&c2=='e')||(c1=='l'&&c2=='l')) {
                    piece(p, p + 3); p += 3; continue;
                }
            }
        }
        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        {
            size_t q = p;
            bool lead = false;
            if (!is_letter_cp(cps[q]) && !is_digit_cp(cps[q]) &&
                cps[q] != '\r' && cps[q] != '\n' && q + 1 < N &&
                is_letter_cp(cps[q + 1])) { lead = true; ++q; }
            if (is_letter_cp(cps[q])) {
                size_t e = q;
                while (e < N && is_letter_cp(cps[e])) ++e;
                piece(p, e); p = e; continue;
            }
            if (lead) { /* fallthrough with q reset */ }
        }
        // 3. single digit
        if (is_digit_cp(cps[p])) { piece(p, p + 1); ++p; continue; }
        // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            size_t q = p;
            if (cps[q] == ' ' && q + 1 < N && !is_space_cp(cps[q+1]) &&
                !is_letter_cp(cps[q+1]) && !is_digit_cp(cps[q+1])) ++q;
            if (q < N && !is_space_cp(cps[q]) && !is_letter_cp(cps[q]) &&
                !is_digit_cp(cps[q])) {
                size_t e = q;
                while (e < N && !is_space_cp(cps[e]) && !is_letter_cp(cps[e]) &&
                       !is_digit_cp(cps[e])) ++e;
                while (e < N && (cps[e] == '\r' || cps[e] == '\n')) ++e;
                piece(p, e); p = e; continue;
            }
        }
        // 5. \s*[\r\n]+
        {
            size_t q = p, lastnl = SIZE_MAX;
            while (q < N && is_space_cp(cps[q])) {
                if (cps[q] == '\r' || cps[q] == '\n') lastnl = q;
                else if (lastnl != SIZE_MAX) break;
                ++q;
            }
            if (lastnl != SIZE_MAX) {
                // include trailing run of \r\n only up to last consecutive nl
                size_t e = p;
                size_t run_end = p;
                while (run_end < N && is_space_cp(cps[run_end]) &&
                       run_end <= lastnl) ++run_end;
                e = run_end;
                piece(p, e); p = e; continue;
            }
        }
        // 6. \s+(?!\S)  |  7. \s+
        if (is_space_cp(cps[p])) {
            size_t e = p;
            while (e < N && is_space_cp(cps[e])) ++e;
            if (e < N && e - p > 1) --e;  // \s+(?!\S): leave one for next
            piece(p, e); p = e; continue;
        }
        piece(p, p + 1); ++p;  // safety: single cp
    }
}

void Tokenizer::bpe_piece(const std::string& raw, std::vector<int>& out) const {
    // map raw bytes into byte-encoder space, one symbol per byte
    std::vector<std::string> sym;
    for (unsigned char b : raw) {
        std::string s;
        append_utf8(s, byte2cp_.at(b));
        sym.push_back(s);
    }
    // greedy lowest-rank merges
    while (sym.size() > 1) {
        int best = INT32_MAX, bi = -1;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            auto it = merge_rank_.find(sym[i] + "\x01" + sym[i + 1]);
            if (it != merge_rank_.end() && it->second < best) {
                best = it->second; bi = (int)i;
            }
        }
        if (bi < 0) break;
        sym[bi] += sym[bi + 1];
        sym.erase(sym.begin() + bi + 1);
    }
    for (const auto& s : sym) {
        auto it = tok2id_.find(s);
        if (it != tok2id_.end()) { out.push_back(it->second); continue; }
        // unknown symbol: fall back to per-byte tokens
        size_t i = 0;
        while (i < s.size()) {
            uint32_t cp; int len;
            decode_utf8(s, i, &cp, &len);
            std::string one;
            append_utf8(one, cp);
            auto jt = tok2id_.find(one);
            if (jt != tok2id_.end()) out.push_back(jt->second);
            else fprintf(stderr, "tokenizer: unmappable symbol\n");
            i += len;
        }
    }
}
