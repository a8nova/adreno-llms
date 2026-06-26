// See tokenizer.h. Unigram Viterbi over codepoints; Metaspace add_dummy_prefix +
// space→U+2581; byte fallback via the <0xHH> pieces. Matches the SentencePiece
// reference on English text (validated in scripts/convert_tokenizer.py harness).
#include "tokenizer.h"
#include <fstream>
#include <cstdint>
#include <cstring>

static uint32_t rd_u32(std::ifstream& f) { uint32_t v = 0; f.read((char*)&v, 4); return v; }

// Split a UTF-8 string into its codepoint substrings (1 entry per codepoint).
static std::vector<std::string> to_codepoints(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        size_t len = (c < 0x80) ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
        if (i + len > n) len = 1;
        out.emplace_back(s.substr(i, len));
        i += len;
    }
    return out;
}

bool Tokenizer::load(const std::string& vocab_path) {
    std::ifstream f(vocab_path, std::ios::binary);
    if (!f) return false;
    uint32_t magic = rd_u32(f), version = rd_u32(f), V = rd_u32(f);
    if (magic != 0x4B4F5450 || version != 1 || V == 0) return false;
    unk_ = (int)rd_u32(f); bos_ = (int)rd_u32(f); eos_ = (int)rd_u32(f); pad_ = (int)rd_u32(f);
    for (int b = 0; b < 256; ++b) { uint32_t v = rd_u32(f); byte_fallback_[b] = (v == 0xFFFFFFFFu) ? -1 : (int)v; }
    score_.resize(V);
    piece2id_.reserve(V * 2);
    for (uint32_t id = 0; id < V; ++id) {
        float sc; uint16_t blen;
        f.read((char*)&sc, 4); f.read((char*)&blen, 2);
        std::string piece(blen, '\0');
        if (blen) f.read(&piece[0], blen);
        if (!f) return false;
        score_[id] = sc;
        piece2id_[piece] = (int)id;
        int cp = (int)to_codepoints(piece).size();
        if (cp > max_piece_cp_) max_piece_cp_ = cp;
    }
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    // Metaspace: add_dummy_prefix=true, spaces → U+2581 ("▁").
    std::string norm = "\xE2\x96\x81";   // leading ▁
    for (char ch : text) { if (ch == ' ') norm += "\xE2\x96\x81"; else norm.push_back(ch); }
    std::vector<std::string> u = to_codepoints(norm);
    const int N = (int)u.size();
    const double NEG = -1e18;
    std::vector<double> best(N + 1, NEG);
    std::vector<int> bp(N + 1, -1);          // back-pointer position
    std::vector<int> btok(N + 1, -1);        // token id used (or -2 = byte fallback marker)
    best[0] = 0.0;
    for (int i = 0; i < N; ++i) {
        if (best[i] == NEG) continue;
        // try normal pieces of length 1..max_piece_cp_
        std::string sub;
        int maxL = (N - i < max_piece_cp_) ? (N - i) : max_piece_cp_;
        for (int L = 1; L <= maxL; ++L) {
            sub += u[i + L - 1];
            auto it = piece2id_.find(sub);
            if (it == piece2id_.end()) continue;
            int id = it->second;
            if (id == unk_ || id == bos_ || id == eos_ || id == pad_) continue;
            double sc = best[i] + score_[id];
            if (sc > best[i + L]) { best[i + L] = sc; bp[i + L] = i; btok[i + L] = id; }
        }
        // byte fallback for a single codepoint (low priority): every byte must have a <0xHH> piece
        bool ok = true;
        for (unsigned char b : u[i]) if (byte_fallback_[b] < 0) { ok = false; break; }
        if (ok) {
            double sc = best[i] - 10.0;   // matches the prototype's fallback penalty ordering
            if (sc > best[i + 1]) { best[i + 1] = sc; bp[i + 1] = i; btok[i + 1] = -2; }
        }
    }
    // backtrack
    std::vector<int> ids;
    int i = N;
    while (i > 0 && bp[i] >= 0) {
        int pi = bp[i], tok = btok[i];
        if (tok == -2) {
            std::vector<int> bytes;
            for (unsigned char b : u[pi]) bytes.push_back(byte_fallback_[b]);
            ids.insert(ids.begin(), bytes.begin(), bytes.end());
        } else {
            ids.insert(ids.begin(), tok);
        }
        i = pi;
    }
    return ids;
}
