// SentencePiece UNIGRAM encoding — see spm_tokenizer.h for the contract and its one limitation.

#include "spm_tokenizer.h"

#include "debug_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {

// SentencePiece charges an unmatched character `min_score - kUnkPenalty`, which keeps the lattice
// connected without ever preferring an unknown over a real piece.
constexpr float kUnkPenalty = 10.0f;

// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece's escaped space.
const char kSpaceMark[] = "\xE2\x96\x81";

int utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;   // invalid lead byte: consume one so encoding always terminates
}

bool is_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

}  // namespace

bool SpmTokenizer::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { NNOPT_ERROR_FMT("spm: cannot open %s", path.c_str()); return false; }

    char magic[4] = {0};
    uint32_t n = 0;
    int32_t unk = 0, bos = 0, eos = 0;
    float min_score = 0.0f;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "SPM1", 4) == 0 &&
              std::fread(&n, sizeof(n), 1, f) == 1 &&
              std::fread(&unk, sizeof(unk), 1, f) == 1 &&
              std::fread(&bos, sizeof(bos), 1, f) == 1 &&
              std::fread(&eos, sizeof(eos), 1, f) == 1 &&
              std::fread(&min_score, sizeof(min_score), 1, f) == 1;
    if (!ok || n == 0 || n > 1000000u) {
        NNOPT_ERROR_FMT("spm: %s is not a valid SPM1 table", path.c_str());
        std::fclose(f);
        return false;
    }

    pieces_.clear();
    pieces_.reserve(n);
    index_.clear();
    index_.reserve(n * 2);
    for (int i = 0; i < 256; ++i) byte_id_[i] = -1;
    max_piece_bytes_ = 1;

    std::string buf;
    for (uint32_t i = 0; i < n; ++i) {
        uint8_t type = 0;
        float score = 0.0f;
        uint16_t len = 0;
        if (std::fread(&type, sizeof(type), 1, f) != 1 ||
            std::fread(&score, sizeof(score), 1, f) != 1 ||
            std::fread(&len, sizeof(len), 1, f) != 1) {
            NNOPT_ERROR_FMT("spm: truncated table at piece %u/%u", i, n);
            std::fclose(f);
            return false;
        }
        buf.resize(len);
        if (len && std::fread(&buf[0], 1, len, f) != len) {
            NNOPT_ERROR_FMT("spm: truncated piece %u/%u", i, n);
            std::fclose(f);
            return false;
        }
        Piece p;
        p.text = buf;
        p.score = score;
        p.type = type;
        // Only NORMAL and USER_DEFINED pieces take part in the lattice. CONTROL pieces (<s>, </s>)
        // must never be produced from text, and BYTE pieces are reachable only through fallback —
        // indexing either would let a literal "<s>" in a prompt become a control token.
        if (type == 1 || type == 4) {
            index_.emplace(p.text, (int)i);
            max_piece_bytes_ = std::max(max_piece_bytes_, p.text.size());
        } else if (type == 6 && p.text.size() == 6 && p.text.compare(0, 3, "<0x") == 0) {
            const int v = (int)std::strtol(p.text.substr(3, 2).c_str(), nullptr, 16);
            if (v >= 0 && v < 256) byte_id_[v] = (int)i;
        }
        pieces_.push_back(std::move(p));
    }
    std::fclose(f);

    unk_id_ = unk;
    bos_id_ = bos;
    eos_id_ = eos;
    min_score_ = min_score;
    return true;
}

bool SpmTokenizer::encode(const std::string& text, int max_ids, std::vector<int32_t>& ids_out,
                          bool* had_non_ascii) const {
    ids_out.clear();
    if (pieces_.empty()) { NNOPT_ERROR("spm: encode before load"); return false; }
    if (max_ids < 1) { NNOPT_ERROR("spm: max_ids must be >= 1"); return false; }

    // ── normalize: lowercase, collapse whitespace, strip, dummy prefix, escape spaces ──
    bool non_ascii = false;
    std::string norm;
    norm.reserve(text.size() + 4);
    norm += kSpaceMark;                       // add_dummy_prefix
    bool pending_space = false, any = false;
    for (unsigned char c : text) {
        if (c >= 0x80) non_ascii = true;
        if (is_space(c)) { pending_space = any; continue; }
        if (pending_space) { norm += kSpaceMark; pending_space = false; }
        norm += (char)((c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c);
        any = true;
    }
    if (had_non_ascii) *had_non_ascii = non_ascii;
    if (!any) { NNOPT_ERROR("spm: empty prompt"); return false; }

    // ── Viterbi over the piece lattice ──────────────────────────────────────
    const size_t n = norm.size();
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    std::vector<float> best(n + 1, kNegInf);
    std::vector<int> prev(n + 1, -1), piece_at(n + 1, -1);
    best[0] = 0.0f;

    const float unk_score = min_score_ - kUnkPenalty;
    for (size_t i = 0; i < n; ++i) {
        if (best[i] == kNegInf) continue;
        const size_t limit = std::min(max_piece_bytes_, n - i);
        for (size_t l = 1; l <= limit; ++l) {
            auto it = index_.find(norm.substr(i, l));
            if (it == index_.end()) continue;
            const float s = best[i] + pieces_[it->second].score;
            if (s > best[i + l]) { best[i + l] = s; prev[i + l] = (int)i; piece_at[i + l] = it->second; }
        }
        // Unknown character: one whole UTF-8 character, resolved to bytes after backtracking.
        const size_t cl = std::min((size_t)utf8_len((unsigned char)norm[i]), n - i);
        const float s = best[i] + unk_score;
        if (s > best[i + cl]) { best[i + cl] = s; prev[i + cl] = (int)i; piece_at[i + cl] = -1; }
    }
    if (best[n] == kNegInf) { NNOPT_ERROR("spm: no path through the lattice"); return false; }

    std::vector<std::pair<int, int>> spans;   // (start, piece id or -1 for unknown)
    for (size_t i = n; i > 0;) {
        const int p = prev[i];
        if (p < 0) { NNOPT_ERROR("spm: broken backtrack"); return false; }
        spans.emplace_back(p, piece_at[i]);
        i = (size_t)p;
    }
    std::reverse(spans.begin(), spans.end());

    ids_out.push_back(bos_id_);
    for (size_t k = 0; k < spans.size(); ++k) {
        const int start = spans[k].first;
        const int id = spans[k].second;
        if (id >= 0) {
            ids_out.push_back(id);
            continue;
        }
        // Byte fallback: emit the character's UTF-8 bytes as <0xNN> pieces.
        const size_t end = (k + 1 < spans.size()) ? (size_t)spans[k + 1].first : n;
        for (size_t b = (size_t)start; b < end; ++b) {
            const int bid = byte_id_[(unsigned char)norm[b]];
            ids_out.push_back(bid >= 0 ? bid : unk_id_);
        }
    }
    if ((int)ids_out.size() > max_ids) ids_out.resize(max_ids);
    return true;
}
