#pragma once
// SentencePiece unigram tokenizer for the T5 conditioner — CPU, header-only.
//
// Vocab file (weights/t5_tokenizer.bin) is exported by
// reference/export_t5_assets.py from the HF fast tokenizer's JSON backend:
//   int32 n_pieces, int32 pad_id, int32 eos_id, int32 unk_id
//   then per piece: uint16 byte_len, float32 log_prob_score, bytes[byte_len]
//
// Encoding pipeline mirrors the sentencepiece T5 config:
//   1. normalize: collapse runs of whitespace to one space, trim ends
//      (remove_extra_whitespaces). NFKC is skipped — identity for ASCII.
//   2. metaspace: prepend "▁" (add_dummy_prefix), replace ' ' -> "▁".
//   3. unigram viterbi over the byte string: maximize sum of piece scores.
//      Unknown bytes fall back to unk_id with the standard sentencepiece
//      penalty (min_score - 10) so viterbi always completes.
//   4. truncate to max_len-1, append EOS, pad with PAD to max_len.

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

class T5Tokenizer {
public:
    int pad_id = 0, eos_id = 1, unk_id = 2;

    bool load(const std::string& path) {
        FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) { std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str()); return false; }
        int32_t hdr[4];
        if (std::fread(hdr, 4, 4, fp) != 4) { std::fclose(fp); return false; }
        int n = hdr[0]; pad_id = hdr[1]; eos_id = hdr[2]; unk_id = hdr[3];
        pieces_.resize(n); scores_.resize(n);
        min_score_ = 0.0f;
        for (int i = 0; i < n; i++) {
            uint16_t len; float score;
            if (std::fread(&len, 2, 1, fp) != 1 || std::fread(&score, 4, 1, fp) != 1) {
                std::fclose(fp); return false;
            }
            pieces_[i].resize(len);
            if (len && std::fread(&pieces_[i][0], 1, len, fp) != len) {
                std::fclose(fp); return false;
            }
            scores_[i] = score;
            if (score < min_score_) min_score_ = score;
            if (len > max_piece_len_) max_piece_len_ = len;
            // first occurrence wins (ids 0..2 are control pieces with score 0)
            lookup_.emplace(pieces_[i], i);
        }
        std::fclose(fp);
        return n > 0;
    }

    // Returns exactly max_len ids (eos-terminated, pad-filled).
    // n_real_out = count of non-pad ids (includes the EOS).
    std::vector<int32_t> encode(const std::string& text, int max_len, int* n_real_out) const {
        // 1. whitespace normalize
        std::string norm;
        bool in_ws = true;   // leading ws dropped
        for (char c : text) {
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                if (!in_ws) norm += ' ';
                in_ws = true;
            } else { norm += c; in_ws = false; }
        }
        while (!norm.empty() && norm.back() == ' ') norm.pop_back();
        // 2. metaspace (U+2581 = 0xE2 0x96 0x81)
        static const char* META = "\xE2\x96\x81";
        std::string s = META;
        for (char c : norm) { if (c == ' ') s += META; else s += c; }

        // 3. viterbi
        const int N = (int)s.size();
        const float NEG = -1e30f;
        std::vector<float> best(N + 1, NEG);
        std::vector<int> from(N + 1, -1), piece(N + 1, -1);
        best[0] = 0.0f;
        const float unk_score = min_score_ - 10.0f;
        for (int i = 0; i < N; i++) {
            if (best[i] <= NEG) continue;
            int maxl = max_piece_len_ < N - i ? max_piece_len_ : N - i;
            for (int l = 1; l <= maxl; l++) {
                auto it = lookup_.find(s.substr(i, l));
                if (it == lookup_.end()) continue;
                float sc = best[i] + scores_[it->second];
                if (sc > best[i + l]) { best[i + l] = sc; from[i + l] = i; piece[i + l] = it->second; }
            }
            // unk fallback: consume one whole UTF-8 codepoint
            int cl = 1;
            unsigned char b = (unsigned char)s[i];
            if      ((b & 0xF8) == 0xF0) cl = 4;
            else if ((b & 0xF0) == 0xE0) cl = 3;
            else if ((b & 0xE0) == 0xC0) cl = 2;
            if (i + cl <= N && best[i] + unk_score > best[i + cl]) {
                best[i + cl] = best[i] + unk_score; from[i + cl] = i; piece[i + cl] = unk_id;
            }
        }
        std::vector<int32_t> ids;
        for (int pos = N; pos > 0; pos = from[pos]) {
            if (from[pos] < 0) { ids.clear(); break; }   // shouldn't happen
            ids.push_back(piece[pos]);
        }
        std::vector<int32_t> out;
        for (auto it = ids.rbegin(); it != ids.rend(); ++it) out.push_back(*it);

        // 4. truncate + eos + pad
        if ((int)out.size() > max_len - 1) out.resize(max_len - 1);
        out.push_back(eos_id);
        if (n_real_out) *n_real_out = (int)out.size();
        while ((int)out.size() < max_len) out.push_back(pad_id);
        return out;
    }

private:
    std::vector<std::string> pieces_;
    std::vector<float> scores_;
    std::unordered_map<std::string, int> lookup_;
    float min_score_ = 0.0f;
    int max_piece_len_ = 1;
};
