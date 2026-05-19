// uroman.cpp — see uroman.h for design notes.
//
// Implementation strategy:
//   • Tables are parsed into two structures:
//       single_cp_   unordered_map<uint32_t cp, std::string ascii>
//                    Fast O(1) lookup for the common case (one codepoint
//                    maps to one ASCII string).
//       multi_cp_    sorted vector<MultiEntry{key,value}> where key is
//                    a UTF-8 byte sequence of length >1 codepoint. We
//                    sort by key descending and scan greedily for longest
//                    match starting at the current byte position. ~hundreds
//                    of multi-cp entries in the full table, so a linear
//                    scan is cheap (≤500 ns/char on Snapdragon 765).
//   • Deletion list is a hash set of uint32_t codepoints.
//   • Lowercasing: ASCII only after romanization, so a simple `tolower`
//     suffices. Non-ASCII codepoints not in the table are dropped (matches
//     uroman's "unromanizable → skip" fallback).
//
// Input is treated as UTF-8 byte sequence; codepoints decoded inline.

#include "uroman.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace adreno_llm::uroman {

namespace {

struct MultiEntry {
    std::string key_utf8;     // 2+ codepoints encoded as UTF-8
    std::string value;        // ASCII romanization
};

std::unordered_map<uint32_t, std::string> single_cp_;
std::vector<MultiEntry> multi_cp_;
std::unordered_set<uint32_t> deletable_;
bool loaded_ = false;
std::mutex load_mu_;

// ── UTF-8 decode: returns codepoint and bytes consumed (1–4). Returns
//    {0xFFFD, 1} on malformed sequence — caller should drop and advance.
struct DecodedCp { uint32_t cp; int bytes; };

DecodedCp decode_utf8(const char* p, const char* end) {
    if (p >= end) return {0, 0};
    uint8_t b0 = static_cast<uint8_t>(*p);
    if (b0 < 0x80) return {b0, 1};
    if ((b0 & 0xE0) == 0xC0 && p + 1 < end) {
        uint8_t b1 = static_cast<uint8_t>(p[1]);
        if ((b1 & 0xC0) == 0x80) {
            uint32_t cp = ((b0 & 0x1Fu) << 6) | (b1 & 0x3Fu);
            return {cp, 2};
        }
    }
    if ((b0 & 0xF0) == 0xE0 && p + 2 < end) {
        uint8_t b1 = static_cast<uint8_t>(p[1]);
        uint8_t b2 = static_cast<uint8_t>(p[2]);
        if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80) {
            uint32_t cp = ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
            return {cp, 3};
        }
    }
    if ((b0 & 0xF8) == 0xF0 && p + 3 < end) {
        uint8_t b1 = static_cast<uint8_t>(p[1]);
        uint8_t b2 = static_cast<uint8_t>(p[2]);
        uint8_t b3 = static_cast<uint8_t>(p[3]);
        if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80 && (b3 & 0xC0) == 0x80) {
            uint32_t cp = ((b0 & 0x07u) << 18) | ((b1 & 0x3Fu) << 12) |
                          ((b2 & 0x3Fu) << 6) | (b3 & 0x3Fu);
            return {cp, 4};
        }
    }
    return {0xFFFD, 1};
}

// Trim leading/trailing ASCII whitespace.
std::string_view trim(std::string_view s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) ++a;
    while (b > a && (s[b-1] == ' ' || s[b-1] == '\t' || s[b-1] == '\r' || s[b-1] == '\n')) --b;
    return s.substr(a, b - a);
}

// Parse a "U+XXXX" or "0xXXXX" or raw hex token into a codepoint.
bool parse_cp(std::string_view tok, uint32_t& out) {
    if (tok.size() > 2 && tok[0] == 'U' && tok[1] == '+') tok.remove_prefix(2);
    else if (tok.size() > 2 && tok[0] == '0' && (tok[1] == 'x' || tok[1] == 'X')) tok.remove_prefix(2);
    if (tok.empty() || tok.size() > 8) return false;
    uint32_t v = 0;
    for (char c : tok) {
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (c - '0');
        else if (c >= 'A' && c <= 'F') v |= (c - 'A' + 10);
        else if (c >= 'a' && c <= 'f') v |= (c - 'a' + 10);
        else return false;
    }
    out = v;
    return true;
}

void encode_utf8(uint32_t cp, std::string& out) {
    if (cp < 0x80) out.push_back(static_cast<char>(cp));
    else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// Parse one line of uroman's romanization-table.txt (real isi-nlp format).
// Lines look like:
//   ::s Ä ::t Ae
//   ::s Ä ::t A ::lcode deu
//   ::s ሀ ::t ha
// We extract only ::s (source UTF-8 or U+XXXX) and ::t (target ASCII).
// Language-conditional entries are accepted universally.
bool parse_table_line(std::string_view line, std::string& key_utf8, std::string& value) {
    line = trim(line);
    if (line.empty() || line[0] == '#') return false;

    std::string_view src;
    std::string_view tgt;
    bool have_src = false, have_tgt = false;

    size_t i = 0;
    while (i < line.size()) {
        size_t m = line.find("::", i);
        if (m == std::string_view::npos) break;
        size_t name_start = m + 2;
        size_t name_end = name_start;
        while (name_end < line.size() && line[name_end] != ' ' && line[name_end] != '\t') ++name_end;
        std::string_view name = line.substr(name_start, name_end - name_start);

        size_t val_start = name_end;
        while (val_start < line.size() && (line[val_start] == ' ' || line[val_start] == '\t')) ++val_start;
        size_t val_end = line.find("::", val_start);
        if (val_end == std::string_view::npos) val_end = line.size();
        std::string_view val = trim(line.substr(val_start, val_end - val_start));

        if      (name == "s" && !have_src) { src = val; have_src = true; }
        else if (name == "t" && !have_tgt) { tgt = val; have_tgt = true; }
        i = val_end;
    }
    if (!have_src || !have_tgt || src.empty()) return false;

    key_utf8.clear();
    if (src.size() > 1 && (src[0] == 'U' || src[0] == 'u') && src[1] == '+') {
        size_t p = 0;
        while (p < src.size()) {
            while (p < src.size() && src[p] == ' ') ++p;
            if (p >= src.size()) break;
            size_t q = p;
            while (q < src.size() && src[q] != ' ') ++q;
            uint32_t cp;
            if (parse_cp(src.substr(p, q - p), cp)) encode_utf8(cp, key_utf8);
            p = q;
        }
        if (key_utf8.empty()) key_utf8.assign(src.data(), src.size());
    } else {
        key_utf8.assign(src.data(), src.size());
    }
    value.assign(tgt.data(), tgt.size());
    return !key_utf8.empty();
}

bool parse_delete_line(std::string_view line, uint32_t& cp_out) {
    line = trim(line);
    if (line.empty() || line[0] == '#') return false;
    auto t = line.find('\t');
    std::string_view first = trim(t == std::string_view::npos ? line : line.substr(0, t));
    return parse_cp(first, cp_out);
}

void ascii_lower(std::string& s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
    }
}

}  // namespace

namespace {
// Ingest one table file into single_cp_ / multi_cp_. Caller holds load_mu_.
size_t ingest_table_file(std::string_view path) {
    std::ifstream f{std::string(path)};
    if (!f) return 0;
    size_t added = 0;
    std::string line, key_utf8, value;
    while (std::getline(f, line)) {
        if (!parse_table_line(line, key_utf8, value)) continue;
        int cp_count = 0;
        for (size_t i = 0; i < key_utf8.size(); ) {
            auto [cp, n] = decode_utf8(key_utf8.data() + i, key_utf8.data() + key_utf8.size());
            if (n <= 0) break;
            ++cp_count;
            i += static_cast<size_t>(n);
        }
        if (cp_count == 1) {
            auto [cp, _] = decode_utf8(key_utf8.data(), key_utf8.data() + key_utf8.size());
            // First-definition-wins: manual table loads first so its entries
            // override auto-table where they overlap.
            single_cp_.emplace(cp, value);
            ++added;
        } else if (cp_count > 1) {
            multi_cp_.push_back({key_utf8, value});
            ++added;
        }
    }
    return added;
}
}  // namespace

bool load_tables(std::string_view table_path, std::string_view delete_path) {
    std::lock_guard<std::mutex> lk(load_mu_);
    if (loaded_) return true;

    single_cp_.clear();
    multi_cp_.clear();
    deletable_.clear();

    // Primary table (manual: Latin extensions, European, multi-codepoint rules).
    size_t added = ingest_table_file(table_path);
    if (added == 0) return false;

    // Auto-generated table lives next to the manual one. uroman's
    // Ethiopic / many Indic / various other syllabic-script entries are
    // ONLY in the auto table. Load if present (sibling file, expected name).
    {
        std::string p{table_path};
        auto slash = p.find_last_of('/');
        std::string dir = (slash == std::string::npos) ? "." : p.substr(0, slash);
        ingest_table_file(dir + "/romanization-auto-table.txt");
    }

    // Sort multi-cp entries by descending key byte length so greedy scan
    // matches the longest prefix first.
    std::sort(multi_cp_.begin(), multi_cp_.end(),
              [](const MultiEntry& a, const MultiEntry& b) {
                  return a.key_utf8.size() > b.key_utf8.size();
              });

    if (!delete_path.empty()) {
        std::ifstream df{std::string(delete_path)};
        if (df) {
            std::string dline;
            while (std::getline(df, dline)) {
                uint32_t cp;
                if (parse_delete_line(dline, cp)) deletable_.insert(cp);
            }
        }
    }

    loaded_ = !single_cp_.empty() || !multi_cp_.empty();
    return loaded_;
}

bool tables_ready() {
    std::lock_guard<std::mutex> lk(load_mu_);
    return loaded_;
}

std::string romanize(std::string_view utf8_in, const char* /*lang_code*/) {
    std::string out;
    out.reserve(utf8_in.size());

    const char* p = utf8_in.data();
    const char* end = p + utf8_in.size();

    while (p < end) {
        // Try multi-codepoint match first (longest-prefix wins).
        bool matched = false;
        if (loaded_) {
            for (const auto& e : multi_cp_) {
                size_t kl = e.key_utf8.size();
                if (static_cast<size_t>(end - p) >= kl &&
                    std::memcmp(p, e.key_utf8.data(), kl) == 0) {
                    out.append(e.value);
                    p += kl;
                    matched = true;
                    break;
                }
            }
        }
        if (matched) continue;

        auto [cp, n] = decode_utf8(p, end);
        if (n <= 0) { ++p; continue; }
        p += n;

        if (deletable_.count(cp)) continue;

        if (cp < 0x80) {
            // ASCII passes through (lowercased at the end).
            out.push_back(static_cast<char>(cp));
            continue;
        }

        auto it = single_cp_.find(cp);
        if (it != single_cp_.end()) {
            out.append(it->second);
        }
        // else: silently drop (uroman behaviour for unromanizable codepoints)
    }

    ascii_lower(out);
    return out;
}

}  // namespace adreno_llm::uroman
