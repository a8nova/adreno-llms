// uroman.h — minimal table-driven Unicode→ASCII romanization for on-device
// use in adreno-llm's MMS-TTS port. Implements the subset of isi-nlp/uroman
// that Facebook's MMS pipeline relies on:
//   1. UTF-8 codepoint stream (assumed NFC; we don't ship a full NFC table)
//   2. Greedy longest-match against a romanization table
//   3. Drop codepoints listed as "deletable"
//   4. Lowercase the ASCII output
//
// Data tables ship as plain-text assets (romanization-table.txt +
// chars-to-delete.txt) sourced from isi-nlp/uroman/data/ (MIT license).
// Load them once at app start with uroman_load_tables(); subsequent calls
// to uroman_romanize() are pure-CPU and allocation-light.
//
// Thread safety: tables are immutable after load; uroman_romanize() is
// re-entrant and thread-safe.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace adreno_llm::uroman {

// Initialize romanization tables from disk. Idempotent — safe to call
// multiple times; subsequent calls with the same paths are no-ops.
//
//   table_path: path to romanization-table.txt
//                 (e.g. <assets>/uroman/romanization-table.txt)
//   delete_path: path to chars-to-delete.txt (optional; empty string skips)
//
// Returns true on success. On failure, romanize() falls back to a passthrough
// that lowercases ASCII and drops non-ASCII codepoints — keeping the pipeline
// functional but degraded.
bool load_tables(std::string_view table_path,
                 std::string_view delete_path = "");

// Romanize a UTF-8 string. `lang_code` is currently advisory (uroman's
// per-language rules collapse for MMS-TTS's use case to the table-driven
// path); the parameter is accepted for forward-compat and for future
// language-specific overrides (e.g. Chinese→Pinyin variants).
//
// Returned string is lowercase ASCII suitable for char-level lookup against
// tokenizer_vocab.bin.
std::string romanize(std::string_view utf8_in, const char* lang_code = nullptr);

// Returns true if uroman tables loaded successfully and at least one
// table entry exists. Useful for early diagnostics at app start.
bool tables_ready();

}  // namespace adreno_llm::uroman
