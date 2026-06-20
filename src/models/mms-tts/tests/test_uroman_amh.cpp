// Minimal sandbox test: verify uroman romanizes Amharic Ge'ez script correctly.
// Compile on macOS (no OpenCL / Android needed):
//   clang++ -std=c++17 -I../src tests/test_uroman_amh.cpp ../src/uroman.cpp -o test_uroman_amh
//   ./test_uroman_amh <path-to-uroman-tables-dir>
//
// Expected: Amharic text → Latin romanization, not empty.

#include "uroman.h"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    const char* table_dir = (argc > 1) ? argv[1] : "assets/uroman";
    std::string table_path = std::string(table_dir) + "/romanization-table.txt";
    std::string delete_path = std::string(table_dir) + "/chars-to-delete.txt";

    fprintf(stderr, "Loading uroman tables from: %s\n", table_dir);
    bool ok = adreno_llm::uroman::load_tables(table_path, delete_path);
    if (!ok) {
        fprintf(stderr, "FAIL: load_tables returned false\n");
        return 1;
    }
    fprintf(stderr, "tables_ready = %s\n", adreno_llm::uroman::tables_ready() ? "true" : "false");

    struct TestCase {
        const char* label;
        const char* input;
        const char* expected_substr;
    };

    TestCase cases[] = {
        {"amh_greeting", "ሰላም", "salaam"},
        {"amh_sentence", "ሰላም፣ ስሜ አላዛር ነው።", "salaam"},
        {"amh_ha", "ሀ", "ha"},
        {"amh_syllables", "ለመሆን", "lamahone"},
        {"eng_passthrough", "hello world", "hello world"},
    };

    int failures = 0;
    for (auto& tc : cases) {
        std::string result = adreno_llm::uroman::romanize(tc.input);
        bool pass = !result.empty() && result.find(tc.expected_substr) != std::string::npos;
        fprintf(stderr, "[%s] %s\n  input:    %s\n  output:   \"%s\"\n  expect:   contains \"%s\"\n  result:   %s\n\n",
                pass ? "PASS" : "FAIL", tc.label,
                tc.input, result.c_str(), tc.expected_substr,
                pass ? "OK" : "FAILED");
        if (!pass) failures++;
    }

    fprintf(stderr, "=== %d/%d passed ===\n", (int)(sizeof(cases)/sizeof(cases[0])) - failures,
            (int)(sizeof(cases)/sizeof(cases[0])));
    return failures > 0 ? 1 : 0;
}
