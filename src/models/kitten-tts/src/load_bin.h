#pragma once
// Generic typed binary loader for fixture files the reference Python publishes
// alongside the model (e.g. test_input_ids.bin, duration_noise.bin,
// prior_noise.bin). The runtime loads these instead of sampling its own RNG
// so cosine compare against the reference is meaningful.
//
// Each file is a flat little-endian array of T (no header). The shape is
// known to the caller (from reference_tokens.json or the model architecture).
// If you need a header, write_bin in the reference template.

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace nnopt {

template <typename T>
inline bool read_bin(const std::string& path, std::vector<T>& out) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    std::fseek(fp, 0, SEEK_END);
    long bytes = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (bytes < 0 || (bytes % (long)sizeof(T)) != 0) { std::fclose(fp); return false; }
    out.resize((size_t)bytes / sizeof(T));
    size_t got = std::fread(out.data(), sizeof(T), out.size(), fp);
    std::fclose(fp);
    return got == out.size();
}

inline std::vector<int32_t> load_int32_bin(const std::string& path) {
    std::vector<int32_t> v;
    if (!read_bin<int32_t>(path, v)) {
        std::fprintf(stderr, "ERROR: failed to load int32 bin %s\n", path.c_str());
    }
    return v;
}

inline std::vector<float> load_float_bin(const std::string& path) {
    std::vector<float> v;
    if (!read_bin<float>(path, v)) {
        std::fprintf(stderr, "ERROR: failed to load float bin %s\n", path.c_str());
    }
    return v;
}

}  // namespace nnopt

using nnopt::load_int32_bin;
using nnopt::load_float_bin;
