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
#ifdef NNOPT_DEBUG
        // Debug-only: missing fixture is harmless in production-text mode, so
        // this message would otherwise spam release-mode stderr. The real
        // hard-fail (no fixtures AND no --token-ids AND tokenizer failed) is
        // reported by an explicit NNOPT_ERROR_FMT in main.cpp.
        std::fprintf(stderr, "CHECKPOINT: load_int32_bin: missing or unreadable %s\n", path.c_str());
#endif
    }
    return v;
}

inline std::vector<float> load_float_bin(const std::string& path) {
    std::vector<float> v;

    // (1) Try load as float32 (the canonical reference dump format).
    if (read_bin<float>(path, v)) return v;

    // (2) Fallback: allow fp16 fixtures (storage_t) and upcast to float32.
    // This is workspace-robust because Deploy may ship fp16 assets alongside fp16 weights.
    std::vector<uint16_t> v_u16;
    if (read_bin<uint16_t>(path, v_u16)) {
        v.resize(v_u16.size());
        for (size_t i = 0; i < v_u16.size(); ++i) {
            const uint16_t h = v_u16[i];
            // Minimal fp16->fp32 conversion (IEEE 754 half). Good enough for fixtures.
            const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
            const uint32_t exp = (uint32_t)(h & 0x7C00u) >> 10;
            const uint32_t mant = (uint32_t)(h & 0x03FFu);
            uint32_t fbits = 0;
            if (exp == 0) {
                if (mant == 0) {
                    fbits = sign;  // zero
                } else {
                    // subnormal
                    uint32_t m = mant;
                    int e = -1;
                    do { e++; m <<= 1; } while ((m & 0x0400u) == 0);
                    m &= 0x03FFu;
                    const uint32_t exp_f = (uint32_t)(127 - 15 - e);
                    fbits = sign | (exp_f << 23) | (m << 13);
                }
            } else if (exp == 0x1Fu) {
                // inf/nan
                fbits = sign | 0x7F800000u | (mant << 13);
            } else {
                const uint32_t exp_f = exp + (127 - 15);
                fbits = sign | (exp_f << 23) | (mant << 13);
            }
            float out_f;
            std::memcpy(&out_f, &fbits, sizeof(float));
            v[i] = out_f;
        }
        return v;
    }

#ifdef NNOPT_DEBUG
    std::fprintf(stderr, "CHECKPOINT: load_float_bin: missing or unreadable %s\n", path.c_str());
#endif
    return v;
}

}  // namespace nnopt

using nnopt::load_int32_bin;
using nnopt::load_float_bin;
