// Dequantisation for the vision tower's tensor kinds.
//
// The text tower is Q1 end to end and needs none of this. The Qwen3-VL mmproj is a mix — verified
// census over the shipped file: F32 224, Q8_0 83, F16 27 — so the loader has to understand three
// formats. We dequantise ONCE at upload rather than in the inner loop: the tower is ~0.63 GB
// quantised, runs once per image, and CLBlast wants a dense buffer anyway.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

/**
 * GGUF Q8_0: blocks of 32 values, each stored as [fp16 scale][32 x int8] = 34 bytes.
 * value = scale * q. Layout is exactly as llama.cpp writes it, which is why the converter can copy
 * the bytes through untouched and leave the decode to here.
 */
static constexpr int kQ8Block = 32;
static constexpr int kQ8Bytes = 34;

/** IEEE half -> float. Standalone so this header has no dependency on the Q1 path's q1.h. */
inline float vis_half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1F;
    const uint32_t man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;                              // +/- 0
        } else {                                      // subnormal — renormalise
            int e = -1;
            uint32_t m = man;
            do { m <<= 1; ++e; } while (!(m & 0x400));
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FF) << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (man << 13);      // inf / nan
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

/**
 * Decode `numel` values of `kind` from `src` into fp32.
 *
 * Returns false for an unknown kind rather than guessing — a silently mis-decoded weight produces a
 * model that still emits plausible text, which is the failure mode this port has been bitten by
 * twice already (the double-exp on ssm_a, the tile-vs-interleave head mapping).
 */
inline bool vis_dequant(const std::string& kind, const uint8_t* src, size_t numel,
                        std::vector<float>* out) {
    out->resize(numel);
    if (kind == "f32") {
        std::memcpy(out->data(), src, numel * 4);
        return true;
    }
    if (kind == "f16") {
        const uint16_t* h = (const uint16_t*)src;
        for (size_t i = 0; i < numel; ++i) (*out)[i] = vis_half_to_float(h[i]);
        return true;
    }
    if (kind == "q8_0") {
        if (numel % kQ8Block) return false;
        const size_t nblocks = numel / kQ8Block;
        for (size_t b = 0; b < nblocks; ++b) {
            const uint8_t* blk = src + b * kQ8Bytes;
            uint16_t sh;
            std::memcpy(&sh, blk, 2);
            const float scale = vis_half_to_float(sh);
            const int8_t* q = (const int8_t*)(blk + 2);
            float* o = out->data() + b * kQ8Block;
            for (int i = 0; i < kQ8Block; ++i) o[i] = scale * (float)q[i];
        }
        return true;
    }
    return false;
}
