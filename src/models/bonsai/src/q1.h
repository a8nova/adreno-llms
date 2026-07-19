// Q1 (1-bit) math. Layout per model/Q1_0_LAYOUT.md, byte-exact verified:
// 18-byte unit = [fp16 scale][16B sign bits, LSB-first], 128 weights,
// bit=1 -> +scale. Units run along the input dim (ne0), rows concatenated.
//
// GEMV identity (removes the per-element multiply):
//   dot(unit, x) = s * (2 * sum_{i: bit=1} x[i] - sum_i x[i])
// Host reference keeps it simple; fp32 accumulation everywhere (invariant #4).
#pragma once
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

inline float half_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = s << 31;
        else {  // subnormal
            e = 127 - 15 + 1;
            while (!(m & 0x400)) { m <<= 1; --e; }
            m &= 0x3FF;
            bits = (s << 31) | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        bits = (s << 31) | 0x7F800000 | (m << 13);
    } else {
        bits = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t float_to_half(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000;
    int32_t e = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFF;
    if (e <= 0) return (uint16_t)s;                       // flush to zero
    if (e >= 31) return (uint16_t)(s | 0x7C00);           // inf/overflow
    // round-to-nearest-even on the 13 dropped bits
    uint32_t h = (uint32_t)(e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;
    return (uint16_t)(s | h);
}

// Decode one full row of a Q1 tensor into fp32 (used for token_embd gather
// and debugging; row = output index, n_in = ne0).
inline void q1_row_decode(const uint8_t* q1, int64_t row, int n_in, float* out) {
    const int units = n_in / 128;
    const uint8_t* p = q1 + (size_t)row * units * 18;
    for (int u = 0; u < units; ++u, p += 18) {
        uint16_t sh;
        memcpy(&sh, p, 2);
        const float s = half_to_float(sh);
        const uint8_t* bits = p + 2;
        float* o = out + u * 128;
        for (int i = 0; i < 128; ++i)
            o[i] = (bits[i >> 3] >> (i & 7)) & 1 ? s : -s;
    }
}

// y[0..n_out) = W_q1[n_out, n_in] · x[0..n_in)   (fp32 accum, threaded)
inline void q1_gemv(const uint8_t* q1, const float* x, float* y,
                    int n_out, int n_in, int n_threads = 8) {
    const int units = n_in / 128;
    // per-unit-column activation sums, shared across all rows
    std::vector<float> xsum(units);
    std::vector<float> xps(units * 128);  // prefix trick not needed; keep sums
    for (int u = 0; u < units; ++u) {
        float acc = 0.f;
        const float* xu = x + u * 128;
        for (int i = 0; i < 128; ++i) acc += xu[i];
        xsum[u] = acc;
    }
    auto worker = [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            const uint8_t* p = q1 + (size_t)r * units * 18;
            float acc = 0.f;
            for (int u = 0; u < units; ++u, p += 18) {
                uint16_t sh;
                memcpy(&sh, p, 2);
                const float s = half_to_float(sh);
                const float* xu = x + u * 128;
                float pos = 0.f;
                for (int w = 0; w < 16; ++w) {
                    uint8_t b = p[2 + w];
                    const float* xw = xu + w * 8;
                    // unrolled predicated adds
                    if (b & 1)   pos += xw[0];
                    if (b & 2)   pos += xw[1];
                    if (b & 4)   pos += xw[2];
                    if (b & 8)   pos += xw[3];
                    if (b & 16)  pos += xw[4];
                    if (b & 32)  pos += xw[5];
                    if (b & 64)  pos += xw[6];
                    if (b & 128) pos += xw[7];
                }
                acc += s * (2.f * pos - xsum[u]);
            }
            y[r] = acc;
        }
    };
    if (n_threads <= 1 || n_out < 256) {
        worker(0, n_out);
        return;
    }
    std::vector<std::thread> th;
    int per = (n_out + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; ++t) {
        int r0 = t * per, r1 = std::min(n_out, r0 + per);
        if (r0 >= r1) break;
        th.emplace_back(worker, r0, r1);
    }
    for (auto& t : th) t.join();
}
