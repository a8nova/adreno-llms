#pragma once
// Minimal WAV writer (RIFF, 16-bit signed PCM, mono, little-endian).
// Used by TTS / MusicGen ports to dump audio output.
// Pure stdlib; no libsndfile / no ALSA / no portaudio dependency.
//
// Two overloads:
//   write_wav(path, int16*, n, sr)   — caller already has int16 PCM
//   write_wav(path, float*, n, sr)   — caller has float32 in [-1, 1]; converted
//
// Both write the same file format (16-bit signed little-endian).

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <algorithm>

namespace nnopt {

namespace detail_wav {
inline void write_u16(FILE* fp, uint16_t v) { std::fwrite(&v, 2, 1, fp); }
inline void write_u32(FILE* fp, uint32_t v) { std::fwrite(&v, 4, 1, fp); }
}

inline bool write_wav(const std::string& path,
                      const int16_t* samples,
                      int num_samples,
                      int sample_rate) {
    if (num_samples <= 0 || sample_rate <= 0 || !samples) return false;
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) { std::fprintf(stderr, "ERROR: cannot open %s for writing\n", path.c_str()); return false; }

    const uint16_t num_channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint16_t block_align = num_channels * bits_per_sample / 8;
    const uint32_t byte_rate = (uint32_t)sample_rate * block_align;
    const uint32_t data_bytes = (uint32_t)num_samples * (uint32_t)block_align;
    const uint32_t riff_size = 36 + data_bytes;

    std::fwrite("RIFF", 1, 4, fp);
    detail_wav::write_u32(fp, riff_size);
    std::fwrite("WAVE", 1, 4, fp);

    std::fwrite("fmt ", 1, 4, fp);
    detail_wav::write_u32(fp, 16);                    // fmt chunk size
    detail_wav::write_u16(fp, 1);                     // PCM format code
    detail_wav::write_u16(fp, num_channels);
    detail_wav::write_u32(fp, (uint32_t)sample_rate);
    detail_wav::write_u32(fp, byte_rate);
    detail_wav::write_u16(fp, block_align);
    detail_wav::write_u16(fp, bits_per_sample);

    std::fwrite("data", 1, 4, fp);
    detail_wav::write_u32(fp, data_bytes);
    size_t wrote = std::fwrite(samples, sizeof(int16_t), (size_t)num_samples, fp);
    std::fclose(fp);
    return wrote == (size_t)num_samples;
}

inline bool write_wav(const std::string& path,
                      const float* samples_f32,
                      int num_samples,
                      int sample_rate) {
    if (num_samples <= 0 || sample_rate <= 0 || !samples_f32) return false;
    // Convert [-1, 1] float → int16 with clipping.
    std::vector<int16_t> pcm(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        float s = samples_f32[i];
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
        pcm[i] = (int16_t)(s * 32767.0f);
    }
    return write_wav(path, pcm.data(), num_samples, sample_rate);
}

}  // namespace nnopt

using nnopt::write_wav;
