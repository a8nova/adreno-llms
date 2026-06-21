#pragma once
// Minimal WAV container I/O (byte parsing only — no DSP). PCM16 in, PCM16 out.
// The numeric work (downmix, resample, float<->int16) runs on the GPU.
#include <string>
#include <vector>
#include <cstdint>

struct WavData {
    std::vector<int16_t> samples;  // interleaved PCM16
    int channels = 1;
    int sample_rate = 16000;
};

// Reads a PCM16 (format 1) WAV. Returns false on parse failure / unsupported format.
bool read_wav(const std::string& path, WavData& out);

// Writes a mono 16 kHz PCM16 WAV.
bool write_wav(const std::string& path, const std::vector<int16_t>& mono, int sample_rate = 16000);
