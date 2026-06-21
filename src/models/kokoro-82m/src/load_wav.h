#pragma once
// Minimal WAV (RIFF) loader for the host side of TTS / ASR ports.
// Supports the only format we care about for v1:
//   - canonical PCM (format code 1)
//   - 16-bit signed little-endian samples
//   - mono OR stereo (stereo is downmixed to mono by averaging)
//
// The 'sample_rate' the caller wants is checked against the file's reported
// rate; on mismatch we WARN to stderr but still return the samples so the
// agent can detect the bug. Resampling lives in a separate kernel/host
// helper (see resample.h when ASR lands).

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nnopt {

struct WavInfo {
    int sample_rate = 0;
    int num_channels = 0;
    int bits_per_sample = 0;
    int num_frames = 0;
};

inline bool read_wav_mono_f32(const std::string& path,
                              std::vector<float>& out_samples,
                              WavInfo& info_out,
                              int expected_sample_rate = 0) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) { std::fprintf(stderr, "ERROR: cannot open WAV %s\n", path.c_str()); return false; }

    char riff[4]; std::fread(riff, 1, 4, fp);
    if (std::memcmp(riff, "RIFF", 4) != 0) {
        std::fprintf(stderr, "ERROR: not RIFF: %s\n", path.c_str()); std::fclose(fp); return false;
    }
    uint32_t riff_size = 0; std::fread(&riff_size, 4, 1, fp);
    char wave[4]; std::fread(wave, 1, 4, fp);
    if (std::memcmp(wave, "WAVE", 4) != 0) {
        std::fprintf(stderr, "ERROR: not WAVE: %s\n", path.c_str()); std::fclose(fp); return false;
    }

    // Walk chunks; we want 'fmt ' then 'data'.
    uint16_t fmt_code = 0, num_chan = 0, bits = 0, block_align = 0;
    uint32_t sample_rate = 0, byte_rate = 0;
    bool got_fmt = false;
    std::vector<int16_t> pcm16;

    while (!std::feof(fp)) {
        char id[4]; if (std::fread(id, 1, 4, fp) != 4) break;
        uint32_t sz = 0; if (std::fread(&sz, 4, 1, fp) != 1) break;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            std::fread(&fmt_code, 2, 1, fp);
            std::fread(&num_chan, 2, 1, fp);
            std::fread(&sample_rate, 4, 1, fp);
            std::fread(&byte_rate, 4, 1, fp);
            std::fread(&block_align, 2, 1, fp);
            std::fread(&bits, 2, 1, fp);
            if (sz > 16) std::fseek(fp, sz - 16, SEEK_CUR);  // skip any extension bytes
            got_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0) {
            if (!got_fmt) { std::fprintf(stderr, "ERROR: data before fmt in %s\n", path.c_str()); std::fclose(fp); return false; }
            if (fmt_code != 1 || bits != 16) {
                std::fprintf(stderr, "ERROR: WAV %s is not 16-bit PCM (fmt=%u bits=%u)\n",
                             path.c_str(), (unsigned)fmt_code, (unsigned)bits);
                std::fclose(fp); return false;
            }
            pcm16.resize(sz / 2);
            if (std::fread(pcm16.data(), 2, pcm16.size(), fp) != pcm16.size()) {
                std::fprintf(stderr, "ERROR: short read in %s\n", path.c_str());
                std::fclose(fp); return false;
            }
            break;  // we have what we need
        } else {
            std::fseek(fp, sz, SEEK_CUR);
        }
    }
    std::fclose(fp);
    if (!got_fmt || pcm16.empty()) return false;

    info_out.sample_rate = (int)sample_rate;
    info_out.num_channels = (int)num_chan;
    info_out.bits_per_sample = (int)bits;
    info_out.num_frames = (int)(pcm16.size() / num_chan);

    if (expected_sample_rate > 0 && (int)sample_rate != expected_sample_rate) {
        std::fprintf(stderr,
                     "WARN: WAV %s sample_rate=%u, expected %d — caller must resample.\n",
                     path.c_str(), (unsigned)sample_rate, expected_sample_rate);
    }

    // Downmix to mono float32 in [-1, 1].
    out_samples.resize(info_out.num_frames);
    const float scale = 1.0f / 32768.0f;
    if (num_chan == 1) {
        for (int i = 0; i < info_out.num_frames; ++i) out_samples[i] = pcm16[i] * scale;
    } else {
        // average channels
        for (int i = 0; i < info_out.num_frames; ++i) {
            int sum = 0;
            for (int c = 0; c < num_chan; ++c) sum += pcm16[i * num_chan + c];
            out_samples[i] = ((float)sum / num_chan) * scale;
        }
    }
    return true;
}

}  // namespace nnopt
