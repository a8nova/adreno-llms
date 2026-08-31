// Host-side I/O for the song path: style conditioning in, stereo WAV out.
// Kept out of backbone.cpp so run_song() is pure compute (no filesystem).

#include "song.h"
#include "debug_utils.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr int kTemporalDim = 1024;   // temporal_input_0
constexpr int kSourceLen   = 256;    // cross-attention conditioning
constexpr int kSampleRate  = 48000;

// Read exactly `n` fp32 values. Returns false (and says why) on short files —
// a truncated conditioning blob would otherwise generate silently-wrong music.
bool read_f32_exact(const std::string& path, int n, std::vector<float>& out, std::string& why) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { why = "cannot open " + path; return false; }
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (bytes != (long)(n * (int)sizeof(float))) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "%s is %ld bytes, expected %d (%d fp32)",
                      path.c_str(), bytes, n * (int)sizeof(float), n);
        why = msg;
        std::fclose(f);
        return false;
    }
    out.resize((size_t)n);
    const size_t rd = std::fread(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);
    if (rd != out.size()) { why = "short read on " + path; return false; }
    return true;
}

bool file_exists(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

// The shipped debug fixtures — a real generation, but NOT a real style prompt.
bool load_fixture(SongConditioning& out, std::string& why) {
    if (!read_f32_exact("weights/optest_input.bin", kTemporalDim, out.temporal_input, why)) return false;
    if (!read_f32_exact("weights/optest_source.bin", kSourceLen, out.source, why)) return false;
    out.name = "fixture";
    out.is_fixture = true;
    return true;
}

}  // namespace

bool song_load_conditioning(const std::string& style, SongConditioning& out, std::string& why) {
    out = SongConditioning();

    if (style.empty()) {
        if (!load_fixture(out, why)) return false;
        why = "no style given — using the shipped fixture conditioning";
        return true;
    }

    // A style bundle is 1280 fp32: [1024] temporal_input ++ [256] source.
    const bool looks_like_path =
        style.find('/') != std::string::npos ||
        (style.size() > 4 && style.compare(style.size() - 4, 4, ".bin") == 0);
    const std::string path = looks_like_path ? style : ("styles/" + style + ".bin");

    if (file_exists(path)) {
        std::vector<float> all;
        if (!read_f32_exact(path, kTemporalDim + kSourceLen, all, why)) {
            // A present-but-malformed style file is an error worth reporting, but
            // it must not take the whole run down — fall back and say so.
            const std::string bad = why;
            if (!load_fixture(out, why)) return false;
            why = "style '" + style + "' is malformed (" + bad + ") — using the fixture instead";
            return true;
        }
        out.temporal_input.assign(all.begin(), all.begin() + kTemporalDim);
        out.source.assign(all.begin() + kTemporalDim, all.end());
        out.name = style;
        out.is_fixture = false;
        why = "style '" + style + "' loaded from " + path;
        return true;
    }

    if (!load_fixture(out, why)) return false;
    why = "no style file at " + path + " — using the fixture conditioning";
    return true;
}

bool song_write_wav(const std::string& path, const std::vector<float>& pcm_interleaved, int n_samples) {
    if (n_samples <= 0 || pcm_interleaved.empty()) {
        NNOPT_ERROR("song_write_wav: nothing to write (0 samples)");
        return false;
    }
    const int channels = 2;
    std::vector<int16_t> pcm(pcm_interleaved.size());
    for (size_t i = 0; i < pcm_interleaved.size(); ++i) {
        float v = 0.5f * pcm_interleaved[i];
        v = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
        const long s = lround(32767.5f * v - 0.5f);
        pcm[i] = (int16_t)(s > 32767 ? 32767 : (s < -32768 ? -32768 : s));
    }
    FILE* wf = std::fopen(path.c_str(), "wb");
    if (!wf) { NNOPT_ERROR_FMT("song_write_wav: cannot open %s", path.c_str()); return false; }
    const int data_bytes = (int)pcm.size() * 2;
    const int byte_rate  = kSampleRate * channels * 2;
    auto w32 = [&](int v) {
        unsigned char b[4] = {(unsigned char)(v & 0xff), (unsigned char)((v >> 8) & 0xff),
                              (unsigned char)((v >> 16) & 0xff), (unsigned char)((v >> 24) & 0xff)};
        std::fwrite(b, 1, 4, wf);
    };
    auto w16 = [&](int v) {
        unsigned char b[2] = {(unsigned char)(v & 0xff), (unsigned char)((v >> 8) & 0xff)};
        std::fwrite(b, 1, 2, wf);
    };
    std::fwrite("RIFF", 1, 4, wf); w32(36 + data_bytes); std::fwrite("WAVE", 1, 4, wf);
    std::fwrite("fmt ", 1, 4, wf); w32(16); w16(1); w16(channels);
    w32(kSampleRate); w32(byte_rate); w16(channels * 2); w16(16);
    std::fwrite("data", 1, 4, wf); w32(data_bytes);
    const size_t wrote = std::fwrite(pcm.data(), 1, (size_t)data_bytes, wf);
    std::fclose(wf);
    if (wrote != (size_t)data_bytes) { NNOPT_ERROR_FMT("song_write_wav: short write on %s", path.c_str()); return false; }
    return true;
}
