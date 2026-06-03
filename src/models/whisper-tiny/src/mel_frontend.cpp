#include "mel_frontend.h"
#include <cmath>
#include <thread>
#include <algorithm>
#include <vector>

namespace {
constexpr int N_FFT = 400;
constexpr int HOP = 160;
constexpr int N_MELS = 80;
constexpr int N_FREQ = 201;   // N_FFT/2 + 1
constexpr int N_FRAMES = 3000;
constexpr int N_SAMPLES = 480000;  // 30s @ 16kHz
constexpr double PI = 3.14159265358979323846;
}  // namespace

// Core log-mel over a caller-chosen window: pad/trim the waveform to
// `n_samples` (= n_frames * HOP) and emit an [N_MELS, n_frames] spectrogram.
// whisper_log_mel() is the fixed 30s/3000-frame path (byte-exact as before);
// whisper_log_mel_n() runs the SAME math on a shorter window so the encoder can
// process the actual audio length (streaming) instead of always 30s.
static std::vector<float> mel_core(const std::vector<float>& audio_in,
                                   const std::vector<float>& mel_filters,
                                   int n_samples, int n_frames) {
    // 1. Pad/trim the waveform to exactly n_samples.
    std::vector<float> wav((size_t)n_samples, 0.0f);
    {
        const size_t n = std::min<size_t>((size_t)n_samples, audio_in.size());
        std::copy(audio_in.begin(), audio_in.begin() + n, wav.begin());
    }

    // 2. Reflect-pad by N_FFT/2 on each side (np.pad mode="reflect": edge NOT repeated).
    const int P = N_FFT / 2;  // 200
    std::vector<float> padded((size_t)n_samples + 2 * P);
    for (int i = 0; i < P; i++) padded[i] = wav[P - i];                       // left reflect
    for (int i = 0; i < n_samples; i++) padded[P + i] = wav[i];               // body
    for (int i = 0; i < P; i++) padded[P + n_samples + i] = wav[n_samples - 2 - i];  // right reflect

    // 3. Periodic Hann window: np.hanning(N_FFT+1)[:-1].
    std::vector<double> win(N_FFT);
    for (int n = 0; n < N_FFT; n++) win[n] = 0.5 - 0.5 * std::cos(2.0 * PI * n / N_FFT);

    // 4. Precompute DFT twiddles for the 201 retained bins: cos/sin(2*pi*k*n/N_FFT).
    std::vector<double> cosT((size_t)N_FREQ * N_FFT), sinT((size_t)N_FREQ * N_FFT);
    for (int k = 0; k < N_FREQ; k++) {
        for (int n = 0; n < N_FFT; n++) {
            const double ang = 2.0 * PI * k * n / N_FFT;
            cosT[(size_t)k * N_FFT + n] = std::cos(ang);
            sinT[(size_t)k * N_FFT + n] = std::sin(ang);
        }
    }

    // 5. Per-frame STFT power -> mel (linear). Parallelized over frame columns; each
    //    thread writes disjoint columns of mel_lin[m * N_FRAMES + f], so no locking.
    std::vector<float> mel_lin((size_t)N_MELS * n_frames);
    auto worker = [&](int f0, int f1) {
        std::vector<double> wf(N_FFT);
        std::vector<double> power(N_FREQ);
        for (int f = f0; f < f1; f++) {
            const int start = f * HOP;
            for (int n = 0; n < N_FFT; n++) wf[n] = (double)padded[start + n] * win[n];
            for (int k = 0; k < N_FREQ; k++) {
                const double* c = &cosT[(size_t)k * N_FFT];
                const double* s = &sinT[(size_t)k * N_FFT];
                double re = 0.0, im = 0.0;
                for (int n = 0; n < N_FFT; n++) { re += wf[n] * c[n]; im -= wf[n] * s[n]; }
                power[k] = re * re + im * im;
            }
            for (int m = 0; m < N_MELS; m++) {
                const float* fb = &mel_filters[(size_t)m * N_FREQ];
                double acc = 0.0;
                for (int k = 0; k < N_FREQ; k++) acc += (double)fb[k] * power[k];
                mel_lin[(size_t)m * n_frames + f] = (float)acc;
            }
        }
    };

    unsigned hw = std::thread::hardware_concurrency();
    int nthreads = (hw == 0) ? 4 : std::min<unsigned>(hw, 8);
    std::vector<std::thread> pool;
    int per = (n_frames + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; t++) {
        int f0 = t * per, f1 = std::min(n_frames, f0 + per);
        if (f0 >= f1) break;
        pool.emplace_back(worker, f0, f1);
    }
    for (auto& th : pool) th.join();

    // 6. Log + Whisper normalization. log10(max(mel,1e-10)); clamp to max-8; (x+4)/4.
    std::vector<float> out((size_t)N_MELS * n_frames);
    float gmax = -1e30f;
    for (size_t i = 0; i < mel_lin.size(); i++) {
        float v = (float)std::log10((double)std::max(mel_lin[i], 1e-10f));
        out[i] = v;
        if (v > gmax) gmax = v;
    }
    const float floor = gmax - 8.0f;
    for (size_t i = 0; i < out.size(); i++) {
        float v = out[i] < floor ? floor : out[i];
        out[i] = (v + 4.0f) / 4.0f;
    }
    return out;
}

// Fixed 30 s / 3000-frame path — byte-identical to the original (eval/batch).
std::vector<float> whisper_log_mel(const std::vector<float>& audio_in,
                                   const std::vector<float>& mel_filters) {
    return mel_core(audio_in, mel_filters, N_SAMPLES, N_FRAMES);
}

// Variable-length path for streaming: encode the ACTUAL window (rounded to an
// even frame count, clamped to [MIN, 3000]) so a 2 s window costs ~1/15th of a
// 30 s encode. Whisper's per-input normalization makes this self-contained.
std::vector<float> whisper_log_mel_n(const std::vector<float>& audio_in,
                                     const std::vector<float>& mel_filters,
                                     int max_frames, int min_frames) {
    // Quantize the window to fixed 4 s buckets {400, 800, …, 3000 frames}. Without
    // this, every distinct window length is a new GEMM shape and CLBlast re-JITs
    // each one (~seconds) — defeating the speedup. ~8 buckets → each JITs once per
    // process, then cached. A window is rounded UP to its bucket (≤4 s of extra pad).
    constexpr int BUCKET = 400;  // 4 s
    int raw = (int)((audio_in.size() + HOP - 1) / HOP);
    int n_frames = ((raw + BUCKET - 1) / BUCKET) * BUCKET;  // round up to bucket
    if (n_frames < BUCKET) n_frames = BUCKET;
    // Context floor (streaming FINALs): pad short phrases up to min_frames so tiny
    // has enough zero-padded context not to loop. Round the floor to the bucket grid
    // too, so it lands on an already-JITed shape.
    if (min_frames > 0) {
        const int floor_b = ((min_frames + BUCKET - 1) / BUCKET) * BUCKET;
        if (n_frames < floor_b) n_frames = floor_b;
    }
    int cap = (max_frames > 0 && max_frames < N_FRAMES) ? max_frames : N_FRAMES;
    if (n_frames > cap) n_frames = cap;
    return mel_core(audio_in, mel_filters, n_frames * HOP, n_frames);
}
