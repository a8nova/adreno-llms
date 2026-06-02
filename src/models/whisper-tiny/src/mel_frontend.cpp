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

std::vector<float> whisper_log_mel(const std::vector<float>& audio_in,
                                   const std::vector<float>& mel_filters) {
    // 1. Pad/trim the waveform to exactly 30 s (480000 samples).
    std::vector<float> wav(N_SAMPLES, 0.0f);
    {
        const size_t n = std::min<size_t>(N_SAMPLES, audio_in.size());
        std::copy(audio_in.begin(), audio_in.begin() + n, wav.begin());
    }

    // 2. Reflect-pad by N_FFT/2 on each side (np.pad mode="reflect": edge NOT repeated).
    const int P = N_FFT / 2;  // 200
    std::vector<float> padded((size_t)N_SAMPLES + 2 * P);
    for (int i = 0; i < P; i++) padded[i] = wav[P - i];                       // left reflect
    for (int i = 0; i < N_SAMPLES; i++) padded[P + i] = wav[i];               // body
    for (int i = 0; i < P; i++) padded[P + N_SAMPLES + i] = wav[N_SAMPLES - 2 - i];  // right reflect

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
    std::vector<float> mel_lin((size_t)N_MELS * N_FRAMES);
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
                mel_lin[(size_t)m * N_FRAMES + f] = (float)acc;
            }
        }
    };

    unsigned hw = std::thread::hardware_concurrency();
    int nthreads = (hw == 0) ? 4 : std::min<unsigned>(hw, 8);
    std::vector<std::thread> pool;
    int per = (N_FRAMES + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; t++) {
        int f0 = t * per, f1 = std::min(N_FRAMES, f0 + per);
        if (f0 >= f1) break;
        pool.emplace_back(worker, f0, f1);
    }
    for (auto& th : pool) th.join();

    // 6. Log + Whisper normalization. log10(max(mel,1e-10)); clamp to max-8; (x+4)/4.
    std::vector<float> out((size_t)N_MELS * N_FRAMES);
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
