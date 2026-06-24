#pragma once
// OpenVoice V2 linear-spectrogram front-end (CPU, header-only).
//
// Mirrors openvoice/mel_processing.py::spectrogram_torch EXACTLY:
//   n_fft = win = 1024, hop = 256, center = False,
//   reflect-pad (n_fft - hop)/2 = 384 each side (no edge repeat),
//   periodic Hann window, onesided rFFT -> 513 bins,
//   magnitude = sqrt(re^2 + im^2 + 1e-6).
//
// Input audio is float32 mono in [-1, 1] at 22.05 kHz (same contract as
// librosa.load(sr=22050)). Output is row-major [N_FREQ=513, T] (channel-major) —
// the exact layout enc_q.pre (Conv1d 513->192) consumes as Buf{C=513, T}.
//
// Implementation is a direct DFT over the 513 retained bins (same approach as
// whisper-tiny/src/mel_frontend.cpp), threaded over frame columns. It runs once
// per utterance on the CPU while the GPU is otherwise idle, so an FFT is not
// worth the added complexity here.

#include <vector>
#include <cmath>
#include <thread>
#include <algorithm>

namespace nnopt {

inline std::vector<float> ov_spectrogram(const std::vector<float>& audio, int* out_T) {
    constexpr int N_FFT = 1024, HOP = 256, WIN = 1024, N_FREQ = 513;
    constexpr int PAD = (N_FFT - HOP) / 2;        // 384
    constexpr double PI = 3.14159265358979323846;

    const int n = (int)audio.size();
    if (n < PAD + 2) { if (out_T) *out_T = 0; return {}; }

    // Reflect-pad by PAD on each side (numpy/torch reflect: edge sample NOT repeated).
    std::vector<float> padded((size_t)n + 2 * PAD);
    for (int i = 0; i < PAD; i++) padded[i] = audio[PAD - i];
    for (int i = 0; i < n;   i++) padded[PAD + i] = audio[i];
    for (int i = 0; i < PAD; i++) padded[PAD + n + i] = audio[n - 2 - i];
    const int L = (int)padded.size();

    const int T = 1 + (L - N_FFT) / HOP;          // center=False frame count
    if (out_T) *out_T = T;
    if (T <= 0) return {};

    // Periodic Hann: 0.5 - 0.5*cos(2*pi*i/WIN), i in [0, WIN).
    std::vector<double> win(WIN);
    for (int i = 0; i < WIN; i++) win[i] = 0.5 - 0.5 * std::cos(2.0 * PI * i / WIN);

    // DFT twiddles for the 513 retained (onesided) bins.
    std::vector<double> cosT((size_t)N_FREQ * N_FFT), sinT((size_t)N_FREQ * N_FFT);
    for (int k = 0; k < N_FREQ; k++) {
        for (int j = 0; j < N_FFT; j++) {
            const double a = 2.0 * PI * k * j / N_FFT;
            cosT[(size_t)k * N_FFT + j] = std::cos(a);
            sinT[(size_t)k * N_FFT + j] = std::sin(a);
        }
    }

    std::vector<float> spec((size_t)N_FREQ * T);  // [freq, T] channel-major
    auto worker = [&](int t0, int t1) {
        std::vector<double> wf(N_FFT);
        for (int t = t0; t < t1; t++) {
            const int s = t * HOP;
            for (int j = 0; j < N_FFT; j++) wf[j] = (double)padded[s + j] * win[j];
            for (int k = 0; k < N_FREQ; k++) {
                const double* c  = &cosT[(size_t)k * N_FFT];
                const double* sn = &sinT[(size_t)k * N_FFT];
                double re = 0.0, im = 0.0;
                for (int j = 0; j < N_FFT; j++) { re += wf[j] * c[j]; im -= wf[j] * sn[j]; }
                spec[(size_t)k * T + t] = (float)std::sqrt(re * re + im * im + 1e-6);
            }
        }
    };

    unsigned hw = std::thread::hardware_concurrency();
    int nthreads = (hw == 0) ? 4 : std::min<unsigned>(hw, 8);
    int per = (T + nthreads - 1) / nthreads;
    std::vector<std::thread> pool;
    for (int t = 0; t < nthreads; t++) {
        int a = t * per, b = std::min(T, a + per);
        if (a >= b) break;
        pool.emplace_back(worker, a, b);
    }
    for (auto& th : pool) th.join();
    return spec;
}

}  // namespace nnopt
