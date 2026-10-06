// istft — inverse short-time Fourier transform (mag + phase → waveform).
//
// Generic primitive used by any vocoder that emits a magnitude / phase
// spectrogram and reconstructs a waveform on-device (iSTFTNet, MB-iSTFT-VITS,
// any other STFT-based decoder). For HiFi-GAN-style vocoders that emit the
// waveform directly via transposed convolutions, this kernel is NOT used.
//
// Math (real-input Hermitian iSTFT with overlap-add):
//
//   For each output sample n in [0, T_audio):
//     out[n] = sum over frames f that cover sample n of
//                window[n - f*hop] * idft_real(mag[f, :], phase[f, :], n - f*hop)
//
//   where idft_real(M, P, t) reconstructs one time sample from one frame's
//   spectrum using Hermitian symmetry of a real-input FFT:
//
//     idft_real(M, P, t) = (1/N) * (
//        M[0] * cos(P[0])                          // DC
//      + M[N/2] * cos(P[N/2] + π*t)                // Nyquist (only if N even)
//      + 2 * sum_{k=1}^{N/2-1} M[k] * cos(2πkt/N + P[k])
//     )
//
// The implementation below uses a direct DFT (O(N) per sample) rather than
// a radix-2 FFT. That is adequate for the small n_fft values typical of
// iSTFTNet (n_fft ≈ 20–64). For large n_fft (≥ 256) prefer an FFT-based
// rewrite via OptimizeKernel after first-port convergence.
//
// Inputs:
//   mag      — [n_frames, n_freq]  storage_t   n_freq = n_fft/2 + 1
//   phase    — [n_frames, n_freq]  storage_t
//   window   — [n_fft]             storage_t   typically Hann; precomputed host-side
// Output:
//   out      — [T_audio]           storage_t   T_audio = (n_frames-1)*hop_size + n_fft
//
// Launch shape: one work-item per output sample. Global size = T_audio.
//
// The host wrapper computes T_audio and ALSO computes a per-sample
// normalization buffer (sum-of-window-squared at each output position) if
// the chosen window does not naturally satisfy the COLA / constant-overlap-add
// condition at the given hop. Hann at hop = n_fft/4 is COLA-2 by definition
// and needs no normalization; other (hop, window) pairs do. The agent's
// host code decides which case applies based on the model config and either
// passes a normalization buffer or relies on the analytic constant.
// Dtype-template preamble — DO NOT EDIT. Driven by host-side -DNNOPT_USE_FP16.
#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)    vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)    ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
#endif


#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

__kernel void istft(
    __global const storage_t* mag,        // [n_frames, n_freq]
    __global const storage_t* phase,      // [n_frames, n_freq]
    __global const storage_t* window,     // [n_fft]
    __global       storage_t* out,        // [T_audio]
    const int n_frames,
    const int n_fft,
    const int hop_size,
    const int n_freq,                     // n_fft/2 + 1
    const int T_audio) {

    const int gid = get_global_id(0);
    if (gid >= T_audio) return;

    // Frames whose support [f*hop, f*hop + n_fft) covers sample gid:
    //   f*hop <= gid  AND  gid < f*hop + n_fft
    //   ⇒  (gid - n_fft + 1) / hop  <= f  <=  gid / hop
    int f_min = (gid - n_fft + 1 + hop_size - 1) / hop_size;
    if (f_min < 0) f_min = 0;
    int f_max = gid / hop_size;
    if (f_max > n_frames - 1) f_max = n_frames - 1;

    float acc = 0.0f;
    const float inv_n = 1.0f / (float)n_fft;
    const int has_nyquist = ((n_fft % 2) == 0);

    for (int f = f_min; f <= f_max; ++f) {
        const int t = gid - f * hop_size;        // sample index within frame [0, n_fft)
        if (t < 0 || t >= n_fft) continue;       // belt-and-suspenders

        const int base = f * n_freq;

        // DC bin (k=0): real-valued, no doubling, no phase contribution from
        // imaginary part — but phase[0] is captured separately to handle
        // models that store a signed real value as (|x|, 0 or π).
        float frame_acc = (float)LOAD(mag, base + 0) * cos((float)LOAD(phase, base + 0));

        // Mid bins (k=1 .. n_freq-2 when n_fft even, or .. n_freq-1 when odd):
        // contribute with factor 2 from Hermitian symmetry.
        const int mid_end = has_nyquist ? (n_freq - 1) : n_freq;
        const float two_pi_t_over_n = 2.0f * M_PI_F * (float)t * inv_n;
        for (int k = 1; k < mid_end; ++k) {
            const float m_k = (float)LOAD(mag,   base + k);
            const float p_k = (float)LOAD(phase, base + k);
            frame_acc += 2.0f * m_k * cos((float)k * two_pi_t_over_n + p_k);
        }

        // Nyquist bin (k = n_fft/2): real-valued, no doubling. cos(π*t)
        // alternates sign per sample.
        if (has_nyquist) {
            const float m_nyq = (float)LOAD(mag,   base + (n_freq - 1));
            const float p_nyq = (float)LOAD(phase, base + (n_freq - 1));
            frame_acc += m_nyq * cos(M_PI_F * (float)t + p_nyq);
        }

        frame_acc *= inv_n;

        // Apply synthesis window and overlap-add.
        const float w_t = (float)LOAD(window, t);
        acc += frame_acc * w_t;
    }

    STORE(out, gid, acc);
}
