// Kaldi-povey fbank: per-frame windowing, DFT power spectrum, mel filterbank+log.

// remove-DC mean, preemphasis 0.97 (elementwise on mean-removed originals),
// povey window ^0.85. One work-item per frame.
__kernel void fbank_window(__global const storage_t* audio, __global storage_t* win_out,
                           const int nframes, const int FL, const int FS) {
    int t = get_global_id(0);
    if (t >= nframes) return;
    int base = t * FS;
    float mean = 0.0f;
    for (int i = 0; i < FL; ++i) mean += (float)LOAD(audio, base + i);
    mean /= (float)FL;
    for (int i = 0; i < FL; ++i) {
        float ci = (float)LOAD(audio, base + i) - mean;
        float p = (i >= 1) ? (ci - 0.97f * ((float)LOAD(audio, base + i - 1) - mean)) : (ci - 0.97f * ci);
        float wv = pow(0.5f - 0.5f * cos(2.0f * M_PI_F * (float)i / (float)(FL - 1)), 0.85f);
        STORE(win_out, t * FL + i, p * wv);
    }
}

// Power spectrum via direct DFT (== zero-padded radix-2 FFT magnitudes). fp32 out.
__kernel void fbank_power(__global const storage_t* win, __global float* power,
                          const int nframes, const int FL, const int NFFT, const int NSPEC) {
    int gid = get_global_id(0);
    if (gid >= nframes * NSPEC) return;
    int t = gid / NSPEC, k = gid - t * NSPEC;
    float re = 0.0f, im = 0.0f;
    int b = t * FL;
    for (int n = 0; n < FL; ++n) {
        float ang = 2.0f * M_PI_F * (float)k * (float)n / (float)NFFT;
        float wv = (float)LOAD(win, b + n);
        re += wv * cos(ang);
        im -= wv * sin(ang);
    }
    power[t * NSPEC + k] = re * re + im * im;
}

// 80 triangular mel filters + log. One work-item per (frame, mel band).
__kernel void fbank_mel(__global const float* power, __global storage_t* out,
                        const int nframes, const int NB, const int NSPEC,
                        const int SR, const int NFFT, const float eps) {
    int gid = get_global_id(0);
    if (gid >= nframes * NB) return;
    int t = gid / NB, m = gid - t * NB;
    float mlow = 1127.0f * log(1.0f + 20.0f / 700.0f);
    float mhigh = 1127.0f * log(1.0f + ((float)SR / 2.0f) / 700.0f);
    float mdelta = (mhigh - mlow) / (float)(NB + 1);
    float left = mlow + (float)m * mdelta, center = mlow + (float)(m + 1) * mdelta, right = mlow + (float)(m + 2) * mdelta;
    float e = 0.0f;
    int pb = t * NSPEC;
    for (int k = 0; k < NSPEC; ++k) {
        float mel = 1127.0f * log(1.0f + ((float)k * (float)SR / (float)NFFT) / 700.0f);
        if (mel > left && mel < right) {
            float w = (mel <= center) ? (mel - left) / (center - left) : (right - mel) / (right - center);
            e += w * power[pb + k];
        }
    }
    STORE(out, t * NB + m, log(fmax(e, eps)));
}
