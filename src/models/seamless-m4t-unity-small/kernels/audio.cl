// Audio pre/post-processing on the GPU: PCM16 decode (downmix to mono + /32768),
// linear resample to 16 kHz, and PCM16 encode (clamp + round). WAV container
// byte parsing happens on the CPU (src/wav_io.cpp) — these kernels do the math.

// Interleaved int16 [nframes, nch] -> mono fp16 normalized [nframes].
__kernel void audio_decode_s16(__global const short* in, __global storage_t* out,
                               const int nframes, const int nch) {
    int i = get_global_id(0);
    if (i >= nframes) return;
    float acc = 0.0f;
    for (int c = 0; c < nch; ++c) acc += (float)in[i * nch + c];
    STORE(out, i, (acc / (float)nch) / 32768.0f);
}

// Linear resample. ratio = rate_in / rate_out. out[j] = lerp(in, j*ratio).
__kernel void audio_resample_linear(__global const storage_t* in, __global storage_t* out,
                                    const int nin, const int nout, const float ratio) {
    int j = get_global_id(0);
    if (j >= nout) return;
    float pos = (float)j * ratio;
    int i0 = (int)pos;
    float frac = pos - (float)i0;
    int i1 = i0 + 1;
    if (i0 < 0) i0 = 0;
    if (i1 >= nin) i1 = nin - 1;
    if (i0 >= nin) i0 = nin - 1;
    float a = (float)LOAD(in, i0), b = (float)LOAD(in, i1);
    STORE(out, j, a + frac * (b - a));
}

// fp16 waveform [-1,1] -> int16 PCM (clamp + round).
__kernel void audio_encode_s16(__global const storage_t* in, __global short* out, const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float v = (float)LOAD(in, i) * 32767.0f;
    v = fmax(-32768.0f, fmin(32767.0f, v));
    out[i] = (short)(v < 0.0f ? v - 0.5f : v + 0.5f);
}
