#!/usr/bin/env python3
"""E1 STFT verification + test-WAV prep.

1. Loads a source audio at 22.05 kHz mono, writes a 16-bit PCM WAV (the exact
   samples the device will read — no resampler mismatch).
2. Reimplements the C++ ov_spectrogram() math in numpy and asserts it matches
   upstream openvoice.mel_processing.spectrogram_torch (cosine ~1.0). This
   proves the only new numerical kernel in E1 is correct.
3. Saves the reference spectrogram [513,T] as fp32 for an optional device-side
   compare.

Run with the upstream env:
  ~/.nnopt/ref_venvs/env_latest/bin/python scripts/verify_stft.py <in_audio> <out_dir>
"""
import sys, struct, math
import numpy as np
import librosa, soundfile as sf
import torch
from openvoice.mel_processing import spectrogram_torch

N_FFT, HOP, WIN, N_FREQ = 1024, 256, 1024, 513
PAD = (N_FFT - HOP) // 2  # 384
SR = 22050


def numpy_ov_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Exact mirror of src/ov_stft.h ov_spectrogram()."""
    n = len(audio)
    padded = np.empty(n + 2 * PAD, dtype=np.float64)
    for i in range(PAD):
        padded[i] = audio[PAD - i]
    padded[PAD:PAD + n] = audio
    for i in range(PAD):
        padded[PAD + n + i] = audio[n - 2 - i]
    L = len(padded)
    T = 1 + (L - N_FFT) // HOP
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(WIN) / WIN)
    spec = np.empty((N_FREQ, T), dtype=np.float64)
    k = np.arange(N_FREQ)[:, None]
    j = np.arange(N_FFT)[None, :]
    cosT = np.cos(2.0 * np.pi * k * j / N_FFT)
    sinT = np.sin(2.0 * np.pi * k * j / N_FFT)
    for t in range(T):
        wf = padded[t * HOP: t * HOP + N_FFT] * win
        re = cosT @ wf
        im = -(sinT @ wf)
        spec[:, t] = np.sqrt(re * re + im * im + 1e-6)
    return spec


def cosine(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    in_audio = sys.argv[1]
    out_dir = sys.argv[2]
    audio, _ = librosa.load(in_audio, sr=SR, mono=True)
    audio = audio.astype(np.float32)
    # Round-trip through 16-bit so device reads the SAME samples.
    pcm16 = np.clip(np.round(audio * 32768.0), -32768, 32767).astype(np.int16)
    wav_path = f"{out_dir}/src_22k.wav"
    sf.write(wav_path, pcm16, SR, subtype="PCM_16")
    audio_q = pcm16.astype(np.float32) / 32768.0  # quantized, matches device load
    print(f"wrote {wav_path}: {len(audio_q)} samples = {len(audio_q)/SR:.2f}s @ {SR}Hz")

    # Upstream reference spectrogram on the SAME quantized samples.
    y = torch.FloatTensor(audio_q).unsqueeze(0)
    ref = spectrogram_torch(y, N_FFT, SR, HOP, WIN, center=False)[0].numpy()  # [513,T]
    mine = numpy_ov_spectrogram(audio_q)                                      # [513,T]
    Tm = min(ref.shape[1], mine.shape[1])
    cos = cosine(ref[:, :Tm], mine[:, :Tm])
    mae = float(np.mean(np.abs(ref[:, :Tm] - mine[:, :Tm])))
    print(f"spec shapes: ref {ref.shape} mine {mine.shape}")
    print(f"STFT cosine(ref, numpy-mirror) = {cos:.6f} | MAE = {mae:.3e}")

    ref.astype(np.float32).tofile(f"{out_dir}/ref_spec.f32.bin")
    print(f"saved ref_spec.f32.bin [513,{ref.shape[1]}]")
    print("RESULT:", "PASS" if cos > 0.9999 else "FAIL")


if __name__ == "__main__":
    main()
