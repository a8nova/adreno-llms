#!/usr/bin/env python3
"""Pin down the EXACT Whisper log-mel front-end and export C++ assets.

1. Re-save each bench clip's raw waveform (16 kHz mono float32) -> sample_NN_audio.bin
2. Dump the HF mel filterbank [80,201] -> bench/mel_filters.bin (row-major, f32)
3. Re-implement log_mel in pure numpy and VALIDATE it against the WhisperProcessor
   output already saved as sample_NN_features.bin (cosine must be ~1.0). This proves
   the recipe before it's ported to C++ in main.cpp.

Whisper/HF recipe (n_fft=400, hop=160, n_mels=80, 30s = 480000 samples):
  pad/trim waveform to 480000
  center-pad reflect by n_fft//2=200, frame (len 400, hop 160), apply hann(401)[:-1]
  rfft size 400 -> 201 bins; drop the LAST time frame -> 3000 frames
  power = |stft|^2 ; mel = filters[80,201] @ power[201,3000]
  log = log10(max(mel, 1e-10)); log = max(log, log.max()-8); log = (log+4)/4
"""
import io, json, os
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "bench")
BENCH = os.path.abspath(BENCH)
ASSETS = os.path.join(HERE, "..", "assets")
N_FFT, HOP, N_MELS, N_SAMPLES, N_FRAMES = 400, 160, 80, 480000, 3000

fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
mel_filters = np.asarray(fe.mel_filters, dtype=np.float32)  # HF stores [201, 80]
print("[mel] HF mel_filters shape:", mel_filters.shape)
# We want filters as [80, 201] for mel = filters @ power.
if mel_filters.shape == (201, 80):
    filters_80x201 = mel_filters.T.copy()
else:
    filters_80x201 = mel_filters.copy()
assert filters_80x201.shape == (80, 201), filters_80x201.shape
filters_80x201.astype(np.float32).tofile(os.path.join(BENCH, "mel_filters.bin"))
filters_80x201.astype(np.float32).tofile(os.path.join(ASSETS, "mel_filters.bin"))
print(f"[mel] wrote mel_filters.bin [80,201] to bench/ and assets/")

window = np.hanning(N_FFT + 1)[:-1].astype(np.float64)  # periodic Hann, len 400


def log_mel(waveform):
    wav = np.asarray(waveform, dtype=np.float64)
    if len(wav) < N_SAMPLES:
        wav = np.pad(wav, (0, N_SAMPLES - len(wav)))
    else:
        wav = wav[:N_SAMPLES]
    # center=True reflect pad
    p = N_FFT // 2
    padded = np.pad(wav, (p, p), mode="reflect")
    n_frames_full = 1 + (len(padded) - N_FFT) // HOP
    # frames
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames_full)[:, None]
    frames = padded[idx] * window  # [n_frames_full, 400]
    stft = np.fft.rfft(frames, n=N_FFT, axis=1)  # [n_frames_full, 201]
    stft = stft[:-1]  # drop last frame -> 3000
    power = (np.abs(stft) ** 2).T  # [201, n_frames]
    mel = filters_80x201.astype(np.float64) @ power  # [80, n_frames]
    log = np.log10(np.maximum(mel, 1e-10))
    log = np.maximum(log, log.max() - 8.0)
    log = (log + 4.0) / 4.0
    return log.astype(np.float32)


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# Reload audio (need raw waveforms again)
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(decode=False))


def load_audio(a):
    if a.get("bytes"):
        arr, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
    else:
        arr, sr = sf.read(a["path"], dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return np.asarray(arr, dtype=np.float32), int(sr)


manifest = json.load(open(os.path.join(BENCH, "manifest.json")))
worst = 1.0
for m in manifest:
    i = m["id"]
    arr, sr = load_audio(ds[i]["audio"])
    assert sr == 16000, sr
    # save raw waveform for on-device front-end
    arr.astype(np.float32).tofile(os.path.join(BENCH, f"sample_{i:02d}_audio.bin"))
    # validate our numpy log-mel vs the known-good WhisperProcessor .bin
    ref = np.fromfile(os.path.join(BENCH, f"sample_{i:02d}_features.bin"), dtype=np.float32).reshape(80, 3000)
    mine = log_mel(arr)
    c = cosine(mine, ref)
    mae = float(np.abs(mine - ref).mean())
    worst = min(worst, c)
    m["audio_bin"] = f"sample_{i:02d}_audio.bin"
    m["n_audio_samples"] = int(len(arr))
    print(f"[mel] sample_{i:02d}: cosine={c:.6f} mae={mae:.4e}")

json.dump(manifest, open(os.path.join(BENCH, "manifest.json"), "w"), indent=2)
print(f"[mel] worst cosine across 10 clips: {worst:.6f}  (need >0.9999)")
