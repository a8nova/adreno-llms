#!/usr/bin/env python3
"""Build a 10-clip LibriSpeech benchmark set for Whisper RTF testing.

For each sample:
  - extract Whisper log-mel input_features [80, 3000] (same WhisperProcessor path
    the reference uses) -> bench/sample_NN_features.bin  (float32, C-contiguous)
  - record real audio duration (seconds) for RTF = proc_time / duration
  - capture the HF whisper-tiny transcript (greedy) as the accuracy reference
  - capture the dataset ground-truth text

Writes bench/manifest.json with one entry per clip.
"""
import io, json, os, sys
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

OUT = os.path.join(os.path.dirname(__file__), "..", "bench")
OUT = os.path.abspath(OUT)
N = 10
MODEL = "openai/whisper-tiny"

os.makedirs(OUT, exist_ok=True)
print(f"[prep] output dir: {OUT}")

print("[prep] loading hf-internal-testing/librispeech_asr_dummy (clean/validation)...")
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# datasets>=4 needs torchcodec to auto-decode audio; sidestep it by reading the
# raw FLAC bytes/path with soundfile ourselves.
ds = ds.cast_column("audio", Audio(decode=False))
n = min(N, len(ds))
print(f"[prep] dataset has {len(ds)} samples; taking {n}")


def load_audio(a):
    """Return (float32 mono array, sample_rate) from a non-decoded audio cell."""
    if a.get("bytes"):
        arr, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
    else:
        arr, sr = sf.read(a["path"], dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return np.asarray(arr, dtype=np.float32), int(sr)

print(f"[prep] loading {MODEL} (processor + model)...")
proc = WhisperProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(MODEL).eval()

manifest = []
for i in range(n):
    s = ds[i]
    arr, sr = load_audio(s["audio"])
    dur = len(arr) / sr

    feat = proc(arr, sampling_rate=sr, return_tensors="pt", padding="max_length")
    input_features = feat["input_features"]  # [1, 80, 3000]
    mel = input_features.squeeze(0).contiguous().numpy().astype(np.float32)  # [80, 3000]
    assert mel.shape == (80, 3000), mel.shape

    fname = f"sample_{i:02d}_features.bin"
    mel.tofile(os.path.join(OUT, fname))

    # HF reference transcript (greedy) — the accuracy target for the C++ port.
    import torch
    with torch.no_grad():
        gen = model.generate(input_features, do_sample=False, num_beams=1, max_new_tokens=128)
    hf_text = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
    gt_text = (s.get("text") or "").strip()

    print(f"[prep] {fname}  dur={dur:5.2f}s  hf='{hf_text[:60]}'")
    manifest.append({
        "id": i,
        "features_bin": fname,
        "duration_s": round(dur, 3),
        "sample_rate": sr,
        "hf_transcript": hf_text,
        "ground_truth": gt_text,
    })

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"[prep] wrote {len(manifest)} clips + manifest.json to {OUT}")
