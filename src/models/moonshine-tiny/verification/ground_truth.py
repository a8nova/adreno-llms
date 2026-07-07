#!/usr/bin/env python3
"""Regenerate PyTorch ground-truth token ids for the 3 verification clips.

Reads clip{A,B,C}.wav (16kHz mono s16), writes clip{A,B,C}.bin (raw f32)
and ground_truth.json with the greedy-decoded ids + text per clip.
Mirrors the port's inference: raw waveform in, generate() greedy.
"""
import json
import struct
import wave

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

MODEL_ID = "UsefulSensors/moonshine-tiny"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID)
model.eval()

results = {}
for clip in ["A", "B", "C"]:
    with wave.open(f"clip{clip}.wav", "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        n = w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16)
    wav = (pcm.astype(np.float32)) / 32768.0
    with open(f"clip{clip}.bin", "wb") as f:
        f.write(struct.pack(f"<{len(wav)}f", *wav))

    inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    ids_list = ids[0].tolist()
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    results[clip] = {
        "num_samples": int(len(wav)),
        "duration_s": round(len(wav) / 16000.0, 3),
        "ids": ids_list,
        "text": text,
    }
    print(f"clip {clip}: {len(wav)/16000.0:.2f}s  ids={ids_list}")
    print(f"         text={text!r}")

with open("ground_truth.json", "w") as f:
    json.dump(results, f, indent=2)
print("wrote ground_truth.json")
