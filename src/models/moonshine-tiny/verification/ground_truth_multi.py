#!/usr/bin/env python3
"""PyTorch ground truth for arbitrary clip list (args: clip letters)."""
import json
import struct
import sys
import wave

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

MODEL_ID = "UsefulSensors/moonshine-tiny"
clips = sys.argv[1:]

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID)
model.eval()

results = {}
for clip in clips:
    with wave.open(f"clip{clip}.wav", "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    wav = pcm.astype(np.float32) / 32768.0
    with open(f"clip{clip}.bin", "wb") as f:
        f.write(struct.pack(f"<{len(wav)}f", *wav))
    inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(**inputs, do_sample=False, max_new_tokens=192)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    results[clip] = {
        "duration_s": round(len(wav) / 16000.0, 3),
        "ids": ids[0].tolist(),
        "text": text,
    }
    print(f"clip {clip} ({len(wav)/16000.0:.2f}s): {text!r}")
    print(f"   ids ({len(ids[0])}): {ids[0].tolist()}")

with open("ground_truth_multi.json", "w") as f:
    json.dump(results, f, indent=2)
