#!/usr/bin/env python3
"""Regenerate reference WAV with deterministic SineGen.
BYPASSES phonemizer (espeak subprocess was hanging) — uses input_ids from
assets/test_input_ids.bin directly via KModel.forward_with_tokens.
"""
import sys
import struct
import wave
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Patch SineGen for deterministic phase
def _patched_f02sine(self, f0_values):
    rad_values = (f0_values / self.sampling_rate) % 1
    rand_ini = torch.zeros(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
    rand_ini[:, 0] = 0
    rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
    if not self.flag_for_pulse:
        rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=1/self.upsample_scale, mode="linear").transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
        phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
        return torch.sin(phase)
    # pulse mode unused for Kokoro inference
    return torch.zeros_like(f0_values)

from kokoro import istftnet
istftnet.SineGen._f02sine = _patched_f02sine
print("[patch] SineGen deterministic", flush=True)
torch.manual_seed(0); np.random.seed(0)

# Load model (skip KPipeline)
from kokoro import KModel
print("[load] KModel", flush=True)
model = KModel(repo_id="hexgrad/Kokoro-82M").eval()
print("[load] model ready", flush=True)

# Load input_ids from existing assets (24 phoneme ids: BOS + 22 + EOS)
ids_path = Path("assets/test_input_ids.bin")
with open(ids_path, "rb") as f:
    raw = f.read()
n_ids = len(raw) // 4
input_ids = torch.tensor([struct.unpack("<i", raw[i*4:(i+1)*4])[0] for i in range(n_ids)],
                          dtype=torch.long).unsqueeze(0)
print(f"[input] loaded {n_ids} phoneme ids from {ids_path}", flush=True)

# Load voice pack and index for ref_s
voice_pack = np.fromfile("assets/voice_pack_af_heart.bin", dtype=np.float32).reshape(510, 256)
ref_idx = n_ids - 1
ref_s = torch.tensor(voice_pack[ref_idx].tolist(), dtype=torch.float32).unsqueeze(0)  # [1, 256]
print(f"[input] ref_s from voice pack index {ref_idx}", flush=True)

print("[run] model.forward_with_tokens (deterministic SineGen)", flush=True)
with torch.no_grad():
    audio, pred_dur = model.forward_with_tokens(input_ids, ref_s, speed=1)
audio_list = audio.detach().cpu().float().flatten().tolist()
print(f"[done] audio: {len(audio_list)} samples, range [{min(audio_list):.3f}, {max(audio_list):.3f}]", flush=True)

out = Path("reference_det")
out.mkdir(exist_ok=True)
pcm = bytearray()
for v in audio_list:
    s = max(-1.0, min(1.0, v)) * 32767
    pcm += struct.pack("<h", int(s))
with wave.open(str(out / "output.wav"), 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(bytes(pcm))
print(f"[wrote] {out / 'output.wav'}", flush=True)
