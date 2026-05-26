#!/usr/bin/env python3
"""Offline debug fixture generator for the duration predictor port.

Writes:
  reference/debug_dp_inputs.bin    [B=1, H=192, T_chars=33] conditioning vector
  reference/debug_dp_noise.bin     [B=1, 2, T_chars=33] noise * noise_scale_duration
  reference/debug_dp_log_dur.bin   [T_chars=33] ground-truth log_durations

These three together let us validate the C++ inverse pass bit-by-bit: feed
text_encoder_out + debug_dp_noise → C++ should produce debug_dp_log_dur.
"""
import sys, os
import numpy as np
import torch
from transformers import VitsModel

PORT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(PORT, "reference")

model = VitsModel.from_pretrained("facebook/mms-tts-eng", torch_dtype=torch.float32)
model.eval()

# Load reference text encoder output (channel-LAST [T, H])
te = np.fromfile(os.path.join(REF, "layers/text_encoder_out_output.bin"), dtype=np.float32)
T_chars = te.size // 192
te_btc = te.reshape(1, T_chars, 192)
hidden_states = torch.from_numpy(te_btc).transpose(1, 2)  # [1, 192, T]
padding_mask = torch.ones(1, 1, T_chars, dtype=torch.float32)

dp = model.duration_predictor

with torch.no_grad():
    # Step 1: conditioning vector
    inputs = dp.conv_pre(hidden_states)
    inputs = dp.conv_dds(inputs, padding_mask)
    inputs = dp.conv_proj(inputs) * padding_mask

inputs.numpy().astype(np.float32).tofile(os.path.join(REF, "debug_dp_inputs.bin"))
print(f"inputs: shape={tuple(inputs.shape)} rms={inputs.float().pow(2).mean().sqrt().item():.4f}")

# Step 2: known UNSCALED noise (the C++ op multiplies by noise_scale_duration=0.8 internally)
torch.manual_seed(42)
raw_noise = torch.randn(1, 2, T_chars, dtype=torch.float32)
raw_noise.numpy().tofile(os.path.join(REF, "duration_noise.bin"))   # overwrite existing
noise = raw_noise * 0.8
print(f"noise (post-scale): shape={tuple(noise.shape)} rms={noise.pow(2).mean().sqrt().item():.4f}")

# Step 3: inverse flow chain (mirrors HF VitsStochasticDurationPredictor.forward(reverse=True))
with torch.no_grad():
    latents = noise.clone()
    flows = list(reversed(dp.flows))
    flows = flows[:-2] + [flows[-1]]   # drop the "useless vflow"
    for f in flows:
        latents = torch.flip(latents, [1])
        latents, _ = f(latents, padding_mask, global_conditioning=inputs, reverse=True)
    log_duration = latents[:, 0, :]    # [1, T]

log_duration.numpy().astype(np.float32).tofile(os.path.join(REF, "debug_dp_log_dur.bin"))
print(f"log_duration: shape={tuple(log_duration.shape)} values={log_duration[0].tolist()}")
print(f"  ceil(exp): {torch.ceil(torch.exp(log_duration))[0].tolist()}  sum={int(torch.ceil(torch.exp(log_duration)).sum().item())}")
