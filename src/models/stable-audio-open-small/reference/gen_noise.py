#!/usr/bin/env python3
# Seed-only noise generator — no model load, runs in ~2s. Draw order mirrors
# gen_e2e_reference.py exactly: manual_seed(seed) -> randn([1,64,256]) init
# -> 8x randn_like per-step. Sigmas copied from the LogSNRShift schedule.
import os, sys
import numpy as np
import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "multi_prompts/noise_seed0"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 8   # binary derives steps from sigmas.bin
LATENT_C, LATENT_LEN = 64, 256

os.makedirs(OUT, exist_ok=True)
t_lin = np.linspace(1.0, 0.0, STEPS + 1)
logsnr = 2.0 - t_lin * (2.0 - (-6.2))
sig = (1.0 / (1.0 + np.exp(logsnr))).astype(np.float32)
sig[0], sig[-1] = 1.0, 0.0
sig.tofile(os.path.join(OUT, "sigmas.bin"))

torch.manual_seed(SEED)
init = torch.randn([1, LATENT_C, LATENT_LEN])
init.numpy().astype(np.float32).tofile(os.path.join(OUT, "init_noise.bin"))
for i in range(STEPS):
    torch.randn_like(init).numpy().astype(np.float32).tofile(
        os.path.join(OUT, f"step_noise_{i}.bin"))
print(f"PROGRESS: noise seed={SEED} steps={STEPS} -> {OUT}")
