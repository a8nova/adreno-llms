# Stable Audio Open Small on Adreno (Android)

Stability AI's Stable Audio Open Small (text→audio) ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 620 / Snapdragon 765G). The first latent-diffusion model in the repo — three submodels run fully on-device: **T5-small text conditioning** (device CPU), an **8-step DiT denoise loop** (GPU), and the **VAE audio decoder** (GPU) → 44.1 kHz stereo WAV. Seed noise ships from the host (`assets/`) for deterministic replay — never device RNG, so output is reproducible per (prompt, seed).

- **Upstream:** [stabilityai/stable-audio-open-small](https://huggingface.co/stabilityai/stable-audio-open-small) (via stable-audio-tools)
- **Architecture:** latent diffusion — T5 conditioning + DiT denoiser + VAE audio decoder
- **Precision:** fp16 (`model.fp16.bin` DiT+VAE 948 MB + `t5_encoder.fp16.bin` 256 MB + embedder/tokenizer sidecars)

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch weights from HuggingFace (three submodels + sidecars)
../../../scripts/fetch_weights.sh stable-audio-open-small

# 2. Build (release — required for representative perf)
NNOPT_DTYPE=fp16 ./scripts/build.sh --release

# 3. Deploy binary + kernels + weights
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 4. Generate — <prompt> <seed> <name> <seconds> <denoise-steps>
./scripts/generate.sh "warm analog synth chords with soft rain" 42 my_clip 11 8
# → multi_prompt_results/my_clip.wav
```

`scripts/multi_prompt_sweep.sh` runs a batch of prompts; `scripts/cos_check.py` compares per-layer dumps against the PyTorch reference.

## Performance

Razr 2020 / Adreno 620 / Snapdragon 765G, release fp16, 11 s stereo clip, 8 denoise steps, seed 42 (verified 2026-07-07):

| Stage | Wall |
|---|---:|
| T5 conditioning (device CPU) | 8.3 s |
| DiT denoise, 8 steps | 51.5 s (6.44 s/step) |
| VAE decode | 40.4 s |
| **Total** | **100.7 s** (RTF ≈ 9.2) |

Peak host memory 2.4 GB. Output validated non-silent (peak 23007, rms 5587) and deterministic per (prompt, seed). Diffusion is the heaviest workload class in the repo and the Adreno 620 is the entry tier, so it is **not real-time** — full optimization log in [BENCHMARK.md](./BENCHMARK.md).

## Layout

```
.
├── BENCHMARK.md          # optimization log
├── CMakeLists.txt
├── assets/               # host-generated seed noise + fixtures
├── kernels/              # one OpenCL file per op family (conv_1d, conv_transpose_1d, dit, decoder, istft, utils)
├── reference/            # PyTorch reference + cosine-validation fixtures
├── scripts/              # build / deploy / generate / sweep / cos_check
├── src/                  # dit.cpp + decoder.cpp GPU drivers, main.cpp orchestrates T5 → DiT → VAE, ops/ per module class
└── weights/              # fetched from HuggingFace (not committed)
```

## Licensing

- **Code:** Apache 2.0 (this repo).
- **Weights:** **Stability AI Community License** — see the [model card](https://huggingface.co/stabilityai/stable-audio-open-small) before redistribution.
