# Granite-4.0-350M on Adreno (Android)

IBM's Granite-4.0-350M (dense decoder transformer with grouped-query attention) ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 620).

- **Upstream:** [ibm-granite/granite-4.0-350m](https://huggingface.co/ibm-granite/granite-4.0-350m)
- **Parameters:** 350M
- **Architecture:** Dense decoder transformer — 28 layers, hidden=1024, GQA (16Q / 4KV heads, head_dim=64), SwiGLU MLP (intermediate=2048), RoPE θ=10000, tied lm_head (vocab=100,352)
- **Precision:** fp16

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch weights from the HuggingFace mirror
../../../scripts/fetch_weights.sh granite-4-0-350m

# 2. Build (release, fp16)
NNOPT_DTYPE=fp16 ./scripts/build.sh --release

# 3. Deploy
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 4. Run
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "Hello I am a language model" 32 --temperature 0 --seed 42
```

## Performance

Razr 2020 / Adreno 620 / Snapdragon 765G, fp16, greedy (`--temperature 0`), 32-token generation, 3-run warm median measured 2026-05-16.

| | Decode tok/s | TTFT (s) | Peak CPU mem (MB) |
|---|---:|---:|---:|
| Measured today | **10.19** | **2.41** | **2580** |
| Roofline ceiling (10 GB/s) | 14.8 | — | — |

Per-token weight footprint: ~676 MB (140 MB attention + 336 MB MLP + 200 MB tied lm_head). 69% of the 10 GB/s memory-bandwidth ceiling. Full optimization log — including the 13.5× sprint that took decode from 0.70 → 10.49 tok/s via custom GEMV M=1 + native_exp silu + image2d w_out — in [BENCHMARK.md](./BENCHMARK.md).

## Layout

```
.
├── BENCHMARK.md
├── CMakeLists.txt
├── kernels/                    # OpenCL kernels (Adreno-tuned)
├── reference/                  # HuggingFace parity harness
├── scripts/                    # build / deploy / run
├── src/                        # C++ inference
└── weights/                    # populated by fetch_weights.sh
```

## Licensing

- **Code:** Apache 2.0 (this repo).
- **Weights:** Apache 2.0 — see the upstream [model card](https://huggingface.co/ibm-granite/granite-4.0-350m). Redistributed via `a8nova/adreno-llms-weights` for offline-fetch convenience.
