# Mamba-130M on Adreno (Android)

State-spaces' selective SSM (Mamba) ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 618).

- **Upstream:** [state-spaces/mamba-130m-hf](https://huggingface.co/state-spaces/mamba-130m-hf)
- **Parameters:** 130M
- **Architecture:** Mamba (selective state-space model — no attention)
- **Precision:** fp16

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch the converted weights
../../../scripts/fetch_weights.sh mamba-130m

# 2. Build (release, fp16)
NNOPT_DTYPE=fp16 ./scripts/build.sh --release

# 3. Deploy
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 4. Run
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "Once upon a time" 64
```

## Performance

Razr 2020 / Adreno 618, fp16, greedy (`temperature=0, seed=42`), 32-token generation, canonical token IDs, 5-run warm median measured 2026-05-06.

| | Decode tok/s | TTFT (s) |
|---|---:|---:|
| Measured today | **22.15** | **1.60** |
| min / max across 5 runs | 21.68 / 23.20 | — |
| Roofline ceiling (10 GB/s) | 38.7 | — |

Per-token weight footprint: ~258 MB. Full optimization log + bottleneck census in [BENCHMARK.md](./BENCHMARK.md).

## Layout

```
.
├── BENCHMARK.md
├── CMakeLists.txt
├── kernels/
├── reference/
├── scripts/
├── src/
└── weights/
```

## Licensing

- **Code:** Apache 2.0 (this repo).
- **Weights:** Apache 2.0 — see the [model card](https://huggingface.co/state-spaces/mamba-130m-hf).
