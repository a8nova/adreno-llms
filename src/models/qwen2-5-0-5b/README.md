# Qwen2.5-0.5B on Adreno (Android)

Alibaba's Qwen2.5-0.5B (LLaMA-style + GQA) ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 618).

- **Upstream:** [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- **Parameters:** 500M
- **Architecture:** LLaMA-style transformer with grouped-query attention (GQA)
- **Precision:** fp16

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch the converted weights
../../../scripts/fetch_weights.sh qwen2-5-0-5b

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
| Measured today | **8.45** | **2.59** |
| min / max across 5 runs | 8.27 / 8.99 | — |
| Roofline ceiling (10 GB/s) | 10.6 | — |

Per-token weight footprint: ~942 MB. Reaches ~80% of the memory-bandwidth ceiling — the highest utilization across the models in this repo. Full optimization log in [BENCHMARK.md](./BENCHMARK.md).

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
- **Weights:** Apache 2.0 — see the [model card](https://huggingface.co/Qwen/Qwen2.5-0.5B).
