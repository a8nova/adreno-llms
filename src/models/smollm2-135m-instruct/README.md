# SmolLM2-135M-Instruct on Adreno (Android)

HuggingFace's smallest instruct-tuned LLM (LLaMA-style + GQA) ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 618).

- **Upstream:** [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- **Parameters:** 135M
- **Architecture:** LLaMA-style transformer with grouped-query attention (GQA), instruct-tuned
- **Precision:** fp16

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch the converted weights
../../../scripts/fetch_weights.sh smollm2-135m-instruct

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
| Measured today | **14.57** | **1.61** |
| min / max across 5 runs | 13.86 / 15.32 | — |
| Roofline ceiling (10 GB/s) | 36.9 | — |

Per-token weight footprint: ~275 MB. Full optimization log + per-kernel timings in [BENCHMARK.md](./BENCHMARK.md).

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
- **Weights:** Apache 2.0 — see the [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).
