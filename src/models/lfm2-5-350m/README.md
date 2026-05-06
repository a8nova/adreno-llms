# LFM2.5-350M-Base on Adreno (Android)

Liquid AI's hybrid conv+attention foundation model, ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android devices. Verified on Motorola Razr 2020 (Adreno 618).

- **Upstream:** [LiquidAI/LFM2.5-350M-Base](https://huggingface.co/LiquidAI/LFM2.5-350M-Base)
- **Parameters:** 350M
- **Architecture:** Hybrid (LFM2 — interleaved short conv + grouped attention)
- **Precision:** fp16

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch the converted weights
../../../scripts/fetch_weights.sh lfm2-5-350m

# 2. Build (release, fp16)
NNOPT_DTYPE=fp16 ./scripts/build.sh --release

# 3. Deploy binary + weights + tokenizer to /data/local/tmp/
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 4. Run a prompt and stream 64 tokens
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "Once upon a time" 64
```

## Performance

Razr 2020 / Adreno 618, fp16, greedy (`temperature=0, seed=42`), 32-token generation, canonical token IDs, 5-run warm median measured 2026-05-06.

| | Decode tok/s | TTFT (s) |
|---|---:|---:|
| Measured today | **10.20** | **3.72** |
| min / max across 5 runs | 10.03 / 10.40 | — |
| Roofline ceiling (10 GB/s) | 14.8 | — |

Per-token weight footprint: ~676 MB. Memory-bandwidth-bound. Full per-step optimization log + bottleneck census in [BENCHMARK.md](./BENCHMARK.md).

## Layout

```
.
├── BENCHMARK.md
├── CMakeLists.txt                # CMake 3.18+, C++17, CLBlast 1.6.3
├── kernels/                      # OpenCL kernels (Adreno-tuned)
├── reference/                    # Python HF reference + test inputs
├── scripts/                      # build.sh, deploy_android.sh, run_android.sh
├── src/                          # C++ inference code
└── weights/                      # model.fp16.bin pulled by fetch_weights.sh;
                                  # tokenizer.json + tokenizer_vocab.bin shipped here
```

## Licensing

- **Code:** Apache 2.0 (this repo).
- **Weights:** Liquid AI's open license — see the [model card](https://huggingface.co/LiquidAI/LFM2.5-350M-Base) for terms.
