# OpenELM-270M on Adreno (Android)

Apple's OpenELM (LLaMA-style efficient transformer) ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 618).

- **Upstream:** [apple/OpenELM-270M](https://huggingface.co/apple/OpenELM-270M)
- **Parameters:** 270M
- **Architecture:** LLaMA-style transformer (depthwise-scaled)
- **Precision:** fp16

> **Weights are NOT redistributed in this repo.** OpenELM ships under the [Apple Sample Code License](https://huggingface.co/apple/OpenELM-270M/blob/main/LICENSE), which restricts redistribution. Use the per-model fetch script below — it pulls the upstream safetensors from `apple/OpenELM-270M` directly under Apple's terms and converts locally.

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch + convert weights from Apple's HF repo (requires network)
../../../scripts/fetch_openelm_weights.sh

# 2. Build (release, fp16)
NNOPT_DTYPE=fp16 ./scripts/build.sh --release

# 3. Deploy
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 4. Run
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "Once upon a time" 64
```

## Performance

Razr 2020 / Adreno 618, fp16, greedy (`temperature=0, seed=42`), 32-token generation, 5-run warm median measured 2026-05-06. (No `reference/test_input_ids.bin` shipped — uses live encoder.)

| | Decode tok/s | TTFT (s) |
|---|---:|---:|
| Measured today | **4.47** | **2.13** |
| min / max across 5 runs | 4.40 / 4.50 | — |
| Roofline ceiling (10 GB/s) | 19.3 | — |

Per-token weight footprint: ~518 MB. Full optimization log in [BENCHMARK.md](./BENCHMARK.md).

## Layout

```
.
├── BENCHMARK.md
├── CMakeLists.txt
├── kernels/
├── reference/                    # empty in this port
├── scripts/
├── src/
└── weights/                      # populated by fetch_openelm_weights.sh
```

## Licensing

- **Code:** Apache 2.0 (this repo).
- **Weights:** Apple Sample Code License — see the [LICENSE](https://huggingface.co/apple/OpenELM-270M/blob/main/LICENSE) on the upstream model card. **Do not redistribute the converted binary.**
