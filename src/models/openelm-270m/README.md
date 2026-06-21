# OpenELM-270M-Instruct on Adreno (Android)

Apple's OpenELM-270M-Instruct (LLaMA-style efficient transformer, instruction-tuned) ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 620 / Snapdragon 765G).

- **Upstream:** [apple/OpenELM-270M-Instruct](https://huggingface.co/apple/OpenELM-270M-Instruct)
- **Parameters:** 270M
- **Architecture:** LLaMA-style transformer (depthwise-scaled), instruction-tuned
- **Precision:** fp16

> **Weights are NOT redistributed in this repo.** OpenELM ships under the [Apple Sample Code License](https://huggingface.co/apple/OpenELM-270M-Instruct/blob/main/LICENSE), which restricts redistribution. Use the per-model fetch script below — it pulls the upstream safetensors from `apple/OpenELM-270M-Instruct` directly under Apple's terms and converts locally (the tokenizer companions come from the project repo).
>
> **Chat:** OpenELM-270M-Instruct ships no official chat template (Apple drives it with plain prompts), so `--chat` feeds the prompt as-is (BOS + text, optional `--system "<text>"` prefix) rather than inventing a template the 270M model was never tuned on. The flag exists so the app can call every model uniformly.

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

Razr 2020 / Adreno 620 / Snapdragon 765G, fp16, greedy (`--temperature 0`), 32-token generation, 3-run warm median measured 2026-05-16. (No `reference/test_input_ids.bin` shipped — uses live encoder.)

| | Decode tok/s | TTFT (s) | Peak CPU mem (MB) |
|---|---:|---:|---:|
| Measured today | **14.65** | **2.00** | **1371** |
| Roofline ceiling (10 GB/s) | 19.3 | — | — |

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
