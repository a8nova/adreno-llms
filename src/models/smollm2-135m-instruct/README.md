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

# 4. Run — instruct mode (recommended; wraps prompt with the SmolLM2 chat template)
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "What is the capital of France?" 96 --chat

# Or raw completion mode (no template — useful for benchmark parity but produces
# rambling output on most prompts since this is an instruct-tuned model):
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "Once upon a time" 64
```

The `--chat` flag wraps the prompt as `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n` and stops generation at `<|im_end|>`. Without it the model still works but treats the prompt as raw completion, which is rarely what you want for an instruct model.

## Performance

Razr 2020 / Adreno 618, fp16, greedy (`temperature=0, seed=42`), 32-token generation, canonical token IDs, 5-run warm median measured 2026-05-06.

| | Decode tok/s | TTFT (s) |
|---|---:|---:|
| Measured today | **23.65** | **1.53** |
| min / max across 5 runs | 22.75 / 23.95 | — |
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
