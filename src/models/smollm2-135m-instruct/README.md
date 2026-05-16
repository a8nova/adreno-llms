# SmolLM2-135M-Instruct on Adreno (Android)

HuggingFace's smallest instruct-tuned LLM (LLaMA-style + GQA) ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 620 / Snapdragon 765G). Ships **fp16 and int8** weight variants.

- **Upstream:** [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- **Parameters:** 135M
- **Architecture:** LLaMA-style transformer with grouped-query attention (GQA), instruct-tuned
- **Precision:** fp16

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch the converted fp16 weights
../../../scripts/fetch_weights.sh smollm2-135m-instruct

# 2. (Optional) Generate the int8 variant from the fp16 bundle
python3 scripts/quantize_weights.py --emit-lm-head-int8     # → weights/model.int8.bin

# 3. Build (single release binary, handles both variants at runtime)
NNOPT_DTYPE=fp16 ./scripts/build.sh --release

# 4. Deploy binary + every weight bundle that exists locally
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 5a. Run fp16 — instruct mode (recommended; wraps prompt with the SmolLM2 chat template)
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "What is the capital of France?" 96 --chat

# 5b. Run int8 — same prompt, picks weights/model.int8.bin via NNOPT_QUANT
NNOPT_DTYPE=fp16 NNOPT_QUANT=int8 ./scripts/run_android.sh "What is the capital of France?" 96 --chat

# Or raw completion mode (no template — useful for benchmark parity but produces
# rambling output on most prompts since this is an instruct-tuned model):
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "Once upon a time" 64
```

The `--chat` flag wraps the prompt as `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n` and stops generation at `<|im_end|>`. Without it the model still works but treats the prompt as raw completion, which is rarely what you want for an instruct model.

## Performance

Razr 2020 / Adreno 620 / Snapdragon 765G, greedy (`--temperature 0 --chat`), 32-token generation, 3-run warm median measured 2026-05-16.

| Variant | Decode tok/s | TTFT (s) | Peak CPU mem (MB) | Run command |
|---|---:|---:|---:|---|
| **fp16** | **24.40** | 1.67 | 923 | `NNOPT_DTYPE=fp16 ./scripts/run_android.sh "..." 32 --temperature 0 --chat` |
| **int8** | 24.21 | **0.91** | **670** | `NNOPT_DTYPE=fp16 NNOPT_QUANT=int8 ./scripts/run_android.sh "..." 32 --temperature 0 --chat` |

int8 ties fp16 on decode tok/s (SmolLM2's fp16 path is already at ~63% of texture ceiling) but saves **27% peak memory** and roughly halves TTFT under int8 prefill-via-decode-loop. Per-token fp16 weight footprint: ~275 MB. Full optimization log + per-kernel timings in [BENCHMARK.md](./BENCHMARK.md).

### Regenerate the int8 bundle locally

```bash
python3 scripts/quantize_weights.py --emit-lm-head-int8
# →  weights/model.int8.bin       (~183 MB)
# →  weights/model.int8.meta.json
```

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
