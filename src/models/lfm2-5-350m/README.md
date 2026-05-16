# LFM2.5-350M-Base on Adreno (Android)

Liquid AI's hybrid conv+attention foundation model, ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android devices. Verified on Motorola Razr 2020 (Adreno 620 / Snapdragon 765G) and Samsung Galaxy Tab A9+ (Adreno 619 v2 / Snapdragon 695).

- **Upstream:** [LiquidAI/LFM2.5-350M-Base](https://huggingface.co/LiquidAI/LFM2.5-350M-Base)
- **Parameters:** 350M
- **Architecture:** Hybrid LFM2 (interleaved short conv + grouped attention, 16 layers, GQA)
- **Precisions:** fp16, int8 (per-row symmetric), Q4 (block-32 symmetric, 4-bit)

## Choose a variant

| Variant | Decode tok/s — Razr 2020 / Adreno 620 | Decode tok/s — Tab A9+ / Adreno 619 v2 | Weight file | Peak GPU mem | Quality |
|---|---:|---:|---:|---:|---|
| **fp16** | **11.43** | ~10.0 (chained) / 6.70 (sampler=top_k50) | 676 MB | ~1.35 GB | Reference accuracy |
| **int8** | **13.67** | 10.85 | 467 MB | ~1.0 GB | ~0.4% per-row quant noise — indistinguishable in practice |
| **Q4** | **14.54** | 13.85 | 333 MB | ~720 MB | ~7% per-block quant noise — usable for greedy, sampling-friendly tasks |

All numbers are greedy (`--temperature 0`), 32-token generation, 3–5 run median (`Hello, I am a language model`).

**Recommendation:**
- **Razr 2020 / Adreno 620+**: pick whichever variant you want. int8 and Q4 nearly tied (+6%) — int8 has cleaner accuracy. Q4 wins on memory.
- **Adreno 619 v2 / lower-tier**: Q4 is meaningfully faster (~30%) because memory bandwidth is the bottleneck there.

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch weights from HuggingFace. Pick the variant(s) you want:
../../../scripts/fetch_weights.sh lfm2-5-350m                # fp16 only (676 MB)
../../../scripts/fetch_weights.sh lfm2-5-350m --quant int8   # fp16 + int8 (1.14 GB)
../../../scripts/fetch_weights.sh lfm2-5-350m --quant q4     # fp16 + Q4   (994 MB)
# (Each fetch is incremental — re-running with a different --quant adds the
# missing files without re-downloading the fp16 bundle.)

# 2. Build (single Release binary, handles all variants at runtime)
NNOPT_DTYPE=fp16 ./scripts/build.sh --release

# 3. Deploy binary + every weight bundle that exists locally
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 4. Run — pick a variant via NNOPT_QUANT (unset = fp16)
NNOPT_DTYPE=fp16                  ./scripts/run_android.sh "Hello, I am a language model" 32 --temperature 0
NNOPT_DTYPE=fp16 NNOPT_QUANT=int8 ./scripts/run_android.sh "Hello, I am a language model" 32 --temperature 0
NNOPT_DTYPE=fp16 NNOPT_QUANT=q4   ./scripts/run_android.sh "Hello, I am a language model" 32 --temperature 0
```

Same binary handles all three precisions — pick via `NNOPT_QUANT` at run-time.

### Alternative: generate the quantized bundles locally

If you have the fp16 bundle and would rather not download the int8/Q4 files, regenerate them on your machine (~30s each, needs Python 3 + numpy):

```bash
python3 scripts/quantize_weights.py --emit-lm-head-int8     # → weights/model.int8.bin
python3 scripts/quantize_q4.py      --emit-lm-head-q4       # → weights/model.q4.bin
```

Both scripts read `weights/model.fp16.bin` and write the new bundle alongside.

Same binary, same deploy step — the runtime picks the weight file based on `NNOPT_QUANT` (`weights/model.fp16.bin` / `model.int8.bin` / `model.q4.bin`).

## What runs where

- All matrix-vector products, attention, conv, RMSNorm, residual adds, softmax, argmax — **GPU** (OpenCL kernels on Adreno).
- Tokenization (one-shot on prompt), sampling (only at `temperature > 0`; greedy uses an on-GPU argmax), and kernel orchestration — **CPU**.

For int8 and Q4, weights are stored quantized on disk and in GPU memory; the GEMV kernels read them as `image2d_t` bytes, convert to `float` on-the-fly, multiply against fp16 activations, accumulate in fp32, and store fp16 outputs. **CLBlast is bypassed at decode under quantization** — custom kernels in `kernels/gemv_m1_int8.cl` and `kernels/gemv_m1_q4.cl` handle the M=1 path; under `NNOPT_QUANT=int8|q4` prefill is also routed through the M=1 fast path one token at a time (CLBlast Hgemm can't ingest int8 weights at M>1).

## Verify it works

Quickest sanity check after deploy:

```bash
# Should print coherent English like "Hello, I am a language model. I am here to help..."
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "Hello, I am a language model" 32 --temperature 0
NNOPT_DTYPE=fp16 NNOPT_QUANT=int8 ./scripts/run_android.sh "Hello, I am a language model" 32 --temperature 0
NNOPT_DTYPE=fp16 NNOPT_QUANT=q4   ./scripts/run_android.sh "Hello, I am a language model" 32 --temperature 0
```

If you also want a per-kernel GPU-time breakdown (slows down runs ~10%):

```bash
NNOPT_DTYPE=fp16 NNOPT_QUANT=q4 NNOPT_KERNEL_PROFILE=1 ./scripts/run_android.sh "Hi" 32 --temperature 0
```

## Performance — Razr 2020 / Adreno 620 (full breakdown)

Greedy (`--temperature 0`), seed 42, prompt `"Hello, I am a language model"`, 32-token generation, 3-run warm median measured 2026-05-16.

| Variant | Decode tok/s | TTFT (s) | Peak CPU mem (MB) | Δ vs fp16 |
|---|---:|---:|---:|---:|
| fp16 | 11.43 | 2.21 | 1666 | baseline |
| int8 | 13.67 | 0.81 | 1015 | +19.6% |
| Q4   | **14.54** | **0.79** | **719** | **+27.2%** |

Full optimization log + bottleneck census in [BENCHMARK.md](./BENCHMARK.md).

## Re-generating quantized bundles

Both quantizers read `weights/model.fp16.bin` and write a fresh bundle alongside.

```bash
# int8 (per-row symmetric, fp16 scales)
python3 scripts/quantize_weights.py --emit-lm-head-int8
# →  weights/model.int8.bin       (467 MB)
# →  weights/model.int8.meta.json

# Q4 (block-32 symmetric, fp16 per-block scales, packed nibbles)
python3 scripts/quantize_q4.py --emit-lm-head-q4
# →  weights/model.q4.bin         (333 MB)
# →  weights/model.q4.meta.json
```

Flags:
- `--emit-lm-head-int8` / `--emit-lm-head-q4`: emit a separate `lm_head.weight` quantized alias of `model.embed_tokens.weight`, so the embedding-lookup path keeps fp16 (correctness) while the lm_head GEMV gets the quantization win.
- `--quantize-embed`: quantize `model.embed_tokens.weight` directly (saves more memory but may degrade embedding lookup quality). Off by default.

## Layout

```
.
├── BENCHMARK.md
├── CMakeLists.txt                       # CMake 3.18+, C++17, CLBlast 1.6.3
├── kernels/
│   ├── gemv_m1.cl                       # fp16 image-path GEMV variants
│   ├── gemv_m1_int8.cl                  # int8 image-path GEMV variants (NEW)
│   ├── gemv_m1_q4.cl                    # Q4  image-path GEMV variants (NEW)
│   └── …                                # attention.cl, mlp.cl, rmsnorm.cl, etc.
├── reference/                           # Python HF reference + canonical token IDs
├── scripts/
│   ├── build.sh / deploy_android.sh / run_android.sh
│   ├── quantize_weights.py              # NEW — fp16 → int8
│   └── quantize_q4.py                   # NEW — fp16 → Q4
├── src/                                 # C++ inference code
└── weights/                             # model.fp16.bin (pulled by fetch_weights.sh)
                                         # model.int8.bin / model.q4.bin (locally generated)
                                         # tokenizer_vocab.bin
```

## Licensing

- **Code:** Apache 2.0 (this repo).
- **Weights:** Liquid AI's open license — see the [model card](https://huggingface.co/LiquidAI/LFM2.5-350M-Base) for terms.
