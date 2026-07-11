# FunctionGemma-270M-IT on Adreno (Android)

FunctionGemma-270M-IT (Gemma 3 text architecture, function-calling instruction-tuned) ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 620 / Snapdragon 765G).

- **Upstream:** FunctionGemma-270M-IT — Gemma 3 text decoder, function-calling/instruction-tuned
- **Parameters:** 270M
- **Architecture:** Gemma 3 dense decoder — 18 layers, hidden 640, FFN 2048, 4 query heads / 1 KV head (MQA), head_dim 256, `gelu_pytorch_tanh` MLP, RMSNorm, 5:1 sliding/full attention (layers 5, 11, 17 are full attention; the rest use a 512-token sliding window), RoPE θ=1e6, tied input embedding / `lm_head` (262,144-way vocab)
- **Precision:** fp16

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch the converted weights (model.fp16.bin + meta + tokenizer_vocab.bin)
../../../scripts/fetch_weights.sh functiongemma-270m-it

# 2. Build (release, fp16)
NNOPT_DTYPE=fp16 ./scripts/build.sh --release

# 3. Deploy
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 4a. Run (deterministic, bit-exact reference path — recommended for verification)
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "x" 32 \
    --token-ids reference/test_input_ids.bin --temperature 0 --seed 42
# → "10th grade class, Mrs. Anya Sharma, to help her students with their math homework. ..."

# 4b. Run (raw-text completion)
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "The teacher worked at the" 32 --temperature 0
```

**Tokenizer note.** The deterministic `--token-ids` path feeds the exact reference input IDs and is bit-exact against the PyTorch reference — use it to verify correctness and reproduce the benchmark. The raw-text path uses a lightweight on-device tokenizer (BOS-prepended greedy-longest-match, not full SentencePiece unigram), so for prompts where word boundaries matter it is an approximation — completions are coherent but may differ from a reference HF tokenization.

## Performance

Razr 2020 / Adreno 620 / Snapdragon 765G, fp16, greedy (`--temperature 0`), 32-token generation, 3-run warm median (deterministic `--token-ids` path), measured 2026-06-25.

| | Decode tok/s | TTFT (s) | Peak CPU mem (MB) |
|---|---:|---:|---:|
| Measured today | **8.89** | **3.12** | **1743** |
| Roofline ceiling (9 GB/s measured fp16 streaming) | ~17 | — | — |

Per-token weight footprint: ~535 MB (of which the tied embedding / `lm_head` is 335 MB). Greedy tokens are bit-identical to the golden reference (verified on-device via the GPU-argmax cross-check). Full optimization log in [BENCHMARK.md](./BENCHMARK.md).

### Optimization summary (decode 1.06 → 8.89 tok/s, 8.4×)

The release baseline was 1.06 tok/s — ~94% host/sync overhead, not GPU compute. The wins, in order:

1. **Coalesced workgroup-per-output reduction GEMV (fp32 accum)** replacing CLBlast HGemm — adjacent lanes read adjacent weights, and the ~3.6 ms/call CLBlast host cost disappears. The single biggest lever: **1.12 → 4.71 tok/s.**
2. **Parallelised RMSNorm** (same workgroup-reduction pattern; the M=1 path had been running a 640-element reduction on a single work-item, ~515 µs → ~25 µs/call): **4.71 → 5.79 tok/s.**
3. **Fused multi-output GEMV** for q/k/v and gate/up projections (one dispatch instead of three): **→ ~7.0 tok/s.**
4. **Per-token buffer pool + greedy GPU-argmax fast-path** (kills the 262,144-element logits readback every token): **→ ~7.5 tok/s.**
5. **Transposed `lm_head` GEMV** (one-thread-per-output + local-x cache, lws=256) — the 335 MB tied embedding now streams at the device's ~9 GB/s fp16 ceiling instead of ~4 GB/s. A tiled transpose keeps the one-time weight transform off the critical path so TTFT stays ~3.1 s: **→ ~8.9 tok/s.**

The fp16 ceiling on this device is ~8.9 tok/s here: `lm_head` is at the measured 9 GB/s bandwidth ceiling, and the small-N projections are at the device's small-matvec / dispatch-overhead wall (proven four ways — split-K, warp-sync, WG sweep, and packed int8 all tie at ~8.8). Packed int8/int4 was built and measured but rejected: byte loads don't coalesce on Adreno (no `lm_head` speedup) and quantizing the 262,144-way tied embedding flips the near-tie argmax (garbage tokens), while the projections are overhead-bound so halving their bytes doesn't help. The remaining lever is dispatch-count reduction (megakernel fusion of whole MLP/attention blocks), not quantization.

## Layout

```
.
├── BENCHMARK.md
├── CMakeLists.txt
├── kernels/          # gemma3_ops.cl (all compute kernels) + utils.cl
├── reference/        # PyTorch reference: input IDs, expected tokens/text, config
├── scripts/          # build.sh, deploy_android.sh, run_android.sh, cos_check.py
├── src/              # main + graph-mode model/backbone + ops/ + tokenizer/sampler
└── weights/          # populated by ../../../scripts/fetch_weights.sh (gitignored)
```

## Licensing

- **Code:** Apache 2.0 (this repo).
- **Weights:** Gemma Terms of Use — see Google's [Gemma license](https://ai.google.dev/gemma/terms). FunctionGemma is built on the Gemma 3 base.
