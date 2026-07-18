# Bonsai (Q1_0, 1-bit) on Adreno (Android)

Prism's Bonsai reasoning models ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship
Android, running at **1.125 bit/weight** (Q1_0) with **no dequantization** — the packed 1-bit
weights are streamed straight into the GEMV kernels. Verified TOKEN-EXACT against the
llama.cpp oracle on Motorola Razr 2020 (Adreno 620 / Snapdragon 765G).

- **Upstream:** [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf)
- **Architecture:** Qwen3-style transformer with grouped-query attention (GQA), reasoning-tuned
- **Precision:** Q1_0 weights (1.125 bit/weight, never dequantized); fp32 activations (the token-exact configuration — fp16 activations measured *slower*, see [BENCHMARK.md](./BENCHMARK.md))
- **Chat:** ChatML template built in — pass `chat` as the run mode

Both sizes share **one** dim-generic runtime binary (`bonsai_inference` → `libbonsai.so` in Edgi):
every hyperparameter is read from the `.nnb` header, so there is no per-size code — the size a run
uses is decided purely by which `.nnb` is in `weights/`.

| bundle    | params | weights (Q1_0) | HF file        |
|-----------|--------|----------------|----------------|
| Bonsai-8B | 8B     | 1.16 GB        | `bonsai8b.nnb` |
| Bonsai-4B | 4B     | 574 MB         | `bonsai4b.nnb` |

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch the converted weights (Q1_0 .nnb + tokenizer) from HuggingFace
../../../scripts/fetch_weights.sh bonsai

# 2. Build (release; BONSAI_STORAGE=fp32 is the default, token-exact config)
./scripts/build.sh --release

# 3. Deploy (pushes binary + kernels; the .nnb is pushed once). Default is 8B;
#    for the 4B bundle: BONSAI_NNB=bonsai4b.nnb ./scripts/deploy_android.sh
./scripts/deploy_android.sh

# 4. Run (chat — reasoning-tuned)
./scripts/run_android.sh "Tell me a fun fact about foxes." 64 chat
```

## Performance

Razr 2020 / Adreno 620 / Snapdragon 765G, Q1_0, greedy (`chat`/`gen`), 64-token warm decode.
Token-exact vs the llama.cpp oracle (3/3 goldens).

| | Decode tok/s | Load (warm) | Weights |
|---|---:|---:|---:|
| Bonsai-8B | **1.95** | ~3.5 s | 1.16 GB |
| Bonsai-4B | _pending on-device benchmark_ | — | 574 MB |

1.95 tok/s is the hard floor for a token-exact 1-bit 8B decode on this 2020 GPU — the decode is
**ALU-issue-bound on the 1-bit unpack**, not memory-bound (memory has ~5× headroom). The full
optimization ladder and the closed dead ends are in [BENCHMARK.md](./BENCHMARK.md).

## Layout

```
.
├── BENCHMARK.md
├── CMakeLists.txt
├── Q1_0_LAYOUT.md
├── kernels/
├── reference/
├── scripts/
├── src/
└── weights/
```

## Licensing

- **Code:** Apache 2.0 (this repo).
- **Weights:** Apache 2.0 — see the [model card](https://huggingface.co/prism-ml/Bonsai-8B-gguf).
