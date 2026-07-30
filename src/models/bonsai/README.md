# Bonsai (Q1_0, 1-bit) on Adreno (Android)

Prism's Bonsai reasoning models ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship
Android, running at **1.125 bit/weight** (Q1_0) with **no dequantization** — the packed 1-bit
weights are streamed straight into the GEMV kernels. Verified TOKEN-EXACT against the
llama.cpp oracle on Motorola Razr 2020 (Adreno 620 / Snapdragon 765G).

- **Upstream:** [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf)
- **Architecture:** Qwen3-style transformer with grouped-query attention (GQA), reasoning-tuned
  (GGUF `general.architecture = qwen3`)
- **Precision:** Q1_0 weights (1.125 bit/weight, never dequantized); fp32 activations (the token-exact configuration — fp16 activations measured *slower*, see [BENCHMARK.md](./BENCHMARK.md))
- **Chat:** ChatML template built in — pass `chat` as the run mode

All sizes share **one** dim-generic runtime binary (`bonsai_inference` → `libbonsai.so` in Edgi):
every hyperparameter is read from the `.nnb` header, and the header is derived entirely from the
GGUF metadata block — so there is no per-size code anywhere in the port. The size a run uses is
decided purely by which `.nnb` is in `weights/`. The tokenizer is shared (all sizes use the same
151669-entry vocab).

| bundle      | params | layers × hidden × ffn | weights (Q1_0) | HF file          |
|-------------|--------|-----------------------|----------------|------------------|
| Bonsai-8B   | 8B     | 36 × 4096 × 12288     | 1.16 GB        | `bonsai8b.nnb`   |
| Bonsai-4B   | 4B     | 36 × 2560 × 9728      | 574 MB         | `bonsai4b.nnb`   |
| Bonsai-1.7B | 1.7B   | 28 × 2048 × 6144      | 242 MB         | `bonsai1.7b.nnb` |

### Bonsai-27B — different architecture, port in progress

**Bonsai-27B is not a fourth size, it is a second architecture.** Its GGUF declares
`general.architecture = qwen35` (inherited from Qwen3.6-27B): 48 of its 64 blocks are
Gated-DeltaNet linear-attention layers (`ssm_conv1d`, `ssm_a`, `ssm_dt`, `ssm_alpha/beta`,
`ssm_norm`, `ssm_out`) and only the 16 blocks at `L % 4 == 3` are attention layers — and even
those are *gated* attention (`attn_gate`) at head_dim 256 with 24 q / 4 kv heads and partial
mRoPE (`rope.dimension_count = 64` of 256). It also has its own tokenizer (248320 entries vs
151669).

`convert_to_nnb.py` **does** understand qwen35 and emits a complete header (layer kinds, GDN
dims, partial-RoPE config) so the layout can be developed against. The **runtime** refuses to
load it — there are no Gated-DeltaNet kernels yet, and executing it on the dense forward pass
would silently produce garbage. The plan, the reusable assets, and the measured memory ceiling
are in [QWEN35_PORT_PLAN.md](./QWEN35_PORT_PLAN.md).

## Converting a new size

Pre-converted `.nnb` bundles come from `fetch_weights.sh`. To convert a GGUF yourself:

```bash
# check a size is supported before spending the I/O
./scripts/convert_gguf.sh --dry-run weights/Bonsai-1.7B-Q1_0.gguf

# convert (writes the .nnb + a per-size tokenizer.json next to the GGUF)
./scripts/convert_gguf.sh weights/Bonsai-1.7B-Q1_0.gguf weights/bonsai1.7b.nnb
```

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch the converted weights (Q1_0 .nnb + tokenizer) from HuggingFace
../../../scripts/fetch_weights.sh bonsai

# 2. Build (release; BONSAI_STORAGE=fp32 is the default, token-exact config)
./scripts/build.sh --release

# 3. Deploy (pushes binary + kernels; the .nnb is pushed once). Default is 8B;
#    smaller bundles: BONSAI_NNB=bonsai4b.nnb (or bonsai1.7b.nnb) ./scripts/deploy_android.sh
./scripts/deploy_android.sh

# 4. Run (chat — reasoning-tuned)
./scripts/run_android.sh "Tell me a fun fact about foxes." 64 chat
```

## Performance

Razr 2020 / Adreno 620 / Snapdragon 765G, Q1_0, greedy, 64-token warm decode. 8B measured
2026-07-18, 1.7B measured 2026-07-22. 8B and 1.7B are both token-exact vs the llama.cpp oracle
(3/3 goldens each — `./scripts/check_goldens.sh`).

| | Decode tok/s | TTFT (s) | Peak mem (MB) | Weights |
|---|---:|---:|---:|---:|
| Bonsai-8B | **1.96** | 7.15 | 2946 | 1.16 GB |
| Bonsai-4B | **3.08** | 4.10 | 1828 | 574 MB |
| Bonsai-1.7B | **7.25** | — | — | 242 MB |

1.96 tok/s is the hard floor for a token-exact 1-bit 8B decode on this 2020 GPU — the decode is
**ALU-issue-bound on the 1-bit unpack**, not memory-bound (memory has ~5× headroom). The smaller
sizes share the same dim-generic runtime and scale almost exactly with parameter count, which is
what an ALU-issue-bound decode predicts. The full optimization ladder and the closed dead ends are
in [BENCHMARK.md](./BENCHMARK.md).

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
