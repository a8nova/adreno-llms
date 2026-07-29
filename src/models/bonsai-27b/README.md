# Bonsai-27B (Q1_0, 1-bit VLM) on Adreno (Android)

Prism's Bonsai-27B — a 27B **vision-language** reasoning model — ported to C++/OpenCL for
Adreno 8xx GPUs, running at **1.125 bit/weight** (Q1_0) with **no dequantization**: the packed
1-bit weights are streamed straight into the GEMV kernels. 3.6 GB resident, on a phone.

- **Upstream:** [prism-ml/Bonsai-27B-gguf](https://huggingface.co/prism-ml/Bonsai-27B-gguf)
- **Architecture:** `general.architecture = qwen35` (Qwen3.6-27B), **hybrid** — see below
- **Vision:** Qwen3-VL tower, 27 blocks, verified cosine **1.000000** against the real
  `transformers` `Qwen3VLVisionModel`
- **Precision:** Q1_0 weights, fp32 activations, fp32 lm_head store
- **Measured:** 8.0 tok/s decode, 9.7 tok/s prefill on an Adreno 840 — see [BENCHMARK.md](./BENCHMARK.md)

## Not a fourth Bonsai size — a second architecture

The [`bonsai`](../bonsai/) port covers 1.7B/4B/8B, which are all `qwen3`: dense transformers with
GQA. **27B is `qwen35`, and it is a hybrid stack.** Its 64 blocks alternate two kinds:

| kind | count | operator | per-block state |
|---|---|---|---|
| `L` | 48 | Gated DeltaNet (linear attention) | recurrent `[n_v · d_k · d_v]` + causal-conv window — **no KV cache** |
| `F` | 16 | gated full attention (GQA) | KV cache |

`meta.layer_types` is one char per block, and weights, state and dispatch all branch on it.
Both kinds then run the same SwiGLU MLP. This is why the `bonsai` runtime cannot load a 27B and
refuses it by name (`FATAL: this runtime is qwen35-only`).

That asymmetry has a consequence worth knowing before touching the decode loop: **restarting a
turn at position 0 is not enough to make turns independent.** The KV cache self-heals because
every position is rewritten before it is read, but the recurrent state and conv window are
accumulators with no position index. `DeviceModel::reset_state()` exists for exactly this, and
`scripts/check_vlm_session.sh` is the regression test.

| | |
|---|---|
| params | 27B (64 blocks, hidden 5120, ffn 17408, vocab 248320) |
| text weights | 3.6 GB Q1_0 (`bonsai27b.nnb`) |
| vision weights | 600 MB, Q8_0 + F16 + F32 (`bonsai27b-vision.nnb`) |
| context | 2048 |
| device floor | Adreno 8xx-class — the model is **resident-only**, needs ~4.8 GB of GPU-visible memory |

## Quickstart

```bash
# 1. Convert the GGUFs into weights/ (see below) — the repo-level
#    scripts/fetch_weights.sh does not carry this model yet

# 2. Build (fp32 activations is the default; BONSAI_VISION=1 adds the tower)
BONSAI_VISION=1 ./scripts/build.sh --release

# 3. Deploy (binary + kernels; the .nnb is pushed once)
./scripts/deploy_android.sh

# 4. Run
./scripts/run_android.sh
```

## Converting from GGUF

> The conversion and verification scripts (`convert_gguf.sh`, `convert_to_nnb.py`,
> `convert_mmproj.py`, `gguf_parse.py`, the oracles and the check scripts) land in a follow-up
> change and are not in this commit. What is here is the runtime: `build.sh`,
> `deploy_android.sh`, `run_android.sh`, the kernels and the engine.

Weights are repacked at conversion time into a K-major device layout (`bits[u][row]` +
`scales[u][row]`) — the same transform `upload_q1()` would otherwise do per block, per token.
Every hyperparameter in the `.nnb` header is derived from the GGUF metadata; nothing about the
27B is hardcoded in the runtime.

## Verifying a change

Three gates exist, landing with the scripts in the follow-up change above:

- **warm-session state** — the 27B needs ~4.8 GB and loads on 8xx-class hardware only, while every
  smaller Bonsai is `qwen3` with no recurrent state at all, so the hybrid path has no test that can
  run on a desk. A generator emits a ~3 MB structurally faithful `qwen35` (same tensor names, same
  device layout, same `L`/`F` interleave, random weights) and the gate asserts invariants that hold
  regardless of what the weights say: same input + same starting state ⇒ same output.
- **vision tower** — cosine against the real `transformers` `Qwen3VLVisionModel`, loaded with the
  converted weights. Currently 1.000000.
- **token-exactness vs the llama.cpp oracle** — the script exists but there are no
  `reference/golden_N.json` captures yet, so it SKIPs every prompt.

That last one is the largest outstanding gap: the vision tower and the session machinery are both
gated, **the text decode path is not.**

## Tuning

The kernel geometry (`Q1_WG`, `Q1_ROWS`, `GEMM_MT`, `SK_TARGET_GROUPS`, `SK_MAX_SPLITS`) is
compile-time in `src/model.h`, and every value there was measured on the device it targets.

**Measure on the Adreno 840 only.** This port shipped two constants measured on an Adreno 620 —
a 1-CU part that ranks these kernels differently — and both were wrong for the target. The
on-device measurement tooling that settled them was removed before release; see
[BENCHMARK.md](./BENCHMARK.md) for what it did and why it is worth rebuilding before the next
tuning round.

## Layout

```
.
├── BENCHMARK.md
├── CMakeLists.txt
├── kernels/               *.cl — q1_gemv is the decode path; vision_* is the tower
├── scripts/               build / deploy / run
├── src/
│   ├── layers/            common / full_attn / linear_attn
│   └── vision_model.*     the tower, built only with -DBONSAI_VISION
└── weights/               (gitignored — fetched or converted)
```

## Licensing

- **Code:** Apache 2.0 (this repo).
- **Weights:** see the [model card](https://huggingface.co/prism-ml/Bonsai-27B-gguf).
