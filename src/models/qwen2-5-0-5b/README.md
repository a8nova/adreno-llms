# Qwen2.5-0.5B on Adreno (Android)

Alibaba's Qwen2.5-0.5B (LLaMA-style + GQA) ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 618).

- **Upstream:** [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- **Parameters:** 500M
- **Architecture:** LLaMA-style transformer with grouped-query attention (GQA)
- **Precision:** fp16

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch the converted weights
../../../scripts/fetch_weights.sh qwen2-5-0-5b

# 2. Build (release, fp16)
NNOPT_DTYPE=fp16 ./scripts/build.sh --release

# 3. Deploy
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 4. Run
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "Once upon a time" 64
```

## Performance

Razr 2020 / Adreno 618, fp16, greedy (`temperature=0, seed=42`), 32-token generation, canonical token IDs, 5-run warm median measured 2026-05-06.

| | Decode tok/s | TTFT (s) |
|---|---:|---:|
| Measured today | **8.45** | **2.59** |
| min / max across 5 runs | 8.27 / 8.99 | — |
| Roofline ceiling (10 GB/s) | 10.6 | — |

Per-token weight footprint: ~942 MB. Reaches ~80% of the memory-bandwidth ceiling — the highest utilization across the models in this repo. Full optimization log in [BENCHMARK.md](./BENCHMARK.md).

### Steps 17 + 18: chained cl_mem decode + fused QKV (+21.2% total)

**Step 17 — chained cl_mem decode + ping-pong async readback** (ported from OpenELM-270M's +1.38× win). The decode loop's two-way host roundtrip (upload next token id + sync readback of int32 argmax) was serializing the GPU. Now: the embedding reads its token id directly from `argmax_out_buf_` via a single `clEnqueueCopyBuffer`, the GPU pipeline runs straight through to the next argmax, and the host readback is `CL_FALSE` with a 2-slot ping-pong so iteration N's int32 drains while iteration N+1 enqueues. **+19.3%** alone. Toggle off via `NNOPT_NO_CHAIN=1`.

**Step 18 — fused QKV projection.** Stack q/k/v weights+biases into a single `[1152, 896]` matrix at `set_weights()` time, dispatch one image2d GEMV with N=1152 (routes through the existing `gemv_m1_k896_no4_img` — no new kernel) plus one bias_add of [1152]. Slice the output into q/k/v via cached sub-buffers at byte offsets `{0, Q_DIM*2, (Q_DIM+KV_DIM)*2}` — all 128-byte aligned for Qwen's H=896, KV_DIM=128. Cuts 96 dispatches/token (2 fewer GEMVs + 2 fewer bias_adds × 24 layers). **+1.6% on top of Step 17.**

**Same-session A/B (canonical greedy, 32-tok decode):**

| Path | tok/s |
|------|------:|
| baseline (forward_greedy, no chain, no fuse) | 8.59 |
| chained decode (Step 17) | 10.25 |
| **chained + fused QKV (Steps 17 + 18, default)** | **10.41** |

**Cumulative: +21.2%**, **69.9% of the 14.0 GB/s practical fp16 ceiling.** Greedy tokens bit-identical to the golden reference. Wall-vs-GPU host gap shrunk from ~38 ms/token → 4.2 ms/token. Per-kernel GEMVs are now near per-kernel saturation; further fp16 wins must come from kernel fusion (norm+matmul) or texture-cache wider reads. Past 70% utilization at fp16, int8 (deferred) is the next physical lever.

### int8 W8A16 attempt — reverted

Tried per-channel symmetric int8 weights (W8A16) for `lm_head` + `down_proj` + `gate_proj` + `up_proj` via packed `CL_RGBA + CL_SIGNED_INT8` image2d. The same-session A/B did show a relative gain (+15% over the degraded baseline measured that day), but the absolute number never matched the 8.45 tok/s fp16 result in this README, so it wasn't shipped. Lesson: on Adreno, the texture engine matters more than the dtype — `fp16 + image2d` beats `int8 + buffer`, and only `int8 + image2d` (with tiling for `lm_head`'s N=151936) compounds the bytes saving with the texture-cache amplification. Even then, the per-dispatch CPU overhead and the small-N projections (k/v at N=128) eat the win at this model size on Adreno 618.

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
- **Weights:** Apache 2.0 — see the [model card](https://huggingface.co/Qwen/Qwen2.5-0.5B).
