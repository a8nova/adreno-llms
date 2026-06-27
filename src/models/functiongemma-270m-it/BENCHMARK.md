# Benchmark log — FunctionGemma-270M-IT on Razr 2020 (Adreno 620, fp16)

## Benchmark protocol

**One-time setup (after any source change):**
```bash
cd <repo-root>/src/models/functiongemma-270m-it
NNOPT_DTYPE=fp16 ./scripts/build.sh --release
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh
```
- `--release` ⇒ CMake `Release` (`-O3`), `-DNNOPT_DEBUG=` unset (debug macros stripped).
- `NNOPT_DTYPE=fp16` builds `build/fp16/functiongemma_270m_it_inference_fp16` and pulls `weights/model.fp16.bin`.

**Per-run command (canonical, bit-exact path):**
```bash
NNOPT_DTYPE=fp16 NNOPT_DEBUG_LAYERS=0 ./scripts/run_android.sh \
    "x" 32 --token-ids reference/test_input_ids.bin --temperature 0 --seed 42
```
- `--token-ids reference/test_input_ids.bin` feeds the exact reference input IDs (`[2, 818, 9800, 5934, 657, 506, 236743]`, BOS-prefixed), so the comparison is against the PyTorch reference's own input — tokenizer drift can't masquerade as model error.
- `--temperature 0` ⇒ greedy decode (deterministic).
- `NNOPT_DEBUG_LAYERS=0` suppresses per-step stderr (each `fprintf` is a syscall, visible at ~9 tok/s).
- `max_tokens=32` ⇒ generate 32 new tokens after the 7-token prompt.

**Token IDs as canonical reference.** Greedy decode is deterministic at the kernel-output level. The on-device GPU-argmax is cross-checked against the fp16 logits readback (`NNOPT_ARGMAX_VERIFY`) — 0 mismatches across runs.

**Metric pulled per run:** `BENCHMARK decode_tokens_per_sec` = `(n_generated - 1) / decode_s` (excludes prefill). **Per-step ritual:** 3 consecutive warm runs, take the median.

## Hardware ceiling

**Snapdragon 765G / Adreno 620 / LPDDR4X**

- **Theoretical DRAM peak: ~17 GB/s.** At ~535 MB weights read/token that is ~31 ms/token = **~32 tok/s** — the hard physics bound at fp16.
- **Practical streaming-read ceiling: ~9 GB/s** — directly measured on this device via the `NNOPT_BWTEST=1` STREAM microbench (`gemma3_bw_test`), which reads the 335 MB tied embedding coalesced with minimal compute. At 535 MB/token → 59 ms/token = **~17 tok/s**. This is the honest fp16 target ceiling.
- Earlier sessions quoted a "~7.2 tok/s practical ceiling" — that was **wrong** (it mistook the kernel's then-current throughput for the device's). The STREAM microbench corrected it: the device streams fp16 at ~9 GB/s; a perfectly-tuned kernel doing nothing but streaming weights should land near 17 tok/s.

**fp16 weight footprint** (`HIDDEN=640, INTER=2048, NUM_LAYERS=18, NUM_Q_HEADS=4, NUM_KV_HEADS=1, HEAD_DIM=256, VOCAB=262144, TIE_WORD_EMBEDDINGS=true`):

| Component | Per token (decode) |
|---|---|
| q/k/v/o projections × 18 layers | ~40 MB |
| gate/up/down × 18 layers | ~140 MB |
| norms (RMSNorm) | trivial |
| `lm_head` (tied embedding, 262144 × 640 fp16) | **335 MB** |
| **Total weight bytes / token** | **~535 MB** |

`lm_head` is 63% of the per-token read. KV-cache reads at 32-token context are trivial vs weights.

## Step 0 — Release-build baseline (3-run median)

| Build | prefill tok/s | decode tok/s | TTFT | total (7+7 tok) |
|---|---|---|---|---|
| Debug (layer-checks + per-op clFinish) | 1.27 | 0.76 | 5.86 s | 13.78 s |
| **Release (-O3, no layer-checks)** | **2.02** | **1.06** | **3.60 s** | **9.29 s** |

**Headline finding:** total GPU kernel time was ~560 ms but wall-clock inference was ~9290 ms — the GPU did ~6% of the work; ~94% was host overhead + per-op synchronization. Per decode token: ~40 ms GPU vs ~910 ms host/sync. So the biggest early wins were structural (kill overhead), not faster kernels.

Two structural facts drove everything: (1) decode was overhead-bound, *flat* vs sequence length (1.06 @ 13 tok ≈ 1.01 @ 27 tok) — proving per-op overhead, not O(N²); (2) every op `clCreateBuffer`'d and released its output (thousands of driver allocations per inference).

## Step 1 — Coalesced workgroup-per-output reduction GEMV (MEASURED — BIG WIN)

Replaced CLBlast HGemm at every M=1 projection and `lm_head` with a custom **workgroup-per-output reduction** kernel: threads in a WG=64 reduce over K, so adjacent lanes read adjacent `W[n*K + k]` (coalesced), with fp32 accumulation to preserve the near-tie precision of the 262,144-way argmax. A naïve one-thread-per-output kernel was ~50× *slower* (uncoalesced: adjacent lanes read K apart). This also removed CLBlast's ~3.6 ms/call host cost.

**decode 1.12 → 4.71 tok/s.**

## Step 2 — Parallelised RMSNorm (MEASURED)

Same workgroup-reduction pattern applied to RMSNorm. At M=1 the scaffold kernel ran a 640-element reduction on a *single* work-item (~515 µs/call); the cooperative version hits ~25 µs/call.

**decode 4.71 → 5.79 tok/s.** (Session-1 total: 1.06 → 5.79 = 5.5×, TTFT 3.6 → 2.2 s.)

## Step 3 — Fused multi-output GEMV (MEASURED)

A fused reduction GEMV writes multiple outputs per dispatch, wired into `Gemma3Attention` (q/k/v) and `Gemma3MLP` (gate/up): 3 dispatches → 1 per group. Reduction-width sweep confirmed WG=64 is optimal (WG ∈ {16,32,64,128} → 4.5 / 6.6 / **6.7** / 4.8 tok/s).

**decode 6.74 → ~7.0 tok/s.**

## Step 4 — Per-token buffer pool + greedy GPU-argmax (MEASURED)

1. **Size-keyed buffer pool** (`NNOPT_NO_POOL=1` to disable) recycles per-op output buffers instead of `clCreateBuffer`/`clReleaseMemObject` every call.
2. **Greedy GPU-argmax fast-path** (default for `temperature ≤ 0`): the forward computes argmax on-device and returns one int32, skipping the 262,144-element logits readback + host convert + host `max_element` every token. Cross-checked against the fp16 readback (`NNOPT_ARGMAX_VERIFY`) — 0 mismatches.

**decode ~7.0 → ~7.5 tok/s.**

## Step 5 — Transposed `lm_head` GEMV (MEASURED — closes the bandwidth gap)

The reduction GEMV streamed the 335 MB tied embedding at ~4 GB/s — 2.2× below the device's measured 9 GB/s. A **transposed [K,N] one-thread-per-output kernel with a local-x cache and lws=256** gets `lm_head` to the ~9 GB/s ceiling. lws is decisive: 64 → 3.5, 128 → 5.8, **256 → 9.0 tok/s**. (Transposed only wins for huge N — the small-N projections stay on the WG=64 reduction; transposing them under-occupies.)

**TTFT trap + fix:** the 335 MB transpose initially ran on the first forward → TTFT 2.65 → 8.4 s (worse total at 16 tokens). Fixed with a tiled (16×16 local, read+write coalesced) transpose: ~6 s → ~0.5 s, TTFT back to ~3.1 s.

**decode ~7.45 → ~8.9 tok/s.**

## What did NOT help (measured, reverted / gated off)

- **Split-K transposed projections** — ties the reduction (~8.8). The projections are at the device's small-N matvec bandwidth wall: a 1–5 MB matvec can't reach the 9 GB/s the 335 MB `lm_head` sustains. Left OFF (costs transposed-weight memory for no gain).
- **Warp-synchronous reduction** (volatile local, no barriers) — correct but no faster ⇒ barriers were never the bottleneck.
- **Subgroup built-ins** (`sub_group_reduce_add`, shuffle) — compile error `-11` "OpenCL 2.0 built-in is not supported" despite the extension being advertised. Dead on this driver; gated behind an undefined macro.
- **Packed int8 / int4** — built, validated, rejected. `lm_head` int8 gave no speedup (Adreno doesn't coalesce sub-32-bit loads, so int8 streams the same effective bandwidth) **and** quantizing the 262,144-way tied embedding flips the near-tie argmax → garbage ("`<b>Primary School</b>…`"); it must stay fp16. Projection packed-int8 (real `uint32` packing) is bit-correct but no faster — the projections are dispatch/overhead-bound, not weight-byte-bound, so halving the bytes doesn't help. Code removed after measurement.

## Final fp16 state

3 warm runs, canonical `--token-ids` path, measured 2026-06-25:

| Run | decode tok/s | TTFT (s) | prefill tok/s | peak CPU mem (MB) |
|---|---:|---:|---:|---:|
| 1 | 8.878 | 3.08 | 2.39 | 1742.6 |
| 2 | 8.893 | 3.13 | 2.35 | 1742.8 |
| 3 | 8.895 | 3.12 | 2.36 | 1742.2 |
| **median** | **8.89** | **3.12** | **2.36** | **1742.6** |

**Journey: 1.06 → 8.89 tok/s (8.4×).** Greedy tokens bit-identical to the golden reference (GPU-argmax verified). Generated text:

> The teacher worked at the **10th grade class, Mrs. Anya Sharma, to help her students with their math homework. Mrs. Sharma was known for her patience, understanding, and**

The fp16 ceiling is ~8.9 tok/s here: `lm_head` is at the 9 GB/s bandwidth ceiling (transposed), the projections are at the small-N dispatch/overhead wall (proven four ways — split-K, warp-sync, WG sweep, packed int8 all tie at ~8.8). The remaining lever is **dispatch-count reduction** — a megakernel fusing the whole MLP block (gate+up+gelu+mul+down) and/or attention block into single dispatches, removing ~10 of the ~15 launches/layer. That targets the measured bottleneck (launch count), unlike quantization. Deferred (larger refactor).
