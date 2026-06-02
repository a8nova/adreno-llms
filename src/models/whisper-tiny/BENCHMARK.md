# Whisper-tiny Android OpenCL — Benchmark & Optimization Log

**Target:** Motorola Razr 2020 (Snapdragon, Adreno GPU) · OpenCL · fp16
**Model:** openai/whisper-tiny (4 encoder layers, 4 decoder layers, d=384, 1500 encoder seq)
**Mission:** achieve **faster-than-realtime (aggregate RTF < 1.0)** for end-to-end raw-audio → transcript on device.

`RTF = processing_time / audio_duration`. Lower is better; **RTF < 1.0 = faster than real time.**

## ✅ MISSION EXCEEDED — aggregate RTF **0.529** (was 3.297, **6.2× faster**), REPRODUCIBLE

**Session 2026-06-02 (all Release build, 3-run medians ±0.001, transcripts byte-exact 9/10):**

| Step | Optimization | Aggregate RTF | Notes |
|---|---|---:|---|
| — | Release baseline (the real milestone) | 0.975 | (Debug build inflates to ~1.36 — see gotcha) |
| A | Remove leftover **bare `clFinish` in `LayerNorm.cpp`** | **0.786** | ~13 norms/token, each drained the GPU; not gated by `NNOPT_DEBUG_SYNC` so it ran in Release too. −0.19 |
| B | **LDS-tile the mel convs** (`whisper_encoder_frontend.cl`) | **0.776** | weight row cached in `__local` per output channel; confirmed prefill is mostly host-bound, so small. −0.01 |
| C | **Custom M=1 matvec kernel** (`matvec.cl`) replacing CLBlast on the decode path | **0.529** | the big one. CLBlast `GemmStridedBatched` fixed overhead (~1.4-2.9ms/call) dominated M=1; a direct matvec for projections/FFN/lm_head cut proc 85→58s. EVERY clip now <1.0 except clip0 (JIT). Long clips ~0.40. −0.25 |

Steady-state (JIT excluded) **0.485**. Per-clip after C: clip0 1.306 (JIT), clip4 **0.398**, clip9 0.404, clip2 0.467, shortest (clip6/8) ~0.72.

**Remaining lever:** the **attention scores + weighted-sum** (`attn_scores_gemm_step` / `attn_wsum_gemm_step`) still go through CLBlast `GemmStridedBatched` at M=1 in decode (they are NOT pytorch_linear, so matvec didn't touch them) — now the dominant decode GPU cost. A custom **batched-matvec or fused decode-attention** kernel (Q·Kᵀ → softmax → probs·V for a single query row, per head, reading the KV cache) is the next step toward ~0.4 aggregate.

### Earlier milestone (Release baseline before this session) — aggregate RTF 0.975

Real-deployment benchmark (`scripts/run_bench_batch.sh`: all 10 clips in ONE process, so the per-process JIT compile is paid once, as a deployed model would). **Aggregate RTF ≈ 0.975 < 1.0**, steady-state (JIT excluded) **~0.92**, **9/10 transcripts byte-exact**. Journey to here: `3.297 → 1.388` (attention→GEMM) `→ 1.313` (softmax+KV cache) `→ 0.975` (amortize JIT via single-process batch). Steps A/B/C above took it from 0.975 → 0.529.

**Reproducibility (verified 2026-06-02):** 3 back-to-back measured runs (after 1 warm-up, settled): **0.975 / 0.973 / 0.982 → median 0.975 ± 0.005**, GPU flat at ~41 °C. This is stable and production-representative.

### ⚠️ MEASUREMENT GOTCHA — benchmark ONLY in a Release build

`scripts/build.sh` **defaults to `--debug`** (`-DNNOPT_DEBUG=1`). In debug, `NNOPT_DEBUG_SYNC` expands to `clFinish` and fires after **every** `pytorch_linear` (~24 GEMMs/token) — costing roughly **4× decode throughput** (see the comment in `utils.cpp` / `debug_utils.h`). Effect on the aggregate:

| build | aggregate RTF | why |
|---|---:|---|
| **Release** (`build.sh --release`) | **0.975** | `NNOPT_DEBUG_SYNC` = no-op; no per-GEMM sync |
| Debug (`build.sh`, the default) | ~1.36 | `clFinish` after all 24 GEMMs/token |

A near-**uniform ~1.4× on every clip** between two binaries of the same source is the signature of this (NOT thermal — temp is flat ~41 °C; NOT GPU DVFS). **Always deploy/benchmark the Release binary.** A 2026-06-02 session briefly mis-attributed the Debug slowdown to "GPU clock state" — it was the build type. The reliable protocol still holds: 1 warm-up → ≥3 runs → median + spread → log temp; and confirm the binary is Release.

Bottleneck breakdown below is build-independent in *shape*: **decode ≈ 79% of wall (launch-bound, ~60 M=1 launches/token)**, prefill ≈ 18% (mel convs largest).

| Clip | Dur | RTF | ok | | Clip | Dur | RTF | ok |
|---:|---:|---:|:--|-|---:|---:|---:|:--|
| 0 | 5.86 | 1.960 | exact | | 5 | 9.01 | **0.858** | exact |
| 1 | 4.82 | 1.313 | exact | | 6 | 5.64 | 1.275 | exact |
| 2 | 12.48 | **0.877** | exact | | 7 | 9.24 | **0.849** | exact |
| 3 | 9.90 | **0.957** | exact | | 8 | 5.12 | 1.250 | exact |
| 4 | 29.40 | **0.838** | exact | | 9 | 18.29 | **0.803** | ~drift |

clips=10 · audio 109.8s · proc 106.7s · **aggregate 0.972** · clip0 carries one-time JIT.

---

## 1. Baseline (pre-optimization)

End-to-end pipeline: raw waveform → on-device log-mel front-end → encoder → autoregressive decode with token streaming.
Already-landed correctness/perf work before this baseline:
- On-device C++ log-mel front-end (FFT + slaney filterbank + log), cosine 1.0 vs `WhisperProcessor`, ~0.15s.
- Release build, debug prints gated behind `NNOPT_DEBUG`.
- **Encoder-output cache** (encoder depends only on the fixed mel → run once per clip, not once per token): RTF 63 → ~4.6 on the first clip, a 13.7× speedup. **This is the starting point of the table below.**

### Baseline results — 10 LibriSpeech clips (max_tokens=128, fp16)

| Clip | Dur (s) | Proc (s) | RTF  | Transcript correct? |
|-----:|--------:|---------:|-----:|:--------------------|
| 0 | 5.86  | 27.15 | 4.637 | ✓ |
| 1 | 4.82  | 24.24 | 5.034 | ✓ |
| 2 | 12.49 | 36.12 | 2.893 | ✓ |
| 3 | 9.90  | 32.21 | 3.254 | ✓ |
| 4 | 29.40 | 90.07 | 3.063 | ✓ |
| 5 | 9.01  | 27.25 | 3.025 | ✓ |
| 6 | 5.64  | 25.54 | 4.528 | ✓ |
| 7 | 9.24  | 27.39 | 2.965 | ✓ |
| 8 | 5.12  | 24.07 | 4.700 | ✓ |
| 9 | 18.29 | 47.78 | 2.612 | ✓ |

**Aggregate RTF = 3.297** (109.8s audio / 361.8s proc) · mean per-clip RTF = 3.671 · min 2.61 · max 5.03
**Accuracy: 10/10 transcripts match whisper-tiny HF reference.**

### Reading the baseline

- **Strong inverse correlation between clip length and RTF.** Short clips (~5s) sit at RTF ~4.6–5.0; long clips (18–29s) drop to ~2.6–3.1.
- This is the signature of a **large fixed per-clip cost amortized over decode steps**: the one-time encoder pass (~16.7s) dominates short clips and gets diluted on long ones.
- **To reach RTF < 1.0 we need ≥ 3.3× end-to-end speedup.** The encoder is the first and largest target; decode per-step cost is the second.

---

## 2. Profiling

Tooling added for this analysis (all dormant unless `NNOPT_PROFILE=1`):
- **GEMM instrumentation** — `pytorch_linear`/`pytorch_conv1d` now pass a `cl_event` to CLBlast and label it; the profiler suffixes `_ctx` (M>1, encoder/prefill) vs `_step` (M=1, decode) so encoder-vs-decode GEMM cost separates in a single run. Attention projections are further split `selfattn`/`crossattn`.
- `run_android.sh` now forwards `NNOPT_PROFILE` to the device.

**First control test — `NNOPT_DEBUG_LAYERS`:** the baseline ran with the default `NNOPT_DEBUG_LAYERS=1` (per-layer host readbacks). Re-running clip 0 with `=0` gave RTF **4.678 vs 4.637** — no change. The layer-check reads only 256 sampled elements; it is **not** a bottleneck. Baseline RTF is honest compute.

### Per-kernel GPU profile — clip 0 (5.86s, fp16, 25 tokens generated)

```
label                          total_ms   %total   calls    avg_us
attn_scores                    13456.840   76.15%     204   65964.90
attn_softmax                    1556.289    8.81%     204    7628.87
attn_wsum                       1395.348    7.90%     204    6839.94
whisper_conv2_gelu (mel)         404.886    2.29%       1  404886.02
layernorm_forward                195.920    1.11%     334     586.59
whisper_conv1_gelu (mel)         163.222    0.92%       1  163222.02
gemm_crossattn_k_ctx              77.514    0.44%     100     775.14   <- recomputed every decode step
gemm_crossattn_v_ctx             77.379    0.44%     100     773.79   <- recomputed every decode step
... all other GEMMs               ~370      ~2.0%
TOTAL GPU kernel time:         17672.4 ms
```

### Key findings

1. **Attention is 92.9% of all GPU time** (`attn_scores` 76.2% + `attn_softmax` 8.8% + `attn_wsum` 7.9%). Everything else combined is ~7%.
2. **The GEMMs (q/k/v/o/fc1/fc2/lm_head) are ~2% total.** CLBlast is already efficient on this Adreno — matmuls are *not* the problem.
3. **`attn_scores` is essentially 100% encoder.** A `max_tokens=4` run shows 36 `attn_scores` calls but the 4 encoder calls (1500×1500×6heads×64) account for ~all the time (~2,700ms each); the 32 decode calls are negligible. Encoder attention = **93.7% of the 16.7s encoder pass**.
4. **The attention kernels are grossly inefficient.** Each encoder `attn_scores` call does ~1.7 GFLOP in ~2,700–3,400ms ⇒ **~0.5–0.6 GFLOP/s effective** — two orders of magnitude below what CLBlast achieves for the same Adreno on plain GEMM. Root cause (see `kernels/attn.cl`): one work-item per output element, each re-reading Q/K rows from **global** memory (no local-memory tiling, so K rows are re-fetched Tq=1500× over), **scalar `vload_half`** (no vectorization), and a flat 13.5M-work-item 1-D range with no workgroup locality.
5. **Decode is host/launch-overhead bound**, not GPU bound. 2.24 tok/s = ~446ms/token while per-token GPU work is tiny. Sources: per-token kernel-launch count, leftover **unconditional `clFinish`** in `GELUActivation.cpp:183` and `Embedding.cpp:192` ("early bring-up" syncs), per-token full-vocab logit readback, and per-op `clCreateBuffer`/`clReleaseMemObject` churn. Host/non-GPU time ≈ 9.7s of the 27.4s clip-0 run.
6. **Cross-attention K/V are recomputed every decode step** (`gemm_crossattn_k_ctx`/`v_ctx` = 100 calls = 4 layers × 25 steps) — they depend only on the fixed encoder output and should be projected once.

**Conclusion:** the path to RTF < 1.0 is (Tier 1) make attention FLOP-efficient — the encoder is 93% attention and attention runs at <1% of achievable throughput — then (Tier 2) remove decode per-token host overhead.

### Roofline anchor

- The same Adreno runs CLBlast `gemm_lm_head` ([1,384]×[384,51865]) and the q/k/v/o GEMMs at negligible cost. The attention score op is *also* a batched matmul (Q@Kᵀ per head). **The expected end-state is: attention costs about what an equivalent batched GEMM costs (~hundreds of ms, not ~16s).**

---

## 3. Qualcomm Adreno OpenCL optimization references

Principles below are from the **Qualcomm Snapdragon Mobile Platform OpenCL General Programming and Optimization Guide** and **"OpenCL Optimization & Best Practices for Qualcomm Adreno GPUs"**, mapped to our specific hotspots.

| # | Adreno guideline | Why it matters here | Our hotspot |
|---|------------------|---------------------|-------------|
| G1 | **Use vector data types** (`half8`/`float4`). Adreno ALUs are SIMD; scalar code wastes lanes and issues 1 load per element. | `attn_scores`/`attn_wsum` use scalar `vload_half` over D=64. Vectorize to `half8` → 8× fewer load instructions. | attn_scores, attn_wsum |
| G2 | **Reuse data through `__local` (LDS) memory.** Don't re-read the same global data across a workgroup. | Every `attn_scores` work-item re-reads a K row from global; tiling K/Q into LDS removes ~Tq× redundant traffic. | attn_scores |
| G3 | **Prefer images (`image2d_t`) for read-heavy 2-D data.** Adreno has a dedicated, high-bandwidth L1 texture cache. | Q/K/V and weight matrices read many times → texture path raises effective bandwidth. | attention, GEMM |
| G4 | **Pick local work size = multiple of the wave size** (Adreno wave 64/128) and size workgroups for occupancy. | The naive 1-D `attn_scores` range has no workgroup structure → poor scheduling/occupancy. | all attention |
| G5 | **Use a tuned GEMM (CLBlast) rather than hand-rolled matmul.** It already applies tiling + vectorization + image path. | Attention scores/weighted-sum ARE matmuls — route them through CLBlast batched/strided GEMM. | attn_scores, attn_wsum |
| G6 | **Use `half` storage + math** to halve bandwidth and double ALU throughput where precision allows. | Scores accumulate in fp32 (keep), but loads/stores and softmax can use half/`half_exp`. | softmax, attention |
| G7 | **Fuse kernels** to cut launch overhead and intermediate global round-trips. | Fuse `attn_scale_q` into scores; fuse softmax max/exp/sum; fuse bias into GEMM epilogue. | softmax, decode |
| G8 | **Minimize kernel-launch count** & **avoid blocking (`clFinish`) in hot loops** — let the in-order queue pipeline. | Decode is launch/sync bound; remove leftover `clFinish`, batch launches. | decode |
| G9 | **Workgroup-level reductions in LDS** instead of one serial thread per row. | `attn_softmax` is one thread per (h,tq) looping Tk=1500 three times — parallelize the reduction. | attn_softmax |
| G10 | **`native_*`/`half_*` transcendentals** (`native_exp`) for softmax where precision allows. | softmax `exp()` over 6×1500×1500 elements. | attn_softmax |
| G11 | **Avoid per-op buffer alloc/free; reuse scratch.** Allocation churn stalls and fragments. | Each op `clCreateBuffer`/`clReleaseMemObject` per token → preallocate a scratch arena. | decode |
| G12 | **Avoid redundant recompute of loop-invariant work.** | Cross-attn K/V depend only on encoder output — cache them. | decode |

---

## 4. Top-20 optimizations (ranked by expected impact)

Baseline: aggregate RTF **3.297** (clip-0 RTF 4.64; encoder 16.7s, decode 10.7s). Target **< 1.0**.
"Expected" speedups are estimates from the profile + roofline; "Achieved" filled as each lands. Tier-1 alone should cross RTF 1.0 on medium/long clips.

### Tier 1 — Attention throughput (targets the 92.9%)

| # | Optimization | Adreno ref | Targets | Expected |
|---|--------------|-----------|---------|----------|
| 1 | **Reformulate `attn_scores` (QKᵀ) as CLBlast per-head/strided-batched GEMM** (`Transpose::kYes`), replacing the naive kernel. | G5,G2,G1 | attn_scores 76% | encoder attention scores **~20–40×**; ~13,400ms → ~400–700ms |
| 2 | **Reformulate `attn_wsum` (probs@V) as CLBlast GEMM** (no transpose). | G5,G2,G1 | attn_wsum 7.9% | ~1,400ms → ~80–150ms (**~10–15×**) |
| 3 | **Rewrite `attn_softmax` with LDS workgroup reduction + `native_exp` + `half` I/O**; one workgroup per (h,tq) row, parallel max/sum. | G9,G10,G6 | attn_softmax 8.8% | ~1,560ms → ~300–500ms (**~3–5×**) |
| 4 | **Fuse the `1/sqrt(D)` query scale into the scores GEMM `alpha`** (drop the separate `attn_scale_q` kernel + a global round-trip). | G7 | scale + a launch | small but free; removes 204 launches |
| 5 | **If GEMM-reformulation is deferred: vectorize the existing `attn_scores`/`attn_wsum` to `half8` + LDS-tile the K/Q rows.** Fallback that still yields most of the win without CLBlast batched API. | G1,G2,G4 | attn_scores/wsum | **~6–12×** as a fallback to #1/#2 |
| 6 | **Flash-attention-style fused scores→softmax→wsum** for the encoder (tile over Tk, keep running max/sum in LDS, never materialize the [1500,1500] score matrix). Bigger rewrite; do after #1–#3 if still short. | G2,G7,G9 | all attention + the 13.5M-elem score buffer | **~2–3×** on top of #1–#3, and slashes memory traffic |

### Tier 2 — Decode host/launch overhead (targets the ~10.7s decode / ~9.7s host)

| # | Optimization | Adreno ref | Targets | Expected |
|---|--------------|-----------|---------|----------|
| 7 | **Cache cross-attention K/V** projected from the (fixed) encoder output — compute once per clip, not once per decode step. | G12 | gemm_crossattn_k/v (100→4 calls) + pack/bias | decode **~1.1–1.2×**; removes ~200 launches |
| 8 | **Remove leftover unconditional `clFinish`** in `GELUActivation.cpp:183` and `Embedding.cpp:192`. | G8 | per-token pipeline stalls | decode **~1.1–1.3×** |
| 9 | **Preallocate a per-layer scratch arena**; stop `clCreateBuffer`/`clReleaseMemObject` per op per token. | G11 | alloc churn | decode **~1.1–1.2×** |
| 10 | **Keep logits on device & sample on GPU (argmax kernel)**, or read back only top-k, eliminating the 51,865-float host readback per token. | G8 | per-token readback | decode **~1.05–1.15×** |
| 11 | **Batch/queue all per-token kernels without intermediate host waits**; single flush per token. | G8 | launch overhead | decode **~1.1–1.2×** |
| 12 | **Persistent decoder KV cache for self-attention** (append one row/step; never reproject the prefix). Confirm it's actually incremental (profile showed self-attn growing). | G12 | self-attn recompute | decode **~1.2–1.5×** on longer outputs |

### Tier 3 — Encoder secondary + Adreno polish

| # | Optimization | Adreno ref | Targets | Expected |
|---|--------------|-----------|---------|----------|
| 13 | **Vectorize `layernorm_forward`** (`half8` loads, LDS reduction). | G1,G9 | layernorm 1.1% | ~196ms → ~60ms |
| 14 | **Vectorize the mel-frontend convs** `whisper_conv1/2_gelu` (`half8`, better workgroup). | G1,G4 | conv 3.2% | ~568ms → ~200ms |
| 15 | **Store Q/K/V (and weights) as `image2d_t`** for the attention GEMMs to hit the Adreno texture cache. | G3 | attention bandwidth | **~1.2–1.5×** on attention GEMMs |
| 16 | **Tune CLBlast for this Adreno** (run its tuner; cache the tuned params) so the new attention GEMMs use device-optimal tile sizes. | G5,G4 | all GEMMs | **~1.1–1.4×** on GEMM-bound work |
| 17 | **Process all 6 heads in one batched-strided GEMM call** rather than 6 separate calls per layer. | G5,G8 | per-head launch overhead | small; fewer launches |
| 18 | **Drop the `attn_transpose` [H,Tq,D]→[Tq,H,D]** by writing the weighted-sum GEMM output directly in [Tq,H,D] layout (epilogue/strides). | G7 | a kernel + round-trip | small; removes 204 launches |
| 19 | **Reuse the encoder-output cache across the whole clip** (already in) and verify the encoder runs exactly once even across chunked/long audio. | G12 | encoder recompute | guards the 13.7× already banked |
| 20 | **fp16 throughout the attention path with fp32 accumulation only** (avoid silent fp32 storage of the [1500,1500] score buffer — it's `__global float*` today, 2× the bandwidth of half). | G6 | score-buffer bandwidth | **~1.2–1.5×** on scores/softmax |

### Expected trajectory

- **After Tier 1 (#1–#3):** encoder ~16.7s → ~2–3s. Clip-0 RTF ~4.64 → ~**1.0–1.3**; medium/long clips (where encoder amortizes) drop **below 1.0**. Aggregate RTF ~3.30 → **~1.0–1.4**.
- **After Tier 2 (#7–#12):** decode ~10.7s → ~4–5s. Clip-0 RTF → ~**0.7–0.9**. **Aggregate RTF < 1.0 — mission target met.**
- **Tier 3:** headroom to push aggregate RTF toward ~0.5 and bank margin for thermal throttling.

---

## 5. Optimization log (chronological, as each lands)

Measurements standardized on `NNOPT_DEBUG_LAYERS=0` (shown equivalent to the `=1` baseline: clip-0 4.678 vs 4.637). Clip-0 used as the fast iteration probe (encoder-heavy); full 10-clip aggregate re-run at milestones.

| # | Optimization | Expected | clip-0 RTF | Transcript ok? | Notes |
|---|--------------|----------|------------|----------------|-------|
| — | baseline | — | 4.637 | ✓ 10/10 | aggregate 3.297; encoder-cache already in |
| 8 | remove leftover decode `clFinish` (GELU, Embedding) | ~1.1–1.3× decode | 4.579 | ✓ | proc 27.15→26.81s; ~340ms (≈125 stalls). Small on short clip (62% encoder); helps long-decode clips more. |
| 7 | cache cross-attn K/V (project once, not per decode step) | ~1.1–1.2× decode | 4.313 | ✓ | with #8. clip-0 proc 26.81→25.25s. **clip-9 (long decode) RTF 2.612→2.369**, proc 47.78→43.32s (~9.3%). Removes k/v GEMM+bias+pack_k/v on every step (~200 launches). Transcripts byte-identical. |
| 1+2+4 | **attention → CLBlast `GemmStridedBatched`** (QKᵀ scores + probs@V, 1/√D folded into alpha) | ~20–40× on attention | **2.102** | ✓ 9/10 byte-identical | **the big one.** `attn_scores` 13,457→~140ms (~96×); `attn_wsum` 1,395→~61ms (~23×); GPU total 17.7s→~2.9s. clip-0 proc 25.25→12.31s. Causal decoder self-attn keeps the naive masked kernel. fp16 scores buffer (opt #20) → 9/10 transcripts identical; clip-9 "Angelov"→"Angelo" (non-deterministic borderline token, arguably *more* correct). |
| 3 | **softmax → 1 workgroup/row + `__local` reduction + `native_exp`** | ~3–5× softmax | **1.792** | ✓ | `attn_softmax` 1,642→134ms (~12×). GPU total ~2.9s→**1.31s**. clip-0 proc 12.31→10.49s. **GPU now only 1.3s of 10.5s — decode is ~9.2s pure host/launch overhead.** New top GPU kernel: `whisper_conv2_gelu` (mel, 31%). |
| 12 | **decoder self-attn KV cache → single-token decode** (O(N²)→O(N); append K/V to `[H,CAP,D]` cache, GEMM over `[0:pos+1]`) | O(N²)→O(N) decode | 2.002 (clip0) / **0.996 (clip4)** | ✓ 8/10 | **length-dependent.** Decode rate now constant ~4.1 tok/s regardless of length. **Long clips win** (clip4 96-tok: 1.123→**0.996**, first sub-1.0!); **short clips regress** (clip0 25-tok: 1.792→2.002) — the per-step overhead (cache-pack launches + `GemmStridedBatched` on 1-row matrices) exceeds the O(N²) savings when N is small. Aggregate **1.388→1.313**. clip2 comma, clip9 "Angelov"→"Angelo". |

### 🔑 Pivotal finding — the wall is now the ENCODER (host-bound)

After #1/#2/#3/#7/#8/#12, **total GPU kernel time is ~1.3s but clip-0 proc is ~10.5s.** Breakdown (clip 0):
- **TTFT (encoder + 1st token) = ~6.0s**, but the encoder's GPU kernels are only ~1.5s → **~4.5s is host overhead** (per-op `clCreateBuffer`/`clReleaseMemObject` of large 1500-seq buffers — the self-attn score buffer alone is `[6,1500,1500]`×2B ≈ **27 MB**, allocated fresh per layer — plus ~100 kernel enqueues).
- Decode ≈ 4.5s, now O(N) and constant-rate.

**Consequence:** GPU-side wins no longer move wall-clock much (e.g. #3 cut softmax 1.5s of GPU but TTFT barely moved 6.01→5.95s). **A 5.86s clip whose encoder takes ~6.0s is RTF ~1.0 from the encoder alone** — so no short clip can go sub-1.0 until the encoder's host overhead drops. The next lever for the mission is **encoder host overhead**: a preallocated scratch-buffer arena (#9 / #11) to stop per-op alloc/free of the big 1500-seq buffers, and fewer ops/buffers per encoder layer. This helps *every* clip's TTFT and is the only path to sub-1.0 on short clips.

### Standings after #12 (full 10-clip)

| Clip | Dur | RTF | | Clip | Dur | RTF |
|-----:|----:|----:|-|-----:|----:|----:|
| 0 | 5.86 | 2.002 | | 5 | 9.01 | 1.301 |
| 1 | 4.82 | 2.160 | | 6 | 5.64 | 1.971 |
| 2 | 12.49 | 1.218 | | 7 | 9.24 | 1.273 |
| 3 | 9.90 | 1.382 | | 8 | 5.12 | 2.004 |
| 4 | 29.40 | **0.996** | | 9 | 18.29 | 1.036 |

**Aggregate RTF = 1.313** (proc 144.1s / 109.8s) · min **0.996** · max 2.16. Long clips at the boundary; short clips gated by the encoder.

### 🔬 #9 investigation — the TTFT wall is JIT compilation, not buffer alloc or weights

Chased the ~4.5s of encoder host overhead. Two hypotheses were tested and **falsified** (honest negative results):
- **Buffer allocation** — pooled the big attention scratch buffers (the 27 MB score buffer + projections, reused across calls instead of alloc/free per op). Result: clip-0 proc 11.49s, TTFT **5.9462→5.9449s — no change.** Not the bottleneck. (Kept anyway: correct, harmless, slightly positive.)
- **Weight upload** — added `Weights::preload_all()` to upload all 167 weight tensors (~72 MB) before the RTF timer. The cold upload IS ~4s, but **moving it out of the timer left proc unchanged** — so it wasn't *in* the timed forward to begin with on a warm device.

**Root cause — first-use JIT compilation.** Direct timing of the forwards:

```
PRELOAD 167 weight buffers   ...  30 ms (warm) / ~4048 ms (cold disk+upload)
PREFILL_FORWARD            5705 ms  ← encode + ALL first-use JIT compile
DECODE_FORWARD[0]          238 ms   ← steady-state, nothing left to compile
TOTAL GPU kernel time     ~1100 ms
```

The prefill forward is **5.7s but a steady-state decode forward is 0.24s.** Of the 5.7s: ~1s is encoder GPU, ~0.8s is our `.cl` `clBuildProgram`, and **~3.9s is CLBlast JIT-compiling its GEMM kernels** for each shape on first use. CLBlast caches compiled kernels **in-memory per process** — and **each benchmark clip is a separate process**, so *every clip recompiles from scratch (~4s)*. That ~4s × 10 clips is the dominant cost in the whole benchmark.

**Implications for the two metrics:**
- **Per-process (cold-start) RTF** — what the benchmark reports — is inflated by ~5.5s one-time init/clip. This is the honest number for "launch the binary, transcribe one clip, exit."
- **Steady-state (deployed) RTF** — model loaded once, streaming many clips — pays init **once**, then each clip is `encode (~1s) + N×0.238s decode`. Estimated steady-state: clip 4 (96 tok) ≈ (1 + 22.8)/29.4 = **RTF 0.81**; clip 0 (25 tok) ≈ (1 + 6.0)/5.86 = **RTF 1.19**. **Long clips are comfortably faster than real-time in deployment; short clips are gated by the 238ms/token decode (launch-bound).**

### Real remaining levers (re-prioritized by this finding)
1. **Persist compiled kernels across processes** ✅ **DONE** (lever #1). CLBlast exposes no disk cache (`ClearCache`/`FillCache` are in-memory only), so the robust fix was the real-deployment pattern: `--audio-list` **batch mode** transcribes all clips in ONE process, paying JIT once. Needed `WhisperSdpaAttention_reset_caches()` (called from the backbone on each encoder re-run) to clear the per-`wp` KV caches between clips, and keeping feats buffers alive so the encoder-cache pointer key stays distinct per clip. **Result: aggregate RTF 1.313 → 0.972 — mission achieved.** (See top of file.)
2. **Cut decode per-token cost** (238ms/token, the steady-state short-clip bottleneck): it's launch/host-bound (~60 kernel enqueues/token), so **op fusion (#11)** — fuse layernorm+bias+residual+GELU chains — to cut kernels-per-token, not per-kernel work. (Note: #12 traded a little of this for O(N) scaling — net win on long clips.)
3. **Encoder GPU** (#13/#14: vectorize layernorm + mel convs) — small now (~1s) and won't move wall-clock until #1 lands.

### 🏁 Milestone — after #1/#2/#4 (+#7/#8): aggregate RTF 3.297 → **1.388**

Full 10-clip re-benchmark (fp16, `NNOPT_DEBUG_LAYERS=0`):

| Clip | Dur (s) | Baseline RTF | **Now RTF** | Transcript |
|-----:|--------:|-------------:|------------:|:-----------|
| 0 | 5.86  | 4.637 | 2.058 | ✓ identical |
| 1 | 4.82  | 5.034 | 2.301 | ✓ identical |
| 2 | 12.49 | 2.893 | 1.264 | ✓ identical |
| 3 | 9.90  | 3.254 | 1.422 | ✓ identical |
| 4 | 29.40 | 3.063 | **1.123** | ✓ identical |
| 5 | 9.01  | 3.025 | 1.380 | ✓ identical |
| 6 | 5.64  | 4.528 | 2.057 | ✓ identical |
| 7 | 9.24  | 2.965 | 1.309 | ✓ identical |
| 8 | 5.12  | 4.700 | 2.157 | ✓ identical |
| 9 | 18.29 | 2.612 | **1.046** | ~ "Angelov"→"Angelo" |

**Aggregate RTF = 1.388** (was 3.297) · proc_total 361.8s → 152.3s · mean per-clip 1.612 · min **1.046** · max 2.30.
**Long clips (18–29s) are already faster than real-time.** Short clips sit at ~2.0–2.3 because the (now ~1.5s) encoder + decode host overhead amortize over few tokens.

### Post-attention profile (clip 0) — where time goes now

```
attn_softmax            1642.4 ms   56.7%   ← NEW hotspot (naive serial softmax) → opt #3
whisper_conv2_gelu (mel) 464.3 ms   16.0%   → opt #14
layernorm_forward        203.3 ms    7.0%   → opt #13
whisper_conv1_gelu (mel) 177.2 ms    6.1%   → opt #14
attn_scores_gemm         140.1 ms    4.8%   ← was 13,457ms
attn_wsum_gemm            60.8 ms    2.1%   ← was 1,395ms
TOTAL GPU kernel time    ~2,900 ms
```

Two shifts: (1) GPU work is now dominated by **`attn_softmax`** (opt #3 — LDS reduction + `native_exp`); (2) GPU total is only ~2.9s of the 12.3s clip-0 proc — **decode host overhead (~9.4s) now dominates**, promoting Tier-2 (KV cache #12, fewer per-token launches #11, GPU sampling #10) as the path under RTF 1.0 on short clips.
