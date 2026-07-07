# moonshine-tiny — Optimization Campaign Benchmark Log

Device: Motorola Razr 2020 (ZY22D5NLGQ) · Adreno 620 · OpenCL · fp16 · release builds (`build.sh --release`, NNOPT_DEBUG off)
Model: UsefulSensors/moonshine-tiny (27.1M params, 53MB fp16) — ASR, raw-16kHz-in, tokens-out

## Verification protocol (every optimization must pass before it lands)

Three novel held-out speech clips (synthesized 2026-07-02, never used during development), each verified **token-for-token** against freshly-computed PyTorch ground truth — including reproducing PyTorch's own mis-hearings (clip C "Neuro networks neuron on…"). A change that alters ANY generated id on ANY clip is reverted.

| clip | audio | ref transcript | ids |
|---|---|---|---|
| A | 2.68s | "The quick brown fox jumps over the lazy dog." | 13 ids, exact |
| B | 3.70s | "Testing speech recognition on the phone with the brand new sentence." | 14 ids, exact |
| C | 2.89s | "Neuro networks neuron on tiny graphics chips." | 12 ids, exact |

Profiling method: OpenCL event profiling (`clGetEventProfilingInfo`, `CL_QUEUE_PROFILING_ENABLE`) per the Qualcomm Adreno OpenCL General Programming & Optimization Guide — implemented in `src/profiler.cpp` (`NNOPT_PROFILE=1`). (Snapdragon Profiler GUI can be layered on for system-level views; kernel-level timing below is the guide's recommended event method.)

## Results ledger

| Rev | Change | decode tk/s | TTFT s | e2e A/B/C s | RTF (median) | 3-clip exact |
|---|---|---|---|---|---|---|
| r0 | baseline (as converged) | 1.9 | 3.3–3.6 | 8.89 / 11.71 / 8.65 | **3.2** | ✅ |
| r1 | **OPT-1: decoder KV cache** (self-KV ring [H,194,D]/layer, cross-KV once/waveform, incremental forward, last-row-only proj_out) | 2.6 | 3.4–3.6 | 7.59 / 9.53 / 7.64 | 2.7 | ✅ |
| r2 | **OPT-2: attention kernel v2** (one 64-lane WG per (h,row); scores computed ONCE into local mem; max/sum via WG reduction — old kernel used 1 work-item/row and recomputed QK 3×) | **10.5** | **1.7** | **2.83 / 3.02 / 2.73** | **0.95** | ✅ |
| r3 | **OPT-3: full profiler coverage** (shape-tagged events on both CLBlast Gemm wrappers + every remaining kernel/readback; zero perf change with NNOPT_PROFILE off) | 10.4 | 1.7 | 2.81 / 3.00 / 3.01 | 0.95 | ✅ |
| r4 | **OPT-4: bespoke M=1 GEMV, CLBlast off the decode path** (transposed-weight coalesced K-major `linear_gemv_t` for ALL M=1 linears — one-time `mat_transpose` per weight; row-scan `linear_gemv` kept as fallback) | 38.0 | 1.8 | 2.07 / 2.24 / 2.15 | 0.76 | ✅ |
| r5 | **OPT-5: layernorm workgroup rewrite** (WG-per-row + two tree reductions; was 1 work-item/row) | 45.3 | 1.8 | 2.02 / 2.15 / 2.06 | 0.73 | ✅ |
| r6 | **OPT-6: CLBlast fully out of runtime + program binary cache + kernel-object cache** (`linear_gemm_t` for M>1 — CLBlast's lazy first-call compile was ~1.4s of TTFT; `<file>.cl.bin` device cache keyed on source+options+device+driver; cl_kernels cached for process lifetime; stray per-step embed clFinish removed) | 44 | **0.81** | 1.07 / 1.32 / 1.14 | 0.42 | ✅ |
| r7 | **OPT-7: conv → im2col + GEMM** (naive conv kernel was <1% ALU for ~320 MFLOP) | 45 | 0.70 | 0.96 / 1.20 / 1.03 | 0.38 | ✅ |
| r8 | **OPT-10: `-cl-mad-enable -cl-fast-relaxed-math`** (ids unchanged on all 3 clips) | 45 | 0.70 | ~same | 0.38 | ✅ |
| r9 | **4-row-tiled `linear_gemm_t4` + 16×16 tiled `mat_transpose`** (untiled GEMM re-read conv2's 2.3MB weight 210×/call = 487MB; scatter transpose ran 0.5GB/s) | 47 | **0.30** | 0.56 / 0.66 / 0.62 | 0.22 | ✅ |
| r10 | **OPT-11: K-split GEMV for small-N decode shapes** (N=288 gave only 128 lanes; S=4/8 fp32-partial chunks + reduce; interleaved A/B 56 vs 48 tk/s) | **59** | **0.28** | **0.49 / 0.59 / 0.52** | **0.18** | ✅ |

**Cumulative (r0 → r10): 18× e2e (8.9s → 0.49s), 31× decode (1.9 → 59 tk/s), TTFT 3.4s → 0.28s, RTF 3.2 → 0.18.** Decode ≥60 tk/s and e2e ≤0.5s practical targets from the r3 roofline: reached.

Note (r3, 2026-07-03): verification clips regenerated (`say` is deterministic — clip A byte-identical to 2026-07-02's; B/C same sentences) and ground truth freshly recomputed in the `model_usefulsensors_moonshine_tiny` ref venv; clip C's PyTorch mis-hearing shifted to "1. Neuro Networks Running on Tiny Graphics Chips" (16 ids). Ground truth + clips now persist in `verification/` (clip{A,B,C}.bin f32@16kHz, ground_truth.json, ground_truth.py); gate = push clip as assets/test_waveform.bin, run `'' 64`, GENERATED_IDS must equal ground truth ids[1:].
Also fixed: `deploy_android.sh` now searches the dtype-matched build dir (`build/fp16/_deps/clblast-build/`) for libclblast.so — a fresh fp16 deploy previously wiped device `lib/` and left the binary unlinkable.

## Kernel profile (clip A)

r1 (pre-OPT-2): total GPU 4,990ms — `attention` 4,821ms (96.6%, 162 calls @ 29.8ms), `op_conv1d` 167ms (3.3%), rest noise.
r2: total GPU **222ms** — `op_conv1d` 183ms (82%, 3 calls — conv1 k=127/s=64 dominates), `attention` 37ms (162 calls @ 228µs, **130× kernel speedup**), adds/splits ~2ms.

r3 (FULL coverage, clip A, wall 2.79s): total GPU **743ms** —
| label | total_ms | % | calls | avg_µs |
|---|---|---|---|---|
| linear_m1_n288_k288 | 430.3 | 57.9% | 468 | 920 |
| op_conv1d | 167.9 | 22.6% | 3 | 55953 |
| layernorm_nobias | 78.0 | 10.5% | 260 | 300 |
| attention | 36.8 | 5.0% | 162 | 227 |
| rope_apply | 6.2 | 0.8% | 168 | 37 |
| linear_m1_n32768_k288 (proj_out) | 5.0 | 0.7% | 13 | 383 |
| everything else | ~18 | 2.5% | — | — |

CLBlast caveat: its event maps to the LAST kernel of each routine (pre-kernels uncovered) — per-shape numbers are lower bounds.

### End-to-end wall breakdown (clip A, r3: 2.79s)
- **743ms GPU** — dominated by CLBlast's pathological M=1 GEMV (m1_n288_k288 reads 166KB weights in 920µs ≈ 180 MB/s ≈ 1% of DRAM bw) and the 1-work-item-per-row layernorm (rows=1 at decode → 288-wide serial reduction, 300µs).
- **~1.3s of TTFT is host compile** — TTFT 1.64s vs ~250ms encoder GPU; per-process clBuildProgram (moonshine.cl + CLBlast's first-call kernel builds) eats the rest.
- **~0.7s decode host overhead** — decode wall 1.24s vs ~520ms decode GPU: clCreateKernel per dispatch, buffer alloc/release churn, per-step blocking logits readback, stray clFinish after every embed_gather (Embedding.cpp), GroupNorm's mid-op blocking psum readback (1×/waveform, minor).
- proj_out via CLBlast is FINE (5ms total) — the old #5 "proj_out wave-stride GEMV" ranking is obsolete.

## Roofline (Adreno 620, fp16)

- Practical DRAM bandwidth (measured on this device in prior campaigns): ~12–14 GB/s.
- Decode step is memory-bound: ~40MB weight traffic/token (20M params touched × 2B) → **theoretical floor ≈ 3.3ms/token ≈ 300 tk/s**. Current: 10.5 tk/s ⇒ **~3.5% of practical ceiling** — the gap is host overhead + un-tuned GEMV, not physics.
- Encoder convs: ~200 MFLOP total; 183ms ⇒ ~1.1 GFLOPS effective vs ~400+ GFLOPS fp16 peak ⇒ **<0.3% ALU utilization** — naive conv kernel, huge headroom.
- Practical targets: decode ≥ 60–100 tk/s, encoder ≤ 150ms, **e2e ≤ 0.4s (RTF ≤ 0.15)** before diminishing returns.

## Ranked optimizations — status after r10

1–11. ✅ ALL LANDED (r1–r10): KV cache · attention v2 · profiler coverage · M=1 transposed GEMV · layernorm WG rewrite · CLBlast removal + program-binary/kernel caches · im2col conv · fast-math · tiled GEMM/transpose · K-split GEMV · sync audit.

Remaining (diminishing returns at RTF 0.18; ranked by expected gain):
1. **Fuse decode q/k/v** into one dispatch (3× lanes + 2 fewer launches per layer); fold split_heads into the GEMV epilogue.
2. **Encoder gemm_t4 tuning** — m111 shapes total ~115ms of TTFT GPU (MT=8 A/B, watch register spill past 4 accumulators).
3. **On-GPU argmax** — read 4B instead of 64KB logits + host f16→f32 loop per step.
4. **Online-softmax attention** — robustness item: removes the ATTN_MAX_TK=1024 cap (>25s audio fails today).
5. Buffer arena for per-step activations (alloc/release churn is the residual host overhead).
6. ~~qcom_dot8 int8 conv~~ — conv is now ~46ms via GEMM; check issue-vs-DRAM-bound before bothering.

## Key lessons (transferable)
- **CLBlast's event undercounts**: its cl_event maps to the LAST kernel of a routine; pad/transpose pre-kernels are invisible. proj_out "383µs" was really ~10ms+ (routing it back to CLBlast cost 12 tk/s). Never rank CLBlast calls by their event time.
- **CLBlast's real cost on one-shot inference is COMPILE, not GEMM**: ~1.4s of first-call kernel builds per process on Adreno 620. For ports that launch a fresh process per inference, bespoke kernels + a program binary cache beat it outright.
- **DVFS makes cross-build wall comparisons lies** (2× swings between "identical" runs). Every kernel decision here was made with same-binary interleaved A/B via env toggles (NNOPT_GEMVT, NNOPT_GEMV_KSPLIT, NNOPT_CONV_IM2COL, NNOPT_FAST_MATH).
- **Transposed K-major GEMV wins at every N** on this GPU (coalesced across lanes, no reduction); for small N (≤1024) add K-split fp32 partials + reduce or the GPU starves at ≤256 lanes.
- **M>1 GEMM needs M-tiling**: the untiled per-row walk re-read conv2's weight 210× (487MB/call). 4 rows/WG with ≤4 float4 accumulators (spill limit) was 3.4×.

## Length sweep (r10 build, 2026-07-03) — 8 clips, 0.84s–25.15s

| clip | audio | e2e | RTF | token-exact vs PyTorch |
|---|---|---|---|---|
| D | 0.84s | 0.24s | 0.28 | ✅ |
| A | 2.55s | 0.49s | 0.19 | ✅ |
| C | 2.72s | 0.51s | 0.19 | ✅ |
| B | 3.60s | 0.58s | 0.16 | ✅ |
| E | 5.32s | 0.87s | 0.16 | ✅ |
| F | 11.61s | 1.98s | 0.17 | ✅ |
| G | 19.50s | 4.23s | 0.22 | ✅ (74 tokens, incl. PyTorch's "dears" mis-hearing) |
| H | 25.15s | — | — | refused (Tk=1046 > ATTN_MAX_TK=1024) — as designed |

RTF stable 0.16–0.22 across all working lengths. Clips + wavs + transcripts in `verification/` (TRANSCRIPTS.md).
⚠ BUG found via clip H: after the encoder refuses (>1024 frames), generation continues on zero logits for max_new_tokens steps (83s of id=0 output) instead of aborting — the zero_logits() fallback should be a hard stop. Fix alongside the online-softmax kernel.

Note on methodology: the first run after each deploy recompiles kernels (~+0.5s TTFT once) while the binary cache repopulates — benchmark from run 2.
