
## Optimization campaign (2026-07-04, evening)

Baseline (release, fp16, 518×686): **40.0 s/frame** (0.025 FPS). GPU kernel time 37.3s = 93.6% of wall.
Profile: attn_scores 48.7% + attn_context 19.7% + attn_softmax 8.7% = **attention 77%**; conv2d 19.1%; all else <4%.

| # | Change | Result | Valid |
|---|--------|--------|-------|
| 1 | Profiler wired at run1d (kernel-name introspection) + dump_summary in main | per-kernel ledger live | n/a |
| 2 | attn_scores + attn_context → CLBlast GemmStridedBatched (head-major strides; context writes token-major via c_ld=D, c_stride=hd — no transpose kernel). A/B: NNOPT_ATTN_GEMM=0 | **40.0 → 17.9 s (2.2×)** | cos vs reference 0.999988, rms 0.992 ✓ |

Next (by new expected share): conv2d (~7.1s, im2col+GEMM), attn_softmax (~3.2s, vectorize+fuse scale), then re-profile.
| 3 | conv2d → im2col+CLBlast GEMM (1×1 = direct GEMM, K×K chunked im2col, persistent col scratch, NNOPT_CONV_GEMM=0 reverts) | 17.9 → 12.9 s | cos 0.999986 ✓ |
| 4 | attn_softmax → workgroup-per-row + local reductions + native_exp (NNOPT_SOFTMAX_WG=0 reverts) | 12.9 → **10.03 s (4.0× cumulative)** | cos 0.999987 ✓ |
| 5 | Activation buffer pool (exact-size free-list; also correctness: heap fragmentation) | neutral on wall; keeps OOM class away | cos 0.999987 ✓ |

**Final 2026-07-05: 10.03 s/frame (from 40.0 s) = 4.0×, cosine 0.999987.** Phase split: weights 0.002s / forward 10.0s; steady-state forward (warm process) 6.64s; GPU kernel ledger ≈3.2s.
Next levers (designed, unimplemented): CLBlast per-shape compile cache = −3.4s one-time (binary-cache pattern from moonshine ledger); CLBlast internal pad-buffer churn on odd ld (host gap suspect, ~2-3s); im2col vectorize (790ms); conv_transpose2d as GEMM+scatter (732ms); fuse layernorm/to_head_major/gelu chains (~500ms); workgroup tuning sweep per guide §6.1.

## Re-profile (2026-07-07, fresh end-to-end, NNOPT_REPEAT added)

`main.cpp` gained an env-gated `NNOPT_REPEAT=N` in-process frame loop (frame 1 = cold, 2+ = steady state).
Deploy gap recurred a THIRD time: `lib/` on device was missing `libclblast.so` after deploy_android.sh — pushed manually.

3-frame run (release fp16, NNOPT_DEBUG_LAYERS=0), process wall 23.32s (`time`, user 3.81s + sys 2.90s host CPU):
- weights_load 0.006s | frame1 9.840s | frame2 6.603s | frame3 6.586s → one-time init ≈ 3.24s, steady ≈ 6.60s
- e2e cosine vs reference/predicted_depth.bin: **0.999990** ✓

Per-kernel GPU ledger (NNOPT_PROFILE=1, single frame) — TOTAL GPU 3058.5 ms, so steady host gap ≈ 3.54s (54% of warm frame).
NOTE: CLBlast rows are undercounts (event tracks only the final internal kernel — see clblast-oneshot-traps memory).

| label | ms | % | calls |
|---|---|---|---|
| im2col2d | 784.2 | 25.6% | 26 |
| conv_transpose2d | 724.4 | 23.7% | 2 |
| attn_softmax_wg | 456.9 | 14.9% | 12 |
| clblast_attn_scores | 316.3 | 10.3% | 12 |
| to_head_major | 203.1 | 6.6% | 36 |
| layernorm_rows | 180.6 | 5.9% | 28 |
| add_bias_rows | 98.8 | 3.2% | 72 |
| bilinear_interp | 57.7 | 1.9% | 5 |
| bias_per_row | 47.1 | 1.5% | 27 |
| gelu | 42.1 | 1.4% | 12 |
| clblast_conv_gemm | 37.5 | 1.2% | 26 |
| others (add/layer_scale/conv1x1/relu/attn_context/...) | ~110 | 3.6% | — |

## Lever 1 (2026-07-07): CLBlast on-disk program binary cache — one-time compile 3.3s → 0.15s

Patch: `src/third_party/clblast/routine.cpp` overlaid onto _deps at configure (CMake configure_file COPYONLY —
survives clean rebuilds). Disk layer around CLBlast's BinaryCache: `<NNOPT_CLBLAST_CACHE_DIR>/<fnv1a>.clbin`
keyed routine_info+device+driver+precision; stale binary → remove + fall back to source compile.
Kill-switch: NNOPT_CLBLAST_CACHE=0 (run script sets cache dir by default).
Also fixed for good: deploy now pushes `$BUILD_DIR/_deps/*-build/*.so` (libclblast.so gap, bitten 3×) and
run_android.sh forwards ALL host NNOPT_* env vars to the device shell.

A/B (same binary, NNOPT_REPEAT=2, release fp16):
- cache OFF: frame1 9.930s / frame2 6.600s (stock behavior confirmed unregressed)
- cache ON, cold disk: frame1 9.718s (writes 2 .clbin: 217KB+109KB)
- cache ON, warm disk: **frame1 6.736s / frame2 6.591s** → one-time cost now ~0.15s
- e2e cosine vs reference: **0.999990** ✓ (depth stats bit-identical to pre-patch: min=0.8125 max=5.09375 mean=2.8236)

Single-shot wall is now ≈ steady-state. Next: the 3.54s host gap (lever 2).

## Lever 2 investigation (2026-07-07): THE HOST GAP DOES NOT EXIST — profiler undercount did

kgsl gpubusy sampled at 0.5s cadence across steady frames: GPU is 96-100% busy the entire frame.
The "3.5s host gap" was CLBlast composite time hidden by the last-kernel-only event (moonshine Trap 1, now
measured here). New `NNOPT_SYNC_PROF=1` (clFinish-bracketed host clock around every CLBlast call) gives truth:

| CLBlast call | event ledger (lie) | sync-prof (truth) |
|---|---|---|
| clblast_attn_scores (12) | 316 ms | **1303 ms** |
| clblast_conv_gemm (26) | 37 ms | **960 ms** |
| clblast_attn_context (12) | 11 ms | **906 ms** |
| clblast_conv1x1 (9) | 19 ms | **117 ms** |
| total | 384 ms | **3286 ms** |

True warm-frame ledger: CLBlast 3.29s + our kernels 3.14s ≈ 6.6s wall ✓ (fully accounted, GPU-bound).

Dead ends, documented:
- NNOPT_GEMM_DIRECT (OverrideParameters XGEMM_MIN_INDIRECT_SIZE): TRAP — CLBlast CUBES the value in
  unsigned long long; 1<<30 overflows 2^64 and silently keeps indirect. With 1<<21 (2^63, correct):
  direct = 28.5s/frame vs indirect 6.58s — untuned XgemmDirect is 4.3× SLOWER on Adreno 620. Default OFF.
- Padding lds / persistent temp_buffer: pointless — pad work is GPU-side and the GPU has no idle to hide it in.

Live lever: CLBlast Xgemm params are generic defaults for this device → on-device tuner (one-off TUNERS=ON
build in /tmp/clblast-tuners) → OverrideParameters("Xgemm", kHalf, tuned). Covers plain + strided-batched.

## Levers 3+4 landed (2026-07-07): warm frame 6.51 → 5.77s, single-shot ≈5.9s (cos 0.999990 ✓)

| lever | change | kernel time | wall |
|---|---|---|---|
| 3 | conv_transpose2d → `conv_transpose2d_sk` (stride==K ⇒ exactly one weight tap per output: ih=oh/S, kh=oh%S; no window loop, no div/mod guards; 4 ow per WI) | **745 → 69.5 ms (10.7×)** | −0.7s combined |
| 4 | im2col2d → `im2col2d_v4` (4 positions/WI, vload_half4 interior fast path) | 791 → 678 ms (1.17×) | ⌃ |

Interleaved A/B ×2 rounds: defaults 5.76-5.78s warm vs old kernels 6.48-6.55s. Kill-switches NNOPT_CONVT_SK=0 / NNOPT_IM2COL_V4=0.

**TRAP discovered en route (cost a full bisect): Adreno program-wide register footprint.**
Adding these kernels to depth_pipeline.cl slowed OTHER kernels 2-7× (im2col 790→2956ms, add 23→159ms,
frame 6.6→10.5s) with the new kernels never dispatched — one fat kernel cuts wave occupancy for the whole
cl_program. Fix: campaign kernels live in a SEPARATE program `kernels/depth_pipeline_ext.cl`. Rule: never
add register-heavy kernels to an existing lean program; quarantine and A/B.

Xgemm tuned-at-1024³ params: +0.9s at real shapes (thin-k attn / wide-n conv) — default OFF (NNOPT_XGEMM_TUNED=1
opt-in). Real-shape tuner run (m=1814 n=1814 k=64) queued as lever-2 round 2.

True remaining ledger (warm 5.77s): CLBlast composite ~3.29s (attn_scores 1303 + conv_gemm 960 + attn_context 906 +
conv1x1 117 — sync-prof), our kernels ~2.30s (im2col_v4 678, softmax_wg 456, to_head_major 206, layernorm 176, …).

## Lever 5a landed (2026-07-07): attention head-loop — warm 5.77 → 5.24s (cos 0.999990 ✓)

`NNOPT_ATTN_HEADLOOP=1` (default): per-head PLAIN clblast::Gemm reading token-major Q/K/V slices in place
(a_offset=h·hd, ld=D) and writing context token-major (c_offset=h·hd, ld=D). Eliminates ALL 36
to_head_major copies (206ms) AND sidesteps the batched kernel variant. A/B ×3 interleaved rounds:
headloop 5.24s / batched 5.75s / headloop+tuned 5.84s.

**Xgemm OverrideParameters tuning: CLOSED as a dead end** — tuned at 1024³: +0.9s; tuned at the real shape
1792×1792×64 (64.9 GFLOPS in the tuner harness!): +0.6s in-model, even on the plain-kernel path. Isolated
tuner throughput does not transfer to this workload. NNOPT_XGEMM_TUNED=1 opt-in retained for reference.

Sync-prof after headloop: scores 1303→847ms, context 906→816ms, conv_gemm ~1000ms, conv1x1 145ms.

**Cumulative: 40.0 → 5.24s warm (7.6×), single-shot ≈5.4s, cosine 0.999990 throughout.**
Remaining ledger (warm 5.24s ≈ CLBlast ~2.8s + ours ~2.1s + softmax): next levers by size:
conv_gemm ~1.0s (per-shape behavior, maybe Convgemm routine or bespoke), attn scores/context ~1.7s
(bespoke thin-k batched kernel, moonshine-style), im2col_v4 678ms (8-wide / fold into a bespoke conv),
softmax_wg 456ms (vectorize row walk), elementwise fusion ~300ms (bias+gelu, bias+residual).
