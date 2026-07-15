# Benchmark — stable-audio-open-small on motorola razr 2020 (Adreno 620)

Release fp16, fully on-device (T5 conditioning on device CPU, DiT + VAE on GPU),
seed noise from assets/ (deterministic replay — never device RNG).

## Verified run (2026-07-07)

Prompt "warm analog synth chords with soft rain", seed 42, 11 s stereo @ 44.1 kHz, 8 denoise steps:

| Metric | Value |
|---|---|
| conditioning_sec (T5-small, CPU) | 8.32 |
| dit_total_sec (8 steps) | 51.50 |
| dit_step_avg_sec | 6.44 |
| decoder_sec (VAE) | 40.43 |
| dec_host_weight_fold_upload_sec | 3.06 |
| dec_host_download_sec | 1.91 |
| total_inference_sec | **100.75** (RTF ≈ 9.2 for 11 s audio) |
| time_to_first_token_sec | 16.24 |
| peak_cpu_memory_mb | 2429 |
| latent_len | 256 |

Output gate: non-silent (peak 23007 / rms 5587 int16), bit-deterministic per (prompt, seed).

## Verified run (2026-07-08, optimization campaign)

Same methodology (prompt "warm analog synth chords with soft rain", seed 42,
11 s stereo @ 44.1 kHz, 8 steps), warm on-device caches (program binaries +
folded VAE weights; a first-ever run additionally pays ~1.5 s cache build):

| Metric | Value |
|---|---|
| conditioning_sec (T5-small, CPU) | 2.66 |
| dit_total_sec (8 steps) | 32.02 |
| dit_step_avg_sec | 4.00 |
| decoder_sec (VAE) | 23.75 |
| dec_host_weight_fold_upload_sec | 0.40 |
| dec_host_download_sec | 1.88 |
| total_inference_sec | **59.06** (RTF ≈ 5.4 for 11 s audio) |
| time_to_first_token_sec | 6.97 |
| peak_cpu_memory_mb | 2581 |

1.91× vs the same-session all-off baseline (113.08 s), 1.71× vs the 2026-07-07
run above. Output gate: non-silent (peak 22927 / rms 5591), cosine vs the
pre-campaign waveform 0.99979 (numerics shift from fused-attention
`native_exp` + on-GPU fp16 pingpong; bit-identical when the new kernels are
switched off).

## Campaign ledger (2026-07-08, each measured on-device, cumulative)

| Lever | Switch | total_s | Δ |
|---|---|---|---|
| all-off baseline (same session) | — | 113.08 | — |
| program binary cache + CLBlast disk cache (guide §5.7.3) | `NNOPT_PROG_CACHE` / `NNOPT_CLBLAST_CACHE_DIR` | 93.71 | −19.4 (TTFT 27.8→9.1) |
| step-invariant cache (cond embeds + RoPE hoisted from the 8-step loop) | `NNOPT_STEP_CACHE` | 92.63 | −1.1 |
| fused workgroup attention (scores in __local, WG softmax, merge folded) | `NNOPT_ATTN_FUSED` | 84.71 | −7.9 |
| workgroup LayerNorm (was 1 WI/row) | `NNOPT_LN_WG` | 77.67 | −6.3 |
| VAE folded-weight disk cache (fold 4.1 s→0.3 s; helps cold runs) | `NNOPT_DEC_FOLD_CACHE` | 77.59 | ~0 warm |
| convT-as-GEMM + col2im gather (t4x4 kernel was 18.9 s ≈ 52% of GPU time at ~8 GFLOPS) | `NNOPT_CONVT_GEMM` | 62.36 | −15.2 |
| workgroup qk-norm (was 1 WI/row, 6.4 ms × 512 calls) | `NNOPT_QK_WG` | 59.56 | −3.0 |
| on-GPU denoise loop (device-resident latent, fused pingpong kernel) | `NNOPT_DENOISE_GPU` | 59.56 | ±0 (removes 8 syncs) |
| aligned half8 elementwise (snake/bias/residual) | `NNOPT_VEC_KERNELS` | 59.06 | −0.3 |
| decoder activations via buffer pool (was raw clCreateBuffer churn) | `NNOPT_DEC_POOL` | 59.06 | −0.3 |
| `--serve` mode (persistent process; gen 2+ = 57.8 s) | `--serve` | — | app-side |
| cl_qcom_perf_hint HIGH context | `NNOPT_PERF_HINT` | — | ~0 |

Measured dead ends (kept behind opt-in switches for future silicon):

- **CLBlast Xgemm tuning** (`NNOPT_XGEMM_TUNED=1`): on-device tuner at the
  dominant FFN shape (m=256 n=8192 k=1024 fp16) found 87.6 GFLOPS vs 12.1
  stock — but in-model DiT went 32.0 → 35.9 s. Same verdict as the
  depth-anything campaign: isolated tuner throughput does not transfer
  (pytorch_linear runs the TransB path on padded M=257). Tuner results:
  ladder_results/clblast_xgemm_*_16.json.
- **Multi-row fused attention** (`NNOPT_ATTN_ROWS=4`): R=8 rows/WG DiT
  32→57 s, R=4 32→132 s at equal temperature. The 4–8× K/V-traffic saving is
  erased by occupancy collapse from the per-WG __local footprint on this
  single small CU.
- **8-wide im2col** (`NNOPT_IM2COL_V8=1`): unaligned vload_half8 (offset
  shifts by k·dilation−padding) is slower than the scalar 4-wide kernel;
  decoder 24.3 → 34.7 s.
- **Pre-transposed NoTrans GEMM** (`NNOPT_PRETRANS=1`): DiT 32.0 → 42.8 s
  stock, 39.4 s with tuned params, +630 MB resident. CLBlast is ColMajor
  internally, so RowMajor TransB (pytorch_linear) is ALREADY its
  transpose-free layout; forcing RowMajor NoTrans adds pad/transpose passes.
  Together with the two tuning failures this closes ALL CLBlast-knob routes —
  the GEMM wall needs a bespoke kernel.

Also measured: **4-step sampler** = 42.4 s total (DiT 16.1 s, RTF 3.9) —
non-silent output (peak 19103 / rms 3471), audibly sparser than 8-step;
user-approved 2026-07-08 as an app-configurable option (app pushes the s4
noise assets; the binary derives step count from assets/sigmas.bin).
`cl_qcom_recordable_queues` IS present on this driver (probe 2026-07-08) —
the DiT-step replay lever (guide §9.1.3) is available for a future round.

## Round 3 (2026-07-08, after the GEMM-wall analysis)

- **Decoder-scoped tuned Xgemm** (`NNOPT_DEC_TUNED`, default on): the tuned
  set is applied only around Decoder::decode and restored to the Adreno stock
  entry on exit (safe without cache clears — CLBlast keys compiled programs
  by param values; restore matters for serve mode). Measured decoder 22.5 /
  23.9 s across two runs vs 23.6 / 24.3 stock — neutral-to-slightly-positive
  within thermal noise. Corroborating: tuning at an actual decoder conv shape
  (m=128 n=16384 k=896) returned params IDENTICAL to CLBlast's stock Adreno
  wildcard entry — the decoder's GEMMs are already at their CLBlast ceiling.
- **Direct t4x4 conv instead of im2col+GEMM** (`NNOPT_CONV_GEMM=0`): decoder
  24 → 104.5 s. The hand conv kernel loses 4.4× to im2col+CLBlast at these
  shapes; GEMM formulation stays.
- **Bespoke panel-packed linear** (`NNOPT_BESPOKE_LINEAR=1`): W repacked to
  coalesced [K/8][N][8] panels, pure mad8 inner loop, MT=4 rows/lane — DiT
  32 → 152 s. Four float8 accumulators spill (the playbook's ≤4-float4 cliff);
  smaller MT makes weight re-read traffic (W × M/MT) the wall before ALU.
  With local staging also a measured loser here (attn r8), the only remaining
  bespoke design on this CU is TEXTURE-PATH weight tiles (guide §6.2, musicgen
  GEMV precedent) — needs a micro-benchmark harness, scoped as its own
  campaign day. Kernel kept for that baseline — QUARANTINED in
  kernels/linear_p8.cl: while it lived in dit.cl its register spill degraded
  every kernel in the program (DiT 32 → 44 s WITHOUT being dispatched; the
  Adreno program-wide register trap from the depth campaign, re-confirmed).
- Final verified state after round 3: one-shot 58.95 s; serve mode gen 2 =
  56.8 s (DiT 32.0 stock / decoder 22.2 tuned-scoped — the override + stock
  restore verified across generations).

## Round 4 (2026-07-08 evening): gaps disproven, padding peeled, serve pipelined

- **GPU-busy sampling** (kgsl gpubusy, 500 ms windows): DiT 99.8% busy,
  decoder 99.9% busy. There are NO dispatch gaps — recordable queues would
  buy ~nothing and the lever is CLOSED without building it. The GEMM wall is
  kernels occupying the GPU while under-using ALUs, not scheduling.
- **tools/gemm_bench** micro-harness (seconds per data point, cosine-checked
  vs CLBlast): baseline reproduces the in-model 55 GFLOPS. Bespoke candidates:
  texture MT4 12 → +local-x-tile 27 (texture beats packed buffer 27 vs 23) →
  2D-lane mad variant 18. Best hand kernel = half of CLBlast; combined with
  round 3, bespoke GEMM on this CU is CLOSED for buffer/texture/local designs.
  Harness + kernels kept for future silicon or QNN-based attempts.
- **B14 split-M (kept, default on; `NNOPT_SPLIT_M=0` reverts)**: CLBlast pads
  M to the next MWG=64 multiple — measured M=257 ≡ M=320 (78 ms) vs M=256 =
  65 ms. DiT's M=257 (prepended global token) and cross-attn's M=65 both pay
  it. Fix: GEMM the aligned M−1 rows + one CLBlast Gemv for the last row.
  DiT 32.0 → 29.9 s (cool). Identical math.
- **B15 serve pipelining (kept)**: reader thread + one-ahead T5 prefetch on a
  worker thread while the GPU runs the current generation. Queued prompts:
  gen 2+ = 52.1 s (T5's ~2.9 s fully hidden). Interactive single prompts
  behave as before.
- State after round 4 (warm caches, cool device): one-shot ~55.5 s
  (DiT 29.9 + VAE 22.3 + T5 2.7), serve pipelined ~52 s/gen, 4-step 41.5 s
  (DiT 14.9). Thermal band: DiT 29.9 (≤31 °C) to ~34 s (≥34 °C battery).
- int8/int4: NOT a GPU speed lever on Adreno 6xx — no integer dot-product
  path from OpenCL (unpack→fp16 mad, zero ALU gain) and the model is
  compute-bound, not bandwidth-bound (weight traffic ≈ 53 ms/step vs 3.7 s
  step). Real quant wins: T5-on-CPU int8 NEON (~−1 s) and weights footprint
  (950 → ~500 MB) — app-side value, queued separately.

## Notes

- The activation buffer pool in this port exists for correctness, not just speed:
  per-call clCreateBuffer/clReleaseMemObject churn fragmented the Adreno heap and
  caused intermittent CL_OUT_OF_RESOURCES mid-denoise. (As of 2026-07-08 the
  decoder is routed through it too — `NNOPT_DEC_POOL=0` reverts.)
- Remaining known levers, unexplored: texture/image objects for hot GEMM/conv
  weight tiles (guide §6.2; musicgen precedent), cl_qcom_recordable_queues to
  replay the ~540-enqueue DiT step (guide §9.1.3), a bespoke FFN GEMM kernel
  (the FFN GLU is ~63% of DiT FLOPs and CLBlast-untunable per above), and a
  4-step sampler A/B (halves the DiT wall at some quality cost).
- Post-campaign wall: DiT 32.0 s (≈24 s of it CLBlast GEMMs at stock params +
  dispatch gaps; event profiling undercounts CLBlast — trust wall deltas, not
  the `clblast_gemm` ledger line), VAE 23.7 s (conv GEMM-bound), T5 2.7 s.
