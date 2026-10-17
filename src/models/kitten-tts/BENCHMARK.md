# KittenTTS Nano 0.1 — Benchmark (Motorola Razr 2020, Adreno 620 v2)

All numbers are the reference utterance (70 phonemes → 159 frames → 95,400
samples = **3.975 s of 24 kHz audio**), `--release` fp32, on-device, GPU-busy
≥99.9%. Correctness held throughout: envelope corr **0.9992**, log-spectrum
**0.9998** vs the validated baseline; durations integer-identical to the oracle.

## Optimization ladder — 26.18 s → 3.47 s (7.5×), RTF 6.59 → 0.87

| step | total | RTF | key change |
|---|---|---|---|
| baseline (release + fast-math) | 26.18 s | 6.59 | |
| tiled conv1d | 19.35 s | 4.87 | LDS-tiled, T_REG=4, CO_TILE=16, CIN_CHUNK=16 |
| tiled ConvTranspose | 17.86 s | 4.49 | branch-free tap-step |
| workgroup InstanceNorm | 17.31 s | 4.35 | one workgroup per channel (16× on that kernel) |
| hand-tiled GEMM | 17.55 s | 4.41 | replaces CLBlast (not the bottleneck) |
| **flash attention** | **6.66 s** | **1.68** | **the 74% was here** — bert 13 s → 2.1 s |
| vload4 conv1d LDS read | 4.65 s | 1.17 | 4 consecutive outputs = one float4 load |
| vload4 GEMM Bs read | 4.08 s | 1.03 | |
| ConvTranspose vload4 weights | 4.01 s | 1.01 | |
| bert-attention float4 | 3.71 s | 0.93 | scalar-order reduction (durations are hypersensitive) |
| GEMM cooperative-load vectorization | **3.47 s** | **0.87** | vload4/vstore4 global→LDS (load-bound stage) |

**Final kernel ranking** (3.63 s of events): conv1d 1622 ms (45%) > gemm 812
(23%) > convT 260 (7%) > dql 107 > attn 74 > instancenorm/adain/snake ~38 each.

## Streaming (`--serve-stream`, fp32)

Per-clause synthesis, on-device espeak G2P + chunking. Sustained clauses stream
faster than real-time; a short cold opener does not amortize.

| chunk | text | audio | synth | RTF |
|---|---|---|---|---|
| 0 | "Hello there." | 1.00 s | 1.27 s | 1.27 (TTFB) |
| 1 | "I am streaming speech in real time on the GPU," | 3.38 s | 2.94 s | **0.87** |
| 2 | "chunk by chunk, as the words arrive from the language model." | 3.60 s | 3.01 s | **0.84** |

## Findings (device truth)

- **fp16 is a dead end here.** Kernels are issue-bound on load *instructions*;
  fp16 halves bytes not instructions and adds upconvert cost. fp16 build measured
  conv1d 1817 ms (vs fp32 1625), RTF 1.1–1.5 — slower, not faster.
- **vload4 on the LDS/global read** was the highest-leverage single technique
  (1.7–5.7× per kernel): conv1d, gemm, convT, and attention were all issue-bound
  on scalar loads of consecutive elements.
- **>4 accumulators regress on Adreno**, even for GEMM (a 4×4 microtile regressed
  3.5×).
- **Never reorder a reduction that feeds duration/F0** — a float4 tree-reduction
  of the attention dot shifted the sum ~1e-6 and flipped a duration (159→160
  frames). Vectorize *loads* freely; keep the reduction in scalar lane order.
- **The remaining lever for RTF ~0.5 is `cl_qcom_dot_product8`** (int8 dot8) — the
  model is int8 at source. Not yet implemented (see kokoro-82m for the pattern).
