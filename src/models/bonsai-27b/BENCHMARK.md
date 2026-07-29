# Benchmark log — Bonsai-27B on Adreno 840 (Q1_0, 1-bit)

All numbers below are from an **Adreno 840, 12 CU, 5542 MB** (Snapdragon 8-Elite-class).
Nothing here was measured on another part.

That constraint is not pedantry. This port shipped two constants tuned on an Adreno 620 —
a 1-CU device that ranks these kernels differently — and both were wrong for the 840:
`GEMM_MT=4` (should be 2) and the "x-reuse beats occupancy" verdict behind `Q1_WG=128`
(should be 64). A stale measurement from the wrong device is worse than no measurement,
because it reads as authoritative. **Tune on the 840 only.**

## Protocol

These numbers were produced by on-device measurement tooling reachable as chat messages
(`/bench` for a full report, `/preset N` to switch tunables, `/ab A B` for an interleaved
comparison), because the target device is normally only reachable through the app — not with
`adb`. **That tooling was removed before release**: it carried ~1,000 lines of host code and
28 extra kernels that the driver JIT-compiles at startup, none of which a user's device should
pay for.

It is worth rebuilding — as an env-gated report rather than chat commands — before the next
tuning round, because the two things it measured are exactly what makes tuning this port
tractable: a per-run peak-bandwidth figure to normalise against, and an interleaved A/B.

### Measurement noise — read this before comparing runs

Three consecutive runs of the *same* build measured peak read bandwidth at
**53.7, 55.0 and 51.3 GB/s**. The device drifts ±7% with thermal state and battery, while
the changes under test are worth 3–6%. Absolute tok/s from separate runs therefore cannot
resolve them: a real gain and a thermal dip are the same number.

Two rules follow, and the ladder below obeys both:

1. **Quote % of measured peak, not tok/s**, when comparing across runs.
2. **Interleave A,B,A,B,… in one process for anything under ~10%**, and compare the ratio of
   *medians* — median because a thermal spike is a one-sided outlier that would drag a mean
   toward whichever arm caught it. Sequential runs cannot resolve a change that small.

## Headline (2026-07-28)

> **Two changes landed after this table was taken** — the projection fusion and the removal of the
> per-token blocking readback. Together they moved a real reply (not the pipelined bench figure)
> from 6.9 to **7.5 tok/s**, and prefill from 8 to **10 tok/s**. The bench numbers below were not
> re-taken afterwards, because the tooling that produced them was removed for release.

| metric | value |
|---|---|
| decode | **8.0 tok/s**, 125 ms/token |
| weight traffic | 30.5 GB/s = **55% of measured peak** (55.0 GB/s) |
| prefill | 9.7 tok/s batched (1.28× over sequential) |
| TTFT, 128-token image | ~21 s (vision tower ~12.6 s + LM prefill) |
| weights resident | 3618 MB, 1.125 bit/weight, never dequantized |
| model load | 8.4 s |
| dispatches | 1333/token, 11–24% of the token depending on thermal state |
| vision correctness | cosine **1.000000** vs `transformers` `Qwen3VLVisionModel` |

### Where a token goes (GPU hardware timestamps, guide §4.5.2)

```
wall 159.0 ms | GPU busy 144.8 ms (91%)
  gemv_sk_fused  496 calls  124.1 ms  86%   250.3 us/call
  delta_net       48          6.7 ms   5%
  rms            161          5.3 ms   4%
  gemv_head        1          4.9 ms   3%
  xsum           257          1.6 ms   1%
  conv1d/add/argmax/swiglu/norm_gate      <1% each
```

**Decode is the GEMV and nothing else.** Everything outside it sums to ~20 ms. This matters
because the *dispatch-count* census ranks these very differently — xsum is 257 launches, 19%
of launch overhead, but 1% of GPU time. Optimising it saves 1.6 ms. Rank work by the
hardware-timestamp table, not the dispatch census.

## Roofline

3.795 GB of weights are read **per token** — the model is dense, so there is no way around it.

| target | required sustained | % of 53 GB/s peak |
|---|---|---|
| 8.0 (today) | 30.4 GB/s | 55% |
| 9 | 34.2 | 62% |
| 10 | 38.0 | 69% |
| 12 | 45.5 | 83% |

The best rate any single matrix shape has ever hit on this device is **39.6 GB/s**. So 10 tok/s
requires every matrix to run at 96% of the best any one of them has managed, and 12 tok/s
requires beating the best measured shape by 15%. **12 is not reachable by kernel tuning.**
Past ~9–10 the only lever is not reading all 3.795 GB per token — speculative decoding, which
the 1.28× batched GEMM finally makes viable.

## Optimization ladder

Each step was re-verified against a reference before being ranked — the sweep refused to rank a
configuration whose output did not match, which is what stopped a kernel that silently skipped
rows from reading as an 8% speedup.

| change | result | note |
|---|---|---|
| `image1d_buffer` weights (guide §6.2) | ships | texture L1 instead of the buffer path; zero-copy alias of the same allocation |
| split-K | **20.4 → 34.5 GB/s** | parallelism was set entirely by N; `ffn_down` (N=5120) launched 10 workgroups on 12 CUs |
| fused split-K reduce | removed 27% of a token | `sk_reduce` ran 496×/token for 8.3 ms and an *empty* launch costs the same — the pass paid for its own dispatch and nothing else. Last group to arrive now reduces in place |
| 2D dispatch (guide §8.12) | ships | removed `gid % ng` / `gid / ng`; hardware carries both indices |
| `select()` over branches (§8.5) | ships | branch-free tail handling |
| `-cl-fast-relaxed-math` | ships | |
| fp32 lm_head store | correctness | fp16 logits at ‖·‖≥200 collide on argmax; first few tokens match the reference then diverge |
| **`Q1_WG` 128 → 64** | **1.16×** (0.37 → 0.32 ms) | ~1.10× e2e; this shape is 64% of weight traffic. Inverted the Adreno-620 verdict, as predicted |
| **`GEMM_MT` 4 → 2** | **1.21×** (12.82 → 10.56 ms) | batched prefill 1.03× → **1.28×**; ~5 s off TTFT |
| `SK_MAX_SPLITS` 8 → 32 | +1% of peak | 8 was a ceiling the probe never got past, not an optimum |
| `SK_TARGET_GROUPS` 160 → 320 | +1.6% | interleaved A/B; 96 measured worse (129.2 ms), so 320 is a peak not a slope |
| **projection fusion** | 1333 → **1093** dispatches | qkv+z+alpha+beta (L), q+k+v (F), gate+up (MLP) — same input each, so 240 of 496 GEMV dispatches were pure launch cost. Folds the two N=48 matrices, which no split-K could fill 12 CUs with, into wide ones. Bit-identical output; GEMV GPU time 124.1 → 108.0 ms |
| **no per-token readback** | reply 6.9 → **7.5** | `read_token()` was a blocking 4-byte `clEnqueueReadBuffer`, so the pipeline drained once per token. `q1_row_gather3_dev` takes the token from a device buffer instead, so step i+1 is enqueued before step i is read. Argmax double-buffered so the in-flight step cannot clobber the token being read |

### Measured and rejected

| change | result | why |
|---|---|---|
| 8-bit LUT GEMV | **22× slower** (4210 vs 187.7 µs/call) | byte-indexed gather thrashes cache once split-K multiplies workgroups 5–8×. Standalone it is 0.96× — the LUT and split-K are incompatible, not the LUT alone |
| 16-bit ALU, `half8`/`half4` (§7.2.4) | 0.47 / 0.45 ms vs 0.32 | the predicted 2× packed-fp16 rate does not materialise here |
| float-FMA masking | 22.07 ms vs 0.32 | catastrophic |
| software pipeline (prefetch u+1) | 0.54 vs 0.59 @wg32, loses @wg64 | helps only where occupancy is already poor |
| x staged in `__local` | 0.59 vs 0.59 | L2 broadcast across a workgroup is already free |
| 8 rows/work-item | 0.68 vs 0.59 | register pressure |
| fp16 activations (whole model) | slower | `vload_half4` converts to float anyway, so it buys traffic and pays conversion |
| fp16 vision tower | cosine 0.954682 | CLBlast Hgemm **accumulates** in half; error compounds across 27 blocks. Also slower on the 840 |
| `q1_gemm` (local-staged GEMM) | 175.82 ms, 2944 B/wi private | spills to global. `q1_gemm_r1` does the same work in 8.31 ms at 672 B/wi |
| `GEMM_MT` 16 | 96.76 ms, 2340 B/wi private | same spill cliff |

**Always print `CL_KERNEL_PRIVATE_MEM_SIZE`.** Three of the rejections above are one failure
mode — private memory on Adreno *is* global memory, so a spilled accumulator turns every
update into a round trip. It is invisible in the source and obvious in that one number.

## Bugs the gates caught

- **Vision rope used raster order instead of merge-block order.** Cosine 0.93 vs the real
  `Qwen3VLVisionModel` — high enough to look like precision loss, low enough to describe the
  wrong image. Only a per-stage diff against the actual `transformers` module found it; a
  hand-transcribed oracle would have reproduced the same mistake.
- **Recurrent state leaked across turns.** `rec_`/`convs_` are accumulators with no position
  index, so restarting at pos 0 clears the KV cache but not them. Turn 1 correct, turn 2 a
  fluent degenerate loop. Guarded by `scripts/check_vlm_session.sh`.
- **Hardcoded 64-row offsets in seven kernels.** Correct only at wg=64; at any other size they
  wrote a subset of rows, which reads as a speedup.
- **The GEMV sweep silently stopped working.** `q1_gemm` was added to `q1_gemv.cl`, the sweep's
  hand-rolled build options never gained `-DGEMM_MT`, and one kernel failing to compile fails
  the whole program — every row printed "program build FAILED" and the table came out empty.
- **The sweep's correctness column was vacuous.** `x` was filled with zeros, so the reference
  output was all zeros and every candidate matched. Now a varied pattern, with the output
  poisoned before each candidate so an unwritten row cannot inherit a correct value.
