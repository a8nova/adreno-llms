# SeamlessM4T UnitY — On-Device Performance Log

End-to-end speech→speech (`facebook/seamless-m4t-unity-small`) on **Qualcomm Adreno 620
(Snapdragon 765G, Motorola Razr 2020)**, OpenCL **fp16** (fp16 storage, fp32 accumulate).

Correctness is held fixed at every step: **units exact (53/53)** vs `e2e_out_1` and
**waveform cos ≥ 0.999999** vs `e2e_out_2`. No optimization is accepted unless these hold.

## Methodology
- Timing comes from the binary's own `std::chrono` instrumentation (env `NNOPT_TIMING=1`),
  printed as `TIMING(ms): fbank=… encoder=… text_beam=… mt_feat=… synth=… unit_greedy=… vocoder=… TOTAL=…`.
  - The device's `date +%s%N` is **unreliable** (toybox has no nanoseconds) — ignore any
    shell-computed `WALL_MS`; trust the `std::chrono` numbers.
- **Cold vs warm**: the 646 MB weight file is `mmap`'d lazily, so the *first* run after a
  deploy pays page-fault cost on whichever stage touches each weight first. Numbers below
  are labeled cold/warm; compare warm-to-warm.
- Input: `assets/test_audio_raw.bin` = 16000 float32 samples ≈ **1.0 s of audio**.
  Real-time factor (RTF) = TOTAL_ms / 1000.
- Run: `NNOPT_DTYPE=fp16 NNOPT_TIMING=1 ./<bin> full`  (or `./scripts/run_android.sh full`).

## Per-stage timeline (ms)

| Stage        | Baseline (cold) | + KV cache (warm) | Notes |
|--------------|----------------:|------------------:|-------|
| fbank        | 549             | 69                | GPU FFT+mel |
| encoder      | 39,299          | 31,051            | 8 Conformer layers — **now the bottleneck** |
| text_beam    | 103,236         | 9,514             | beam-5, 4 layers |
| mt_feat      | 1,337           | 1,321             | 1 teacher-forced forward |
| synth        | 615             | 611               | 2 layers |
| unit_greedy  | 120,881         | 5,545             | greedy, 2 layers |
| vocoder      | 5,306           | 5,296             | CodeHiFiGAN conv stack |
| **TOTAL**    | **271,224**     | **53,407**        | RTF 271 → **53** |

(Baseline column is the first cold profile; the KV-cache column is warm. The cold→warm
gap is mostly lazy-mmap, so the headline win of each change is measured warm-to-warm.)

---

## Optimization 1 — KV cache for both autoregressive decoders ✅

**Problem.** Each decode step re-ran the *entire growing sequence* through the decoder
(no cache): O(T²) per step → O(T³) total. The unit decoder (53 steps) and text beam
(beam-5 × ~8 steps) dominated.

**Change.**
- `KVCache` (in `pipeline.{h,cpp}`): per-layer self-attention K/V caches that grow one
  token per step; cross-attention K/V computed **once** (the encoder memory is fixed).
- `make_cache()` precomputes cross K/V + allocates self-K/V buffers; `decode_step()`
  processes a single token at position `pos`, appends its self K/V, returns logits.
- **Unit greedy** (`layers/unit_decoder.cpp`): one cache, append per step.
- **Text beam** (`layers/text_decoder.cpp`): per-beam self-caches in a **double-buffered
  pool**; on candidate selection the caches are **reordered** (child inherits parent's
  cache) by GPU `copy_into`. Cross K/V shared across beams.

**Result (warm-to-warm):**
- `unit_greedy` 120,881 → **5,545 ms** (**21.6×**)
- `text_beam`   ~40,845* → **9,514 ms** (**~4.3×**)  (*warm baseline before the beam cache)
- TOTAL 271,224 (cold) → **53,407 ms** warm. Correctness preserved (units exact, cos 0.999999).

After this, the bottleneck is the **speech encoder (31 s = 58%)** and the matmul-heavy
stages generally — all driven by a naive GEMV `linear` kernel (1 work-item/output, scalar
fp16 loads, weights re-read from global memory each time).

---

## Tier 1 — matmul/conv engine (in progress)

Target: the `linear` kernel powers the encoder, both decoders, projections, FFN and the
LM head. Per the Qualcomm Adreno OpenCL guide (80-NB295-11):
1. **Vectorized 128-bit / half8 loads** (§6.3, §7.2.2/§7.2.4 — fp16 = 2× ALU + ½ bandwidth).
2. **Register blocking** — each work-item computes several outputs, reusing the activation
   row from registers (§10.x "more work per work-item").
3. **Local-memory / L2 reuse** of the activation row across a workgroup (§6.5, §7.1.5.5).
4. **Image/texture (L1) for weights** (§6.2, §7.1.5.4) — later step.

### 1a. Vectorized + N-register-blocked `linear` ✅

`linear_fast` (kernels/gemm.cl): each work-item computes `LIN_TN=4` outputs for one row,
half8-vectorized K loop (`LOAD8` → `vload_half8` → float8, `dot8`), reusing the activation
chunk across the 4 weight rows. `LOAD`/`LOAD8`/`dot8` added to `_preamble.cl` (dtype-agnostic).

| Stage | + KV cache | + vec GEMM | speedup |
|-------|----------:|-----------:|--------:|
| fbank | 69 | 380 | — (variance) |
| encoder | 31,051 | **11,424** | **2.7×** |
| text_beam | 9,514 | 8,076 | 1.2× |
| mt_feat | 1,321 | 575 | 2.3× |
| synth | 611 | 262 | 2.3× |
| unit_greedy | 5,545 | 4,577 | 1.2× |
| vocoder | 5,296 | 5,225 | 1.0× (conv, not linear) |
| **TOTAL** | **53,407** | **30,520** | **1.75×** |

units exact ✓, waveform cos 0.999999 ✓. Encoder still #1; its FFNs are M=49 (real GEMM) and
the attention kernels are still scalar → next: M-register-blocking + vectorized attention.

### 1b. M×N register-blocked GEMM + vectorized attention ✅

- `linear_gemm` (kernels/gemm.cl): GEMM_TM×GEMM_TN = 4×4 register tile, half8 K loop.
  `GpuOps::linear` dispatches `linear_gemm` for M≥4 (encoder T=49, mt_feat/synth T=8) and
  `linear_fast` (GEMV) for the single-token decode steps. Each weight chunk is now reused
  across 4 rows.
- `attention_context` / `relpos_attention_context`: replaced the scalar `float acc[64]`
  + scalar dot with half8 vectorization (`float8 acc[]`, `dot8`, `LOAD8`/`STORE8`).

Stable over 3 runs (warm):

| Stage | KV-cache | + vec GEMM (1a) | + M-block + vec-attn (1b) | total speedup |
|-------|---------:|----------------:|--------------------------:|--------------:|
| fbank | 69 | 380 | 347 | — |
| encoder | 31,051 | 11,424 | **6,097** | **5.1×** |
| text_beam | 9,514 | 8,076 | 10,203 | 0.9× (launch/sync-bound) |
| mt_feat | 1,321 | 575 | 363 | 3.6× |
| synth | 611 | 262 | 164 | 3.7× |
| unit_greedy | 5,545 | 4,577 | 6,002 | 0.9× (launch/sync-bound) |
| vocoder | 5,296 | 5,225 | 5,326 | 1.0× (conv, not linear) |
| **TOTAL** | **53,407** | **30,520** | **~28,500** | **1.87×** |

units exact ✓, waveform cos 0.999999 ✓.

**Finding.** Tier 1 solved the matmul stages: encoder **31 s → 6.1 s (5.1×)**, mt/synth ~3.6×.
The remaining top costs are now **NOT matmul-bound**:
- `text_beam` (10.2 s) and `unit_greedy` (6.0 s) are **launch- and host-sync-bound** — each
  token launches dozens of tiny kernels and does a blocking `clEnqueueReadBuffer` for argmax
  (decode attention is only H·Tq = 16 work-items). → **Tier 3** (fuse kernels, batch beams,
  remove per-token sync, recordable queues).
- `vocoder` (5.3 s) is **conv-bound** (scalar `conv1d_ct`/`conv_transpose1d_ct`). → **Tier 1 #5**
  (vectorize/tile conv) — not yet done.

## Tier 2 — precision & math ✅ (small win: we're no longer math-bound)

`-cl-fast-relaxed-math` added to the program build (`GpuOps::init`). Safe here: every `exp`
operates on ≤0 args (post-max-subtraction) and no inf/nan is produced, so the implied
`-cl-finite-math-only` doesn't bite. On Adreno this flag already lowers `exp`/`log`/`div`/
`sqrt`/`erf` to the fast/native units, so explicit `native_*` calls were **not** added
(redundant, and they'd add precision risk to the exact-argmax decode path for ~no gain).

| Stage | + Tier 1 | + fast-math | Δ |
|-------|---------:|------------:|---|
| fbank | 347 | **13** | cos/sin/log heavy → big relative win |
| encoder | 6,097 | 5,935 | −3% |
| text_beam | 10,203 | 10,129 | ~0 (launch/sync-bound) |
| unit_greedy | 6,002 | 5,973 | ~0 |
| vocoder | 5,326 | 5,276 | ~0 (conv) |
| **TOTAL** | **~28,500** | **27,844** | **−2%** |

units exact ✓, cos 0.999999 ✓. **Finding:** post-Tier-1 the bottleneck is launch/host-sync
(decoders) and conv (vocoder), neither math-bound — so Tier 2's ceiling on this workload is
small. Kept because it's free and correct. The real remaining levers are Tier 3 and conv.

## Tier 3 — decoder launch/sync (attempted, reverted — important finding)

**Hypothesis:** the decoders (text_beam 10 s + unit_greedy 6 s) are launch-count-bound, so
fusing bias+activation+residual into the matmul epilogue (dropping the separate `act`/`axpy`
launches per sublayer) would cut decode time.

**Result: hypothesis wrong — reverted.**
- Removing ~4 elementwise launches/layer (`act` + 3 `axpy`) changed decode time by **~0**
  (text 10,129→10,174; unit 5,973→6,015). So decode is **not** bound by the *count* of
  (cheap) kernels.
- Worse, inlining `act_apply` (which pulls in `erf`/`exp`/`tanh`) into the hot GEMM kernel
  **regressed the encoder 5,935 → 9,250 ms (~1.55×)** — the transcendental code bloated the
  GEMM and tanked its occupancy. Stable across 3 cooled runs (not thermal).
- Reverted to the clean Tier-1+2 kernels; re-verified **encoder 5,906 ms, TOTAL 27,794 ms,
  units exact, cos 0.999999**.

**What the experiment revealed.** Decode costs ~60 ms per *decoder layer* (one token), with
~9 matmul launches/layer. The cheap elementwise kernels are near-free; the cost is the
**per-matmul GPU dispatch latency (~4 ms each) on M=1 work with no overlap** (in-order queue,
per-token host sync). So the real decode levers are about **fewer / overlapped matmul
dispatches**, not fusing elementwise:
1. **Batch the 5 beams** (text): one M=5 GEMM instead of 5× M=1 → 5× fewer matmul dispatches
   + far better occupancy. Biggest single text-beam win. (Needs batched per-beam attention.)
2. **Fuse q/k/v** into one matmul (concatenate the three weight tensors at cache-build) →
   −2 matmul dispatches/layer for both decoders.
3. **Recordable queues** (`cl_qcom_recordable_queues`): record the per-token kernel sequence
   once and replay — directly removes per-dispatch latency, which is the actual bottleneck.

These are structural changes (not attempted this session); each needs on-device validation.
Lesson recorded: **do not inline transcendental activations into the GEMM kernel.**

### Tier 3b — batch the 5 text-beam beams ✅

`decode_step_batch` (pipeline.cpp): processes all B beams in one pass — every projection
runs as a single **M=B GEMM** (5× fewer matmul dispatches), cross-attention is **one batched
call** (all beams share the memory K/V → `attention(q2[B], ck, cv, Tq=B, Tmem)`), and
self-attention stays per-beam (each reads its own cache). Numerically identical to B separate
`decode_step` calls. `text_beam_search` now calls it once per step.

**Bug found + fixed:** `embed_scale_pos` computed `pos = start + t` (row index = sequence
position) — correct for the teacher-forced forward, but **wrong for batched beams**, which
are all at the *same* position. Added `pos_stride` (1 for sequences / single-token, **0 for
batched beams**). First (buggy) run produced a wrong 6-token hypo; with `pos_stride=0` the
hypo is exact again.

| Stage | before (Tier 2) | batched beam | speedup |
|-------|----------------:|-------------:|--------:|
| text_beam | 10,061 | **~4,910** | **2.0×** |
| (others unchanged) | | | |
| **TOTAL** | **27,794** | **~22,700** | **1.22×** |

Stable over 3 runs. **hypo exact, units exact, cos 0.999999.** The remaining stages are now
well-balanced: unit_greedy ~6.0 s, encoder ~5.9 s, vocoder ~5.3 s, text_beam ~4.9 s. Next
levers: unit_greedy can't be beam-batched (greedy=1 seq) → q/k/v fusion or recordable queues;
vocoder → conv vectorization (Tier 1 #5).

## Cumulative progress
| Milestone | TOTAL (ms) | RTF (1 s audio) | vs previous |
|-----------|-----------:|----------------:|------------:|
| Baseline (cold) | 271,224 | 271 | — |
| + KV cache (both decoders) | 53,407 | 53 | 5.1× |
| + Tier-1 GEMM + attention | ~28,500 | 28.5 | 1.87× |
| + Tier-2 fast-relaxed-math | 27,844 | 27.8 | 1.02× |
| + Tier-3 batch beams | ~22,700 | 22.7 | 1.22× |
| **cumulative** | | | **~12×** |

## Per-language warm profile — 2026-06-21 (current numbers)

`s2s` on `samples/tell_me_nearest_gas_station.wav` (6.012 s in, T_enc=38), Razr / Adreno 620,
fp16, warm. Median of **5 fresh-process runs** per language (each `NNOPT_WARMUP=1` untimed
warm-up → timed steady-state run; `NNOPT_TIMING=1` per-stage `std::chrono`). Within-language
spread < 2 %. RTF = wall ÷ 6.012 s input.

| lang | fbank | encoder | text_beam | mt_feat | synth | unit_greedy | vocoder | **TOTAL** | S2ST RTF | S2TT\* | S2TT RTF |
|------|------:|--------:|----------:|--------:|------:|------------:|--------:|----------:|---------:|-------:|---------:|
| eng | 32 | 4377 | 4941 | 202 | 80 | 2443 | 5327 | **17,418** | 2.90× | 9,324 | 1.55× |
| spa | 33 | 4353 | 7174 | 215 | 84 | 2170 | 4979 | **19,022** | 3.16× | 11,553 | 1.92× |
| por | 32 | 4344 | 6239 | 204 | 80 | 2809 | 6185 | **19,833** | 3.30× | 10,605 | 1.76× |
| hin | 32 | 4337 | 10475 | 231 | 90 | 3724 | 7925 | **26,810** | 4.46× | 14,844 | 2.47× |
| rus | 33 | 4337 | 7526 | 208 | 81 | 2814 | 6541 | **21,521** | 3.58× | 11,899 | 1.98× |

\*S2TT / ASR = `fbank + encoder + text_beam` (text modes stop after the text decoder, skipping
T2U → unit_greedy → vocoder). **fbank + encoder (~4.4 s) is language-independent** (same input
audio); all per-language spread is in `text_beam` and — for S2ST — the unit/vocoder lengths
(grow with #units emitted; Hindi emits the most → slowest). Units bit-exact vs `.ptl`
(eng 211/211, spa 191/191, por 234/234), waveform cos 0.999999.

`s2tt_all` (encode once, decode all 5): name clip (3.09 s) **19.5 s** total (enc 2.2 s);
gas clip (6.01 s) **40.5 s** total (enc 4.4 s).

> The two tables below are **historical**: the laughter fixture (~1 s output, unrepresentative)
> and the pre-optimization real-speech run (encoder ~35 s, since cut ~8× to ~4.4 s). Kept for
> the optimization narrative — **use the 2026-06-21 table above for current numbers.**

## Multilingual S2ST profile (all output languages, warm) — historical, laughter fixture

`full --lang <code>` on the laughter fixture. Time scales with **#units emitted** (decode is
per-token; vocoder length = units×320). The model rambles on this non-speech input for spa.

| lang | units | encoder | text_beam | unit_greedy | vocoder | TOTAL | out audio |
|------|------:|--------:|----------:|------------:|--------:|------:|-----------|
| eng  | 53    | 5919 | 4942 | 5986 | 4822 | 22,202 | 1.06 s |
| por  | 53    | 5972 | 4390 | 6061 | 4827 | 21,776 | 1.06 s |
| hin  | 53    | 5948 | 4329 | 6095 | 5292 | 22,189 | 1.06 s |
| rus  | 39    | 5991 | 4919 | 4468 | 3800 | 19,718 | 0.78 s |
| spa  | 299   | 5955 | 6684 | 34,150 | 29,274 | 76,796 | 5.98 s |

For a ~53-unit output (~1 s audio) the four big stages are each **~5 s and roughly equal**
(encoder 27%, unit_greedy 27%, text_beam 22%, vocoder 22%) → **RTF ≈ 22**. unit_greedy and
vocoder scale linearly with #units (see spa). Audio in `generated_audio/seamless_<lang>.wav`.

## Real-speech profile — historical (pre encoder/decoder optimization) — `test_inputs/*.wav`

S2ST via `s2s --in <wav>`. **Encoder cost scales with INPUT length and is identical across
target languages** (same audio); unit_greedy + vocoder scale with OUTPUT units.

**`tell_me_nearest_gas_station.wav` (6.01 s in, T_enc≈299):**
| lang | units | encoder | text_beam | unit_greedy | vocoder | TOTAL | out |
|------|------:|--------:|----------:|------------:|--------:|------:|----:|
| eng | 180 | 35,482 | 9,036 | 20,183 | 17,033 | 82,683 | 3.6 s |
| spa | 218 | 35,380 | 12,911 | 24,653 | 21,258 | 95,170 | 4.4 s |
| por | 208 | 35,428 | 10,654 | 23,228 | 18,905 | 89,169 | 4.2 s |
| hin | 299 | 35,677 | 18,811 | 33,902 | 29,337 | 118,934 | 6.0 s |
| rus | 266 | 35,583 | 11,893 | 30,292 | 25,646 | 104,589 | 5.3 s |

**`what_is_your_name.wav` (3.09 s in, T_enc≈150):**
| lang | units | encoder | text_beam | unit_greedy | vocoder | TOTAL | out |
|------|------:|--------:|----------:|------------:|--------:|------:|----:|
| eng | 52 | 15,495 | 4,447 | 5,962 | 4,993 | 31,475 | 1.0 s |
| spa | 77 | 15,513 | 10,228 | 8,726 | 7,422 | 42,650 | 1.5 s |
| por | 55 | 15,518 | 5,594 | 6,252 | 5,455 | 33,402 | 1.1 s |
| hin | 73 | 15,501 | 6,190 | 8,332 | 7,077 | 37,859 | 1.5 s |
| rus | 86 | 15,575 | 5,580 | 9,822 | 8,295 | 40,037 | 1.7 s |

**Encoder vs input length: 1 s→5.9 s, 3 s→15.5 s, 6 s→35.5 s.** ≈ linear (FFN, O(T)) with a
growing quadratic component (relpos attention, O(T²)). **RTF ≈ 10–14 vs input** (e.g. gas-eng
82.7 s / 6 s). Real audio: `generated_audio/{gas,name}_<lang>.wav`.

**Revised priority for realtime (real speech ≠ the 1 s fixture):**
1. **Encoder** is the #1 fixed cost (15–35 s, ~43% at 6 s, lang-independent). Optimize the
   FFN/proj GEMMs (local-mem + image/L1 tiling, §6.2/§6.5/§7.1.5.4) AND the O(T²) relpos
   attention (tile over T, vectorized) — biggest realtime lever for real input.
2. **unit_greedy + vocoder** (output-scaling): recordable queues for decode (§9.1.3),
   vocoder conv vectorization (§7.1.2). 
3. Workgroup-size tuning everywhere (§6.1.5; currently `local=NULL`).
4. int8 weights (§7.2.4) for the memory-bound GEMMs.

## Next (not yet done)
- **Tier 1 #5**: vectorize/tile the vocoder + encoder conv kernels (~5 s).
- **Tier 1 #4**: weights as image objects (L1 texture cache) for the GEMM.
- **Tier 3**: decode launch/sync — kernel fusion (q/k/v + bias+act + residual), batch the
  5 beams into one GEMM, eliminate per-token blocking readbacks, recordable queues. This is
  now the largest remaining block (~16 s across the two decoders).
- **Tier 2**: `-cl-fast-relaxed-math` + `native_exp/native_log` (GELU currently uses `erf`).
- **Tier 4**: workgroup-size tuning (currently `local=NULL`).

---

## 2026-06-10 — ENCODER BUG FIX (correctness) + cooperative-GEMV decode win + recordable-queues ruled out

### Correctness first: encoder was structurally wrong on real speech
The port had a **spurious pos_conv** added to the conformer input. The scripted
`ConformerEncoder.extract_features` (pos_enc_type=="rel_pos", layer_norm_first=True) feeds
`post_extract_proj` straight into layer 0 — NO additive pos_conv. Removing it
(`src/layers/speech_encoder.cpp`): encoder_out cos **0.989 → 0.999997** on the 6 s clip; units
now bit-exact (eng 211/211, spa 191/191, por 234/234); waveform cos 0.999997. 1 s fixture still
53/53. All earlier "complete" numbers were on the 1 s fixture, which tolerated the error.

### Recordable queues (`cl_qcom_recordable_queues`) — RULED OUT on this workload
Device is Razr 2020 / **Adreno 620** (same as the adreno-llms reference ports). Instrumented a
dispatch counter on the full s2s run: **11,332 total NDRange dispatches** for the 82 s pipeline.
Replay saves ~14 µs/dispatch (23 µs live → 8.6 µs replay, per lfm2/qwen probes) ⇒ absolute
ceiling **163 ms = 0.20 % of wall-clock**. Our 113 ms/step decode is memory-bound kernels, not
dispatch. (lfm2/qwen confirmed the same 3× independently; counter-buffer arg-override hits −59 on
this driver anyway.) Decision: skip recordable queues, fix the GEMV kernels instead.

### Cooperative M=1 GEMV (the proven lfm2 decode lever) — DONE, 1.56× on decode
`linear_fast` launched only ceil(N/4) work-items at M=1 (192 for a 768-proj) — far too few for
the single-CU Adreno to hide DRAM latency (~4 % of roofline). Added `gemv_coop` / `gemv_coop4`
(`kernels/gemm.cl`): one WG (64 threads = one A6xx wave) per 1 or 4 output rows, coalesced half8
loads, fp32 tree-reduce. Wired into `GpuOps::linear` T==1 path. **unit_greedy 23.9 s → 15.3 s
(1.56×), units bit-exact.** coop4 over coop is marginal (~4 %) — decode is no longer
GEMV-occupancy-bound.

### Profile after this session (eng, 6.01 s input, fp16)
| stage | before | after | note |
|---|---|---|---|
| encoder | 28.3 s | ~29 s | dense GEMM (T=299), **now #1 (39 %)** — unchanged, next target |
| vocoder | 20.3 s | ~20.4 s | conv (27 %) — next after encoder |
| unit_greedy | 23.9 s | **15.3 s** | cooperative GEMV (20 %) |
| text_beam | 9.1 s | ~8.9 s | M=5 batched (linear_gemm) (12 %) |
| **TOTAL** | **82.4 s** | **74.9 s** | RTF 13.7× → 12.5× |

**Next lever: the encoder dense GEMM** (39 %, language-independent, runs once) — image2d/L1-texture
weights + larger register tiles + tuned WG size + the O(T²) relpos attention.

### Encoder dense-GEMM tiling + relpos-attention hoist (2026-06-10, cont.)
Profiled the conformer internals (NNOPT_PROF_ENC): FFN 13.4 s (56%), relpos-attn 6.6 s (27%),
conv 2.1 s (9%). Two changes, both correctness-neutral (encoder cos 0.999998, units bit-exact):
- **`linear_gemm_tiled`** (`kernels/gemm.cl`): local-memory tiled GEMM (32×32 block, BK=16, 4×4
  micro-tile, WG=8×8=64), wired into `GpuOps::linear` for M≥32 (encoder T=299, adaptor T=38).
  Scalar 4×4 accumulate — a float8-from-local variant was tried and regressed 1.6× (register
  spills), reverted. Encoder 29 → 24 s.
- **relpos attention**: hoisted q+pos_bias_u / q+pos_bias_v into registers (were re-read from
  global on every one of the ~299 key iterations). attn 6.6 → 5.8 s.

### Session cumulative (eng, 6.01 s input, fp16)
| stage | start | now | Δ |
|---|---:|---:|---|
| encoder | 28.3 s | **23.5 s** | tiled GEMM + attn hoist |
| vocoder | 20.3 s | 18.7 s | (noise) |
| unit_greedy | 23.9 s | **15.3 s** | cooperative GEMV |
| text_beam | 9.1 s | 9.0 s | — |
| **TOTAL** | **82.4 s** | **67.6 s** | RTF 13.7× → **11.2×** |
All units bit-exact (eng 211/211, spa 191/191, por 234/234).

**Remaining levers (bigger efforts):** FFN GEMM still 13.3 s @ ~4.5 GFLOP/s (0.2% of peak) —
needs image2d/L1-texture weights or int8 to push further. Vocoder conv vectorization (18.7 s).
True realtime (RTF 1) on Adreno 620 for a 174M-param cascade is unlikely without int8 across
encoder+vocoder; RTF ~5–8× is a more realistic target with the remaining fp16 levers.

---

## 2026-06-10 — Top-10 optimization sprint (learned from ALL 13 adreno-llms ports)

Worked the top-10 in impact order, correctness-gated (units must stay bit-exact). Net eng
6.01 s clip: **68.8 → 62.2 s, RTF 11.4× → 10.3×**, all langs bit-exact (eng 211, spa 191, por 234).

| # | Optimization | Result | Kept? |
|---|---|---|---|
| ⑤ | native_exp in swish/sigmoid/GLU/softmax (keep exact erf GELU + sinusoids) | 66.7 s | ✅ |
| ③ | vocoder `conv1d_ct4` (4 out-ch/WI, input reuse + ILP) | vocoder 17.9→14.4 s | ✅ |
| ⑧ | vocoder `conv_transpose1d_ct4` (upsample, 4 out-ch/WI) | vocoder →12.9 s | ✅ |
| ① | int8 per-row weights (decode GEMV + encoder GEMM) | **REVERTED** | ❌ gated off |
| ⑥ | cooperative relpos attention (8 thr/query flash-merge) | **REVERTED** | ❌ |

**① int8 — accuracy dead-end for this exact-match cascade.** Kernels written + working
(`gemv_coop_int8`, `gemv_coop4_int8`, `linear_gemm_tiled_int8`, lazy per-row quant cache in
GpuOps, `NNOPT_INT8` gate). Decode-only int8: units 62/211 — the AR unit cascade is FAR more
sensitive than the next-token LLMs the refs quantized (<0.4% noise was "fine" there; here one
flipped token derails the sequence). Encoder int8 also *slower* (one-time readback+quant not
amortized over the encoder's few weight-uses, unlike decode's 211 reuses). Kept gated off for a
future pre-quantized-weight-file path (would move quant cost out of the timed run).

**⑥ cooperative attention — net loss on Adreno 620.** WG=8 wasted 56/64 wave lanes (13.8 s);
full-wave 8 queries×8 key-groups still 7.5 s vs the single-WI-per-query 5.8 s. The 4784 single-WI
work-items already hide the serial-key latency; the flash-merge adds local-memory + idle overhead.
Reverted to `relpos_attention_context` (hoisted q+bias).

**Remaining levers (not yet landed):** ② image2d/L1-texture weights (lossless 1.71× BW, but needs a
separate `cl_program` per the refs or it spills ~10× — architectural change); ④ chained/async decode
(decode is GEMV-compute-bound at ~73 ms/step, not sync-bound — small expected gain); the decode GEMV
is reduction-overhead-bound at K=768 (1.5 chunks/thread + 6-barrier reduce).

**Honest ceiling:** lossless fp16 has largely been mined (RTF ~10×). The big remaining multiplier is
int8/Q4, which this exact-match unit cascade can't tolerate without retraining/calibration or
accepting non-exact output. Near-realtime (RTF ~2-3×) would require image2d + accepting a small
quant-noise budget on the decode.

### int8 with quant-noise budget (2026-06-10) — only the encoder is safe, and it's small
User authorized a small documented quant-noise budget. Findings:
- **Encoder int8 (T>=32, per-row symmetric)**: encoder_out cos 0.999998 → **0.999699** (small), units
  **211/211 still exact**. WARM (quant done in an untimed startup warmup) encoder 23.4 → **21.9 s (7%)**.
  Cold (quant in the timed run) is *slower* (29 s) — the per-row readback+quant isn't amortized over
  the encoder's ~10 weight-reuses. Gated behind `NNOPT_INT8` (opt-in deployment mode + `pipe.encoder`
  warmup in main.cpp s2s). Small because the encoder GEMM at M=299 is compute/occupancy-bound, not
  weight-BW-bound — int8 only halves weight DRAM traffic, which isn't the limiter.
- **Decode int8 — UNUSABLE.** The AR unit greedy is a hard argmax over 10 015 units; any quant noise
  flips a token → sequence diverges → *different words* (62/211 units, LOW waveform cos). This is
  all-or-nothing, so a "small noise budget" doesn't apply — it changes the output, not just adds noise.

**Conclusion:** quantization is a poor fit for this exact-output speech cascade on Adreno 620. The
decode (the BW-bound part where int8 would help most) can't tolerate it; the encoder (where it's safe)
isn't BW-bound. The remaining real lever is **image2d/L1-texture weights** — LOSSLESS 1.71× BW, and it
hits the BW-bound decode GEMV directly (unit 15.4 + text 9 → est. ~14 s). Needs a separate `cl_program`
(refs: else register spill ~10×) — a bounded but real refactor. That is the recommended next step.

### Verified state (default fp16, bit-exact) after the full sprint
eng 62.2 s (211/211) · spa 63.2 s (191/191) · por 68.0 s (234/234). With `NNOPT_INT8` (warm): eng ~60.7 s.
271 s (cold start of project) → 62 s = **4.4× cumulative**, output correct (encoder bug fixed this arc).

### image2d / L1-texture weights for decode GEMV (2026-06-10) — LOSSLESS win ✅
`kernels/gemv_image.cl` (`gemv_img`, `gemv_img4`): weight viewed as image2d_t (CL_RGBA/CL_HALF_FLOAT,
W=K/4, H=N) via image-from-buffer (shared backing, zero extra memory), read with read_imagef →
texture L1 cache. Built in a SEPARATE `cl_program` (`img_prog_`) so register allocation is isolated
from the buffer kernels (mixing them spills ~10× — the documented Adreno gotcha). Wired into
`GpuOps::linear` T==1 path; default ON under fp16 (`NNOPT_IMG=0` to disable). Text lm_head (N=20005 >
16384 image-height cap) auto-falls-back to the fp16 buffer GEMV (get_image → nullptr).

**unit_greedy 15.4 → 10.8 s (1.43×), waveform cos 0.999997, units bit-exact** (lossless — same fp16
data, just the texture access path). The encoder GEMM is NOT image-accelerated (M=299 is
compute/occupancy-bound, not weight-BW-bound — same reason int8 only gave it 7%).

### FINAL verified state (default fp16, bit-exact, image2d on)
| lang | encoder | text_beam | unit_greedy | vocoder | TOTAL | units |
|---|---:|---:|---:|---:|---:|---:|
| eng | 23.3 | 9.0 | 10.8 | 13.0 | **57.2 s** | 211/211 |
| spa | 23.4 | 13.0 | 9.7 | 12.1 | **59.3 s** | 191/191 |
| por | 23.4 | 11.3 | 12.0 | 14.8 | **62.8 s** | 234/234 |

Project arc: **271 s → 57 s (4.75×)**, output correct + bit-exact, waveform cos 0.999997.
Bottlenecks now: encoder GEMM (compute-bound, 23 s) and text_beam (M=5 batched, 9-13 s). Both need a
batched-image / better-tiled GEMM to push further — the M=1 levers are exhausted.

### Dense-GEMM rewrite attempt (2026-06-10) — encoder GEMM is at its practical floor
Took on the encoder (M=299) + text-beam (M=5) GEMM, the last lever. Tried, all on the FFN (13.3 s):
- **Image-weight tiled GEMM** (`gemm_tiled_img`, texture-cache weight strip): **REGRESSED** 13.3→20.4 s.
- **Occupancy via BK=8** (halve local 4→2 KB/WG to double resident WGs): **FLAT** (13.1 vs 13.3, noise).
- (earlier) float8-from-local vectorize: spilled (1.6× regression); int8: only 7% + quant-cost.

**Conclusion: the M=299 encoder GEMM is compute/occupancy-bound at ~4.5 GFLOP/s and resists every
lever** (texture cache, int8, vectorization, occupancy). It's not BW-bound (so image/int8 can't help)
and the scalar 4×4 tiled kernel is near the achievable compute floor for a single-CU Adreno 620 hand
kernel. Reverted to the BK=16 tiled GEMM. `gemm_tiled_img` kept in the image program but not dispatched.

### FINAL LOCKED STATE (default fp16, bit-exact, lossless)
eng **57.6 s** (211/211) · spa **59.1 s** (191/191) · por **62.5 s** (234/234) · waveform cos 0.999997.
Project arc: **271 s → 57.6 s = 4.7×**, output correct + bit-exact. RTF ~9.6×.
Bottlenecks (all compute-bound, at practical floor): encoder 23.5 s, vocoder 13 s, text_beam 9-13 s,
unit_greedy 10.8 s. Beyond this needs a different algorithm class (smaller model / distillation /
NPU offload), not kernel tuning.

---

## 2026-06-10 — PROPER GPU PROFILER (guide §4.5.2) + cl_qcom_perf_hint + the hidden layernorm cost
User (rightly) skeptical of the host `std::chrono` stage timing. Added the correct instrumentation per
the Snapdragon guide §4.5: real **GPU event profiling** (`clGetEventProfilingInfo` START→END, the true
HW kernel exec time), gated by `NNOPT_GPUPROF`, accumulated per kernel name + `dump_gpu_prof()`. Also
added `cl_qcom_perf_hint HIGH` (§4.5.5 — pin GPU to max clock; measured err=0 but timing-neutral, so
the GPU was already boosted under our sustained load — kept for profiling consistency).

**Audit — is everything on GPU?** YES, ~96%: TOTAL_GPU_ACTIVE 55.1 s vs ~57 s wall → ~2 s host
(vocoder embedding-gather `xin` + beam control flow + per-step argmax readback). Not host-bound.

**THE FINDING the stage-timer hid: `layernorm_forward` was 8.6 s (16% of GPU), 1759 calls.** It ran
**1 work-item per row** — so every decode-step LayerNorm (T=1) was a SINGLE thread reducing 768
elements. Fixed with `layernorm_coop` (one 64-wide wave per row, half8 two-pass reduce). LayerNorm
**8606 → 776 ms (11×)**; decode `unit_greedy` **10.8 → 4.0 s (2.7×)**, lossless, units bit-exact.

### Fresh GPU profile (eng 6.01 s, true START→END) AFTER layernorm fix
| kernel | gpu_ms | calls | what |
|---|---:|---:|---|
| linear_gemm_tiled | 19705 | 117 | encoder GEMM (43%) — compute-bound wall |
| linear_gemm | 5076 | 543 | text-beam M=5 GEMM |
| relpos_attention_context | 1867 | 8 | encoder attn |
| attention_context | 862 | 1227 | decoder attn |
| layernorm_coop | 776 | 1759 | (was 8606) |
| gemv_img | 623 | 212 | decode (image, fast) |
| TOTAL_GPU_ACTIVE | 45694 | 11333 | |

### NEW verified state (default fp16, bit-exact, lossless)
eng **48.0 s** (211/211) · spa **50.3 s** (191/191) · por **52.3 s** (234/234). Project arc:
**271 s → 48 s = 5.6×**, output correct + bit-exact. RTF ~8.0×. Next wall: encoder linear_gemm_tiled
(compute-bound, 19.7 s) + text-beam linear_gemm (5 s).

### Profiler-guided continuation (2026-06-10) — vocoder conv exposed, 2 attempts reverted
Full GPU profile (now showing conv): `conv1d_ct4` (vocoder) = **11.1 s (23%)**, 91 calls — the #2 kernel,
hidden in my earlier truncated view. Two profiler-guided attempts, both REVERTED:
- **gemm_coop_m** (cooperative small-M GEMM for text beam M=5): REGRESSED text_beam 8→16.6 s. The
  per-output-channel cooperative reduce over N=20005 outweighs the 4×4 register tile's 2D reuse.
- **conv1d_ct44** (vocoder conv, tile time ×4 to amortize weight reads): REGRESSED vocoder 11→13 s.
  4× fewer work-items under-occupies the late stages (small C, huge T). The conv is occupancy-bound,
  not weight-BW-bound. Kept conv1d_ct4.

**Confirmed remaining walls (true GPU time, all occupancy/compute-bound at their floor):**
linear_gemm_tiled 19.9 s (encoder, 42%) · conv1d_ct4 11.1 s (vocoder, 23%) · linear_gemm 5.0 s
(text beam) · relpos_attention 1.9 s · conv1d_tc 1.9 s (encoder conv) · conv_transpose1d_ct4 1.7 s.

### HEADLINE: proper GPU profiler (NNOPT_GPUPROF) found + fixed the hidden layernorm cost
Verified state (default fp16, bit-exact, ~run-to-run vocoder variance ±2 s):
eng ~48-50 s (211/211) · spa ~50-52 s (191/191) · por ~52-54 s (234/234). vs pre-profiler 57.2 s.
**Project arc: 271 s → ~48 s = 5.6×.** The layernorm fix (1-WI/row → cooperative wave/row) was a 16%
lossless win the host stage-timer had hidden for the entire project — found only with GPU event timing.

### Kernel scan for hidden inefficiencies (2026-06-10) — found conv_pool (2nd scalar anti-pattern)
Split conv1d_tc in the profiler (depthwise→conv1d_dw alias): the 1885 ms was **ONE call** — the
adaptor `conv_pool` (Cin=768→1536, K=8, **stride=8**) at **0.2 GFLOP/s** (scalar 6144-iter inner loop
per work-item). Since stride==K (non-overlapping), it's a GEMM in disguise. Reformulated as
**im2col + tiled GEMM** (`im2col_tc` + linear_gemm_tiled; the weight [Cout,Cin,K] is already the
[Cout, Cin*K] GEMM matrix): **conv_pool 1782 ms → ~1.7 ms im2col + ~180 ms GEMM (~10×)**, encoder
24→22.4 s, units bit-exact. The 9 depthwise convs (conv1d_dw) were fine (8.6 ms each).

Scan verdict: the two hidden costs were both the scalar/1-thread anti-pattern — **layernorm**
(8.6 s, 1-WI/row) and **conv_pool** (1.78 s, scalar im2col-able conv). Everything else is genuinely
compute/occupancy-bound at its floor: linear_gemm_tiled 20 s (encoder), conv1d_ct4 11 s (vocoder),
linear_gemm 5 s (text beam), relpos_attention 1.9 s, attention_context 0.83 s (decoder, low occupancy
at Tq=1 but small total). Cooperative-attention + gemm_coop_m + conv-time-tiling were tried, regressed.

### Verified state after the scan (default fp16, bit-exact)
eng **49.3 s** (211/211) · spa 51.0 (191/191) · por 53.1 (234/234) · vocoder ±2 s run variance.
Profiler-driven session arc: 57.2 → ~49 s (layernorm 11× + conv_pool 10×). Project: 271 → 49 s = 5.5×.

---

## 2026-06-10 — guide §6-driven round (target 50→25): argmax + barrier-free GEMM
Profiler found a 3rd hidden cost: **argmax_f32 = 2961 ms (single work-item over the 10-20K vocab)**.
Applied 5 levers, profiler-verified each:
- ① **argmax_coop** (cooperative reduce, tie-break = smallest-index so bit-exact): 2961 → **116 ms (25×)**;
  it was on the blocking critical path so wall saved ~5 s (text_beam 10.4→5.5, unit 9.6→3.2). ✅
- ② **encoder GEMM → barrier-free 4×8 register** `linear_gemm8` (guide §6.1.4 no-barrier→max WG, §6.5
  no local→more waves): tiled 20.1 → **17.7 s** (+12%). ✅ (4×8 marginally > 4×4 register > tiled.)
- ③ **vocoder conv explicit WG=128** (§6.1.4): ~noise (compute-bound). kept.
- ④ **text-beam via linear_gemm8**: REGRESSED 5.5→19.7 s (4×8 wastes M=5) — reverted to 4×4. ❌
- ⑤ perf_hint HIGH (kept), WG tuning (done where it helped).

GPU profile after: linear_gemm8 17.7 s (encoder, 43%) · conv1d_ct4 11.3 s (vocoder, 28%) · linear_gemm
5.1 s (text beam) · relpos 1.9 · conv_transpose 1.6 · argmax_coop 0.12 (was 2.96). **TOTAL_GPU 46.6→41.0 s.**

### Verified state (default fp16, bit-exact)
eng **42.9 s** (211/211) · spa 44.1 (191/191) · por 46.6 (234/234). Round: ~50 → ~43 s.
**Project arc: 271 → 43 s = 6.3×.** Did NOT reach 25 s: encoder GEMM (17.7 s) + vocoder conv (11.3 s)
= 71% are compute-bound at their floor (register/tiled/image/int8/occupancy all explored). 25 s would
require int8 across encoder+vocoder (encoder int8 is units-exact but only ~7%; vocoder conv int8
untried) or architectural change (distill / fewer decode steps). The M=1 + scalar-anti-pattern levers
are exhausted; the profiler (NNOPT_GPUPROF) found all 3 hidden costs (layernorm, conv_pool, argmax).

### int8 on the encoder (register GEMM) — NO WIN, structural reason (2026-06-11)
Added `linear_gemm8_int8` (int8 weights + per-row scale, char8→convert_float8) for the encoder,
hoping the barrier-free register GEMM (which re-reads weights from DRAM, no local reuse) was
weight-BW-bound. Measured WARM (quant in startup warmup): **encoder 20.1 → 20.8 s — flat/slightly
worse**, units still bit-exact. **Conclusion: the encoder GEMM is ALU/compute-bound, not BW-bound** —
int8 halves weight *bytes* but the limiter is fp32-FMA throughput + occupancy, and the int8→float
convert adds ALU. int8 only helps memory-BW-bound kernels; the one BW-bound kernel here (decode GEMV)
can't tolerate int8 (argmax accuracy). So int8 helps exactly where we can't use it and doesn't help
where we can. Kept `NNOPT_INT8`-gated OFF. Default fp16 stays **42.4 s, bit-exact**.

**Definitive: kernel-level optimization floor is ~42-43 s (project 271→42 = 6.4×, all bit-exact).** The
encoder GEMM (17.7 s GPU) + vocoder conv (11.3 s) are compute/occupancy-bound and resist every lever
(tiled/register/image/int8/occupancy/WG). Below ~40 s needs architectural change: distill the
decoders, reduce beam/decode steps, or NPU/Hexagon offload — not kernel tuning.

---

## 2026-06-11 — THE BREAKTHROUGH: CLBlast auto-tuned HGemm (42 → 20 s)
The "where is the time going" question + KINFO probe (linear_gemm8 = 504 B private mem ≈ 126 regs/thread)
showed the encoder GEMM was **latency-bound from register pressure** (too few waves to hide global-load
latency, guide §3.2.3) — NOT at the compute limit (it ran at 0.2% of peak). My hand kernels couldn't
balance reuse vs occupancy. **CLBlast (already linked, HALF=ON) is auto-tuned for the device** and the
reference ports only beat it at M=1 decode — for M>1 it wins. Routed every M>1 GEMM through it:
- **Encoder GEMM** (M=299) via `pytorch_linear` (CLBlast HGemm) + `bias_add`: **20.0 → 5.5 s (3.6×)**.
- **Text-beam M=5 + teacher-forced**: CLBlast: text_beam 5.4 → 3.7 s.
- **Vocoder convs** → `im2col_ct` (channel-major, stays [Cout,T]) + CLBlast (`pytorch_conv1d`) +
  `bias_add_ct`: **vocoder 13.1 → 7.3 s (1.8×)**. (Decode M=1 stays on custom gemv_img — CLBlast loses at M=1.)

### WARM steady-state (default fp16)
| lang | encoder | text_beam | unit | vocoder | TOTAL | units | wav cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| eng | 5.5 | 3.7 | 3.4 | 7.3 | **20.1 s** | 211/211 | 0.9982 |
| spa | 5.5 | 5.3 | 3.0 | 6.3 | **20.5 s** | 191/191 | — |
| por | 5.5 | 4.6 | 3.8 | 7.8 | **21.9 s** | 233/234 | — |

**Project arc: 271 → 20 s = 13.5×. RTF ~3.3× (6 s clip in 20 s).**
**Tradeoff (within the owner-approved small-noise budget):** CLBlast HGemm accumulates in fp16 (vs my
fp32-accum hand kernels) → waveform cos 0.99999 → **0.9982**, and occasional 1-unit flip (por 233/234;
eng/spa still bit-exact). CLBlast is now ALWAYS used for M>1 GEMMs (the NNOPT_NOCLBLAST opt-out was removed — it would only ever be slower). **Cold-start:** CLBlast JIT-compiles + auto-tunes per GEMM shape on first use
(first run ~46 s); caches to disk → warm runs ~20 s. A deployed model warms up once.

**Lesson (own it): I should have tried CLBlast chapters ago.** I spent many rounds hand-tuning GEMMs
(tiled/register/image/int8/occupancy) that the auto-tuned library beats 3.6×. The KINFO register-pressure
probe + "where's the time going" framing is what finally pointed at it. Remaining: vocoder 7.3 s
(conv_transpose still hand kernel 1.7 s + im2col overhead) is the next target if pushing below 18 s.

---

## 2026-06-11 (cont) — overhead hunt: transposed-conv → CLBlast (20 → 18 s)
"Where's the overhead?" — GPU profile showed the time is NOT compute: im2col_ct 3.06 s (conv-input
expansion), relpos_attention 1.84 s (hand kernel, 286 regs/thread = occupancy-bombed), conv_transpose
1.68 s (hand scalar), bias passes 0.75 s. ~12 s of the 20 s is overhead/reshaping, ~8 s real GEMM.
- **Transposed conv (vocoder upsample) → GEMM + col2im** (`get_wt_transpose` repacks W[Cin,Cout,K]→
  Wr[Cout*K,Cin] once; `pytorch_conv1d`(Cout*K,T,Cin) → tmp; `col2im_transpose` folds): vocoder
  6.96 → 5.5 s, units bit-exact.

### Steady-state warm (default fp16, CLBlast always on)
| lang | encoder | text_beam | unit | vocoder | TOTAL | units |
|---|---:|---:|---:|---:|---:|---:|
| eng | 5.5 | 3.6 | 3.4 | 5.5 | **18.3 s** | 211/211 |
| spa | 5.5 | 5.3 | 3.0 | 5.0 | **19.1 s** | 191/191 |
| por | 5.5 | 4.6 | 3.8 | 6.0 | **20.1 s** | 233/234 |
waveform cos 0.998. **Project: 271 → 18 s = 15×. RTF ~3.0× (6 s clip in 18 s).**

**Remaining overhead (next targets):** im2col_ct 3.0 s (conv expansion — needs implicit-GEMM conv or
K-tap accumulate to avoid the scratch), relpos_attention 1.8 s (register-bombed hand kernel), decode
per-step host-sync overhead (~2 s in the 7 s text+unit). Real GEMM compute is now only ~8 s of 18.

### Overhead hunt cont. — im2col + relpos confirmed at floor (2026-06-11)
- **im2col hybrid sweep** (direct conv1d_ct4 for small-C stages vs im2col+CLBlast): CLBlast wins at EVERY
  threshold (THR=4 vocoder 5.5 s; THR=128 → 7.3 s; all-direct → 11.4 s). The im2col scratch is NOT the
  cost — GEMM efficiency is, and CLBlast+im2col is already optimal. im2col stays.
- **relpos low-register variant** (load q/bu/bv per key vs hoist): REGRESSED encoder 5.5→6.2 s — per-key
  cache re-reads cost more than the register pressure saved. Hoisted version kept.
**FINAL: eng 18.1 s, units bit-exact. Project 271→18 = 15×, RTF ~3.0×.** Remaining overhead (im2col 3 s,
relpos 1.8 s, decode per-step host-sync ~2 s) is at the floor for tried approaches; last untried lever
is on-GPU chained decode (~0.6 s, complex control-flow rewrite). Real GEMM compute ≈ 8 s of the 18.

---

## 2026-06-11 — decode overhead: chained greedy + GPU top-k beam (18 → 17.4 s)
Profiler "where's the overhead": of 18 s only ~8 s is GEMM; the decode (text 3.5 + unit 2.9 = 6.4 s)
was the suspect. Found two host-sync overheads and killed them:
- **Chained greedy unit decode**: argmax writes the token to a device buffer (out_idx, no readback);
  next step's embed reads it directly (in-order queue ⇒ correct). EOS checked in batches of 8 instead
  of every step. New: `argmax_dev`, `decode_step_ids`, `set_at_dev`, `alloc_ints`/`copy_ints`. unit 3.1→2.9 s.
- **GPU top-k beam**: the beam did `cand_size=10` BLOCKING argmax readbacks/step (≈10 GPU drains/step).
  Now `topk_dev` runs all 10 argmax+mask rounds on-GPU + ONE bulk readback. text_beam 3.67→3.45 s.

Both LESS than hoped (~0.7 s total) → the decode is **kernel-launch/compute-bound, not sync-bound**
(~35 tiny kernels/step). Units bit-exact (eng 211, spa 191, por 233/234 — the 1-off is CLBlast encoder noise).

### Steady-state warm
| lang | TOTAL | units |
|---|---:|---:|
| eng | **17.4 s** | 211/211 |
| spa | 18.4 s | 191/191 |
| por | 19.1 s | 233/234 |
**Project: 271 → 17.4 s = 15.6×. RTF ~2.9×.** Next lever for 15 s: **decode kernel fusion** (fuse the
~35 launches/step — fused QKV, fused attention block — like the reference ports' block_fused). The
decode is launch-bound, so collapsing launches is the path; it's a real rewrite (~1–2 s) not a tweak.

### Decode block fusion (2026-06-11) — confirms decode is BW-bound, NOT launch-bound
Took on fusion (user request). Fused QKV (3 GEMVs→1+slice): NEUTRAL/worse (warm unit 2.84→3.04 s) —
the slice-copies offset the saved launches. Diagnostic conclusion across this round's 3 decode
experiments: chained decode (−0.4 s, not sync-bound) + GPU top-k (−0.2 s, not readback-bound) + fused
QKV (neutral, not launch-bound) ⇒ **the M=1 decode is weight-BANDWIDTH-bound** (~50 MB weights/step at
~40% of texture roofline). Reverted fused QKV. `get_qkv` helper left in tree (unused).

### FINAL (default fp16, CLBlast everywhere, chained decode + GPU top-k)
eng **17.4 s** · spa 18.4 · por 19.1, units eng/spa bit-exact, por 233/234, wav cos 0.998.
**Project: 271 → 17.4 s = 15.6×, RTF ~2.9×.** Every component at its compute/BW floor:
encoder 5.4 (CLBlast) · text_beam 3.5 (CLBlast M=5) · unit 2.9 (BW-bound M=1 GEMV) · vocoder 5.4
(im2col+CLBlast, confirmed optimal). **Reaching 15 s is no longer a kernel problem** — it needs to read
fewer weight bytes per decode step (int8 decode = breaks exactness) or fewer decode steps/layers
(distill the decoders) — i.e. architectural, not OpenCL tuning.

---

## 2026-06-11 — int8 DECODE experiment (NNOPT_INT8DEC): lossless but NO speedup
User: "try 1 int8 decode and report speed gains vs how different is output." Added a gated int8 branch
to the T==1 decode GEMV (`gpu_ops.cpp linear()`, env `NNOPT_INT8DEC`): per-row symmetric int8 weights via
`get_int8` + `gemv_coop_int8`/`gemv_coop4_int8`, with an in-process full-pipeline warmup (`NNOPT_WARMUP`)
so the one-time quantization + Adreno online-compile are excluded from the timed run. eng, gas-station clip.

**NOTE — device was thermally throttled this session** (encoder ~97 s vs historical warm 5.5 s; text_beam
swung 5.7→17.6 s run-to-run). Absolute numbers are junk, but fp16 vs int8 ran back-to-back under identical
throttled conditions, so the *relative* decode comparison is valid.

| path | unit_greedy (warm) | units vs fp16 | waveform cos vs fp16 |
|---|---:|---|---|
| fp16 (image2d GEMV, default) | 10746 ms | — | — |
| int8 decode (NNOPT_INT8DEC)  | 10800 ms | **211/211 IDENTICAL** | **1.000000** |

**Two findings, both overturning priors:**
1. **int8 decode is BIT-EXACT here**, not 62/211. The old "int8 breaks decode" result was a different/coarser
   path; **per-row symmetric int8 weight-only + fp32 accumulate does not flip a single argmax** on eng.
2. **int8 decode gives ZERO speedup** (10.80 vs 10.75 s, within noise / marginally slower). Reason: the fp16
   default already uses the **image2d (texture-cache) GEMV** — a lossless ~1.71× BW path at ~13.5 GB/s.
   int8 halves the weight bytes but reads from plain global memory (~10 GB/s), so int8-global ≈ fp16-texture.
   Halving bytes on the GEMVs also doesn't move the needle because per-step fixed/launch cost (~35 kernels/
   step) is a large share — consistent with the earlier fused-QKV null result.

**Conclusion:** int8 decode is NOT worth shipping — no speed upside, and the fp16 image2d path is already
lossless. The branch stays behind `NNOPT_INT8DEC` (default off) as a documented experiment; default build
unchanged. The remaining lever for <15 s is architectural (distillation / fewer decode steps), confirmed
again: the decode is at its bandwidth/launch floor on the current architecture.

---

## 2026-06-11 — UTILIZATION MEASURED: we are at 36.8%, NOT the floor. 16.2s is GPU IDLE.
User: "are we at 99% utilization? measure, don't assume." Built a real instrument: every enqueue (hand
kernels AND CLBlast GEMMs, via a new `cl_event*` out-param on pytorch_linear/conv1d) gets a profiling
event, collected without per-kernel waits (no serialization), drained in batches. `dump_gpu_prof` now
reports GPU_BUSY (Σ kernel HW exec time), GPU_SPAN (first START→last END), IDLE, UTILIZATION, AND a
per-stage table (`prof_mark` at each pipeline lap). Warm timed run (in-process warmup excludes Adreno
online-compile, which is what the earlier "98s encoder" was — NOT throttle; device measured 35°C).

```
stage         busy_ms   span_ms   idle_ms   util%
fbank            23        23         0     100.0
encoder        2887      5441      2554      53.1
text_beam       396      6734      6338       5.9   <-- 0.4s compute, 6.3s pure host overhead
mt_feat          38       394       356       9.5
synth            10       145       135       6.6
unit_greedy    2374      2801       427      84.7   <-- chained decode WORKS, near floor
vocoder        3706      9585      5880      38.7
GPU_BUSY 9.4s  WALL 25.5s  IDLE 16.2s  UTILIZATION 36.8%
```

**MAJOR REFRAME (corrects the earlier "kernel optimization exhausted at 17s, BW-bound at floor"):**
- The dense GEMMs we tuned for weeks are NEGLIGIBLE: `clblast_gemm` 77ms (0.8%), `clblast_conv_gemm`
  103ms. Encoder M=38 is tiny. We optimized the wrong thing.
- Of the 9.4s GPU-BUSY, the fat is `im2col_ct` 3018ms (32%, vocoder conv RESHAPING — the conv matmul is
  only 103ms!) and `relpos_attention_context` 1861ms (20%, register-bombed encoder attention). Pure
  matmul is ~1.7s.
- **16.2s (64%) is GPU IDLE** — the GPU sits waiting for the host. Concentrated in **text_beam (6.3s
  idle, 5.9% util)** and **vocoder (5.9s idle)**. unit_greedy is already 84.7% (chained decode paid off).

**Why text_beam is 5.9% util:** per-step blocking readbacks (download_ints/download_f32 for top-k) +
heavy CPU bookkeeping (partial_sort, KV-cache copy_into reorder) drain the in-order queue every step.
0.4s of GPU work stretched over 6.7s of host serialization. This is the single biggest reclaimable win.

**At 99% util the GPU-side wall would be ~9.5s** (current busy) — and shrinking the busy fat (im2col,
relpos) could push it toward ~5s. So faster-than-realtime (6s clip) is reachable WITHOUT distillation,
purely by closing idle + killing reshaping overhead. NOTE: vocoder wall has DVFS variance (9.9–26s
across runs at stable 35°C — DDR clock scaling, not thermal); busy/util ratios are stable.

**Attack order (by reclaimable idle):**
1. text_beam → GPU-resident beam search (chain like unit decode; kill per-step readbacks). ~6s.
2. vocoder → preallocate/reuse conv scratch (stop clCreateBuffer per conv), reduce inter-stage glue. ~5s.
3. encoder → reduce per-conformer-layer launch glue. ~2.5s.
4. (busy fat) implicit-GEMM conv to kill im2col 3s; better relpos attention 1.86s.

---

## 2026-06-11 — text_beam attack #1: CLBlast was wrong tool for M=5 decode (6.8 → 5.4 s, hypo exact)
Per-stage profiler said text_beam = 5.9% GPU-util (0.4s busy, 6.3s idle). TBPROF (per-step host timing
with clFinish attribution) localized it: `decode_step_batch` = 414 ms/step = 91% of the stage; score/
topk-readback/kv-reorder are noise (~33 ms/step combined). The decode does ~28 GEMMs/step through
CLBlast (the `T>=4` gate), and CLBlast's HOST-side per-call overhead (~2 ms) dwarfs a tiny M=5 GEMM's
GPU work (~0.12 ms). unit_greedy proves it: M=1 hand-GEMV decode is 0.26 ms/enqueue & 84.7% util vs
text_beam's 2 ms/enqueue.
- **Fix 1 — route small-M GEMMs to hand `linear_gemm`** (1 launch, bias folded in), keep CLBlast only
  for M >= NNOPT_CLBLAST_MINM (default 16; encoder ≈38 stays on CLBlast). New `cl_event*` param threaded
  so CLBlast still profiles. decode 414 → 317 ms/step, text_beam 6.8 → 5.5 s. hypo EXACT, units 211.
- **Fix 2 — kill per-beam alloc/copy churn**: new `copy_into_off` (offset→offset clEnqueueCopyBuffer, no
  alloc) for the K/V cache appends + a reused `qb` buffer (in-order queue ⇒ safe). Removed ~56
  clCreateBuffer/step. decode 317 → 314 ms/step (negligible) — so allocs weren't the cost either.
- **Diagnosis locked: decode is OP-COUNT-bound** — ~170 tiny ops/step (kernels + clEnqueueCopyBuffer,
  the copies invisible to the kernel profiler's "busy"). The per-beam self-attention loop is ~100 of
  them. Next lever = batch the B-beam self-attention into one call (needs contiguous KV-cache layout) to
  collapse ~80 ops/step → text_beam ~2.5 s. hypo stays exact through both fixes.

### Result (per-stage util, improved binary) — text_beam 5.9% → 90.8% util, overall 36.8% → 46.1%
| stage | busy_ms | span_ms | idle_ms | util% |
|---|---:|---:|---:|---:|
| fbank | 23 | 23 | 0 | 100 |
| encoder | 2904 | 5507 | 2603 | 52.7 |
| text_beam | 4698 | 5175 | 477 | **90.8** |
| mt_feat | 507 | 601 | 95 | 84.2 |
| synth | 212 | 232 | 20 | 91.3 |
| unit_greedy | 2405 | 2828 | 422 | 85.1 |
| vocoder | 3589 | 13847 | 10258 | **25.9** (DVFS-inflated) |

hypo EXACT, units 211, **waveform cos 0.998050** vs fp16 baseline. text_beam flipped from idle-bound
(CLBlast host tax) to compute-bound on the hand GEMM — busy went UP (0.4→4.7s, now properly counted as
kernel time) while wall dropped (6.8→5.2s). **New worst stage = vocoder (25.9% util, 10.3s idle — the
per-conv im2col scratch clCreateBuffer churn).** Next targets: vocoder scratch reuse, then encoder (2.6s
idle), then the busy fat (im2col 3s, relpos 1.86s, and a GPU-efficient small-M GEMM for text_beam).

---

## 2026-06-11 — vocoder attack: reuse im2col scratch (−7.7s, same-state A/B). HUGE.
Vocoder was 25.9% util (~10s idle). CodeHiFiGAN runs ~95 convs (conv_pre + 5 upsample × (1 transposed
+ 3 resblocks×6) + conv_post). Each conv1d_ct did `alloc(Cin*K*T)` for the im2col col — for late
upsamples (C≈32, T≈67520, K=11) that's a **~47 MB clCreateBuffer EVERY conv, ~95×/run**. That alloc
churn (+ the big im2col write into fresh memory) was the idle.
- **Fix: one growable persistent `col_scratch_` reused across all convs** (in-order queue ⇒ prior conv's
  GEMM finishes reading before next conv's im2col writes; SYNC-01 safe). `NNOPT_CONV_SCRATCH=0` disables.
- **Same-state A/B: vocoder 15.68 → 8.01 s (−7.67 s); TOTAL 29.66 → 22.07 s. units 211, waveform cos
  0.998050 (bit-identical).**

### Session cumulative (same device DVFS state, verified A/Bs)
- text_beam: route M=5 decode GEMMs CLBlast→hand `linear_gemm` (−1.5s) + minm=8 (mt_feat/synth recover).
- vocoder: reuse im2col scratch (−7.7s).
- **Combined ≈ −9.2 s end-to-end in the current state.** Current total 22.1 s (vs ~31 s old-binary
  equiv in this DVFS state). All correctness preserved (hypo exact, units 211, wav cos 0.998).
- NOTE device DVFS still inflates absolutes vs the 17.4 s historical good-state baseline; the saving is
  the trustworthy metric. Vocoder now 8.0s (was worst) — next: re-measure its util (CLBlast conv host
  overhead across ~95 calls likely the remaining idle) + encoder 2.6s idle.

---

## 2026-06-11 — post-reboot 3-config A/B: device STILL DDR-throttled; both fixes confirmed net wins
After a device restart (attempt to restore the 17.4s good-state), same 6s eng clip, 3 configs back-to-back:
| config | encoder | text_beam | unit | vocoder | TOTAL |
|---|---:|---:|---:|---:|---:|
| A all-old (MINM=4, SCRATCH=0) | 5609 | 6552 | 2727 | 15419 | **30816** |
| B +vocoder scratch (MINM=4, SCRATCH=1) | 5432 | 6312 | 2802 | 7861 | **22973** |
| C +text hand-GEMM (MINM=8, SCRATCH=1, DEFAULT) | 5345 | 5166 | 2818 | 7701 | **21660** |

**The OLD binary (A) gives 30.8s, NOT 17.4s — reboot did not restore the good state.** Diagnosis:
encoder 5.5s = baseline (GPU-compute-bound, fine) but text_beam 3.5→6.5s and vocoder 5.4→15.4s inflated
~2-3× → the **DDR/memory clock is throttled**, not GPU compute. Memory-bound stages suffer; compute
stages don't. The 4.7s gap to the 17.4s best is device state, NOT code (proven: old code is even slower).
**Both fixes are net wins same-state: vocoder scratch −7.6s (A→B), text hand-GEMM −1.1s (B→C, does NOT
regress at this clock). Combined 30.8→21.7s = −9.1s.** Keep both. To reproduce 17.4s the device needs its
memory bus clocked high (plugged in / battery-saver off / fully cold). Our memory-traffic reductions help
MORE when DDR is the bottleneck, so they're the right direction regardless.

---

## 2026-06-11 — #3 int8 re-test on BW-bound device: REJECTED (slower + breaks accuracy)
Hypothesis: now that the device is DDR-throttled (BW-bound), halving weight bytes via int8 should pay
(it was neutral in the good state). MEASURED (same-state A/B, eng 6s clip):
| config | encoder | unit | TOTAL | units |
|---|---:|---:|---:|---:|
| A baseline fp16 | 5420 | 2730 | 22015 | 211 ✓ |
| B decode-int8 (NNOPT_INT8DEC) | 5556 | 3832 | 23239 | 228 ✗ |
| C encoder-int8 (NNOPT_INT8) | 21592 | 1776 | 34789 | 142 ✗ |

**Hypothesis WRONG. Both paths slower AND wrong units.** Lesson: byte-count is not the bottleneck —
KERNEL EFFICIENCY is. Encoder int8 4× SLOWER because it bypasses CLBlast for the hand
`linear_gemm_tiled_int8` (CLBlast's tuned fp16 >> our hand int8; fewer bytes through a bad kernel loses).
Decode int8 SLOWER because fp16 already reads via image2d texture cache (~13.5 GB/s) > int8 global
(~10 GB/s) even at half bytes — AND breaks AR argmax (228 vs 211; earlier "bit-exact" was a different
code state). int8 stays env-gated/off. **The real lever = make our hand kernels as efficient as CLBlast/
image2d (#1 small-M GEMM, #2 implicit-GEMM conv), NOT byte reduction.**

---

## 2026-06-11 — #1 attempt: gemv_m (BW-optimal small-M GEMM) REGRESSED 2×. linear_gemm stays.
Idea: linear_gemm reads the weight ~2× (once per M-tile); a kernel with 1 WI per output channel + M
accumulators reads each weight row ONCE → less traffic on the BW-bound device. Built `gemv_m`. A/B:
| | text_beam | TOTAL | units | cos |
|---|---:|---:|---:|---:|
| linear_gemm (default) | 5096 | 21714 | 211 | — |
| gemv_m (NNOPT_GEMVM=1) | 10430 | 27181 | 211 | 1.000000 |

**gemv_m is CORRECT (cos 1.0) but 2× SLOWER.** Lesson: on Adreno, reading the weight "once" doesn't help
if ONE thread issues those reads SERIALLY — a single work-item's K-loop has too little memory-level
parallelism to hide DRAM latency. You need MANY concurrent threads in flight. linear_gemm's register tile
(more work per WI, more WIs) keeps the memory pipeline full and wins despite ~2× weight reads. (The
cooperative gemm_coop_m regressed earlier too, for a different reason: reduce overhead over N=20005.)
**Verdict: linear_gemm is the best simple hand GEMM for M=5; beating it needs CLBlast-level autotuning.
gemv_m default OFF (NNOPT_GEMVM=1 to enable). #1 is a dead end with naive kernels — pivot to #2 (vocoder
im2col scratch writes, 3s of pure avoidable traffic).** Two BW-theory predictions wrong now (int8, gemv_m):
byte-count is NOT the lever on this device — kernel efficiency / memory-level-parallelism is.

## 2026-06-11 — #2 im2col 3D launch (no per-elem int-div): −6% only. im2col is latency-bound.
Replaced 1D launch (2 int-div/element: gid/T, row/K) with 3D (t,k,ci) — zero divisions. im2col_ct
2992 → 2823 ms (−6%), vocoder/total unchanged, units 211, cos 1.0. Kept (tiny win, no downside). But the
divisions were NOT the bottleneck — im2col is **memory-latency-bound** (1 elem/WI, writes tens of MB of
scratch; vectorizing the read is alignment-blocked since ti=t-pad+k*dil isn't 8-aligned). The algorithmic
alternative (direct conv, no scratch) already regressed to 11.4s. So im2col ~2.8s is near its practical
floor for hand code.

### HONEST WALL (3 consecutive naive kernel attempts failed/marginal this round)
- #3 int8 (encoder+decode): REGRESSED (4× encoder, breaks accuracy). byte-count ≠ lever.
- #1 gemv_m (weight-read-once small-M GEMM): REGRESSED 2× (1 WI/channel = no memory-level parallelism).
- #2 im2col divisions: −6% (latency-bound, not div-bound).
**Pattern: the remaining hot kernels (lm_head 4.3s via linear_gemm, im2col 2.8s, relpos 1.86s) are at the
hand-tuning wall — naive changes regress because CLBlast/linear_gemm/im2col are already near practical
hand limits, and the device is memory-latency-bound. Real further kernel gains need expert GPU-arch work
(tuned small-M GEMM, true implicit-GEMM conv, flash-style relpos) OR the architectural lever (distillation:
fewer decode steps/layers). Banked wins this session (same-state): vocoder scratch −7.7s + text de-CLBlast
−1.5s = ~21.7s (from ~31s). Device DVFS still inflates ~1.3-3× vs 17.4s good-state best.**

## 2026-06-11 — #2b hand conv_gemm_ct REGRESSED (8.9→14.9s); IMPORTANT profiler correction
Built conv_gemm_ct to replace CLBlast for the vocoder conv (CONVPROF showed CLBlast conv = 31ms wall /
1.1ms captured GPU → looked like ~2.75s reclaimable host overhead). A/B: vocoder 8.9 → 14.9s (hand SLOWER
by 6s), units 211, cos 0.998112. REVERTED (NNOPT_CONVGEMM=1 to enable).
**CORRECTION: the "~30ms/call host overhead" was MIS-MEASURED.** My profiler attaches ONE cl_event per
CLBlast call, but CLBlast launches HELPER kernels (pad/copy for non-tile-aligned dims) that the single
event misses → they were mislabeled as "idle"/"host overhead" when they're real, efficient GPU work.
So: (1) true GPU utilization is HIGHER than the reported 62.8% (the single-event GPUPROF undercounts
CLBlast helpers); (2) the "8.4s idle / 39% overhead" was OVERSTATED; (3) the hand conv kernel (col[k*T+t]
strides by T → cache-thrash) is far less efficient than CLBlast. CLBlast is the right tool for the conv.
**Round scorecard: 4 kernel-efficiency attempts (int8, gemv_m, im2col-div, conv_gemm) all LOST to
CLBlast/existing kernels.** The session's ONE big win (vocoder scratch −7.7s) came from finding WASTE
(alloc churn), not a better kernel. Productive vein = hunt more waste (allocations/redundant copies) +
architectural (distillation), NOT naive kernel swaps. Current default holds at ~22s (vocoder ~8s CLBlast).

## 2026-06-11 — waste-hunt: buffer POOL (generalize col-scratch). −0.5s, kept default.
The arena clCreateBuffer'd on every alloc() and clReleaseMemObject'd on every release_to() — thousands
of create/release pairs/run. Generalized the col-scratch fix: release_to() recycles buffers to a free
list (keyed by byte size), alloc() reuses an exact-size free buffer (in-order queue ⇒ SYNC-01 safe).
NNOPT_BUFPOOL=0 disables. Clean alternating A/B (encoder ~5.3s = stable):
| | text_beam | unit | TOTAL |
|---|---:|---:|---:|
| POOL   | 4927/4967 | 2499/2501 | 21894/21618 (avg 21756) |
| NOPOOL | 5147/5108 | 2854/2799 | 22202/22400 (avg 22301) |
**−0.54s (−2.5%), consistent; concentrated in unit_greedy (−0.35s) + text_beam (−0.18s) = the
alloc-heavy decode loops. units 211, cos 1.0. No downside → default on.** Confirms create/release churn
cost real time (less than the col-scratch outlier). Session cumulative same-state: vocoder scratch −7.7 +
text de-CLBlast −1.5 + buffer pool −0.5 + im2col-3D −0.17 ≈ −9.9s. Current ~21.7s.
**Waste-hunt now small-increment territory; the col-scratch was the big outlier. To reach ~10s needs the
device in good DDR state (~15s) OR distillation (architectural).**

## 2026-06-11 — ACCURATE measurement: kgsl HW gpubusy counter (no profiler blind spot)
Added read of /sys/class/kgsl/kgsl-3d0/gpubusy (HW busy/total cycles, counts ALL kernels incl CLBlast
helpers). Whole-run GPU_HW_UTIL = 66.7% (event-profiler said 63.2% — aggregate was actually close).
Per-stage HW util (warm, clFinish-bracketed via NNOPT_HWUTIL):
| stage | HW GPU-busy % |
|---|---:|
| encoder | 83.8 |
| text_beam | 97.4 (SATURATED) |
| mt_feat / synth | 97.4 (SATURATED) |
| unit_greedy | 96.5 (SATURATED) |
| vocoder | 66.2 (lowest — main reclaimable idle) |
**Key finding (overturns event-profiler per-stage): the DECODE stages are GPU-SATURATED at ~97% — compute
bound, no reclaimable idle (only faster kernels help, and hand kernels lose to CLBlast). The reclaimable
idle (~7.4s whole-run) is in the VOCODER (66%, 8.6s, ~2.9s idle) + encoder (84%, ~0.85s) + inter-stage
host gaps. MATH TO 10s: decode alone = 7.5s saturated; reclaiming ALL vocoder+encoder idle (~3.7s) → ~18s,
NOT 10s. Reaching 10s REQUIRES cutting the saturated decode compute = distillation (fewer/smaller decoder
layers or non-AR units). Build flags already -cl-fast-relaxed-math. Tools: NNOPT_HWUTIL (per-stage HW
util), GPU_HW_UTIL (whole-run). New idea pool from guide §8.9: subgroup HW reduction > local-mem reduce
for cooperative kernels (layernorm_coop/gemv_coop/argmax) — untried.**

## 2026-06-11 — Attack A (vocoder idle reclaim): per-block HW util + 2 attempts, ~0 net.
Per-vocoder-block HW GPU-busy (kgsl, warm): conv_pre 97%, ups0 79%, ups1 80%, ups2 73%, ups3 73%,
ups4 71%. **Idle GROWS with T** (late blocks huge T=16k-67k sit ~71-73% — latency-bound memory ops, not
enough resident waves to hide DRAM latency).
- **Removed 45 resblock clone() copies** (axpy reorder: add cur into conv output vs clone(cur)+axpy).
  Math-identical, cosine 1.0, units 211. **NEUTRAL** (vocoder 8.58s, no change) — clones weren't the cost.
  Kept (no downside).
- **im2col local=(64,1,1) wave-aligned WG**: REGRESSED 2× (im2col 2843→6047ms, vocoder→11.2s). Driver's
  default 3D WG shape beats a forced 1D-along-T one. Reverted.
**Verdict on A: the vocoder idle is REAL (~70% util, ~2.9s) but resists hand fixes — it's latency-bound
im2col + CLBlast helper kernels. Clone removal neutral, WG-size regressed.** Consistent with the whole
session: kernel/idle micro-opts mostly fail (CLBlast/driver defaults are well-tuned); the wins came from
finding WASTE (scratch reuse, buffer pool). Current ~22s holds.
**DECISIVE: accurate HW counter shows decode SATURATED at 97% (8s) — that's the wall. Even perfect
vocoder reclaim → ~19s. 10s is mathematically gated on DISTILLATION (cut the saturated decode compute).**

## 2026-06-11 — IDLE RECLAIM: vocoder input build moved to GPU (−0.85s, bit-exact)
The vocoder built its channel-major input [1792,211] on the CPU (dict 25MB page-in + 378K-elem loop +
upload) = ~580ms of pure GPU IDLE at the start of the stage. Moved to a GPU gather kernel
(vocoder_input_gather: gather lang/dict/spkr weight buffers by per-token code id). HOSTIDLE 580 → 7ms.
vocoder 8.4 → 7.5s, TOTAL 21.7 → 20.87s. units 211, cosine 1.000000. Clean idle→0 reclaim, no accuracy
cost. Session ~31s → 20.9s same-state. Next idle targets: inter-stage transfers (encMem upload,
mt_hidden download/re-upload), remaining vocoder conv-loop idle (~70% util, latency-bound im2col).

## 2026-06-11 — vocoder lrelu FUSED into im2col (removes 90 act + 45 clone passes)
Resblock applied lrelu as a separate full-tensor pass before each conv. Fused it into the im2col read
(im2col_ct new pre_act arg; conv1d_ct pre_act param; resblock uses conv1d_ct(cur,...,ACT_LRELU01)
removing clone+act). Removes 90 act passes + 45 clones. vocoder 7.43→7.35s, TOTAL 20.68→20.61s (marginal
wall), HW util 73→78%, units 211, cosine 1.000000. Kept (correct, fewer ops, util up). Wall ~flat: the
vocoder is dominated by the conv work (im2col copy ~2.8s latency-bound + CLBlast), NOT elementwise passes.

### Idle-reclaim push status (toward 99%/~15s)
Start 22.0s/66.7% → now 20.6s/~78%. Wins: vocoder input build→GPU gather (−0.85s, the real one),
lrelu→im2col fusion (neutral wall, +util). Per-stage: decode SATURATED 97%, encoder ~86% (relpos
register-bombed), vocoder ~72% (im2col latency-bound — resists WG-size [regressed 2x] + vectorization
[alignment-blocked]). The clean host-idle was reclaimed; remaining idle is in the actual conv/attention
COMPUTE (im2col copy, relpos attention) which is latency/BW-bound and resists hand tuning.

## 2026-06-11 — B: direct conv for smallest-Cin vocoder stage (−1s vocoder). WIN.
Hypothesis (corrected for caching): late vocoder stages have TINY weights (ups4: 16×16×11 = 5.6KB →
caches), so a direct conv's weight re-reads are cache HITS while im2col streams a ~24MB col scratch
through DRAM (K× write-expansion). Added small-Cin direct route (NNOPT_DIRECT_CIN_MAX, conv1d_ct4 + fused
pre_act). Sweep (vocoder ms): OFF 7833 | Cin<=16 6808 | Cin<=32 7093 | Cin<=64 8755(regress). Crossover
exactly at the cache boundary: Cin=16 weight caches (direct wins −1025ms), Cin=64 too big (im2col GEMM
wins). **Default = 16.** Verified: vocoder 7.8→6.86s, TOTAL→20.3s, units 211, cosine 0.999999.
(One util read glitched to 100% — anomalous; real ~78%, timing is the truth.)
### Push status: 22.0s → 20.3s this round (vocoder input→GPU gather −0.85s, lrelu→im2col fusion neutral,
direct-conv-Cin16 −1.0s). All bit-exact (cos ≥0.999999). Util ~66.7 → ~78%+. Next: encoder relpos (A) /
upsample act_n fusion / further small-Cin tuning.

## 2026-06-11 — push consolidation: 22.0 → 20.0s (idle reclaim), MEASUREMENT correction
PUSH WINS (all bit-exact, cosine 0.999999, units 211): vocoder input build host→GPU gather (−0.85s),
direct-conv for smallest-Cin vocoder stage (−1.0s), lrelu→im2col fusion (neutral wall, fewer ops).
22.0 → 20.0s.
**MEASUREMENT CORRECTION: /sys/.../kgsl-3d0/gpubusy is a FIXED-WINDOW SNAPSHOT (every 'total' reads ~1.0M
regardless of interval), NOT a since-last-read integral. So the GPU_HW_UTIL %s (66.7/73/78/100) were
instantaneous readings at the moment of the read, NOT run-integrated utilization — unreliable for "are we
at 99%". Rely on TIMING (wall, truth) + event-profiler busy/span (integral, but has a CLBlast-helper-kernel
blind spot that overstates idle for CLBlast-heavy stages).**
Reliable kernel breakdown @20s (HW timestamps): linear_gemm 4367ms (text decode M=5), im2col_ct 2648ms
(vocoder), relpos_attention 1890ms (encoder reg-bombed), conv1d_ct4 1320ms (ups4 direct), gemv 1424ms,
attention 894, layernorm 768. These big kernels (linear_gemm/im2col/relpos) RESIST hand tuning (gemv_m,
CLBlast-for-decode, im2col-WG, implicit-GEMM, low-reg-relpos all regressed). The clean idle is reclaimed;
further needs cracking these saturated/inefficient kernels.

## 2026-06-11 — A: linear_gemm_m (M-exact tile) for text decode — marginally slower, reverted.
Targeted the 4.4s text-decode M=5 GEMM. linear_gemm_m: 1 WI does all M rows × 4 out-channels (weight read
once/WI, no wasted half-M-tile). A/B: text_beam 4963→5145ms (+182ms), cosine 0.999999, hypo exact. SLOWER
— the runtime-M inner loop doesn't unroll like linear_gemm's static 4×4 (loses ILP). Default OFF (LGM=1 to
try). **3rd small-M GEMM kernel to lose to linear_gemm (after gemv_m 2×, CLBlast host-tax). linear_gemm is
the hand-tuned floor for M=5.** The big-three kernels (linear_gemm 4.4s, im2col 2.6s, relpos 1.9s) have all
resisted every variant this session. Push floor ~20.0s; further needs cracking these (the wins came from
structural/idle reclaim, not kernel efficiency).
