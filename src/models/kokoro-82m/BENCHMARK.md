# Kokoro-82M · Motorola Razr 2020 (Adreno 620) · OpenCL fp16 — Benchmarks

Test clip: "The teacher worked at the" → 1.90 s of audio @ 24 kHz (45 600 samples).
Wall time = `BENCHMARK total_inference_sec` from the device binary. RTF = wall / 1.9.
Quality gate: cos ≥ 0.99 vs host fp64 generator (`NNOPT_HOST_GENERATOR=1` A/B), on-device.

## Milestones

| Date | Change | Wall (s) | RTF | cos | Notes |
|---|---|---|---|---|---|
| 2026-05-24 | First fully-working port (fp16 GPU + host fp64 iSTFT) | ~160 | ~84× | — | journal Entry 5 era |
| 2026-05-25 | Hybrid path baseline (fp32 generator + hybrid resblocks) | 50.9 | 26.8× | 0.9959 | plan v1 start |
| 2026-05-25 | Phase 1+2: fast-math, perf_hint, LOCAL_T 384, half LDS, Cout-tile=4 + ic-unroll | 18.7 | 9.8× | 0.9959 | HYBRID_OPTIMIZATION_PLAN.md "current state" |
| 2026-05-2x | conv1d_f32_t4x4 register tiling (fp32 resblocks were 76 % of wall) | 16.0 | 8.4× | A/B-validated | from campaign memory; `NNOPT_CONV_T4X4=0` reverts |
| 2026-06-05/06 | Session (power-lost): fp16 texture-weight convs (`conv1d_ht_t8x4`), skinny GEMM (+texture) for non-generator linears, `lstm_seq` whole-recurrence LSTM, kpad GEMM unblock | 5.75 (2.175 s clip) | 2.64× | A/B per item | from session TUI log |
| 2026-06-06 | Audio epilogue fix: removed 0.8/p99 loudness rescale + hard clip (high-pitched distortion overlay); raw waveform out, parity with reference | 5.13 | 2.70× | zero clipped samples; peak 0.309 vs ref 0.318 | standard 1.90 s clip |
| 2026-06-06 | Hybrid-path deletion + fastest-by-default scripts (see below) | 4.56–4.58 warm | 2.40× | wav identical pre/post | cold first run after deploy ≈ 5.7 s (kernel-cache compile) |
| 2026-06-06 | **Texture-path generator convtranspose** (`convtr1d_f32_c4x4_tex`): ups[0]/ups[1] weights through TP/L1, phase-major RGBA16F image (built once per prefix), fused fp16 bias | **3.74–3.76** | **1.97×** | cos 1.0000 vs buffer kernel | convtr kernels 850 ms → 92 ms (9.2×); alternating A/B ×2; `NNOPT_TRTEX=0` reverts |
| 2026-06-06 | Streaming support: `--serve` persistent server mode + `scripts/stream.py` (chunk → synth → pipelined playback) | n/a (infra) | | re-entrant, 3-chunk soak clean | sustained synth RTF is the gapless-playback gate |
| 2026-06-06 | Killed hidden per-utterance costs: 7 blocking multi-MB `dump_fp32` readbacks (un-gated bisection leftover) + 8 release-mode `clFinish` in TICK | **3.39** | **1.79×** | cos 1.0000 | both now opt-in/debug-only |
| 2026-06-06 | **Buffer arena** (recording-stable, §5.7.1: no clCreate/Release between kernels) + **recordable queues** (record/replay generator span, byte-identical) | **3.36 single / 3.12 serve-marginal** | **1.77× / 1.64×** | replay BYTE-IDENTICAL to live; 8-utterance soak stable | arena = the win (serve marginal 4.3→3.1); replay ≈ neutral at these kernel sizes (host enqueues ahead; musicgen's 4× was for µs-kernels). `NNOPT_RECORD=0` reverts |

| 2026-06-06 | **WG-shape + wave-size sweep on `conv1d_ht_t8x4`** (§6.1.5/§9.2.1): lws (1,128)+compiler-default → **(4,32) + `qcom_reqd_sub_group_size("full")`** | **2.90 single / 2.50 serve-marginal** | **1.53× / 1.32×** | cos 1.0000; alternating A/B ×2 | conv 1556→1068 ms (−31%); full-wave WG needs ≥128 WIs ((2,32) regressed 1.7×); (1,256) thrashes tex L1. Knobs: `NNOPT_HT_L0/HT_LT/HT_WAVE` |

| 2026-06-06 | **noise_convs → texture path** (`plain_conv1d_tex_f32`: 4oc×4ol, 4-tap texels, fused bias; `NNOPT_NCTEX=0` reverts) + **GPU iSTFT in the active path** (split+istft inside the recorded span; `NNOPT_HOST_ISTFT=1` reverts) | **2.55 single / 2.20 serve-marginal** | **1.34× / 1.16×** | cos 1.0000; peak 0.3091 (fp32-vs-fp64 iSTFT noise); alternating A/B ×2 | noise_convs[0] 204→7.1 ms (29×), [1] 29→5.2 ms; iSTFT 0.84 ms GPU vs ~120 ms host; cpost readback 0.8 MB → 0.18 MB audio |

| 2026-06-06 | **Skinny-GEMM WG/wave re-sweep — NEGATIVE** (`NNOPT_SK_LT`=32/64/128 × `NNOPT_SK_WAVE`=half/full on `linear_skinny_*`) | no change (kept default) | — | — | default (64 lanes, compiler wave) already optimal: half64 2.11 (tie), lt32 2.28, lt128 2.43, full64 3.01, full128 2.97. Full-wave **regresses barrier-heavy K-reduction kernels** — the conv win (§9.2.1) only transfers to barrier-free kernels. Knobs kept for future shapes |

| 2026-06-06 | **int8 resblock convs via `cl_qcom_dot_product8`** (`NNOPT_DOT8=1`, plan-v3 #3): per-oc symmetric weights + dynamic per-tensor unsigned acts (zp=128), channels-last packed xq with zp-filled pad rows, `qcom_dot8_acc` over 4-channel groups, LDS-tiled quant transpose | **~2.08 single / 1.83 serve-marginal** | **1.09× / 0.96×** | cos 0.9940 vs fp16 path (gate ≥0.99); peak 0.3109 vs 0.3091; alternating A/B ×2; ear check PASSED ("exactly the same") → **default ON** (`NNOPT_DOT8=0` reverts) | conv 1076→670 ms (**1.61×**); quant+absmax overhead 19+9 ms after LDS tile (was 115). **OCT=4 critical**: 8 int4 accs spill (949 ms); full-wave regresses 1.6× (unlike the fp16 conv); geometry optimum stays (4,32). **First marginal chunks under RTF 1.0.** Old "int8 doesn't help" verdict was emulated int8 — hardware dot8 changes it |

Streaming: sustained synth RTF **1.05/1.01** with `NNOPT_DOT8=1` (longest chunk 0.99; was 1.19 before dot8, 2.20 at day start). Gapless needs < 1.0 sustained — remaining gap is arena churn on T change (slots dropped per chunk length; bucket/grow-only fix is next).

| 2026-06-06 | **Grow-only arena + grow-only boundary buffers** (slots persist across T changes, 1/8 headroom; recording keyed by (T, arena-generation), `rec_L_cp` carried with it) + **stream.py gapless scheduling** (hard-split punctuation-free spans, graduated chunk caps 8→12→14 words, jitter-buffer hold computed from chunk-0-calibrated duration estimates) | per-chunk RTF **0.96–1.01** across varying-T chunks (was the churn case); sustained **1.00–1.01** | **underrun 0.00s** (was 2.70s on the same paragraph) | A/B: A-B(3.2× longer T)-A-A serve sequence → all A wavs **byte-identical** (cross-T pollution + capture/replay identity) | **GAPLESS achieved** via front-loaded hold (~3.1s on a 15s paragraph, TTFA 5.4s). Hold scales with text length while sustained RTF ≈ 1.0 — hold-free gapless needs device RTF ≲ 0.9 (next structural target) |

| 2026-06-08 | **int8 dot8 for the skinny GEMMs** (`linear_skinny_i8`, plan #1: extend `qcom_dot8_acc` from the resblock convs to the fp16 plBERT/predictor/flow GEMMs — signed per-n weights × unsigned per-m acts zp=128, dequant `sx·sw·(A−128·Σwq)`; weight pack cached per-W, `NNOPT_DOT8_GEMM=1`) | warm-stream **0.92–0.93** / single-shot 2.26 (was 2.41) | **~1.09× warm** | all 120 GEMMs cos≥0.998; frame-RMS env cos 0.9989; log-STFT cos 0.9957; raw-waveform cos 0.42 = the phase trap (don't use it) | GEMM bucket 648→~490 ms (1.32×; big g16384x21 got full 1.61×, 61× small g12288x8 only 1.27× — quant overhead at small M). wpack 38 ms is one-time (cached chunk 0). int8 clip 1.6% shorter (duration-predictor rounding, within 5%). **Ear-check PASSED ("identical") → default ON** (`NNOPT_DOT8_GEMM=0` reverts) |

Re-validated 2026-06-06 night from a clean release build + redeploy: identity
gate PASS (A-B-A-A-A: cross-T reuse + replay all byte-identical), varying-chunk
soak sustained RTF 1.01/0.99/1.00, A/B ×2 vs `NNOPT_RECORD=0` identical within
DVFS noise on varied chunks (recordings never replay when every T is new — the
win is the killed realloc churn).

Defaults hardened 2026-06-06 (the config trap where stale `NNOPT_HYBRID_RESBLOCKS=1` in say.sh cost 8.3 s vs 5.1 s can no longer happen):
- `NNOPT_HYBRID_RESBLOCKS` / `NNOPT_HYBRID_FUSED` / `NNOPT_HYBRID_PROFILE` **deleted** along with the obsolete May-25 hybrid resblock path (`hybrid_adainresblock1`, `dec_fused_adain_snake_conv1d_v2`, `fused_combine_snake_conv1d_fast_c4` kernel, hp_* accumulators). The fused-kernel learning (ALU recomputation beats buffer-traffic savings on Adreno 620) is preserved here and in campaign memory.
- `build.sh` defaults to **Release** (`--debug` opt-in) and **fp16**; deploy/run scripts default fp16. Bare `./scripts/build.sh` = fastest binary.
- README.md created: build/run/A-B-toggle documentation; fast paths are default-on, no env needed except `NNOPT_GPU_FP32_GENERATOR=1` (set by say.sh).

## Current shipped config

```
runtime: NNOPT_GPU_FP32_GENERATOR=1   (set by say.sh / stream_device.sh)
default-ON fast paths (no env): NNOPT_DOT8=1 (resblock convs), NNOPT_DOT8_GEMM=1
  (skinny GEMMs), texture convs, GPU iSTFT, grow-only arena. Each <flag>=0 reverts.
build: scripts/build.sh (defaults --release + fp16). espeak-ng vendored → --stream.
```
(`NNOPT_HYBRID_RESBLOCKS` no longer exists — the May-25 hybrid path was deleted; see note above.)

## Optimization ledger (HYBRID_OPTIMIZATION_PLAN.md top-10)

| # | Item | Status |
|---|---|---|
| 1 | ups[0] ConvTranspose1d Cout-tile (`convtr1d_f32_fast_c4`) | ✅ shipped (default when C_out%4==0) |
| 2 | AdaIN+Snake+Conv fusion (`dec_fused_adain_snake_conv1d_v2`) | ⚠️ implemented, **regressed** → opt-in `NNOPT_HYBRID_FUSED=1`, default OFF |
| 3 | `__constant` γ/β/α | ✅ shipped |
| 4 | `cl_qcom_recordable_queues` | ❌ not implemented |
| 5 | GPU conv_post + GPU iSTFT | ✅ kernels exist; active fp32-gen path still uses host fp64 iSTFT (~0.1 s, precision-safe) |
| 6 | BERT matmul optimization | ✅ via skinny-GEMM fast path (default ON, `NNOPT_SKINNY_GEMM=0` reverts) + image2d texture path for M≥8 |
| 7 | Predictor LSTM | ✅ `lstm_seq` whole-recurrence single-launch kernel |
| 8 | LOCAL_T auto-query (`CL_KERNEL_WORK_GROUP_SIZE`) | ❌ not implemented (perf-neutral portability item) |
| 9 | image2d weights for resblock convs | ⛔ skipped by design (no cross-WG W reuse) |
| 10 | Subgroup shuffle broadcast | ❌ not implemented (modest expected gain) |

Also tried beyond the list: int8 conv path (`NNOPT_INT8_CONV`) — **does not help on Adreno 620**, documented in decoder.cpp; fp16 conv variants (`NNOPT_HT8/HT48/H16CONV/HMATH`).

## Optimization ledger (Jun-5/6 session top-10, "bus/latency-bound" list)

| # | Item | Status |
|---|---|---|
| 1 | Weights through texture pipe (image + `read_imageh`) | ✅ shipped, default ON (`NNOPT_TEXW`, `conv1d_ht*` kernels) |
| 2 | RGBA16F texel packing | ✅ shipped with #1 (`get_wn_conv_weight_image`, kpad to %4) |
| 3 | Persistent-kernel LSTM | ✅ shipped (`lstm_seq`: 280 ms/484 launches → 74 ms/12) |
| 4 | Unblock decode-conv1 GEMM path (K·C_in pad 3270→3272) | ✅ shipped (`weightnorm_*_kpad`) |
| 5 | instnorm_adain_snake_h2h coalescing anomaly | ✅ resolved — f16out twin at 2.2 ms in fresh profile, no 17.9 ms entries |
| 6 | GPU iSTFT | ❌ active path still host fp64 (~0.12 s incl. readbacks) |
| 7 | Recordable queues (`cl_qcom_recordable_queues`) | ❌ not implemented |
| 8 | Dual in-order queues (overlap text_enc/BERT, sinegen on thread) | ❌ not implemented |
| 9 | 8-ol half8 tiles in hh conv | ⚠️ tried, **spills**: ht48 5.47 vs 3.93 s, ht8 5.97 vs 4.30 s → default OFF, documented in code |
| 10 | Init/tail trims (zero-copy staging, deferred uploads, p99 sort) | ◐ partial: p99 sort **removed entirely** by the audio epilogue fix; staging/deferred uploads not done |

## Fresh profile 2026-06-06 — fast path (5.32 s wall with profiling overhead, 1.90 s clip)

Top kernels (% of GPU time):
| Kernel | ms | % | Notes |
|---|---|---|---|
| conv1d_ht_t8x4 (128ch, T=9121) | 831 | 19.6% | resblocks_345 + noise_res[1] convs (texture weights) |
| conv1d_ht_t8x4 (256ch, T=1520) | 733 | 17.3% | resblocks_012 + noise_res[0] convs |
| convtr1d_f32_fast_c4 ups[1] | 476 | 11.2% | still fp32, no texture path |
| convtr1d_f32_fast_c4 ups[0] | 430 | 10.1% | |
| **weightnorm_recon (all sizes)** | **506** | **11.9%** | recomputed 12×/run — cache the reconstructed W once ⚠️ |
| plain_conv1d_f32 (noise_convs[0]) | 222 | 5.2% | scalar kernel, no fast path |
| linear_skinny_m4_tex (BERT/predictor) | ~540 | 12.7% | |
| pad_rows | 128 | 3.0% | |
| lstm_seq | 74 | 1.7% | was 280 ms / 484 launches before whole-recurrence kernel |

Next levers (from this profile): cache weightnorm_recon (−0.5 s) · texture-path the two convtr1d ups (−0.4–0.5 s) · fast noise_convs[0] (−0.2 s) · recordable queues for launch gaps (−0.2–0.3 s) → ~3.7–4.0 s ≈ RTF 2.0. Below that needs the conv inner loop itself to go faster (texture pipe already in) or int8.

## On-device C++ streaming (--stream) — no Python

2026-06-08: streaming orchestration moved off the host `stream.py` into the C++
binary. `--stream` does text chunking + G2P (espeak-ng, on-device) + per-chunk
`forward_graph` synth + raw int16 LE PCM to stdout, entirely on-device. The host
side is now just a player pipe (`scripts/stream_device.sh` → ffplay); no Python
in the path.

- **espeak-ng**: cross-compiled static lib for arm64-v8a at **android-21** (must
  match the port's API level — android-28 build fails to link with
  `undefined symbol: stderr`, which is the bionic `__sF[]` macro below API 23).
  Merged libespeak-ng + ucd + speechPlayer + sonic objects into one fat
  `src/third_party/espeak-ng/libespeak-ng.a`. CMake auto-enables
  `NNOPT_TTS_STREAMING` when the lib is vendored; absent → `--stream` is a clean
  error and the rest of the build is unchanged.
- **G2P**: `src/phonemizer.{h,cpp}` wraps `espeak_TextToPhonemes` (IPA mode,
  `phonememode=0x02`) and longest-match maps onto Kokoro's vocab
  (`assets/phoneme_vocab.tsv`, distilled from config.json::vocab). espeak is
  Kokoro's documented fallback phonemizer — its inventory ≈ misaki but not
  identical; symbols absent from the 114-phoneme vocab are dropped (logged at
  `NNOPT_DEBUG_LAYERS=1`), a known intelligible-but-not-misaki-exact tradeoff.
- **chunker**: faithful C++ port of the retired `stream.py` chunk_text (same
  graduated word caps + clause splits), so streamed prosody is unchanged.
- **style**: reuses the `--serve` style-by-token-length selection from
  `voice_pack_af_heart.bin`.

Measured on device (`adb exec-out … --stream` | ffplay), 4-chunk paragraph:

| chunk | text | audio_s | synth_s | RTF |
|---|---|---|---|---|
| 0 | "The teacher worked at the" | 1.98 | 2.21 | 1.118 (cold) |
| 1 | "school for many years." | 1.83 | 1.86 | 1.018 |
| 2 | "She loved teaching children to read," | 2.58 | 2.59 | 1.006 |
| 3 | "and every morning she arrived early to prepare her classroom." | 4.10 | 4.01 | **0.977** |

Warm chunks sit at/under realtime (same `forward_graph` as `--serve`). Run with
`scripts/stream_device.sh "text" [voice]`. Output stdout is **pure PCM** — device
stderr is redirected to `device_stream.log` on-device (else `adb exec-out` merges
it into the PCM). Consumer rate is 24 kHz mono s16le.

## Fresh profile 2026-06-08 — single-shot fast path (NNOPT_PROFILE=1)

29 tokens → 48 600 samples (2.025 s audio), GPU compute **1940 ms** (≈0.96 RTF;
the 2.42 s `total_inference_sec` includes the profiler's per-kernel clFinish).
97% of wall is GPU kernels — compute-bound, not launch-bound.

GPU time by family:
| Family | ms | % | Note |
|---|---|---|---|
| int8 resblock convs (dot8) | 718 | 37% | already hw `qcom_dot8` |
| **skinny GEMMs** (plBERT/predictor/flow) | 648 | 33% | **still fp16, NOT int8 — biggest untapped lever** |
| convtranspose upsamples | 118 | 6% | still fp32 |
| weightnorm recompute | 100 | 5% | **one-time cold build, already cached across chunks** |
| conv glue (im2col/pad/transpose/bias) | 94 | 5% | |
| LSTM (predictor) | 85 | 4% | |
| AdaIN/snake/instnorm | 63 | 3% | |
| attention/layernorm/small linears | 59 | 3% | |
| int8 quant/pack overhead | 47 | 2% | i8_wpack (weights) cached; i8_quant_lc/absmax (acts) per-call |

**Correction to the stale "cache weightnorm_recon (−0.5 s)" lever below:** the
weight-norm reconstruction, int8 weight pack, and texture-image writes are
**ALREADY cached across chunks** — static prefix-keyed maps over persistent
`clCreateBuffer`/`clCreateImage` (`g_wn_cache`/`g_wn_int8_cache` in decoder.cpp,
`g_wn_cache_fp32`/`g_i8_wcache` in _generator_fp32.cpp). The ~100 ms in this
single-shot profile is the one-time cold build; in `--serve`/`--stream` it runs
only on chunk 0 (visible as chunk0 RTF 1.11 → chunk1+ ~1.0). No sustained-RTF win
remains here. The old −0.5 s figure was against a pre-caching 506 ms recon.

**Real path to 2×:** 70% of GPU time is two matmul-shaped buckets — int8 convs
(done) + **fp16 skinny GEMMs (untapped)**. Extending hardware dot8 int8 to the
plBERT/predictor/flow GEMMs is the #1 lever (the resblock convs got 1.61× from
`qcom_dot8_acc`; the GEMMs are the same int8 shape, never converted). Then int8
the fp32 convtranspose upsamples, fuse the conv glue, and overlap predictor with
generator. Realistic combined ≈ RTF 0.62–0.67; true 2.0× needs dot8 to transfer
to plBERT near the resblock's full 1.6×.

**Levers #3/#4 assessed 2026-06-08 → deferred (diminishing returns).** After the
int8 GEMM win (warm 1.0 → 0.92), the remaining buckets were investigated and judged
not worth the effort:
- **#3 squeeze the int8 resblock convs (37%):** kernel is already at the wall —
  OCT=4/OL=4 both spill if grown, inner loop is 8 loads / 64 `qcom_dot8_acc` (256
  MACs) with 4× reuse on both operands → issue-bound at the int8 hw ceiling
  (4 MAC/instr); quant overhead only ~30 ms. The one load-cutting idea (LDS-stage
  activations) adds barriers, recorded by the §9.2.1 sweep as a regressor for this
  kernel class. Realistic ≤5% with regression risk → not done.
- **#4 int8 the fp32 convtranspose upsamples (6%):** ~2% for a real rewrite —
  convtranspose vectorizes over space not channels, so dot8 needs an axis flip + a
  **per-stride-phase** zero-point correction (`wsum[oc][p]`) + zp-filled OOB rows.
  Effort:reward too poor at this stage → not done.

**Bottom line:** int8 GEMM was the last high-value kernel lever. Warm RTF ~0.92
(~1.09× realtime). True 2.0× (RTF 0.5) is a 1.84× further speedup NOT reachable by
kernel tuning on this GPU/architecture — needs a structural change (distillation/
smaller model, lower sample rate, or dropping the fp32 generator), same conclusion
as the musicgen realtime campaign.

## Rules

- Always benchmark `--release` builds (debug adds per-GEMM clFinish).
- Validate kernel changes via on-device A/B with the env toggles, never wav-vs-reference cosine (duration mismatch predates the campaign).
- Update this file after every validated milestone.
