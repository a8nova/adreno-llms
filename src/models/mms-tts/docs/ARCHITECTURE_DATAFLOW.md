# MMS-TTS Dataflow & Architecture (Adreno OpenCL port)

End-to-end picture of how an input string becomes a 16 kHz waveform on a Razr 2020 (Snapdragon 765G / Adreno 620), with the host↔GPU split, tensor shapes, the modules we ported, measured latencies, and where parallelism is realistically available.

Upstream model: **`facebook/mms-tts-eng`** (VITS-family — non-autoregressive TTS with stochastic duration prediction + normalizing-flow latent + HiFi-GAN vocoder). 762 tensors total.

**Current best RTF (long English, 7.81s audio):** **1.66**, achieved with `NNOPT_FLOW_GPU=1 NNOPT_FLOW_WN_FUSED=1 NNOPT_PREWARM=1` (default).

---

## TL;DR — the pipeline in one picture (post-fused-WN landed)

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                                  CPU (host)                                    │
│                                                                                │
│   text  ──▶  uroman normalize ──▶  tokenize ──▶  int32[T_chars] input_ids      │
│                                                                                │
└────────────────────────────────────┬──────────────────────────────────────────┘
                                     │ clCreateBuffer + COPY_HOST_PTR
                                     ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                                  GPU (Adreno) — stays on device end-to-end     │
│                                                                                │
│  ids_buf ──▶ [1] text_encoder ──▶ enc_hidden [T_chars, 192]                    │
│                                   stats      [T_chars, 384]   (means‖logvars)  │
│                                                                                │
│  enc_hidden ──▶ [2] duration_predictor ──▶ log_durations [T_chars]             │
│                                                                                │
└──────────────────────────────────┬─────────────────────────────────────────────┘
                                   │ clEnqueueReadBuffer (BLOCKING, T_chars × 2B) ◀── sync point #1  (essential — T_frames is needed to size everything downstream)
                                   ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  CPU: [3] host_compute_durations: ceil(exp(log_durations) * length_scale)     │
│       → int32[T_chars] durations, T_frames = sum(durations)                   │
│                                                                                │
│  ──▶ clEnqueueWriteBuffer(durations, BLOCKING)  ◀── sync point #2              │
└──────────────────────────────────┬─────────────────────────────────────────────┘
                                   ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                                  GPU                                           │
│                                                                                │
│  stats + durations ──▶ [4] length_regulator ──▶ expanded_stats                 │
│                                                  [T_frames, 384]               │
│                                                                                │
│  split last dim:    expanded_stats ──▶ means    [T_frames, 192]                │
│                                  ╰──▶ logvars   [T_frames, 192]                │
│                                                                                │
│  GPU transpose (kernels/transpose.cl::transpose_btc_to_ncl):                   │
│       means, logvars ──▶ [192, T_frames]  (channels-first)                     │
│                          (was 4 CL_TRUE round-trips — fixed in this session)   │
│                                                                                │
│  means + logvars + prior_noise ──▶ [5] sample_prior ──▶ z_prior [192,T_frames] │
│                                                                                │
│  z_prior ──▶ [6] flow_inverse — DEFAULT: HOST. OPT-IN: GPU fused.              │
│              with NNOPT_FLOW_GPU=1 + NNOPT_FLOW_WN_FUSED=1                     │
│              (16 dispatches, not 160) ──▶ z_latent [192, T_frames]             │
│                                                                                │
│  z_latent ──▶ [7] vocoder (HiFi-GAN: conv_pre + 4 upsample stages + conv_post) │
│              ──▶ waveform [T_frames * 256]                                     │
└──────────────────────────────────┬─────────────────────────────────────────────┘
                                   │ clEnqueueReadBuffer (BLOCKING, audio)       ◀── sync point #3
                                   ▼
                       CPU: int16 PCM ──▶ output.wav (16 kHz)
```

**With `NNOPT_FLOW_GPU=1 NNOPT_FLOW_WN_FUSED=1`: only 3 host↔GPU round-trips on the critical path** (was 7). All inter-stage data stays on the device. The remaining 3 round-trips are essential: (#1) duration readback for sizing, (#2) durations upload, (#3) final PCM readback.

If `NNOPT_FLOW_GPU=0` (current default — host path is still faster for short utterances): two extra `CL_TRUE` round-trips around `flow_inverse` reappear (z_prior down to host, z_latent back up to GPU).

---

## The seven ported modules

All op functions live in `src/ops/` and are dispatched from `src/ops/backbone.cpp::tts_forward_graph()`.

| # | Stage | File | Where it runs | Measured wall (eng-long 7.8s audio) | Output shape |
|---|---|---|---|---|---|
| 1 | `op_text_encoder` | `src/ops/text_encoder.cpp` | GPU | 1.27 s | hidden `[T_chars, 192]`, stats `[T_chars, 384]` |
| 2 | `op_duration_predictor` | `src/ops/duration_predictor.cpp` | GPU | 0.11 s | log_durations `[T_chars]` |
| 3 | `host_compute_durations` | `src/ops/backbone.cpp` | **CPU (unavoidable)** | <0.01 s | int32 durations, scalar T_frames |
| 4 | `op_length_regulator` + transpose | `src/ops/length_regulator.cpp`, `kernels/transpose.cl` | **GPU** (was CPU before this session) | 0.05 s | expanded `[T_frames, 384]`, transposed means/logvars `[192, T_frames]` |
| 5 | `op_SamplePrior` | `src/ops/sample_prior.cpp` | GPU | 0.06 s | z_prior `[192, T_frames]` |
| 6 | `op_FlowInverse` | `src/ops/flow_inverse.cpp` | **HOST default; GPU fused opt-in** | 3.10 s host / **3.77 s GPU fused** | z_latent `[192, T_frames]` |
| 7 | `op_Vocoder` | `src/ops/vocoder.cpp` | GPU | **7.66–8.77 s** (faster when prev stage on GPU; no upload) | int16 PCM `[T_frames * 256]` |

Vocoder + flow_inverse together are still ~85% of the wall on long-audio runs. Vocoder alone is the largest remaining lever.

---

## Tensor shape evolution

For the gate prompt `"Hello, my name is"` on the Razr 2020:

```
"Hello, my name is"
   │
   │ uroman + tokenize (CPU)
   ▼
input_ids                       int32[31]               // T_chars = 31
   │
   │ text_encoder (GPU)
   ▼
enc_hidden                      fp16[31, 192]           // [T_chars,  HIDDEN_SIZE]
stats                           fp16[31, 384]           // [T_chars, 2*HIDDEN_SIZE]
   │
   │ duration_predictor (GPU)
   ▼
log_durations                   fp16[31]                // [T_chars]
   │
   │ host_compute_durations (CPU)
   ▼
durations                       int32[31]               // sum -> T_frames
T_frames = 113                                          // ≈ 1.81 s of audio  @ 16 kHz / 256
   │
   │ length_regulator (GPU): repeat stats[i] durations[i] times along T axis
   ▼
expanded_stats                  fp16[113, 384]          // [T_frames, 2*HIDDEN_SIZE]
   │
   │ split last dim 2 (GPU)
   ▼
means, logvars                  fp16[113, 192]    each
   │
   │ transpose to channels-first (CPU)
   ▼
means, logvars                  fp16[192, 113]    each
   │
   │ sample_prior (GPU): z = means + exp(logvars) * noise * NOISE_SCALE
   ▼
z_prior                         fp16[192, 113]
   │
   │ flow_inverse (CPU, 4 coupling layers)
   ▼
z_latent                        fp16[192, 113]          // same shape, transformed
   │
   │ vocoder (GPU, HiFi-GAN)
   ▼
   conv_pre              :  [192, 113]  → [512, 113]    // K=7
   upsample stage 0 ×8   :  [512, 113]  → [256, 904]    // convT + 3-branch resblocks (avg)
   upsample stage 1 ×8   :  [256, 904]  → [128, 7232]
   upsample stage 2 ×2   :  [128, 7232] → [64,  14464]
   upsample stage 3 ×2   :  [64, 14464] → [32,  28928]
   conv_post             :  [32, 28928] → [1,   28928]  // K=7, tanh
   ▼
PCM samples                     int16[28928]            // 28928 / 16000 = 1.81 s
   │
   │ host int16 → wav
   ▼
output.wav
```

**Upsample factor**: 8 × 8 × 2 × 2 = 256 samples of audio per T_frames frame. SAMPLING_RATE = 16000.

---

## Vocoder zoom-in — where the wall actually goes

The vocoder is 84% of wall on short prompts. Internally it's `conv_pre` + 4 upsample stages + `conv_post`:

```
z_latent [192, 113]
   │
   ▼  ─────  conv_pre  ──────  (1 conv, K=7) ─────────────────────────────────  ~1.1 s (first-call JIT + weight image pack)
[512, 113]
   │
   ├─ leaky_relu
   │   convT (K=16, stride=8)        →   [256, 904]
   │   3 parallel ResBlock branches (kernel sizes 3, 7, 11; dilations 1, 3, 5)
   │     each branch:  3 × ( conv1d(K=ks, dil=d) → conv1d(K=ks, dil=1) + skip )
   │   branch_reduce_3   (b0 + b1 + b2) / 3
   │                                                                             ≈ 1.66 s
   │
   ├─ leaky_relu                     →   [128, 7232]    ≈ 1.77 s
   ├─ leaky_relu                     →   [64,  14464]   ≈ 1.62 s
   ├─ leaky_relu                     →   [32,  28928]   ≈ 1.66 s
   │
   ▼  leaky_relu + conv_post (K=7) + tanh + clEnqueueReadBuffer (BLOCKING)        ~0.13 s
PCM int16[28928]
```

Per upsample stage we dispatch **~57 OpenCL kernels** (1 leaky + 1 convT + 3 branches × 3 kpairs × 3 dispatches per conv + 1 branch_reduce). Across all 4 stages: ~228 dispatches.

Total GPU kernel time across the WHOLE vocoder: **452 ms.** Vocoder wall: **7500 ms.** **GPU is ~5% busy.** The other 95% is per-dispatch latency.

Each `leaky_im2col → hgemm` pair has a measured **75–226 ms gap** (CL_PROFILING_COMMAND_QUEUED → CL_PROFILING_COMMAND_START). This isn't CLBlast — even our own one-line kernels (`leaky_relu`, `branch_reduce`, `bias`) show ~200 ms gaps. Per the Qualcomm Adreno OpenCL Optimization Guide §4.5.1 (Fig 4-1), this is the driver's "QUEUED → SUBMIT" software-overhead window.

---

## Host ↔ GPU sync points — current state

| # | Sync point | Direction | Status | Notes |
|---|---|---|---|---|
| 1 | After duration_predictor — `clEnqueueReadBuffer(log_durations, CL_TRUE)` | GPU → CPU | **Essential, kept** | T_frames needed to size downstream buffers |
| 2 | Before length_regulator — `clEnqueueWriteBuffer(durations, CL_TRUE)` | CPU → GPU | **Essential, kept** | Tiny payload (T_chars ints) |
| 3 | ~~After length_regulator — read means/logvars~~ | ~~GPU → CPU~~ | **REMOVED** | Replaced with `kernels/transpose.cl::transpose_btc_to_ncl` on GPU |
| 4 | ~~Before sample_prior — write transposed buffers~~ | ~~CPU → GPU~~ | **REMOVED** | Same; transpose stays on GPU |
| 5 | After sample_prior — read z_prior | GPU → CPU | **Conditional** — only when `NNOPT_FLOW_GPU=0` (host flow). When GPU flow on: gone. |
| 6 | Before vocoder — write z_latent | CPU → GPU | **Conditional** — same as #5 |
| 7 | After vocoder — read waveform | GPU → CPU | **Essential, kept** | Need int16 PCM on host to write WAV |

**Net**: 7 sync points → **3 essential + 2 conditional** depending on flow_inverse placement. With `NNOPT_FLOW_GPU=1 NNOPT_FLOW_WN_FUSED=1` the inference is 3-sync-point on the critical path.

---

## Where parallelism actually exists

There is **no inter-stage parallelism on the critical path** — the dependency graph is a chain:
`text_encoder → duration_predictor → host_durations → length_regulator → sample_prior → flow_inverse → vocoder`. You cannot start vocoder until you have z_latent.

The real parallelism budget is:

### A. Intra-stage parallelism (already used; only thing exploitable)

```
text_encoder
   ├── 6 transformer layers (sequential — no help here)
   └── inside each layer: attention is parallel over (T_chars × heads)

vocoder upsample stage
   ├── 3 ResBlock branches (kernels 3, 7, 11)
   │   ◇── currently SERIAL: branch 0 → branch 1 → branch 2 → branch_reduce
   │   ◇── could be MERGED into a single 3× wider GEMM per kpair
   │       (same K, same N=L_out, 3× wider M=C_out)  ──── potential 3× fewer dispatches in resblocks
   └── 3 kpair × 2 convs serial inside each branch (data-dependent)
```

The 3 parallel ResBlock branches inside each upsample stage are mathematically independent up to the final `branch_reduce_3`. We currently dispatch them serially (9 sequential convs per stage instead of 3 parallel groups of 3). On a 1-CU GPU like Adreno 620, splitting them across "the same CU" doesn't add throughput, but **batching them into one wider HGemm call would cut dispatch count 3×** — directly attacking the 75 ms/dispatch overhead. This is the realistic kernel-merge target.

### B. Host/GPU overlap (untapped)

While the GPU runs the vocoder, the CPU is idle. While the CPU runs `flow_inverse`, the GPU is idle. Two ideas:

1. **Streaming chunked vocoder.** The vocoder is a fully convolutional stack — `z_latent[:, :t1]` can produce `wav[:, :t1*256]` if you tolerate edge effects at chunk boundaries (overlap-and-add). That lets you start playing audio while later frames are still being synthesized. Latency to first audible sample drops massively even if total wall is the same.
2. **Pipelined utterance batching.** In `--interactive` mode (`say.sh`), the next utterance's text_encoder can run on the GPU concurrently with the previous utterance's PCM being written to disk. Modest CPU/GPU overlap.

### C. Reduce dispatch count — landed wins this session

| Lever | Status | Measured impact |
|---|---|---|
| GPU transpose of means/logvars (`kernels/transpose.cl`) | **LANDED** | 4 sync points removed; perf neutral but unblocks pipelining |
| Pre-warm at load time (`NNOPT_PREWARM=1` default) | **LANDED** | −1.25 s vocoder wall on 2nd+ utterance, kills first-call CLBlast JIT |
| Fused WaveNet residual kernel (`kernels/flow_wn_fused.cl`) | **LANDED, opt-in** | `NNOPT_FLOW_GPU=1 NNOPT_FLOW_WN_FUSED=1`: 16 dispatches instead of 64. Brought GPU flow from 6.79 s → 3.77 s on long Eng (still slightly above host's 3.10 s, but vocoder also speeds up because z_latent stays on GPU). Net: long-Eng RTF 1.75 → 1.66. |
| `clFlush` insertion | TRIED, NO HELP | Measured 0.7× (slower). Default off, kept env-gated. |
| `cl_qcom_recordable_queues` integration | TRIED, NO HELP | A separate microbench `src/bench_recordable.cpp` proved replay is 0.7× — driver dispatch isn't the bottleneck. Infrastructure left in `opencl_context.{h,cpp}` for future use. |
| Image2D conv path (`conv1d_image_tiled_h4`) | TRIED, REGRESSED | ~10× slower per kernel than CLBlast HGemm on these shapes. Available behind `NNOPT_VOC_RESBLOCK_IMAGE=1`, default off. |

### C′. Levers still on the table to hit RTF ≤ 1.0

| Lever | Effort | Expected RTF (long Eng, 7.81s audio) |
|---|---|---|
| **w_in storage reorder `[c,ci,k]` → `[c,k,ci]` + vload_half4 in fused WN kernel** | 2–3 hours | ~1.36 (flow 3.77 → 1.5 s) |
| **Vocoder ResBlock fusion** (3-conv-stack → 1 kernel per branch, same pattern as fused WN) | 1 day | ~1.23 (vocoder 7.66 → 4.4 s) |
| **A + B together** | 1+ day | **≤ 1.0 ✓ realtime** |
| Text_encoder transformer-layer fusion | 2-3 days | ~1.55 (TE 1.27 → 0.3 s) |
| int8 weights for vocoder convs (LFM2-style script + new fp16-accumulator kernels) | 2-3 days | ~1.50 (uncertain — depends if BW is the bind) |
| Streaming chunked vocoder | 0.5 day | RTF unchanged at 1.66 but first audio in ~500 ms — "realtime to the listener" |

---

## OpenCL dispatch graph (vocoder, one upsample stage)

```
                            ┌─── leaky_relu (1 dispatch)
                            │
                            ▼
                      convT (zero_stuff + conv) — 2 dispatches
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
   branch 0 (K=3)       branch 1 (K=7)       branch 2 (K=11)
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │ kpair 0:     │     │ kpair 0:     │     │ kpair 0:     │
   │  3 disps     │     │  3 disps     │     │  3 disps     │       ← conv1d_gemm_fused
   │ kpair 1:     │     │ kpair 1:     │     │ kpair 1:     │         = im2col + hgemm + bias-fused
   │  3 disps     │     │  3 disps     │     │  3 disps     │           but the bias-fuse landed
   │ kpair 2:     │     │ kpair 2:     │     │ kpair 2:     │           so effectively 3 disps × 3 kpair
   │  3 disps     │     │  3 disps     │     │  3 disps     │
   └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
          │                    │                    │
          └────────────┬───────┴──────────┬─────────┘
                       ▼                  ▼
                   branch_reduce_3 (1 dispatch — fused 3-way mean)

Total per stage: 1 + 2 + (3 × 3 × 3) + 1 = ~31 profiled dispatches
                (some are batched in fused kernels — measured at ~57 events)
Wall time per stage: ~1.66 s
Kernel time per stage: ~110 ms
```

---

## Memory traffic (one inference, eng-short)

| Buffer | Size | Path |
|---|---|---|
| `input_ids` | 31 × 4 B = 124 B | host → GPU |
| `enc_hidden` | 31 × 192 × 2 = 11.6 KB | GPU-only |
| `stats` | 31 × 384 × 2 = 23.3 KB | GPU-only |
| `log_durations` | 31 × 2 = 62 B | **GPU → host** |
| `durations` | 31 × 4 = 124 B | **host → GPU** |
| `expanded_stats` | 113 × 384 × 2 = 84.7 KB | GPU-only |
| `means`, `logvars` | 113 × 192 × 2 × 2 = 84.7 KB | **GPU → host → GPU** (transpose) |
| `z_prior` | 192 × 113 × 2 = 43.4 KB | **GPU → host** (for flow) |
| `z_latent` | 192 × 113 × 2 = 43.4 KB | **host → GPU** (post-flow) |
| Vocoder intermediates | up to 32 × 28928 × 2 = 1.77 MB | GPU-only |
| `output_waveform` (fp16) | 28928 × 2 = 56.5 KB | **GPU → host** |

The blocking transfers themselves are small (totals <300 KB). The cost isn't bandwidth — it's the synchronization stalls each `CL_TRUE` introduces.

---

## Read order for the actual code

1. `src/main.cpp:730–820` — wires tokenizer → forward → benchmark
2. `src/ops/backbone.cpp::tts_forward_graph` — the one-screen pipeline (line 270 onwards)
3. `src/ops/text_encoder.cpp` — 6-layer Transformer over T_chars
4. `src/ops/duration_predictor.cpp` — small conv stack
5. `src/ops/length_regulator.cpp` — segment-repeat kernel
6. `src/ops/sample_prior.cpp` — single fused affine kernel
7. `src/ops/flow_inverse.cpp` — host path (default) at lines 502–579; GPU path remains gated by `NNOPT_FLOW_GPU=1`
8. `src/ops/vocoder.cpp` — `op_Vocoder` is the giant function at line 1377; the per-stage loop is at line 1401

## Env vars / toggles available today

| Env var | Default | Effect |
|---|---|---|
| `NNOPT_PROFILE` | off | per-kernel CL event profile + GPU timeline + top-10 gap ranking with bracketing kernel labels |
| `NNOPT_PREWARM` | **ON** | one synthetic forward at load time bakes JIT + caches → first utterance ~1.25 s faster |
| `NNOPT_WARMUP` | off | extra discard-output warmup right before the user's input. Use when measuring steady-state with the actual input shape |
| `NNOPT_FLOW_GPU` | off | flow_inverse on GPU. Slower than host alone on short utts unless paired with: |
| `NNOPT_FLOW_WN_FUSED` | off | use `kernels/flow_wn_fused.cl` for each WaveNet layer. 64 → 16 dispatches inside flow. Combine with `NNOPT_FLOW_GPU=1` for the best long-audio RTF. |
| `NNOPT_VOC_SUBSTAGE` | off | host-side `clFinish`-bounded timing around each vocoder substage (inflates total wall; per-stage ratios valid) |
| `NNOPT_VOC_RESBLOCK_IMAGE` | off | `conv1d_image_tiled_h4` path for vocoder convs. Empirically ~10× slower per kernel on these shapes — keep off. |
| `NNOPT_VOC_FLUSH`, `NNOPT_VOC_FLUSH_FINE` | off | clFlush per-conv or per-dispatch. Measured no-help on Adreno 620; left for future-driver A/B. |
| `NNOPT_FLOW_HOST` | off (n/a when `NNOPT_FLOW_GPU=1`) | force host flow path |
| `NNOPT_VERBOSE` | off | dump token IDs |
| `NNOPT_TICKS` | off | emit `TTS_CHAR_TICK` / `TTS_PCM_CHUNK` markers for a host-side karaoke renderer |

**Recommended invocation for best long-audio RTF today:**
```
NNOPT_FLOW_GPU=1 NNOPT_FLOW_WN_FUSED=1 ./mms_tts_eng_inference_fp16 "<text>" 1 --lang eng
```
(prewarm is already on by default)

---

## Where to attack next — RTF 1.66 → ≤ 1.0

1. **`w_in` storage reorder + vectorize the fused WN kernel inner loop** — kernel goes 221 ms → ~80 ms/call → flow_inverse 3.77 s → ~1.5 s. RTF ~1.36. 2–3 hours.
2. **ResBlock conv fusion in vocoder** (same pattern as the fused WN kernel, applied to the 3-conv-stack inside each upsample stage) — vocoder 7.66 s → ~4.4 s. RTF ~1.23. 1 day.
3. **(1) + (2) combined** — RTF ≤ 1.0. Realtime achieved on long English. The clean path to the goal.
4. **Streaming chunked vocoder** — RTF unchanged but first audio plays in ~500 ms while the rest synthesizes. Realtime *experience* even without changing the wall. 0.5 day.
5. **Text_encoder transformer-layer fusion** — secondary lever for short-utt RTF. Multi-day.
6. **int8 weight quantization of vocoder** — uncertain gain, multi-day; revisit only after #1 + #2 if still not at realtime.

Items 1 + 2 are the ranked direct path. Item 4 is the alternative if you want "feels realtime" without writing more kernels.
