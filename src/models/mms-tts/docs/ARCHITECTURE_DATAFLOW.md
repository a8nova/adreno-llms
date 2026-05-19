# MMS-TTS Dataflow & Architecture (Adreno OpenCL port)

End-to-end picture of how an input string becomes a 16 kHz waveform on a Razr 2020 (Snapdragon 765G / Adreno 620), with the host↔GPU split, tensor shapes, the modules we ported, measured latencies, and where parallelism is realistically available.

Upstream model: **`facebook/mms-tts-eng`** (VITS-family — non-autoregressive TTS with stochastic duration prediction + normalizing-flow latent + HiFi-GAN vocoder). 762 tensors total.

---

## TL;DR — the pipeline in one picture

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
│                                  GPU (Adreno)                                  │
│                                                                                │
│  ids_buf ──▶ [1] text_encoder ──▶ enc_hidden [T_chars, 192]                    │
│                                   stats      [T_chars, 384]   (means‖logvars)  │
│                                                                                │
│  enc_hidden ──▶ [2] duration_predictor ──▶ log_durations [T_chars]             │
│                                                                                │
└──────────────────────────────────┬─────────────────────────────────────────────┘
                                   │ clEnqueueReadBuffer (BLOCKING)              ◀── sync point #1
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
│  split last dim:                                                               │
│    expanded_stats ──▶ means  [T_frames, 192]                                   │
│                  ╰──▶ logvars[T_frames, 192]                                   │
└──────────────────────────────────┬─────────────────────────────────────────────┘
                                   │ clEnqueueReadBuffer ×2 (means, logvars)     ◀── sync point #3
                                   ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  CPU: transpose [T_frames, 192] → [192, T_frames] (channels-first)            │
│  ──▶ clEnqueueWriteBuffer ×2 (BLOCKING)             ◀── sync point #4          │
└──────────────────────────────────┬─────────────────────────────────────────────┘
                                   ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                                  GPU                                           │
│  means + logvars + prior_noise ──▶ [5] sample_prior ──▶ z_prior [192,T_frames] │
└──────────────────────────────────┬─────────────────────────────────────────────┘
                                   │ clEnqueueReadBuffer (BLOCKING, full tensor) ◀── sync point #5
                                   ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  CPU: [6] flow_inverse  (4 coupling flows, all running on host today)          │
│       z_prior [192,T_frames] ──▶ z_latent [192,T_frames]                       │
│  ──▶ clEnqueueWriteBuffer (BLOCKING)               ◀── sync point #6           │
└──────────────────────────────────┬─────────────────────────────────────────────┘
                                   ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                                  GPU                                           │
│                                                                                │
│  z_latent ──▶ [7] vocoder (HiFi-GAN: conv_pre + 4 upsample stages + conv_post) │
│              ──▶ waveform [T_frames * 256]                                     │
└──────────────────────────────────┬─────────────────────────────────────────────┘
                                   │ clEnqueueReadBuffer (BLOCKING, audio)       ◀── sync point #7
                                   ▼
                       CPU: int16 PCM ──▶ output.wav (16 kHz)
```

Six host↔GPU round-trips on the critical path. Every one is a `CL_TRUE` blocking transfer — pipeline-killing.

---

## The seven ported modules

All op functions live in `src/ops/` and are dispatched from `src/ops/backbone.cpp::tts_forward_graph()`.

| # | Stage | File | Where it runs | Measured wall (eng-short) | Output shape |
|---|---|---|---|---|---|
| 1 | `op_text_encoder` | `src/ops/text_encoder.cpp` | GPU | 0.20–1.70 s | hidden `[T_chars, 192]`, stats `[T_chars, 384]` |
| 2 | `op_duration_predictor` | `src/ops/duration_predictor.cpp` | GPU | 0.02–0.20 s | log_durations `[T_chars]` |
| 3 | `host_compute_durations` | `src/ops/backbone.cpp` | **CPU** | <0.01 s | int32 durations, scalar T_frames |
| 4 | `op_length_regulator` | `src/ops/length_regulator.cpp` | GPU | 0.05 s | expanded `[T_frames, 384]` |
| 5 | `op_SamplePrior` | `src/ops/sample_prior.cpp` | GPU | 0.06 s | z_prior `[192, T_frames]` |
| 6 | `op_FlowInverse` | `src/ops/flow_inverse.cpp` | **CPU (default)** | 0.77–6.60 s | z_latent `[192, T_frames]` |
| 7 | `op_Vocoder` | `src/ops/vocoder.cpp` | GPU | **7.32–7.99 s** | int16 PCM `[T_frames * 256]` |

Vocoder + flow_inverse together are >90% of the wall on every measured run.

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

## Host ↔ GPU sync points (the parallelism killers)

These are the seven blocking transfers on the critical path. Every one stalls the GPU and the CPU at the same time.

| # | Sync point | Direction | Reason | Roughly avoidable? |
|---|---|---|---|---|
| 1 | After duration_predictor — `clEnqueueReadBuffer(log_durations, CL_TRUE)` | GPU → CPU | Need the actual numbers to compute `T_frames` and the per-char duration map | **Hard** — T_frames is needed to size all downstream buffers |
| 2 | Before length_regulator — `clEnqueueWriteBuffer(durations, CL_TRUE)` | CPU → GPU | Upload host-computed int32 durations | Easy: make non-blocking |
| 3 | After length_regulator — read means, logvars | GPU → CPU | Only because we transpose channels-last → channels-first on host | **Yes** — do the transpose in a GPU kernel |
| 4 | Before sample_prior — write transposed buffers | CPU → GPU | Same | Same — fold into a GPU transpose |
| 5 | After sample_prior — read z_prior | GPU → CPU | flow_inverse currently runs on CPU | **Big**: move flow_inverse to GPU and this whole stage vanishes |
| 6 | Before vocoder — write z_latent | CPU → GPU | Same | Same |
| 7 | After vocoder — read waveform | GPU → CPU | Need to write the WAV | Necessary; could be async + WAV write off the hot path |

**The cheapest wins come from killing #2, #3, #4** (host transpose is unnecessary) and from moving flow_inverse back onto GPU (#5, #6 disappear).

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

### C. Reduce dispatch count (the real lever)

This is not "parallelism" but it has the same effect — letting the GPU stay busy longer per host call.

| Lever | What it changes | Realistic savings |
|---|---|---|
| Fuse the 3 ResBlock branches into one wider HGemm | 9 → 3 conv dispatches per stage × 4 stages = 24 fewer | ≈ 1.8 s |
| Move flow_inverse to GPU as one fused kernel per coupling layer | Removes 6.6 s of host work + 2 sync points | ≈ 6.6 s |
| `cl_qcom_recordable_queues` (Adreno guide §9.1.3) — record the vocoder kernel sequence once, replay with arg updates | Eliminates per-call `clSetKernelArg` + driver dispatch-prep cost | Unknown — purpose-built for this workload, theoretically large |
| Pre-warm at model load (move first-call JIT + weight-image packs off the per-utterance path) | Already gated via `NNOPT_WARMUP=1` | ≈ 1.25 s on the 2nd-and-later utterance |
| Fold the channels-last → channels-first transpose into a GPU kernel | Removes 2 sync points | small (~50 ms) but unblocks streaming |

The biggest single win on the table is **moving `flow_inverse` back to GPU** (`NNOPT_FLOW_GPU=1` exists but was tagged 8% slower in May 2026). With the dispatch-count fixes above, the GPU path should now win comfortably.

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

## Profiling toggles available today

| Env var | Effect |
|---|---|
| `NNOPT_PROFILE=1` | per-kernel CL event profile + GPU timeline + top-10 gap ranking with bracketing kernel labels |
| `NNOPT_VOC_SUBSTAGE=1` | host-side `clFinish`-bounded timing around each vocoder substage (conv_pre / each upsample / conv_post) — adds clFinish so total wall inflates; per-stage ratios remain valid |
| `NNOPT_WARMUP=1` | discard-output warmup forward before the timed one. Saves ~1.25 s on the second-and-later utterances |
| `NNOPT_FLOW_GPU=1` | flow_inverse on GPU (slower today by ~8%; revisit) |
| `NNOPT_VERBOSE=1` | dump token IDs |
| `NNOPT_TICKS=1` | emit `TTS_CHAR_TICK` / `TTS_PCM_CHUNK` markers for a host-side karaoke renderer |

---

## Where to attack next, in priority order

1. **Move `flow_inverse` to GPU** — eliminates 6.6 s of host work + 2 sync points. The GPU implementation already exists behind `NNOPT_FLOW_GPU=1`; needs a fresh perf benchmark.
2. **Fold means/logvars transpose into a GPU kernel** — removes sync points #3 and #4. Small wall savings but unblocks any future streaming.
3. **Pre-warm at load time** — bake the `NNOPT_WARMUP=1` warmup behind the "load weights" banner so even the first utterance is fast.
4. **Batch the 3 ResBlock branches into a single wider HGemm** — 24 fewer dispatches per inference, ~1.8 s saved.
5. **Try `cl_qcom_recordable_queues`** (Adreno guide §9.1.3) — purpose-built for the vocoder's repeated kernel sequence; high theoretical ceiling but is a substantial refactor.
6. **Chunked streaming vocoder** — even without changing total wall, drops perceived latency dramatically. Requires careful handling of conv-edge effects at chunk boundaries.

Items 1, 2, 3 are all <1 day of work each and worth ~9 s of wall on the gate prompt combined. Items 4, 5, 6 are the harder long-tail wins.
