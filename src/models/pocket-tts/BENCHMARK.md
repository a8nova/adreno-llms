# pocket-tts — Adreno OpenCL benchmark

Device: **Motorola Razr 2020** (`ZY22D5NLGQ`), **Adreno 620 v2** (Snapdragon 765G).
Workload: `generate`, 23 frames = **1.84 s audio** @ 24 kHz, `noise_std 0.837`,
tokens `364,1143,295,400,278,309,265,263` ("The teacher worked at the.").

## Clean baseline — fp16 RELEASE (`build.sh --release`, 2026-06-24)

Instrumented with `PHASE` stderr timers (host `std::chrono`, `clFinish` at the
prefill boundary; the decode loop self-syncs via the per-frame eos + waveform
`download`). Numbers are a single steady run (warm page cache; ±2% run-to-run).

| phase | time | notes |
|---|---|---|
| OpenCL init | 18 ms | device/context/queue |
| weight load | 1.3 ms | 209 MB mmap, lazy (first-touch paid during prefill) |
| **program build** | **241 ms** | `clBuildProgram` ×3 (`ops/utils/embedding.cl`) — recompiled **every run** |
| **prefill** (134 pos) | **4272 ms** | prime 126-frame KV + 8 text tokens, 6 layers |
| **decode** (23 frames) | **16703 ms** | the loop — backbone T=1 + flow_net + mimi |
| steady per-frame | **726 ms** | mean of frames 2–23 |
| **total compute** (prefill+decode) | **20.97 s** | |
| **RTF** | **0.088** | **11.4× slower than real-time** |
| total wall (process) | ~21.5 s | compute + ~0.26 s one-time init/build |

Correctness (unchanged from the validated port): eos logit rises −4.65 → crosses
−4.0 at frame 15 → positive at frame 18; per-frame RMS envelope silent→speech
(peak 0.334 @frame 10)→silence. Both §8 tells pass.

### Where the time goes
- **Decode loop = 78%** of compute, **prefill = 20%**, one-time init/build = ~1.2%.
- Steady frame 726 ms for ~0.62 GFLOP ⇒ **~0.85 GFLOP/s effective ≈ 0.1% of the
  ~0.9 TFLOPS fp16 peak**. Confirms the handoff diagnosis: the decode path is
  **M=1 (vector×matrix) GEMV + tiny per-op kernel launches**, not bandwidth- or
  ALU-throughput-bound.

### Theoretical vs practical (handoff §5)
- Theoretical floor: ~9 GFLOP / ~0.9 TFLOPS ≈ **10 ms** of pure compute → RTF ≈ 180×.
- Practical: 20.97 s compute → **~2000× off theoretical** (release; debug was ~3000×).
  Release build alone bought ~1.45× over debug (31 s → 21.5 s wall).

## History
| build | compute (prefill+decode) | frame ms | RTF | notes |
|---|---|---|---|---|
| fp16 debug | ~31 s wall | — | 0.059 (~17× slower) | handoff baseline, `NNOPT_LAYER_CHECK` on |
| fp16 release | 20.97 s | 738 | 0.088 (~11.5× slower) | clean baseline |
| + #1 custom GEMV (M=1) | 16.56 s | 577 | 0.111 (~9.0× slower) | backbone 289→162 (1.78×), flow_net 80→41; prefill 4.26→3.31 s |
| + #8 GEMM-map convtranspose | 10.84 s | 327 | 0.170 (~5.9× slower) | SEANet 296→45 (6.5×); the 43.6%-of-GPU kernel; waveform cos 0.999996 |
| + #2b flash attention | 8.65 s | 260 | 0.213 (~4.7× slower) | attention 2861→705 ms (4.06×); waveform cos 0.99866 |
| + #3 GEMM-map conv1d | 8.64 s | 259 | 0.213 (~4.7× slower) | wall-neutral (small convs); kept, cos 0.99996 |
| + #4 workgroup-reduction GEMV | 6.63 s | 195 | 0.277 (~3.6× slower) | gemv 3306→1316 ms (2.5×); coalesced; cos 0.99857 |
| + #5 workgroup-reduction LayerNorm | 6.02 s | 174 | 0.306 (~3.3× slower) | layernorm 628→26 ms (24×); cos 0.99787 |
| + #6 workgroup attention (over KV) | 5.49 s | 156 | **0.335 (~3.0× slower)** | attention 671→170 ms (3.95×); 16→1024 work-items; cos 0.99706 |

### Opt #6 — workgroup attention (parallelize over KV positions) — LANDED ✅
Flash attention (#2b) still ran **one work-item per (tq,h) = 16 work-items** at decode.
Rewrote as workgroup-per-(tq,h) (`attention_wg`): LSZ threads split the KV positions,
3 phases in local mem (scores+max → exp+denom → output parallel over D). **16→1024 work-items.
attention 671→170 ms (3.95×).** Correct: frame-1 eos −4.641 (≈−4.648), waveform cos 0.99706.
Profile now: `gemv_fp16_wg` **62.6%** (near fp16 bandwidth limit) · `attention_wg` 8.2% · rest <2%
(+ ~3.4 s uninstrumented CLBlast GEMMs: convtranspose/conv1d/prefill — next thing to profile).

### Opt #9 — defer GPU→host downloads out of the decode loop — LANDED ✅ (the real bottleneck)
Wiring CL events into the CLBlast calls revealed the truth: **GPU-busy = 2.1 s but wall compute = 5.5 s
→ the GPU was IDLE ~62% of the wall.** Not compute-bound — **host/sync-bound.** Each frame did blocking
GPU→host downloads (eos logit + waveform) that forced the host to wait, so it could never enqueue the
next frame ahead. But the eos logit is **diagnostic only** (the loop runs a fixed n_frames, never breaks
on it) and the waveform is a sink — so nothing in the loop needs a sync. Kept everything on-device,
**download once after the loop** (single `clFinish`). **steady 157→116 ms/frame (1.36×), decode 3644→2653 ms;
waveform cosine 1.000000 (bit-identical).** 5 s audio end-to-end: 12.5 s → **9.75 s**.
Steady-state is now **1.45× slower than real-time** (116 ms compute / 80 ms audio).

### Opt #10 — GEMV workgroup size 64→128 — LANDED ✅
gemv was at ~3.4 GB/s (not bandwidth-bound). lws 64→128 = more parallel loads/row →
**gemv 1280→898 ms (1.43×), steady 116→96 ms/frame.** (lws=256 regressed to 122 ms — 128 is the
Adreno-620 sweet spot.) Waveform cos 0.9990.

### Opt #11/#12 — fused qkv+RoPE+cache (neutral) + buffer pool (win) — LANDED ✅
`qkv_rope_cache` fuses 3×slice_cols+2×rope+2×cache_append → 1 kernel: **bit-identical but
WALL-NEUTRAL** (97→99 ms) — confirms dispatch-count reduction doesn't move wall on Adreno 620.
So the ~23 ms/frame gap was NOT launch count. **Buffer pool** (reuse freed cl_mem by size instead
of per-op clCreateBuffer/clReleaseMemObject — ~250 alloc+free/frame): **99→91 ms/frame, bit-identical;
prefill 1755→1719 ms.** In-order queue makes reuse safe.

### Cross-frame pipelining — RULED OUT by hardware probe
backbone(N+1) and mimi-decode(N) are data-independent, so the plan was to run them on two
command queues. **Probe result: the Adreno 620 driver SERIALIZES two queues — concurrent
(2 queues) = serial (1 queue), ratio 1.00.** Zero overlap → pipelining yields nothing on this
GPU. (`scratchpad/qprobe.c` — 40 busy-kernels, 1-queue vs 2-queue: both 533 ms.) Ruled out
without the large/risky async refactor.

### Opt #13 — host-dispatch profiling + fusion (host-bound attack) — PARTIAL
Precise host-cost instrumentation (`HostTimer` per category) proved we're **host-CPU-dispatch-bound**:
host_enqueue 2073 ms vs gpu_tail 23 ms. Per-frame host = **CLBlast ~52 ms (~19 calls × 2.7 ms setup)**,
run1d ~28 ms, alloc/release 3 ms. (Note: this means **int8 would NOT help** — GPU isn't the bottleneck.)
Three `run1d`-cutting fusions landed (all bit-identical, cos 1.0): **(a) fused bias into the GEMV**
(drops the separate add_bias enqueue), **(b) cache the frame-invariant flow timestep embedding**
(s=0,t=1 constant — drops ~10 kernels + 2 uploads/frame), **(c) read adaLN shift/scale/gate from the
packed `mod` buffer at offsets** (drops slice_cols). **Result: 90→85 ms/frame; 5 s audio 7.99→7.75 s.**
Three custom GEMMs to kill the CLBlast host cost all REGRESSED (can't match CLBlast GPU throughput).
**The CLBlast host setup (~52 ms/frame, ~60% of host) is the wall** — fusion chips run1d ~1 ms each;
crossing it needs either many more fusions or a hand-tiled GEMM.

### Opt #14 — cached voice-prompt KV (prefill) — LANDED ✅
Prefill profiled host-bound too: the **voice-prompt prime (T=126) = 1526 ms = ~87% of prefill**, and it
processes the FIXED `audio_prompt` weight → its KV cache is identical every generation. Compute once,
persist raw fp16 to `weights/voice_kv.bin`, LOAD on later runs (skips the prime). **Prefill 1757→301 ms
(5.8×); the prime drops 1526→2.6 ms. BIT-IDENTICAL output (cosine 1.00000000).** The remaining ~300 ms
prefill is the 8 text tokens (text-dependent, ~210 ms) + KV load. Helps short clips most (fixed cost).

## Session summary (this run): **5 s of audio: ~51 s → 7.75 s end-to-end (6.6× faster). Steady-state 738→85 ms/frame (8.7×). + prefill 1.7 s→0.3 s (cached voice KV).**
## **11.5× → 1.12× slower than real-time (steady-state, 90 ms compute / 80 ms audio).**
Biggest lever was NOT compute — it was **#9 deferring per-frame GPU→host syncs** (GPU was idle 62%),
then GEMV coalescing/lws=128 + buffer pool. Profile: gemv_fp16_wg ~54%, attention 10%, conv 16%.
**This is the fp16 single-stream floor.** Crossing real-time (need ~10 ms/frame more) is blocked:
pipelining ruled out (driver serializes queues), so the only remaining lever is **int8 weights**
(halve GEMV bandwidth — GEMV is the dominant cost). Prefill 1.7 s is one-time (amortizes on long audio).
Winning device pattern: **workgroup-per-output-row reduction with coalesced consecutive reads**
(GEMV, LayerNorm, attention) + **GEMM-mapping strided-reduction kernels via CLBlast** (convtranspose).
Losers on this Adreno 620: fewer-thread vectorization (#7), texture-weight GEMV (#2) — both need
reuse/thread-count this GPU doesn't have. Remaining path to real-time: int8 weights (only way past
the fp16 GEMV bandwidth wall) + instrument/optimize the hidden CLBlast GEMMs.

### Opt #4/#5 — workgroup-reduction GEMV + LayerNorm — LANDED ✅
Profile after flash-attn showed `gemv_fp16` 63% running at ~1 GB/s (7% of peak) — **uncoalesced**:
the per-row kernel has adjacent threads reading addresses K apart. Rewrote as **workgroup-per-row
reduction** (`gemv_fp16_wg`, 64 threads/row reading *consecutive* coalesced chunks, local-mem reduce)
for K≥512. **gemv 3306→1316 ms (2.5×); now ~8 GB/s effective (near bandwidth limit).** Same pathology
in `layernorm` (dispatched `rows` work-items → 1 thread/row at decode): `layernorm_wg` → **628→26 ms (24×)**.
Both correct (frame-1 eos identical, waveform cos ≥0.998). Added `run1d_lws` (explicit local size).
Profile now: `gemv_fp16_wg` 50.9% · `attention` 26.3% · everything else <3% (+ uninstrumented CLBlast GEMMs).

### Opt #2b — single-pass online-softmax (flash) attention — LANDED ✅
The `attention` kernel (one work-item per (tq,h) → only 16 work-items for Tq=1 decode)
did the q·k dot **twice** (separate max + exp passes) with a scalar D loop. Rewrote as
single-pass online softmax (running-max rescale) with `vload_half8` over D (fp32 accum).
**attention 2861→705 ms (4.06×).** Correct: frame-1 eos identical, waveform cos 0.99866.
Profile after: `gemv_fp16` 58.9% · `attention` 12.6% · `conv1d_streaming` 11.3% · `layernorm` 11.2%.

### Opt #1 — custom fp16 GEMV (M=1 decode path)
`gemv_fp16` kernel (one work-item/row, `vload_half8` over K, fp32 accumulate) replaces
CLBlast `Gemm` for all M=1 linears (qkv/out_proj/FFN/flow_net/input_linear/out_eos).
Prefill's T=126 GEMM stays on CLBlast. **Correctness:** frame-1 eos identical to CLBlast
(−4.648), per-frame waveform cos 0.998–0.9998 for frames 1–5 then benign autoregressive
fp16 drift (two valid flow-matching trajectories); RMS envelope + eos trajectory unchanged.

**Section shift:** mimi SEANet is now the clear #1 at **296 ms (52%)**; backbone 162 (28%),
mimi-transformer 75 (13%), flow_net 41 (7%).

### Opt #8 — GEMM-map ConvTranspose1d (the 43.6%-of-GPU kernel) — LANDED ✅
Per-kernel CL profiler (`NNOPT_PROFILE=1`, now wired into `run1d`) exposed the real
culprit: `convtranspose1d_full` was **43.6% of all GPU time** (86 ms/call, ~0.36 GFLOP/s —
strided uncoalesced `Cin` inner loop). Replaced it with `gemm_AtB` (CLBlast: `cols[Cout·K,T]
= W[Cin,Cout·K]ᵀ @ in[Cin,T]`) + a light `convtr_col2im` scatter. Streaming state
(overlap-add / partial) untouched. **Result: SEANet 296→45 ms (6.5×), frame 575→327 ms
(1.76×), RTF 0.111→0.170 (9.0×→5.9× slower).** col2im is now 0.66 ms/call (0.59% GPU).
**Correctness: waveform cosine 0.999996 vs the naive kernel; eos trajectory + envelope identical.**

Per-kernel profile after #8 (decode kernels, excl. CLBlast GEMM which is uninstrumented):
`gemv_fp16` 42.8% · `attention` 37.0% (12 ms/call ×238) · `convtr_col2im` 0.6% · rest <1%.
New section split: backbone 162 (50%), mimi-transformer 75 (23%), seanet 45 (14%), flow 43 (13%).
→ next target: **attention** (now the dominant non-GEMV kernel).

### Opt #2 attempt — texture-weight GEMV (image1d_buffer) — REVERTED (regression)
Routed the M=1 GEMV weights through an `image1d_buffer` view (zero-copy, `read_imagef`,
Adreno texture cache). Device supports it (`image1d_buffer_max=134M` texels). **Backbone
got *slower*: 162→206 ms (+28%).** Correct (frame-1 eos identical). Root cause: a pure
M=1 GEMV reads each weight **exactly once** — no reuse for the texture cache to exploit,
so it's pure sampling overhead (plus 4-wide image reads vs the buffer path's 8-wide
`vload_half8`). Texture weights would only pay where weights are *reused* (GEMM M>1, or
convs across time). Reverted; kept the `IMGINFO` device-limits diagnostic.

### Opt #7 attempt — vectorize-over-T conv1d — REVERTED (regression)
Tried `conv1d_streaming_v8` (one work-item computes 8 consecutive output-t, `vload_half8`
inputs, weight reused across the 8-wide vector). **SEANet got *slower*: 296→323 ms (+9%).**
Correct (frame-1 eos identical), but cutting work-items 8× starves Adreno-620 occupancy —
the naive high-thread-count kernel hides memory latency better. **SEANet conv is
memory-latency bound, not ALU bound.** Lesson: the next SEANet lever must *keep* thread
count high (local-memory weight caching, texture/image2d weights) — not fewer-thread
vectorization. Reverted to the scalar kernel.

## Reproduce
```bash
cd src/models/pocket-tts
export ANDROID_NDK="$HOME/Library/Android/sdk/ndk/android-ndk-r26d"
NNOPT_DTYPE=fp16 ./scripts/build.sh --release
cp build/fp16/_deps/clblast-build/libclblast.so build/libclblast.so   # deploy gotcha
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh
DEV=ZY22D5NLGQ; RD=/data/local/tmp/pocket_tts_inference
TOK=364,1143,295,400,278,309,265,263
adb -s $DEV shell "cd $RD && LD_LIBRARY_PATH=$RD/lib:/system/vendor/lib64 \
  ./pocket_tts_inference_fp16 weights/model.fp16.bin weights/model.fp16.meta.json \
  generate /dev/null out.bin 23 0.837 $TOK"
```
Add `NNOPT_PROFILE=1` once per-kernel `event_for()` instrumentation is wired into
`gpu_ops.cpp` (currently only 3 events in `utils.cpp`; `dump_summary` is never
called — see next step).

> Note: `run_android.sh` is a generic `<prompt> <max_tokens>` template and does
> **not** match this binary's `generate` CLI — use the raw `adb` form above.
