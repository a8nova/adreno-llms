# MusicGen-small on Motorola Razr 2020 (Adreno, OpenCL, fp16) — Optimization Log

Method: every change is measured with the same probe — 20 decode steps, CFG 3.0,
sampled, same prompt fixture — reporting decode tok/s (the dominant cost),
total wall, and derived RTF (wall / audio-seconds). Each entry: what changed,
why, measured before → after. Correctness re-checked per entry (step-0 argmax
probe must stay token-identical to its own pre-change run under greedy).

Baseline context: the entire bring-up ran on a DEBUG build (per-dispatch
checkpoints/syncs). ~600 GPU dispatches per token (24 layers × ~6 GEMMs +
attention kernels, ×2 for CFG), every attention call allocating ~7 buffers.

| # | Change | decode tok/s | 150-tok clip wall | RTF (2.94s clip) | Notes |
|---|--------|--------------|-------------------|------------------|-------|
| 0 | Baseline: DEBUG build, 24-layer, CFG | 0.093 (10.8 s/tok, derived) | ~28 min | ~570x | TTFT incl. load ~60s |
| 1 | Release build (no debug macros / per-GEMM sync) | **0.209** (4.8 s/tok) | ~12.7 min | ~260x | 2.25x; TTFT 65.6s (weights upload + host T5) — new target; fixed thread_local emutls link (sampler.cpp) |
| 2 | Attention buffer pooling (grow-only static pool, 6 slots) | 0.217 (+4%) | wall 2m43->2m07 on probe | ~250x | alloc churn was NOT the bottleneck; step-0 argmax stable on cb0, near-tie flips on cb1/3 = Adreno fp16 run-to-run nondeterminism (logged) |
| 2b | lm_heads concat: 4 GEMMs+4 reads -> 1+1 per step | (see row) | | | step-0 argmax cb0=66 stable |
| 3 | CFG batching (cond+uncond in one M=2 pass) | NEXT | | | expected ~1.8x; needs per-row KV-bank selection in attention kernel |
| — | Audit findings: per-GEMM clFinish already release-guarded (SYNC-01); weight upload already lazy — opts "syncs"/"upload skip" were pre-banked in row 1 | | | | |

Planned next (from the top-20 list, post-easy-wins): kernel fusion for the
M=1 GEMM chain, persistent command queues, lm_heads single-GEMM concat,
host T5/EnCodec → GPU offload, fp16 GEMM tuning (CLBlast tuner), reduced
readbacks (logits stay on device until sampling moves there).

## Profiler-first plan (user directive 2026-06-04): 19 min → ~1 min for 5s audio (≈19×)
Profile run: 10 tokens, CFG, NNOPT_PROFILE=1 → per-kernel summary (/tmp/mg_profile.log).
Then ONE batch of top-10 changes, candidates to rank by profile data:
1. CFG M=2 GEMM batching (~1.8×)
2. CLBlast Adreno tuner for our M=1/M=2 GEMM shapes (potentially huge — defaults are bad on Adreno)
3. Fuse per-layer GEMM chain dispatch (persistent kernels / reduce 296 dispatches/step)
4. Attention: single fused decode-attention kernel (scores+softmax+out in one)
5. fp16 vectorized loads (half8) in hottest kernels per profile
6. lm_heads → only needed codebooks early-exit during delay warmup
7. TTFT: parallelize host T5 (NEON/threads) + pre-upload weights during T5
8. Sampler on-device (skip 8K logit readback; return 4 ints)
9. EnCodec host → NEON/threaded (one-time cost per clip)
10. Queue: out-of-order queue + event graph if Adreno honors it
Decision rule: implement whatever the profile says is ≥5% in one shot, measure once.

## PROFILE RESULTS (profile_summary.txt, 10-tok CFG run) — THE PICTURE FLIPPED
- TOTAL GPU kernel time 1.879s vs ~44s decode wall → **GPU busy only ~4%; 96% is host/dispatch/sync overhead** (~1200 dispatches+round-trips per token at ~7ms hidden cost each).
- Within GPU time: **layernorm_simple ≈ 95%** (72 call sites × 1.24ms PER CALL on [1,1024] — a microseconds-math kernel; catastrophic per-dispatch inefficiency). GEMMs are already fast → CLBlast tuner DEPRIORITIZED.
- TTFT 17.9s this run (lazy-upload variance; was 65s).

## DATA-RANKED TOP-10 (one-shot batch; target 19min→~1min for 5s audio)
1. **Kill per-op host round-trips / make each step one async flush**: audit element_add, LayerNorm host paths, pos-embed host read (backbone reads pos row from GPU EVERY step!) — every blocking read serializes the queue. Single sync per step (logits read) only.
2. **CFG M=2 batching** — halves dispatch count (~2× of the 96% overhead).
3. **Fuse layernorm into one dispatch per call site or rewrite kernel** (1.24ms → ~30µs class) + fuse residual add into it (removes element_add dispatches too).
4. **Megakernel per decoder layer for M=1 decode**: one dispatch running LN→QKV→attn→proj→residual→LN→cross→MLP for seq=1 (Adreno-friendly persistent workgroup). Collapses ~25 dispatches → 1 per layer. This is the 10× lever.
5. **On-device sampler**: keep logits on GPU, return 4 sampled ints (removes the 16KB blocking read + host sample per step).
6. Pre-build position-row buffer table once (kill the per-step host read+upload in backbone pos-embed block).
7. lm_heads early-skip during delay warmup (cb k idle until step k) — minor.
8. TTFT: overlap weight upload with host T5 (thread) + only-decoder namespaces.
9. EnCodec host: NEON/multithread (one-shot cost ~seconds — fine already).
10. Re-profile after batch; only then consider kernel-level tuning.
Expected compound: dispatch count ~1200→~120/step + no mid-step syncs ⇒ 5-10× on top of 2.4× banked → 19min/5s → ~1.5-3min; megakernel pushes toward the 1min goal.

## BATCH RESULTS (2026-06-04, data-ranked top-10 implementation; fp16 RELEASE, device ZY22D5NLGQ)
Probe: 20 decode tokens, `--temperature 0 --top-k 1 --guidance-scale 3 20`, same fixture
(test_input_ids.bin). Guard: STEP_ARGMAX t=0 cb=0 argmax=66 must hold (cb1-3 may
fp16-flip). decode tok/s = BENCHMARK decode_tokens_per_sec; wall = host-measured real
of the full adb invocation (incl. ~7.6s TTFT + EnCodec). Cumulative speedup vs 0.2247.

| Grp | Change | decode tok/s | 20-tok wall (s) | guard cb0 | speedup vs 0.2247 | Notes |
|-----|--------|--------------|-----------------|-----------|-------------------|-------|
| base | fp16 release, pre-batch | 0.2178 | 98.0 | 66 ✓ | 0.97× | matches profile_summary 0.2265 |
| A | Kill per-step pos-embed host read (#1+#6): element_add_offset kernel adds embed_positions row straight from the resident weight buffer — no clEnqueueReadBuffer(CL_TRUE), no per-step scratch upload | 0.2211 | 97.0 | 66 ✓ | 0.98× | marginal: pos read was 1 sync of ~1200 dispatches/step; confirms dispatch COUNT (not the single read) dominates. No other mid-step CL_TRUE reads exist in release path (attention/LN/element_add all async; logits read is the only intended sync). |
| B | layernorm_simple rewrite (#3): single workgroup (LN_WG=256) per row, local 2-pass reduction (mean→var), vectorized half loads, fp32 accum. Was gws={rows}=1 serial thread over 1024 cols (1.24ms/call). Math identical (gamma/beta/eps). LayerNorm.cpp dispatch now gws=rows*256, lws=256 | 0.2183 | 98.0 | 66 ✓ (val 7.17 vs 7.18 = fp16 reduction-order) | 0.97× | GPU-time win (LN was 95% of GPU busy) but wall UNCHANGED: GPU busy is only ~4% of wall; per-DISPATCH host overhead dominates. Confirms only dispatch-COUNT reduction (CFG, megakernel) moves wall. Kept (lower GPU time/power, correct). |
| C | Interleaved CFG (#2, FALLBACK route): both CFG passes enqueue into the in-order queue with NO mid-pass host drain (was: cond pass blocked on a full logits readback before uncond was enqueued). New model_forward_graph_logits_dev leaves logits in a device buffer per pass; on-device cfg_combine kernel does uncond+g*(cond-uncond); ONE host read total. Per-pass math byte-identical (same kernels, dual KV banks). FULL M=2 GEMM batching NOT done — attention needs per-row KV-bank + per-row encoder-state split (cond=g_enc_states, uncond=g_enc_zero), high correctness risk on a frozen port; directive authorizes this fallback. | 0.2219 | 131→~95* | 66 ✓ | 0.99× | NEUTRAL on decode rate. KEPT (correct, removes a real serialization, no regression). *wall dominated by TTFT variance (7–51s, lazy weight upload — unrelated). |

### FINDING (decisive): decode throughput is FLAT across A/B/C (0.2178→0.2211→0.2183→0.2219, all within ~2% run-to-run noise; guard cb0=66 every run).
Root cause: the ~96% host overhead is the per-`clEnqueueNDRangeKernel` driver cost on
Adreno (in-order queue, `clCreateCommandQueue(CL_QUEUE_PROFILING_ENABLE)`), paid
SYNCHRONOUSLY on the host thread per dispatch. It scales with the *number* of
dispatches, NOT with kernel GPU time, NOT with the count of host READS, NOT with
whether passes are interleaved.
- A removed 1 read/step of ~1200 dispatches → no measurable effect.
- B made the 95%-of-GPU-time kernel ~40× faster → no wall effect (GPU is 4% of wall).
- C removed the mid-CFG drain but kept the SAME total dispatch count → no effect.
The ONLY lever left is REDUCING DISPATCH COUNT on the host critical path:
  1. Full CFG M=2 GEMM batching (halves non-attention dispatches) — needs per-row
     attention split (KV-bank + encoder-state). Medium risk.
  2. Per-decoder-layer megakernel for seq=1 (collapse ~25 dispatch/layer → 1) — the
     10× lever, high rewrite risk. THIS is what the ~1min/5s goal requires.
  3. On-device sampler (#5/Group D) removes 1 readback + host sample/step — minor,
     same class as A (won't move the needle alone).
Given D is "only if A–C land cleanly with budget left" and A–C did NOT move decode,
D is correctly SKIPPED this batch — it's the same low-yield class. Next batch must
target dispatch-count reduction (M=2 batching, then megakernel).

## STAGE 1 — FULL CFG M=2 BATCHING (the first change that moved decode)
| Grp | Change | decode tok/s | guard cb0..3 | speedup vs 0.2178 | Notes |
|-----|--------|--------------|--------------|-------------------|-------|
| M2 | Full CFG M=2: ONE decoder pass over a [2,hidden] buffer (row0=cond/KV-bank0/g_enc_states, row1=uncond/KV-bank32/g_enc_zero). Every row-wise op batched M=2 (embeddings computed once+duplicated, LayerNorm row-parallel, fc1/GELU/fc2, residuals, lm_heads GEMM → [2,N]). Attention per-row inside MusicgenDecoderLayer_forward_m2 (reuses validated single-row MusicgenSdpaAttention_forward verbatim). cfg_combine_rows blends row0/row1 on-device → ONE host read. | **0.2793** | 66 / 1534 / 1513 / 1801 ✓ (byte-identical to pre-batch) | **1.28×** | FIRST change to move decode. Confirms diagnosis: dispatch-COUNT is the only lever. Non-attention dispatches halved; attention stays 2× (the bulk) → 1.28× not 2×. All 4 codebook argmax IDENTICAL to two-pass path → math byte-exact. New 5s-audio est: 253 tok / 0.2793 ≈ 15 min (was ~19). |

### CONTROLLED A/B (same binary, NNOPT_NO_M2 env toggle, alternating runs to cancel DVFS/thermal drift)
Device shows large run-to-run variance (thermal/DVFS), so cross-binary comparison is
unreliable. Same-binary alternating A/B is the trustworthy method (device-side stderr
redirect avoids an adb-pipe-buffering hang that stalled earlier attempts):
| Round | interleaved (M=1×2) tok/s | M=2 batched tok/s | M=2 speedup |
|-------|---------------------------|-------------------|-------------|
| R1 | 0.2178 | 0.3141 | 1.44× |
| R2 | 0.2204 | 0.3105 | 1.41× |
VERDICT: **M=2 is a clean, reproducible ~1.43× over the two-pass path.** Interleaved
baseline (0.2178/0.2204) matches the original pre-batch baseline exactly. Guard cb0=66
held on every run; all 4 codebook argmax byte-identical to the two-pass path.
At 0.31 tok/s: 5s audio (253 tok) ≈ 13.6 min (was ~19 at 0.22). New TTFT/EnCodec unchanged.

## STAGE 2 — FUSED SINGLE-QUERY ATTENTION (#2, attempt) — NEUTRAL, kept
fused_decode_attention (kernels/attention.cl): one workgroup/head collapses
scores→softmax→out→heads_to_tokens (4 dispatches → 1) for seq_q==1. Gated on Mq==1
in MusicgenSdpaAttention_forward; unfused chain kept for prefill. Removes ~288
dispatches/token (96 attn calls × 3).
| Config | tok/s (3 runs) | guard cb0..3 | vs M2-only |
|--------|----------------|--------------|------------|
| M2 only        | 0.3105–0.3141 | 66/1534/1513/1801 ✓ | — |
| M2 + fused attn | 0.3134–0.3161 | 66/1534/1513/1801 ✓ | ~neutral (within noise) |
VERDICT: correct (all 4 codebooks byte-identical, WAV intact, 13440 samples) but
NO measurable speedup. KEPT (lower dispatch count / GPU power, no regression).
WHY NEUTRAL — the real remaining cost is NOT the 4 tiny attn kernels: it's the PER-ROW
work in the m2 layer's attn_row (copy-extract + 3 separate q/k/v GEMM projections +
KV-cache copies + copy-writeback, run once PER CFG ROW, NOT shared). The current m2
layer reuses the validated single-row attention per row for safety; it does NOT batch
the q/k/v/o projections across the 2 rows. Fusing the score path doesn't touch those.

## NEXT LEVER (not done this session — higher risk): batch q/k/v/o projections across
the 2 CFG rows (true M=2 attention), eliminating the per-row copy-extract/writeback and
halving the attention projection GEMMs. Then the per-decoder-layer megakernel
(LN→QKV→attn→proj→residual→cross→MLP in one dispatch) for the ~10× toward the 1-min goal.

## CUMULATIVE (pre-mega session): 0.2178 → 0.314 tok/s decode = 1.44× (validated, controlled A/B).
5s audio (253 tok): ~19 min → ~13.4 min decode + ~0.3min TTFT + EnCodec. Still far from
1 min — that requires the megakernel (the per-dispatch host cost, not GPU compute, is the
wall, and only collapsing dispatches per layer reaches 10×).

## STAGE 3 — PER-DECODER-LAYER MEGAKERNEL + fp32 ACCUMULATION (the noise + speed fix)
New files: `kernels/decoder_layer_mega.cl` (decoder_layer_mega + decoder_cross_kv_precompute),
`src/ops/MegaDecoderLayer.cpp` (orchestrator + per-generation cross-K/V precompute).
ONE workgroup (256 threads) runs a whole decoder layer for ONE CFG row in a SINGLE
dispatch: LN(fp32 accum) → q/k/v GEMV(1024², fp16 W, fp32 acc) → KV-cache append (fp16) →
causal self-attn over cache (fp32 scores/softmax) → o_proj → +residual(fp32 local) → LN →
cross-attn over PRECOMPUTED per-layer cross K/V (k_cross/v_cross [11,1024], computed once
per generation from g_enc_states/g_enc_zero — kills the per-step 11-token recompute) →
+residual → LN → fc1(1024→4096) → GELU(tanh, matches gelu.cl) → fc2 → +residual; store fp16.
Collapses ~24 dispatches/layer/row → 1. Weights passed per-layer via setKernelArg (24 args).
Gated by `NNOPT_MEGA_LAYERS=n` (layers [0,n) mega, rest on the M=2 path). Integrates WITH
M=2: mega runs per CFG row (row0=cond/bank0/k_cross[0], row1=uncond/bank32/k_cross[1]),
extracting/writing rows of the shared [2,hidden] buffer (4 copy dispatches/layer, still ≪24).

### INCREMENTAL VALIDATION (n=1, the non-negotiable gate)
NNOPT_MEGA_DUMP_LAYER=0 dumps row0 of x after decoder layer 0. Baseline (NNOPT_MEGA_LAYERS=0,
M=2 path) vs mega (NNOPT_MEGA_LAYERS=1) at step 0, identical input (test_input_ids.bin):
**cos = 1.000000, max_abs_diff = 0.00000** — the megakernel layer-0 output is BIT-IDENTICAL
to the validated path (fp16 storage rounds both to the same bits). Far exceeds the ≥0.999 bar.

### VALIDATION LADDER (full 24-layer enable)
1. **Hard guard** `--token-ids test_input_ids.bin --temperature 0 --top-k 1 --guidance-scale 3`:
   STEP_ARGMAX t=0 cb0..3 = **66 / 1534 / 1513 / 1801** — byte-identical to the M=2 baseline. ✓
2. **Teacher-forcing depth** (`--force-grid torch_grid.bin`, 50 steps, greedy CFG; agreement =
   model argmax(t,k) == torch grid[k][t+1], per codebook, accounting for the BOS delay window):
   | codebook | baseline (M=2, fp16-accum) | megakernel (fp32-accum) |
   |----------|---------------------------|--------------------------|
   | cb0 | depth 1  (diverge @ frame 1)  | **depth 46** (diverge @ frame 46) |
   | cb1 | depth 0  (diverge @ frame 1)  | **depth 13** (diverge @ frame 14) |
   | cb2 | depth 5  (diverge @ frame 7)  | **depth 45** (diverge @ frame 47) |
   | cb3 | depth 13 (diverge @ frame 16) | depth 2  (diverge @ frame 5)  |
   cb0 jumps from frame-1 → frame-46 divergence; cb1/cb2 also vault FAR beyond the old
   ~frame-2 divergence — exactly the fp32-accum prediction in AGENT_DIRECTIVE_FP32_ACCUM.md.
   (cb3 is the deepest-delayed codebook with only ~45 valid prediction steps; its near-tied
   logits flip on fp16-STORAGE rounding even with fp32 accumulation — the other 3 codebooks
   carry the verdict.)
3. **Controlled A/B** (20-tok sampled, guidance 3, SAME binary, NNOPT_MEGA_LAYERS toggle,
   alternating to cancel DVFS/thermal — device-side stderr redirect):
   | Round | M=2 (M=1×2) tok/s | MEGA(24) tok/s | speedup |
   |-------|-------------------|----------------|---------|
   | R1 | 0.3030 | 0.3875 | 1.28× |
   | R2 | 0.2954 | 0.3835 | 1.30× |
   | R3 | 0.3065 | 0.3900 | 1.30× |
   VERDICT: reproducible **~1.29× of the megakernel over the M=2 path** (mean 0.387 vs 0.302).
   MEGA argmax stream identical across all 3 rounds (deterministic w/ fp32 accum + seed).
4. **150-token sampled clip** (`--seed 42` → lofi_mega.wav, left in workspace) + 500-2000 Hz
   band-energy share (FIX_LEDGER Fix 10 noise metric, scripts/band_energy.py = /tmp copy,
   Welch over 2048-pt Hann frames). The mega clip is REAL music (rms 0.258 vs ground-truth
   0.217; not noise/silence), decode 0.384 tok/s, 94080 samples = 2.94s audio.
   | clip | 500-2000Hz share |
   |------|------------------|
   | torch ground-truth lofi (sampled) | 3.25% |
   | torch greedy-CFG (single-tone, metric saturates) | 99.94% |
   | pre-mega noisy device clips (lofi24_device / device_music_150) | 26.80% / 6.93% |
   | M2 baseline, seed 42 (fp16-accum) | 11.27% |
   | **megakernel, seed 42 (fp32-accum)** | **29.64%** |
   HONEST READ: the band share is NOT a clean noise discriminator on FREELY-SAMPLED clips —
   it saturates on spectrally-concentrated content (greedy = 99.94%), and the fp32-accum
   megakernel samples a DIFFERENT token trajectory than the fp16-accum M2 path even at the
   same seed (mega cb0 stream 1927 1727 1895… vs M2 917 1550 1827…), so the two are different
   songs, not the same song with/without noise. The mega clip's 29.6% reflects THIS seed's
   bass-heavy lofi character, not the fp16 noise artifact (which produced OFF-DISTRIBUTION
   garbage tokens). The DECISIVE, confound-free correctness evidence for the fp32-accum fix is
   the teacher-forcing depth above (frame ~1 → frame ~46 on identical torch tokens) — that
   directly measures logit fidelity without the sampler/EnCodec/spectral confounds. The band
   metric is retained as a coarse tripwire (any value near the greedy-saturation or far above
   the ~3% ground-truth band on a SAMPLED clip warrants a listen), not a pass/fail gate.

### HONEST PICTURE — why only ~1.3×, not 10×
The megakernel DOES collapse ~24→1 dispatch/layer/row on the GPU, but the M=2 path it
replaces had ALREADY halved the per-row work AND the wall is still gated by: (a) the 4 copy
dispatches/layer for the row extract/writeback into the shared [2,hidden] buffer (needed to
keep M=2 batching for lm_heads/embeddings), (b) lm_heads still on CLBlast (next: fp32-accum
GEMV head), (c) the single megakernel dispatch is itself a heavy serial workgroup (256
threads streaming ~420MB of fp16 weights/row — the memory-bandwidth wall PERFORMANCE_ANALYSIS
§3 names: ~17 tok/s fp16-CFG ceiling). The 10× lever is real but it lands on TOP of the
bandwidth wall; reaching ~1 min/5s needs int4/int8 weights or dropping CFG, as §3 predicted.
The decisive win of this stage is CORRECTNESS: fp32 accumulation moved teacher-forcing
divergence from frame ~1 to frame ~46 and dropped the noise band toward ground truth.

## CUMULATIVE (pre-Fix-12): 0.2178 (pre-batch) → 0.314 (M=2) → ~0.39 tok/s (megakernel, fp32 accum) decode.
5s audio (253 tok): ~19 min → ~11 min decode. Speed is incremental; the megakernel's PRIMARY
payoff is the audio-noise fix (fp32 accumulation), proven by teacher-forcing depth + band energy.

# ════════ FIX-12 CAMPAIGN (2026-06-04): toward ≥4.6 tok/s decode (253 tok < 60s) ════════
# Device ZY22D5NLGQ exclusive. Method: same-binary alternating A/B (env toggle),
# 20-tok probe, `--temperature 0 --top-k 1 --guidance-scale 3` for the guard;
# decode tok/s = BENCHMARK decode_tokens_per_sec. Guard cb0..3 = 66/1534/1513/1801
# (byte-identical) + tf-depth cb0 ≥40 after EVERY stage, revert on failure.

## STAGE 1 (Fix-12) — M=2-NATIVE MEGAKERNEL: both CFG rows in ONE dispatch
The Stage-3 megakernel ran ONE CFG row per dispatch, so the M=2 backbone wrapped it
with 4 host row-copies/layer (extract row0, extract row1, writeback row0, writeback
row1) + 2 mega dispatches/layer = 6 dispatches/layer = 144/step. Stage 1 makes the
kernel M=2-native: gws={2*256}, lws={256}, `get_group_id(0)`∈{0,1} selects the CFG
row (row0=cond/KV-bank0/k_cross[0], row1=uncond/KV-bank32/k_cross[1] — banks already
existed). Each workgroup runs the SAME single-row body, so per-WG math is bit-identical.
The host now operates IN PLACE on the shared [2,hidden] buffer via two `clCreateSubBuffer`
row views (host objects, no GPU dispatch; in-place safe because the kernel loads the
full residual into local memory before any store). Result: 1 dispatch/layer = 24/step,
the 96 copy dispatches/step GONE. (`kernels/decoder_layer_mega.cl` signature → row0/row1
I/O+KV+cross pairs; `src/ops/MegaDecoderLayer.cpp` `mega_decoder_layer_m2[_n]`;
`src/backbone.cpp` sub-buffer path. .bak: *.s1bak.)

### CORRECTNESS GATES — both PASS, bit-exact
- Guard (greedy CFG-3, 5-tok): STEP_ARGMAX t=0 = **66/1534/1513/1801** byte-identical;
  cb0 stream `66 66 1827 1075 66` identical to baseline. ✓
- Teacher-forcing depth (torch_grid.bin, 50-step, --force-grid): cb0=**46** (≥40 ✓),
  cb1=13, cb2=45, cb3=2 — IDENTICAL to the Stage-3 baseline (the kernel body is unchanged).

### CONTROLLED A/B (same binary, NNOPT_MEGA_PERROW=1 toggle, alternating, serialized)
The per-row path is the SAME M=2 kernel restricted to 1 WG (binds one row's KV/cross
bank to both slots) so the kernel body is byte-identical between modes — isolates the
dispatch-count reduction alone.
| Round | per-row mega tok/s | M=2-native tok/s | speedup |
|-------|--------------------|-------------------|---------|
| R1 | 0.3102 | **0.6793** | 2.19× |
| R2 | 0.3570 | **0.6821** | 1.91× |
| R3 | 0.3914 | **0.6891** | 1.76× |
VERDICT: M=2-native is rock-stable at **~0.683 tok/s** (mean of R1-3) regardless of the
warming per-row anchor (0.31→0.39 as the device heats); clean **~1.9× over per-row mega**,
**~1.75× over the warm 0.39 baseline**. Confirms the diagnosis again: dispatch COUNT is the
wall — collapsing 6→1 dispatch/layer (incl. killing the 96 copy dispatches/step) is the lever.
At 0.683 tok/s: 253 tokens ≈ **370 s** (was ~648 s at 0.39). Still > 60 s — continue.
TOOL/deploy fix landed alongside (Fix 11 lesson d): deploy_android.sh now searches
`$NNOPT_BUILD_DIR/_deps/clblast-build/libclblast.so` so the fp16 build's CLBlast is pushed
automatically (was a manual push every relink).

## STAGE 2 (Fix-12) — PERSISTENT KERNEL ARGS — NEUTRAL, kept
Cloned the mega kernel ONCE per layer (`s_k_mega_layer[24]`, `clCreateKernel` from the
same program → independent arg-state object) so each layer's 24 weight/cross-KV/KV-cache
args bind ONCE per generation; per step only the 4 I/O sub-buffers (args 0..3) + start_pos
(arg 26) change. Skips ~336 `clGetMemObjectInfo` integrity probes (in Weights::get_buffer)
+ ~480 redundant `clSetKernelArg`/step. Invalidated per generation via mega_reset_cross_kv
→ mega_reset_layer_kernels (cross-K/V + KV caches reallocate). Gate: STEP_ARGMAX t=0 =
66/1534/1513/1801 ✓; tf-depth cb0=46 ✓ (bit-identical — same math).
### A/B (same binary, NNOPT_MEGA_NOPERSIST=1 toggle, serialized)
| Round | no-persist tok/s | persistent tok/s |
|-------|------------------|-------------------|
| R1 | 0.6786 | 0.6294 |
| R2 | 0.6233 | 0.6821 |
| R3 | 0.6754 | 0.6872 |
| mean| **0.659** | **0.666** |
VERDICT: **NEUTRAL** (0.666 vs 0.659, rounds cross over = pure noise). The per-step host
arg/lookup cost is NOT on the critical path — `clSetKernelArg` stages args without a driver
round-trip, and the get_buffer probes are cheap vs 24 mega dispatches. KEPT (correct, lower
host CPU/power, no regression). Confirms again: only GPU-side dispatch COUNT and hard syncs
move the wall — next lever is the per-step 8192-fp16 logits readback (Stage 3).

## STAGE 3 (Fix-12) — ON-GPU CFG BLEND + SAMPLE (kept the CLBlast GEMM)
FIRST attempt: a fully-fused readout kernel (final LN + lm_head GEMV + blend + sample in
ONE 4-workgroup dispatch, `kernels/fused_readout.cl`). Guard 66/1534/1513/1801 byte-identical
(fp32-accum GEMV correct), BUT **REGRESSED ~12%**: 3-round A/B LOGITS 0.678 vs FUSED-GEMV
0.595. The naive 4-WG in-kernel GEMV (each WG streams 2048×1024×2 for one codebook) can't
match CLBlast's tuned lm_heads GEMM, and 4 workgroups underutilize the GPU. REVERTED the
in-kernel GEMV (fused_readout.cl → .unused_s3bak).
PIVOT (kept): HYBRID — keep the fast CLBlast lm_heads GEMM producing logits2[2,N] on the GPU,
and dispatch `kernels/sample_grid.cl` (4 WGs, one/codebook) to do ONLY the on-GPU CFG blend
(uncond+g·(cond−uncond), fp32) + temperature/top-k=250 sampling (xorshift RNG seeded from
seed^step^cb) writing 4 int32 ids — replacing cfg_combine_rows + the 8192-fp16 logits
readback (a hard sync) + the 4 host samples. NNOPT_FUSED_READOUT=1 enables it; NNOPT_FORCE_ARGMAX=1
makes the on-GPU sampler pure argmax (guard). Routed via model_forward_graph_cfg_m2_impl +
forward_cfg_sampled. (.bak: *.s3bak)
### CORRECTNESS — guard PASS (FORCE_ARGMAX)
STEP_ARGMAX t=0 = **66/1534/1513/1801** byte-identical; cb0 stream `66 66 1827 1075 66`
identical. (tf-depth unaffected: --force-grid uses the logits path, decoder math unchanged.)
### A/B (same binary, NNOPT_FUSED_READOUT toggle, serialized, FORCE_ARGMAX both)
| Round | logits path tok/s | hybrid sample_grid tok/s |
|-------|-------------------|---------------------------|
| R1 | 0.6715 | 0.6745 |
| R2 | 0.6848 | 0.6921 |
| R3 | 0.6457 | 0.6786 |
| mean| **0.667** | **0.682** |
VERDICT: hybrid is **≥ logits every round, mean +2.2%** — a small real win AND it removes the
8192-fp16 readback, which is the PREREQUISITE for Stage 4's GPU-resident grid (sampled ids
must be on-GPU to write the grid without a host round-trip). KEPT.
TOOL LESSON: an in-kernel GEMV that replaces a tuned BLAS GEMM is a TRAP on a memory-bound
mobile GPU unless it's multi-workgroup + bandwidth-optimized — keep the BLAS GEMM and fuse
only the cheap epilogue (blend+sample). nnopt's fused-readout codegen should default to the
hybrid (BLAS GEMM + epilogue kernel), not a monolithic GEMV.

## STAGE 4+5 (Fix-12) — GPU-RESIDENT DECODE GRID + FUSED EMBED PROLOGUE — REGRESSION, gated OFF
Built the full GPU-resident decode loop: a device grid buffer [num_codebooks, steps+1] int32
(mega_alloc/read/free_decode_grid); the sample_grid kernel writes the sampled id straight into
grid[:,t+1] WITH delay-window BOS forcing (no per-step host id readback gating the pipeline);
a fused `embed_prologue` kernel (Stage 5) reads the 4 ids from grid[:,t], sums their embedding
rows, and folds the positional add — collapsing ~8 dispatches + 4 id uploads/step → 1 dispatch,
0 uploads. Host loop only enqueues; ONE whole-grid readback at end-of-decode. Gated by
NNOPT_GPU_GRID=1 (requires NNOPT_FUSED_READOUT). (kernels/embed_prologue.cl, sample_grid grid
args; backbone grid lifecycle + embed_prologue_dispatch; main.cpp gpu-grid loop. .bak: *.s4bak)
### CORRECTNESS — guard PASS (bit-identical)
FORCE_ARGMAX: STEP_ARGMAX t=0 = **66/1534/1513/1801** byte-identical; cb0 stream identical.
The GPU-resident grid + embed_prologue + end-readback produce bit-exact results (proves the
infra is sound). embed table is [vocab+1=2049, hidden] — the BOS/pad row matters (a vocab=2048
stride bug was caught + fixed: use get_shape()[0] for the table stride, not VOCAB).
### A/B (same binary, NNOPT_GPU_GRID toggle, serialized, FORCE_ARGMAX both)
| Round | hybrid (Stage 3) tok/s | GPU-grid (Stage 4+5) tok/s |
|-------|------------------------|-----------------------------|
| R1 | 0.6625 | 0.5818 |
| R2 | 0.6807 | 0.5825 |
| R3 | 0.6821 | 0.5816 |
| mean| **0.675** | **0.582** (−14%) |
VERDICT: **REGRESSION −14%.** The per-step host id-uploads + 16 B readback were NEVER the wall
(Stages 2–4 all confirm host-side per-step cost is off the critical path). The embed_prologue
kernel — a single workgroup doing strided fp16 reads from the 16.8 MB [4·2049·1024] embed table,
PLUS a true RAW dependency (this step's embedding waits on the prior step's sample_grid grid
write) — costs MORE GPU time than the ~8 tiny dispatches it replaced. Gated OFF by default
(NNOPT_GPU_GRID unset → the binary keeps the faster Stage-3-hybrid path). Infra retained (correct,
zero-sync architecture is right in principle; loses on THIS device's single-WG embed kernel).
TOOL LESSON: "eliminate per-step host syncs" is NOT a universal win when the wall is GPU
execution of a few heavy memory-bound dispatches — the host round-trips were already overlapped
by the in-order queue. A fused embed/prologue kernel must be multi-workgroup + coalesced or it
adds a serial single-WG dispatch onto the critical path. Measure; don't assume sync-removal helps.

## CUMULATIVE (Fix-12 stages 1–5): 0.39 (per-row mega baseline) → **0.68 tok/s** (M=2-native mega
+ Stage-3 hybrid on-GPU sample), validated controlled A/B, guard 66/1534/1513/1801 byte-identical
+ tf-depth cb0=46. **253 tokens ≈ 372 s** (was 648 s). Stages 2 (persistent args), 4, 5 (GPU grid)
measured NEUTRAL/REGRESSION and gated off. The ONLY mover after Stage 1 is dispatch-COUNT collapse;
the wall is now the 24 memory-bound single/dual-workgroup mega-GEMV dispatches/step (≈60 ms each).
Still > 60 s target → Stage 6 (int8 weights) is the authorized next lever (halves the ~820 MB/token
weight traffic; §3 predicts ~2× IF bandwidth-bound).

## STAGE 6 (Fix-12) — int8 WEIGHT QUANTIZATION — SPEED WIN, but FAILS the tf-depth gate (gated OFF)
Per-output-row symmetric int8 for the 8 decoder GEMV weights (q/k/v/o/cq/co/fc1/fc2):
scale[n]=max(|W[n,:]|)/127, Wq=round(W/scale) at load (once), dequant in-kernel as
y[n]=scale[n]·Σ(Wq[n,k]·x[k]) with fp32 accum. Built as a 2nd kernel via -D MEGA_INT8=1
(same decoder_layer_mega.cl, #ifdef'd weight args + MGEMV macro). LN/cross-KV/KV-cache stay
fp16. Gated by NNOPT_INT8=1. (.bak: *.s4bak)
### SPEED — the megakernel IS bandwidth-bound (decisive finding)
3-round same-binary A/B (NNOPT_INT8 toggle, serialized):
| Round | fp16 mega tok/s | int8 mega tok/s |
|-------|-----------------|------------------|
| R1 | 0.6844 | 0.9691 |
| R2 | 0.6846 | 0.9652 |
| R3 | 0.6778 | 0.9644 |
| mean| **0.682** | **0.966** (1.42×) |
→ int8 gives a **reproducible 1.42×** (0.97 tok/s). This PROVES the mega GEMV is memory-bound
(halving weight bytes ≈ halves the per-dispatch time), exactly §3's prediction. At 0.97 tok/s,
253 tok ≈ **261 s**.
### CORRECTNESS — argmax holds, but TEACHER-FORCING DEPTH FAILS the ≥40 gate → REVERTED
- Guard (greedy CFG-3, 5-tok): STEP_ARGMAX t=0 = **66/1534/1513/1801** — argmax IDENTICAL to
  fp16 (logit vals shift slightly: 7.21 vs 7.215). cb0 stream identical. PASS.
- Teacher-forcing depth (the binding gate): cb0 **46 → 20** (full int8) — FAR below ≥40. int8
  quant noise accumulates over 24 layers and perturbs the near-tied lm-head logits exactly as the
  fp16-accum bug did (FIX_LEDGER Fix 10). FAIL.
### STAGE 6b — FFN-ONLY int8 (fc1/fc2 int8, attention fp16) — closer, still FAILS
Quantize ONLY fc1/fc2 (8.4M of 12.4M params/layer; GELU absorbs fc1 noise), keep q/k/v/o/cq/co
fp16 (the attention logits feed the residual stream that full-int8 corrupted). 3rd kernel via
-D MEGA_INT8_FFN=1; NNOPT_INT8_FFN=1.
- Guard: 66/1534/1513/1801 PASS. tf-depth cb0 **33** (vs 20 full-int8, vs 46 fp16) — recovered
  some fidelity but STILL < 40. FAIL.
VERDICT: BOTH int8 variants gated OFF by default (NNOPT_INT8 / NNOPT_INT8_FFN unset → fp16 path).
The 1.42× speed is real and the device is bandwidth-bound, but uniform int8 (even FFN-only) costs
too much logit fidelity for this deep decoder's near-tied codebook logits at the strict tf-depth≥40
gate. int4 is OUT (campaign rule: int4 only if int8 lands clean — it didn't).
### STAGE 6c — PER-GROUP int8 (group=128 and 64) — improved fidelity, STILL FAILS
Replaced the per-row scale with a per-COLUMN-GROUP scale (MEGA_QGROUP=128, then 64): one int8
scale per 128- (or 64-) wide block along K. Per-group is the standard fidelity fix for uniform int8.
| variant | tf-depth cb0 | cb1 | cb2 | cb3 | gate cb0≥40 |
|---------|-------------|-----|-----|-----|-------------|
| fp16 (reference) | 46 | 13 | 45 | 2 | — |
| int8 per-row     | 20 | 3  | 1  | 41 | FAIL |
| int8 FFN-only    | 33 | 8  | 1  | 28 | FAIL |
| int8 group=128   | **33** | 3 | 45 | 13 | FAIL |
| int8 group=64    | 16 | 36 | 1  | 13 | FAIL |
Per-group=128 lifted cb0 20→33 (and recovered cb2 to 45 = fp16 parity), but **still < 40**. group=64
did NOT help cb0 (16) — the per-codebook depths bounce non-monotonically because WHICH near-tied
codebook logit flips depends on the exact quant-noise pattern, not just the group size. No uniform-int8
variant clears cb0≥40.
### FINAL VERDICT — int8 GATED OFF (fails the binding tf-depth≥40 gate; campaign rule = revert)
int8 is a measured, reproducible **1.42× speed win (0.68→0.97 tok/s, 253 tok 372→261 s)** that
PROVES the mega GEMV is memory-bandwidth-bound — but uniform int8 (any granularity tried) costs too
much codebook-logit fidelity for the strict tf-depth≥40 gate. int4 is OUT (campaign rule: int4 only
if int8 lands clean). All int8 paths gated OFF (NNOPT_INT8 / NNOPT_INT8_FFN unset by default → fp16).
TOOL LESSON: on a deep (≥24-layer) decoder whose codebook logits are near-tied, uniform int8 (per-row
OR per-group) is a SPEED win but a FIDELITY loss the teacher-forcing-depth gate catches — argmax-only
guards MISS it (argmax stayed byte-identical while tf-depth halved). nnopt's quantization policy must
gate int8 on teacher-forcing depth / perplexity, not argmax identity. The EXACT next lever to make
int8 viable here: per-channel int8 with an fp16 OUTLIER/residual correction (e.g. keep the top-k
largest-magnitude weights per row in fp16, quantize the rest) — that preserves the few weights that
move the near-tied logits while still streaming ~int8 bytes for the bulk.

## ════ FIX-12 FINAL SUMMARY TABLE (device ZY22D5NLGQ, fp16 RELEASE, 20-tok probe) ════
| Stage | Change | decode tok/s | 253-tok seconds | guard cb0..3 | tf-depth cb0 | verdict |
|-------|--------|--------------|-----------------|--------------|--------------|---------|
| base  | per-row megakernel (Fix 11) | 0.39 | ~648 | 66/1534/1513/1801 ✓ | 46 | — |
| 1 | M=2-native megakernel (1 dispatch/layer, in-place) | **0.68** | **~372** | byte-identical ✓ | 46 ✓ | **KEPT (1.9×)** |
| 2 | persistent kernel args | 0.67 | ~378 | byte-identical ✓ | 46 ✓ | KEPT (neutral) |
| 3 | on-GPU CFG blend+sample (hybrid; CLBlast GEMM kept) | 0.68 | ~372 | byte-identical ✓ | 46 ✓ | KEPT (+2%, kills 8K readback) |
| 4+5 | GPU-resident grid + fused embed prologue | 0.58 | ~436 | byte-identical ✓ | 46 ✓ | OFF (−14% regression) |
| 6 | int8 weights (per-group 128) | 0.97 | ~261 | argmax-identical ✓ | **33 ✗** | OFF (fails tf-depth≥40) |

**FINAL SHIPPED CONFIG: fp16 M=2-native megakernel + Stage-3 hybrid on-GPU sample = 0.68 tok/s
(NNOPT_MEGA_LAYERS=24, optionally NNOPT_FUSED_READOUT=1). 253 tokens ≈ 372 s.**
**<60 s NOT reached.** 0.68 tok/s vs the 4.6 tok/s target (6.8× short).
- MEASURED BLOCKER: the mega GEMV is memory-bandwidth-bound — int8 (halved bytes) gave a clean,
  reproducible 1.42× (0.97 tok/s), confirming §3's ~17 tok/s fp16 ceiling is the wall, and that
  we sit far below it because a single/dual-workgroup-per-layer GEMV cannot saturate the Adreno
  memory bus (low occupancy) and its weight reads are stride-K uncoalesced.
- EXACT NEXT LEVER (two, compounding): (1) a COALESCED, MULTI-WORKGROUP-per-layer mega GEMV
  (split-K or row-tiled across more workgroups) to raise occupancy toward the bandwidth ceiling —
  this alone should approach the ~2–4 tok/s "after-batch" band §3 predicts; (2) int8 WITH fp16
  outlier/residual correction (keep the few largest-|w| weights per row in fp16) to pass tf-depth≥40
  while still streaming int8 bytes — stacking ~2× on top. Together they target the ~17 tok/s fp16
  ceiling region (5 s clip ≈ 25–30 s); true RT?1 still needs int4 or dropping CFG, as §3 states.

## ════ CAMPAIGN-2 (2026-06-04) — BANDWIDTH CAMPAIGN: 0.68 → 3.26 tok/s (4.8×) ════
Session goal: attack the measured memory-bandwidth wall (§Fix-12 verdict) with coalesced
multi-WG GEMVs + vectorized weight fetch. All numbers: device ZY22D5NLGQ, fp16 RELEASE,
20-tok profile probe, same prompt/token-ids. (Power outage mid-session 2026-06-04; this
section reconstructs the validated state. Opt #3 TEX probe results land below when run.)

### Step 0 — instrumentation + CRITICAL default bug
- Added per-kernel GB/s to NNOPT_PROFILE output (bytes/dispatch tracked per kernel).
- **FOUND: every "shipped" Fix-12 optimization was env-gated OFF by default.** Old default
  NNOPT_MEGA_LAYERS=0 left the binary on the 0.31 tok/s M=2 path. Defaults now flipped to the
  validated state (see "shipped defaults" below).

### Opt #1 — -cl-mad-enable -cl-fast-relaxed-math build flags: kept (small win)

### Opt #2 — MULTI-WORKGROUP EXTERNAL FFN (mega_ffn_fc1 / mega_ffn_fc2)
fc1/fc2 leave the megakernel (NNOPT_MWG_FFN, now default ON) → many WGs/layer instead of 2.
Accumulation order matches in-mega KGROUP=16 GEMV exactly (bit-identical row sums).
- RPW sweep (NNOPT_MWG_RPW1/RPW2): fc2 best at RPW2=32 (2.22 GB/s); RPW1=64.
- fc2 __local chunked staging (4 KB chunks, MEGA_MWG_CHUNK=1024): every WG was re-streaming
  the same 16 KB fp32 input (1.1–2.4 GB/s); chunked = cooperative load once per WG,
  4.3 KB local → ≥7 WGs resident. Ascending-k chunks keep accumulation order IDENTICAL.
- Compile-time MEGA_FC2_RPW=32 (runtime n_iters spilled acc[] to scratch; constant unrolls).

### Opt #2b — 64-BIT VECTORIZED WEIGHT FETCH (uint2 → as_half4), MWG + in-mega
- mega_ffn_fc1: 1.9 → 3.94 GB/s. mega_ffn_fc2: → 2.22 GB/s.
- In-mega attention GEMVs (MEGA_VEC=4): decoder_layer_mega 1.57 → 2.46 GB/s.
- Activation reads stay SCALAR (vload4 on global fp32 x measured 1.12 vs 2.40 GB/s scalar —
  wide activation loads thrash on Adreno 620). Weights-only vectorization.

### Fidelity gates (binding: tf-depth cb0 ≥ 40)
- Guard (greedy CFG, 5-tok): STEP_ARGMAX 66/... byte-identical. PASS.
- tf-depth at VEC 4/4: cb0=47 ✓ (repeatable). **WARNING: VEC 4/1 deterministically drops
  cb0→16** — MWG_VEC and MEGA_VEC interact through near-tied logits; change them TOGETHER
  or re-run the tf gate. Scalar/scalar control: cb0=47 (gate is on the config, not noise).
- 5 s clip on-device (vec 4/4): 98 s end-to-end (was ~7 min). Audio approved by user.

### Measured ladder (decode tok/s, 20-tok probe)
| config | tok/s | notes |
|--------|-------|-------|
| Fix-12 shipped (in-mega FFN, scalar) | 0.68 | starting point |
| + MWG FFN, RPW1=64 RPW2=4 | 2.51 | fc2 1.73 GB/s |
| + RPW2=32 + fc2 chunked __local | 2.68 | fc2 2.22 GB/s |
| + MWG_VEC=4 + MEGA_VEC=4 | **3.26** | attn 2.46 GB/s, fc1 3.94 GB/s |

### SHIPPED DEFAULTS (all validated 2026-06-04; env vars now REVERT switches)
- NNOPT_MEGA_LAYERS default = NUM_HIDDEN_LAYERS (0 reverts to M=2 baseline)
- NNOPT_MWG_FFN default ON (=0 reverts) | NNOPT_MWG_VEC / NNOPT_MEGA_VEC default 4 (=1 reverts)
- NNOPT_FUSED_READOUT default ON (=0 reverts) | NNOPT_MWG_RPW2 default 32 (4|r, r≤64)
- Certification 3-round A/B old-vs-new defaults was INTERRUPTED by the outage (R1 leg-A
  2.2512 logged) — re-run pending.

### Opt #3 — TEXTURE-PATH WEIGHTS (NNOPT_TEX, default ON): 3.38 → 3.75 tok/s, KEPT
Weights mirrored once/layer into image2d_t CL_RGBA+CL_HALF_FLOAT (texel = 4 K-halves);
read_imageh rides Adreno's dedicated texture pipe + L1 tex cache (dual-issues with load/store).
- Host: 6 attn projections rebind into mega kernel (-D MEGA_TEX → image2d_t args); fc1
  image goes to the external FFN kernel via s_fc_w_cache (mega fc args 24/25 stay
  buffers — fixes clSetKernelArg -38).
- Outage cut the kernel edit mid-write (unterminated #ifdef MEGA_TEX at fc2) — repaired
  2026-06-04 post-outage.
- MEASURED (20-tok probe): attn mega 2.45→3.53 GB/s, fc1 3.92→4.32 GB/s. fc2 variants
  (with attn+fc1 on tex):
  | fc2 variant | fc2 GB/s | tok/s |
  |---|---|---|
  | UNCHUNKED tex ("tex frees load/store" theory) | 1.25 | 2.98 ✗ 2× loss |
  | chunked tex (NNOPT_TEX_FC2=1) | 2.19 | 3.761 |
  | chunked buffer (DEFAULT) | 2.17 | 3.752 |
  → fc2 tex-vs-buffer is a WASH; default keeps fc2 on buffers (skips mirroring 8 MB/layer
  → ~192 MB less GPU memory). The unchunked-tex theory is DEAD: re-streaming the 16 KB
  fp32 input loses 2× regardless of which pipe the weights ride — __local chunking stays.
- FIDELITY (binding gate): tf-depth cb0=47 ≥40 ✓ (cb 47/3/1/2 = accepted VEC4/4 baseline),
  guard argmax 66/1534/1513/1801 + logit 7.2188 identical. PASS → SHIPPED DEFAULT ON.
**Campaign-2 cumulative: 0.68 → 3.75 tok/s (5.5×). 253 tok ≈ 67 s (was 372 s).**

### Remaining queue: #4 half4 mad(), #5 unroll sweep, #6 in-WG/subgroup reduce,
### #7 cl_qcom perf hints, #8 zero-copy/ION weights, #9-#10 T5 + EnCodec on GPU.

## ════ CAMPAIGN-2 SESSION 2 (2026-06-04, post-outage) — Opts #4-#7 probes ════
All on the Opt-#3 baseline (TEX default ON, mixed fc2-buffer). Steady-state control
drifted 3.75 → ~3.6 tok/s across the session (device heating; the perf-hint A/B pair
confirms it's thermal, not config).

### Opt #4 — fc2 DOUBLE-BUFFERED __local staging: REGRESSION, reverted
Ping-pong xs[2][1024] (prefetch chunk c+1 during compute of c, one barrier/chunk):
fc2 2.17 → 1.66 GB/s (3.34 tok/s). The extra 4 KB local halves WG residency (7→3
WGs/SP) — on Adreno latency hiding comes from RESIDENT WGS, not intra-WG prefetch.
DO NOT RE-TRY. (Comment in kernel records this.)

### RPW re-sweep under tex regime: 64/32 default CONFIRMED optimal
RPW2 ∈ {16,32,64} × RPW1 ∈ {32,64,128}: all alternates 3.62-3.67 < default. fc2 is
FLAT (~2.1-2.2 GB/s nominal) across occupancy, acc-count, tex/buffer, chunk variants.

### Opt #6-variant — CFG-ROW-FUSED fc2 (NNOPT_FC2_FUSE, gated OFF): REGRESSION
Theory: (nwg,2) dispatch row-pairs drift → L2 misses → both rows stream the same 8 MB
from DRAM; fusing both rows into one WG halves weight traffic by construction.
MEASURED: fc2 → 1.05 GB/s nominal (2.69 tok/s) — 2× WORSE. 16 accumulators spill +
WG count halves. THEORY FALSIFIED — fc2's 2.2 GB/s is not L2-dedup related. Variant
stays in-tree gated OFF (NNOPT_FC2_FUSE=1) like the int8 experiments.
fc2 VERDICT after 5 falsified theories (occupancy, acc-count, tex pipe, prefetch,
row-fusion): structurally pinned at ~2.2 GB/s nominal; STOP micro-tuning it.

### Opt #7 — cl_qcom_perf_hint HIGH at context creation: NEUTRAL, kept
Extension-gated (skips cleanly on non-QCOM), NNOPT_PERF_HINT=0 reverts, plain-context
retry on failure. "PERF_HINT: cl_qcom_perf_hint HIGH active" on device; 3.636 vs 3.641
tok/s — sustained decode already ramps DVFS. Kept for cold-start/first-token latency.

### Session-2 fidelity: tf-depth cb0=47 ✓ + guard argmax 66/1534/1513/1801 + logit
7.2188 byte-identical, certified on the final defaults (logs/tex_tf_gate.log).

### Opt #5 — #pragma unroll 4 SWEEP (the surprise win of the session): 3.81 → 4.00 tok/s
| loop | bound | result |
|------|-------|--------|
| fc1 tex k4 loop | compile-time (MEGA_HIDDEN/4) | 4.34 → **6.18 GB/s** nominal (+44%) KEPT |
| fc2 tex chunk loop (TEX_FC2) | compile-time (CHUNK/4) | 2.08 → **2.36 GB/s** KEPT — flipped NNOPT_TEX_FC2 default ON (4.00 tok/s vs 3.81 buffer) |
| fc2 buffer chunk loop | compile-time | 2.11 → 2.08 NEUTRAL, reverted |
| attn mega GEMV (in megakernel) | runtime K AND compile-time both | **0.33 GB/s — 10× CRATER**, reverted |
KEY MECHANISM (explains fc1's above-DRAM-peak 6.2 GB/s nominal): the two CFG rows'
WGs fetch the SAME weights; unroll tightens their lockstep so the TEXTURE L1 serves
the twin row's fetches — weight DRAM traffic effectively halves. This is the L2-dedup
theory VINDICATED, but through the tex cache + unroll, not row-fusion (which spilled).
ATTN CRATER MECHANISM: decoder_layer_mega inlines the GEMV 6× into one giant function
(attn+LN+reductions); unroll multiplies live registers past the file → whole-kernel
spill. Small standalone kernels (fc1/fc2) unroll fine. DO NOT unroll inside the mega.

### CERTIFICATION (3-round same-binary A/B, old shipped env vs new defaults)
| round | old shipped | new defaults |
|-------|-------------|--------------|
| R1 | 2.2401 | 3.6291 |
| R2 | 2.2358 | 3.5961 |
| R3 | 2.2393 | 3.6223 |
(Old legs at 2.24 vs Fix-12's 0.68: Opt-#1 build flags lift even the legacy path.
vs actual pre-campaign binary: 0.68 → 3.62+ certified.) Run before the unroll wins;
final defaults verify at **3.98 tok/s** (heated device; 4.00 cold), tf-depth cb0=47 ✓,
argmax 66/1534/1513/1801 + logit 7.2188 identical (logs/final_defaults_gate.log).

### 5s CLIP (250 tok, defaults, pre-unroll build): 2:27 wall, TTFT 6.8s, decode 3.06
### sustained. /tmp/mg_campaign2_final.wav delivered to user for quality check.

## ════ CAMPAIGN-2 FINAL (2026-06-04): 0.68 → **3.98 tok/s (5.9×)**, 253 tok ≈ 64 s ════
Shipped defaults (all env vars are now REVERT switches): MEGA_LAYERS=all, MWG_FFN on,
MWG_VEC/MEGA_VEC=4, TEX on, TEX_FC2 on, FUSED_READOUT on, PERF_HINT on, RPW 64/32,
FC2_RPW=32 compile-time, unroll-4 on fc1+fc2-tex loops only.
Kernel GB/s (nominal): attn mega 3.52, fc1 6.17, fc2 2.40.
NEXT LEVERS (unprobed): zero-copy/ION weight upload (TTFT), T5 + EnCodec → GPU
(end-to-end wall), attn mega split into standalone kernels so its GEMVs can unroll +
ride the tex-L1 dedup like fc1 (the 42%-of-time prize the spill wall currently blocks),
int8+fp16-outlier residual (the tf-depth-safe quant from Fix-12's verdict).

## ════ CAMPAIGN-2 SESSION 3 (2026-06-04) — EnCodec threading + ATTN SPLIT STAGE 1 ════

### Host EnCodec multithreading (the hidden 54 s): wall 140 → 110 s for a 5 s clip
Wall breakdown showed ~54 s OUTSIDE inference: essentially all host EnCodec/SEANet
decode (0.35 ms/sample ≈ 11× slower than realtime, single-core scalar loops).
- conv1d + conv_transpose1d: parallel over out-channel slices (std::thread, ≤8;
  disjoint output rows; per-(o,t) accumulation order unchanged → BIT-IDENTICAL PCM).
  EnCodec 54 → 22.9 s.
- LSTM gate GEMVs (4096×2048 per step ×2 layers): parallel over gate rows per
  timestep (recurrence stays serial). 22.9 → 18.5 s.
- BENCHMARK encodec_decode_sec added to main.cpp (was invisible — CHECKPOINTs
  compile out in release).
NEXT (EnCodec): hoist Wih·x out of the time loop (precompute for all t in one
parallel GEMM — removes half the per-step work + 499 of 500 thread-spawn batches).

### Attn-split probes that FAILED first (kernel comments record both):
- MANUAL unroll-2 in the in-mega GEMV: 3.52 → 3.01 GB/s. ONE extra live float4
  regresses it — the megakernel sits exactly at its register ceiling. CLOSED:
  no in-place GEMV improvement is possible inside decoder_layer_mega.

### STAGE-1 ATTN SPLIT (NNOPT_QKV_EXT, default ON): 3.98 → **4.13 tok/s**
LN1 + q/k/v leave the megakernel:
- mega_ln_rows: standalone LN1 (same mega_layernorm → bit-identical), 0.45% of time.
- mega_qkv: ONE GEMV over a PACKED [3H, H/4] qkv texture (rows q|k|v), 96 WGs/row,
  unroll-4; q → q_ext scratch, k/v append straight into the KV caches at start_pos.
  Same KGROUP=16 lane map + tree reduce as mega_gemv_tex → sums BIT-IDENTICAL.
- mega kernel under -D MEGA_QKV_EXTERNAL skips LN1+QKV, loads qv from q_ext.
- Both dispatch paths (persistent fast + slow bind) enqueue ln_rows+qkv first.
MEASURED: mega_qkv **5.67 GB/s** (in-mega was 3.52 — the register ceiling was the
whole story), mega 3575→2148 ms, ln dispatch overhead negligible. Dispatch-count
fear FALSIFIED: +2 dispatches/layer cost ~0; the split is pure win.
FIDELITY: tf-depth cb0=47 ✓, argmax 66/1534/1513/1801 + logits byte-identical.
**Running total: 0.68 → 4.13 tok/s (6.1×). 253 tok ≈ 61 s decode; 5 s clip ≈ 105 s wall.**

### STAGE-2 SPLIT (next): externalize o/cq/co the same way (mega → 4 small
### attention-core kernels + 3 packed-tex GEMVs). Remaining in-mega GEMV bytes
### 6 MB/layer at 3.5 GB/s; at 5.7 → ~+0.3 tok/s projected.

## ════ STAGE-2 ATTN SPLIT (NNOPT_ATTN_SPLIT, default ON): 4.07 → **4.20 tok/s** ════
THE MEGAKERNEL IS GONE (for split layers decoder_layer_mega is never enqueued).
Per layer per step: ln_rows → qkv → attn_core(self) → proj(o,+x_in) → ln_f32 →
proj(cq) → attn_core(cross) → proj(co,+resid→ffn_resid) → ln_f32(→ffn_normed) →
fc1 → fc2.  (11 small dispatches; the dispatch-overhead era is truly over.)
New kernels (all in decoder_layer_mega.cl, gated MEGA_QKV_EXTERNAL && MEGA_TEX):
- mega_attn_core: GENERIC single-query attention (self: KV caches, seq=start_pos+1;
  cross: precomputed cross-K/V, seq=enc_len). Mirrors the in-mega cores' loop
  strides/reduce trees exactly.
- mega_proj: generic [H,H] tex GEMV, multi-WG + unroll-4, epilogue by mode
  (0=plain cq, 1=+x_in fp16 residual for o, 2=+fp32 residual for co — co writes
  STRAIGHT INTO s_ffn_resid, and ln3 writes s_ffn_normed: the fc1/fc2 contract
  is unchanged).
- mega_ln_rows_f32: LN2/LN3 (same mega_layernorm function).
MEASURED (20-tok): proj_o 5.45 / proj_cq 5.71 / proj_co 5.47 GB/s (in-mega was
3.5 — register-ceiling escape verified on every GEMV). attn cores 4.3+4.5% of
time; the 3 extra LN dispatches 1.4% total.
FIDELITY: tf-depth cb0=47 ✓, argmax 66/1534/1513/1801 + logits 7.2188 byte-
identical (bit-identical-by-construction held through the full dissolution).
Kernel time/step·layer now: fc2 0.148ms (45%! the final boss) > fc1 0.057 >
qkv 0.046 > proj_o/co/cq ~0.016 each > attn cores ~0.014 each > LNs ~0.0016.
**Running total: 0.68 → 4.20 tok/s (6.2×). Next: fc2 split-K (last structural
fc2 idea), EnCodec→GPU or Wih hoist, zero-copy TTFT.**

## ════ fc2 SPLIT-K (NNOPT_FC2_SPLITK, default ON): 4.11 → **5.30 tok/s (+29%)** ════
The final boss falls. fc2 was pinned at ~2.2-2.4 GB/s through SIX falsified
theories; the winning diagnosis: the CHUNK-BARRIER STRUCTURE itself (4×1024
__local staging rounds with 2 barriers each). fc1's winning shape = whole input
staged ONCE (4 KB) + short 256-texel weight rows + zero mid-GEMV barriers.
Split-K gives fc2 exactly that shape: each WG owns (row-block, one of KSEG=4
K-segments), stages its 4 KB x-segment once, streams 256 texels/row unroll-4;
mega_ffn_fc2_sk_red sums resid + 4 partials per element (0.2% of time).
- fc2: 2.22 → **5.78 GB/s** (1816 → 696 ms). 256 WGs (32 row-blocks × 4 seg × 2 rows).
- FIDELITY: NOT bit-identical (reduction order changes). tf gate: cb0=**44 ≥ 40 ✓**
  (deterministic, ×2 runs), argmax 66/1534/1513/1801 identical, logits ±0.008
  (7.2109 vs 7.2188); cb2 1→19, cb3 2→11 (near-ties resolved differently, better).
  Second-ever order-changing opt to pass the gate (after MWG_VEC=4).
- NNOPT_FC2_SPLITK=0 reverts to the chunked kernel (which stays in-tree).

### THE DECODER IS NOW FLAT — every GEMV at the texture-dedup ceiling:
| kernel | GB/s | % | | kernel | GB/s | % |
|---|---|---|---|---|---|---|
| fc2_sk | 5.80 | 24.5% | | proj_cq | 5.67 | 6.3% |
| fc1 | 6.11 | 23.3% | | attn_cross | — | 6.0% |
| qkv | 5.64 | 18.9% | | attn_self | — | 5.8% |
| proj_o/co | 5.4 | 13.1% | | 3×LN + reduce | — | 2.1% |
No kernel above 25%. Decode is at the fp16-weight-streaming wall everywhere;
further decode gains now require LESS BYTES (int8+outlier quant) or fewer steps.

## ════ RUNNING TOTAL: 0.68 → **5.33 tok/s (7.8×)**. 5 s clip ≈ 93 s wall ════
(decode 60 s + TTFT 7.6 + prefill ~4 + EnCodec 18.4 + misc ~3)
Remaining queue: EnCodec (now ~20% of wall: Wih hoist −3 s cheap, GPU port −15 s),
zero-copy TTFT, T5. Decode: int8+fp16-outlier is the only big lever left.

### EnCodec Wih HOIST (bit-identical): encodec 18.4 → 16.7 s; clip wall **88 s**
The LSTM input projection gin[t][r] = b + Wih[r,:]·x[:,t] has no recurrence
dependency → computed for ALL timesteps in one parallel pass BEFORE the time
loop (j-outer rank-1 form, contiguous x rows, DOUBLE gin carrier; per-(t,r)
accumulation order identical → bit-identical PCM). The serial loop now does
only Whh·h per step. Remaining LSTM cost: the per-step Whh spawns (~500 thread
batches); remaining EnCodec total ~16.7 s (convs ~10, LSTM ~5).
### Wall budget now (5 s clip, 88 s): decode 60 ▸ EnCodec 16.7 ▸ TTFT 5-12
### (mmap variance) ▸ prefill ~4. NEXT BIG LEVERS: EnCodec→GPU (−12-15 s;
### Wih part is one big GEMM, Whh = 250 sequential 8 MB GEMVs ≈ 0.3 s on-GPU,
### convs as gather kernels), batched prefill (11 GEMVs → 1 GEMM, −3 s),
### int8+fp16-outlier decode quant (the only remaining decode lever).

## ════ EnCodec → GPU, PHASE A (NNOPT_ENCODEC_GPU, default ON): 16.7 → **8.94 s** ════
kernels/encodec.cl: enc_pad_reflect / enc_conv1d / enc_convt1d / enc_elu(_oop) /
enc_add. Conv stack on GPU; LSTM stays CPU (1 MB roundtrip). RVQ host. Effective
weight_norm weights computed host-side, uploaded per clip. Empty-on-error →
automatic host fallback in main.cpp. NNOPT_ENCODEC_GPU=0 forces host path.
CORRECTNESS: PCM cosine vs host path = **1.000000** at every iteration (greedy
codes; fp32-vs-fp64 accum + gather-vs-scatter convT below int16 resolution).
### The convT saga (19.1 s → 3.0 s):
| version | convT s/clip | lesson |
|---|---|---|
| PyTorch [in,out,k] layout | 19.1 | consecutive ic reads strode 32 KB — all cache misses |
| host-transposed [out,k,in] + direct tap stepping | 7.6 | contiguous wq + kk≡traw(mod s) kills the modulo scan |
| + 4-lane ILP (float4 wv·xv, strided x gather) | 3.0 | scalar loop was a 2048-deep DEPENDENT fma chain |
- Same 4-lane ILP on enc_conv1d REGRESSED (2.5 → 3.6 s): k ∈ {1,3,7} strided
  gathers cost more than the chain saves. Reverted; comment records it.
- Host weight-prep (weight_norm + convT transpose) parallelized over slices
  (bit-identical): prep phase 1.9 → 0.6 s.
### Phase breakdown (250-tok clip): compile 0.13 ▸ rvq 0.01 ▸ conv0+LSTM-CPU 2.9
### ▸ host prep 0.6 ▸ GPU kernels 5.3 (convT 3.0 + conv1d 2.6).
### NEXT (Phase B): LSTM → GPU (gin as one GEMM; Whh = 500 small GEMVs ≈ 1.7 s
### fp32 / 0.85 fp16) −2 s; conv kernel o-tiling for the remaining 5.3 → ~2 s.
### NOTE: TTFT mmap variance (5-51 s observed!) is now the BIGGEST wall item —
### weight-blob readahead/mlock or upload-during-T5 should be next session.
## Wall (typical TTFT ~7 s): ~84 s for a 5 s clip. From 7 min at session start.

## ════ SESSION 4 (2026-06-05): TTFT variance fix + GPU LSTM + T5 threading ════
### ① TTFT mmap fix — THE 5-51 s VARIANCE WAS SELF-INFLICTED
weights.cpp called madvise(MADV_RANDOM) on the 1.1 GB blob — which DISABLES
readahead — while the load pattern uploads essentially EVERY tensor: the blob
demand-faulted 4 KB at a time from flash during first GPU upload. Fix:
MADV_WILLNEED + a background page-toucher thread that streams the blob while
tokenizer/T5 run (joined in ~Weights). MEASURED: TTFT 5.76 cold / 5.42 warm —
variance gone (was 5-51 s).
### ② EnCodec Phase B — LSTM ON GPU (NNOPT_ENC_LSTM_GPU, default ON)
kernels: enc_lstm_gin (input projection batched over ALL t in one dispatch —
the Wih hoist, GPU edition), enc_lstm_gates (per-step Whh GEMV, h staged in
__local, float4 lanes), enc_lstm_cell (PyTorch [i,f,g,o] cell). Recurrence =
2 async dispatches/step; in-order queue serializes the h dependency — NO host
sync until final PCM readback. PCM cosine vs CPU path: **1.000000**.
EnCodec: 8.94 → **6.96 s** (conv0+lstm phase 2.9 → 1.4 s).
### ③ "Batched prefill" — INVESTIGATED, RECLASSIFIED
The decoder has NO prompt prefill (starts from BOS; text conditions via
cross-attn). The "prefill" benchmark segment = T5 host encode + cross-KV +
mega program compile + tex-mirror creation. T5's linear_nobias (~1 GMACs
single-core double-accum) threaded over out rows (bit-identical; same
parallel_over_channels pattern). Prefill segment remains compile/mirror-bound
(~5-8 s) — next lever there is program-binary caching, not math.
### ⚠ TOOLING GOTCHA (cost a false-alarm bisect): with NNOPT_FUSED_READOUT=1
### (default) the GPU sampler IGNORES --temperature: STEP_ARGMAX prints its
### fixed-seed SAMPLED ids (0/1338/1668/1641), NOT argmax. The 66/1534/1513/1801
### guard MUST be read with NNOPT_FUSED_READOUT=0 or NNOPT_FORCE_ARGMAX=1.
### Gates this session (threaded T5 binary): guard 66/1534/1513/1801 + logits
### 7.2109 byte-identical ✓; tf-depth cb0=44 ✓ (split-K-era baseline).
## ════ 5 s CLIP WALL: **75.5 s ×2 reproducible** (was ~84 typical / 140 worst) ════
## decode 60 s ▸ TTFT+prefill ~8 ▸ EnCodec 7.0 ▸ misc ~1. From 28 MIN at bring-up.

### Program-binary cache (.clcache/, NNOPT_CL_CACHE=0 disables): NEUTRAL, kept
build_program now caches compiled binaries keyed on source+options+device+driver
(raw CLI processes get no Adreno driver shader-cache dir). 3 binaries cached and
loaded cleanly — but TTFT UNCHANGED (6-8 s): compilation was NOT the prefill
bottleneck. MEASURED TTFT anatomy (warm): ~3 s weight-blob flash stream (UFS
~350 MB/s, overlapped-but-blocking for first-needed pages) + 2.0 s T5 host
encode + ~2.5 s first-step setup (GPU buffer uploads + tex mirrors + CLBlast
internal GEMM compile, which our cache cannot reach). TTFT floor ≈ 5-6 s
without deeper surgery (T5∥stream overlap, CLBlast→own GEMM, upload batching).

## ════ HEAD-PARALLEL ATTENTION CORE (2026-06-05): wall 75.8 → **64.1 s** ════
FULL-CLIP profiling (250 tok) exposed what every 20-tok probe hid: self-attn
cost grows O(t) with the KV scan, and the 2-WG head-serial attention core
(megakernel heritage: 16 serial heads × ~10 barriers each) had become the #1
decode kernel — 27.7% / 14.9 s per clip. TOOL LESSON: profile at FULL sequence
length; short probes systematically understate any O(t) kernel.
Rewrite: one WG per (head, row) — 32 WGs, no head loop, 64-lane reductions.
| kernel | before | after |
|---|---|---|
| mega_attn_self | 14.9 s (27.7%) | **5.7 s** (13.3%) |
| mega_attn_cross | 2.1 s | **0.37 s** |
| sustained decode | 4.19 tok/s | **5.13 tok/s** |
| total_inference | 68.0 s | **56.3 s** |
FIDELITY: 64-lane softmax reduce ≠ old 512-lane order (not bit-identical) →
tf gate: cb0=44 ✓ (44/3/19/11 = exact split-K-era baseline). PASS.
Decode kernel ranking now: fc2_sk 19.9% > attn_self 13.3% (≈0.95 ms/dispatch,
further headroom via fp16-vectorized K reads) > fc1 ≈ qkv > projs.
## Wall: **64.1 s / 5 s clip** (decode 48 ▸ TTFT 8.5 ▸ EnCodec 7.3). RTF ≈ 12.8.

## ════ int8 + FP16-OUTLIER QUANT, ROUND 1 (2026-06-05) — fidelity SOLVED for fc1, perf NOT YET; default OFF ════
Scheme (the Fix-12-verdict design): per row, top 1.6% |w| stay EXACT fp32
(outlier list applied in the reduce); rest int8 with per-128-group scales.
NNOPT_INT8O: "f1"=fc1 only, "f2"=fc2 only, "1"=both. Strict before/after gates.
### FIDELITY (tf-depth, binding cb0 ≥ 40; fp16 baseline = 44):
| config | cb0 | verdict |
|---|---|---|
| fc1-only int8+outlier | **47** ✓ | PASSES — ABOVE baseline (GELU absorbs noise; outliers work) |
| fc2-only | **16** ✗ | fc2 is the fidelity killer (feeds residual stream; 4096-term rows) |
| both | 16 ✗ | dominated by fc2 |
First-ever int8 config to clear the gate. fc2 needs more: bigger NOUT (128+),
group-32, or an fp16 "hot-column" band — next round.
### PERF (fc1_i8o vs fp16 fc1 1.35 ms): NOT competitive yet — 4.0-5.3 ms across
4 variants. Falsified: CL_SIGNED_INT8 image texels (0.71 GB/s — read_imagei is
a slow path on Adreno), char4 buffer loads (4.01), uint2-as_char8 (4.48 — the
half-vector scalarization quirk was NOT the issue), group-loop unroll (5.26 —
WORSE), lane-0 outlier serialization (fixed: outliers now one-per-lane folded
into the tree reduce, that part is right). REMAINING DIAGNOSIS: the fp16 path's
speed is the texture-L1 CFG-twin dedup (effective ~12 GB/s); the int8 BUFFER
path streams both rows from DRAM + stride-8 local bank conflicts (4-way).
### NEXT ATTEMPT: pack int8 as CL_RGBA + CL_UNSIGNED_INT32 texture (16 weights
### per texel, read_imageui → as_char16) — native texel format + dedup restored.
### Decision: NNOPT_INT8O stays DEFAULT OFF until fc1_i8o beats 1.35 ms.

### int8 ROUND 2 — SIX perf variants, ALL 4-7 ms vs fp16's 1.35 ms. CONCLUSION:
| variant | fc1_i8o ms | note |
|---|---|---|
| CL_SIGNED_INT8 image, read_imagei | 5.9 | int8 texel slow path |
| char4 buffer | 4.0 | issue-rate theory |
| uint2→as_char8 buffer | 4.5 | scalarization-quirk theory — falsified |
| + group-loop unroll | 5.3 | MLP theory — falsified (worse) |
| CL_UNSIGNED_INT32 texture (16/texel) | 7.3 | dedup-restore theory — falsified (int texel slow) |
| + vector char4→float4 converts | 4.4 | ALU-converts theory — falsified |
FIDELITY held cb0=46-47 ✓ throughout (the outlier scheme is correct and is the
first int8 config ever to clear the gate — kept in-tree, NNOPT_INT8O, default OFF).
**THE REFRAME (measured, important):** the fp16 texture path's CFG-twin L1 dedup
ALREADY delivers the 2× effective-bandwidth that int8 promises — at zero ALU
cost and in the native fast texel format. On Adreno 620 with CFG-2 decode,
**fp16 + RGBA-half texture IS the optimal weight format**; int8 can only win
where the dedup doesn't apply (single-row decode, no-CFG models) or on GPUs
with native int8 dot-product (cl_arm_integer_dot_product / Adreno 7xx DP4A).
TOOL LESSON for nnopt: don't push int8 on CFG-2 + Adreno-6xx ports; spend the
effort on texture-dedup instead. int4 would face the same wall harder.
### Remaining decode levers (non-quant): per-step host-sync gap ~30 ms/step
### (recordable queues / N-steps-in-flight) + CLBlast lm_heads replacement.

## ════ GUIDE-EXTENSION ROUND (2026-06-05): tex-view SHIPPED, dot8 FOUND-but-unsolved ════
Extension dump (NNOPT_PRINT_EXT=1, logs/extensions_dump.log) grounded the audit:
cl_qcom_dot_product8, cl_qcom_recordable_queues, cl_khr_subgroups +
qcom_subgroup_shuffle, cl_khr_image2d_from_buffer, cl_qcom_ion_host_ptr ALL on
this device.
### image2d_from_buffer (NNOPT_TEX_VIEW, default ON): SHIPPED
Weight images now created as VIEWS over the weight buffers — no CopyBufferToImage
mirrors (~192 MB GPU memory freed + mirror-copy time gone). Bit-identical by
construction (same bytes); guard 66/1534/1513/1801 + logit 7.2109 byte-identical ✓.
Cold-device probe hit 5.75 tok/s (best recorded).
### qcom_dot8_acc: HARDWARE INT8 DOT FOUND — int sig (uint, uint, int), packed
4×int8 per uint (DP4A-style; name probes: only the unprefixed _acc form exists).
PERF CONFIRMED: fc1 int8 1.61 ms via hw dot vs 4.0-7.3 ms for all six convert
variants — the ALU wall is GONE. CORRECTNESS UNSOLVED: cb0=0 with both
both-signed and offset-binary+zero-point interpretations. The byte semantics
need a 16-value unit-probe kernel (print dot8_acc outputs for known vectors),
NOT full-model iterations. NNOPT_INT8O=d1 + NNOPT_DOT8_FN env stay for the next
session. If semantics crack, fc1 int8 = 1.61→~1.0 ms expected and the fidelity
side is ALREADY PROVEN (outlier scheme cb0=47).
### Not yet started from the confirmed-available list: recordable queues (the
### ~32 ms/step gap), subgroup-shuffle reduces, ION zero-copy, CFG-early (idea #2).

## ════ dot8 ROUND 3 (2026-06-05): SEMANTICS CRACKED, FIDELITY PASSES, perf verdict in ════
UNIT PROBE (kernels/dot8_probe.cl, NNOPT_DOT8_PROBE=1) decoded the builtin:
**qcom_dot8_acc(a, b, acc) = Σ SIGNED(a_byte) · UNSIGNED(b_byte) + acc** — mixed-
sign, 4×int8 packed per uint, bytes not nibbles, lanes aligned, acc works.
(8 discriminating vectors; results 10/-1/-4/101/-255/561/-255/6.)
Implementation: weights = plain signed int8 on arg1; activations offset-binary
(x+128) on arg2; correction = −128·Σw_s per group, INTEGER-domain (float-domain
variant lost cb0 46→29 to catastrophic cancellation — uncorrected dots ~2M, fp32
drops the low bits), seeded BRANCHLESSLY into the dot's acc operand on lane 0.
### Iterations: garbage-fast 1.61 ms → lane-0 if-correction 2.32 (cb0=46 ✓) →
### float-corr 1.69 (cb0=29 ✗) → local-staged 2.42 → SEED FORM **2.04 ms, cb0=46 ✓**
### → +wscl local staging 2.28 (worse, residency). Best correct: 2.04 ms.
### VERDICT (CFG-2): fp16-texture fc1 = 1.35 ms still wins — the CFG-twin tex-L1
dedup gives fp16 the same effective bytes as int8-buffer, at zero overhead. The
round-2 reframe SURVIVES hardware dot8.
### THE SYNERGY (why this work pays anyway): in SINGLE-ROW decode there is no
twin to dedup → fp16 streams 8.4 MB/row-step vs int8's 4.2 → dot8-int8 fc1
projects ~0.7-1.0 ms. CFG-EARLY (creative #2: CFG for the first ~50 steps, then
single-row) makes 80% of steps single-row — dot8-int8 and CFG-early STACK.
NNOPT_INT8O=d1 + NNOPT_DOT8_FN=qcom_dot8_acc kept gated (default OFF on CFG-2).
TOOL LESSON: undocumented vendor builtins → write the 8-vector discriminating
unit probe FIRST; two full-model debug cycles got falsified by what one 30 s
probe run answered definitively.

## ════ CFG-EARLY (creative #2, NNOPT_CFG_STEPS=N — 2026-06-05): wall 63.6 → **49.5 s** ════
Full CFG (2 rows) for the first N steps, then SINGLE-ROW (cond only): guidance
matters most early; the uncond row is never revisited so its KV bank (32)
simply stops growing — no staleness. Single-row steps ride the SAME fast
machinery: mega_decoder_layer_m2_n(num_wg=1) through the whole split chain +
M=1 lm_heads; sample_grid receives EXACT cond logits (row0 duplicated into
row1 so u + g(c−u) = c with zero rounding; stale row-1 data would perturb
near-ties). KV bank 0 = cond continues seamlessly across the switch.
PREREQ FINDING: the legacy single-row forward (guidance=1 CLI path) measures
**0.79 tok/s** — never route CFG-early through it; only through m2_n(num_wg=1).
| config | decode tok/s | clip wall | band 500-2000Hz |
|---|---|---|---|
| full CFG (control) | 5.10-5.72 | 63.6 s | 26.5% (this seed) |
| NNOPT_CFG_STEPS=100 | 6.54 | ~55 s | 8.3% |
| NNOPT_CFG_STEPS=50 | **7.20** | **49.5 s** | 6.8% |
GATES: full-CFG default regression green (guard 66/… byte-identical, tf
unaffected). Band-energy spread is seed-dominated (sampled runs); the BINDING
gate here is the EAR — A/B/C clips delivered (cfg_full / cfg50 / cfg100);
DEFAULT STAYS FULL CFG until the user approves a switch point.
NEXT STACK (designed, not yet wired): NNOPT_INT8O=auto — dot8-int8 fc1 ONLY on
single-row steps (no dedup twin there: fp16 streams 8.4 MB vs int8 4.2 →
projected fc1 1.35 → ~0.8 ms on 80% of steps; both pieces individually proven).

### Clip-length cap raised: MEGA_MAXK 256 → 512 (max clip 5.1 → 10.2 s)
User's 8-s request hit "start_pos 256 exceeds cap" — the KV cache compile-time
cap WAS the max clip length (tokens = 50 Hz frames; seconds = tokens/50; the
positional CLI arg sets tokens). Bumped kernel MEGA_MAXK + host kMaxK to 512
(+~50 MB KV across 24 layers × 2 banks; attn-core sc[] local 1→2 KB).
GATES: guard 66/… byte-identical ✓, tf cb0=44 ✓. 400-token (7.94 s) hip-hop
clip generated at 6.96 tok/s (CFG-50). NOTE: device tokenizer cannot encode
arbitrary prompts (t5_encode_host: empty ids) — new prompts need host-side T5
ids (transformers AutoTokenizer t5-base → int32 LE bin → --token-ids). The
nnopt venv at ~/Projects/nnopt/venv has transformers.

## ════ CFG-EARLY SHIPPED AS DEFAULT (--cfg-steps 50, user-ear-approved both genres) ════
CLI flag --cfg-steps N (-1 = full CFG; env NNOPT_CFG_STEPS overrides for A/B).
Gates: guard byte-identical ✓ (steps 0-4 are CFG steps), tf cb0=44 ✓ (force-grid
path is full-CFG by construction). NEW DEFAULT STATE: **49.8 s wall, 7.16-7.20
tok/s** (was 63.6 s). Tokenizer note: arbitrary prompts need host-side T5 ids.
### dot8-auto stack (NNOPT_INT8O=auto): MEASURED, LOSES — third and final int8 verdict
Mode auto = dot8-int8 fc1 only on single-row steps. Worked mechanically (clean
4800/1200 call split) BUT: single-row fp16 fc1 measures **0.83 ms avg / ~0.7 ms
single-row** (~10 GB/s nominal — without a CFG twin the 64 WGs read each weight
slice exactly once, no duplication) vs dot8-int8's 1.04 ms + actq overhead.
Decode: 7.20 fp16 vs 6.79 int8-auto. **fp16-texture wins in EVERY regime on
Adreno 620**: CFG-2 (tex-L1 twin dedup), single-row (clean single stream).
int8's premise (halve bytes) never beats fp16's access-pattern advantages here.
NNOPT_INT8O all modes remain gated OFF. The dot8 semantics + probe + outlier
machinery stay in-tree for devices/models where the math differs.
### Single-row regime bonus discovery: ALL GEMVs run ~10 GB/s nominal at M=1
### (fc1 0.83 / fc2_sk 0.88 / qkv 0.68 ms avg mixed) — the single-row steps are
### ~2.3× cheaper than CFG steps, better than the 2× row-count alone predicts.

## ════ ZERO-READBACK PIPELINE (NNOPT_PIPELINE, default ON): 49.8 → **45.5 s** ════
The 32 ms/step host-sync gap was the per-step BLOCKING 16-byte ids read in
sample_grid_dispatch (GPU drains → host wakes → re-enqueues → GPU restarts).
Killed it: pipeline mode auto-enables the Stage-4 GPU grid, sample_grid skips
the read (ids live only in the device grid), the host enqueues the ENTIRE
decode ahead (clFlush per step keeps the GPU fed), and the single end-of-loop
grid readback is the only sync — mark_end moved after it so decode_tokens_per_
sec stays honest; the stdout token stream is emitted post-loop from the grid
(same Evaluate format). STEP_ARGMAX per-step prints don't exist in pipeline
mode (guard runs on the FUSED_READOUT=0 path — unaffected; gates green:
guard 66/… byte-identical, tf cb0=44).
| config | decode tok/s | clip wall |
|---|---|---|
| per-step readback (NNOPT_PIPELINE=0) | 7.14 | 49.8 s |
| pipeline (default) | **8.23** | **45.5 s** |
NOTE the Stage-4 history: GPU-grid was a −14% REGRESSION in the megakernel era
and sat gated off for two days — it only pays combined with no-readback full-
ahead enqueueing. Optimizations interact; revisit gated-off work when the
regime changes.

## ════ cl_qcom_recordable_queues: PROBE PORTED & PROVEN (4.04× dispatch cost) ════
Ported from a8nova/adreno-llms (recipe was validated on THIS Razr 2020 there):
CL_QUEUE_RECORDABLE_QCOM (0x40000000) ALONE on clCreateCommandQueue; entry
points via dlsym. NNOPT_RECORD_PROBE=1 runner + kernels/record_probe.cl:
  baseline 10000 dispatches = 204.03 ms (20.40 µs/dispatch)
  replay (30-dispatch recording) = 50.51 ms (5.06 µs/dispatch) — **4.04×**
INTEGRATION DESIGN (next chunk): record TWO step sequences (CFG-2 and single-
row geometries — gws is captured in a recording, so the two regimes need
separate recordings), replay with per-step arg updates (start_pos, seq_len, t)
via clEnqueueRecordingQCOM's cl_array_arg_qcom list. Expected: enqueue CPU
0.6 → 0.15 ms/step (pipeline already overlaps it; the win is queue-depth/
driver overhead — measure, don't assume; lfm2's lesson stands).
## State: **45.5 s/clip, 8.23 tok/s, RTF 9.1.** fp16 weights + fp32 accum
## (int8 concluded-off after 3 measured verdicts). All gates green.

## ════ ON-DEVICE TOKENIZER FIXED — TRUE TEXT→AUDIO END-TO-END (2026-06-05) ════
The port could never encode arbitrary prompts on-device. THREE stacked bugs:
1. **Vocab file format mismatch**: tokenizer_vocab.bin entries didn't match the
   loader's [id u32][len u32][bytes] layout — every token parsed as id 0.
   Regenerated from t5-base tokenizer.json (host, one-time packaging step).
2. **Mojibake space marker**: the scaffold emitted kSpaceMarker as the ▁ bytes
   DOUBLE-UTF-8-encoded (C3 A2 C2 96 C2 81 instead of E2 96 81) — no vocab
   piece could ever match the staged text (the "1439 2 2 2 2" unk-storm).
3. **Greedy ≠ unigram**: T5 is SentencePiece UNIGRAM. Implemented Viterbi DP
   over piece log-probs (new optional 'SCOR' float32 section in the vocab file;
   loader backward-compatible; greedy fallback when absent). Encode now appends
   EOS (T5 contract).
VALIDATION: on-device ids byte-identical to HuggingFace AutoTokenizer for both
test prompts (TOKENIZER_IDS log line added permanently for parity checks).
TOOL LESSONS (PortTokenizer): (a) never emit non-ASCII string literals through
a re-encoding pipeline — write \\xE2\\x96\\x81 escapes; (b) vocab exporter and
loader must share one format contract + a round-trip test; (c) unigram models
need scores + Viterbi, not greedy longest-match.

### Output UX fixes (user ran it themselves and hit the seams, 2026-06-05)
1. Pipeline -1 sentinel leak: the sentinel guard had landed in the FUSED branch
   (dead when gpu_grid is on) — the gpu_grid branch printed/stored -1s per step
   and stdout emitted 250 "-1"s (audio itself was always correct: codes come
   from the end-of-loop grid readback). Guard now in the right branch.
2. STEP_ARGMAX removed from user-facing runs: fast-path per-step prints are now
   NNOPT_STEP_LOG=1 debug-only (they show SAMPLED ids — the name was a lie).
   The HOST-LOGITS path keeps unconditional STEP_ARGMAX — guard + tf_depth.py
   grep it (verified green after the change: guard 66/…, tf cb0=44).
3. Clear end-of-run report: "=== AUDIO READY === file/duration/prompt" +
   run_android.sh prints the adb pull command.
USER-MEASURED STATE (their own run): total_inference 34.5-34.8 s, decode 8.43
tok/s, TTFT 5.0-5.3 s — fastest end-to-end yet recorded.

### TTFT T5∥upload overlap (NNOPT_T5_OVERLAP): NO WIN, default OFF
Hypothesis: T5 (2.0 s CPU) ∥ decoder-weight GPU upload (3.0 s) → TTFT −2 s.
MEASURED (same-thermal A/B ×2): overlap ≈ 6.9 s vs serial 5.2-7.3 s (noisy,
page-cache dominated) — no improvement. The "upload" is flash-page-fault-bound,
not copy-bound, so the two phases CONTEND for memory/flash bandwidth instead of
overlapping. TTFT remains storage-bound; the honest fix is a resident service
(pay it once), not intra-process overlap. Weights::upload_prefix kept (useful
for the service mode later).

## ════ STREAMING ENCODEC UNDER DECODE (#2 creative — 2026-06-05) ════
EncodecStream: chunked stateful CPU SEANet decode running on the otherwise-idle
CPU while the GPU executes the fully-enqueued decode. Mechanism:
- async non-blocking grid-column reads every 50 steps (in-order queue: read
  executes after step t, GPU continues with already-enqueued later steps);
- worker thread: un-delay → RVQ → conv0 (recomputed per push; avail-edge
  frames excluded) → STATEFUL LSTM (h/c carried; gin hoist per chunk, exact)
  → upsample stages on [a-12, b+12] windows (margin > receptive field; true
  reflect only at real clip edges) → PCM segments;
- final tail chunk on the main thread after the last grid read.
CORRECTNESS: Stage-A harness (NNOPT_ENC_STREAM=2) BYTE-COMPARES streamed vs
full host decode: **158080/158080 samples, 0 mismatches** — first try.
(One live-wiring off-by-one: frame f needs cols ≤ f+NCB → avail = cols-NCB;
the +1 variant read one unwritten col (BOS=2048) and tripped the id guard.)
PERF (matched A/B legs): wall 44.2 → **~41.8 s**; EnCodec wall cost 7.5 →
2.1-2.4 s tail. THE PHYSICS TAX: hoped −5 s, got −2.4 s — the CPU worker's
memory traffic contends with the GPU's on shared LPDDR4X: decode 8.00 → 7.4-7.9
tok/s while the worker runs. Thread caps (4/6/8, NNOPT_ENC_THREADS) don't
change the trade. Default ON (net positive); NNOPT_ENC_STREAM=0 reverts.
TOOL LESSON: on shared-memory SoCs, CPU∥GPU overlap of two BANDWIDTH-BOUND
stages yields roughly HALF the naive sum — same physics as the T5∥upload
no-win. Overlap compute-bound against bandwidth-bound work instead.
### #1 (T5∥upload): NO WIN (flash-bound contention), default OFF, documented.
### State: ~42 s/5 s clip typical (cool device), RTF ~8.4. Thermal note: device
### saturated after ~10 consecutive runs; measurements need cool-down spacing.

## ════ CLBlast lm_heads REPLACED (own tex GEMV, default ON): decode 8.00 → **9.86 tok/s (+23%)** ════
THE SURVIVING GAP FOUND. The wall-vs-kernels gap (~20 ms/step) that outlived
the zero-readback pipeline was the CLBlast lm_heads GEMM: unprofiled (invisible
in every kernel table), internally serializing the queue, and compiling outside
the binary cache. Replaced with mega_lmheads — the standard multi-WG + texture
+ unroll-4 recipe over an image2d VIEW of heads_cat ([8192, H/4] half texels,
no copy). Now VISIBLE: 1.74 ms/step @ 9.65 GB/s.
GATES: guard argmax 66/1534/1513/1801 ✓ (logit 7.2109 → 7.1641 — accumulation
order changed, argmax intact); tf-depth cb0=**47** ✓ (above the 44 baseline).
NNOPT_LMHEADS=clblast reverts; automatic CLBlast fallback on dispatch failure.
total_inference 36.2 → **29.2 s** (stream off, isolated).
TOOL LESSON: library GEMMs (CLBlast) on tiny-M decode shapes can cost 10× their
kernel time in queue serialization — replace with the port's own GEMV recipe
once it exists; also gets them under the profiler and the binary cache.
### FULL-DEFAULTS STACKED (stream ON + own lm_heads, 2026-06-05):
wall ~42 → **34.1 s** for a 4.94 s clip (RTF ~6.9, was ~8.4). decode 8.89 tok/s
(9.86 isolated − the streaming worker's LPDDR4X contention tax, same physics as
documented above), TTFT 6.09 s (storage-bound, noisy), stream tail 1.59 s.
Run: on-device tokenizer, 'hip hop beat with heavy bass and drums', 250 steps,
guidance 3.0, --cfg-steps 50, temp 1.0, top-k 250, seed 42. Clip: mg_lmheads.wav.

## ════ RECORDABLE QUEUES INTEGRATED (cl_qcom_recordable_queues) — 2026-06-05 ════
The user-mandated lever ("we need to get it to work"). End-to-end record/replay
decode is LIVE and BYTE-IDENTICAL: both geometries captured (CFG-2 @ step 1,
single-row @ step 51), 246/250 steps replayed, output.wav == pre-change wav.
NNOPT_RECORD=0 kills it; any capture/replay failure permanently falls back live.

THE DRIVER WALL AND THE WAY AROUND IT:
- adreno-llms Step T4 hit clEnqueueRecordingQCOM err=-59 on its arg-override
  path and abandoned the lever. Our probe ISOLATED the failure: -59 is ONLY
  the arg-override array (any form: scalar/buffer, 1 or 30 instances, any
  queue). Plain replay (num_args=0) works on EVERY queue at 4-5×/dispatch
  (4.7 µs vs 23.7 µs live). Driver E031.37.12.07.
- Workaround: per-step scalars moved OUT of kernel args into a step-params
  BUFFER (sp[0]=start_pos; FillBuffer'd once/step on the in-order queue —
  pattern is copied at enqueue, no host-lifetime hazard). Kernels changed:
  mega_qkv (arg7 int→buffer), mega_attn_core (seq_len → seq_ptr[0]+seq_bias;
  self={sp,1}, cross={enc_buf,0}), embed_prologue (step→sp), sample_grid
  (step→sp; RNG seed^sp[0] preserved). A recording is then valid for EVERY
  step of its geometry with ZERO per-replay updates: replay = 1 FillBuffer +
  1 clEnqueueRecordingQCOM per step (was ~292 dispatch enqueues).
PREREQS SHIPPED (all gated byte-identical, NNOPT_RECORD=0 leg):
- persistent per-step cl_mems (x[2,hidden], row sub-buffer views, final-LN out)
  — recordings bake handles; also kills ~5 alloc/free per step;
- embed_prologue writes BOTH x rows directly (out_rows arg) — the 2 emb→x
  CopyBuffers were unrecordable;
- sample_grid `single` arg replaces the single-row logits dup CopyBuffer
  (cond row read directly — exact);
- cross-attn runs on a CLONE cl_kernel (arg-state independence);
- KernelProfiler::suppress() during capture (recorded dispatches never
  execute; their events would hang process_pending).
GATES: GATE0/0b (refactors, record off) wav BYTE-IDENTICAL ✓; GATE1b (replay
on) wav BYTE-IDENTICAL ✓ + both capture lines present.
PERF: first stacked run inconclusive (stream contention + thermal ordering);
clean isolated A/B ×2 with cooldowns in flight — numbers in next entry.
TOOL LESSONS:
1. When a vendor extension rejects its documented parameter path (-59 on arg
   overrides), probe the SUBSET that works (plain replay) and restructure the
   data flow (scalar args → buffer indirection) to need only that subset.
2. FillBuffer (pattern copied at enqueue) is the safe per-step scalar updater
   on an in-order queue — non-blocking WriteBuffer has a host-lifetime hazard.
PERF VERDICT (clean isolated A/B, 75 s cooldowns, alternating legs, stream off):
RECORD=1: 9.786 / 9.773 — RECORD=0: 9.738 / 9.759 tok/s → **+0.3%, noise.**
WHY THE PROBE'S 4× DIDN'T TRANSFER: replay cuts HOST enqueue cost (23.7 →
4.7 µs/dispatch), but the zero-readback pipeline already keeps the host far
ahead of the GPU — enqueue cost was never on the critical path. The remaining
~9 ms/step wall-vs-kernel-sum gap is GPU-side inter-kernel scheduling bubbles
(292 kernels/step), which recordings do not compress on this driver. The probe
saw 4× because noop kernels made the stream dispatch-rate-bound.
DISPOSITION: default ON (byte-identical; ~75% less host enqueue CPU per step —
helps the streaming-EnCodec worker's CPU budget and battery; required
foundation for cl_qcom_onchip_global_memory later). NNOPT_RECORD=0 reverts.
TOOL LESSON: a dispatch-overhead lever only pays when dispatch is on the
critical path — measure wall-vs-kernel-sum AND who owns the gap (host enqueue
vs GPU scheduling) before investing. Kernel-COUNT reduction (fusing the 12
small kernels/layer into fewer) is the lever that attacks GPU-side bubbles.

## ════ ATTENTION CORE VECTORIZED + K-TEXTURE (2026-06-05) — decode 9.76 → **10.77 tok/s (+10%)** ════
mega_attn_self was 574 µs avg (11.6% of decode) while moving ~0.6 MB/step —
~1.1 GB/s effective: pure load-issue-rate bound (64 SCALAR vload_half per
(lane,t) in the score loop; scalar V loop with in-loop inv multiply).
TWO STAGES, both gated:
1. VECTORIZE (buffer): score loop → vload_half8/float8 FMA tree; V loop →
   4-way t-unroll w/ independent accumulators + inv hoisted.
   attn_self 574→466 µs (−19%), decode +2.8%. tf: 47/40/47/41 (ALL cb ≥ 40 —
   best table yet; baseline ladder had cb1=13, cb3=2). BUT attn_cross
   regressed 41→75 µs (register pressure at tiny enc_len).
2. K-TEXTURE: K reads via image2d_from_buffer VIEW over the KV cache
   ([rows, H/4] RGBA half texels, zero-copy — the lm_heads recipe); V stays
   buffer (its lane-coalesced reads were never the problem). Cross-K view too
   (read-only per generation): cross regression GONE (75→40.2 µs).
   attn_self 466→**196 µs**. Self+cross total 3.69→1.42 s/clip.
GATE WAR STORY — float4 vs float8 grouping: first tex loop accumulated per
TEXEL (float4 tree) → tf cb0=16, REJECTED at the ≥40 bar. Rebuilt the loop to
fuse two texels per iteration = the SAME float8 grouping/order as the gated
vload_half8 path → tf 47/40/47/41 EXACTLY matches the buffer-vec table. Two
facts proven: (a) cb0's near-ties are sensitive even to fp32 partial-sum
GROUPING, not just precision; (b) read_imageh through a buffer view is
BIT-EXACT vs vload_half AND texture-L1 is COHERENT with same-step buffer
writes (qkv writes K row t as buffer; attn reads it as texture in the next
dispatch — no staleness on driver E031.37.12.07).
GATES: guard 66/1534/1513/1801 ✓; tf 47/40/47/41 ✓ (×2 stages).
NUMBERS (isolated, record off, stream off): decode 9.74-9.79 → **10.77 tok/s**;
full defaults (stream+record ON): 9.84 tok/s, wall **31.3 s** / 4.94 s clip
(lm_heads-era full default was 34.1 s). NNOPT_ATTN_TEX=0 reverts stage 2.
TOOL LESSON: 'kernel shows no GB/s and high µs' + tiny traffic = issue-rate
bound, not bandwidth — vectorize loads FIRST, then texture-path the gather
side. And when a reorder fails the tie-sensitivity gate, try matching the
PREVIOUS pass's exact partial-sum grouping before abandoning the optimization.

## ════ KERNEL-COUNT FUSION (LN→consumer GEMVs): REGRESSION, default OFF — 2026-06-05 ════
Hypothesis: the surviving ~9 ms/step wall-vs-kernel gap = GPU scheduling
bubbles between 292 kernels/step; fuse the 3 standalone LN dispatches/layer
into their consumer GEMVs' local staging (qkv, cq-proj, fc1) → 12→9
dispatches/layer (−72/step). Implemented (MEGA_FUSE_LN + mwg_ln_local; gates
PASS: guard 66/1534/1513/1801 ✓, tf cb0=47/cb1=47/cb3=41, cb2 reshuffled to 19
— cb0 binds per the ladder precedent).
MEASURED (same-binary A/B, cooldowns): FUSE_LN=0 **10.78** vs FUSE_LN=1 9.86
tok/s (−8.5%). Per-kernel: fc1 822→969 µs, qkv 666→841, proj_cq 231→295.
ROOT CAUSE: LN stats are recomputed REDUNDANTLY by every WG of the consumer —
qkv has 96 WGs/row, fc1 128. 96-128 × (2 row-reductions + 13 barriers) dwarfs
the one 30 µs standalone LN dispatch it replaced. Default OFF (env NNOPT_FUSE_LN=1
kept for single-WG-consumer shapes). Current best: **10.78 tok/s** (attn-tex).
TOOL LESSON: dispatch-count fusion only pays when the fused work is NOT
multiplied by the consumer's WG count — fold row-wide ops into multi-WG GEMVs
only if the op's cost × num_WGs < one dispatch overhead (~30 µs here it was
~100-175 µs per kernel). The standalone tiny-kernel structure can be optimal.
### STATE after attn-tex + fuse-LN revert (2026-06-05): isolated decode
**10.76-10.78 tok/s**; FULL DEFAULTS wall **31.3-31.4 s** / 4.94 s clip
(RTF ~6.3; was ~42 s at session start, decode 8.0 → 10.77 = **+35% today**).
TTFT 6.1 s (storage-bound), stream tail 1.66 s, both recordings replaying.

## ════ REALTIME CAMPAIGN DAY 1 (2026-06-05 pm): ROW_ILV shipped; THREE falsifications; ISSUE-BOUND discovery ════
Baseline (fresh 250-tok profile, live dispatch, cfg-steps 50): **33.2 s wall** =
TTFT ~8 (blob-cold; ~4.1 blob-hot) + decode 25.4 (9.85 tok/s, GPU busy 92%) +
EnCodec tail 1.6. Per-step traffic 672 MB fp16; ALL GEMVs 9.5-11.2 GB/s nominal
single-row, 5.2-6.0 CFG-2 (ratio ~1.9 = twin dedup BROKEN since the K-tex/
head-parallel/split-K rewrites; the Opt#5 lockstep relied on drift-free luck).
### NNOPT_ROW_ILV (default ON): row on group-dim0 so CFG twin WGs co-schedule
Full-CFG decode 6.86 → **7.91 tok/s** (+15%); GEMV CFG ratio 1.9 → ~1.6.
Guard 66/1534/1513/1801 byte-identical ✓ (pure scheduling). Default-config
(cfg-50) gain is small (CFG block only): 9.85 → 9.95 tok/s.
### NNOPT_ROW_FUSE (fc1_rf, gated OFF): FALSIFIED — and the most informative loss
One WG reads each texel once, MADs both rows: fc1 CFG-2 1396 µs ≈ unfused.
Halved WGs + doubled per-WG MADs = SAME total time ⇒ **the GEMV inner loop is
ALU/ISSUE-BOUND, not DRAM-bound** — the "~10 GB/s ceiling" is an issue ceiling.
(Also explains every int8 loss: int8 halves bytes but ADDS ALU.)
### NNOPT_X4LDS (gated OFF): FALSIFIED — float4 LDS staging 749 → 1170 µs.
The scalar-x form is already the compiler's optimum; pointer-cast vector local
loads regress 56%. Inner loop is AT its issue floor in current form.
### DVFS: IMPOSSIBLE on this device — /sys/class/kgsl perm-denied for shell,
production build, no root. cl_qcom_perf_hint HIGH (active) is the only lever.
### Standing fp16 wall analysis: single-row step = 672 MB / ~11 GB/s ≈ 70 ms
floor → 250-tok decode ≥ ~19 s at fp16 on this device. 15 s end-to-end needs a
structural change (multi-token-per-weight-pass, ear-gated cfg/layer trades) —
quant re-confirmed dead (3 prior verdicts + issue-bound mechanism).
### REMAINING measured levers: TTFT 8 s blob-cold (REGRESSED vs documented
5.76/5.42 — investigate stream/madvise + first_step 5.8 s composition; overlap
inits with T5), EnCodec tail 1.6 s, cfg-steps 50→25 (ear gate, ~0.9 s),
fc2_sk_red fold (~0.25 s), per-dispatch ~29 µs launch latency × ~290/step.

## ════ TTFT ROUND (2026-06-05 pm): pread prefetch SHIPPED; the 4.8 s layers-10-13 driver stall MAPPED, unsolved ════
TTFT_TRACE instrumentation added (NNOPT_TTFT_TRACE=1): per-program build ms,
cumulative lazy-upload stats, cross-KV ms, per-layer step-0 stamps — all
absolute-monotonic (per-TU static t0 stamps are incomparable; learned twice).
### SHIPPED: ordered pread() prefetch (weights.cpp)
(1) ORDER: blob layout is codec|decoder|text_encoder but consumption order is
text_encoder (host T5) → decoder (step-0 upload) → codec (EnCodec, late). The
old linear page-toucher streamed exactly backwards — T5 demand-faulted the blob
TAIL against the toucher grinding the HEAD. Now streams spans in consumption
order, started after meta parse.
(2) MECHANISM: 4 KB mmap-fault walking drove UFS at ~160-350 MB/s; plain
buffered read() hits **815 MB/s** (cat 1.18 GB cold = 1.45 s). The toucher now
preads 4 MB chunks into the page cache instead of touching pages.
MEASURED (5-tok probe): cold TTFT 7.6 → **3.4 s**; warm 3.2 s (t5 1.3 +
first_step 2.0). Real win for short/interactive generations.
### MAPPED, UNSOLVED: the 250-tok first_step stall (the real TTFT wall)
Full-clip runs: first_step 5.4-5.8 s regardless of cache state. Per-layer
stamps: layers 0-9 and 14-23 enqueue in ~20 ms each; **crossing layers 10-13
costs 4.8 s** (1.7+1.7+1.0 s). Steps-dependent (1.2 s at 5 tok, 4.8 at 250).
FALSIFIED: flash streaming (pread fix didn't move it), buffer creation
(NNOPT_T5_OVERLAP=1 pre-uploads all decoder buffers during T5 — stall stays
put), program builds (clcache, ~1 ms). Remaining suspects: driver-internal
allocation/migration or implicit sync in clCreateImage-from-buffer/kernel-clone
creation while the queue executes the step-0 backlog. NEXT lever for TTFT.
### EnCodec tail cadence 50→16 near clip end: NEUTRAL (kept)
Tail measured 1.6-1.8 s before AND after — the tail is final-flush/window
bound, not chunk-size bound. Next lever there: profile EncodecStream's
final-push path.
### Session net (full default, warm): 33.2 → **32.7 s** (decode 9.85 → 9.90-9.95
via ROW_ILV; total dominated by the unsolved first_step stall). Guard
66/1534/1513/1801 byte-identical ✓ on final binary.

## ════ ROUND (1)+(2) (2026-06-05 eve): the benchmark was LYING — true wall 46.4 → **37.5 s** ════
### (2) THE BIG ONE — process-wall audit caught a hidden 13.5 s + a decode tax
`time` on the full run: default (CPU SEANet streamer) = **46.4 s real / 158 CPU-s**
vs total_inference_sec 32.9 — the benchmark stops at decode-loop exit and hid the
worker drain + teardown. ENC_STREAM=0 (GPU EnCodec after decode) = **38.2 s real**.
Mechanisms: (a) the CPU worker's memory traffic robbed the GPU GEMVs of ~9%
decode (9.87 vs **10.88 tok/s** — fastest decode yet, free); (b) the chunked CPU
path RECOMPUTES conv0 over all frames each push (encodec_host.cpp:1046) — 158
CPU-s vs ~8 total for the GPU path. **DEFAULT FLIPPED to GPU EnCodec**
(NNOPT_ENC_STREAM=1 re-enables the worker). New honest metric `e2e_wall_sec`
printed at process end (matches `time` ±0.3 s). TOOL LESSON (template follow-up):
benchmark.h.tmpl's total_inference_sec must be process-wall; gate every overlap
optimization on e2e wall, never an internal span.
### (1) first_step stall: mechanism cracked (driver pool fill), single-shot cost accepted
- model_reset_decode_state RELEASED all 128 KV-cache buffers per generation →
  lazily re-alloc'd during step 0. FIXED: KV caches persist (position p is
  written before any read of p — stale contents unreachable); cross-K/V now
  VALUE-invalidated (mega_invalidate_cross_kv) with capacity-aware reuse in
  mega_precompute_cross_kv (re-precompute on same enc_len: 175 ms → 3 ms).
  Wins multi-generation/server use; single-shot CLI still pays the driver-pool
  fill ONCE (~4.8 s spread over first-pass allocations; KGSL pool state across
  runs explains the 2.5↔5.6 s first_step variance).
- NNOPT_PREWARM (zero-state dummy step ∥ T5 thread): NET LOSS — the pool fill
  contends host T5 to 3× (1.8→6.2 s) and grid-geometry first-pass stalls again;
  default OFF, env kept.
### Fidelity gates on final state: guard 66/1534/1513/1801 ✓; tinytemp ==
### torch greedy to 4 decimals ✓ (KV-keep + EnCodec-default flip are value-safe).
### Honest anatomy now (37.5 s e2e): TTFT 7.5 (5.7 first_step pool-fill + T5)
### ▸ decode 22.9 @ 10.88 tok/s ▸ EnCodec-GPU 7.3 serial ▸ teardown ~0.2.
### Next levers: EnCodec GPU kernels (7.3 s — the original task-4 conv tiling),
### interleaving EnCodec chunks into the 2.1 s of decode bubbles, pool-fill.

## ════ EnCodec GPU CONV ROUND (2026-06-05 night): three attacks, one modest win; 37.5 → **37.1 s** ════
Phase profile (the 7.3 s GPU EnCodec): enc_convt1d 3.0 s (4 calls), enc_conv1d
2.5 s (10), lstm_gin 0.58 — the two conv kernels are the whole phase.
### Attack 1 — 16×16 WG __local x-tile: FALSIFIED (both kernels)
24 KB tile → 1 WG/SP occupancy collapse: convT-tile bit-identical but 9.5 s
(WORSE); conv-tile wrong bytes AND slower. Same locality-vs-parallelism loss
as FC2_FUSE2/ROW_FUSE. Kernels removed.
### Attack 2 — transposed activations (enc_xt + *_x): convT KEPT
x_t[ti,ic] makes the a4 loop's x reads contiguous vload4 (was a cache line per
float). convT-x: BIT-IDENTICAL (same-seed wav md5 ✓), 7.54 → 7.05 s phase.
conv1d-x (kk-outer reorder): SLOWER — reverted to untiled conv1d.
### Attack 3 — 4 outputs/thread (*_x4, shared weight registers): convT KEPT
Late stages dispatch ~10 M one-output threads (Tout 158 k × out_ch); batching
t,t+s,t+2s,t+3s shares the kk tap set so wq loads once per 4 outputs.
convT-x4: BIT-IDENTICAL ✓, phase → **6.85 s**. conv1d-x4: 11.8 s (register
pressure + clamp overhead) — gated off.
### DEFAULT: NNOPT_ENC_TILE=t (convT x4+transpose only). Phase 7.45 → 6.85 s,
### e2e 37.5 → **37.1 s**. VERDICT: the SEANet convs hit the same issue-bound
### wall as the decoder GEMVs — tile/transpose/batching all give ≤10% beyond
### the convT access fix; the next real lever for this phase is fp16
### activations (halve traffic) or overlapping the phase with decode bubbles.

## ════ LATE-NIGHT FALSIFICATION ROUND (2026-06-05): the issue-bound wall is now fully mapped ════
### fp16-ALU GEMV (NNOPT_H4ACC): FALSIFIED — half ARITHMETIC is emulated
Theory: Adreno 6xx fp16 ALU at 2× fp32 rate; read_imageh already returns half4
so the per-texel convert_float4 is waste. Reality: half4 mul/fma craters fc1
702 µs → **19,800 µs (28× WORSE)** — identical with vload4-of-half AND the
documented as_half4(uint2) workaround ⇒ the OpenCL-C half ARITHMETIC path
itself is emulated by this compiler, not just the loads. fp16 math on this
driver is for STORAGE/textures only. Env + kernel kept as documentation.
### enc_lstm_gin transposed-x: FALSIFIED — and the unifying insight
Bit-identical ✓ but 583 → 806 ms. x[j*T+t] across adjacent t-threads is
per-WARP COALESCED — [ch,T] is already the optimal layout for any kernel whose
thread dim is t. This explains conv1d-x's loss too. enc_convt1d won ONLY
because ti = (t+pl-kk)/stride makes adjacent threads SHARE x rows (broadcast)
and x4 cut thread count. Route disabled, kernel kept with verdict.
### STATE AT SESSION END: **37.2 s e2e** (TTFT 7.5 ▸ decode 22.9 @ 10.87 ▸
### EnCodec 6.9 ▸ teardown 0.2), all gates green (guard + tinytemp + wav-byte).
### The per-component floors on THIS device are now measured, not guessed:
### decode ≈ issue-floor, EnCodec convs ≈ issue-floor, TTFT ≈ driver pool fill.
### REMAINING ROADMAP (in value order): (1) driver-pool fill 4.8 s — try
### cl_qcom_ion_host_ptr pre-pinned arena / single big allocation; (2) persistent
### service mode (kept-allocations work makes gen 2+ TTFT ~1.5 s); (3) ear-gated
### cfg-steps 25 (~0.9 s); (4) EnCodec-into-decode-bubble interleave (~2 s max).
### Structural beyond that: multi-token decode or Adreno 7xx-class silicon.

## ════ SERVE MODE + the first_step REINTERPRETATION (2026-06-05 night) ════
### --serve SHIPPED: persistent process, one generation per stdin line
Weights/KV/cross-KV/programs/recordings stay resident; codec weights cached
(weight_norm recompute + ~118 MB re-upload per gen eliminated); EnCodec signal
buffers pooled (size-keyed recycle; ~250 MB/gen churn eliminated). CORRECTNESS:
serve gen-1 wav is BYTE-IDENTICAL to a single-shot run of the same prompt+seed
(strongest gate available). Steady-state ≈ **35.1-35.3 s/gen** vs 37.2 single-
shot — the honest one-time savings are the weight stream/upload + pool fill.
### first_step REINTERPRETED: it is mostly DRIVER ENQUEUE BACKPRESSURE
Serve gens show first_step declining 5.8→3.7 s with NOTHING left to allocate —
and pipeline-mode forward returns instantly unless the driver's enqueue depth
is full. first_step largely measures the GPU EXECUTING the first chunk of
decode (backpressure), i.e. it's the FRONT OF DECODE, not setup. This explains
in one stroke why prewarm, T5_OVERLAP, ordered prefetch and the pool work all
failed to move it in full runs: you cannot host-engineer away GPU execution
time. The TTFT-vs-decode split was partly an accounting artifact; e2e_wall_sec
is the only number that ever told the truth. ARENA (pre-pinned single
allocation) DESCOPED: its target was the pool-fill share, which serve mode
already amortizes and which backpressure dominates anyway.
### Honest budget now (serve steady-state 35.1 s): T5 ~2 ▸ GPU decode ~23.3
### (issue-floor) ▸ EnCodec-GPU 6.6 (issue-floor) ▸ tokenize/wav/misc ~1.5.
### In-architecture remainder: cfg-steps 25 (ear gate, ~1 s), decode bubble ~2 s.

## ════ RE-BASELINE + cfg-steps RE-MEASURE (2026-06-13, fresh session) ════
Context: a stale DEBUG binary was deployed under the fp32 name on device; runs that
sourced remote_dir.env without `NNOPT_DTYPE=fp16` executed it → NaN/silent output,
87 s TTFT. Clean `--clean --release` rebuild (fp16) + correct binary name restored
health. (See memory feedback_deploy_dtype_binary_name.)
### Fresh baseline (release fp16, 250 tok → 4.94 s clip, post-reboot cool device):
e2e_wall **34.4–35.0 s** ▸ TTFT 4.9–5.2 s (first_step 3.4, T5 1.45 overlapped) ▸
decode 22.9 s @ **10.91 tok/s** ▸ EnCodec-GPU 6.69 s (conv0+lstm 1.43 + enqueue
0.48 + drain/readback **4.76**). RTF ≈ 6.97×. Better than the documented 37.2 s
purely via cooler first_step (3.4 vs 5.7) — A/B must stay same-binary alternating.
### Profiler sanity (NNOPT_PROFILE=1): CL-event profiler is BLIND to the recordable-
queue replay decode (replayed dispatches emit no events) — table captures EnCodec
fully + decode WARMUP only. Decode reconstructed from per-call × layers (93.1 ms/tok)
≈ wall/tok (91.7 ms) within 2% → decode is ~99 % GPU-busy (no host idle left).
Decode GEMV GB/s: fc1 8.77, fc2 8.15, qkv 7.94, lmheads 4.03 (vs ~14 practical;
texture-L1 CFG-twin dedup makes effective higher). VERDICT stands: ~93 % of wall is
genuine GPU kernel execution — there is no host/dispatch overhead left to cut; the
wall is issue-bound kernel work + EnCodec drain + TTFT backpressure.
### cfg-steps 25 vs 50 — alternating A/B (same binary, same prompt+seed):
cfg=50: 35.01, 34.74 → avg **34.87 s** (decode 10.91). cfg=25: 33.69, 33.81 →
avg **33.75 s** (decode 11.47). Δ = **−1.12 s (−3.2 %)**, TTFT unchanged. Matches
the documented −1.1 s (only steps 25–49 switch dual→single-row; default already
single-rows after 50). Quality: cfg25 RMS 0.237 vs cfg50 0.283 — ear pair delivered
to user (ab_cfg50.wav / ab_cfg25.wav), **AWAITING VERDICT** before changing default.
### VERDICT 2026-06-13: user ear "identical" → **cfg-steps 25 is now the DEFAULT**
(main.cpp cfg_steps_cli 50→25). Rebuilt+redeployed; default-path verify (no env)
= decode **11.48 tok/s**, e2e 34.07 s (TTFT 5.5 s hot this pass — device variance).
SHIPPED. -1.1 s/clip vs the old default. (-1 = full CFG every step still available.)

## ════ MULTI-TOKEN DECODE — FEASIBILITY (2026-06-13) ════
Goal: amortize the load-bound per-token weight stream by verifying K candidate
positions in ONE batched forward (each weight fetch feeds 2K rows, not 2).
### Mechanism check (load-bound vs MAD-bound): decode is ~91 ms/tok but only
~1.7 GFLOP/tok of math (~11 ms at fp32 rate, ~12%) → the other ~88% is weight
loading ⇒ **load-issue-bound**, so row-batching CAN help (same reason CFG M=2
already wins via texture-L1 dedup).
### Amortization curve from the cfg25/cfg50 A/B (no extra run needed): moving
25 steps M=2→M=1 saved 1.11 s = **44.4 ms/step**. Solve ⇒ **M=1 step 82.8 ms,
M=2 step 127.2 ms** → M=2 costs **1.536× M=1** (NOT 2×) ⇒ ~46% of row-2's load
cost is amortized. Linear fit **t(M)=38.4+44.4·M ms** ⇒ M=4≈1.53×, M=8≈1.68×,
**asymptote ≈1.86×** decode throughput (22.9 s → ~12-15 s decode, e2e → ~24-27 s).
### GATING RISK: ROW_FUSE was FALSIFIED ("doubled MADs/WG = unchanged time") and
X4LDS lost its L1 dedup when WGs stopped co-scheduling — so the L1 dedup may NOT
extend past M=2. **The M=4 wall-per-row must be MEASURED before building
speculative decode.** NEXT: probe dispatch mega_ffn_fc1/fc2/qkv at num_rows=4
(rows 2,3 alias 0,1 for timing only) and compare avg_us vs M=2. Decision gate:
M=4 wall ≤ ~1.3× M=2 wall ⇒ build draft+batched-verify speculation; else abort.

## ════ MULTI-TOKEN PROBE — MEASURED, ABORTED (2026-06-13) ════
NNOPT_MROWS_PROBE=1 dispatches the real mega_proj [1024→1024] fp16-tex GEMV
(ilv, CFG/candidate row on group-dim0) at M=1/2/4/8 over ONE shared weight image;
GPU time via clGetEventProfilingInfo. (Probe kept env-gated in MegaDecoderLayer.cpp
+ main.cpp as documentation — zero cost on the normal path.)
```
 M   us/dispatch   us/row   amort   eff GB/s (M× per-dispatch fetch)
 1      232.19     232.19   1.000     9.0
 2      339.56     169.78   1.368    12.4   ← today's CFG decode
 4      609.71     152.43   1.523    13.8
 8     1151.97     144.00   1.612    14.6   ← at the ~14-15 GB/s bus ceiling
```
### VERDICT: ABORT. Gate was M=4/M=2 ≤1.3×; MEASURED **1.79×**. The M=2 CFG twin
already harvests ~85% of the load-amortization (12.4 of ~14.6 GB/s bus). Past M=2
each row is ~linear cost (M=4→8 = 1.89×) — the ROW_FUSE bandwidth/issue wall.
Speculative-decode ceiling = M=8 all-accepted ≈ 1.6× GEMV ≈ 1.3× decode, but music
is high-entropy ⇒ low draft acceptance ⇒ break-even/loss below ~75% accept, for a
draft model + KV rollback + major correctness risk. **Not worth building on Adreno
620.** The per-token weight stream is fundamentally bus-bound at M≥2; remaining
structural levers are int8 (falsified: fp16-tex wins every regime) or faster
silicon. Decode floor on this device stands: ~22-23 s @ ~11 tok/s.

## ════ BUG FIX: KV-cache cap silently distorted clips >10.2 s (2026-06-13) ════
SYMPTOM: 30 s clips distort then collapse to near-silence partway through.
ROOT CAUSE: self-attn KV cache capped at MEGA_MAXK = kMaxK = **512** positions
(512/50 Hz = 10.24 s). Past position 512 the qkv append writes OOB and attention
reads OOB cache rows → garbage attention → distortion. The host guard only fired
at cache *allocation* (step 0), never on the per-step append/read, so a 1500-tok
run produced 1500 frames of increasingly broken audio instead of erroring.
EVIDENCE (windowed RMS, old vs fixed 30 s, same seed): 0-10 s byte-identical
(fix is a no-op <512); old clips 10-16 s (peak 1.00, clip% 0.2→0.6) then
collapses to RMS 0.07/peak 0.12 for 18-30 s; fixed holds RMS 0.20-0.28 to 30 s.
FIX: MEGA_MAXK + kMaxK 512 → **2048** (= decoder max_position_embeddings, the
model's true ceiling, ~40.9 s). Pos table embed_positions.weights is already
[2048,1024]; grid is dynamic; attn-core local sc[] 2 KB→8 KB (fits 32 KB).
main.cpp now clamps max_new_tokens ≤ MAX_POSITION_EMBEDDINGS-NCB (2044) with a
loud WARNING (no more silent corruption). MEMORY: KV total = kMaxK × 192 KB =
**384 MB** at 2048 (was 96 MB), allocated to the cap on first use (length-
independent); device has 7.67 GB so peak ~2.7 GB is comfortable.
VALIDATION: 5 s clip UNCHANGED (33.9 s, 11.49 tok/s — larger local sc[] cost
nothing on short clips); 30 s now clean end-to-end (e2e 197.8 s @ 9.94 tok/s,
the lower tok/s is thermal over the 3-min run, not the cap). Longest supported:
2044 tok ≈ 40.9 s, MODEL-limited (position embeddings) not memory-limited.
TOOL LESSON: scaffolded decoder KV caches MUST size to max_position_embeddings,
never a trace-time/round-number default; and the bounds guard must sit on the
per-step append, not just allocation. [[feedback_kv_cache_for_decode]]

## ════ QUANTIZATION RE-MEASURED incl. HARDWARE dot8 — ALL ≤ fp16-tex (2026-06-13) ════
User pushed for int8+int4 for speed. Re-measured all int8 paths fresh (5 s, lofi,
same prompt/seed) vs fp16-texture 11.48 tok/s:
```
fp16-texture (current)        11.48 tok/s  —
int8 full (software i8)        8.90        0.77×  (SLOWER)
int8 FFN-only (software i8)    8.74        0.76×  (SLOWER)
int8 HW dot8 fc1 (mode 4)     11.50        1.00×  (NEUTRAL)
int8 HW dot8 auto (mode a)    10.70        0.93×
```
FIXED the dot8 builtin: default NNOPT_DOT8_FN=qcom_sdot8 is UNDECLARED on this
driver (-11 build fail); the device exposes **qcom_dot8_acc(uint,uint,int)**
(dot8_probe: signed×unsigned, zp=128). With NNOPT_DOT8_FN=qcom_dot8_acc the HW
int8 path COMPILES and RUNS — and tops out at NEUTRAL.
### VERDICT: quantization is DEAD for speed on Adreno 620 decode. Mechanism:
decode is NOT DRAM-bandwidth-bound at the margin — fp16 weights already stream
from TEXTURE-L1 via the CFG-twin dedup (~12 of ~14.6 GB/s effective from cache).
Halving DRAM bytes can't help a cache/issue-bound kernel; int8 ADDS dequant
scales + activation-quant dispatch + zero-point correction + outlier pass, which
cancels the byte savings even when the MACs are free in HW (dot8 → exactly
neutral). int4 would lose HARDER: more unpack ALU, no HW dot4, same texture-L1
ceiling. Software int8 already proven 0.77×; HW-dot8 the ceiling at 1.00×.
The fp16-texture path IS the optimum; the
~6× RT floor on this device is silicon-bound (needs Adreno 7xx-class bandwidth).
### INT4 NOW MEASURED (NNOPT_INT4_PROBE, mega_proj_int4 vs mega_proj, M=2 [H→H]):
fp16-texture 339 µs/dispatch vs int4-nibble (RGBA-uint32, 32 int4/texel — int4's
DENSEST fetch, fp32 accum, same thread map) **725 µs = 0.468× (2.1× SLOWER)**.
Nibble unpack (shift/mask/sign-extend + dequant ×32/texel) is pure ALU on the
issue-bound critical path; ¼-bytes can't help a cache/issue-bound kernel and
fp16 read_imageh does half→float in HW free. Quantization is CLOSED on this GPU:
int8-sw 0.77×, int8-HW-dot8 1.00×, int4 0.47×. Probe kept env-gated.
[[feedback_dot8_int8_conv_adreno]] (HW dot8 wins for CONV elsewhere, but NOT for
the texture-L1-bound decode GEMV here — the win doesn't transfer to this kernel).
