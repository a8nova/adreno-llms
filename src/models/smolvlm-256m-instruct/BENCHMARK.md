# SmolVLM-256M decode optimization log

Device: Motorola Razr (2020) `smith`, GPU Adreno 620v2 (Snapdragon 765G), LPDDR4X.
Build: `NNOPT_DTYPE=fp16` (`SmolVLM_256M_Instruct_inference_fp16`).
Workload (VLM): prompt `"Describe this image."` + `--image fixtures/sample.jpg`, max_tokens=64.
Prompt expands via reference/test_input_ids.bin → **874 prompt tokens** (832 image placeholder tokens + text), decode emits up to 64 new tokens.
Measurement: `NNOPT_DEBUG_LAYERS=0`, 3 back-to-back runs, median reported. `BENCHMARK` lines parsed from stderr.

## Baseline

| run | prefill tok/s | decode tok/s | TTFT s | peak MB | total s |
|----:|--------------:|-------------:|-------:|--------:|--------:|
| 1   | 27.9638       | 0.1108       | 31.38  | 955.65  | 600.14  |
| 2   | 28.4033       | 0.1121       | 30.85  | 955.91  | 593.05  |
| 3   | 27.4836       | 0.1039       | 31.89  | 955.86  | 638.45  |
| **median** | **27.9638** | **0.1108** | **31.38** | **955.86** | **600.14** |

Observations:
- Decode is catastrophically slow (~0.11 tok/s ≈ 9 seconds per token).
- Prefill is also low (~28 tok/s) but is M=many GEMM so CLBlast at least has something to chew on.
- 874-token prompt + 64 tokens decode → ~600 s total wall clock.
- Image preprocessing + vision tower + projector cost is rolled into TTFT (~31 s); decode itself = total − TTFT ≈ 569 s / 63 new tokens ≈ 9.0 s/token.

## Ceiling

Decode is memory-bandwidth bound at batch=1.

- **Weights touched per decoded token (fp16):**
  - `lm_head`: 576 × 49280 × 2 B ≈ 54 MB
  - 30 × (4 attn projs × 576 × 576 + 3 MLP projs × 576 × 1536) × 2 B ≈ 199 MB
  - KV cache reads at seq_k≈900: 30 × 2 × 900 × 192 × 2 B ≈ 21 MB
  - **Total ≈ 274 MB/token**
- **LPDDR4X bandwidth (Snapdragon 765G, Adreno 620v2):** ≈17 GB/s peak, ~10 GB/s sustained at GPU.
- **Ceiling tok/s:**
  - Theoretical peak: 17 GB/s ÷ 0.274 GB ≈ **62 tok/s**
  - Realistic (10 GB/s sustained): **36 tok/s**
- **Compute ceiling (fp16):** Adreno 620 ≈ 200 GFLOPS fp16 peak. Per-token FLOPs ≈ 2 × params ≈ 512 MFLOPS → compute-bound floor ≈ **390 tok/s** (10× margin over bandwidth ceiling — confirms decode is bandwidth-bound).

**Baseline 0.111 tok/s is ~0.3% of the realistic ceiling.** Headroom is enormous; the gap is structural inefficiency (kernel dispatch overhead, wasted visibility ops, unfused GEMVs), not memory bandwidth.

## Profile

Profile run: `NNOPT_PROFILE=1 NNOPT_DEBUG_LAYERS=0`, prompt `"Describe this image."` + image, 8 decode tokens. clFinish-bracketed host-side timing per stage. Note: clFinish adds ~5-20 ms overhead per begin/end pair, so absolute mean_us values are slightly inflated. Rank order is reliable.

Decode-phase breakdown (totals across 8 decoded tokens; per-token = total / 8):

| stage | total_ms (8 tok) | per-token (ms) | calls | pct of decode | notes |
|---|---:|---:|---:|---:|---|
| `decode_10_visibility_norms` | 42061 | **5258** | 480 | **47%** | unconditional dump-only RMSNorm pairs in backbone.cpp |
| `decode_25_rmsnorm_post` | 20703 | 2588 | 240 | 23% | internal post-attn norm — dispatch overhead-dominated |
| `decode_22_rmsnorm_in` | 16191 | 2024 | 240 | 18% | internal pre-attn norm — same root cause |
| `decode_21_residual_copy` | 8861 | 1108 | 510 | 10% | two `clEnqueueCopyBuffer`s per layer |
| `decode_23e_attn_softmax` (one entry) | 6139 | 767 | 60 | 7% | only some steps had this (every other token?) |
| `decode_26_mlp` (outer)  | 3685 | 461 | 240 | 4% | wraps gate+up+swiglu+down |
| `decode_23_self_attn` (outer) | 3205 | 401 | 240 | 4% | wraps qkv+rope+kv_write+score/sm/out+o_proj |
| `decode_26a_mlp_gate_up` | 2220 | 278 | 240 | 2.5% | two CLBlast GEMVs |
| `decode_23a_qkv_proj` | 1450 | 181 | 240 | 1.6% | three CLBlast GEMVs |
| `decode_26c_mlp_down` | 1313 | 164 | 240 | 1.5% | one CLBlast GEMV |
| `decode_40_lm_head` | 858 | **107** | 8 | 0.9% | smaller than expected — GEMV is fine |
| `decode_30_final_norm` | 581 | 73 | 8 | 0.6% | one RMSNorm |
| `decode_23g_o_proj` | 568 | 71 | 240 | 0.6% | |
| `decode_23f_attn_out` | 277 | 35 | 240 | 0.3% | |
| `decode_24_residual_add` | 165 | 21 | 240 | 0.2% | |
| `decode_23d_attn_scores` | 156 | 19 | 240 | 0.2% | |
| `decode_23b_rope` | 155 | 19 | 240 | 0.2% | |
| `decode_50_logits_d2h` | 122 | 15 | 8 | 0.1% | |
| `decode_26b_swiglu` | 110 | 14 | 240 | 0.1% | |
| `decode_23c_kv_cache_write` | 158 | 20 | 300 | 0.2% | |
| `decode_00_embedding` | 10 | 1.2 | 8 | 0% | |

Top findings:
1. **47% of decode wasted in visibility-only RMSNorms** (backbone.cpp lines ~344-352 and ~419-426). These compute results that are only dumped/checked when `NNOPT_DEBUG_LAYERS=1`; in release runs they must not run at all.
2. **RMSNorm dispatch overhead is huge.** Per-norm mean ≈70-90 ms for a 576-element op — that is driver/dispatch overhead, not compute. A 576-element fp16 reduction on Adreno 620 should be sub-millisecond. Likely cause: kernel launches each grab a single workitem (gws=576 with no local size hints) and the queue is profiling-enabled all the time (cost varies by driver).
3. **Residual `clEnqueueCopyBuffer` pairs** are ~10% — two copies per layer × 30 layers per token. Single fused-add-store kernel can eliminate both.
4. **lm_head is NOT the bottleneck** (only 0.9% of decode). The 28M-MAC GEMV is well-served by CLBlast for now.
5. **Attention math at decode is cheap** (seq_q=1, GQA). Only the QKV projections matter, and they sum to ~250 ms/token across 30 layers — driver overhead, not compute.

Implication: the first 4-5 optimizations need to attack **kernel-dispatch overhead** (fuse ops, kill visibility ops, eliminate alloc/free pairs), not flop count or memory bandwidth.

## Session summary (2026-05-16)

Decode throughput: **0.111 → 4.411 tok/s (39.7× speedup)**. Total inference for the standard VLM workload (874-token prompt + 64 decoded tokens with image): **600 s → 37 s (~16× wall-clock)**. Prefill also improved from 28 → 39.4 tok/s (+40%) as a side-effect of opt #8 (parallel softmax). Realistic memory-bandwidth ceiling is ~36 tok/s — we are now at ~12% of the ceiling, up from ~0.3%.

Top three findings (root causes), in order of impact:

1. **`op_LlamaRMSNorm` was rebuilding its OpenCL program from source on every call** (~120×/decoded token). Singleton init + dropping the post-dispatch `clFinish` was the single largest win (+5.6× decode). Pattern: every decode-path op MUST cache program/kernel via the `ensure_initialized` singleton pattern that the other ops already use.
2. **`pytorch_linear` routed M=1 decode GEMVs through CLBlast Hgemm**, which is tuned for large GEMMs and dispatch-bound at M=1. A 60-line custom GEMV kernel (workgroup-per-output-row, `half8` vector loads) was +3× decode.
3. **Visibility-only RMSNorms in `backbone.cpp`** were running unconditionally even when `NNOPT_DEBUG_LAYERS=0`. They cost 47% of decode at baseline; gating with `#ifdef NNOPT_DEBUG` + the runtime env check restored ~+20%.

Fusions then chipped the rest down: gate+up+swiglu into one kernel (+11%), Q/K/V into one kernel (+27%), GPU argmax with int-only readback (+1.4%).

A fused single-kernel decode attention (FlashAttention-style online softmax) was implemented and validated for correctness, but reverted: only 9 workgroups (one per query head) underutilizes Adreno 620's single CU compared with the 3-kernel pipeline's 8100-workitem `gqa_attn_scores`. The kernel is retained in `src/utils.cpp` (`decode_attn_fp16`) for a future split-K attempt.

Where time is still going (post-opt-7 profile, per decode token, profile-inflated):
- `decode_23_self_attn` (the trio scores+softmax+attn_out + qkv + rope + kv_write + o_proj): ~286 ms — softmax kernel uses 1 workitem per row, the obvious next target
- `decode_26_mlp` (gate+up+swiglu fused + mlp_down): ~94 ms — mostly dispatch overhead, GEMV math is small
- everything else: ~30 ms (lm_head 13 ms, norms 40 ms, residuals 16 ms, kv_write 18 ms, d2h 1 ms)

Suggested next opts (not in this session):
- Split-K fused decode attention (one workgroup per (q_head, seq_k_chunk)) — biggest remaining single lever; needs partial-result combine kernel
- Parallel-row softmax kernel (replace 1-workitem-per-row with 64-lane workgroup reduction)
- Persistent scratch buffers across decode steps (~300 `clCreateBuffer` ops/token currently)
- Fuse `residual_add` into RMSNorm prologue
- Concatenate weights at load time so the fused QKV/gate_up kernels read a single contiguous matrix

## Experiments
| # | name | decode tok/s | Δ vs prev | Δ vs baseline | cos_pass | notes |
|--:|------|-------------:|----------:|--------------:|:--------:|-------|
| 0 | baseline (median of 3) | 0.1108 | — | — | ✓ | reference |
| 1 | gate visibility-only RMSNorms in backbone.cpp behind `NNOPT_DEBUG_LAYERS` (`src/ops/backbone.cpp`) | 0.1334 | +20% | +20% | ✓ | the two dump-only RMSNorms per layer were running unconditionally; #ifdef-gated against `NNOPT_DEBUG` + runtime `nnopt_debug_layers_enabled()` check |
| 2 | cache `rms_norm.cl` program + kernel in singleton; drop per-call `clFinish` (`src/ops/rms_norm.cpp`) | **0.7353** | **+451%** | **+564%** | ✓ | RMSNorm was rebuilding the OpenCL program from source ~120×/token. Singleton init pattern (matches other ops) + remove blocking `clFinish` after dispatch. Median of 2 (runs identical to 4 sig figs). Total inference 600s → 112s. |
| 3 | eliminate per-layer residual `clCreateBuffer`+`clEnqueueCopyBuffer` pairs in decoder layer (`src/ops/llama_decoder_layer.cpp`) | 0.7598 | +3.3% | +585% | ✓ | reuse `hidden_states` directly as residual1 (preserved through RMSNorm+Attention) and `attn_out` as residual2 (preserved through RMSNorm+MLP). Saves 2 buffer allocs + 2 device-to-device copies per layer × 30 layers = 120 ops/token. |
| 4 | custom fused GEMV kernel for M=1 decode path; route from `pytorch_linear` (`src/utils.cpp`) | **2.2090** | **+191%** | **+1894%** | ✓ | Singleton fp16 GEMV: workgroup-per-output-row, 64-lane reduction over K via `half8` vector loads. Replaces 7 CLBlast Hgemm calls per layer (q/k/v + o + gate + up + down) + lm_head. CLBlast Hgemm at M=1 was 3-7 ms/call dominated by dispatch overhead. New kernel correctness validated via `NNOPT_DEBUG_LAYERS=1` LAYER_CHECK on `model_text_model_norm` + `lm_head` — all passes "OK". Total inference 600s → 55s. |
| 5 | fuse gate_proj + up_proj + SwiGLU into one GEMV-style kernel for M=1 (`src/utils.cpp`, `src/ops/llama_mlp.cpp`) | **2.4550** | +11% | +2115% | ✓ | One workgroup per output row reads `x[K]` once, accumulates both `acc_gate` and `acc_up`, then writes `silu(acc_gate) * acc_up`. Eliminates 1 GEMV dispatch + 1 swiglu dispatch + the entire intermediate `up` buffer per layer (30 layers × 64 tokens = 1920 saved ops per run). lm_head LAYER_CHECK matches reference. |
| 6 | fuse Q/K/V projections into one GEMV-style kernel for M=1 (`src/utils.cpp`, `src/ops/llama_sdpa_attention.cpp`) | **3.1097** | +27% | +2707% | ✓ | One kernel dispatch covers all (N_q + 2·N_kv) = 960 output rows. Each workgroup picks which of `W_q`/`W_k`/`W_v` to read and which output buffer to write based on `group_id`. Eliminates 2 GEMV dispatches per layer (30 layers × 64 tokens = 1920 fewer kernel launches per run). lm_head LAYER_CHECK matches. |
| 7a | **(reverted)** fused single-kernel decode attention (scores+softmax+attn_out, online softmax) (`src/utils.cpp` `decode_attn_fp16`) | 3.00 (regress) | -4% | — | ✓ | Kernel was correct (LAYER_CHECK passed) but underutilizes Adreno 620's single CU at only 9 workgroups (one per query head). The 3-kernel pipeline's 8100-workitem `gqa_attn_scores` actually wins on occupancy. Code retained in utils.cpp/.h for future split-K experiments but call site reverted. |
| 7 | on-device greedy argmax + skip 49280-fp16 D2H per decode step (`src/utils.cpp`, `src/ops/backbone.cpp`) | **3.1413** | +1.4% | +2735% | ✓ | New `argmax_fp16` kernel (single workgroup, 64-lane tree reduce) writes one int32; host reads only 4 bytes instead of ~98 KB. Synthesizes a one-hot logits row so the host sampler picks the same id. Gated by `seq_len==1 && !NNOPT_DEBUG_LAYERS=1`. Generated text remains "The image depicts a cat sitting on a surface that appears to be a wooden object, possibly a table…" |
| 8 | parallel-row softmax: workgroup-per-row with 64-lane tree-reduce for max+sum (`kernels/attention.cl`, `src/ops/llama_sdpa_attention.cpp`) | **3.9803** | **+27%** | **+3489%** | ✓ | Old kernel used 1 workitem per row (9 workitems total at decode!) doing 3 serial passes over seq_k≈900. Now: `gws=total_rows*64, lws=64`; per-row parallel max-reduce, parallel exp+sum, parallel normalize. Bonus: prefill also +20% (32.7→39.1 tok/s) since prefill has 7866 softmax rows. TTFT 27.0s → 22.6s. lm_head LAYER_CHECK matches. |
| 9 | fuse residual_add into post-attention RMSNorm prologue (`kernels/rms_norm.cl`, `src/ops/rms_norm.cpp`, `src/ops/llama_decoder_layer.cpp`) | 4.0029 | +0.6% | +3509% | ✓ | New kernel `rms_norm_residual_forward` reads `x` + `residual`, writes `x += residual` in place, then computes `rmsnorm(x)` in one workgroup pass. Eliminates 30 dispatches/token (one per layer). Smaller-than-expected gain: residual_add was already only 0.55ms/call so saving the dispatch landed roughly at the dispatch-overhead floor. lm_head LAYER_CHECK matches. |
| 10a | **(reverted)** fused QKV+RoPE+KV-cache-write in a single kernel (`src/utils.cpp` `gemv_m1_qkv_fused_fp16`) | 3.71 (regress) | -7% | — | ✓ | Kernel runs (LAYER_CHECK OK) but halved the workgroup count from 960→480 (each WG handled a rope pair = 2 output rows). Adreno 620's single CU prefers more, smaller WGs. Profile showed self_attn faster per call but real wall-time slower — likely due to branchy per-WG output target (Y_q vs K_cache vs V_cache) hurting wave coherence. Kernel retained in `utils.cpp` for future split tuning. |
| 11a | **(reverted)** parallel-row `gqa_attn_out` (64 lanes per output tile, reduce over seq_k) | 2.87 (regress) | -28% | — | ✓ | Decode profile improved slightly but prefill collapsed (39→31 tok/s). Prefill has 9·874·16=125,856 output tiles; making each a 64-lane workgroup turned a saturating kernel into a barrier-bound one. Needs a decode-only variant — out of scope this round. Reverted to the simple 1-workitem-per-tile kernel. |
| 12 | (gap from reverts) restore the opt-#6 fused QKV path in the `!qkv_rope_cache_done` fallback branch | **4.1035** | +2.5% vs #9 | **+3599%** | ✓ | After reverting #10a I had to put the fused-QKV decode dispatch back into the fallback path (it was inlined into the deleted #10a block). End state: same kernels as #9 but cleaner control flow; small bonus from removing branch overhead. |
| 13 | decode-only parallel `gqa_attn_out`: new `gqa_attn_out_decode` kernel uses workgroup-per-(qh,d4) with 64-lane reduce over seq_k (`kernels/attention.cl`, `src/ops/llama_sdpa_attention.cpp`) | **4.2780** | +4.3% | **+3756%** | ✓ | Decode at seq_q=1 was using only 144 workitems each looping seq_k≈900 iterations. New decode-only variant: 9·16=144 workgroups × 64 lanes = 9216 workitems, each lane processes ~14 seq_k entries then tree-reduces. Prefill keeps the original kernel (parallelizing it killed prefill in #11a). Prefill unchanged (39.7 tok/s), decode total inference 47s → 37s. lm_head LAYER_CHECK matches. |
| 14 | persistent per-layer `scores_decode` buffer (sized to `KV_CACHE_MAX_LEN`) — reused across all decode steps (`src/ops/llama_sdpa_attention.cpp`) | **4.4108** | +3.1% | **+3876%** | ✓ | The scores buffer (9·seq_k·2B ≈ 16 KB at seq_k=900) was allocated and freed per layer call (30 layers × 64 tokens = 1920 alloc/free pairs per run). Now allocated once per layer at max size (36 KB) and reused. Prefill keeps per-call alloc because its scores buffer is 13 MB per layer × 30 = 390 MB — too big to keep resident. Memory cost: 30·36 KB = 1.1 MB extra resident. |
| 15a | **(reverted)** TILE_N=2 fused gate+up+SwiGLU kernel (each WG handles 2 output rows for N=1536) | 4.33 (regress) | -1.8% | — | ✓ | Halved WG count to 768 but doubled per-WG memory pressure (4 weight rows read per WG instead of 2). Adreno 620 wasn't dispatch-bound for N=1536 — the simple per-row kernel already hits enough WGs to saturate the CU. |
| 16a | **(skipped)** fuse `attn_scores` + `softmax` into one decode kernel | — | — | — | — | Would require either cross-WG sync (impossible in one kernel) or 9-WG layout (occupancy-starves Adreno's single CU, same problem as #7a). Estimated savings only ~140 µs/call × 30 = 4 ms/token (~2%); not worth the architectural cost. |
| 17a | **(reverted)** fuse `kv_cache_write` into fused-QKV via `clCreateSubBuffer` slot writes | 3.86 (regress) | -12% | — | ✓ | Per-call `clCreateSubBuffer` on Adreno's userspace ICD turned out to cost more than the single `clEnqueueCopyBuffer` it replaced. Sub-buffer creation likely involves a kernel-driver round-trip that the bulk-copy can amortize away. |
| 18a | **(reverted)** persist Q + ctx_q + gate + logits_all across decode steps | 4.20 (flat) | 0% (in noise) | — | ✓ | Persisted all decode scratch buffers per-layer (Q 1.2 KB, ctx_q 1.2 KB, gate 3 KB, logits_all 96 KB). 6 stability runs spanned 4.12-4.42 tok/s — same band as without persistence. **Conclusion: Adreno 620's clCreateBuffer is not on the critical path.** This rules out option 1 from the previous summary. Buffer allocation is NOT what's eating decode time — it's the per-kernel queue-submit overhead (~500 µs/dispatch) that's the real floor. |

## Round 3: Qualcomm-guide-driven opts (continues from #18)

### Device Probe (Phase 0a)

```
DEVICE_PROBE name: QUALCOMM Adreno(TM)
DEVICE_PROBE version: OpenCL 2.0 Adreno(TM) 620
DEVICE_PROBE driver_version: OpenCL 2.0 QUALCOMM build #2b233bc5e1 Date: 11/15/21 Compiler E031.37.12.07
DEVICE_PROBE max_compute_units: 1
DEVICE_PROBE max_work_group_size: 1024
DEVICE_PROBE preferred_vec_width_half: 1
DEVICE_PROBE local_mem_size: 32768 bytes (32 KB)
DEVICE_PROBE global_mem_size: ~3.93 GB
DEVICE_PROBE global_mem_cache_size: 65536 bytes (64 KB L2)
DEVICE_PROBE global_mem_cacheline_size: 64 bytes
```

**Extensions present on this Adreno 620v2 (driver 11/15/21):**
- ✅ `cl_qcom_recordable_queues` — **enables opt #1 (decode dispatch-floor break)**
- ✅ `cl_qcom_dot_product8` — enables int8 path (deferred this session)
- ✅ `cl_qcom_subgroup_shuffle` — enables opt #6
- ✅ `cl_qcom_reqd_sub_group_size` — enables opt #7
- ✅ `cl_qcom_perf_hint` — enables opt #13
- ✅ `cl_qcom_priority_hint` — adjacent
- ✅ `cl_qcom_create_buffer_from_image`, `cl_qcom_accelerated_image_ops` — image-object path (#5)
- ✅ `cl_qcom_protected_context`, `cl_qcom_compressed_image` (unused here)
- ✅ `cl_khr_subgroups`, `cl_khr_fp16`
- ❌ `cl_qcom_ml_ops` — **NOT present** (item #11 infeasible — older A6x driver)
- ❌ `cl_qcom_onchip_global_memory` — **NOT present** (item #15 infeasible)

Single-CU GPU + 32 KB local mem + 64 KB L2 confirms our occupancy/tile sizing constraints. Cacheline 64 B → align loads to 16×fp16 (the half8 we already use).

## Experiments (Round 3 continues numbering)
| # | name | decode tok/s | Δ vs prev | Δ vs baseline | cos_pass | notes |
|--:|------|-------------:|----------:|--------------:|:--------:|-------|
| 19 | `-cl-fast-relaxed-math` added to every `clBuildProgram` call (`src/opencl_context.cpp` build_program) + device-extensions probe printed once at startup (Phase 0) | 4.24 | flat | — | ✓ | 3-run median 4.24 (range 4.22-4.32). Adreno guide §8.2 endorses for math-heavy kernels but we're not math-bound — net change inside noise band. Kept anyway: zero risk, correctness validated via lm_head LAYER_CHECK. Prefill 39.4 tok/s also unchanged. |
| 20 | `CL_CONTEXT_PERF_HINT_QCOM=HIGH` + `CL_CONTEXT_PRIORITY_HINT_QCOM=HIGH` on context creation (`src/opencl_context.cpp`) — Adreno guide §9.1.1/§9.1.2 | 4.20 | flat | — | ✓ | 3-run median in noise band. HIGH perf is documented as default but explicit setting keeps boost clocks engaged. Free; kept. |
| 21 | subgroup-reduce softmax: replace `__local` + barrier tree-reduce with `sub_group_reduce_max` / `sub_group_reduce_add` + `qcom_reqd_sub_group_size("full")` (`kernels/attention.cl` gqa_softmax) + `native_exp` / `native_recip` | 4.21 | flat | — | ✓ | softmax was already only ~3% of decode time. lm_head LAYER_CHECK passes (relaxed-math precision OK). Kept — simpler kernel, prefill softmax also benefits. |
| 22 | subgroup-reduce RMSNorm: drop the `__local` tree-reduce in both `rms_norm_forward` and `rms_norm_residual_forward`, use `sub_group_reduce_add` with `qcom_reqd_sub_group_size("full")` (`kernels/rms_norm.cl`) | 4.20 | flat | — | ✓ | RMSNorm cost is dispatch-overhead-floored (60 dispatches/token × 500 µs). Reduce-time savings are real but sub-µs/call. Kept — cleaner kernel. |
| 23 | **tiled-prefill `gqa_attn_scores`** with `TILED_BR=16 × TILED_BC=16` workgroups; cooperative vec4 fp16 (`vload_half4`) load of Q/K tiles into 4 KB `__local`; each WI computes one score via 16 vec4 dot products over head_dim=64. Activated only when `seq_q ≥ 16 && head_dim==64` (prefill path). (`kernels/attention.cl`, `src/ops/llama_sdpa_attention.cpp`) | 4.17 (decode flat) | **prefill +22%** | — | ✓ | **BIG TTFT WIN.** Prefill 39.4 → **47.9 tok/s** (3-run median), TTFT **22.5s → 18.5s**. Decode untouched (uses original 1-WI-per-row kernel). lm_head LAYER_CHECK matches. Tile size 16×16 chosen for 8 KB local-mem (32 KB available) — leaves room for double-buffer or larger tile if needed. Implementation: float4 compute → `convert_half4` → `vstore4` because half4 = float4 assignment is rejected. |
| 24a | **(reverted)** tiled `gqa_attn_out` with `AO_BR=64 × AO_BC=16` cooperative tile loads | 47.4 prefill (regress) | -1% | — | ✓ | 3-run median 47.4 vs 47.9 baseline — small regression. The naive attn_out (1 WI per output tile) was already L2-cache-friendly: V is read sequentially down seq_k, and each output (qh, seq_q, d) tile is small enough that re-reads stay in L2. Tiling added scheduling overhead with no L1-texture-cache benefit (buffer objects don't hit L1). Kernel code retained in attention.cl; dispatcher uses `use_tiled = false`. |

### Round 3 state after #19-24a (primary lands so far)

| metric | pre-Round-3 | now | delta |
|---|---:|---:|---:|
| decode tok/s (median of 3) | 4.20 | 4.17 | flat (in noise band) |
| prefill tok/s | 39.4 | **47.9** | **+22%** |
| TTFT (s) | 22.5 | **18.5** | **-18%** (-4.0 s) |

Decode unchanged — that's expected: items #19-#22 hit kernels that are dispatch-overhead-floored, and #23-#24 target prefill only. The actual decode lever is **opt #1 (cl_qcom_recordable_queues)** — confirmed available in the device probe — which is the next target.

| 25a | **(reverted)** subgroup-reduce in `gemv_m1_fp16`, `gemv_m1_swiglu_fp16`, `gemv_m1_qkv_fused_fp16`: replace `__local` tree reduce + barriers with `sub_group_reduce_add` + `qcom_reqd_sub_group_size("full")` (`src/utils.cpp`, all 3 singleton programs) | 3.05 (regress) | **-27%** | — | ✓ | Lm_head LAYER_CHECK passed but 3-run median collapsed to 3.05 tok/s vs 4.17 baseline. The pattern that helped softmax/rms_norm (which are reduce-dominated) HURT the GEMVs (which are vload-dominated). Hypothesis: pinning `qcom_reqd_sub_group_size("full")` constrained the driver's wave-issue scheduler, and `sub_group_reduce_add` is implemented as a built-in call that doesn't fold inline as cheaply as the open-coded tree reduce at WG=64. Reverted all 3 kernels. The `-cl-std=CL2.0 -cl-fast-relaxed-math` build options on these singletons retained — they're free even without subgroup builtins. Decode back to 4.10-4.21 (median 4.17). Confirms: pattern-matching across kernels doesn't generalize — Adreno scheduling is shape-dependent. |

### Session end-state (Round 3 primary lands)

3-run median, fp16 build, VLM workload (image + 874-token prompt, 64 new tokens):

| metric | session start | session end | delta |
|---|---:|---:|---:|
| **decode tok/s** | 4.20 | 4.17 | flat (in noise) |
| **prefill tok/s** | 39.4 | **47.9** | **+22%** |
| **TTFT (s)** | 22.5 | **18.5** | **-18%** (-4.0 s) |

**Headline this session:** tiled-prefill `gqa_attn_scores` (opt #23) cut TTFT by 4 seconds for the VLM workload. Decode unchanged — the dispatch-overhead floor still blocks gains; the only known lever past it is `cl_qcom_recordable_queues` (opt #1), confirmed available on the device but deferred (needs `cl_ext_qcom.h` signatures + queue plumbing through every op).


## Round 4: Path A — dispatch-overhead fusion (start: decode 4.17, prefill 47.9)

| # | name | decode tok/s | Δ vs prev | Δ vs Round-4 start | cos_pass | notes |
|--:|------|-------------:|----------:|--------------:|:--------:|-------|
| 26a | **(reverted)** drop `CL_QUEUE_PROFILING_ENABLE` when `NNOPT_PROFILE=0` — gate on env var (`src/opencl_context.cpp`) | 3.65 (regress) | -13% | — | ✓ | Counter-intuitive: disabling profiling REGRESSED 3-run median from 4.17 to 3.65. Suspected: Adreno driver picks a different (slower) submission path for no-profiling queues, or the event objects provide backpressure that helps throughput. Reverted with explanatory comment. |
| 27 | **Fuse post-MLP `add_residual_inplace` into the `down_proj` GEMV** for decode (`gemv_m1_residual_fp16` kernel + `op_LlamaMLP_with_residual` entry + `src/ops/llama_decoder_layer.cpp` use_fused_residual path) | **5.19** | **+24%** | **+24%** | ✓ | Saves 1 dispatch per layer per token (30/token). 3-run median 5.19 (range 5.19-5.22). Way bigger than expected from a ~15ms savings — `add_residual_inplace` was launched with `gws=n` and no `lws` (Adreno picked a slow 1-WI-per-element layout), so each call was much costlier than its arithmetic. New kernel: a copy of `gemv_m1_fp16` with an extra `residual` param; last line writes `(half)(partial[0] + (float)residual[row])`. Prefill keeps the old MLP path (CLBlast GEMM handles M>1 well). lm_head LAYER_CHECK matches reference within fp16 noise (<0.04 drift). |

| 28a | **(reverted)** retry `gemv_m1_qkv_fused_fp16` (fuses Q+K+V proj + RoPE + KV-cache-write in one kernel) on decode — replaces 5 ops per layer with 1. Code wired in `op_LlamaSdpaAttention` behind `seq_q==1 && head_dim==64` guard. | 4.62 (regress) | -11% | — | ✓ | Reverted (same WG-halving issue as opt #10a): combined kernel uses 480 WGs (one per RoPE pair = 2 output rows) vs 960 in the non-fused path. Adreno 620's single CU prefers more, smaller WGs even at the cost of 4 extra dispatches. Reverted — confirms the #10a finding still holds at higher decode tok/s. |
| 28b | **Combined `apply_rope_qk_inplace`** — one dispatch instead of two for Q+K RoPE on decode (`src/ops/llama_sdpa_attention.cpp`). Same per-WI math, halves rope dispatches per layer. | 5.16 | flat (in noise) | -0.6% | ✓ | 3-run median 5.16 vs 5.19 baseline. The two original rope dispatches were already cheap (~80µs each); saving one didn't move the needle. Kept anyway — strict dispatch reduction with no measurable cost. |

### Round 4 result: Path A landing

3-run median, fp16 build, VLM workload:

| metric | Round-4 start | Round-4 end | delta |
|---|---:|---:|---:|
| **decode tok/s** | 4.17 | **5.19** | **+24%** |
| **prefill tok/s** | 47.9 | 47.9 | flat |
| **TTFT (s)** | 18.5 | 18.5 | flat |

**Headline:** opt #27 (fuse post-MLP add_residual into down_proj GEMV) delivered the entire decode gain. Other Path-A candidates either regressed (#28a fused-QKV) or landed flat (#28b combined RoPE).

**Diagnosis why Path A topped out near 5.2 tok/s:**
- We're now ~9 kernel + 2 copy = 11 ops/layer × 30 layers = ~335 ops/token at 192ms/token = **0.57 ms/op average**.
- That's at or below Adreno 620's dispatch-overhead floor. Further dispatch fusion either requires (a) reducing WG count below 9 per kernel (occupancy-starves the single CU — see #7a, #10a, #28a) or (b) full sequence recording via `cl_qcom_recordable_queues`.
- **To hit 8 tok/s we need Path B (recordable queues).** Path A's realistic ceiling on this device is ~5.5 tok/s.


## Round 5 starting profile (NNOPT_PROFILE=1, 16 decode tokens)

Note: profile run shows decode 2.93 tok/s due to clFinish-bracketed timing overhead (~5–20ms/pair). Real-world decode at 5.19 tok/s. Use the **per-call mean_us** as the trustworthy ranking, not the headline tok/s.

**Top decode-phase per-call costs (per layer per token unless noted):**

| op | per-call mean_us | per-token (×30 layers) | rank | candidate fix |
|---|---:|---:|---:|---|
| `decode_26a_mlp_gate_up` (fused gate+up+SwiGLU GEMV) | 1253 | 37.6 ms | 1 | image2D weights (#34) — heaviest GEMV |
| `decode_23f_attn_out` (parallel attn_out_decode) | 1021 | 30.6 ms | 2 | image2D V-cache (#35) |
| `decode_23a_qkv_proj` (fused QKV GEMV) | 935 | 28.0 ms | 3 | image2D weights (#34) |
| `decode_26c_mlp_down` (fused down+residual GEMV) | 812 | 24.4 ms | 4 | image2D weights (#34) |
| `decode_23d_attn_scores` | 804 | 24.1 ms | 5 | image2D K-cache (#35) |
| `decode_23g_o_proj` GEMV | 753 | 22.6 ms | 6 | image2D weights (#34) |
| `decode_22_rmsnorm_in` | 496 | 14.9 ms | 7 | at dispatch floor — limited upside |
| `decode_25_rmsnorm_post` | 474 | 14.2 ms | 8 | at dispatch floor |
| `decode_23c_kv_cache_write` (2× copy) | 410 | 12.3 ms | 9 | dispatch floor; absorb via fused-QKV (failed #28a) |
| `decode_23b_rope` (combined Q+K) | 407 | 12.2 ms | 10 | dispatch floor |
| `decode_23e_attn_softmax` | 353 | 10.6 ms | 11 | dispatch floor |
| `decode_40_lm_head` GEMV | 13597 / token | 13.6 ms | 12 | image2D weight (#33) — single biggest matrix |

**Conclusion:** the 6 weight-bound GEMVs collectively own ~170 ms/token (out of ~192 ms total budget). The other ~22 ms is at-or-near dispatch floor. **Phase 2 (image2D weights for L1 texture cache) directly targets the actual bottleneck.** Phase 1 micro-opts on the floor-bound items are individually <1%.


## Round 5: Phase 2 — Image2D weights for L1 texture cache

| # | name | decode tok/s | Δ vs prev | Δ vs Round-5 start | cos_pass | notes |
|--:|------|-------------:|----------:|--------------:|:--------:|-------|
| 33a | **Image-backed `o_proj` weight** (decode only). New `gemv_m1_image_fp16` kernel + `get_or_create_weight_image()` cache (`src/utils.cpp`); o_proj call site routes through it (`src/ops/llama_sdpa_attention.cpp:23g`). | **8.71** | **+68%** | **+68%** | ✓ | Holy. ONE weight image-backed — decode median 5.19 → 8.71 tok/s (3-run range 8.65–8.80). The L1 texture cache effect is far bigger than estimates: 30× layers × Wo (663 KB) shouldn't fit in L1 either, but routing reads through the texture engine frees up the buffer-cache hierarchy for everything else. Prefill flat (47.9 → 47.9 tok/s — image-only on decode path). lm_head LAYER_CHECK matches reference (<0.5% drift, fp16 noise). |

| 43 | **OpenCL program binary cache** — new `nnopt_build_program_cached()` in `src/opencl_context.cpp` hashes (source + options + device + driver) with FNV-1a, persists device binary to `kernel_cache/<hex>.bin` on first build, reloads via `clCreateProgramWithBinary` on warm starts. All 12 program builds (OpenCLContext::build_program + 8 singletons in src/utils.cpp + rope kernel) routed through it. | **9.00** | +3.2% | +73% | ✓ | Direct user ask. Cold run populates cache (12 files, ~85 KB total). Warm: prefill 47.9 → 49.1 tok/s (+2.5%), TTFT 18.5 → 18.0s (-0.5s), decode 8.61 → 8.97 tok/s (+4.2%). Smaller TTFT win than expected because most kernel-build cost was lazy (first-use, inside prefill timing); now it lives in startup which is ~stale by TTFT measure. Decode bonus likely from removed compile jitter during early decode steps. |

| 34a | **Image-backed `down_proj` + `swiglu` (gate+up) GEMVs.** New `gemv_m1_image_residual_fp16` + `gemv_m1_image_swiglu_fp16` kernels in `src/utils.cpp`; `op_LlamaMLP_with_residual` routes through images, falls back to buffer GEMV if image creation fails. | **9.93** | +10.7% | +91% | ✓ | MLP image-backed: gate_up call cost 1253 → 887 us (-29%), down_proj 812 → 625 us (-23%) per profile. lm_head LAYER_CHECK matches (fp16 noise <0.5%). |
| 34b | **Image-backed `gemv_m1_qkv_image_fp16`** — 3 image weights (Wq, Wk, Wv) in one kernel; branch on row index. Routed from `op_LlamaSdpaAttention` decode path. | **10.08** | +1.5% | +94% | ✓ | qkv_proj 935 → 780 us per profile (-17%). Smaller gain — we're now in diminishing returns as the L1 cache becomes saturated across kernels. |

### Round 5 state after image-weights pass

3-run median:

| metric | Round-5 start | Round-5 now | Δ |
|---|---:|---:|---:|
| **decode tok/s** | 5.19 | **10.08** | **+94%** |
| **prefill tok/s** | 47.9 | 49.3 | +2.9% |
| **TTFT (s)** | 18.5 | 18.0 | -0.5s |

**Per-call profile after image-pass** (decode-only, 16-token profile run, lower is better):
- attn_out (V-cache read): 981 us — **next biggest, not yet image-backed (KV cache is a buffer)**
- gate_up: 887 us (was 1253)
- qkv: 780 us (was 935)
- attn_scores (K-cache read): 718 us — **not yet image-backed**
- down: 625 us (was 812)
- o_proj: 561 us (was 753)
- lm_head: 13.9 ms/token — 49280-row weight exceeds CL_DEVICE_IMAGE2D_MAX_HEIGHT, needs image2d_array

**Next biggest single levers:**
1. KV cache as image2D (attn_out + scores = 51 ms/token of bandwidth-bound reads)
2. lm_head as image2d_array
3. Recordable queues

| 35a | **Image-backed `gqa_attn_out_image` for prefill** — V cache mirrored to image2D via clEnqueueCopyBufferToImage after each kv_cache_write; new kernel reads V via `read_imageh` through L1 texture cache. Allocated `V_image` + `K_image` per layer in `ensure_kv_cache`. | flat decode | **prefill +22%, TTFT -3.2s** | — | ✓ | **The big prefill rock.** profile had shown `prefill_23f_attn_out` at 201 ms × 30 = **6 SECONDS = 35 % of TTFT**. V cache (336 KB/layer) exceeds L2 (64 KB) badly; image reads route through dedicated texture cache. Prefill **49.3 → 60.2 tok/s**, TTFT **18.0 → 14.78s**. Decode unchanged (decode uses gqa_attn_out_decode which doesn't yet have an image variant). |
| 35b | **Image-backed `gqa_attn_scores_image`** — analogous K-cache image path for non-tiled scores (decode + fallback). | flat | flat | — | ✓ | Prefill uses the tiled scores kernel (Round-3 #23) which is already L1-friendly via `__local`; image only kicks in for decode. Decode in-noise. Kept — strict dispatch + bandwidth reduction. |

### Round 5 final state (this session)

3-run median, fp16, VLM workload (image + 874 prompt tokens, 64 new tokens):

| metric | session start | session end | Δ |
|---|---:|---:|---:|
| **decode tok/s** | 5.19 | **10.22** | **+97%** |
| **prefill tok/s** | 47.9 | **60.2** | **+25.7%** |
| **TTFT (s)** | 18.5 | **14.78** | **-3.7s (-20%)** |

**Wins ordered by impact:**
1. **#33a (image-backed o_proj)** — decode +68% (single best lever; L1 texture cache vs L2 buffer reads)
2. **#35a (image-backed V cache for attn_out)** — prefill +22%, TTFT -3.2s (the elephant)
3. **#34a (image-backed swiglu + down_proj)** — decode +10.7%
4. **#43 (binary kernel cache)** — TTFT -0.5s, decode +4%, prefill +2.5%
5. **#27 (fused down_proj+residual GEMV)** — decode +24% (Round 4)
6. **#23 (tiled prefill attn_scores)** — prefill +22% (Round 3)
7. **#34b (image-backed QKV fused)** — decode +1.5%

**Theme:** Adreno 620v2's L1 texture cache was 0% utilized by buffer-backed paths. Routing weights AND large activations (V cache) through image2D unlocked 2× decode and -3.7s TTFT in one session. Both wins came from sections §6.2 / §7.1.5 of the Qualcomm OpenCL guide.


## Round 6: realistic short-prompt workload (79 tokens, 64 image placeholders)

**Workload change.** Round 5 measurements used `reference/test_input_ids.bin` (874 tokens with 832 image placeholders — the HF processor's full expansion). The on-device projector emits 64 feature rows (post-pixel-shuffle scale=4), so feeding the 832-placeholder reference into `splice_image_tokens` now errors out with `placeholder count mismatch — input_ids has 832 image_token_id positions, but projector emitted 64 feature rows` (backbone.cpp:276). The production path goes through `Tokenizer::build_vlm_prompt(image_present=true, prompt)` which emits **64 placeholders**, total **~79 prompt tokens** for `"Describe the image."`. Round 6 retunes against this workload.

**Round 6 start (3-run median, fp16, clean rebuild + REPL + prewarm support added):**

| metric | Round 5 final (874 tok) | Round 6 start (79 tok) | notes |
|---|---:|---:|---:|
| decode tok/s | 10.22 | **8.32** | shape-driven: at seq_k≈80 the parallel `gqa_attn_out_decode` 64-lane reduce is mostly idle (~75% empty); also dispatch-overhead floor is a larger fraction at short prompt |
| prefill tok/s | 60.2 | **81.1** | shorter prefill amortizes fixed per-launch costs better |
| TTFT (s) | 14.78 | 20.4 | TTFT includes whole startup (weight load 513 MB, kernel-cache lookups); 14.78 likely measured with warmer page cache |
| n_prompt | 874 | 79 | |
| n_gen | 64 | 64 | |
| peak MB | 956 | 1227 | (?) larger working set at short prompt — investigate |

**Inside the interactive REPL (with prewarm):** turn 1 decode **10.35 tok/s** at 81 prompt tok; turn 2 (follow-up "What colors…") **11.01 tok/s** at 15 prompt tok with KV reuse from turn 1. Prewarm closes most of the apparent regression vs Round 5.

**Bandwidth ceiling unchanged:** weights touched per decoded token ≈ 274 MB; 10 GB/s sustained → realistic ceiling **~36 tok/s**. Round 6 start at 8.32 = ~23% of ceiling (single-shot) / 10.35 = ~29% of ceiling (REPL). Headroom factor 3.5–4.3×.

### Profile (NNOPT_PROFILE=1, 16 decode tokens, 79-tok prompt)

Decode-only per-call mean (sorted by total impact):

| op | mean_us | per-token (×30 layers, ms) | rank | Δ vs Round-5 |
|---|---:|---:|---:|---:|
| `decode_26a_mlp_gate_up` | 1081 | 32.4 | 1 | +22% |
| `decode_23a_qkv_proj` | 1034 | 31.0 | 2 | +33% |
| `decode_23c_kv_cache_write` | 791 | 23.7 | 3 | **+93%** |
| `decode_26c_mlp_down` | 771 | 23.1 | 4 | +23% |
| `decode_22_rmsnorm_in` | 575 | 17.2 | 5 | +16% |
| `decode_25_rmsnorm_post` | 575 | 17.2 | 6 | +21% |
| `decode_23g_o_proj` | 517 | 15.5 | 7 | −8% |
| `decode_23b_rope` | 500 | 15.0 | 8 | +23% |
| `decode_23d_attn_scores` | 495 | 14.8 | 9 | **−31%** (KV scan shorter) |
| `decode_23f_attn_out` | 470 | 14.1 | 10 | **−52%** (KV scan shorter, still buffer V) |
| `decode_23e_attn_softmax` | 430 | 12.9 | 11 | +22% |
| `decode_40_lm_head` | 13343 / token | 13.3 | 12 | flat |

**Observations:**
- `attn_out` is no longer #1 (was at long-prompt). KV-bandwidth ops dropped 31–52%. Weight-bandwidth GEMVs unchanged in absolute work but **proportionally larger** — they now dominate.
- `kv_cache_write` nearly **doubled** (410 → 791 µs). Two `clEnqueueCopyBuffer` per layer have constant dispatch overhead; at short prompt the data volume is tiny so overhead is the entire cost.
- `attn_out` decode still reads **buffer V**, not the image V mirror that already exists (mirrored after every kv_cache_write for prefill). Free 30–50% on this call.
- `lm_head` unchanged (bandwidth-bound, prompt-independent). Still 13% of per-token wallclock budget — image2d_array conversion still applicable.

### Optimization queue for Round 6 (ordered by expected impact, all on the 79-tok workload)

| # | name | rationale | expected lift | risk |
|--:|---|---|---:|:---:|
| R6.1 | image-backed V for **decode** `gqa_attn_out_decode_image` | mirror already maintained; decode just doesn't read it | +5–8% | low |
| R6.2 | `lm_head` as `image2d_array` (49280×576 weight; 4 slices @ ≤16384) | 13.3 ms → 6–8 ms per token | +5–7% | low |
| R6.3 | merge `kv_cache_write` into QKV image-GEMV epilogue (write directly to K/V image at row offset) | kills 791 µs/layer | +6–9% | mid |
| R6.4 | recordable queues for decode (cl_qcom_recordable_queues; pre-record 32-token bucket) | dispatch-overhead floor → near-zero | +30–60% | high |
| R6.5 | fuse `rmsnorm_post` + `gate_up` GEMV prologue (read x once) | −575 µs/layer dispatch | +3–5% | mid |
| R6.6 | INT8 weights + `cl_qcom_dot_product8` per-row symmetric | 274 → 137 MB/tok | up to 2× | high (quality) |

### Experiments

| # | name | decode tok/s | Δ vs prev | Δ vs Round-6 start | cos_pass | notes |
|--:|------|-------------:|----------:|------------------:|:--------:|-------|
| R6.0 | Round 6 start (clean rebuild + interactive REPL + prewarm; no opt changes) | 8.32 | — | — | ✓ | 3-run median 8.14 / 8.32 / 8.33. Inside the REPL with prewarm: 10.35 turn 1 / 11.01 turn 2. |
| R6.1 | image-backed V for decode `gqa_attn_out_decode_image` (new kernel + V_image route from existing prefill mirror) | 8.34 | flat (in noise) | flat | ✓ | attn_out 470 → 487 µs per call (slight regression). At seq_k≈80–142 V fits in L2 already, texture-cache adds indirection. Kernel kept (will help when KV grows in multi-turn), gated on V_image presence. |
| R6.2 | `lm_head` as **image2d_array** (4 slices × 16384 max_h; new `gemv_m1_image_array_fp16` kernel; routes through it on decode only, `pytorch_linear` still used for prefill) | **9.30** | **+11.5%** | **+11.8%** | ✓ | 3-run median 9.16 / 9.30 / 9.56. Profile: `decode_40_lm_head` 13343 → 10628 µs (−20%, ~3 ms saved per token); also helped re-warm of `gate_up` / `qkv` per-call timings (likely L1 image-cache benefit shared across kernels). Generated text bit-identical to baseline ("The image is a screenshot of a web…"). TTFT also dropped 20.4 → 13.9 s in 2 of 3 runs (image_array creation amortized across more steady-state work). |
| R6.3 | fuse decode `kv_cache_write` 4-dispatch chain into a single `kv_cache_write_decode_fused` kernel (writes new K/V row to buffer + image in one launch using `vstore4` + `write_imageh`) | 9.10 | flat (in noise) | +9.4% | ✓ | **Profile-positive, wall-flat.** Per-call `decode_23c_kv_cache_write` 791 → 368 µs (−53%) — fusion does what's on the tin. But 6-run wall median 9.10 vs R6.2 9.30 is within noise. Pattern matches earlier `#18a persistence` finding: at decode, the 4 small async copies were already pipelined behind other kernel work, so removing them doesn't shorten the critical path. Kernel kept (clean dispatch reduction may compound with recordable queues / further fusion). Prefill keeps the original bulk-copy path. |
| R6.4 | **fuse `rmsnorm_in` into the decode QKV image GEMV** (new `gemv_m1_rmsnorm_qkv_image_fp16` kernel; each WG redundantly computes `inv_rms = rsqrt(mean(x²)+eps)` via subgroup-reduce, then GEMV with `x * inv_rms * gamma`). Eliminates the standalone rmsnorm_in dispatch (~575 µs/layer × 30 layers = 17 ms/token). New `op_LlamaSdpaAttention` args `fused_in_norm_w` + `rms_eps`; decoder layer routes decode through fused path, prefill through original. | **10.20** | **+9.7%** | **+22.6%** | ✓ | 3-run median 9.76 / 10.20 / 10.76 (tight band). Profile: `decode_22_rmsnorm_in` **eliminated entirely** (no longer appears); `decode_23a_qkv_proj` per-call 794 → 1724 µs (absorbs the rmsnorm work). Net per-layer rmsnorm+qkv: 1369 → 1724 µs/call ⇒ saved ~250 µs/layer × 30 = 7.5 ms/token, matches wall-clock delta. Output text bit-identical. **First fusion that translated profile-win directly to wall-clock** — rmsnorm_in's dispatch was on the critical path (unlike R6.3 kv_writes which were already pipelined). |
| R6.5a | **(reverted)** fuse `rmsnorm_post + residual_add + gate_up_swiglu` into one kernel (`gemv_m1_rmsnorm_residual_image_swiglu_fp16`). New MLP entry `op_LlamaMLP_with_residual_and_rmsnorm`. Goal: drop the standalone rmsnorm_post dispatch (~655 µs/layer). | 9.01 (regress) | **-11.7%** | — | ✓ | Generated text bit-identical (math correct). But `decode_26a_mlp_gate_up` per-call **1145 → 2651 µs** (+131%). Root cause: with 1536 WGs in the fused kernel, **each WG independently re-reads (attn_out + hidden_states + gamma)** — 3× input bandwidth vs the original gate_up (which read pre-computed `norm2`). The 1-WG rmsnorm_residual saved 655 µs but the redundant per-WG sum-reading added ~1500 µs. Asymmetry of fusion at decode: **per-WG-independent precomputation only pays off when the precomputed work itself is cheap or the WG count is small**. For rmsnorm_in (R6.4) it worked because gate_up's input is `hidden_states` which is just 1 read source (vs 2 + gamma in R6.5). Kernel + entry retained for a future attempt that broadcasts `inv_rms` via tiny global scratch (1 dispatch produces scale, all WGs read it). Reverted to R6.4 state. |

### Round 6 final state

3-run median, fp16, VLM workload (image + `"Describe the image."` + 64 new tokens):

| metric | R6 start | R6 final | Δ |
|---|---:|---:|---:|
| **decode tok/s single-shot** | 8.32 | **10.2** | **+22.6%** |
| **decode tok/s REPL turn 1** (with prewarm) | 10.35 | **13.30** | **+28.5%** |
| **decode tok/s REPL turn 2** (KV reuse) | 11.01 | **13.47** | **+22.3%** |
| **prefill tok/s** | 81.1 | 82.5 | flat |
| **TTFT s** | 20.4 | 14.0 | -31% (kernel-cache + image-array work amortizes earlier) |

**Wins ordered by impact (Round 6):**
1. **R6.4 (rmsnorm_in + qkv fusion)** — decode +9.7%. Killed a dispatch on the critical path; classic Qualcomm-guide pattern of folding small ops into a heavier kernel's prologue.
2. **R6.2 (lm_head image2d_array)** — decode +11.8%. Routed the only big GEMV still on CLBlast Hgemm through the image-cache path.
3. **R6.3 (kv_cache_write fusion)** — profile-positive (-53% per-call), wall-flat. Kept for future amortization (recordable queues + further fusion may compound).
4. **R6.1 (image V for decode attn_out)** — neutral at short prompt (V fits in L2); kept for longer-context multi-turn.

**Anti-pattern confirmed (R6.5):** fusing across 1 WG → N WG boundary fails when the small kernel's output is consumed by all N WGs and each must re-read the inputs. Save 1 dispatch (~600 µs), cost N× input reads (~1500 µs). Watch for this when planning future fusions.

**Bandwidth ceiling math:** 274 MB/token × 10 GB/s = 36 tok/s ceiling. R6 final REPL decode 13.5 tok/s = **37% of ceiling** (up from 23% at R6 start). Headroom remaining: 2.7×.

**Next unexplored levers (deferred this session):**
- `cl_qcom_recordable_queues` — biggest remaining single lever. ~335 dispatches/token × ~500 µs floor → if recording absorbs half, +30-60% decode.
- INT8 + `cl_qcom_dot_product8` for hot GEMVs — per-token weight read 274 → 137 MB. High quality risk.
- Broadcast-scale variant of R6.5: 1-WG kernel writes `inv_rms` to a tiny global buffer, gate_up reads it (avoids 1536× redundant reduce). May reclaim what R6.5 lost.


## Round 7: vision-tower parallel softmax + parallel scores (TTFT focus)

After Round 6's decode wins, the dominant TTFT contributor is the vision encoder, not text prefill. Round 7 attacks it directly.

**Round 7 start state (1-run, 79-tok prompt + image):**

| metric | value | notes |
|---|---:|---|
| TTFT | 13.9 s | one-shot path |
| `vision_pipe` | 19.5 s | dominates TTFT |
| `vis_attn` per call | ~1.6 s | × 12 layers = 19 s |
| `vis_softmax` per call | 1.32 s | naive 1-WI-per-row, 3 sequential T-wide passes |
| `vis_scores` per call | 850 ms | naive 1-WI-per-(b,h,tq,tk) |
| `vis_mha_out` per call | 240 ms | 1-WI-per-output, T-tk reduction |

### Experiments

| # | name | per-call delta | TTFT delta | cos_pass | notes |
|--:|---|:--|---:|:--:|---|
| R7.1 | `mha_softmax_parallel`: 1 WG/row, 64 lanes + `sub_group_reduce_max/add` for the 3 passes | softmax 1.32 s → 20 ms (**−98.5%**) | -1.5 s | ✓ | Subgroup reduce replaces the 3× linear scan with O(log64). T=1024 fits comfortably. **First Round 7 win.** Kept. |
| R7.2 | `mha_scores_par`: 1 WG/row, 64 lanes stride over tk, Q in `__local` | scores 850 → 388 ms (**−54%**) | -5.5 s | ✓ | 64 independent dot products per WG; Q reused via `__local` (128 B). K reads per lane still 16× per WG but at least small Q footprint avoids re-reads. **Second Round 7 win.** Kept. |
| R7.3 | `madvise(MADV_WILLNEED)` on weight mmap (was MADV_RANDOM) | startup `weights_load` 18 ms vs 35 ms | -0.6 s | ✓ | Async-prefetches the 489 MB weight file so first `clCreateBuffer` reads land in page cache. Tiny but free. |

**Round 7 final state (3-run median):**

| metric | R7 start | R7 final | Δ |
|---|---:|---:|---:|
| TTFT | 13.9 s | **11.3 s** | **−19%** |
| `vision_pipe` | 19.5 s | **10.3 s** | **−47%** |
| `vis_softmax` × 12 | 16 s | **0.24 s** | −98.5% |
| `vis_scores` × 12 | 10.2 s | **4.7 s** | −54% |
| `vis_mha_out` × 12 | 2.9 s | 2.4 s | flat |
| decode tok/s | 10.2 | 10.2 | flat |

Round 7 surfaced the new bottleneck: **`vis_scores` (4.7 s) + `vis_mha_out` (2.4 s) + `vis_mlp` (2.6 s) = 9.7 s of 10.3 s vision pipe.** Setup for Round 8.

## Round 8: Flash Attention attempts on the vision tower (regressions across the board)

**Hypothesis going in:** fuse `vis_scores + vis_softmax + vis_mha_out` into a single FlashAttention-style kernel (online softmax, no materialized scores buffer). Conservative target: 7.3 s trio → ~1.5–2 s ⇒ TTFT 11.3 → ~5 s.

### Experiments

| # | name | per-call | × 12 layers | vs Round-7 fallback | cos_pass | notes |
|--:|---|---:|---:|:--|:--:|---|
| R8.1 | `mha_flash_attn` v1 — Q+K_tile+V_tile in `__local` (16 KB/WG), FA_BC=64, online softmax | ~7250 ms | 87 s | **+1000%** | ✓ | Math correct (text bit-identical). Pathological wall: 16 KB `__local`/WG cap Adreno 620's single CU to 1–2 concurrent WGs out of 12,288 needed → serializes hard. |
| R8.2 | Lean variant — drop K/V tile, keep only Q (128 B) + scores_tile (256 B); K/V read direct from `__global` | 886 ms/call | 10.6 s | +44% | ✓ | Slightly better than v1 because more WGs fit concurrently, but K reads across 64 lanes are 64 different cachelines (strided). Adreno can't amortize the global K traffic without `__local` reuse. |
| R8.3 | 64-lane P·V variant — added back K/V tile + cooperative coalesced load; all 64 lanes participate in P·V across tk-stride, `__local pv_partials` tree-reduce to D4=16 outputs | 5749 ms/call | 69 s | +850% | ✓ | The 21 KB `__local`/WG and per-block `barrier(CLK_LOCAL_MEM_FENCE)` worsened concurrency further. Cooperative load + barriers serialize across the 12,288 WGs even worse than v1. |
| R8.4 | `mha_scores_coalesced` — keep scores trio, replace `mha_scores_par` only: 1 WG/row, all 64 lanes read `K[tk, lid]` (ONE cacheline / iter), `sub_group_reduce_add` per tk | scores 4959 ms/call | 59 s | **+1180%** | ✓ | Memory transactions were 16× fewer (theoretically). But the `sub_group_reduce_add` **inside the inner loop** adds T=1024 sync points per WG; the reduce latency × 1024 × 12,288 WGs dominates the cacheline savings. Adreno's wave reduce intrinsic isn't free at this scale. |
| R8.5 | `mha_out_coalesced` — 1 WG/(b,h,tq), 64 lanes each hold one output lane (lane `lid` accumulates `out[tq, lid]`), shared probs broadcast | mha_out 260 ms/call | 3.1 s | +30% | ✓ | Closer to baseline (no inner reduces) but still worse than the naive 256-WI-per-WG mha_out. Diagnosis: many tiny WIs let Adreno's wave scheduler hide V-read latency better than a small number of 64-lane WGs that each iterate T=1024 times sequentially. |

**Net: routed back to the Round-7 trio (`mha_scores_par` + `mha_softmax_parallel` + naive `mha_out`).** All Round-8 kernels retained in `kernels/vision_attention.cl` and `state().k_*` slots but disabled in dispatch — useful reference points for future architectures with more LDS or different wave-reduce cost.

### What Round 8 taught us (Adreno 620v2 specifically)

1. **`__local` ≥ 16 KB/WG kills concurrency on a single-CU GPU with 12,288 WGs.** The classical FlashAttention recipe (load K/V into shared memory) is designed for desktop GPUs with many CUs and large register files — it doesn't translate to 1 CU + 32 KB LDS without a different blocking strategy (e.g., one WG per `(b,h)` processing many tq's).
2. **`sub_group_reduce_add` in an inner loop is expensive at scale.** Useful for one-shot reductions (softmax row max/sum), but T=1024 reductions per WG × 12,288 WGs ⇒ the wave-level sync dominates ALU work.
3. **Naive "1-WI-per-output" can beat well-thought-out parallel WGs on Adreno when the WI count is large.** With 12 × 1024 × 16 = 196,608 mha_out WIs, the scheduler has enough work to overlap; with 12,288 cooperative 64-lane WGs it doesn't.
4. **Coalesced K reads ≠ faster K reads on Adreno when reads also dominate cache misses.** With K = 128 KB per head and L2 = 64 KB, every WG ends up cold-reading K regardless. The cacheline coalescing wins are lost in DRAM bandwidth limits.

**TTFT remains at ~11.3 s** going into the next session. Likely next levers:
- INT8 vision tower (attacks the same trio but via bandwidth, not algorithm); high quality risk.
- 384-px input (T 1024 → 576, T² × 0.32) — requires pos-emb interpolation + quality validation.
- image2d-aliased K and V (texture-cache for re-read across tq's) — adds complexity but unlike Flash doesn't change WG structure.
- Vision-encoder caching: per-image one-shot run, reused across multi-turn (REPL already covers this for turns 2+).

## Mode-split measurement (2026-06-03) — Fast/384 vs Quality/512

The headline TTFT numbers in earlier rounds were measured under shifting
conditions (Round 5 used the 874-token reference input; Round 6+ used 79 tokens;
the default image size also moved to 384). That made the single published TTFT
ambiguous. This is a clean A/B of the two `--image-size` modes on the SAME
release build, same device, same workload.

Device: Motorola Razr 2020 (Adreno 620v2 / Snapdragon 765G).
Build: `NNOPT_DTYPE=fp16 ./scripts/build.sh --release`, run with
`NNOPT_DEBUG_LAYERS=0`. Workload: `"Describe this image."` + `fixtures/sample.jpg`,
64 decoded tokens, single-shot (no REPL/prewarm). 3-run warm median.

| mode | image tokens | n_prompt | prefill tok/s | decode tok/s | TTFT s | peak MB |
|---|---:|---:|---:|---:|---:|---:|
| Fast (`--image-size 384`, default) | 36 | 51 | 57.0 | 10.9 | **6.0** | 721 |
| Quality (`--image-size 512`)       | 64 | 79 | 82.2 | 11.2 | 13.2 | 733 |

### Tooling note: `weight_384`

Fast/384 needs a 24×24-interpolated copy of the SigLIP position embedding stored
as a separate tensor `model.vision_model.embeddings.position_embedding.weight_384`
(vision_pipeline.cpp picks `.weight` vs `.weight_384` by `runtime_image_size()`).
`scripts/rebake_pos_embed_384.py` now **appends** that tensor (bicubic resize,
padded to the [1024,768] slot) instead of overwriting `.weight` in place, so both
the 512 and 384 tables coexist. Run once after fetching/converting weights:
`python3 scripts/rebake_pos_embed_384.py` (idempotent; `--restore` to undo).

