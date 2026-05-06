# SmolLM2-135M-Instruct — Android OpenCL Optimization Log

**Model:** HuggingFaceTB/SmolLM2-135M-Instruct (135M params, LLaMA-style, GQA)  
**Device:** Motorola Razr 2020, Adreno 618, Android OpenCL  
**Precision:** fp16 (weights + activations)  
**Architecture:** 30 layers, hidden=576, heads=9 (Q) / 3 (KV), MLP intermediate=1536, vocab=49152

---

## Benchmark Protocol

```bash
# Build release binary (fp16)
NNOPT_DTYPE=fp16 ./scripts/build.sh --release --clean

# Deploy to device
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# Benchmark run (greedy, deterministic, 32 generated tokens)
NNOPT_DTYPE=fp16 NNOPT_DEBUG_LAYERS=0 ./scripts/run_android.sh \
    "Once upon a time" 32 --temperature 0 --seed 42 \
    --token-ids reference/test_input_ids.bin

# Profile run (per-kernel GPU time table on stderr)
NNOPT_DTYPE=fp16 NNOPT_PROFILE=1 NNOPT_DEBUG_LAYERS=0 ./scripts/run_android.sh \
    "Once upon a time" 32 --temperature 0 --seed 42 \
    --token-ids reference/test_input_ids.bin
```

**Ritual:** 3 consecutive runs, take median. Lock token IDs at Step 0. Document any drift event when reduction order changes.

---

## Hardware Ceiling (Roofline)

**Weight footprint per decode token:**
- 30 layers × attention (q+k+v+o): 30 × 1.73 MB = 51.9 MB
- 30 layers × MLP (gate+up+down): 30 × 5.31 MB = 159.3 MB
- lm_head (tied embed, 49152×576): 56.6 MB
- KV cache (grows with context, small at start): ~3–10 MB
- **Total: ~275 MB / token** (dominated by weight reads, memory-bandwidth-bound)

| Bandwidth | Source | Ceiling tok/s |
|---|---|---|
| 17.0 GB/s | theoretical DRAM peak | 61.8 |
| 14.0 GB/s | practical streaming (buffer) | 50.9 |
| ~10.6 GB/s | practical (image/texture cache) | **38.5** ← realistic target |

**Efficiency target:** reach ≥25 tok/s = 65% of image-cache ceiling (Qwen2.5-0.5B reached 78%).

---

## Optimization Plan (ranked by predicted impact)

| # | Lever | What changes | Predicted Δ | Risk | Status |
|---|-------|--------------|-------------|------|--------|
| 0 | Baseline | Release build, release mode, no debug sync | — | — | Step 0 |
| 1 | Wire `forward_decode()` + GEMV path | Dispatch decode-path kernels instead of full prefill on single tokens | **3–8×** | Low | pending |
| 2 | Custom `gemv_m1` (all M=1 sites) | Replace CLBlast HGemm(M=1) with texture-cached vec4 kernel at all 7 proj sites + lm_head | **3–5× stacked** | Low | pending |
| 3 | Coalesced wave-stride GEMV | 3× cache-line efficiency on Adreno (inner-loop unroll, coalesced reads) | **1.30–1.40×** | Low | pending |
| 4 | Persistent KV cache buffers | Avoid per-token re-allocation in k_cache/v_cache writes | **1.10–1.20×** | Low | pending |
| 5 | Fused RMSNorm + gate_proj | Fuse norm into downstream GEMV (saves 1 memory round-trip per layer) | **1.05–1.15×** | Med | pending — NOTE: regressed on SmolLM2 H=576 per memory; measure carefully |
| 6 | Wavefront-shuffle reductions | Use sub-group shuffle in softmax/rmsnorm reduce instead of shared-mem tree | **1.05–1.10×** | Low | pending |
| 7 | Fused softmax+scale+mask | Single kernel for QK scale + causal mask + softmax | **1.05–1.10×** | Med | pending |
| 8 | Fused SwiGLU (gate+up+activation) | Single kernel: `out = silu(gate) * up` without 2 separate GEMMs reading back to GPU | **1.05–1.10×** | Low | pending |
| 9 | RoPE table prebake | Hoist cos/sin table generation to initialize() (currently per-forward) | **1.02–1.05×** | Low | pending |
| 10 | LM-head sparse top-k | Skip full vocab GEMV; compute only top-K logits under greedy | **1.05–1.20×** | High | pending — greedy only |
| 11 | Q4_0 weight quantization | 4-bit weights: 2× memory bandwidth, small accuracy drop | **1.30–1.50×** | High | defer until BW-bound after #2 |
| 12 | `cl_qcom_recordable_queues` | Amortize per-decode launch overhead via queue recording | **1.05–1.30×** | High | pending — BLOCKED on Adreno 618 per prior Qwen work; keep behind `NNOPT_RECORD=1` |

---

## Step 0 — Baseline (release, fp16)

> **Status:** MEASURED — 2026-05-04

**Porting run reference (debug build, cycle 6, 2026-05-04):**
```
BENCHMARK prefill_tokens_per_sec: 1.6071
BENCHMARK decode_tokens_per_sec: 0.5009    ← debug build + clFinish in pytorch_linear
BENCHMARK time_to_first_token_sec: 3.1663
BENCHMARK peak_cpu_memory_mb: 831.58
```

Release build removes `NNOPT_DEBUG_SYNC` → no `clFinish` per GEMM call. Confirmed ~3× improvement over debug baseline.

| Run | decode tok/s | total tok/s | TTFT s | tokens deterministic |
|-----|-------------|-------------|--------|---------------------|
| 1   | 1.4809      | 1.4175      | 1.6699 | YES (reference)     |
| 2   | 1.5087      | 1.4439      | 1.6151 | YES                 |
| 3   | 1.4633      | 1.4044      | 1.6012 | YES                 |
| **median** | **1.4809** | **1.4175** | **1.6151** | **—** |

Generated text (runs 1–3 identical): `2 0th and 21st floors, and the student was at the 22nd floor. The student was 10 feet from the teacher`

**Step 0 per-kernel profile (NNOPT_PROFILE=1):**
```
=========== GPU per-kernel profile ===========
Total recorded GPU time: 175.477 ms across 8 distinct kernels
kernel                                 count    total_ms    avg_us    max_us   % tot
------------------------------------------------------------------------------------
gqa_attn_scores                          960      39.699      41.4      67.8   22.6%
gqa_softmax                              960      39.479      41.1      70.9   22.5%
rmsnorm_forward                         1952      30.982      15.9      21.0   17.7%
gqa_attn_out                             960      29.688      30.9      52.0   16.9%
embedding_forward                         32      11.670     364.7     378.1    6.7%
element_add                             1920       9.991       5.2       9.0    5.7%
rope_apply_qk                            960       7.894       8.2      13.1    4.5%
silu_mul                                 960       6.074       6.3      13.8    3.5%
------------------------------------------------------------------------------------
```

**Critical observation:** Total profiled GPU time = **175 ms** but total inference = **22,200 ms**.
CLBlast GEMM (not captured by profiler) is consuming **~22,000 ms = 99.2% of all time**.
The 7 projection GEMMs per layer × 30 layers + lm_head = 211 CLBlast calls per decode step × 32 steps = ~6,752 CLBlast calls ≈ **3.2 ms each** on average — dominated by M=1 inefficiency (single-row matrix × large weight matrix, CLBlast treats as GEMM).

Attention kernels (scores + softmax + out) together = 109 ms = 0.5% of total. They are not the bottleneck.

**Efficiency vs ceiling:** 1.48 / 38.5 = **3.8%** of 38.5 tok/s image-cache ceiling

---

## Step 1 — Wire `forward_decode()` + fused GEMV kernels

> **Status:** MEASURED — 2026-05-04

**What changed:** Implemented `Model::forward_decode()` using 5 custom GEMV kernels from `block_fused.cl`:
- `fused_qkv_gemv_m1`: Q/K/V projections in one cooperative kernel (replaces 3 CLBlast HGemm)
- `fused_rope_kvwrite_m1`: RoPE rotation + KV cache write in one kernel
- `fused_oproj_residual_m1`: O projection + residual add in one kernel
- `fused_gate_up_silu_m1` (new): silu(gate)×up in one kernel (replaces 2 CLBlast HGemm + silu_mul)
- `fused_down_residual_m1`: down projection + residual add in one kernel
- `fused_lm_head_gemv_m1`: lm_head GEMV (replaces CLBlast HGemm at M=1)

Also added decode dispatch in `Model::forward()`: `if (seq_len == 1 && start_pos > 0) return forward_decode(...)`.

**Drift event:** Token IDs diverge at position 29 vs Step 0. Same tokens through position 28; positions 29–32 differ. Cause: fp16 reduction order change in GEMV vs CLBlast GEMM (different accumulator layout → different rounding). New token sequence from Step 1 onwards:
`"2 0th and 21st floors, and the student was at the 22nd floor. The student was 10 years old, and"`

| Run | decode tok/s | total tok/s | TTFT s | tokens deterministic |
|-----|-------------|-------------|--------|---------------------|
| 1   | 10.3243     | 6.8724      | 1.6543 | YES (new reference) |
| 2   | 10.9084     | 7.1518      | 1.6331 | YES                 |
| 3   | 10.3977     | 6.9470      | 1.6255 | YES                 |
| **median** | **10.3977** | **6.9470** | **1.6331** | **—** |

**Δ vs Step 0:** +**7.02×** decode (1.48 → 10.40 tok/s) | **Δ vs baseline:** +**7.02×**

**Step 1 per-kernel profile (NNOPT_PROFILE=1):**
```
=========== GPU per-kernel profile ===========
Total recorded GPU time: 2072.48 ms across 14 distinct kernels
kernel                                 count    total_ms    avg_us    max_us   % tot
------------------------------------------------------------------------------------
fused_gate_up_silu_m1                    930     633.879     681.6     739.1   30.6%
fused_down_residual_m1                   930     423.229     455.1     493.1   20.4%
fused_lm_head_gemv_m1                     31     422.504   13629.2   13644.0   20.4%
fused_qkv_gemv_m1                        930     263.555     283.4     314.9   12.7%
fused_oproj_residual_m1                  930     164.185     176.5     203.8    7.9%
gqa_attn_scores                          960      40.703      42.4      68.9    2.0%
gqa_softmax                              960      39.354      41.0      71.2    1.9%
rmsnorm_forward                         1952      31.625      16.2      21.0    1.5%
gqa_attn_out                             960      29.481      30.7      49.9    1.4%
embedding_forward                         32      11.663     364.5     374.0    0.6%
fused_rope_kvwrite_m1                    930      11.130      12.0      16.9    0.5%
element_add                               60       0.432       7.2      10.0    0.0%
silu_mul                                  30       0.384      12.8      14.1    0.0%
rope_apply_qk                             30       0.352      11.7      13.1    0.0%
------------------------------------------------------------------------------------
```

**Critical observation:** Total profiled GPU time = **2072 ms** but total inference = **4600 ms**.
Kernel launch overhead ≈ **2528 ms = 55% of total**. ~9,393 decode kernel dispatches × ~269µs per dispatch.
GPU execution efficiency on GEMV kernels: 35–47% of memory bandwidth ceiling (non-coalesced access pattern).

Top two root causes to fix:
1. **Non-coalesced memory access** in all 5 GEMV kernels → fix with wave-stride pattern (each group of 64 threads reads 64 consecutive elements together)
2. **Kernel launch overhead** → future: reduce dispatch count via further fusion; long-term: `cl_qcom_recordable_queues`

**Efficiency vs ceiling:** 10.40 / 38.5 = **27.0%** of 38.5 tok/s image-cache ceiling (vs 3.8% at Step 0)

---

## Step 2 — Wave-stride coalesced GEMV

> **Status:** MEASURED — 2026-05-04

**What changed:** Rewrote all 5 GEMV kernels in `block_fused.cl` from "chunked" (thread t reads block [t*(H/64)..(t+1)*(H/64)-1]) to wave-stride (thread t reads t, t+64, t+128, ...) so the 64 threads in a wavefront read 64 CONSECUTIVE weight elements per cycle — matching Adreno's cache-line width. Also applied vec4 wave-stride to `fused_down_residual_m1` (N=1536=6×256): `for j=tid*4; j<N; j+=WG_SIZE*4` using `vload_half4` for 4× throughput.

**Result:** Wave-stride on H=576 scalar kernels showed negligible improvement (H=576 = only 9 loop iterations per thread — loop overhead + launch latency dominate, not memory bandwidth). Vec4 wave-stride on `fused_down_residual_m1` (N=1536 = 24 scalar / 6 vec4 iters): **-57%** on that kernel (423ms → 180ms). Overall modest gain.

| Run | decode tok/s | total tok/s | TTFT s | tokens deterministic |
|-----|-------------|-------------|--------|---------------------|
| 1   | 10.7149     | 7.1048      | 1.5962 | YES                 |
| 2   | 10.7575     | 7.1326      | 1.6353 | YES                 |
| 3   | 10.9256     | 7.2330      | 1.6887 | YES                 |
| **median** | **10.7575** | **7.1326** | **1.6353** | **—** |

**Δ vs Step 1:** +**3.5%** (10.40 → 10.76 tok/s) | **Δ vs Step 0:** +**7.27×**

**Step 2 per-kernel profile (NNOPT_PROFILE=1):**
```
=========== GPU per-kernel profile ===========
Total recorded GPU time: 1829.07 ms across 14 distinct kernels
kernel                                 count    total_ms    avg_us    max_us   % tot
------------------------------------------------------------------------------------
fused_gate_up_silu_m1                    930     634.217     681.9     741.2   34.7%
fused_lm_head_gemv_m1                     31     423.108   13648.6   13679.0   23.1%
fused_down_residual_m1                   930     180.031     193.6     220.5    9.8%
fused_qkv_gemv_m1                        930     264.912     284.8     316.0   14.5%
fused_oproj_residual_m1                  930     164.431     176.8     204.0    9.0%
gqa_attn_scores                          960      40.514      42.2      69.1    2.2%
gqa_softmax                              960      39.128      40.8      70.8    2.1%
rmsnorm_forward                         1952      30.991      15.9      21.2    1.7%
gqa_attn_out                             960      29.634      30.9      50.1    1.6%
embedding_forward                         32      11.643     363.8     373.0    0.6%
fused_rope_kvwrite_m1                    930      11.124      12.0      16.8    0.6%
element_add                               60       0.430       7.2       9.9    0.0%
silu_mul                                  30       0.384      12.8      14.0    0.0%
rope_apply_qk                             30       0.352      11.7      13.0    0.0%
------------------------------------------------------------------------------------
```

**Critical observation:** Total profiled GPU time = **1829 ms** but total inference = **4489 ms**.
Kernel launch overhead ≈ **2660 ms = 59% of total**. ~9,393 decode kernel dispatches × ~283µs per dispatch.
Buffer allocation overhead: ~244 `clCreateBuffer`/`clReleaseMemObject` pairs per decode step × 31 steps = ~7,564 allocation pairs. Estimated ~2.5s.

Root causes:
1. **`fused_gate_up_silu_m1`** still 635ms (34.7%) — dual-accumulator scalar, INTER=1536. Wave-stride didn't help at H=576 input dimension; vec4 would require restructuring dual-accum loop.
2. **`fused_lm_head_gemv_m1`** at 423ms (23.1%) — VOCAB=49152=768×64; each WG does H=576=9 scalar iters. Small iter count means launch overhead dominates per-token.
3. **Buffer allocation overhead** — every `forward_decode()` call allocates and frees `q[Q_DIM]`, `k[KV_DIM]`, `v[KV_DIM]`, `scores[QH×seq_k]`, `attn_out[Q_DIM]`, `act[INTER]`, `hidden_dec`, `final_hidden`, `logits`. Eliminating this is the highest-impact next step.

**Efficiency vs ceiling:** 10.76 / 38.5 = **27.9%** of 38.5 tok/s image-cache ceiling

---

## Step 3 — Persistent decode buffers + fused decode attention

> **Status:** MEASURED — 2026-05-04

**What changed:** Two improvements bundled together:

1. **Persistent decode buffers** — eliminated all 244 `clCreateBuffer`/`clReleaseMemObject` pairs per decode step:
   - `Attention`: pre-allocated `decode_q_buf_[Q_DIM]`, `decode_k_buf_[KV_DIM]`, `decode_v_buf_[KV_DIM]`, `decode_attn_out_buf_[Q_DIM]` — reused every step.
   - `Mlp`: pre-allocated `decode_act_buf_[INTER]` — reused every step.
   - `LayerNorm`: added `forward_decode()` writing into persistent `decode_out_buf_[H]`, returning a non-owned pointer — caller must NOT release. Eliminates 61 LayerNorm allocations per step.

2. **`fused_decode_attn_m1` kernel** — replaced 3 separate decode attention dispatches (`gqa_attn_scores` + `gqa_softmax` + `gqa_attn_out`) with one fused kernel. Each WG handles one Q head; scores live in dynamic local memory `[seq_k + 64]` floats. Saves 2 × 30 = **60 dispatches per decode step** (303 → 243) = **19.8% fewer kernel launches per step**.

**Drift event:** `fused_decode_attn_m1` computes softmax via a tree-reduce over local scores with a different floating-point accumulation order vs the previous separate kernels. First divergence at token 9: `"grade"` vs `"floors"` from Step 1. **New reference token sequence from Step 3 onwards:**
`"2 0th and 21st grade level, and the student was at the 20th and ..."`

| Run | decode tok/s | total tok/s | TTFT s | tokens deterministic |
|-----|-------------|-------------|--------|---------------------|
| 1   | 14.9429     | 8.6538      | 1.6239 | YES (new reference) |
| 2   | 15.6019     | 8.7167      | 1.6848 | YES                 |
| 3   | 15.4814     | 8.6604      | 1.6931 | YES                 |
| **median** | **15.4814** | **8.6604** | **1.6848** | **—** |

**Δ vs Step 2:** +**43.9%** (10.76 → 15.48 tok/s) | **Δ vs Step 0:** +**10.46×**

**Per-kernel profile:** not captured this step (will capture at Step 4).

**Analysis:**
- Dispatch count: 303 → 243 per decode step (-60 dispatches = -19.8%). With overhead ~275µs/dispatch: 60 × 275µs × 31 steps = **511ms saved**.
- Buffer allocation/release: 244 → ~3 allocations per step eliminated. Likely adds 200–500ms on Adreno from driver synchronization.
- Combined: ~700–1000ms saved on 4.5s total → observed ~790ms saved (4.49s → 3.70s total). Matches.
- Remaining overhead: ~9665 - (31×60) = 7805 dispatches × ~275µs = **2148ms** still from kernel launch latency.

**Efficiency vs ceiling:** 15.48 / 38.5 = **40.2%** of 38.5 tok/s image-cache ceiling

---

## Steps (to be filled as optimizations land)

*(Each step: narrative + 3-run table + Δ vs Step N-1 + Δ vs Step 0 + per-kernel profile every 2-3 steps)*
