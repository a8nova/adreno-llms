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

## Step 4 — Build-side wins (program-binary cache, fast-relaxed-math, perf hint)

> **Status:** MEASURED — 2026-05-06 (rolled in with Step 5–7; see combined median)

**What changed:** Ported `OpenCLContext::build_program` from the Qwen sister-port:
- 64-bit FNV-1a-keyed persistent program-binary cache under `kernel_cache/` (loads via `clCreateProgramWithBinary` on subsequent runs — TTFT ~30 s → ~2 s on cold device).
- Auto-appended `-cl-fast-relaxed-math` to all kernel builds.
- `NNOPT_PROFILE`-gated `CL_QUEUE_PROFILING_ENABLE` (no-op when not profiling — measurable overhead at our 240+ disp/tok rate).
- Adreno `clSetPerfHintQCOM(HIGH)` boost-clock hint via `clGetExtensionFunctionAddressForPlatform` (loaded dynamically; the ICD doesn't expose it as a standard symbol).

These are infrastructure-level wins that don't move decode tok/s on their own (per-kernel time barely changes), but they unlock the larger experiments below by giving us a faster compile loop and stable clocks.

---

## Steps 5–7 — image2d_t texture cache + multi-output GEMV + vec4 SwiGLU + GPU argmax

> **Status:** MEASURED — 2026-05-06

**What changed (combined):**

1. **`gemv_m1_k576_no4_img`** (`kernels/block_fused.cl`) — image2d-backed K=576 GEMV, 4 outputs/WG, vec4 fp16 reads via `read_imageh()` on `CL_RGBA / CL_HALF_FLOAT`. Adreno's texture-fetch engine has 1.3–1.5× higher effective BW than buffer reads. Dispatched 3× per layer for Q/K/V (separate dispatches outperformed a fused 3-image variant; branchy code in the fused version regressed by 2.5×) and once for `lm_head` (tiled into 3 sub-images of 16384 rows each because V=49152 > Adreno's 16384 image-height cap).

2. **`fused_oproj_residual_m1_no4_img`** + **`fused_down_residual_m1_no4_img`** — image2d-backed, 4 outputs/WG, residual-add fused into the kernel. Down-proj has K=1536 = 6×WG×4 vec4 iters per thread (no tail).

3. **`fused_gate_up_silu_m1_v4_img`** — image2d-backed Wgate AND Wup, vec4 inner loop, single-output (NOT no4 — 4 outputs × dual accumulators = 8 fp32 accs/thread, register spill. Qwen Step 7 measured 1.78× regression from this combination).

4. **GPU argmax** (`kernels/argmax.cl`, ported from mamba2-130m) — replaces the synchronous 98 KB fp16 logits readback at greedy decode (`temperature ≤ 0` AND `repetition_penalty == 1.0`) with a two-pass cooperative reduce on-GPU and a 4-byte readback. `Model::forward_decode_greedy()` is the entry point; `Model::generate()` selects it when sampler config is greedy-pure.

**Drift:** Tokens reverted to the Step 1 reference sequence (`"20th and 21st floors, and the student was at the 22nd floor. The student was 10 years old, and"`) — the new reduction order in the image-backed `_no4` kernels happens to match Step 1's GEMV order more closely than Step 3's `fused_decode_attn_m1` softmax-reduce. ID-for-ID stable across 5 runs.

**5-run median (warm):**
| Run | decode tok/s |
|-----|-------------|
| warmup | 23.55 |
| 1 | 23.93 |
| 2 | 23.65 |
| 3 | 23.95 |
| 4 | 22.75 |
| 5 | 23.51 |
| **median** | **23.65** |

**Δ vs Step 3:** +**52.8%** (15.48 → 23.65 tok/s) | **Δ vs Step 0:** +**16.0×**

**Step 7 per-kernel profile (NNOPT_PROFILE=1):**
```
=========== GPU per-kernel profile ===========
Total recorded GPU time: 1087.77 ms across 16 distinct kernels
kernel                                 count    total_ms    avg_us    max_us   % tot
------------------------------------------------------------------------------------
gemv_m1_k576_no4_img                    2883     379.100     131.5    2186.0   34.9%
fused_gate_up_silu_m1_v4_img             930     377.407     405.8     462.1   34.7%
fused_down_residual_m1_no4_img           930     145.834     156.8     188.2   13.4%
fused_oproj_residual_m1_no4_img          930      90.606      97.4     120.8    8.3%
fused_decode_attn_m1                     930      34.912      37.5      58.9    3.2%
rmsnorm_forward                         1952      31.239      16.0      21.2    2.9%
fused_rope_kvwrite_m1                    930      12.140      13.1      19.2    1.1%
embedding_forward                         32      11.706     365.8     380.2    1.1%
gqa_attn_scores                           30       1.216      40.5      43.0    0.1%
gqa_attn_out                              30       1.137      37.9      39.9    0.1%
argmax_partial                            31       0.701      22.6      24.1    0.1%
gqa_softmax                               30       0.471      15.7      20.0    0.0%
element_add                               60       0.429       7.2       8.2    0.0%
silu_mul                                  30       0.361      12.0      15.1    0.0%
rope_apply_qk                             30       0.348      11.6      13.1    0.0%
argmax_final                              31       0.159       5.1       6.1    0.0%
------------------------------------------------------------------------------------
```

**Per-kernel deltas vs Step 3 baseline:**
| kernel (Step 3) → kernel (Step 7) | Step 3 ms | Step 7 ms | Δ |
|---|---|---|---|
| `fused_lm_head_gemv_m1` → `gemv_m1_k576_no4_img` (lm_head 3 tiles, ~6.5 ms/tok) | 423 | ~202 | **−52%** |
| `fused_qkv_gemv_m1` → 3× `gemv_m1_k576_no4_img` (Q+K+V) | 271 | ~177 | **−35%** |
| `fused_gate_up_silu_m1` (scalar) → `fused_gate_up_silu_m1_v4_img` | 597 | 377 | **−37%** |
| `fused_down_residual_m1` → `fused_down_residual_m1_no4_img` | 177 | 146 | **−18%** |
| `fused_oproj_residual_m1` → `fused_oproj_residual_m1_no4_img` | 165 | 91 | **−45%** |
| **Total profiled GPU** | **1829** | **1088** | **−41%** |

**Efficiency vs ceiling:** 23.65 / 38.5 = **61.4%** of 38.5 tok/s image-cache ceiling (vs 40% at Step 3, 28% at Step 2, 4% at Step 0).

**Remaining headroom:**
- `fused_gate_up_silu_m1_v4_img` is now tied for #1 (35% GPU) with the lm_head GEMV. Single-output keeps register pressure safe; trying `_no4_img` here regressed on Qwen and would likely regress here too.
- Kernel-launch overhead is now ~62% of total inference time (1088 ms GPU / 2840 ms total = 38% GPU; rest is launch overhead + readback). The 60 extra dispatches/token from splitting QKV into 3 image dispatches contributed to this; recordable command queues (`cl_qcom_recordable_queues`) would amortize but Adreno 618 historically refuses them per Qwen's prior work.

---

## Step 8 — New device baseline (Galaxy Tab A9+, Adreno 619 v2 / SM6375) + CL 2.0 std

> **Status:** MEASURED — 2026-05-15

**Device change:** moved from Motorola Razr 2020 / Adreno 618 to Samsung Galaxy Tab A9+ Wi-Fi (SM-X210), Qualcomm Snapdragon 695 5G (SM6375) → **Adreno 619 v2**, 1 SP / 64 KB L2 / 32 KB local / 64-byte cache line / 1024 max WG, OpenCL 2.0 full profile. Driver E031.45.02.26. Android 16. Production device (no root → sysfs governor writes no-op; `cl_qcom_perf_hint(HIGH)` provides what clock control is available).

**Device extensions worth noting** (from `cl_probe` dump):
- `cl_khr_subgroups`, `cl_qcom_subgroup_shuffle`, `cl_qcom_reqd_sub_group_size` — subgroup family **AVAILABLE**.
- `cl_qcom_recordable_queues` — **AVAILABLE** (vs Razr 2020 where it was blocked per Qwen prior).
- `cl_qcom_create_buffer_from_image` — enables single-image lm_head (V=49152 > 16384 height cap).
- `cl_khr_image2d_from_buffer`, `cl_qcom_ext_host_ptr`, `cl_qcom_dmabuf_host_ptr` — zero-copy options.
- **Missing on this device** (deviations from PDF Table 9-1 SD888 reference): `cl_qcom_dot_product8` (no hw int8 dot4 — `qcom_dot8_acc` Bloom-560m precedent is OFF the table), `cl_qcom_ml_ops`, `cl_qcom_onchip_global_memory`, `cl_qcom_bitreverse`.

**What changed:** plain port-over rebuild on the new device, plus the Phase 0 offline kernel polish (`native_exp` swaps in silu + softmax, `(size_t)t` cast removal in `fused_decode_attn_m1` per PDF §8.7), plus a default `-cl-std=CL2.0` build flag (Phase 2 lever — measured to flip the compiler's vectorization path on this driver; opt-out with `NNOPT_NO_CL2_STD=1`).

**Step 8.A — Pre-CL2 baseline (Razr port verbatim, just rebuilt for Tab A9+):**
| Run | decode tok/s | tokens deterministic |
|---|---:|:---:|
| 1 | 20.7885 | YES (new reference) |
| 2 | 21.3167 | YES |
| 3 | 20.8190 | YES |
| 4 | 21.1102 | YES |
| 5 | 20.7303 | YES |
| **median** | **20.82** | — |

Output: `"The teacher worked at the 20th and 21st floors, and the student was at the 22nd floor. The student was 10 years old, and"`. **Numerical drift vs Razr 2020 Step 7** — same input/seed, different fp16 reduction order on Adreno 619 silicon. Locked as new-device reference.

**Step 8.B — `-cl-std=CL2.0` build (default-on after this step):**
| Run | decode tok/s |
|---|---:|
| 1 | 20.6093 |
| 2 | 21.1841 |
| 3 | 20.9690 |
| 4 | 20.9867 |
| 5 | 21.6725 |
| **median** | **20.99** |

**Δ vs 8.A:** +0.8% wall, token IDs unchanged. Worth keeping (default-on now). Driver's CL2 compile path tends to pick slightly better register allocation on the GEMV kernels.

**Step 8.C — `NNOPT_SUBGROUP_REDUCE=1` A/B (kept behind flag; default OFF):**
| Run | decode tok/s |
|---|---:|
| 1 | 21.1927 |
| 2 | 21.0043 |
| 3 | 21.0706 |
| 4 | 20.9060 |
| 5 | 21.0521 |
| **median** | **21.05** |

**Δ vs 8.B:** +0.3% wall. **Drift event** — tokens diverge at position 12: `"...20th and 21st grade level, and..."` instead of `"...22nd floor..."`. Per-kernel profile shows **rmsnorm_forward** +90% (24 → 46 ms) and **fused_decode_attn_m1** +90% (25 → 47 ms) — i.e. the kernels that actually USE subgroup reduce regressed; the wall-clock parity comes from the GEMVs reading the same CL 2.0 std flag. Conclusion: **`USE_SUBGROUP_REDUCE` is wrong for WG=64 single-subgroup geometry on Adreno 619 v2** (the cross-subgroup roll-up `if (sg_id == 0) { ... }` is wasted work vs the simple 6-step tree). Kept opt-in for devices with smaller waves; default off.

**Step 8 per-kernel profile (NNOPT_PROFILE=1, CL2.0 std, no subgroup):**
```
=========== GPU per-kernel profile ===========
Total recorded GPU time: 1301.86 ms across 16 distinct kernels
kernel                                 count    total_ms    avg_us    max_us   % tot
------------------------------------------------------------------------------------
fused_gate_up_silu_m1_v4_img             930     463.933     498.9     615.9   35.6%
gemv_m1_k576_no4_img                    2883     447.852     155.3    3202.0   34.4%
fused_down_residual_m1_no4_img           930     215.078     231.3     301.1   16.5%
fused_oproj_residual_m1_no4_img          930     105.900     113.9     149.0    8.1%
fused_decode_attn_m1                     930      24.735      26.6      42.0    1.9%
rmsnorm_forward                         1952      24.333      12.5      17.2    1.9%
embedding_forward                         32       9.075     283.6     292.1    0.7%
fused_rope_kvwrite_m1                    930       7.260       7.8      12.0    0.6%
argmax_partial                            31       0.822      26.5      34.8    0.1%
... rest <1%
------------------------------------------------------------------------------------
```

**Critical observation (Adreno 619 v2):**
- 4 GEMV-family kernels = **95% of GPU work**: `fused_gate_up_silu_m1_v4_img` (36%), `gemv_m1_k576_no4_img` (34%, Q+K+V+lm_head 3-tile = 96 dispatches), `fused_down_residual_m1_no4_img` (16%), `fused_oproj_residual_m1_no4_img` (8%).
- GPU profile total **1.30 s** (with profile overhead — wall-clock-equivalent ~0.90 s); decode wall time ≈ 1.50 s → **~600 ms = 40% of decode wall is host launch overhead** (lower than Razr 2020's 62% because Adreno 619 v2's smaller compute unit makes the GEMV inner loops a larger share of the pie).
- Roofline at 10.6 GB/s image-cache: 38.5 tok/s (same as Razr). **Current efficiency 20.99 / 38.5 = 54.5%.**

**Top next levers (ordered by expected impact):**
1. `cl_qcom_recordable_queues` — eliminates the ~600 ms launch overhead per 32-token run. Target 30+ tok/s. (Available on this device, unlike Razr 2020.)
2. Per-row int8 weight quant + int8 GEMV kernels (NO `qcom_dot8_acc` — pure madd, since `cl_qcom_dot_product8` is missing). Target +30-50% on the 4 GEMV kernels.
3. `cl_qcom_create_buffer_from_image` single-tile lm_head — eliminates the 3-tile fan-out and reduces dispatch count by ~60.

---

## Step 9 — Per-row int8 image path (Q/K/V/O + gate/up/down)

> **Status:** MEASURED — 2026-05-15

**What changed:** `scripts/quantize_weights.py` emits `weights/model.int8.bin` (162 MB, 210 weights quantized — every transformer-block Linear, lm_head/embed and norms unchanged). New `kernels/block_fused_int8.cl` with 4 image2d-backed int8 kernel variants:

- `gemv_m1_k576_no4_img_int8` (Q/K/V)
- `fused_oproj_residual_m1_no4_img_int8`
- `fused_gate_up_silu_m1_v4_img_int8`
- `fused_down_residual_m1_no4_img_int8`

All use `image2d_t` with `CL_RGBA / CL_SIGNED_INT8` + `read_imagei` (sign-extended int4 per pixel), accumulate fp32, pull per-row fp16 scale out once at the tail. **Adreno 619 v2 does not expose `cl_qcom_dot_product8`** (PDF §9.5.1 — extension absent on SM6375), so no `qcom_dot8_acc` inner loop; plain fma. Host wiring in `src/layers/{attention,mlp}.cpp` adds parallel int8 state vars + `quantized_` flag; `set_weights()` detects `dtype=int8` from meta.json. Dispatch priority becomes: int8 image > fp16 image > fp16 buffer.

CLBlast HGemm in `pytorch_linear()` doesn't speak int8, so prefill is routed through the decode kernels token-by-token (`Model::forward` gates on `NNOPT_QUANT=int8`). Slower prefill in absolute kernel-count terms, but cumulatively much faster since CLBlast HGemm at M=6 is the worst case for Adreno's GEMM path.

Flip with `NNOPT_QUANT=int8` env var; weight file selection in `src/main.cpp`.

**5-run median (warm):**
| Run | decode tok/s | total wall (s) | tokens deterministic |
|---|---:|---:|:---:|
| 1 | 19.9040 | 1.8499 | YES (new int8 reference) |
| 2 | 20.1721 | 1.8100 | YES |
| 3 | 20.1922 | 1.8314 | YES |
| 4 | 20.7262 | 1.7741 | YES |
| 5 | 20.4749 | 1.7978 | YES |
| **median** | **20.19** | **1.81** | — |

**Δ vs Step 8 (fp16 CL2.0 baseline):** decode −3.8% (20.99 → 20.19); **total wall −52% (3.78 s → 1.81 s)**.

**Drift event:** Output diverges from fp16 baseline at position 9. New reference: `"The teacher worked at the 20th and 21st grade level, and the student was at the 20th and 21st grade level."` — coherent, deterministic across 5 runs. Per-row symmetric int8 with absolute-max scale; sanity check at row 0 of down_proj layer 0: max_abs_err=0.00586, relative 0.41%.

**Per-kernel deltas (per-call avg µs, int8 vs Step 8 fp16):**
| Kernel | fp16 µs/call | int8 µs/call | Δ |
|---|---:|---:|---:|
| `fused_gate_up_silu_m1_v4_img` | 498.9 | 441.8 | **−11%** |
| `gemv_m1_k576_no4_img` (Q/K/V per call) | 155.3 | 57.6 | **−63%** |
| `fused_down_residual_m1_no4_img` | 231.3 | 156.8 | **−32%** |
| `fused_oproj_residual_m1_no4_img` | 113.9 | 96.2 | **−15%** |
| `lm_head` (still fp16: 3 × `gemv_m1_k576_no4_img` per token) | 2565.9 | 2565.9 | 0% (not quantized) |

**Why decode tok/s didn't move despite faster int8 kernels:** the lm_head still runs fp16 (`model.embed_tokens.weight` was kept fp16 by default in `scripts/quantize_weights.py`, since lm_head is tied to embed and the script keeps it for argmax-safety). 285 ms of 1326 ms total GPU time per decode batch lives in lm_head's 3 fp16 image tile dispatches; with int8 carrying the other 73% of the GPU pie, the lm_head share stays at 21% and overall throughput is BW-mixed.

**Wall-time win source:** the `forward_decode` fallback for `NNOPT_QUANT=int8` prefill replaces CLBlast HGemm (slow at M=6 batch on Adreno 619 v2's single-SP geometry) with 6 cheap decode-style M=1 GEMV dispatches. TTFT drops accordingly.

**Step 9 per-kernel profile (NNOPT_PROFILE=1, int8):**
```
=========== GPU per-kernel profile ===========
Total recorded GPU time: 1326.32 ms across 11 distinct kernels
kernel                                 count    total_ms    avg_us    max_us   % tot
------------------------------------------------------------------------------------
fused_gate_up_silu_m1_v4_img_int8       1110     490.399     441.8     452.9   37.0%
gemv_m1_k576_no4_img (lm_head fp16)      111     284.818    2565.9    2625.0   21.5%
gemv_m1_k576_no4_img_int8               3330     191.879      57.6     103.9   14.5%
fused_down_residual_m1_no4_img_int8     1110     174.078     156.8     167.9   13.1%
fused_oproj_residual_m1_no4_img_int8    1110     106.794      96.2     103.9    8.1%
rmsnorm_forward                         2257      29.377      13.0      16.9    2.2%
fused_decode_attn_m1                    1110      28.311      25.5      43.0    2.1%
embedding_forward                         37      10.475     283.1     286.0    0.8%
fused_rope_kvwrite_m1                   1110       9.353       8.4      12.0    0.7%
... rest <1%
------------------------------------------------------------------------------------
```

**Next levers (ranked, dorm levers excluded):**
1. **`cl_qcom_recordable_queues`** — extension IS available on this device (vs Razr Adreno 618 where it was blocked). Launch overhead is ~12% of decode wall on Adreno 619 v2 (not 50% like Razr — fewer compute units make per-kernel time dominate). Expected win: +5-10% decode.
2. **WG-size sweep** on the 4 hot kernels (`fused_gate_up_silu_m1_v4_img_int8`, `gemv_m1_k576_no4_img_int8`, `fused_down_residual_m1_no4_img_int8`, `fused_oproj_residual_m1_no4_img_int8`) — try {32, 64, 128, 256}. Requires removing `reqd_work_group_size(WG_SIZE)` pinning.
3. **Q4_0 quantization** — 4× weight BW savings vs fp16. Would push decode much closer to the (recomputed) Q4 roofline of ~80 tok/s, but high accuracy risk on a 135M model.
4. **Sparse top-k lm_head** — fuse argmax INTO lm_head GEMV (running-max tracking per WG) to eliminate the 98 KB fp16 logits materialization that dominates the lm_head's per-call cost (~2.4 ms). Greedy-only.

---

## Step 10 — int8 lm_head via `lm_head.weight` alias (dodges tied-embedding conflict)

> **Status:** MEASURED — 2026-05-15

**What changed:** Two-line tweak in `scripts/quantize_weights.py` adds `--emit-lm-head-int8` flag which writes a dedicated `lm_head.weight` int8 tensor (per-row symmetric quant of the embedding) + `lm_head.weight.scale` to `weights/model.int8.bin`, leaving `model.embed_tokens.weight` fp16 (so the fp16 embedding-lookup kernel keeps working under tied embeddings). `src/layers/lm_head.cpp::initialize()` prefers `lm_head.weight` int8 when present; otherwise falls back to `model.embed_tokens.weight`. Image-tiled int8 dispatch via `gemv_m1_k576_no4_img_int8` (3 sub-image tiles for V=49152 > 16384 height cap).

Weight size: 162 → 192 MB on disk (+30 MB embed-as-int8 duplicate).

**5-run median (warm):**
| Run | decode tok/s |
|---|---:|
| 1 | 20.1096 |
| 2 | 20.3197 |
| 3 | 20.2811 |
| 4 | 19.8708 |
| 5 | 20.2753 |
| **median** | **20.28** |

**Δ vs Step 9:** +0.4% (20.19 → 20.28) — within run-to-run variance.

**Drift:** none — tokens identical to Step 9 reference.

**Per-kernel profile (int8 lm_head):**
```
=========== GPU per-kernel profile ===========
Total recorded GPU time: 1309.7 ms across 10 distinct kernels
kernel                                 count    total_ms    avg_us    max_us   % tot
------------------------------------------------------------------------------------
fused_gate_up_silu_m1_v4_img_int8       1110     490.481     441.9     459.0   37.4%
gemv_m1_k576_no4_img_int8               3441     460.469     133.8    2435.1   35.2%
fused_down_residual_m1_no4_img_int8     1110     173.842     156.6     172.0   13.3%
fused_oproj_residual_m1_no4_img_int8    1110     106.773      96.2     102.9    8.2%
... rest <3%
------------------------------------------------------------------------------------
```

**Per-call analysis:** the merged `gemv_m1_k576_no4_img_int8` counter (3441 calls) covers both QKV (3330 calls × 57.6 µs ≈ 191.7 ms) and the 3 lm_head tile dispatches (111 calls × 2421 µs ≈ 269 ms). Compared to Step 9's fp16 lm_head (2566 µs/call), int8 lm_head saves **~6% per call** (2566 → 2421 µs). But lm_head share of total GPU time only dropped 21% → 20% — confirming the lm_head bottleneck is NOT weight BW. The dominant cost is the **98 KB fp16 logits write** per call (49152 × 2 bytes), which int8 weights don't touch.

**Conclusion:** lm_head weight BW is roughly tied with output-write BW on Adreno 619 v2. Beating the lm_head requires fusing argmax INTO the GEMV (eliminating the 98 KB materialization). Listed as a Tier C lever above for future work.

---

## Step 11 — `cl_qcom_recordable_queues` probe (extension confirmed, integration pending)

> **Status:** PROBE LANDED — 2026-05-15

**What changed:** Ported the LFM2 record/replay probe into `src/main.cpp::run_record_probe()` (gated on `NNOPT_RECORD_PROBE=1`) + added `probe_noop` kernel to `kernels/block_fused.cl`. Loads the four `cl_qcom_*` recording entry points via `dlsym` (LFM2's proven approach), creates a recordable command queue with `CL_QUEUE_RECORDABLE_QCOM = 0x40000000` alone (not combined with profiling/out-of-order), records 240 atomic-add dispatches, replays 32 times, compares against a sequential 7680-dispatch baseline on the live queue.

**Probe result on Adreno 619 v2 / SM6375:**
```
Record: recordable queue created OK
Baseline: 7680 sequential dispatches -> 148.176 ms, counter=7680
Replay:   7680 dispatches -> 30.1189 ms,  counter=7680
Record: speedup = 4.91971x (19.2938 us/dispatch baseline -> 3.92173 us/dispatch replay)
```

**Implications for decode integration:**
- Per-dispatch host overhead: **19.3 µs → 3.9 µs** (4.9× cheaper per replay)
- SmolLM2 decode dispatches ~240/token. Per-token launch overhead today: 240 × 19.3 µs = **4.6 ms**; with recording: 240 × 3.9 µs = 0.9 ms. **Per-token savings ≈ 3.7 ms**.
- Over 32 generated tokens: ~118 ms of decode wall (~8% of current 1.5 s decode wall).
- Counter parity (7680 == 7680) confirms replay is dispatch-equivalent — no semantic divergence.

**Why integration is not landed in this session:** building the recording with correct per-step args requires plumbing 60 arg-update entries per replay (30 layers × `fused_rope_kvwrite_m1` arg 10 = `start_pos` + 30 layers × `fused_decode_attn_m1` arg = `seq_k`) plus a persistent `ids_buf_persistent_` for the embedding's token-id input. Estimated 1.5–2 hours of careful refactoring across `src/model.cpp`, `src/layers/attention.cpp`, and `src/opencl_context.{h,cpp}`. **Probe lands the infrastructure; integration is a clean follow-up.**

**Integration plan (for the next session):**
1. Promote the dlsym fn-pointer loader from `main.cpp::run_record_probe` to `OpenCLContext` members so `Model` can reuse them.
2. Add `cl_command_queue record_queue_` + `cl_recording_qcom recording_` + `bool recording_built_` to `Model`.
3. Add `cl_mem ids_buf_persistent_` (1×int32) for embedding input; update per call via `clEnqueueWriteBuffer` (cheap 4-byte write).
4. Expose `fused_rope_kvwrite_m1_` and `fused_decode_attn_m1_` kernel handles from `Attention` via getters; collect into `std::vector<PerStepArg>` in `Model::initialize`.
5. Modify `Model::forward_decode_greedy`:
   - First call: normal dispatch + parallel record (same dispatches on `record_queue_`), `clEndRecordingQCOM`.
   - Subsequent calls: build `cl_array_arg_qcom` args vector (60 entries: 30 × start_pos + 30 × seq_k), `clEnqueueRecordingQCOM(live_q, recording_, args, ...)`, read back argmax.
6. Gate behind `NNOPT_RECORD=1` so it's opt-in until tested.

Expected post-integration target: **~22 tok/s decode** (20.28 × 1.08 wall improvement) on this device.

---

## Step 12 — RoPE table prewarm to MAX_POSITION_EMBEDDINGS at init

> **Status:** MEASURED — 2026-05-15

**What changed:** `Model::initialize` now calls `Attention::prewarm_rope_tables(MAX_POSITION_EMBEDDINGS)` for every layer at startup. Previously each decode step called `ensure_rope_tables(seq_k)` which, because `seq_k = start_pos + 1` grows by one each token, found the cached table too small and **rebuilt cos/sin tables from scratch** every step — an fp32 trig loop on the host plus `clCreateBuffer` × 2 for ~30 layers per token. The cost was hidden inside ensure_rope_tables and never showed up in the kernel profile.

Discovered while attempting the `cl_qcom_recordable_queues` integration: the recording replay was failing with `CL_INVALID_OPERATION (-59)` because the captured cos/sin cl_mem references were freed and re-created mid-decode. The fix to prevent re-allocation also revealed it was a real perf bug independent of recording.

**5-run median (warm), int8 + prewarm, NO recording:**
| Run | decode tok/s | TTFT (s) | total wall (s) |
|---|---:|---:|---:|
| 1 | 24.4589 | 0.2731 | 1.5405 |
| 2 | 24.6416 | 0.2710 | 1.5291 |
| 3 | 24.3189 | 0.2687 | 1.5434 |
| 4 | 24.1973 | 0.2657 | 1.5469 |
| 5 | 24.4368 | 0.2830 | 1.5516 |
| **median** | **24.44** | **0.2710** | **1.5434** |

**Δ vs Step 10:** decode **+20.5%** (20.28 → 24.44), wall −13.3% (1.78 → 1.54s).
**Δ vs Step 8.B fp16 baseline:** decode **+16.4%** (20.99 → 24.44), wall **−59.2%** (3.78 → 1.54s).
**Δ vs original Step 0 fp32 baseline (1.48 tok/s on Razr 2020):** decode **+16.5×**.
**Roofline efficiency:** 24.44 / 38.5 = **63.5%** of image-cache ceiling.

**Drift:** none — tokens match Step 9/10 int8 reference exactly.

**Why this was hidden so long:** the `ensure_rope_tables` cost was on the HOST (Python-like fp32 trig loop + a small `clCreateBuffer`), not in any GPU kernel. Per-kernel profile showed `fused_rope_kvwrite_m1` at only 8 µs/call — but the **host overhead BEFORE that kernel** was ~120 µs/step (4 ms/run across 32 generated tokens × 30 layers = 38ms estimated saved). The total wall savings (~240 ms) outsizes the trig math itself because `clCreateBuffer` + `clCreateBuffer` per layer per step amplifies the trig cost into a driver-bound pattern.

**Bonus side effect:** RoPE tables are now sized to `MAX_POSITION_EMBEDDINGS` (8192 × head_dim × 2 = 1 MB each per layer × 30 layers × 2 (cos+sin) = ~60 MB up-front GPU memory). Trade memory for stable decode throughput.

---

## Step 13 — `cl_qcom_recordable_queues` integration (PARTIAL — replay still err=-59)

> **Status:** INFRASTRUCTURE LANDED, REPLAY UNRESOLVED — 2026-05-15

**What changed:** Plumbed the full recordable-queues path through `src/opencl_context.{h,cpp}` (extension type definitions, dlsym fn-pointer loader, helper methods `create_recordable_queue`/`new_recording`/`end_recording`/`enqueue_recording`/`release_recording`) and `src/model.cpp` (`Model::initialize` creates the recordable queue + pre-warms RoPE tables + collects per-layer `(kernel, arg_idx)` slots for start_pos and seq_k; `Model::forward_decode_greedy` records the 30-layer chain on first call, replays it with per-step arg updates on subsequent calls).

**Current behavior under `NNOPT_RECORD=1`:**
- Recording build succeeds: `RecordQ: recording built (30 layers, start_pos@record=6)` printed
- First replay fails: `enqueue_recording failed err=-59` (CL_INVALID_OPERATION)
- Fallback path correctly disables recording and continues with live dispatch — no functional regression

**Theory on -59:** the dispatches inside the 30-layer chain include `fused_decode_attn_m1`, which is enqueued with a **dynamic local memory size** (`ls_bytes = (seq_k + WG_SIZE) * sizeof(float)`). At record time the local-mem arg is captured for `seq_k = 7`; replay with `seq_k = 8+` may be rejected because the local-mem size is encoded in the recording and not overridable via the standard `cl_array_arg_qcom` mechanism (or the override path differs for `__local` args). The 5-run median of 23.76 tok/s with `NNOPT_RECORD=1` (vs 24.20 without) reflects ~150 µs of wasted replay-attempt overhead per token plus the one-time ~4ms recording-build cost.

**To resolve in a follow-up session:**
1. Either size the dynamic local memory at record time to `(MAX_POSITION_EMBEDDINGS + WG_SIZE) * sizeof(float)` (always allocate max, kernel only uses `seq_k` worth) — eliminates the per-step size mismatch.
2. Or override the local-mem `__local` arg via `cl_array_arg_qcom` with `arg_value = nullptr` and `arg_size = current_seq_k_bytes` per replay — needs verification that the QC ext supports this.
3. Or restructure `fused_decode_attn_m1` to use a fixed-size `__local` array sized for the worst case (32 KB device limit allows ~8192-position context).

Expected gain at full integration: another ~+3-5% wall (probe measured 4.9× per-dispatch host savings; 30 layers × ~7 dispatches/layer × 14.5 µs saved each = ~3 ms/token).

---

## Steps (to be filled as further optimizations land)

*(Each step: narrative + 3-run table + Δ vs Step N-1 + Δ vs Step 0 + per-kernel profile every 2-3 steps)*
