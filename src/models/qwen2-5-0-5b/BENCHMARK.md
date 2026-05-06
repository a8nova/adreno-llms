# Benchmark log — Qwen2.5-0.5B on Razr 2020 (Adreno 620, fp16)

## Benchmark protocol

**One-time setup (after any source change):**
```bash
cd <repo-root>/src/models/qwen2-5-0-5b
NNOPT_DTYPE=fp16 ./scripts/build.sh --release
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh
```
- `--release` ⇒ CMake `Release` (`-O3`), `-DNNOPT_DEBUG=` unset (debug macros stripped).
- `NNOPT_DTYPE=fp16` builds `build/fp16/Qwen2.5_0.5B_inference_fp16` and pulls `weights/model.fp16.bin`.

**Per-run command:**
```bash
NNOPT_DTYPE=fp16 NNOPT_DEBUG_LAYERS=0 ./scripts/run_android.sh \
    "Once upon a time" 32 --temperature 0 --seed 42
```
- `NNOPT_DEBUG_LAYERS=0` suppresses Sampler `max_logit` and `Generated token:` per-step stderr (each fprintf is a syscall, visible at ≥10 tok/s).
- `--temperature 0` ⇒ greedy decode (deterministic).
- `--seed 42` ⇒ pinned for reproducibility (irrelevant under greedy).
- `max_tokens=32` ⇒ generate 32 new tokens after the 4-token prompt.

**Token IDs as canonical reference.** Greedy decode is deterministic at the kernel-output level. Every step's token IDs must match the locked Step-0 reference (or, after fp16 reduction-order drift, the latest locked reference).

**Metric pulled per run:** `BENCHMARK decode_tokens_per_sec`, computed as `(n_generated - 1) / decode_s` (excludes prefill).

**Per-step ritual:** 3 consecutive runs of the per-run command, take the median decode tok/s.

## Hardware ceiling

**Snapdragon 765G / Adreno 620 / LPDDR4X-2133 dual-channel**

- **Theoretical DRAM peak: 17.0 GB/s** (marketing: 4266 MT/s × 16-bit × 2 ch / 8). Hard physics bound — no kernel can beat this.
- **Practical streaming-read ceiling: 14.0 GB/s (buffer) / 13.5 GB/s (image)** — directly measured 2026-05-03 via `--bw-probe` STREAM benchmark in this binary. 256 MB fp16 buffer read with coalesced wave-stride pattern, 5-trial best-of, every read sized to defeat L2 (64 KB). 14.0 GB/s = 82% of DRAM peak — the typical ratio for streaming-read on LPDDR4X (refresh, bus turnaround, controller scheduling overhead consumes the other 18%).
- **The 10 GB/s figure used in earlier sections of this file is inaccurate** — that was a guess inferred from prior LLM-port end-states (Mamba 4.76 GB/s, SmolLM2 2.83 GB/s), but those represent how-far-our-LLMs-got, NOT what-the-hardware-can-deliver. The right way to anchor a roofline is a dedicated STREAM kernel; we now do that.
- Adreno 620 fp16 ALU: ~2.32 TFLOPS — irrelevant; mobile LLM decode is BW-bound.

**Qwen2.5-0.5B fp16 weight footprint** (`HIDDEN=896, INTER=4864, NUM_LAYERS=24, NUM_Q_HEADS=14, NUM_KV_HEADS=2, HEAD_DIM=64, VOCAB=151936, TIE_WORD_EMBEDDINGS=true`):

| Component | Per-layer | × 24 layers | Per token (decode) |
|---|---|---|---|
| q_proj (896 → 896) | 1.53 MB | 36.7 MB | 36.7 MB |
| k_proj (896 → 128) | 0.22 MB | 5.3 MB | 5.3 MB |
| v_proj (896 → 128) | 0.22 MB | 5.3 MB | 5.3 MB |
| q/k/v bias (Qwen has bias here) | trivial | 0.06 MB | 0.06 MB |
| o_proj (896 → 896) | 1.53 MB | 36.7 MB | 36.7 MB |
| gate_proj (896 → 4864) | 8.31 MB | 199.5 MB | 199.5 MB |
| up_proj (896 → 4864) | 8.31 MB | 199.5 MB | 199.5 MB |
| down_proj (4864 → 896) | 8.31 MB | 199.5 MB | 199.5 MB |
| 2× rmsnorm weight (in/post) | trivial | 0.09 MB | 0.09 MB |
| **Per-layer subtotal** | **~28.4 MB** | **~682.5 MB** | — |
| lm_head (tied with embed_tokens, 151936 × 896) | — | — | 259.7 MB |
| final_norm | — | — | trivial |
| embedding row (1 row read at decode) | — | — | trivial |
| **Total weight bytes / token** | — | — | **~942 MB** |

KV cache reads per token (decode): per layer reads `2 × seq_k × KV_DIM × 2` bytes = `seq_k × 512` bytes; × 24 = `seq_k × 12288` bytes. At seq_k=32 → 0.4 MB total. Trivial vs weights.

**Rooflines (using the 942 MB/token weight read):**
- At 17.0 GB/s theoretical DRAM peak: 55.4 ms/token = **18.0 tok/s** (impossible to beat at fp16).
- At 14.0 GB/s measured practical streaming ceiling (buffer): 67.3 ms/token = **14.9 tok/s**.
- At 13.5 GB/s measured practical streaming ceiling (image): 69.8 ms/token = **14.3 tok/s**.

The 14-15 tok/s band is the **honest target ceiling at fp16 on this device**. A perfectly-tuned kernel doing nothing but streaming weight reads from DRAM should land near 14 tok/s. Beating that requires either (a) cache amplification (weight reused multiple times within a decode → not feasible for single-stream LLM decode where each weight is read exactly once per token) or (b) reducing the bytes-per-token (int8 quantization halves weight footprint → ceiling becomes ~28 tok/s).

**Comparable references on the same device:**
- Mamba-130M fp16 (258 MB/token, 38.7 tok/s ceiling) ended at 24.56 tok/s = **63% of ceiling** post-optimization.
- SmolLM2-135M fp16 (~270 MB/token, ~37 tok/s ceiling) ended at 10.46 tok/s = **28% of ceiling**.

Realistic landing zone for Qwen2.5-0.5B fp16 with the same engineering work: **3–5 tok/s** (28–47% of 10.6 tok/s ceiling). int8 quantization (deferred per project rule) would halve weight BW and roughly double tok/s.

## Step 0 — Release-build baseline (3-run median)

| Run | decode tok/s | total tok/s | TTFT s | prefill tok/s | tokens match |
|---|---|---|---|---|---|
| 1 | 0.6399 | 0.6121 | 3.88 | 1.045 | reference |
| 2 | 0.6402 | 0.5199 | 13.20 | 0.305 | ✓ exact |
| 3 | 0.6402 | 0.6119 | 3.93 | 1.034 | ✓ exact |
| **median** | **0.6402** | **0.6119** | **3.93** | **1.034** | ✓ |

(Run 2 had a 13s TTFT — first-time kernel JIT compile or thermal/contention spike on the Razr; the decode tok/s held steady at 0.6402 across all three runs because compile cost is amortized into TTFT.)

Decode time per token: 1000 / 0.6402 = **1562 ms/token**.
Effective decode bandwidth: 942 MB / 1562 ms = **0.603 GB/s = 6.0% of 10 GB/s ceiling**.
Memory: 2766 MB peak (weights + 2 KV caches × 24 layers × 32k positions = a lot — kv cache alone is `24*2*32768*128*2 = 384 MB` allocated upfront).

**Generated text** (`Once upon a time` + 32 tokens, greedy):
> Once upon a time, there was a man who lived in a small village．He had a big farm．He was very rich．He had a lot of money．He had

**Generated token IDs (32):**
`3837, 18532, 572, 264, 883, 879, 12163, 304, 264, 2613, 14126, 58883, 1519, 1030, 264, 2409, 8785, 58883, 1519, 572, 1602, 9080, 58883, 1519, 1030, 264, 2696, 315, 3220, 58883, 1519, 1030`

This is the **acceptance reference** — every optimization step must reproduce this token sequence ID-for-ID at greedy temp=0 seed=42 unless an explicit fp16-reduction-order drift event is documented at that step.

## Release-build fixes (FinalizePort gap)

Same FinalizePort gap as Mamba's BENCHMARK.md: scaffold-emitted per-step stderr prints aren't gated for benchmark runs.

| # | File:line | Issue | Fix |
|---|---|---|---|
| 1 | `src/sampler.cpp:11-18` | `Sampler: max_logit=...` printed unconditionally on every decode step | Wrap in `if (std::getenv("NNOPT_DEBUG_LAYERS") && env[0]!='0')` so benchmark runs (`NNOPT_DEBUG_LAYERS=0`) stay quiet |
| 2 | `src/main.cpp:266-269` | `Generated token: %d` per-token stderr spam | Same env-var gate |

## Bottleneck census per token (decode, M=1)

Counted from `src/model.cpp::forward` + `src/layers/attention.cpp::forward` + `src/layers/mlp.cpp::forward`:

**Per layer (× 24):**

| Op | Launches | Bytes/call (fp16) | Notes |
|---|---|---|---|
| `rms_norm` (input_layernorm) | 1 | 1.6 KB W | Already cooperative WG=64 vec4 ✓ |
| `pytorch_linear q_proj` (CLBlast HGemm M=1, N=896, K=896) | 1 | 1.53 MB | scalar M=1 path |
| `pytorch_linear k_proj` (CLBlast HGemm M=1, N=128, K=896) | 1 | 224 KB | scalar M=1 path |
| `pytorch_linear v_proj` (CLBlast HGemm M=1, N=128, K=896) | 1 | 224 KB | scalar M=1 path |
| `bias_add_rowmajor` × 3 (Qwen has q/k/v bias) | 3 | trivial | scalar |
| `rope_apply_qk` | 1 | trivial | scalar inner half-dim loop |
| `clEnqueueCopyBuffer` × 2 (k → k_cache, v → v_cache) | 2 | KV_DIM bytes each | host-mediated copy |
| `gqa_attn_scores` | 1 | seq_k × KV_DIM | already vec4 inner loop ✓ |
| `gqa_softmax` | 1 | trivial | scalar 3-pass |
| `gqa_attn_out` | 1 | seq_k × KV_DIM | already vec4 output ✓ |
| `pytorch_linear o_proj` (CLBlast HGemm M=1, N=896, K=896) | 1 | 1.53 MB | scalar M=1 path |
| `element_add` (attn residual) | 1+copy | full hidden | alloc + copy + clCreateKernel each call |
| `rms_norm` (post_attention_layernorm) | 1 | 1.6 KB W | cooperative ✓ |
| `pytorch_linear gate_proj` (HGemm M=1, N=4864, K=896) | 1 | 8.31 MB | **biggest per-layer GEMV** |
| `pytorch_linear up_proj` (HGemm M=1, N=4864, K=896) | 1 | 8.31 MB | **biggest per-layer GEMV** |
| `swiglu_inplace` (over 4864 elements) | 1 | 9.7 KB | 1 thread per element, scalar |
| `pytorch_linear down_proj` (HGemm M=1, N=896, K=4864) | 1 | 8.31 MB | **biggest per-layer GEMV** |
| `element_add` (mlp residual) | 1+copy | full hidden | alloc + copy + clCreateKernel each call |
| **Per-layer total** | **~20** | — | |

**Outside the layer loop:**
- 1 embedding kernel
- 1 final_norm
- 1 lm_head HGemm M=1, N=151936, K=896 — **biggest single GEMV at 259.7 MB** (Qwen's huge vocab)
- 1 logits readback (fp16 → fp32 host conversion of all 151,936 logits + std::max_element)

**Grand total per token: ~480 OpenCL launches.**

## Top-15 optimization plan (ranked by predicted impact)

Each estimate is conservative — landed actuals will be appended to each row. Order is by expected impact, not implementation difficulty. Ceiling is 10.6 tok/s; landing 3-5 tok/s is the realistic target.

| # | Lever | What changes | Predicted | Risk |
|---|---|---|---|---|
| 1 | **Custom `gemv_m1` for all big M=1 GEMVs** | Cooperative WG=64, vec4 fp16 inner, fp32 tree-reduce. Replaces CLBlast HGemm at q/k/v/o_proj, gate/up/down_proj, lm_head. Same template that landed 3.39× on Mamba Step 1 and ~5× on SmolLM2 Step 6. | **3.0–3.5×** → 1.9–2.2 tok/s | Low |
| 2 | **Coalesced wave-stride access** (gemv_m1 thread↔K mapping `off = j*WG*4 + tid*4`) | 3× cache-line efficiency on Adreno cache lines vs naïve stride-kp mapping. Mamba Step 7c: +1.39× | **1.30–1.40×** stacked → 2.5–3.1 tok/s | Low — kernel-internal swap |
| 3 | **Multi-output-per-WG GEMV** (`no4` variant: 4 outputs per WG, x loaded once into registers) | 4× fewer WGs, 4 independent fp32 accumulators per thread → register-level parallelism. Mamba P2-4: +1.21× | **1.15–1.25×** stacked → 2.9–3.9 tok/s | Low — same kernel body, different dispatch |
| 4 | **GPU-side argmax over 151936 logits** | Replace 304 KB readback + host `std::max_element` with single-WG kernel returning int32. Qwen's vocab is 3× SmolLM2's → biggest argmax win of any model. | **1.05–1.15×** stacked → 3.0–4.5 tok/s | Low |
| 5 | **Fused QKV projection** (single GEMV with stacked W [Q_DIM+2*KV_DIM=1152, H=896]) | 3 → 1 launch per layer = 48 fewer launches/token. Same template as SmolLM2 Step 6 | **1.05–1.10×** stacked → 3.2–5.0 tok/s | Medium — needs weight stack at init |
| 6 | **Fuse q/k/v bias into QKV GEMV output write** | Each WG writes (proj + bias) directly. Eliminates 3 bias_add launches per layer | **1.02–1.05×** | Low |
| 7 | **`element_add_inplace` for residuals** (already implemented in utils.cpp) | Swap 2 call sites in model.cpp from `element_add` (alloc + copy + clCreateKernel) to `element_add_inplace`. 48 fewer alloc/copy/kernel-create per token. | **1.02–1.04×** | Trivial |
| 8 | **Persistent activation buffers** in Attention/Mlp/LayerNorm | Lazy-allocate q/k/v/scores/attn_out/proj per Attention; gate/up per Mlp; output per LayerNorm. Eliminates ~80 buffer allocs/token. Mamba Step 6: +1.08×. | **1.05–1.10×** | Medium — ownership refactor |
| 9 | **Cooperative + vec4 `swiglu_inplace`** | 4864 elements/row currently 1 thread/elem scalar. Rewrite WG=64 vec4 → matches gate_proj/up_proj output shape | **1.02–1.04×** | Low |
| 10 | **Vec4 `element_add` + `bias_add_rowmajor`** | Hygiene — both are scalar over fp16 storage | **1.01–1.02×** | Low |
| 11 | **Fused RoPE + KV-cache write** | Combine rope_apply_qk + 2× clEnqueueCopyBuffer into one kernel. SmolLM2 Step 4: −2 launches/layer. | **1.01–1.03×** | Low — kernel rewrite |
| 12 | **Cooperative + vec4 `gqa_softmax`** (only if seq_k grows) | At 32-token decode seq_k ≤ 36 — currently scalar 3-pass. WG=32 cooperative once seq_k > 64. | **~0** at small seq_k → larger at long context | skip for now, document |
| 13 | **Fused `down_proj + residual_add`** (M=1 path only) | Custom kernel outputs into hidden directly. SmolLM2 A1a.4: −1 launch/layer. Couples well with the gemv_m1 path | **1.01–1.03×** | Low |
| 14 | **Fused `o_proj + residual_add`** (M=1 path only) | Same template — saves 24 launches/token | **1.01–1.03×** | Low |
| 15 | **`image1d_buffer_t` for big weights** (Adreno texture cache, gate_proj/up_proj/down_proj/o_proj/lm_head) | Texture cache typically 1.3–1.5× buffer cache for read-only fp16 weights. **The biggest known Adreno-specific BW lever.** | **1.20–1.40×** stacked → 4.5–7.0 tok/s | Medium — image1d_buffer_t views over existing buffers |

**Cumulative predicted landing zone**, applied in order, allowing 0.85× pessimism per step: ~3.5–5.5 tok/s = 33–52% of 10.6 tok/s ceiling. Matches Mamba/SmolLM2 utilization on this device.

---

## Step 1 — Custom `gemv_m1` for all M=1 GEMVs (MEASURED)

New `kernels/gemv_m1.cl` with two specialized kernels (`gemv_m1_k896`, `gemv_m1_k4864`) that hard-unroll the K loop. Each is **cooperative WG=64, vec4 fp16 inner loop, fp32 tree-reduce, fp32 accumulator** — the same template that landed 3.39× on Mamba Step 1.

**Coalesced wave-stride access pattern from the start** (Mamba Step 7c lesson, +1.39× there): each thread iteration `j` reads bytes `(j * WG_SIZE * 4 + tid * 4)` so all 64 threads of a wave-iteration touch contiguous bytes 0..511. The earlier "stride-kp" mapping (each thread reading a contiguous *chunk* of K) wastes ~3× cache-line bandwidth on Adreno.

`utils.cpp::pytorch_linear` lazily builds `gemv_m1.cl` on first call and routes M==1 / K∈{896, 4864} sites through it. Routing is transparent — no plumbing changes at any layer call site. Eligible call sites (all 8 hot M=1 projections):

| Site | M | N | K | Path used |
|---|---|---|---|---|
| q_proj | 1 | 896 | 896 | gemv_m1_k896 |
| k_proj | 1 | 128 | 896 | gemv_m1_k896 |
| v_proj | 1 | 128 | 896 | gemv_m1_k896 |
| o_proj | 1 | 896 | 896 | gemv_m1_k896 |
| gate_proj | 1 | 4864 | 896 | gemv_m1_k896 |
| up_proj | 1 | 4864 | 896 | gemv_m1_k896 |
| down_proj | 1 | 896 | 4864 | gemv_m1_k4864 |
| lm_head | 1 | 151936 | 896 | gemv_m1_k896 |
| any GEMM during prefill | M≥4 | * | * | CLBlast (predicate excludes) |

| Run | decode tok/s | total tok/s | TTFT s | tokens deterministic |
|---|---|---|---|---|
| 1 | 4.4621 | 3.318 | 1.95 | reference |
| 2 | 4.4626 | 3.342 | 1.91 | ✓ same IDs |
| 3 | 4.4627 | 3.334 | 1.92 | ✓ same IDs |
| **median** | **4.4626** | **3.334** | **1.92** | ✓ |

Decode time per token: 1000 / 4.46 = **224 ms/token** (saved 1338 ms vs Step 0).
Effective decode bandwidth: 942 MB / 224 ms = **4.20 GB/s = 42.0% of 10 GB/s ceiling** (was 6.0%).

**Δ vs Step 0 (0.6402): 6.97×.**

### Token-ID drift, post-mortem

Step 1's 32 generated tokens are **deterministic across runs** (3-run identical) but **diverge from Step 0 starting at position 17** (Step 0: 8785 → Step 1: 3753). New reference IDs:

`3837, 18532, 572, 264, 883, 879, 12163, 304, 264, 2613, 14126, 58883, 1519, 1030, 264, 2409, 3753, 323, 264, 6233, 13551, 58883, 1519, 572, 1602, 6247, 58883, 3966, 1899, 3837, 383, 3937`

Generated text: "，there was a man who lived in a small village．He had a big farm and a brave warrior．He was very tall．Today，he tried" — **coherent English, same quality class as Step 0** ("…He had a big farm．He was very rich．").

The drift is fp16 reduction-order non-associativity: CLBlast HGemm M=1 sums sequentially `acc += x[k] * W[c,k]`; gemv_m1 has 64 threads each accumulating a partial then tree-reducing. Different rounding paths → different argmax over 152K logits at decode step 17 → cascading divergence in greedy mode. Same lesson as Mamba Step 1.

**Locking these new IDs as the Step-1 reference.** Every later step must reproduce this 32-token sequence ID-for-ID.

---

## Step 2 — Multi-output-per-WG GEMV (`no4` variant) (MEASURED)

Added `gemv_m1_k896_no4` and `gemv_m1_k4864_no4`: each WG produces 4 outputs at once. Each thread holds 4 fp32 accumulators (one per output), reads x once and reuses it across all 4 weight rows, threads cooperate via 4 separate `__local` partial arrays + tree-reduce. Mamba P2-4 lesson (+1.21× there).

Routing in `utils.cpp::try_gemv_m1`: when N % 4 == 0 AND N ≥ 8, dispatch the no4 kernel with `gws = (N/4) * WG_SIZE`. All 8 hot GEMVs qualify (Q_DIM=896, KV_DIM=128, INTER=4864, VOCAB=151936 are all multiples of 4). Single-output variant kept as fall-back (currently unused on Qwen).

| Run | decode tok/s | total tok/s | TTFT s | tokens match |
|---|---|---|---|---|
| 1 | 6.8784 | 4.459 | 1.78 | reference |
| 2 | 6.7274 | 4.432 | 1.78 | ✓ same IDs |
| 3 | 6.8502 | 4.448 | 1.78 | ✓ same IDs |
| **median** | **6.8502** | **4.448** | **1.78** | ✓ |

Decode time per token: 1000 / 6.85 = **146 ms/token** (saved 78 ms vs Step 1).
Effective decode bandwidth: 942 MB / 146 ms = **6.45 GB/s = 64.5% of 10 GB/s ceiling** (was 42.0%).

**Δ vs Step 1 (4.46): 1.54×. Δ vs Step 0: 10.7×.** Token IDs ID-for-ID identical to Step-1 reference (no4 reduces in the same fp32-accumulator-then-tree-reduce order, just 4 of them per WG).

---

## Step 3 — `element_add_inplace` + GPU-side argmax + vec4 swiglu/bias_add/element_add (MEASURED, MARGINAL)

Five small kernel-internal fixes batched into one rebuild:

1. **`element_add_inplace`** for residual adds (already implemented in `utils.cpp` from scaffold; 2 model.cpp call sites switched). Saves 48 alloc + 48 clEnqueueCopyBuffer + 48 clCreateKernel per token.
2. **GPU-side argmax** over Qwen's 152K vocab (2-pass reduction: 32 partial WGs → 1 final WG → 4-byte int32 readback). Replaces the 304 KB fp16 logits readback + host fp16→fp32 conversion + std::max_element. New `forward_greedy()` path in `Model::generate` for temperature ≤ 0 / no rep penalty.
3. **Vec4 `swiglu_inplace`** (1 thread → 4 fp16/thread; 4864-element rows are divisible by 4).
4. **Vec4 `bias_add_rowmajor`** (Qwen's q/k/v have bias; 6 launches/token at decode).
5. **Vec4 `element_add`** (residual stream).

| Run | decode tok/s | total tok/s | TTFT s | tokens match |
|---|---|---|---|---|
| 1 | 6.6661 | 4.305 | 1.79 | reference |
| 2 | 6.7325 | 4.408 | 1.78 | ✓ same IDs |
| 3 | 6.7616 | 4.390 | 1.81 | ✓ same IDs |
| **median** | **6.7325** | **4.390** | **1.79** | ✓ |

**Δ vs Step 2 (6.85): 0.98× — within run noise.** All 5 changes are correct and stay in.

The marginal result confirms Step 2 was already saturating the Adreno 620's per-call BW for these GEMVs. The element_add_inplace + vec4 changes are tiny relative to the 942 MB weight read that dominates decode wall time. The GPU argmax saves ~150-200 µs per token (304 KB readback → 4 byte readback) but adds 100 µs of argmax kernel time, netting ~50-100 µs/token = ~0.5% — buried in noise. (At a higher tok/s ceiling — e.g. once int8 lands — this lever would be more visible.)

Token IDs ID-for-ID identical to Step-1 reference.

---

## Step 4 — Persistent activation buffers (Mlp + Attention) (MEASURED)

Lazy-allocated, growing-on-demand persistent buffers for activations:
- **Attention**: `buf_q_`, `buf_k_`, `buf_v_`, `buf_scores_`, `buf_attn_o_`, `buf_proj_` (6 buffers × 24 layers).
- **Mlp**: `buf_gate_`, `buf_up_`, `buf_out_` (3 buffers × 24 layers).
- The returned proj/out cl_mem is `clRetainMemObject`-ed before handoff so the caller's existing release just decrements back to our owned reference.

Eliminates **9 alloc/free per layer per token = 216 alloc/free per decode token** + 4 KV-cache copies bypassed for the persistent k/v buffers (the Attention class still copies into `k_cache_`/`v_cache_` for the cache; the per-call k/v output buffers are the ones recycled).

Mamba Step 6 lesson: +1.08× there. Each Adreno `clCreateBuffer` is ~50 µs of driver bookkeeping (no DRAM touch — metadata + L0 reservation).

| Run | decode tok/s | total tok/s | TTFT s | tokens match |
|---|---|---|---|---|
| 1 | 7.3660 | 4.627 | 1.81 | reference |
| 2 | 7.4012 | 4.674 | 1.79 | ✓ same IDs |
| 3 | 7.2906 | 4.607 | 1.82 | ✓ same IDs |
| **median** | **7.3660** | **4.627** | **1.81** | ✓ |

Decode time per token: 1000 / 7.37 = **135.7 ms/token** (saved ~10 ms vs Step 3 = matches the predicted 13 ms savings within driver-bookkeeping overhead).
Effective decode bandwidth: 942 MB / 135.7 ms = **6.94 GB/s = 69.4% of 10 GB/s ceiling**.

**Δ vs Step 3 (6.73): 1.09×. Δ vs Step 0 (0.6402): 11.51×.** Token IDs ID-for-ID identical to Step-1 reference.

---

## Step 5 — Persistent LayerNorm output + persistent lm_head logits buffer (MEASURED)

Same pattern as Step 4 extended to the two remaining clCreateBuffer-per-call sites:

- **`LayerNorm::buf_out_`** (per-instance) — saves 25 alloc/free per token (24 layer norms + 1 final). At ~50 µs each = 1.25 ms/token saved.
- **`Model::buf_logits_`** (single owner, sized for max seq_len seen) — saves 1 alloc/free per forward = 33 per generation. The buffer is 304 KB so the alloc cost is more like 80 µs.

| Run | decode tok/s | total tok/s | TTFT s | tokens match |
|---|---|---|---|---|
| 1 | 7.5544 | 4.724 | 1.79 | reference |
| 2 | 7.5466 | 4.717 | 1.79 | ✓ same IDs |
| 3 | 7.5513 | 4.758 | 1.78 | ✓ same IDs |
| **median** | **7.5513** | **4.724** | **1.79** | ✓ |

Decode time per token: 1000 / 7.55 = **132.4 ms/token** (saved 3.3 ms vs Step 4).
Effective decode bandwidth: 942 MB / 132.4 ms = **7.11 GB/s = 71.1% of 10 GB/s ceiling**.

**Δ vs Step 4 (7.37): 1.026×. Δ vs Step 0: 11.79×.** Token IDs ID-for-ID identical.

---

## Step 6 — Persistent OpenCL program-binary cache + `-cl-fast-relaxed-math` (MEASURED)

Two cleanups, one rebuild:

1. **Persistent kernel cache.** All `clBuildProgram` calls now route through `OpenCLContext::build_program` which (a) hashes `source + options + device_name + driver_version` to a 64-bit key, (b) on hit, calls `clCreateProgramWithBinary` from `kernel_cache/<key>.bin`, (c) on miss, compiles from source and writes the binary out via `clGetProgramInfo(CL_PROGRAM_BINARIES)`. The `gemv_m1.cl` route in `utils.cpp::ensure_gemv_m1` was refactored to use the same shared helper. **Cuts cold-start TTFT from ~30 s (the user-reported value) to ~2-3 s** — every kernel build after the first is a binary load.

2. **`-cl-fast-relaxed-math` is now appended to every build option string**. Lets the Adreno compiler emit fused mul-add, treat results as no-NaN/Inf, and use less-precise reciprocals/exp. We don't depend on strict IEEE semantics anywhere (greedy decode is fp16-noisy already; token IDs are locked once at the Step-1 boundary).

| Run | decode tok/s | total tok/s | TTFT s | tokens match |
|---|---|---|---|---|
| 1 | 7.6603 | 4.806 | ~1.7 (warm) / 2.98 (first-ever) | reference |
| 2 | 7.6664 | 4.820 | ~1.7 | ✓ same IDs |
| 3 | 7.6857 | 4.844 | ~1.7 | ✓ same IDs |
| **median** | **7.6664** | **4.820** | **1.7** | ✓ |

**Δ vs Step 5 (7.55): 1.015×. Δ vs Step 0: 11.97×.** Token IDs ID-for-ID identical to Step-1 reference. Fast-math gave a small kernel-throughput bump (compiler emitting fused mul-add for the dot products); the binary cache was pure UX.

---

## Step 7 — Fused gate_proj + up_proj + silu*mul (REVERTED — REGRESSION)

Wrote `fused_gate_up_silu_k896_no4`: one kernel reads x once, computes BOTH the gate and up dot products into the same WG, and stores `silu(gate) * up` directly. Saves 1 launch + 1 round-trip of the 4864-element gate buffer per layer (24 layers × 9.5KB = 228KB/token saved).

| Run | decode tok/s | tokens match |
|---|---|---|
| 1 | 4.0826 | reference |
| 2 | 4.3224 | ✓ same IDs |
| 3 | 4.8318 | ✓ same IDs |
| median | **4.3224** | ✓ |

**Δ vs Step 6 (7.66): 0.56× — 1.78× REGRESSION.** Code reverted (kept behind `kFuseGateUpSilu=false` flag).

**Why it lost:** the kernel keeps **8 fp32 accumulators per thread** (4 gate + 4 up) plus intermediate vec4s for the 4-output-per-WG design. Adreno 620's per-thread register budget can't hold all 8 — register spill to local memory on every loop iteration craters the kernel. Mamba's "fused norm+GEMV redundancy" memory captures the same lesson: fusion at decode-M=1 only pays off if per-thread resource pressure stays the same.

A non-regressing fused version would split into 2 mini-fused kernels (one for gate-half, one for up-half, each with ≤4 accs), then a tiny epilogue kernel that does `silu(gate) * up`. Three launches instead of one — defeats the original launch-collapse rationale. Skipping.

Token IDs were correct (the math is right). The kernel itself is preserved in `kernels/gemv_m1.cl::fused_gate_up_silu_k896_no4` for future revisit.

---

## Step 8 — `image2d_t` weight reads via Adreno texture cache (MEASURED — BIG WIN)

The single canonical lever to **break past the buffer-cache ceiling on Adreno**: wrap fp16 weight matrices as `image2d_t` views and read them via `read_imageh()`. Adreno has a dedicated read-only **texture cache** (separate from the buffer L1) that can sustain ~1.3-1.5× the BW of the buffer path for streaming weight reads.

Implementation:

- **New kernels** `gemv_m1_k896_no4_img` and `gemv_m1_k4864_no4_img` in `kernels/gemv_m1.cl`. Same coalesced wave-stride access pattern, same fp32-tree-reduce structure, same 4-output-per-WG (no4) layout — only the W read changes from `vload_half4` (buffer) to `read_imageh` (image2d_t). Sampler is `CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST`.
- **Image creation** via `clCreateImage` with `cl_image_desc.buffer = W` (the `cl_khr_image2d_from_buffer` extension — no data copy, same backing memory). Image format `{CL_RGBA, CL_HALF_FLOAT}` = 4 fp16 per pixel. Dimensions `(K/4, N)`.
- **Lazy per-buffer image cache** in `utils.cpp::get_or_create_w_image` keyed by source `cl_mem`. Each weight gets one image view, created on first decode call, reused for the rest of the process. Negative-cache for weights that exceed the device's `CL_DEVICE_IMAGE2D_MAX_HEIGHT` (16384 on Adreno 620).
- **Routing**: `try_gemv_m1` tries the image kernel when `M=1`, `K∈{896,4864}`, `N%4==0`, `N≥8`. On image-creation failure (e.g. lm_head N=151936 > 16384) it falls through to the buffer-backed `gemv_m1_kK_no4` automatically.

Coverage (which weights routed to image vs buffer):

| Weight | N | K | Image dims (K/4 × N) | Path |
|---|---|---|---|---|
| q_proj | 896 | 896 | 224 × 896 | **image** |
| k_proj | 128 | 896 | 224 × 128 | **image** |
| v_proj | 128 | 896 | 224 × 128 | **image** |
| o_proj | 896 | 896 | 224 × 896 | **image** |
| gate_proj | 4864 | 896 | 224 × 4864 | **image** |
| up_proj | 4864 | 896 | 224 × 4864 | **image** |
| down_proj | 896 | 4864 | 1216 × 896 | **image** |
| lm_head | 151936 | 896 | 224 × 151936 | buffer (exceeds 16384 max image height) |

So 7 of 8 hot weights are now texture-cached. Only lm_head (27% of total weight read) remains on buffer cache.

| Run | decode tok/s | total tok/s | TTFT s | tokens match |
|---|---|---|---|---|
| 1 (cold rebuild) | 7.04 | 4.56 | 2.66 | reference |
| 2 (warm) | 8.0986 | 4.954 | ~1.7 | ✓ same IDs |
| 3 (warm) | 8.0324 | 4.919 | ~1.7 | ✓ same IDs |
| 4 (warm) | 8.1197 | 4.970 | ~1.7 | ✓ same IDs |
| **median (warm)** | **8.0986** | **4.954** | **1.7** | ✓ |

Decode time per token: 1000 / 8.10 = **123.5 ms/token**.
Effective decode bandwidth: 942 MB / 123.5 ms = **7.63 GB/s = 76.3% of 10 GB/s buffer-cache ceiling**.

**Δ vs Step 6 (7.67): 1.056×. Δ vs Step 0 (0.6402): 12.66×.** Token IDs ID-for-ID identical to Step-1 reference.

The per-step gain looks modest (5.6%) because lm_head still pulls 27% of weight bytes through the buffer path. If we tiled lm_head into 10× sub-images (15200 rows each) and routed those through the texture cache too, predicted landing zone would be **8.7-9.5 tok/s = 82-90% of 10 GB/s ceiling** — possibly poking into "exceeds buffer-cache realistic ceiling" territory if Adreno's texture cache delivers >10 GB/s effective on the layer weights.

---

## Step 9 — Tiled `image2d_t` for `lm_head` (MEASURED — closes the texture-cache coverage gap)

The remaining gap after Step 8: `lm_head` (N=151936, K=896, 259.7 MB = 27% of total weight read per token) is the only big GEMV NOT on the texture cache, because Adreno 620's `CL_DEVICE_IMAGE2D_MAX_HEIGHT = 16384` < 151936. Step 9 tiles `lm_head` into 10 sub-images and dispatches the existing `gemv_m1_k896_no4_img` kernel once per tile.

### Failed first attempt — packed16 layout (REGRESSION)

The naïve idea: keep a single image2d at height ≤ 16384 by packing 16 fp16 along x instead of 4. Image dimensions become `(K/16, N) = (56, 151936)` with format `{CL_RGBA, CL_HALF_FLOAT}` reinterpreted as 16-fp16-per-pixel via 4 fp16 per channel × 4 channels. This fits in 16384 height when split into 10 chunks of 15200 rows each (still under the limit, just on a different axis).

| Run | decode tok/s |
|---|---|
| 1 | 4.62 |
| 2 | 4.59 |
| 3 | 4.65 |
| **median** | **4.62** |

**Δ vs Step 8 (8.10): 0.57× — 1.75× REGRESSION.** Reverted.

**Why packed16 lost.** Adreno's texture cache is **2D-tiled** (typical mobile GPU pattern: 8×8 or 16×16 fp16 pages). When the no4 kernel computes 4 outputs in one WG, the 4 output rows correspond to 4 different y-coords in the image. With the standard layout (4 fp16/pixel), those 4 rows are 1792 bytes apart — same 2D page → ONE cache fill serves all 4. With the packed16 layout, the rows are 7168 bytes apart (4× wider) → **4 different 2D tiles → 4 cache fills per inner step**, each evicting parts of the others. The texture cache geometry is opaque, but every result we see is consistent with this model. **Lesson:** never re-pack the inner-vector dimension to dodge a 1D image-size limit; it always destroys 2D-tiled spatial locality.

### Working approach — tiled standard layout

- `WImageEntry` in `utils.cpp` extended to hold EITHER a single `cl_mem image` OR a `std::vector<WImageTile>` where each `WImageTile = { cl_mem sub_buffer, cl_mem image, int row_offset, int row_count }`.
- `get_or_create_w_image()` tries standard image first (path used by all Step-8 weights). On `clCreateImage` failure with `pix_h > img_max_h`, it falls through to the tiled path:
  - Carve `W` into chunks of `tile_rows = min(N_remaining, img_max_h)` (Adreno = 16384).
  - For each chunk: `clCreateSubBuffer(W, origin = row_offset * row_stride_bytes, size = tile_rows * row_stride_bytes)`. Origin must be aligned to `CL_DEVICE_MEM_BASE_ADDR_ALIGN`. Qwen's `K=896` fp16 row = 1792 bytes = 1024-bit aligned, so every row offset is aligned automatically.
  - `clCreateImage` with `image_desc.buffer = sub_buffer`, dims `(K/4, tile_rows)`, format `{CL_RGBA, CL_HALF_FLOAT}` — same standard layout as Step 8, just a window over each chunk.
- `get_or_create_out_sub()` mirror cache for **output** sub-buffers, keyed by `(cl_mem out, offset, size)` so dispatch loops don't recreate sub-buffers per token.
- `try_gemv_m1()` adds a tile-loop branch:
  ```cpp
  for (auto& tile : entry.tiles) {
      cl_mem out_sub = get_or_create_out_sub(out, tile.row_offset * sizeof(half), tile.row_count * sizeof(half));
      // Dispatch gemv_m1_k896_no4_img with W=tile.image, output=out_sub, n=tile.row_count
  }
  ```
- For Qwen's `lm_head` (N=151936): 10 tiles of 15200 rows each (last tile 16936 ≤ 16384? — actually the loop produces `ceil(151936/16384) = 10` tiles, sizes mostly 16384 with one smaller; the exact split is whichever the loop chooses). Each tile fits in image-height; each tile's sub-image lives on the texture cache.

### Run table

| Run | decode tok/s | total tok/s | TTFT s | tokens match |
|---|---|---|---|---|
| 1 | 8.31 | 4.99 | ~1.7 | reference |
| 2 | 8.37 | 5.02 | ~1.7 | ✓ same IDs |
| 3 | 8.32 | 4.99 | ~1.7 | ✓ same IDs |
| **median** | **8.32** | **4.99** | **1.7** | ✓ |

Decode time per token: 1000 / 8.32 = **120.2 ms/token** (saved ~3.3 ms vs Step 8).
Effective decode bandwidth: 942 MB / 120.2 ms = **7.84 GB/s = 78.4% of 10 GB/s ceiling** (was 76.3%).

**Δ vs Step 8 (8.10): 1.027×. Δ vs Step 0 (0.6402): 13.0×.** Token IDs ID-for-ID identical to Step-1 reference:
`3837, 18532, 572, 264, 883, 879, 12163, 304, 264, 2613, 14126, 58883, 1519, 1030, 264, 2409, 3753, 323, 264, 6233, 13551, 58883, 1519, 572, 1602, 6247, 58883, 3966, 1899, 3837, 383, 3937` ✓.

The 2.7% gain is smaller than naïvely scaling Step 8's 5.6% × (27% / 73%) — that arithmetic would predict ~2.1%, so we're slightly above expectation. The shortfall vs the optimistic 8.7-9.5 prediction in Step 8 is because (a) `lm_head` only fires once per token vs the layer weights firing 24× — its share of decode time is larger in MB but smaller in launch-overhead amortization headroom; (b) 10 dispatches × 15200 outputs each has more dispatch overhead than 1 × 151936; (c) cross-tile `clCreateSubBuffer` of `out` adds a touch of write-side overhead. Still strictly net-positive and now every hot weight goes through the texture cache.

---

## Step 10a — WG=32 sweep on `no4_img` kernels (MEASURED, NEGATIVE — REVERTED)

The hypothesis was that WG=32 (vs the default WG=64) would put more workgroups in flight on Adreno 620 — better latency hiding when one WG stalls on a texture-cache miss. Geometry was clean for both K dimensions:

| Kernel | K_PIX | WG=64 layout | WG=32 layout |
|---|---|---|---|
| `gemv_m1_k896_no4_img` | 224 | 3 waves of 64 + tail of 32 | 7 waves of 32, no tail |
| `gemv_m1_k4864_no4_img` | 1216 | 19 waves of 64 | 38 waves of 32 |

Implementation: added `gemv_m1_k896_no4_img_wg32` and `gemv_m1_k4864_no4_img_wg32` alongside the WG=64 versions, dispatched via `NNOPT_GEMV_WG=32` env var. Same no4 4-output-per-WG layout, just halved partial-arrays and re-tiled K loop.

| Run | decode tok/s | TTFT s | tokens deterministic |
|---|---|---|---|
| 1 (WG=32, cold rebuild) | 7.19 | 23.7 | reference |
| 2 (WG=32, warm) | 8.32 | 11.5 | ✓ same IDs |
| 3 (WG=32, warm) | 8.33 | 2.7 | ✓ same IDs |
| baseline (WG=64, post-revert) | 8.34 | 2.8 | ✓ Step-1 reference |

**Δ vs Step 9 (8.32): 1.001× — within run noise. Reverted.**

Why no signal: at WG=64, gate/up_proj already dispatch `1216 WGs` per launch (N=4864/4 outputs per WG), and lm_head dispatches `15200/4 ≈ 3800 WGs` per tile × 10 tiles. Adreno 620 has plenty of WGs in flight already at WG=64 — halving WG to 32 doubles the count but each does half as much work, so total kernel time unchanged. The hypothesis would have paid off if the GEMVs were already at the WG-count floor (which would be true on much smaller models like SmolLM2 with N=576), but Qwen's larger projections saturate the scheduler at WG=64.

**Reverted cleanly to byte-exact Step 9 source.** The kernel cache key (`fnv1a64(source + ...)`) hits the existing Step-9 binary post-revert → token IDs return to Step-1 reference (verified). Note: the *during-experiment* runs (with the extra kernel functions added but the dispatch toggle off) produced different but coherent token IDs — the source-text change forced a recompile, and Adreno's OpenCL compiler made different fp16 ordering choices. This confirms binary-cache content (not just source text) gates token-ID stability.

Lesson: WG sweeps only matter when the launch is WG-count-starved. Above ~1000 WGs/launch on this device, the choice is invisible.

---

## Final cumulative summary

| Step | decode tok/s | × over Step 0 | × over prev | What changed |
|---|---|---|---|---|
| 0 baseline (release) | 0.6402 | 1.00× | — | scaffold-default kernels, CLBlast HGemm M=1 everywhere |
| **1 (gemv_m1 with coalesced wave-stride)** | **4.4626** | **6.97×** | **6.97×** | cooperative WG=64 + vec4 + tree-reduce GEMV replaces all 8 CLBlast HGemm M=1 sites |
| **2 (no4: 4 outputs per WG)** | **6.8502** | **10.70×** | **1.54×** | 4× fewer WGs, x reused once across 4 W rows, 4 independent fp32 accs/thread |
| 3 (element_add_inplace + GPU argmax + vec4 swiglu/bias_add/element_add) | 6.7325 | 10.52× | 0.98× | within noise — Step 2 already saturated per-call BW |
| **4 (persistent Mlp+Attention activation buffers)** | **7.3660** | **11.51×** | **1.09×** | 216 alloc/free per decode token eliminated |
| **5 (persistent LayerNorm out + lm_head logits)** | **7.5513** | **11.79×** | **1.03×** | last 25+1 alloc/free per token eliminated |
| **6 (binary cache + -cl-fast-relaxed-math)** | **7.6664** | **11.97×** | **1.015×** | TTFT 30s→2-3s; small kernel-throughput bump |
| 7 (fused gate+up+silu) | 4.32 | 6.75× | 0.56× | REVERTED — register spill on Adreno 620 |
| **8 (image2d_t weights via Adreno texture cache)** | **8.0986** | **12.66×** | **1.056×** | 7 of 8 hot weights texture-cached (lm_head still buffer) |
| 9a (lm_head packed16 image) | 4.62 | 7.22× | 0.57× | REVERTED — packed-x layout destroyed Adreno's 2D-tiled texture cache locality |
| **9 (lm_head tiled image2d, 10 chunks × ≤16384 rows)** | **8.32** | **13.00×** | **1.027×** | last hot weight (lm_head, 27% of decode bytes) on texture cache |
| 10a (WG=32 sweep) | 8.33 | 13.01× | 1.001× | NEGATIVE — within noise; Adreno 620 already saturates scheduler at WG=64 with Qwen's GEMV sizes; reverted |

**Final: 8.32 tok/s decode at fp16. ~78.4% of realistic 10 GB/s buffer-cache ceiling.**

**TTFT: ~2-3s on first run, ~1.7s on subsequent runs** (binary cache hits). The user-reported 32s TTFT was a fresh process with empty kernel cache; that's now fixed.

Comparable utilization on this exact device:
- Mamba-130M ended at 24.56 tok/s = 63.4% of its 38.7 tok/s ceiling.
- SmolLM2-135M ended at 10.46 tok/s = 28% of its ~37 tok/s ceiling.
- **Qwen2.5-0.5B 8.32 tok/s = 78.4% of 10.6 tok/s ceiling.**

The highest utilization of any port on this device. Qwen's larger weight footprint (942 MB/token vs 258 MB/token for Mamba) means a larger fraction of decode time is spent on the large GEMVs that we hand-vectorized (and now fully texture-cached), with proportionally less time in launch overhead and small auxiliary kernels.

### What's left between us and the ceiling (closing the last 22%)

At 7.84 GB/s effective wall-clock BW we're at 78% of the 10 GB/s realistic ceiling. Per-kernel we're likely at 90-95% on the big GEMVs (every hot weight is now texture-cached). The remaining gap is **launch overhead + small-op overhead + write-side BW** (logits, KV-cache writes, residual streams). To close further:

| Lever | Predicted | Notes |
|---|---|---|
| Fused QKV projection (single GEMV with stacked W [Q+2KV, H]) | 1.02–1.04× | -2 launches × 24 layers; q/k/v are tiny so launch savings dominate |
| Fused `o_proj + residual_add` and `down_proj + residual_add` (M=1) | 1.02–1.04× | -2 launches × 24 layers; SmolLM2 Step 4 template |
| Fused `gate_proj + up_proj + silu*mul` via 2 mini-fused GEMVs + tiny epilogue (Step 7 retry, no register spill) | 1.01–1.03× | three launches still beats unfused four; lukewarm prediction since Step 7 already validated the register-pressure hazard |
| WG=128 sweep on the no4 image kernels | 1.00–1.05× | doubles per-WG x reuse but halves WG count; needs measurement |
| Persistent kernel for the whole layer (one launch per layer) | 1.05–1.10× | research-y — eliminates ~16 launches × 24 layers; risk of hitting GPU command-buffer limits |

Realistic landing zone with the first three: **8.5–9.0 tok/s = 80–85% of 10 GB/s ceiling**. Beyond ~9 tok/s requires int8 quantization (deferred per project rule — would halve weight BW and roughly double tok/s ceiling to ~21 tok/s).

### Generic lessons (consistent with prior Mamba/SmolLM2 ports)

1. **Custom `gemv_m1` is the single biggest mobile-GPU lever** for any LLM at decode. CLBlast HGemm M=1 is poorly tuned across all K — 7× speedup on the FIRST replacement on Qwen, 3.4× on Mamba, ~5× on SmolLM2. Always do this first.
2. **Coalesced wave-stride access** (each iteration of the wave covers contiguous bytes, NOT each thread covers a contiguous chunk) is required for full Adreno BW. If you ship the naïve stride-kp pattern, you ship 3× under-utilization; this should be the scaffold default.
3. **Multi-output-per-WG (no4)** is a free 1.2-1.5× on top of the cooperative GEMV when N % 4 == 0. Always specialize the large-N kernels (gate/up/down/lm_head) for it.
4. **Persistent activation buffers** matter on Adreno — `clCreateBuffer` is ~50 µs of driver bookkeeping per call. At 200+ allocs/token this adds up. The pattern (lazy-allocate, grow-on-demand, `clRetainMemObject` before handoff) generalizes cleanly.
5. **GPU-side argmax doesn't move the needle at 7 tok/s** — saves ~150 µs per token, buried in 135 ms decode. Becomes more visible at higher tok/s (post-int8). Keep it for code hygiene + future-proofing.
6. **Adreno's texture cache is 2D-tiled, not 1D linear** — when an inner kernel touches multiple rows of the same image per WG, those rows must remain on the standard layout (4 fp16 across x). Re-packing the inner-vector dimension to 16-fp16-per-pixel to dodge an image-height limit (Step 9a) destroys 2D spatial locality and regresses 1.7×. The right way to dodge image-size limits is **tiling along the y-axis** with `clCreateSubBuffer` + per-tile `image2d_t` views (Step 9 working approach), not packing along x.
7. **Texture-cache coverage matters per-byte, not per-launch** — Step 9 (lm_head: 1 dispatch/token, 27% of weight bytes) was a real ~3% gain even though it's a single launch, because the BW share is what the cache improves. Step 8 was 5.6% from 7 weights × 24 layers = 168 launches, and Step 9 added the last 27% of bytes through 10 dispatches. Bytes-on-cache is the right scoreboard, not launches-on-cache.

### Step 11 — `cl_qcom_recordable_queues` (record + replay decode loop) — BLOCKED on arg-override SDK header

**Goal**: amortize CPU-side per-kernel dispatch overhead across decode iterations by recording the full layer compute once and replaying it (with `start_pos` arg overrides) every subsequent decode step. Microbenchmark probe achieved 2.64× speedup on a no-op kernel (23.3 µs/dispatch baseline → 8.85 µs/dispatch replay).

**Result**: integration captured the recording successfully and the iter-1 self-replay produced **+6.5% (8.9 → 9.48 tok/s)** when the override array was zero-length, BUT the per-replay arg-override mechanism returns -59 (`CL_INVALID_OPERATION`) on Adreno 620 for both struct layouts probed (`cl_uint arg_indx` and `size_t arg_indx`). Without the actual `cl_qcom_recordable_queues` SDK header, the correct `cl_array_arg_qcom` field order/sizing can't be pinned down. `clSetKernelArg` between replays does NOT update the recording — Adreno snapshots args at record time.

**Workarounds that don't help on autoregressive decode**:
- Re-record every iter: doubles dispatch CPU cost, kills the gain.
- Stable-shape gws (mask out `k≥seq_k` inside `gqa_attn_scores`): the unused work-items at MAX_POS scale would 100-1000× the kernel for short contexts.
- `start_pos` via persistent `cl_mem`: still leaves `gqa_attn_scores` gws variable per iter.

**Disposition**: code gated behind `NNOPT_RECORD=1` env var so it's off by default and ready for re-investigation when SDK access is obtained.

**What did ship cleanly from this work**:
- Persistent decode hidden buffer (`Model::buf_decode_hidden_`) + `Embedding::set_decode_token` / `forward_into_decode` — perf-neutral but eliminates one `clCreateBuffer` per token.
- `kv_write` kernel as a recordable replacement for `clEnqueueCopyBuffer(K/V → cache)` — perf-neutral on its own; required because only `NDRangeKernel` is recordable. Useful even without recording.


### Step 12 — GPU per-kernel profiler infrastructure (`NNOPT_PROFILE=1`) — SHIPPED

`src/prof.{h,cpp}` adds a deferred-readout profiler that wraps `clEnqueueNDRangeKernel` via `nnopt_prof::enqueue()`. When the env is unset there's zero overhead (no event creation). When set, every dispatch records a `cl_event`; `nnopt_prof::dump()` (called at end of `generate()`) syncs the queue once and prints a sorted (kernel_name, count, total_ms, avg_µs, max_µs, %total) table. Prefill-only events are dropped by `nnopt_prof::reset()` after the first-token benchmark.

**Confirmed decode breakdown for Qwen2.5-0.5B fp16** (16 tokens, prompt "Once upon a time"):

| Kernel | calls | total_ms | %decode |
|---|---:|---:|---:|
| gemv_m1_k896_no4_img | 2310 | 1301.0 | **58.1%** |
| gemv_m1_k4864_no4_img | 360 | 863.0 | **38.5%** |
| rmsnorm_forward | 735 | 15.7 | 0.7% |
| embedding_forward | 15 | 8.4 | 0.4% |
| bias_add_rowmajor | 1080 | 8.4 | 0.4% |
| gqa_softmax | 360 | 8.3 | 0.4% |
| gqa_attn_scores | 360 | 6.6 | 0.3% |
| gqa_attn_out | 360 | 6.4 | 0.3% |
| element_add (residual) | 720 | 6.3 | 0.3% |
| kv_write | 720 | 6.0 | 0.3% |
| swiglu_inplace | 360 | 4.3 | 0.2% |
| rope_apply_qk | 360 | 3.2 | 0.1% |
| argmax_partial+finalize | 30 | 0.9 | 0.0% |
| **total** | | **2238** | |

GEMVs are **96.6% of decode**. Implication: any further single-digit-percent gains must come from the GEMV kernels themselves (kernel rewrites) or from fusing GEMV+epilogue pairs to halve launch counts. Auxiliary kernels (rmsnorm, softmax, element_add, etc.) are individual sub-1% lines — collectively 3.4% — and not worth optimizing without quantization first.

Per-token decode = 149 ms = 6.7 tok/s, matching the 6.97 tok/s wall-clock measurement. Per-token GEMV breakdown (estimated by N-weighted work units within K=896):

| Sub-component | %token |
|---|---:|
| K=4864 down_proj | 38.5% |
| K=896 gate_proj + up_proj (N=4864 ×2) | ~31% |
| K=896 lm_head (10 tiles × N≈15194) | ~20% |
| K=896 q_proj + o_proj (N=896 ×2) | ~6% |
| K=896 k_proj + v_proj (N=128 ×2) | ~1% |

### Step 13 — `qcom_sub_group_shuffle_xor` wave-reduce GEMV variant — DEAD END

**Goal**: replace the 6-step / 6-barrier / 24-local-store fp32 tree-reduce in `gemv_m1_k{896,4864}_no4_img` with a barrier-less XOR-shuffle butterfly using the Qualcomm vendor extension `cl_qcom_subgroup_shuffle`. PDF §8.9 + §9.2.2 explicitly recommend hardware-reduce over local-mem reduce on Adreno; KHR `sub_group_reduce_add` (OpenCL 2.0 builtin) is rejected by the Adreno 620 OpenCL 1.2-target compiler, so the vendor extension is the only path.

**Result**: kernel compiled successfully under both `qcom_reqd_sub_group_size("full")` and `qcom_reqd_sub_group_size("half")`, but at runtime produced numerically-wrong logits (gibberish output: `"Onceuponatime，               "` after the first comma) AND ran ~5× slower than the local-mem variant (1.07 → 1.43 tok/s vs 6.98 baseline) when enabled via `NNOPT_GEMV_SG=1`.

Suspected cause: `qcom_sub_group_shuffle_xor` with `CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM` does not behave as documented when WG_SIZE != hardware wave size, OR the SoC's shuffle unit is slow enough that the saved barriers don't pay for the shuffle latency on Adreno 620. Confirming would require a pure microbenchmark of the shuffle unit and a known-good wave-size probe — not pursued.

**Disposition**: kernel code (`gemv_m1_k896_no4_img_sg`, `gemv_m1_k4864_no4_img_sg`) and routing (NNOPT_GEMV_SG env var in `try_gemv_m1`) kept as a documented dead end. The clCreateKernel sites tolerate failure silently (sg_kernel = nullptr → routing falls back to default), and the env is off-by-default, so the default binary is unaffected. A loud "BROKEN — DO NOT ENABLE" comment in `kernels/gemv_m1.cl` warns future-me before another A/B attempt.

**Generic lesson**: Adreno's vendor-shuffle path is NOT a drop-in replacement for local-mem-tree-reduce on Adreno 620. The Snapdragon programming guide's recommendation appears to apply to newer Adrenos (660+) where the shuffle unit is faster and the wave-size attribute semantics are tighter.

---

### Step 14 — Compilation-unit cleanup: stub dead kernels, remove _sg/_packed16/_fused — MEASURED

**Goal**: Remove the dead code left by Steps 7, 11, 13 that had accumulated in `kernels/gemv_m1.cl`. The Adreno OpenCL compiler is extremely sensitive to the set of kernels in the compilation unit (Step 10b: adding `_add` stubs gave +8.7%; Step 13: adding `_sg` variants caused −27% regression, 9.04→6.97 tok/s). Removing harmful dead code should recover Step 10b level.

**What was removed**:
- `gemv_m1_k896_no4_img_sg` and `gemv_m1_k4864_no4_img_sg` (Step 13 dead end — wrong results, 5× regression)
- `gemv_m1_k896_no4_img_packed16` (Step 11 artifact — was never dispatched, just dead compile weight)
- `fused_gate_up_silu_k896_no4` (Step 7 regression — register spill; already stubbed in utils.cpp)
- `struct GemvM1State` cleaned of 4 dead kernel handles; `ensure_gemv_m1()` cleaned of 4 dead `clCreateKernel` calls
- `try_gemv_m1()` dead routing (`sg_init`/`sg_on` flag and conditionals) removed

**What was kept**:
- `gemv_m1_k896_no4_img_add`, `gemv_m1_k4864_no4_img_add` — these are Step 10b beneficial stubs (not dispatched, but their presence in the compilation unit gave +8.7%)
- `gemv_m1_k896_no4_img`, `gemv_m1_k4864_no4_img` — hot-path image kernels
- `gemv_m1_k896_no4`, `gemv_m1_k4864_no4` — buffer fallback
- `gemv_m1_k896` — single-output fallback (K=896)
- `gemv_m1_k4864` — **replaced with a minimal noop stub** (see below)

**The `gemv_m1_k4864` stub issue**: During cleanup, adding a 19-wave unrolled `gemv_m1_k4864` kernel (single-output K=4864 buffer variant) caused a severe 3.x tok/s regression — the large 19-iteration `#pragma unroll` loop disturbed the Adreno compiler's register allocation for the hot image kernels. Fixed by replacing its body with a minimal noop (no compute, satisfies `clCreateKernel`). The `fallback_buffer` dispatch path in utils.cpp guards against ever dispatching it (`if (K == 4864 && !use_no4) return false` → falls to CLBlast).

**Results** (3 runs, runs 2-4 after warm kernel cache, median):

| Run | decode tok/s |
|---|---:|
| 1 (cache miss) | 8.01 |
| 2 | 9.25 |
| 3 | 8.98 |
| 4 | 9.02 |

**Median (runs 2-4): 9.02 tok/s** (+8.4% vs 8.32 tok/s pre-session baseline)

Effective BW: 942 MB / (1000/9.02 ms) = **8.50 GB/s** (60.5% of 14.0 GB/s ceiling)

Token IDs verified match Step-1 reference: 3837, 18532, 572, 264, 883, 879, 12163, 304 ✓

**Compilation-unit lesson (key insight)**: Every kernel added to `gemv_m1.cl` is compiled together, and the Adreno shader compiler makes global register/scheduling decisions across ALL kernels in the program. Large unrolled loops (`j < 19` × 4 vload/dot ops) dramatically increase the compiler's register pressure analysis cost and can spill neighboring kernels' live ranges. A noop-body stub (empty or `(void)arg`) satisfies `clCreateKernel` without contributing compile weight. The Step 10b `_add` stubs work *because* their body is trivially small (just a write at the epilogue), not because dispatch hints to the compiler.

**Next**: Plan item A — 8-output-per-WG (`no8`) GEMV variant. Predicted +10-20% on k4864, +5-10% on k896.

---

### Step 15 — `no8` GEMV (8 outputs per WG) — DEAD END on Adreno 620

**Goal**: Halve workgroup count by processing 8 output rows per WG instead of 4.

**What was attempted**:
1. Added `gemv_m1_k896_no8_img` and `gemv_m1_k4864_no8_img` inside `gemv_m1.cl` → **severe regression to 3.4 tok/s** (same Adreno compiler-sensitivity phenomenon as Steps 10b/13: large unrolled kernels disturb the compiler's optimization of neighboring hot kernels).
2. Moved no8 kernels to a separate `kernels/gemv_m1_no8.cl`, compiled in an independent `cl_program` to avoid compiler interference → regression fixed, but now **8.23 tok/s warm** — still slower than no4 (9.02 tok/s baseline).

**Diagnosis**: no8 is genuinely slower on Adreno 620, not just a compilation-unit artifact. Root cause: each inner-loop iteration fetches 8 `float4` W-rows simultaneously (w0..w7 = 32 fp32 registers) + 8 accumulators (8 regs) + xv (4 regs) = ~44 live registers per thread. At WG_SIZE=64 threads, this exceeds the register budget per wave on the Adreno 620 shader processor (estimated ~128 regs/wave on this tier), reducing effective WG occupancy. The 2× WG scheduling savings don't compensate.

**Files**: `kernels/gemv_m1_no8.cl` kept as documented experiment (not loaded at runtime). `GemvM1State::kernel_k896_no8_img` and `kernel_k4864_no8_img` are null — routing falls back to no4_img.

**Lesson**: On Adreno 620, increasing outputs-per-WG beyond 4 for texture-backed GEMVs adds too much register pressure. The sweet spot is no4 (4 outputs): 4 accs + 4 W reads per iter = ~16-20 live registers per thread, well within budget.

**Effective BW at Step 15 baseline (= Step 14): 942 MB / (1000/9.02 ms) = 8.50 GB/s (60.5% of ceiling)**

---

### Step 16 — `cl_qcom_recordable_queues` counter-buffer approach — DEAD END

**Goal**: Use the `cl_qcom_recordable_queues` extension to capture one decode step as a recording and replay it on subsequent steps, eliminating per-step `clSetKernelArg` + kernel-dispatch overhead (~23 dispatches × 24 layers = 552 dispatches/token). Expected gain: ~5% based on profiler data showing auxiliary kernels at 3.4% of decode time.

**Background**: In a prior session (2026-05-03), a zero-override replay was measured at +6.5% (8.9→9.48 tok/s) on an earlier binary. However, `clSetKernelArg` returns -59 (`CL_INVALID_OPERATION`) after `clEndRecordingQCOM`, making per-replay arg updates impossible. The counter-buffer workaround: move `start_pos`/`seq_k` from scalar kernel args into a 2-int `buf_counter_` buffer whose address is baked into the recording; update only the contents before each replay.

**What was tried (all failed — GPU always sees recording-time counter values)**:

| Approach | buf_counter_ flags | Write method | Result |
|---|---|---|---|
| 1 | `CL_MEM_READ_WRITE` | `clEnqueueWriteBuffer(CL_FALSE)` | Stale — GPU sees {5,6} every replay |
| 2 | `CL_MEM_READ_ONLY` | `clEnqueueWriteBuffer(CL_TRUE)` | Stale |
| 3 | `CL_MEM_READ_ONLY \| CL_MEM_ALLOC_HOST_PTR` | `clEnqueueWriteBuffer(CL_TRUE)` | Stale |
| 4 | `CL_MEM_READ_ONLY \| CL_MEM_USE_HOST_PTR` (SVM-backed) | Direct CPU write to `svm_counter_[0/1]` | Stale |
| 5 | Same as 4, SVM fine-grain | `clSetKernelArgSVMPointer` at each call site | Stale |

**Symptom**: First replay always produced the correct next token. All subsequent replays produced identical garbage (same token repeated). Logging confirmed `svm_counter_[0]=5, svm_counter_[1]=6` on the host at replay 3 — but the GPU still decoded as if counter = {5, 6} (recording-time value). When `NNOPT_RECORD=1` was active, the sequence was:
- Prompt prefill (forward()): correct token (start of decode)
- Decode iter 0 (live, no recording): correct — "Once"
- Decode iter 1 (live, no recording): correct — "upon"
- Decode iter 2 (record+first-replay, counter={5,6}): correct — "was"
- Decode iter 3 (replay with counter CPU-written to {6,7}): wrong — "．．．．"
- Decode iter 4+ (replay): same wrong token repeated

**The key asymmetry**: The embedding's `ids_buf_decode_` (`CL_MEM_READ_ONLY`, 4 bytes, `clEnqueueWriteBuffer(CL_FALSE)`) **does** update correctly between recording replays — the embedding produces different token embeddings each step. The counter buffer with identical flags and write method does not update. The difference may be that `ids_buf_decode_` is set via a standard `clSetKernelArg(cl_mem)` call during recording, while the counter arg was being set via `clSetKernelArgSVMPointer` in some variants.

**Root cause hypothesis**: Adreno 620's recording implementation bakes counter buffer contents as GPU L2-cached constants at record time. This is independent of:
- Buffer allocation flags (READ_WRITE, READ_ONLY, ALLOC_HOST_PTR, USE_HOST_PTR)
- SVM fine-grain coherency guarantees (confirmed working: `clSVMAlloc` with `CL_MEM_SVM_FINE_GRAIN_BUFFER` allocates successfully)
- Whether CPU writes are direct (SVM) or via DMA (clEnqueueWriteBuffer)

**Untested — last hope**: Approach exactly matching `ids_buf_decode_`: `CL_MEM_READ_ONLY` + standard `clSetKernelArg(cl_mem)` (NOT `clSetKernelArgSVMPointer`) + `clEnqueueWriteBuffer(CL_FALSE)` for writes. All counter arg sites already use `clSetKernelArg(cl_mem)` after the final revert. The only difference from approach 2 above: approach 2 was tested before all kernels were updated to read from `counter` (rope and gqa_attn_scores were still using scalar args at that point). The full counter-buffer path was never benchmarked without SVM.

**Performance impact**: The 9.02 tok/s baseline is unaffected. `NNOPT_RECORD=1` is off by default; the normal decode path writes `buf_counter_` via `clEnqueueWriteBuffer` and kernels read from it correctly (no recording involved).

**What was reverted (2026-05-05)**:
- Removed `svm_counter_`, `fn_svm_alloc_`, `fn_svm_free_` from `src/model.h` and `src/model.cpp`
- Removed SVM allocation block (`clSVMAlloc` call) from `Model::Model()`
- Removed `counter_svm` parameter from `Attention::forward()` and `kv_write_kernel()`
- Removed `_set_counter_arg` SVM helper from `attention.cpp`
- Reverted `gqa_attn_scores` GWS from fixed-max `(QH, 1, MAX_POSITION_EMBEDDINGS=32768)` back to variable `(QH, seq_q, seq_k)` — the fixed GWS caused 14×32768=458K idle workitems per step
- Removed `<dlfcn.h>` from `attention.cpp` and `utils.cpp`

**What was kept** (working correctly for normal decode):
- `buf_counter_` (`CL_MEM_READ_WRITE`, 2 ints) in `model.h`/`model.cpp` — host writes via `clEnqueueWriteBuffer` before each step
- All kernels (rope, kv_write, gqa_attn_scores, gqa_softmax, gqa_attn_out) reading start_pos/seq_k from `counter` buffer — this works correctly in non-recorded mode
- Recording infrastructure (`recordable_q_`, `rec_decode_`, `fn_new_/end_/release_/enqueue_`) gated behind `NNOPT_RECORD=1`

**Baseline (unchanged): 9.02 tok/s**

---

## Next steps (ranked by predicted gain)

### C — Fused QKV projection [~+1.7%]

Pre-stack Q+K+V weights into a single `[Q_DIM + 2*KV_DIM, HIDDEN] = [1152, 896]` matrix at model load. One GEMV dispatch + one bias_add per layer instead of 3+3. Eliminates 2 GEMV launches + 3 bias_add launches × 24 layers = 120 fewer dispatches/token. Risk: weight reordering at load (one-time cost). Output buffer slice indices in attention.cpp change.

### F — `cl_qcom_vector_image_ops` / `qcom_read_imagef_4x1` [unknown, risky]

Extension confirmed present (`cl_qcom_vector_image_ops` ✓). `qcom_read_imagef_4x1` reads 4 consecutive x-pixels in one call (returns `float16`), potentially 4× fewer texture fetch instructions. Risk: k4864 gives 1216/4=304 super-pixels / WG=64 → irregular iterations; register pressure likely similar to no8 failure. Needs a microbenchmark before full integration.

### G — W8A16 INT8 weight quantization [~+1.8–2×]

Halve weight bandwidth: quantize to int8 (1 byte vs 2 for fp16), store per-channel scale `float[N]`. GEMV inner loop: `uchar4` → `float4` → FMA. Bandwidth ceiling doubles from ~14.0 to ~29.8 GB/s effective. Predicted landing: 14–16 tok/s. Requires `cl_qcom_dot_product8` (confirmed available) or software int8×fp16 multiply. Project rule change needed (deferred per prior session).
