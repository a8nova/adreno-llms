# Benchmark log — LFM2.5-350M-Base on Razr 2020 (Adreno 620, fp16)

**Mission: push LFM2.5-350M-Base above its decode ceiling on the same Razr 2020 (Adreno 620, LPDDR4X-2133) used for the mamba2 / mamba1 / SmolLM2 ports.**

This file mirrors the format of `adreno-llms/src/models/mamba2-130m/BENCHMARK.md` so cross-port comparisons are 1:1. Numbers in this file are 3-run medians at fixed prompt + greedy decode + fixed seed.

## Architecture summary

LFM2.5-350M-Base is a **hybrid transformer + short-conv** decoder:
- 16 layers, layer_types = `[conv, conv, attn, conv, conv, attn, conv, conv, attn, conv, attn, conv, attn, conv, attn, conv]`
  - 6 attention layers (idx 2,5,8,10,12,14) — GQA, Q=16 heads / KV=8 heads, head_dim=64, RoPE θ=1e6, **q_layernorm + k_layernorm** before RoPE
  - 10 conv layers — Lfm2ShortConv: `in_proj → split(B,C,X) → conv1d(L=3) with cache → C * conv_out → out_proj`
- hidden_size=1024, intermediate_size=4608 (auto-adjusted from 6656 via 2/3 + 256-multiple round)
- vocab=65536 (large), **TIED embeddings** → lm_head reuses embed_tokens.weight (128 MB by itself)
- max_position_embeddings=128000 (huge alloc; benchmark uses 7-token prompt + 32 generated)

**Per-decode-token weight footprint (fp16):**

| Component | Bytes/layer | × N layers | Per token |
|---|---|---|---|
| MLP w1+w2+w3 (1024↔4608) | 27 MB | × 16 | 432 MB |
| conv (in_proj 6 MB + out_proj 2 MB + tiny conv) | ~8 MB | × 10 | 80 MB |
| attention (q 2 + k 1 + v 1 + out 2 MB + tiny norms) | ~6 MB | × 6 | 36 MB |
| operator_norm + ffn_norm + final_norm | ~4 KB | × 33 | <0.1 MB |
| lm_head (= embed_tokens, vocab 65536 × hidden 1024) | — | shared | 128 MB |
| **Total weight bytes / token** | — | — | **~676 MB** |

KV cache traffic is negligible at this prompt length (8 layers × few KB). Conv state is 1024 × 2 fp16 × 10 layers = 41 KB / step.

## Hardware ceiling

**Snapdragon 765G / Adreno 620 / LPDDR4X-2133 dual-channel.**

- Realistic GPU-visible bandwidth: **~10 GB/s** (mamba1-v2 / mamba2 confirmed empirically on this exact device).
- Adreno 620 fp16 ALU: ~2.32 TFLOPS — irrelevant; mobile LLM decode is BW-bound.

**Roofline at 10 GB/s realistic: 676 MB / 10 GB/s = 67.6 ms/token = 14.8 tok/s.**

This is **less than half of mamba2's 37.2 tok/s ceiling** because LFM2.5 is 2.5× larger (676 MB/tok vs 269 MB/tok). The vocab=65536 lm_head alone is bigger than mamba2's entire per-token weight budget.

**Reference points on the same device:**
- Mamba1-130M-HF fp16 (final): 24.56 tok/s = 63% of its 38.7 tok/s ceiling
- Mamba2-130M fp16 (final): 17.5 tok/s = 47% of its 37.2 tok/s ceiling
- SmolLM2-135M fp16 (final): 10.46 tok/s = 28.3% of its ceiling
- LFM2.5-350M-Base (this port, baseline): 4.55 tok/s = 30.7% of 14.8 tok/s ceiling

The LFM2.5 baseline already runs the cooperative `gemv_rT_fp32acc` kernel from prior nnopt work — that's why it lands at 30% of ceiling instead of the ~5% baseline that mamba2/mamba1 had against CLBlast HGemm M=1. The remaining headroom is much smaller than the prior ports, but real.

## Benchmark protocol

**One-time setup (after any source change):**
```bash
cd <repo-root>/src/models/lfm2-5-350m
NNOPT_DTYPE=fp16 ./scripts/build.sh --release
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh
```
- `--release` ⇒ CMake `Release` (`-O3`) with `-DNNOPT_DEBUG=` unset (debug macros stripped).
- `NNOPT_DTYPE=fp16` builds `build/fp16/LFM2.5_350M_Base_inference_fp16` and pulls `weights/model.fp16.bin`.

**Per-run command (CANONICAL — every step uses this exact form):**
```bash
NNOPT_DTYPE=fp16 NNOPT_DEBUG_LAYERS=0 ./scripts/run_android.sh \
    "ignored" 32 --temperature 0 --seed 42 --no-eos \
    --token-ids reference/test_input_ids.bin
```
- `--token-ids reference/test_input_ids.bin` — bypass the C++ tokenizer (see "Tokenizer note" below) and feed the canonical 7-token prompt `[1, 1098, 7727, 6619, 963, 779, 730]` = `"<BOS>The teacher worked at the "`. This matches `reference/reference_tokens.json` exactly so the comparison against the PyTorch reference is well-defined.
- `--no-eos` — added in this sprint to override the tokenizer's `eos_token_id=2` (which the model reaches at decode position 1 for short prompts, terminating the run before decode reaches steady state).
- `--temperature 0` ⇒ greedy decode, deterministic.
- 7 prompt tokens + 32 new tokens.

**Metric:** `BENCHMARK decode_tokens_per_sec` from `main.cpp`, defined as `(n_generated - 1) / (total_inference_sec - time_to_first_token_sec)` — steady-state per-token cost, excludes prefill.

**Per-step ritual:** 3 consecutive runs, take **median** decode tok/s. Variance <1% on warm runs.

**Greedy reference output (locked across every step in this file — IDs match bit-for-bit):**
> `<|startoftext|>The teacher worked at the 3rd floor of the building, and the 4th floor of the building was 2 floors higher. The 5th floor was 1 floor lower`

(First 5 generated tokens `[528, 6884, 8913, 803, 779]` = "3rd floor of the" match the PyTorch reference exactly. Token 6 diverges: this port produces 4204 ("building"), reference produces 2254 ("school"). This is fp16 reduction-order drift — the standard signature of this device's GEMV stack and consistent with mamba2's "first divergence at token 6" behavior. Acceptance criterion: every optimization step must reproduce **this** sequence ID-for-ID.)

**Profile mode:**
```bash
adb shell "cd /data/local/tmp/LFM2.5_350M_Base_inference && \
  NNOPT_KERNEL_PROFILE=1 LD_LIBRARY_PATH=/data/local/tmp/LFM2.5_350M_Base_inference/lib:/system/vendor/lib64 \
  ./LFM2.5_350M_Base_inference_fp16 'ignored' 32 --temperature 0 --seed 42 --no-eos --token-ids /data/local/tmp/LFM2.5_350M_Base_inference/test_input_ids.bin"
```
- The queue is created with `CL_QUEUE_PROFILING_ENABLE` in `opencl_context.cpp:41`. `KernelProfiler` (`src/kernel_profiler.{h,cpp}`, copied verbatim from the mamba2 workspace) attaches a `cl_event` to every `clEnqueueNDRangeKernel` and CLBlast `Gemm` enqueue. After `model.generate()` the profiler reads `CL_PROFILING_COMMAND_START`/`END` and prints a per-label breakdown sorted by total GPU time.
- **CAVEAT.** Per-event overhead (`clGetEventProfilingInfo` is blocking) ≈ 10-30 µs of host work. Never quote tok/s from a profile run.

## Tokenizer note (out of scope for this sprint, document for later)

The C++ tokenizer (`src/tokenizer.cpp`) currently has `bos_id_ = 2` but the LFM2 model expects BOS = 1 (token 2 is `<|endoftext|>`). When invoked via `tokenizer.encode("Once upon a time")` it returns `[14142, 57632, 574, 8012]` with no BOS — feeding this to the model produces gibberish (`"Onceuponatimea<|endoftext|>" → 510 510 510...` repeating). **All measurements in this file bypass the C++ tokenizer via `--token-ids reference/test_input_ids.bin`.** Fixing tokenizer encode is a separate workitem; it does not affect benchmark numbers because the `--token-ids` path is the canonical one for cross-port comparison anyway.

## Step 0 — Baseline (3-run median)

Existing fp16 release build, with the pre-existing `gemv_rT_fp32acc` (WG=128 single-output cooperative GEMV) on the M=1 hot path.

| Run | decode tok/s | TTFT s | total inference s |
|---|---|---|---|
| 1 | 4.5486 | 3.9826 | 10.7978 |
| 2 | 4.5522 | 3.8524 | 10.6623 |
| 3 | 4.5413 | 3.8631 | 10.6894 |
| **median** | **4.5486** | **3.8631** | **10.6894** |

Variance 0.24% across runs. Token sequence identical across all three (deterministic ✓).

**Per-token decode time:** 1000 / 4.5486 = **219.8 ms/token**.
**Effective decode BW:** 676 MB / 219.8 ms = **3.07 GB/s = 30.7% of 10 GB/s ceiling**.
**Memory:** 2933 MB peak (weights + 1.5 GB of KV cache pre-allocated for max_position_embeddings=128000 — note: severely over-provisioned for this benchmark).

## Bottleneck census (PROFILED, per 32-token decode at baseline)

```
=== KERNEL PROFILE (env NNOPT_KERNEL_PROFILE=1) ===
label                                total_ms   %total       calls    avg_us
-------------------------------- ------------ -------- ----------- ---------
gemv_m1_K1024_N4608                  2767.469   43.44%         992   2789.79  ← MLP w1+w3
gemv_m1_K1024_N65536                 1223.783   19.21%          31  39476.87  ← lm_head
gemv_m1_K4608_N1024                  1162.097   18.24%         496   2342.94  ← MLP w2
gemv_m1_K1024_N3072                   576.287    9.05%         310   1858.99  ← conv in_proj
gemv_m1_K1024_N1024                   420.125    6.60%         682    616.02  ← q/conv-out/attn-out
gemv_m1_K1024_N512                    116.671    1.83%         372    313.63  ← k/v_proj
embedding                              22.099    0.35%          32    690.59
rmsnorm                                21.119    0.33%        1056     20.00
attn_softmax                            6.873    0.11%         192     35.80
attn_out_PV                             4.909    0.08%         192     25.57
attn_scores_QKt                         4.473    0.07%         192     23.30
attn_qk_rmsnorm                         4.241    0.07%         384     11.04
all conv-helpers (transpose/split/mul/cache_update) ~17 ms total combined
all element-wise (add/silu_mul/etc.)                ~10 ms total combined
=== TOTAL GPU kernel time: 6370 ms / 32 tokens = 199 ms/token ===
```

Wall = 220 ms/tok, GPU active = 199 ms/tok ⇒ **GPU is busy 90% of wall time at baseline.** This is unusually high — the existing GEMV is already cooperative, so host-side dispatch overhead is amortized well. There is much less host-side fat to cut than mamba2 (which started at ~30% GPU active in profile because CLBlast was the bottleneck).

**Per-call efficiency vs roofline:**

| Site | Bytes/call | Roofline @10 GB/s | Actual baseline | % of ceiling |
|---|---|---|---|---|
| K=1024 N=4608 | 9.4 MB | 943 µs | 2790 µs | **34%** |
| K=1024 N=65536 | 128 MB | 12800 µs | 39477 µs | **32%** |
| K=4608 N=1024 | 9.4 MB | 943 µs | 2343 µs | **40%** |
| K=1024 N=3072 | 6.3 MB | 629 µs | 1859 µs | **34%** |
| K=1024 N=1024 | 2.1 MB | 210 µs | 616 µs | **34%** |
| K=1024 N=512 | 1.05 MB | 105 µs | 314 µs | **33%** |

Pattern: every K=1024 site consistently hits **33-34% of ceiling**, irrespective of N. K=4608 hits 40% (better arithmetic density per thread). The per-K consistency is the key signal — it says the bottleneck is inherent to the WG=128 single-output kernel design at K=1024 on this Adreno, not anything about specific shapes.

**95% of decode GPU time is in 6 GEMV shapes. Everything else combined is <1.5%.** This is a near-pure GEMV optimization problem.

## Top-N optimization plan (ranked, with predictions and outcomes)

The plan below lists every lever attempted in this sprint, with the predicted impact recorded **before** measurement and the actual measured outcome. Negative results (regressions) are kept — they document what does not work on Adreno 620, so the next port doesn't waste cycles.

| # | Lever | Predicted | Measured | Status |
|---|---|---|---|---|
| **1** | **K=4608 no4 specialization** (`gemv_m1_k4608_no4`, WG=64, 4 outputs/WG) | 1.05–1.08× | **1.047×** (4.55 → 4.76 tok/s) | ✅ landed |
| **2** | **WG=64 single-output for K=1024** (replace WG=128 in `gemv_rT_fp32acc`) | 1.02–1.05× | **1.025×** (4.76 → 4.88 tok/s) | ✅ landed |
| **3** | **GPU argmax for lm_head** (`kernels/argmax.cl` + `Model::forward_greedy`) | 1.02–1.04× | **1.012×** (4.88 → 4.94 tok/s) | ✅ landed |
| **4** | **`-cl-fast-relaxed-math` build flag everywhere** (Adreno guide §8.2) | 1.01–1.05× | **neutral** (within run variance) | ⚪ kept (no harm, may help future kernels with native math) |
| **5** | **`element_add_inplace` for residual adds** (kill the alloc + copy in residual) | 1.005–1.02× | **1.004×** (4.94 → 4.96 tok/s) | ✅ landed |
| **A** | **K=1024 no4 with WG=64** (4 outputs/WG, single wave) | 1.4–1.7× | **0.27×** (4.55 → 1.23 tok/s) | ❌ reverted — 4.6× regression per call |
| **B** | **K=1024 no4 with WG=128** (4 outputs/WG, two waves) | 1.2–1.4× | **0.71×** (4.55 → 3.24 tok/s) | ❌ reverted — 1.6× regression per call |
| **C** | **K=1024 hardcoded fully-unrolled single-output** (`gemv_m1_k1024`, WG=64, `#pragma unroll 4`) | 1.04–1.10× | **0.88×** (4.94 → 4.33 tok/s) | ❌ reverted — runtime-K loop in `gemv_rT_fp32acc` generates better Adreno IL than `#pragma unroll` over a hardcoded-K loop |

**Cumulative landed: 4.5486 → 4.9587 tok/s = 1.090× = +9.0%.**

### Why the no4 K=1024 attempts regressed (analysis kept on file)

For K=4608 N=1024 (the MLP-w2 site), the no4 specialization wins +57%. Per thread it does 18 vec4 of x and 18 vec4 × 4 outputs = 72 vec4 of W = high arithmetic density per thread. With small N (1024 ÷ 4 = 256 WGs × 64 threads = 16K threads) the thread count is low, but each thread does enough work to amortize launch overhead.

For K=1024 N=4608/65536/3072/etc., the same no4 design lost. Cause: **the existing WG=128 single-output baseline already runs at 589K threads (4608 WGs × 128) for the N=4608 site**, and the no4 design with WG=64 crashes that to 73K threads (1152 × 64). The compiler additionally hits **register pressure** with 4 acc + 4 W vec4 + 1 x vec4 simultaneously live (~25 fp32 regs/thread, exceeding Adreno 620's per-wave VGPR sweet spot), spilling to local memory. The combination of dropped occupancy + spills produced 4.6× per-call slowdown on K=1024.

WG=128 no4 is the obvious "fix" (147K threads, no2 waves per WG → fewer barriers), but it still lost 1.6× — register pressure is still over budget at WG=128. We did not test WG=64 **no2** (2 outputs/WG, less register pressure) as the per-call gains projected from baseline analysis are <5% even in the best case.

The lesson, recorded at the top of this file's section ranking: **on Adreno 620, when your baseline is already a cooperative WG=128 single-output GEMV at K≤1536, the no4 multi-output specialization that wins on K≥1536 turns into a regression at K=1024 because of register pressure. Use no4 only when N is small enough that single-output occupancy would also be low.**

### Why hardcoded fully-unrolled K regressed (analysis kept on file)

`gemv_m1_k1024` was a single-output WG=64 kernel with `K=1024` baked in and `#pragma unroll` over the 4-iteration inner loop. Identical math to the runtime-K `gemv_rT_fp32acc` baseline. Measured **0.88×** of baseline per call.

Suspected cause: the Adreno OpenCL compiler emits different IL for `vload_half4(0, ptr + off)` (used in unrolled form) versus `vload_half4(k4, ptr)` (used in runtime-K form). The runtime-K form receives the `k4` index-times-stride-of-4-fp16 calculation as part of `vload_half4`'s normalization; the offset-pointer form requires an explicit add-then-load. We did not pursue further — the kernel is kept in `kernels/gemv_m1.cl` for re-evaluation under different driver/SDK versions but the dispatcher does **not** select it.

## Step 1 — K=4608 no4 specialization (3-run median)

**Lever.** Add `gemv_m1_k4608_no4` to `kernels/gemv_m1.cl` (WG=64, 4 outputs per WG, 18 vec4-iterations of fp32-accumulated dot products). Wired into `pytorch_linear()` via `run_gemv_m1_no4()` in `src/utils.cpp`; eligibility predicate is `M==1 && K==4608 && N%4==0`. Falls back to the existing `gemv_rT_fp32acc` kernel for any K not handled.

| Run | decode tok/s | tokens deterministic |
|---|---|---|
| 1 | 4.7632 | reference |
| 2 | 4.7316 | ✓ exact |
| 3 | 4.7645 | ✓ exact |
| **median** | **4.7632** | ✓ |

**Per-call delta (from profiled run after Step 1):** K=4608 N=1024 dropped 2343 → 1485 µs/call = **1.58× faster**. This site contributes 18% of GPU time, so the kernel-time saving × 18% ≈ +6.5%. The actual run-rate gain is +4.7% — the difference is because Step 1 only optimizes K=4608 sites; the K=1024 sites (which dominate at 80% of GPU time) are unchanged.

**Step 1 / Step 0:** 1.047× (predicted band 1.05–1.08×; actual at the low end of the band).

## Step 2 — WG=64 single-output for K=1024 (3-run median)

**Lever.** Change `kGemvSrc` (the inline kernel for `gemv_rT_fp32acc`) from `#define GEMV_WG 128` to `GEMV_WG 64`, and update the host dispatch from `(N*128, 128)` to `(N*64, 64)`. The kernel body is unchanged (runtime-K loop, vec4 vload_half4, fp32 accumulator, `__local`-mem tree reduce). Hypothesis: at WG=64 = exactly one Adreno A6xx wave, the tree-reduce barrier becomes intra-wave (effectively free), and each thread now does 4 iterations instead of 2, giving better arithmetic density.

| Run | decode tok/s | tokens deterministic |
|---|---|---|
| 1 | 4.8687 | ✓ exact |
| 2 | 4.8812 | ✓ exact |
| 3 | 4.8890 | ✓ exact |
| **median** | **4.8812** | ✓ |

**Per-call delta:** K=1024 N=4608 dropped 2790 → 2680 µs/call = **1.04× faster**. K=1024 N=65536: 39477 → 37978 µs (1.04×). K=1024 N=3072: 1859 → 1779 µs (1.04×). Roughly uniform across all K=1024 shapes.

**Step 2 / Step 1:** 1.025× (predicted band 1.02–1.05×; actual mid-band).

## Step 3 — GPU-side argmax for lm_head (3-run median)

**Lever.** Lm_head produces `[seq_len, 65536]` fp16 logits. At greedy temperature=0 with rep_penalty=1, the host only needs the argmax of the final row. Old path: `clEnqueueReadBuffer(128 KB blocking)` + `for i in [0,65536): logits_f32[i] = nnopt_f16_to_f32(...)` + `std::max_element(65536)`. Combined cost ~5 ms / token of host work.

New path: `kernels/argmax.cl` with two kernels (`argmax_block` and `argmax_final`), both 64-thread WGs. Pass 1 reduces 65536 logits to 64 (max_val, max_idx) pairs; Pass 2 reduces those 64 to a single int32 written to a 4-byte device buffer. Host reads 4 bytes blocking. Wired through `Model::forward_greedy()` (new method) gated on `sampler_config.temperature <= 0 && repetition_penalty == 1.0` in `Model::generate()`. Pure greedy fast path; non-greedy still goes through the full readback + sampler.

| Run | decode tok/s | tokens deterministic |
|---|---|---|
| 1 | 4.9426 | ✓ exact |
| 2 | 4.9229 | ✓ exact |
| 3 | 4.9374 | ✓ exact |
| **median** | **4.9374** | ✓ |

**Step 3 / Step 2:** 1.012× (predicted band 1.02–1.04×; actual at the low end). The host-side argmax + fp16→fp32 was apparently faster on the Razr 2020's ARM cores than the rough 5 ms estimate — the actual host overhead it eliminated was closer to 2 ms/tok.

## Step 4 — `-cl-fast-relaxed-math` build flag (3-run median, neutral result)

**Lever.** Per Adreno OpenCL programming guide §8.2: `-cl-fast-relaxed-math` enables fast-math (allowed reordering, ignored NaN/inf semantics, native reciprocal/divide). Required for `native_*` math substitution by the compiler. Applied to (a) the `OpenCLContext::build_program(...)` path so every layer kernel picks it up, and (b) the inline `clBuildProgram` calls for `gemv_rT_fp32acc` and `gemv_m1.cl` in `src/utils.cpp`.

| Run | decode tok/s | tokens deterministic |
|---|---|---|
| 1 | 4.9245 | ✓ exact |
| 2 | 4.9176 | ✓ exact |
| 3 | 4.9010 | ✓ exact |
| **median** | **4.9176** | ✓ |

**Step 4 / Step 3:** 0.996× — within run-to-run variance. The Adreno compiler must already do these optimizations for our access pattern (no transcendentals, no division on the GEMV path). **Kept anyway**: the flag is essentially free at this point, and any future kernel that does use `native_rsqrt` / `native_recip` will benefit automatically.

## Step 5 — `element_add_inplace` for residual adds (3-run median)

**Lever.** The two residual adds per layer (`hidden = hidden + op_out`, `hidden = hidden + mlp_out`) called `element_add()`, which **allocates a fresh output buffer + copies a → out + adds b**. There's already an `element_add_inplace()` helper in `src/utils.cpp` that just does `a += b` directly. Switching to it kills 32 buffer allocations + 32 copy-buffer roundtrips per token.

| Run | decode tok/s | tokens deterministic |
|---|---|---|
| 1 | 4.9587 | ✓ exact |
| 2 | 4.9500 | ✓ exact |
| 3 | 5.0782 | ✓ exact |
| **median** | **4.9587** | ✓ |

**Step 5 / Step 4:** 1.008× over Step 4 baseline; **1.004× over Step 3** (the meaningful comparison since Step 4 was neutral). Run 3's outlier (5.08) is thermal — first measurement of the cycle when the device was warm. Median of 4.9587 is conservative.

**Final / Baseline:** **4.5486 → 4.9587 tok/s = 1.090× = +9.0%.**

## ── Session 2 (2026-05-06) — break the buffer ceiling, hit 10 tok/s ──

The session-1 BENCHMARK above used "10 GB/s realistic" as the ceiling. **That number was wrong** — a `--bw-probe` STREAM port from the qwen2.5-0.5B workspace measured Adreno 620's actual streaming-read BW on this exact Razr 2020:

```
$ NNOPT_BW_PROBE=1 ./scripts/run_android.sh "x" 1
STREAM[buf]: 256 MB read → 7.85 GB/s (best of 5)
STREAM[img]: 256 MB read via image2d → 13.46 GB/s (best of 5)
=== Practical roofline for LFM2.5-350M-Base (676 MB/token fp16) ===
  Buffer-cache ceiling: 7.85 GB/s → max 11.62 tok/s
  Texture-cache ceiling: 13.46 GB/s → max 19.90 tok/s
```

**Key finding:** image (texture) reads are **1.71× faster** than buffer reads on this device. Buffer ceiling is only 11.6 tok/s — at session-1 final 4.96 tok/s we were already at 43% of the buffer ceiling. **10 tok/s is unreachable on the buffer path; it requires `image2d_t` W reads.** The 14 GB/s figure quoted by the qwen2.5 BENCHMARK was on a slightly different device or earlier in the device's life.

## Step 6 — Persistent activation buffers across all layer classes (3-run median)

**Lever.** ~205 `clCreateBuffer` calls per token in `src/layers/{attention,convolution,mlp,operator_norm,embedding}.cpp` were collapsed into lazy-once allocations stored as class members. Convolution had 10 buffers per call (in_proj, in_proj_T, B, C, X, Bx, conv_out_T, Bx_seq, y_T, y_seq, out); Attention had 8 (q, k, v, q_norm_tmp, k_norm_tmp, scores, attn_out, proj); Mlp had 3; OperatorNorm had 1; Embedding now persists buf_out_ + buf_input_ids_; Model persists buf_logits_. Forward output is a BORROWED handle — caller must NOT release. All `clReleaseMemObject(...)` calls on intermediate hand-offs in `Model::forward` and `Model::forward_greedy` removed.

| Run | decode tok/s | tokens deterministic |
|---|---|---|
| 1 | 5.0194 | ✓ |
| 2 | 5.0416 | ✓ |
| 3 | 5.4034 | ✓ (thermal warm) |
| **median** | **5.0416** | ✓ |

**Step 6 / Step 5 (4.9587):** 1.017× — smaller than predicted +12-18% because at session 1 GPU was already 90% of wall (host overhead was only ~21 ms/tok of which clCreateBuffer/Release was a small fraction). Worth landing anyway as a foundation for Step 7.

## Step 7 — `image2d_t` W reads with auto-tiled lm_head (3-run median, BIG WIN)

**Lever.** Every K=1024 and K=4608 W matrix is wrapped at first use as an `image2d_t` view of its own backing buffer (`cl_qcom_create_buffer_from_image` / `cl_khr_image2d_from_buffer`) — same DRAM, alternative L1 access path. New kernels `gemv_m1_k1024_no4_img`, `gemv_m1_k1024_no2_img`, `gemv_m1_k4608_no4_img` use `read_imageh(W_img, sampler, (int2)(pix, n))` with `CL_RGBA / CL_HALF_FLOAT` (4 fp16/pixel). lm_head N=65536 exceeds Adreno 620's `CL_DEVICE_IMAGE2D_MAX_HEIGHT=16384`, so `get_or_create_w_image()` falls into the tiled path automatically (4 sub-images of 16384 rows each). Single `pytorch_linear()` call site routes through the new `run_gemv_m1_image()` dispatcher before falling back to the buffer paths.

| Run | decode tok/s | tokens deterministic |
|---|---|---|
| 1 | 10.2065 | ✓ |
| 2 | 10.1759 | ✓ |
| 3 | 10.3781 | ✓ |
| profile run | 10.3526 | ✓ |
| **median (warm)** | **10.2065** | ✓ |

**Step 7 / Step 6 (5.0416):** **2.024×.** **Step 7 / Session-1 baseline (4.5486):** **2.244×.**
**Per-token decode time:** 1000 / 10.21 = **97.9 ms/token** (down from 220 ms at session-1 baseline).
**Effective decode BW:** 676 MB / 97.9 ms = **6.91 GB/s = 51% of texture-cache ceiling.**

**Per-call profile delta (image-cache vs buffer-cache):**

| Site | Step 6 buffer | Step 7 image | Speedup | Notes |
|---|---|---|---|---|
| K=1024 N=4608 (MLP w1/w3) | 2680 µs | **1153 µs** | 2.32× | Single image, 4 outputs/WG |
| K=1024 N=65536 (lm_head) | 37978 µs | **16109 µs** total (4 × ~4027 µs/tile) | 2.36× | Tiled into 4 sub-images |
| K=4608 N=1024 (MLP w2) | 1485 µs | **806 µs** | 1.84× | no4_img |
| K=1024 N=3072 (conv in_proj) | 1779 µs | **778 µs** | 2.29× | no4_img |
| K=1024 N=1024 (q/conv-out/attn-out) | 597 µs | **280 µs** | 2.13× | no4_img |
| K=1024 N=512 (k/v) | 307 µs | **155 µs** | 1.98× | no4_img |

**Critical surprise: K=1024 no4 worked perfectly with the image path.** In session 1, K=1024 no4 with the BUFFER path regressed 4.6× due to register pressure. The image path uses `read_imageh` + `convert_float4` — a different code path that apparently doesn't hit the same VGPR constraint. The 25 fp32 regs/thread of the no4 layout (4 acc + 4 W vec4 + 1 x vec4) fits the wave's register file when reads come from the texture engine. **This is the structural reason image is the only path to 10 tok/s on Adreno 620 — buffer no4 is register-bound, image no4 isn't.**

Token IDs across 4 runs are **bit-identical** to the session-1 reference text. No fp16 drift — image vs buffer reads return the same bytes (just via a different cache path), so reduction order is unchanged.

## Step 8 — Live token streaming (3-run median, perf-neutral)

**Lever.** Ported the `TokenCallback` pattern from the qwen2.5-0.5B port. `Model::generate()` now takes an `on_token = std::function<void(int32_t, const std::vector<int32_t>&)>` parameter and invokes it after every sampled token (both prefill-completed first token and every decode-loop token). `main.cpp` builds a closure that:
1. Decodes the full ids buffer to a string,
2. Compares against the running `emitted_text` buffer,
3. Prints only the byte-delta with `std::flush`,
4. Updates `emitted_text` to the new full text.

The byte-delta logic handles multi-byte UTF-8 / CJK gracefully — partial codepoints stay buffered until the next token completes them. Per-token "Generated token: N" stderr noise was gated behind `NNOPT_DEBUG_LAYERS=1` (default off) — at 10 tok/s each fprintf is a measurable syscall in the critical path.

The `Adreno image2d limits: ...` one-time init log in `get_or_create_w_image()` was also gated behind the same flag — without that gate, it interleaved with the first streamed token on the user's terminal.

| Run | decode tok/s | tokens deterministic |
|---|---|---|
| 1 | 10.2855 | ✓ |
| 2 | 10.2526 | ✓ |
| 3 | 10.2679 | ✓ |
| **median** | **10.2679** | ✓ |

**Step 8 / Step 7 (10.2065):** 1.006× — within run-to-run variance (the `tokenizer.decode(all_ids)` per-token call adds ~50 µs of host work but happens fully in parallel with the next forward(), so it's free at this throughput).

**Live demo at 96 generated tokens** (`./scripts/run_android.sh "ignored" 96 --temperature 0 --seed 42 --no-eos --token-ids reference/test_input_ids.bin`):
> `<|startoftext|>The teacher worked at the 3rd floor of the building, and the 4th floor of the building was 2 floors higher. The 5th floor was 1 floor lower than the 4th floor, and the 6th floor was 1 floor lower than the 5th floor. If the 7th floor was 2 floors lower than the 6th floor, and the 8th floor was 1 floor lower than the 7th floor, and the`

Words appear at ~10/sec — comfortably faster than reading speed.

## Final state (session 2 close)

| Metric | Session-1 baseline | Session-1 final (Step 5) | Session-2 final (Step 7) | Δ vs session-1 baseline |
|---|---|---|---|---|
| Decode tok/s | 4.5486 | 4.9587 | **10.2065** | **2.244×** |
| Decode ms/token | 219.8 | 201.7 | **97.9** | −121.9 ms |
| Effective decode BW | 3.07 GB/s | 3.35 GB/s | **6.91 GB/s** | +3.84 GB/s |
| % of 14 GB/s buffer ceiling | 22% | 24% | 49% (effectively bypassing it via image) | — |
| % of 13.46 GB/s image ceiling | — | — | **51%** | — |
| % of session-1 mistakenly-quoted "10 GB/s" ceiling | 30.7% | 33.5% | 69% | — |
| Tokens deterministic | ✓ | ✓ | ✓ | ID-for-ID matched at every step |

**Reference utilization comparison (% of measured 14 GB/s buffer / 13.46 GB/s image ceiling):**

| Port | Final tok/s | Per-token weight | Effective BW | % of texture ceiling |
|---|---|---|---|---|
| Mamba1-130M-HF | 24.56 | 258 MB | 6.34 GB/s | 47% |
| **LFM2.5-350M-Base (this port)** | **10.21** | 676 MB | **6.91 GB/s** | **51%** |
| Qwen2.5-0.5B | 9.02 | 942 MB | 8.50 GB/s | 63% |
| Mamba2-130M | 17.50 | 269 MB | 4.71 GB/s | 35% |

LFM2.5 lands at 51% of the texture ceiling — better than Mamba1 (47%), worse than Qwen2.5 (63%). Final perf-per-byte is healthy; further gains would need int8 quantization (halves byte budget, doubles effective tok/s).

## Final state

| Metric | Baseline | Final | Δ |
|---|---|---|---|
| Decode tok/s | 4.5486 | 4.9587 | **+9.0%** |
| Decode ms/token | 219.8 | 201.7 | **−18.1 ms** |
| Effective decode BW | 3.07 GB/s | 3.35 GB/s | +0.28 GB/s |
| % of 10 GB/s ceiling | 30.7% | 33.5% | +2.8 pp |
| Tokens deterministic | ✓ | ✓ | matched ID-for-ID across all 5 steps |

## Comparison with prior ports on the same device

| Model | Baseline | Final | × | % of own ceiling |
|---|---|---|---|---|
| Mamba1-130M-HF | 1.80 tok/s | 24.56 tok/s | 13.6× | 63% |
| Mamba2-130M    | 1.86 tok/s | 17.5 tok/s  | 9.4× | 47% |
| SmolLM2-135M   | ~0.4 tok/s | 10.46 tok/s | 26× | 28% |
| **LFM2.5-350M-Base (this port)** | **4.55 tok/s** | **4.96 tok/s** | **1.09×** | **33%** |

The LFM2.5 baseline starts much closer to its ceiling than any of the prior ports — at 30.7% on day-zero, it was already past SmolLM2's *final* 28%. The +9% sprint pushes us to 33.5%, a typical fraction-of-ceiling for an Adreno 620 LLM port.

The reason the multiplicative speedup is small here is structural, not procedural:
1. **The pre-existing `gemv_rT_fp32acc` kernel was already cooperative + WG=128 + vec4.** Mamba2's baseline ran CLBlast HGemm M=1 (single-thread-per-output) and so had a 7.8× free win sitting in front of it (Step 4+6 in the mamba2 BENCHMARK). LFM2.5 starts where mamba2 finished its M=1 decode work.
2. **The dominant K is 1024.** The no4 multi-output trick that saved 1.6× on K=1536 (mamba2) and on K=4608 (here) does NOT apply at K=1024 on Adreno 620 — register pressure regresses the per-call time by 1.6× to 4.6× depending on WG size (Levers A and B above). Tested both WG=64 and WG=128; both regressed.
3. **No fp16-recurrent state to compress.** Mamba2's late wins came from packing the SSM recurrent state more aggressively. LFM2.5 has only conv state (41 KB total per step) — already negligible.

What's left on the table:
- **`image1d_buffer_t` for the lm_head W**: untested. Adreno's texture cache is a separate L1 path. Mamba2 found it neutral on K=768, but LFM2.5's lm_head is a one-shot 128 MB streaming read at the end of every forward — different cache profile. Predicted: 1.05–1.10× if the texture engine has higher steady-state BW than the buffer cache for cold streaming reads. Risk: low.
- **Subgroup reduction** (`sub_group_reduce_add`, Adreno guide §9.2): replace the `__local`-mem tree reduce in `gemv_rT_fp32acc` with an HW-accelerated subgroup reduce. Predicted 1.01–1.03× since the tree reduce is not the dominant cost (the K-loop is). Risk: low; requires pragma `cl_khr_subgroups` query.
- **Persistent activation buffers** (eliminate per-call `clCreateBuffer` inside `Attention::forward` / `Convolution::forward` / `Mlp::forward`): predicted 1.01–1.02× since GPU-active is already 90% of wall.
- **int8 quantization**: project rule defers; would 2× the BW budget instantly (676 MB → 338 MB → 14.8 → 29.6 tok/s ceiling).

## ── Session 3 (2026-05-07) — past Step 7, push toward 15 tok/s ──

User goal: 15 tok/s. After Step 7 we sit at 51% of the 19.9 tok/s texture roofline; 15 tok/s = 75%. Plan attacks the two biggest remaining levers identified in the post-Step-7 profile: host-side blocking readback (Step 10) and K=1024 GEMV underutilization at 60% of texture ceiling vs K=4608 at 85% (Step 11).

## Step 9 — post-Step-7 baseline profile (informational)

Profile run with `NNOPT_PROFILE=1 NNOPT_KERNEL_PROFILE=1`. Wall = 95.9 ms/token (10.42 tok/s in profile). Total GPU kernel time = 83.6 ms/token. Host gap = 12.3 ms/token (down from baseline's 21 ms; the chained-decode work in Step 10 targets this 12 ms).

Per-call BW utilization (vs 13.46 GB/s texture ceiling):
- K=1024 N=4608: 1167 µs/call → 8.1 GB/s = **60%**
- K=1024 N=65536 (per tile): 4078 µs → 7.85 GB/s = **58%**
- K=4608 N=1024 (no4): 822 µs → 11.4 GB/s = **85%** ← ceiling
- K=1024 N=3072: 789 µs → 8.0 GB/s = **59%**
- K=1024 N=1024: 284 µs → 7.4 GB/s = **55%**
- K=1024 N=512: 157 µs → 6.7 GB/s = **50%**

**Key signal:** K=4608 hits 85% with the same no4 kernel design that K=1024 stalls at 60% on. The difference is per-thread iteration count: K=4608 no4 = 18 K-iter × 4 outputs = 72 dot products per thread; K=1024 no4 = 4 K-iter × 4 outputs = 16. K=1024 has too few in-flight reads to saturate the texture engine. Step 11 attacks this.

## Step 10 — chained decode + async readback (3-run median, neutral)

**Lever.** Eliminate the blocking `clEnqueueReadBuffer(argmax_out_idx_, CL_TRUE, ...)` at the end of `Model::forward_greedy()` (model.cpp:354). Add a new `forward_greedy_chained_enqueue(start_pos)` that mirrors `forward_greedy` but (a) the embedding reads its single token id from the *previous* iteration's argmax output via `clEnqueueCopyBuffer(argmax_out_idx_ → buf_input_ids_, 4 bytes)` instead of from a host-supplied vector, (b) skips the per-call `clCreateSubBuffer` for the argmax row (offset=0 at seq_len=1 means logits_buf_ itself is the row), and (c) does NOT block on a host readback. `Model::generate()` runs a ping-pong loop: enqueue forward N+1, enqueue *non-blocking* readback for slot (N+1)&1, `clFlush`, then `clWaitForEvents` on slot N&1's readback to consume the int32 from the previous iteration. The host stays one step ahead in enqueue and one step behind in readback, hiding the readback latency behind the next forward's enqueue. Reference: openelm-270m's Step 9 pattern.

| Run | decode tok/s | tokens deterministic |
|---|---|---|
| 1 | 9.91 | ✓ |
| 2 | 9.67 | ✓ |
| 3 | 10.24 | ✓ |
| **median** | **9.94** | ✓ |

**Step 10 / Step 7:** **0.974×** — within run-to-run variance, effectively neutral. The post-Step-7 profile shows GPU kernel time (83.6 ms/token) fully accounts for almost all of the 95.9 ms wall — host overhead is ~12 ms/token but is already overlapped with GPU work via the in-order queue's natural pipelining. The blocking readback was waiting for *GPU work to finish*, not for host work, and that wait is unavoidable. **Predicted +1.20–1.30× was wrong** — the openelm-270m gain was driven by a much higher relative host stall there. **Kept anyway**: code is clean, tokens match, and it interacts correctly with subsequent steps; reverting buys nothing.

## Step 11 — `gemv_m1_k1024_no8_img` (3-run median, BIG WIN)

**Lever.** New OpenCL kernel `gemv_m1_k1024_no8_img` in `kernels/gemv_m1.cl` — same image-backed K=1024 path as `_no4_img`, but produces **8 outputs per WG** instead of 4. Doubles per-thread arithmetic density (32 dot products + 36 vec4 reads per thread vs 16 + 20), giving the texture engine more in-flight reads to hide latency. Key insight: the BUFFER no4 path failed at K=1024 due to register pressure (Lever A in the Top-N table), but the IMAGE path uses the texture engine — a separate pipeline that doesn't compete for VGPRs the same way. Step 7 already established that no4 image works at K=1024 where no4 buffer doesn't; no8 image extends that headroom further.

Dispatcher in `src/utils.cpp::run_gemv_m1_image` selects no8 when `K==1024 && N>=2048 && N%8==0`. Threshold N≥2048 avoids a small regression at N=512 / N=1024 where the WG count drops below the device's latency-hiding threshold (no8 → 8192 threads at N=1024 is too few; no4 keeps 16384). Larger sites win:

| Site | Step 7 (no4) | Step 11 (no8) | Speedup | New BW utilization |
|---|---|---|---|---|
| K=1024 N=4608 (MLP w1/w3) | 1167 µs | **1033 µs** | **1.13×** | **68%** ↑ from 60% |
| K=1024 N=65536 (lm_head, per tile) | 4078 µs | **3428 µs** | **1.19×** | **70%** ↑ from 58% |
| K=1024 N=3072 (conv in_proj) | 789 µs | **718 µs** | **1.10×** | **65%** ↑ from 59% |
| K=1024 N=1024 (q/conv-out/attn-out) | 284 µs | 290 µs | 0.98× | unchanged (uses no4) |
| K=1024 N=512 (k/v_proj) | 157 µs | 174 µs | 0.90× | unchanged (uses no4) |
| K=4608 N=1024 (MLP w2) | 822 µs | 732 µs | 1.12× | unchanged kernel (warm-thermal) |

5-run measurement (post-build warmup):

| Run | decode tok/s | tokens deterministic |
|---|---|---|
| 1 | 11.21 | ✓ |
| 2 | 11.21 | ✓ |
| 3 | 10.75 | ✓ (thermal dip) |
| 4 | 11.25 | ✓ |
| 5 | 11.17 | ✓ |
| **median** | **11.21** | ✓ |

**Step 11 / Step 7 (10.21):** **1.098×.** Tokens are bit-identical to the locked Session-2 reference. Per-token decode time: 1000 / 11.21 = **89.2 ms/token** (down from 97.9). Effective decode BW: 676 MB / 89.2 ms = **7.58 GB/s = 56% of texture-cache ceiling** (up from 51%).

**Why register pressure didn't bite at no8:** the K=1024 no4 buffer regression was diagnosed (line 171) as 25 fp32 regs/thread spilling. no8 image holds 8 fp32 acc + transient W vec4 + 1 x vec4 ≈ 50–60 fp32-equivalent regs. The IMAGE path's `read_imageh + convert_float4` apparently keeps W loads in texture-engine pipeline registers (separate from compute VGPRs), so the 8-output design fits where 4-output buffer didn't.

## Step 11 ledger

| Metric | Step 7 (session-2 final) | Step 11 (session-3) | Δ |
|---|---|---|---|
| Decode tok/s | 10.21 | **11.21** | **+9.8%** |
| Decode ms/token | 97.9 | 89.2 | −8.7 ms |
| Effective decode BW | 6.91 GB/s | 7.58 GB/s | +0.67 GB/s |
| % of 13.46 GB/s texture ceiling | 51% | **56%** | +5 pp |
| Tokens deterministic | ✓ | ✓ | ID-for-ID matched |

**Cumulative session-3:** 10.21 → 11.21 = **1.098× = +9.8%.**

## Step 12 — fused MLP (w3 GEMV + silu_mul) — sub-noise, NOT shipped

**Lever.** New OpenCL kernel `gemv_m1_k1024_no8_silufused_img` in `kernels/mlp_fused.cl` (separate cl_program — see *Compiler interaction caveat* below). Replaces the second MLP GEMV + `silu_mul` with a single kernel that reads `buf_gate_[n]` (= w1·x from a prior pytorch_linear), reads W3 via image2d, and writes `silu(buf_gate_[n]) * (W3·x)` back into buf_gate_. Eliminates one kernel launch per MLP layer (16/token) and the buf_up_ intermediate buffer entirely.

**Compiler interaction caveat (recorded for posterity).** First attempt placed the silufused kernel inside `kernels/gemv_m1.cl`. Result: catastrophic 13× per-call regression on the no8 kernels (no8 K=1024,N=4608 went 1032 → 13217 µs; whole-program tok/s dropped from 11.21 → 1.32). Hypothesis: Adreno OpenCL compiler runs cross-kernel global register allocation across all kernels in a cl_program, and the silufused kernel's transient state (8 acc + native_exp + gate read/write) pushed the no8 kernels' allocation over the per-wave register file budget, causing them to spill. **Lesson: keep optimizer-sensitive kernels in their own cl_program on Adreno.** Moved silufused to `kernels/mlp_fused.cl` with its own `ensure_mlp_fused_program()`; no8 kernels recovered to 1032 µs.

**Per-call profile (post-fix, separate program):**
- `gemv_m1_K1024_N4608_no8_img` (w1 GEMV alone): 1032 µs/call
- `gemv_m1_K1024_no8_silufused_img` (w3 + silu, fused): **881 µs/call** ← *slightly faster than the bare no8 GEMV*
- Net per layer: 1032 + 881 = 1913 µs vs unfused 2 × 1033 + 15 silu = 2081 µs → ~170 µs saved per MLP × 16 = **2.7 ms/token GPU saved**

**Wall-clock measurement (12 runs across two clean rebuilds):** 10.87, 10.91, 10.98, 10.98, 10.99, 10.99, 11.00, 11.00, 11.02, 11.03, 11.03 → median **10.99 tok/s**.

| State | median tok/s |
|---|---|
| Step 11 (no fused) | 11.21 |
| Step 12 (fused enabled) | 10.99 |
| Step 12 code present, fused disabled in mlp.cpp | 11.21 |

**Status:** ❌ NOT shipped (sub-noise to slight regression on Adreno 620; 2.7 ms GPU savings did not propagate to wall-clock, suggesting the in-order queue's natural pipelining was already absorbing the silu_mul cost). Kernel + program plumbing kept in tree (`kernels/mlp_fused.cl`, `pytorch_linear_silu_gate_fused()` in utils) — flip the `if (false)` guard in `Mlp::forward()` to re-enable for future driver/SDK comparison.

## Session 3 final state

| Metric | Step 7 baseline (session-2 final) | Step 11 final | Δ |
|---|---|---|---|
| Decode tok/s | 10.21 | **11.21** | **+9.8%** |
| Decode ms/token | 97.9 | 89.2 | −8.7 ms |
| Effective decode BW | 6.91 GB/s | 7.58 GB/s | +0.67 GB/s |
| % of 13.46 GB/s texture ceiling | 51% | **56%** | +5 pp |
| Tokens deterministic | ✓ | ✓ | ID-for-ID matched |

## ── Session 4 (2026-05-07) — Adreno-guide-driven cleanup + lever #4 ──

After reading the Snapdragon OpenCL Programming Guide (80-NB295-11 Rev. C) end-to-end, surveyed five untried levers (recordable_queues, subgroup_shuffle reduce, onchip_global_memory, max_constant_size on x, full-wave WG=128 + shuffle reduce). Profiled the post-Step-11 hot path then attacked the bottleneck.

### Step 13 — `cl_qcom_recordable_queues` proven, but not integrated

Ported the recordable-queues probe to lfm2 (`run_record_probe()` in main.cpp + `probe_noop` kernel in gemv_m1.cl). Triggered by `NNOPT_RECORD_PROBE=1`. **Recipe that works on Adreno 620 (driver E031.37.12.07):** create queue with `clCreateCommandQueue(ctx, dev, CL_QUEUE_RECORDABLE_QCOM /*0x40000000*/ alone, &err)` — NOT combined with `CL_QUEUE_PROFILING_ENABLE`. The earlier (qwen2.5) investigation comment claiming "no candidate worked" was stale — bit-30-alone DOES work.

```
Record: WIN attempt 400 (probe_q=0xb40000789a490f30)
Baseline: 10000 sequential dispatches → 186.92 ms (18.69 µs/dispatch)
Replay:   10000 dispatches            →  85.70 ms ( 8.57 µs/dispatch)
Record: speedup = 2.18×
```

Did NOT integrate into `Model::generate()` after analysis showed expected wall-clock gain ≈ 0.5–2 ms/token. Reason: prior BENCHMARK Step 10 + Step 12 evidence already established that on this device's in-order queue, host CPU is overlapped with GPU; saving CPU dispatch time doesn't shorten the wall. Confirmed independently in this session (Step 14 + Step 15 below). Probe + plumbing kept in tree for future int8/onchip_global_memory work that needs a recording context.

### Step 14 — Recordable-ready cleanup of chained-decode forward (PHASE 2A)

Eliminated all 4 non-NDRange enqueues from `forward_greedy_chained_enqueue` (only `clEnqueueNDRangeKernel` is recordable per guide §9.1.3). Specifically:
- `embedding.cpp:109` — bound `token_ids_dev` directly as kernel arg 0 when `dev_offset_bytes==0`, eliminating the per-token `clEnqueueCopyBuffer(token_ids_dev → buf_input_ids_)`.
- `attention.cpp:374` — q/k_layernorm `clEnqueueCopyBuffer(tmp → buf)` removed by passing `buf` as both input AND output to `rmsnorm_forward` (in-place is safe: each work-item owns disjoint columns; the barrier between read+reduce and write phases is preserved).
- `attention.cpp:429,431` — KV cache writes converted from `clEnqueueCopyBuffer` to a new `kv_write` NDRange kernel in `kernels/attention.cl` (one work-item per element of the row, reads `start_pos` as a scalar arg). Prefill (`seq_q>1`) keeps `clEnqueueCopyBuffer` because one big copy beats `seq_q` per-row launches there.

| Run | decode tok/s | tokens deterministic |
|---|---|---|
| 1 | 11.1841 | ✓ exact |
| 2 | 11.0085 | ✓ exact |
| 3 | 11.3356 | ✓ exact |
| **median** | **11.1841** | ✓ |

**Step 14 / Step 11:** 0.998× — neutral. Expected: refactoring API calls without changing kernel work. Value is unlocking recording-readiness for future levers.

### Step 15 — Conv block fusion for decode (PHASE 2B-pivot)

Profile showed conv layers fired 9 dispatches each (`copy_transpose / split_chunk3 / pointwise_mul / conv1d_causal_with_cache / copy_transpose_back_state / update_state / mul_C / copy_transpose_back` + the bracketing GEMVs), totaling ~22 ms GPU / 32 tokens. For decode (seq_q=1) the four transposes are no-ops (single-token tensors don't reshape) and the rest are per-channel work on H=1024 elements.

Added `conv_block_decode` kernel in `kernels/convolution.cl` — one work-item per channel does B/C/X read + Bx + conv1d (with state read+update) + multiply by C, all in one launch. Math is bit-identical to the unfused path (verified: tokens match ID-for-ID across all 5 runs against the locked reference).

Profile delta for the conv layer:

| Site | Pre-Step-15 | Post-Step-15 | Notes |
|---|---|---|---|
| 7 conv helper kernels combined | ~22 ms / 32 tokens | n/a (skipped at decode) | seq_q=1 skips them |
| `conv_block_decode` (new) | n/a | 9.85 ms / 32 tokens | 31.8 µs/call vs ~71 µs unfused sum |

| Run | decode tok/s | tokens deterministic |
|---|---|---|
| 1 | 11.2564 | ✓ |
| 2 | 10.8978 | ✓ |
| 3 | 11.2305 | ✓ |
| 4 | 11.3384 | ✓ |
| 5 | 11.2381 | ✓ |
| **median** | **11.2381** | ✓ |

**Step 15 / Step 11:** 1.002× — wall-neutral but GPU active time reduced by ~0.4 ms/tok. Confirms (third time independently this port) that on Adreno 620 with the in-order command queue, dispatch-count cuts and CPU savings don't propagate to wall-clock — the queue's natural pipelining already hides them. **Kept anyway** because: (a) the conv path is structurally cleaner, (b) less L2 traffic between intermediate buffers, (c) lower power consumption, (d) prerequisite for any future block-level recording-queue capture.

### Step 16 — `sub_group_reduce_add` in image GEMVs ❌ NOT shipped

Tried replacing the `__local`-mem tree reduce in the four image GEMV kernels (no2_img, no4_img K=1024, no8_img K=1024, no4_img K=4608) with `sub_group_reduce_add` per Snapdragon Programming Guide §9.2.2 ("Adreno has hardware support for the reduction functions, much faster than doing reduction through local memory").

**Two regressions hit:**

1. **Without `qcom_reqd_sub_group_size("full")`:** tokens diverged at the very first generated token ("The teacher worked at the **0**" instead of "**3rd**"). Cause: Adreno 620 default subgroup_size = 32 (half-wave); `sub_group_reduce_add` only sums 32 of 64 lanes, halving the GEMV output. Decode dropped to 5.07 tok/s.

2. **With `qcom_reqd_sub_group_size("full")`:** tokens correct ✓ but decode collapsed to 1.51 tok/s (**7.4× slower** than baseline). Cause: forcing full-wave on the no8_img kernel (~50 fp32-equivalent regs/thread for 8 acc + 9 fp16x4 in flight) either spilled registers or halved active lanes when the actual full wave width was 128 — exactly the failure mode the BENCHMARK Lever A/B section warned about.

Reverted everything (kernel bodies + `-cl-std=CL2.0` build flag). The existing tree-reduce stays — at WG=64 == one wave (or even half-wave), the inter-iteration barriers compile to ~no-ops on Adreno, and the tree-reduce is essentially free.

**Lesson recorded for later:** wave-shuffle reduce on Adreno 620 image GEMVs requires either (a) a two-level reduce that's subgroup-size-agnostic (sub_group_reduce_add for the inner subgroup + __local for cross-subgroup combine), or (b) confirmation the kernel's compiler-picked wave size matches WG, then no attribute needed. Not worth the complexity given the alternatives below.

### Step 17 — `__constant` x with `max_constant_size` in `gemv_m1_k1024_no8_img` ✅ landed

Adreno guide §6.4 / §7.1.3: an array passed as `__constant` with the `max_constant_size(N)` attribute is promoted to on-chip constant memory, which "can broadcast into ALUs in no time for fast ALU computing. All other memories (global, local, and private) must go through the lengthy load/store path."

`x` in the GEMV is exactly the right pattern: 1024 fp16 = 2048 B (well under the constant-cache budget), read N times per kernel (4608 times in the K=1024 N=4608 site), with all 64 lanes reading the same `x[off..off+3]` vec4 at each K-iteration — the canonical "uniform broadcast" the constant cache is built for.

Applied to `gemv_m1_k1024_no8_img` only:
```c
void gemv_m1_k1024_no8_img(
    __constant storage_t* x __attribute__((max_constant_size(2048))),
    __read_only image2d_t W_img,
    __global storage_t* out, const int N) { ... }
```
Body unchanged except `__global const half* xh` → `__constant const half* xh`.

| Run | decode tok/s | tokens deterministic |
|---|---|---|
| 1 | 11.4677 | ✓ exact |
| 2 | 11.4435 | ✓ exact |
| 3 | 11.5457 | ✓ exact |
| 4 | 11.5109 | ✓ exact |
| 5 | 11.5132 | ✓ exact |
| **median** | **11.5109** | ✓ |

**Step 17 / Step 11 (session-3 final):** **1.026× = +2.6%.** Per-token decode time: 1000 / 11.51 = **86.9 ms/token** (down from 89.2). Effective decode BW: 676 MB / 86.9 ms = **7.78 GB/s = 58% of 13.46 GB/s texture ceiling** (up from 56%).

**Tried also applying `__constant` to the other three image GEMV kernels** (no4_img K=1024, no2_img K=1024, no4_img K=4608 with max_constant_size 9216 for the K=4608 case). Result: catastrophic regression to ~5.16 tok/s. Cause: when multiple kernels in the same `cl_program` declare `__constant` args with sizable `max_constant_size`, the per-program constant cache budget is split / exceeded on Adreno 620, falling back some kernels to off-chip system memory. Reverted to keep `__constant` on `no8_img` only — the kernel where `x` is reused 32× per launch (8 outputs × 4 K-iterations) so the broadcast saving compounds most.

## Session 4 final state

| Metric | Step 11 (session-3 final) | Step 17 (session-4 final) | Δ |
|---|---|---|---|
| Decode tok/s | 11.21 | **11.51** | **+2.6%** |
| Decode ms/token | 89.2 | 86.9 | −2.3 ms |
| Effective decode BW | 7.58 GB/s | 7.78 GB/s | +0.20 GB/s |
| % of 13.46 GB/s texture ceiling | 56% | **58%** | +2 pp |
| Tokens deterministic | ✓ | ✓ | ID-for-ID matched |
| Recording-ready forward | ✗ | ✓ | clEnqueueCopyBuffer eliminated |

**Cumulative Session 1 → 4:** 4.5486 → **11.5109** = **2.53×** improvement on decode tok/s, 22% → 58% of texture ceiling.

**What's left unmoved (after this session's profiling):**
- Wall is GEMV-bound. Further fp16 wins can only come from compressing per-call inner loop further (no12_img / no16_img, two-level subgroup-shuffle reduce, image2d with smaller pix-stride). All hit the same VGPR / occupancy wall the BENCHMARK already documents.
- `cl_qcom_recordable_queues` infra is in tree (`run_record_probe` + `probe_noop` kernel) but not integrated into `Model::generate` because the gain analysis came in below the engineering cost.
- `cl_qcom_onchip_global_memory` (lever #3) untried — would need recordable_queues integration first to preserve activations across recorded kernels.
- 15 tok/s remains structurally out of reach on fp16 (676 MB/tok ÷ 13.46 GB/s = 50.2 ms/tok = 19.9 tok/s absolute max). Honest path stays int8 weight quantization.

## Repo state — session 4

- `kernels/gemv_m1.cl` — added `probe_noop` (recordable-queues test kernel); `gemv_m1_k1024_no8_img` now takes `x` as `__constant` with `max_constant_size(2048)`. Other image kernels stay `__global` (per-program constant budget).
- `kernels/attention.cl` — added `kv_write` NDRange kernel; replaces `clEnqueueCopyBuffer` for KV-cache append in decode (`seq_q==1`).
- `kernels/convolution.cl` — added `conv_block_decode` fused per-channel kernel for decode; replaces 7 helper kernels at `seq_q==1`.
- `src/main.cpp` — added `<dlfcn.h>` + `<CL/cl.h>` includes, `run_record_probe()` (≈170 lines), `NNOPT_RECORD_PROBE=1` env-var dispatch.
- `src/layers/embedding.cpp` — chained `forward_from_device_token` binds `token_ids_dev` directly as kernel arg 0 when `dev_offset_bytes==0`, skipping the per-token `clEnqueueCopyBuffer`.
- `src/layers/attention.{h,cpp}` — added `kv_write_kernel_` member; `apply_qk_norm` lambda now writes in-place (`buf` as both x and out to rmsnorm); KV cache append uses `kv_write_kernel_` for `seq_q==1`.
- `src/layers/convolution.{h,cpp}` — added `block_decode_kernel_` member; `forward()` takes the fused fast path when `seq_len==1 && state_len>0`.

**Path to 15 tok/s — what's needed:**
- 15 tok/s = 66.7 ms/token wall = 75% of texture ceiling. Need to save another 22.5 ms/token from the current 89.2.
- Per-shape utilization at Step 11 close: K=1024 sites at 65–70% (no8_img), K=4608 site at 85% (no4_img is already optimal). If K=1024 could match K=4608's 85%, savings would be ~10 ms/token → 13 tok/s. **Still short of 15.**
- The remaining gap is fundamental: per-token weight footprint (676 MB) ÷ texture-cache ceiling (13.46 GB/s) = 50.2 ms/token = 19.9 tok/s ABSOLUTE max. The only path to 15 tok/s is **int8 weight quantization** (halves weight bytes → ~22 tok/s ceiling) — out of scope for this sprint.
- Cheap remaining levers (Step 13 vec4 OperatorNorm, Step 14 conv-block kernel reduction) collectively offer ≤1% — not worth the complexity at this stage.

## Repo state — session 3

- `src/layers/embedding.{h,cpp}` — added `forward_from_device_token(queue, token_ids_dev, dev_offset_bytes)` for chained decode (reads token id from a GPU buffer instead of host array). Reuses existing persistent buf_input_ids_ / buf_out_.
- `src/model.{h,cpp}` — added `forward_greedy_chained_enqueue(start_pos)` mirroring forward_greedy at seq_len=1 with no host readback; `Model::generate()` greedy fast path now runs a ping-pong async-readback loop after prefill (Step 10).
- `kernels/gemv_m1.cl` — added `gemv_m1_k1024_no8_img` (Step 11). 8-output-per-WG image2d kernel. Selected by `run_gemv_m1_image()` for K=1024 ∧ N≥2048 ∧ N%8==0.
- `src/utils.cpp` — registered `s_gemv_m1_k1024_no8_img`; threshold N≥2048 to fall back to no4_img on small-N sites; added `pytorch_linear_silu_gate_fused()` (Step 12) gated behind `seq_len==1` (currently dispatch is disabled in mlp.cpp pending wall-clock validation).
- `src/utils.cpp` — added `ensure_mlp_fused_program()` building `kernels/mlp_fused.cl` as a SEPARATE cl_program (see Step 12 compiler interaction caveat).
- `kernels/mlp_fused.cl` — new file with `gemv_m1_k1024_no8_silufused_img`. Lives in its own program to avoid Adreno's global cross-kernel register allocator spilling no8 in gemv_m1.cl.
- `BENCHMARK.md` — Session 3 entries (this section).

## Repo state

- `src/main.cpp` — added `--no-eos` / `--ignore-eos` flag for benchmark continuation past the model's tokenizer-mismatched EOS=2.

## Repo state

- `src/main.cpp` — added `--no-eos` / `--ignore-eos` flag for benchmark continuation past the model's tokenizer-mismatched EOS=2.
- `src/kernel_profiler.{h,cpp}` — copied from mamba2 v2.
- `src/model.h` / `src/model.cpp` — added `forward_greedy()` (greedy fast path) + `ensure_argmax_resources_()` (lazy GPU argmax program + buffers). `Model::generate()` dispatches to forward_greedy when sampler is pure greedy. Layer-stack residual adds switched to `element_add_inplace()`.
- `src/utils.cpp` — `kGemvSrc` switched WG=128 → WG=64; added `gemv_m1.cl`-loaded program with `gemv_m1_k4608_no4` dispatch; profiler wired into every `clEnqueueNDRangeKernel` and CLBlast `Gemm` site with per-shape labels via `snprintf`.
- `src/opencl_context.cpp` — `build_program()` appends `-cl-fast-relaxed-math` to every kernel build.
- `kernels/gemv_m1.cl` — new: `gemv_m1_k4608_no4` (used), `gemv_m1_k1024_no4` (regression — kept for future re-eval), `gemv_m1_k1024` (regression — kept), `gemv_m1_generic` (fallback).
- `kernels/argmax.cl` — new: 2-pass GPU argmax for the greedy decode path.
- `CMakeLists.txt` — added `src/kernel_profiler.cpp` to `SOURCES`.
- All per-layer `.cpp` files (`embedding`, `mlp`, `operator_norm`, `attention`, `convolution`) — profiler events wired into every `clEnqueueNDRangeKernel` call site with mnemonic labels.
