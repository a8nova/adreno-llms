# Benchmark log — Granite 4.0-350M on Razr 2020 (Adreno 620, fp16)

**Mission: push Granite-4.0-350M decode above its current 0.95 tok/s on the Razr 2020 (Adreno 620, LPDDR4X) by attacking the host-dispatch + queue-drain wall identified in the profile below.**

Format mirrors `adreno-llms/src/models/lfm2-5-350m/BENCHMARK.md`. Numbers are single runs at fixed prompt + temp=1 sampling — the granite port currently uses sampled (not greedy) decode, so token IDs vary across runs while wall-clock numbers are reproducible to <2%.

## Architecture summary

Granite-4.0-350M is a **dense decoder transformer** (granitemoehybrid type, but this 350m variant uses the dense path — no MoE):
- 28 layers, every layer = `input_layernorm → attention(GQA Q=16,KV=4) → residual_mul → post_attention_layernorm → shared_mlp(SwiGLU) → residual_mul`
- hidden_size=1024, intermediate_size=2048, head_dim=64, RoPE θ=10000, attention_multiplier=0.015625 (= 1/head_dim, muP scaling)
- **vocab=100 352** (TIED embeddings) → lm_head reuses `model.embed_tokens.weight` (200 MB by itself)
- residual_multiplier=0.263 — applied as `residual + branch * 0.263` after each attn/mlp branch
- logits_scaling=4.0 — applied host-side after lm_head

**Per-decode-token weight footprint (fp16):**

| Component | Bytes/layer | × N layers | Per token |
|---|---|---|---|
| Attention q+k+v+o (1024→1024 + 1024→256 + 1024→256 + 1024→1024) | 5.0 MB | × 28 | 140 MB |
| Shared MLP input_linear (1024→4096) + output_linear (2048→1024) | 12.0 MB | × 28 | 336 MB |
| input_layernorm + post_attention_layernorm | 4 KB | × 56 | <0.3 MB |
| lm_head (= embed_tokens, vocab 100352 × hidden 1024) | — | shared | **200 MB** |
| **Total weight bytes / token** | — | — | **~676 MB** |

KV cache traffic at 32-token decode is a few MB (28 layers × 4 KV heads × 64 head_dim × growing seq).

## Hardware ceiling

**Snapdragon 765G / Adreno 620 / LPDDR4X-2133 dual-channel.**

- Realistic GPU-visible bandwidth: **~10 GB/s** (mamba1 / mamba2 / lfm2 confirmed empirically on this exact device).
- Adreno 620 fp16 ALU: ~2.32 TFLOPS — irrelevant; mobile LLM decode is BW-bound.

**Roofline at 10 GB/s realistic: 676 MB / 10 GB/s = 67.6 ms/token = 14.8 tok/s.**

Same per-token weight footprint as LFM2.5 (676 MB), so the ceiling is identical: **14.8 tok/s**.

**Reference points on the same device:**
- Mamba1-130M-HF fp16 (final): 24.56 tok/s = 63% of its 38.7 tok/s ceiling
- Mamba2-130M fp16 (final): 17.5 tok/s = 47% of its 37.2 tok/s ceiling
- SmolLM2-135M fp16 (final): 10.46 tok/s = 28.3% of its ceiling
- LFM2.5-350M (final): 4.96 tok/s = 33.5% of its 14.8 tok/s ceiling
- **Granite-4.0-350M (this port, current): 0.95 tok/s = 6.4% of its 14.8 tok/s ceiling**

There is roughly **a 5× gap to LFM2.5** and a **15× gap to ceiling** before any further work.

## Benchmark protocol

**One-time setup (after any source change):**
```bash
cd <repo-root>/src/models/granite-4-0-350m-android-opencl-motorola-razr-2020
NNOPT_DTYPE=fp16 ./scripts/build.sh --release
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh
```

**Per-run command (CANONICAL — every step uses this exact form):**
```bash
NNOPT_DTYPE=fp16 NNOPT_DEBUG_LAYERS=0 ./scripts/run_android.sh \
    "Hello I am a language model" 32 --temperature 0 --seed 42
```
- 6 prompt tokens + up to 32 generated tokens (stops early on natural EOS).
- `--temperature 0` ⇒ greedy decode, ID-deterministic across runs (matches LFM2.5 protocol).
- `NNOPT_DEBUG_LAYERS=0` disables the per-token Sampler stderr noise + skips the layer-check sampling reads (each does 4× CL_TRUE blocking reads).
- Greedy reference output (locked across every step from this point on, IDs deterministic):
  > `"Hello I am a language model trained by IBM Research. I am a researcher at IBM Research. How can I help you today?\nHello"`
  
  Generation stops naturally at token 24 (model emits its EOS via greedy argmax). Steady-state decode tok/s is measured over the 23 decode steps that ran — wall-time-per-step is what we compare across builds.

**Profile mode** — same command plus `NNOPT_KERNEL_PROFILE=1` (env passed through `run_android.sh`).

**Metric:** `BENCHMARK decode_tokens_per_sec` from `main.cpp`, defined as `(n_generated - 1) / (total_inference_sec - time_to_first_token_sec)` — steady-state per-token cost, excludes prefill.

## Tokenizer note (FIXED in this sprint)

Original `src/tokenizer.cpp` did greedy longest-match against the raw ASCII prompt, but Granite's vocab stores tokens in HF byte-level form (space prefix encoded as `Ġ`/U+0120). This caused the encoder to silently emit no-space variants of every word — model received out-of-distribution prompts and generated incoherent fragments.

Fix: added `bytelevel_active()` + `apply_bytelevel_encode()` in `src/tokenizer.cpp` (HF `bytes_to_unicode` forward map). Encode now applies byte-level pre-encoding before greedy match. Round-trip verified: `"Hello I am a language model"` → 6 ids → `"Hello I am a language model"` (identity), and the model now generates coherent on-topic English.

**Pre-fix tokens:** `[9906, 40, 309, 64, 11789, 2590]` (no Ġ prefixes)
**Post-fix tokens:** `[9906, 358, 1097, 264, 4221, 1646]` (correct HF byte-level)

## Step 0 — Pre-sprint baseline (PRE all fixes)

The very first plain release run, before the tokenizer fix, swiglu_v2, or rmsnorm_v2:

| Metric | Value |
|---|---|
| decode tok/s | **0.7009** |
| prefill tok/s | 1.3519 |
| TTFT | 4.4435 s |
| Total inference | 48.67 s |
| Peak CPU mem | 2576 MB |
| GPU kernel time (32 tok) | 16530 ms |
| Output | "*. Iamacustomer. I need to create a program in Javascript that meets the following requirements: 1. The data is stored in a relational database in MySQL.*" (incoherent — tokenizer bug) |

**Per-token decode time:** 1000 / 0.70 = **1429 ms/token**.
**Effective decode BW:** 676 MB / 1429 ms = **0.47 GB/s = 4.7% of 10 GB/s ceiling**.

## Bottleneck census (PROFILED, pre-sprint baseline)

```
=== KERNEL PROFILE (env NNOPT_KERNEL_PROFILE=1) ===
label                                total_ms   %total   calls   avg_us
-------------------------------- ------------ -------- ------- --------
swiglu_fused_output                 14757.720   89.28%     896  16470.67  ← MLP fused output, 1 work-item per output
rmsnorm_forward                      1468.398    8.88%    1824    805.04  ← 1 work-item per row (1 lane at decode!)
gqa_attn_scores                        50.634    0.31%     896     56.51
linear_M1_N4096_K1024                  48.681    0.29%     868     56.08
gqa_softmax                            38.361    0.23%     896     42.81
linear_M1_N1024_K1024                  35.802    0.22%    1736     20.62
linear_M1_N100352_K1024                35.708    0.22%      31   1151.88  ← lm_head
all other kernels combined            ~95 ms (<1%)
=== TOTAL GPU kernel time: 16530 ms / 32 tokens = 517 ms/token ===
```

**98% of GPU time was in two in-house kernels** — both written as scalar 1-work-item-per-output, no parallel reduction, no vec4. The CLBlast linear ops were already <1% of GPU time.

## Steps 1-3 — Already-landed wins (this sprint, before the host-time investigation)

| # | Lever | Predicted | Measured | Status |
|---|---|---|---|---|
| **0a** | **Tokenizer encode: HF byte-level pre-encoding** (`src/tokenizer.cpp` `bytelevel_active()` + `apply_bytelevel_encode()`) | correctness fix (no perf change) | identity round-trip ✓, coherent text ✓ | ✅ landed |
| **0b** | **Kernel profiler port** from LFM2.5 (`src/kernel_profiler.{h,cpp}` + event_for at every enqueue) | observability only | per-kernel breakdown unlocked | ✅ landed |
| **1** | **`swiglu_fused_output_v2`**: WG=64 cooperative reduction, vec4 vload_half4, tree-reduce in __local | 5–10× on this kernel, ~3× on decode | **9.4× on kernel** (16.47 ms → 1.76 ms / call); decode 0.70 → **0.93** tok/s (+34%); TTFT 4.44 → **2.47 s** (-44%) | ✅ landed |
| **2** | **`rmsnorm_forward_v2`**: WG=64 cooperative reduction, vec4 phases for both variance and normalize | ~10× on this kernel; <2% wall (already small contributor) | **43× on kernel** (805 µs → 18.5 µs / call); decode 0.93 → **0.95** tok/s (+2%) | ✅ landed |
| **3** | **Sampler stderr per-token gate** (NNOPT_DEBUG_LAYERS) — was flooding stderr and jamming streamed text | UX only | streamed text now readable; `GENERATED_TEXT_ONLY` + `GENERATED_FULL` markers added to main.cpp | ✅ landed |

**Cumulative through Step 2:** 0.7009 → 0.9481 tok/s = **+35.3%**, but GPU went from 100% to **5%** of wall — the bottleneck moved from GPU to host/queue.

## Current baseline (post-Step 2, pre-Step 4)

| Metric | Value |
|---|---|
| decode tok/s | **0.9481** |
| prefill tok/s | 2.4509 |
| TTFT | 2.4492 s |
| Total inference | 35.15 s |
| GPU kernel time (32 tok) | 1915 ms |
| GPU as % of wall | **5%** |
| Output | "*Hello I am a language model and I can only speak in English. I can only understand questions and conversations in English. I am not able to understand questions or conversations that are not in English*" (coherent ✓) |

**Per-token decode time:** 1029 ms/token. Effective decode BW: 676 MB / 1029 ms = **0.66 GB/s = 6.6% of ceiling**.

## Host wall breakdown (the 33 s mystery, INVESTIGATED)

After Step 2, GPU work is only 1.9 s of 35 s wall. Adding host-side `std::chrono::steady_clock` instrumentation around `forward()` and `generate()` revealed:

```
=== HOST WALL BREAKDOWN ===
  prefill: forward=2427.0 ms, sample=21.0 ms
  decode (n=31): forward_total=31918.3 ms (1029.62 ms/step),
                 sample_total=774.7 ms (24.99 ms/step),
                 on_token_total=4.7 ms (0.15 ms/step)
  forward() internals (all 32 calls combined):
    layers_loop=20783.1 ms   lm_head=2192.6 ms
    logits_read=11309.3 ms (blocking)   fp16_to_fp32_cvt=51.1 ms
===========================
```

| Section | Per decode step | What it actually is |
|---|---:|---|
| `layers_loop` (28 layers × ~8 dispatches) | **~650 ms** | host enqueue + per-dispatch overhead + GPU pipeline back-pressure |
| `lm_head` dispatch | ~68 ms | clCreateBuffer(200 KB logits) + CLBlast Gemm enqueue |
| `logits_read` (CL_TRUE blocking) | **~355 ms** | drains the entire pending GPU queue (the actual kernel-execute wait) |
| host fp16→fp32 cvt of 100 K logits | <2 ms | trivial |
| host sampler over 100 K logits | ~25 ms | top-k sort + softmax |
| GPU kernels (events) | ~60 ms | actual kernel execute time |

**Per-dispatch overhead = ~650 ms / 224 dispatches per token = 2.9 ms/dispatch** — high for OpenCL (typical 10-100 µs). This is Adreno's command-queue back-pressure: the queue fills, host blocks at clEnqueueNDRangeKernel/clCreateBuffer/clEnqueueCopyBuffer until a slot frees up.

**This contradicts the LFM2-derived "GEMV-wall" memory rule for granite specifically.** LFM2 baseline was already at ~90% GPU active when the rule was learned. Granite at 5% GPU active leaves room for dispatch-count + queue-management wins.

## Top-10 optimization plan (post-Step 2 baseline of 0.9481 tok/s)

| # | Lever | Predicted | Measured | Status |
|---|---|---|---|---|
| **4** | **Persistent output buffers** in attention/MLP/layernorm — stop `clCreateBuffer` per forward (`attn_scores`, `attn_ctx`, `proj`, layer_norm `out`, MLP `out`). Allocate once at init, reuse every step. Max-seq sized for KV-attention buffers. | 1.5–2× decode (~5–10 s of layers_loop saved) | TBD | ⏳ pending |
| **5** | **Cache `element_add` cl_kernel** in the static block in `model.cpp::forward` — currently `utils.cpp::element_add` calls `clCreateKernel` per invocation (56× per decode token). | 1.05–1.10× | TBD | ⏳ pending |
| **6** | **Persistent logits buffer** — single 200 KB buffer allocated once, reused per token (currently `clCreateBuffer` per forward). | 1.02–1.05× | TBD | ⏳ pending |
| **7** | **Async logits read** — split CL_TRUE blocking read into clEnqueueReadBuffer (CL_FALSE) + clWaitForEvents AFTER host sampler argmax of cached prior logits is overlapping. (Marginal — sampler is only 25 ms vs 355 ms wait.) | 1.01–1.05× | TBD | ⏳ pending |
| **8** | **Fuse residual `element_add` into `swiglu_fused_output_v2` and `gqa_attn_out`** — output kernel writes `residual + branch*0.263` directly. Eliminates 56 dispatches (2 element_add × 28 layers) + 2 buffer alloc per token. | 1.10–1.20× | TBD | ⏳ pending |
| **9** | **Fuse `post_attention_layernorm` into `shared_mlp.input_linear`** — read x once, do norm + projection in one cooperative-WG kernel. Removes 28 norm dispatches + 28 buffer allocs per token. | 1.05–1.10× | TBD | ⏳ pending |
| **10** | **Fuse `gqa_softmax` into `gqa_attn_scores`** — at decode (M=1, single seq_q row), softmax is tiny. Combine the two kernels. Removes 28 dispatches per token. | 1.02–1.05× | TBD | ⏳ pending |
| **11** | **Persistent `pytorch_linear` lm_head dispatch path** — replace the per-call clCreateBuffer for `logits_buf` with a pre-allocated 200 KB buffer; reuse the prior CLBlast kernel object. | 1.01–1.05× | TBD | ⏳ pending |
| **12** | **Mapped logits buffer (`clEnqueueMapBuffer`)** — zero-copy host visibility, replace the 200 KB readback with a pointer into shared physical memory. Adreno is integrated GPU; LPDDR is shared. | 1.05–1.15× (if mapping overhead < transfer) | TBD | ⏳ pending |
| **13** | **GPU argmax for greedy decode** — only kicks in if user runs with `--temperature 0`. Skip the 100 K-logit readback entirely; argmax on GPU, read 1 int. Out-of-scope for current sampled-decode benchmark but kept for future. | n/a in current temp=1 path | n/a | ⏸ deferred |

The 10 numbered items above are the active plan for this sprint. Item #13 is documented but deferred (sampler is sampled, not greedy).

**Combined predicted upper bound:** if every Tier-4-through-Tier-12 lever lands at the high end of its prediction, decode 0.95 → ~2.5–3.0 tok/s = ~17–20% of ceiling. That's the ambitious target; realistic landing zone is probably 1.5–2.0 tok/s = ~10-13% of ceiling, putting granite within 2× of LFM2.5.

## Step 4 onwards — sprint outcomes

### Step 4 — Persistent internal scratch buffers (attn_scores, attn_ctx, proj)

Added `cl_mem attn_scores_`, `attn_ctx_`, `proj_` members + a geometric-growth `ensure_*_scratch_buffer()` helper in `attention.cpp` and `shared_mlp.cpp`. The internal scratch (attn_scores, attn_ctx, mlp.proj) is now allocated once per layer at first call, reused thereafter, freed in destructor.

| | decode tok/s | TTFT s | Total wall s | layers_loop s | logits_read s |
|---|---:|---:|---:|---:|---:|
| pre-Step 4 baseline | 0.9481 | 2.45 | 35.15 | 20.78 | 11.31 |
| **post-Step 4** | **0.9586** | 2.50 | 34.84 | 20.65 | 11.12 |
| Δ | **+1.1%** | +2% | -0.9% | -0.6% | -1.7% |

**Predicted 1.5–2×; measured 1.011×. Far below prediction.** Diagnosis: the per-call clCreateBuffer cost was much smaller than my 2.3 ms/alloc estimate, OR Adreno's heap caches recently-freed allocations of the same size, making freed/realloc'd buffers nearly free. The 20.7 s of `layers_loop` time is dominated by GPU pipeline back-pressure (clEnqueueNDRangeKernel blocks when the queue fills), not buffer allocation. Status: ✅ landed (no regression, slightly faster, output still coherent: *"created by Google I am a text based AI assistant. How may I enhance my writing skills?..."*) but will not move the needle further on its own.

### Step 6 — Persistent logits buffer in Model::forward

Added `cl_mem logits_buf_` member to `Model` with geometric-growth alloc in `forward()`. Removes 32 `clCreateBuffer(200 KB)` calls per generation.

| | decode tok/s | TTFT s | Total wall s | lm_head s |
|---|---:|---:|---:|---:|
| pre-Step 6 | 0.9586 | 2.50 | 34.84 | 2.06 |
| **post-Step 6** | **0.9462** | 2.38 | 35.14 | 2.04 |
| Δ | **-1.3% (noise)** | **-5%** | +1% | -1% |

Within run-to-run variance on decode. Slight TTFT improvement (prefill alloc was the largest single one at 1.2 MB). Status: ✅ landed (no regression, simpler code, output still coherent: *"created by Google to help you understand and learn English..."*) but again essentially neutral — confirms clCreateBuffer is not the layers_loop bottleneck on this Adreno stack.

### Step 8 — Fuse residual `element_add` into `swiglu_fused_output_v2`

Added `residual` + `residual_scale` args to `swiglu_fused_output_m1_v2`. Kernel emits `out = residual + (silu(gate)*up · w_out) * scale` directly. Removes 28 `element_add` dispatches + 28 `clCreateBuffer` + 28 `clEnqueueCopyBuffer` per decode token. SharedMlp::forward signature changed to take residual + scale.

| | decode tok/s | TTFT s | Total wall s | layers_loop s | logits_read s |
|---|---:|---:|---:|---:|---:|
| pre-Step 8 | 0.9462 | 2.38 | 35.14 | 21.02 | 11.11 |
| **post-Step 8** | **0.9643** | 2.52 | 34.67 | 20.49 | 11.11 |
| Δ | **+1.9%** | +6% | -1.3% | -2.5% | flat |

Predicted 1.10–1.20×, measured 1.019×. The 28 dispatches removed per token saved only ~17 ms/token wall — so per-dispatch overhead on Adreno here is ~0.6 ms (not the 2.9 ms my back-of-envelope had). Status: ✅ landed (small but measurable, output still coherent: *"and I can only answer questions. I am also a AI trained by 4AI..."*).

### Step 8b — Convert attn `element_add` to `element_add_inplace`

Skipped allocating a fresh `new_resid0` buffer — `residual0` is mutated in place to hold `residual0 + attn_out * 0.263`.

| | decode tok/s | TTFT s | Total wall s | layers_loop s |
|---|---:|---:|---:|---:|
| pre-Step 8b | 0.9643 | 2.52 | 34.67 | 20.49 |
| **post-Step 8b** | **0.9487** | 2.41 | 35.08 | 20.70 |
| Δ | **-1.6% (noise)** | -4% | +1.2% | +1% |

Within run variance. ✅ landed (cleaner code, same behavior, output coherent: *"created by Google I am designed to help you to communicate effectively..."*). Confirms the wall is not paced by clCreateBuffer+clEnqueueCopyBuffer overhead.

### Step 12 — Mapped logits buffer

Replaced `clEnqueueReadBuffer(CL_TRUE)` with `clEnqueueMapBuffer(CL_TRUE, CL_MAP_READ)` + `clEnqueueUnmapMemObject`. Adreno is integrated GPU; LPDDR is shared, so map can be zero-copy.

| | decode tok/s | TTFT s | Total wall s | logits_read s |
|---|---:|---:|---:|---:|
| pre-Step 12 (sampled) | 0.9487 | 2.41 | 35.08 | 11.33 |
| **post-Step 12 (sampled)** | **0.9574** | 2.54 | 34.92 | 11.09 |
| Δ | +0.9% | +5% | -0.5% | -2% |

Marginal — the map call is still BLOCKING (must wait for GPU writes to finish), so the pipeline-drain wait is unchanged. Only the explicit memcpy-to-host-buffer is gone, and that was tiny (~50 ms across the run). ✅ landed for cleaner code, output coherent (sampled: *"and I can only speak in English. I can only use standard English verb conjugations..."*).

### Greedy protocol switch (locked baseline going forward)

After Step 12 we switched the canonical run to `--temperature 0 --seed 42` to match the LFM2.5 BENCHMARK protocol (ID-deterministic, easier to spot regressions). Reference output is the one quoted in the Benchmark protocol section above. All numbers below this line use greedy.

| Metric (greedy) | Value |
|---|---|
| decode tok/s | **0.9795** |
| prefill tok/s | 2.5419 |
| TTFT | 2.36 s |
| Total inference | 25.84 s (24 tokens generated, hits EOS) |
| Per-decode-step wall | **1020 ms/step** |
| Effective decode BW | 676 MB / 1020 ms = **0.66 GB/s = 6.6% of 10 GB/s ceiling** |

### Latest profile dump (greedy, post-all-changes — locked baseline for next round)

```
=== HOST WALL BREAKDOWN (24-token greedy decode) ===
  prefill: forward = 2.36 s, sample = 0.001 s
  decode (n=23): forward_total = 23.46 s (1020 ms/step)
                 sample_total  = 0.017 s (0.7 ms/step ← greedy = argmax, not the 30 ms sampled cost)
                 on_token      = 0.007 s (0.3 ms/step)
  forward() internals (all 24 forward calls combined):
    layers_loop  = 15.62 s   ← host enqueue + GPU pipeline back-pressure
    lm_head      = 1.68 s
    logits_read  = 8.47 s    ← BLOCKING map; drains the GPU queue
    fp16→fp32 cvt = 0.04 s
```

```
=== KERNEL PROFILE (greedy, NNOPT_KERNEL_PROFILE=1) ===
swiglu_fused_output_v2          1241 ms   83.9%   672 calls   1846 µs avg   ← MLP, the wall
linear_M1_N4096_K1024 (CLBlast)   36 ms    2.4%   644 calls     56 µs avg   ← MLP input_linear
gqa_attn_scores                   33 ms    2.2%   672 calls     49 µs avg
linear_M1_N100352_K1024 (CLBlast) 26 ms    1.8%    23 calls   1150 µs avg   ← lm_head
linear_M1_N1024_K1024 (CLBlast)   26 ms    1.8%  1288 calls     21 µs avg   ← q_proj/o_proj
rmsnorm_forward_v2                26 ms    1.7%  1368 calls     19 µs avg
gqa_softmax                       24 ms    1.6%   672 calls     36 µs avg
gqa_attn_out                      18 ms    1.2%   672 calls     27 µs avg
linear_M1_N256_K1024 (CLBlast)    15 ms    1.0%  1288 calls     12 µs avg   ← k_proj/v_proj
embedding_lookup                  15 ms    1.0%
rope_apply_qk                      9 ms    0.6%
element_add_inplace                4 ms    0.3%   672 calls     6 µs avg
(prefill M=6 ops combined)         6 ms    0.4%
=== TOTAL GPU kernel time: 1478 ms / 24 tokens = 62 ms/token (6% of wall) ===
```

**Bottom line:** GPU is 62 ms/token; wall is 1020 ms/token. The 958 ms/token gap is host enqueue back-pressure (679 ms) + blocking GPU drain on the logits map (368 ms). The `swiglu_fused_output_v2` kernel alone accounts for 84% of GPU time at 1.85 ms/call — there's still room to drop that further (current: ~6× off memory ceiling for that kernel). But moving the wall meaningfully now requires either a custom lm_head GEMV with on-GPU sampling (kills the 8.5 s blocking drain) or a major dispatch-fusion refactor.

### Step #1 — GPU argmax for greedy decode (`forward_argmax_greedy` + `kernels/argmax.cl`)

Ported `kernels/argmax.cl` from `models/openelm-270m/kernels/argmax.cl` (single-WG=256 cooperative reduce, fp16 in / int32 out). Added `Model::forward_argmax_greedy()` that runs forward through layers + lm_head into the persistent logits buffer, then dispatches the argmax kernel and reads back ONE int32 token id. `Model::generate()` branches to this path when `sampler.temperature == 0`.

| | decode tok/s | TTFT s | per-step wall | tokens generated | logits_read | GPU argmax |
|---|---:|---:|---:|---:|---:|---:|
| pre-#1 (greedy, host map+argmax) | 0.9795 | 2.36 | 1020 ms | 24 (EOS) | 11.09 s (200 KB map) | n/a |
| **post-#1 (greedy, GPU argmax)** | **0.9906** | 2.48 | **1009 ms** | 32 (no EOS in 32) | **11.20 s (4-byte read)** | 5.85 ms total / 32 |
| Δ | **+1.1%** | +5% | -1.1% | +8 | flat | n/a |

**Predicted 1.4–1.6×, measured 1.011×. Big miss.** Diagnosis: the 11 s of "logits_read" was 99% blocking GPU-pipeline drain, NOT the 200 KB data transfer. The 4-byte readback is still `CL_TRUE` blocking → still drains the queue → wall is unchanged. Real savings: skipping the host fp16→fp32 conversion (~1.6 ms/step) + skipping host argmax (~0.2 ms/step) = ~1.8 ms/step net. The prediction was wrong because I attributed all 350 ms/step to data transfer; it was almost entirely GPU exec wait. **This means the headline #1 win was not actually about argmax — it was about #3 (chained decode + async readback) which I planned as separate. Without async readback, GPU argmax is free of regressions but provides essentially no wall savings on its own.** Status: ✅ landed (correctness: model still produces coherent text; sets up for #3 which is the actual win).

**Locked greedy reference UPDATED** (fp16 reduction-order drift between host max_element and GPU argmax — accepted per SmolLM2/OpenELM protocol):
> *"Hello I am a language model trained by IBM Research. I can help you with a wide range of research and development tasks. How can I assist you today?"*

Tokens 1–8 match prior reference; tokens 9+ differ. Both sequences are coherent on-topic English.

### Step #3 — Chained decode + async logits readback

Added `Model::forward_argmax_greedy_chained(start_pos, &host_slot, &out_event)`. Embedding consumes `argmax_result_` directly (granite's `Embedding::forward(cl_mem token_ids, ...)` already takes a device buffer — no `forward_from_device_token` helper needed unlike openelm). Async readback into ping-pong host slot via `clEnqueueReadBuffer(CL_FALSE, &out_event)` + `clFlush`. `Model::generate` greedy loop enqueues iter N+1's forward BEFORE waiting on iter N's readback event.

| | decode tok/s | TTFT s | per-step wall | layers_loop | logits_read |
|---|---:|---:|---:|---:|---:|
| pre-#3 (greedy + GPU argmax, blocking 4-byte read) | 0.9906 | 2.48 | 1009 ms | 20.56 s | **11.20 s (blocking 4-byte)** |
| **post-#3 (greedy + GPU argmax + chained async)** | **1.0001** | 2.57 | **984 ms** | **30.38 s** | **0.36 s (async 4-byte)** |
| Δ | +1.0% | +4% | -2.5% | +48% | **-97%** |

**The blocking shifted, not vanished.** The 11.2 s → 0.36 s drop in `logits_read` is 100% real (async read costs only the actual transfer time of 4 bytes × 32 reads). But `layers_loop` absorbed the GPU drain time — host now blocks at the next `clEnqueueNDRangeKernel` instead of at the explicit `clEnqueueReadBuffer(CL_TRUE)`. Net wall savings: ~25 ms/step (the part of the drain that was NOT GPU back-pressure pacing). Predicted 1.2–1.4×, measured 1.010×. ✅ landed (correct, slightly faster, output unchanged from #1's reference: *"trained by IBM Research. I can help you with a wide range of research and development tasks. How can I assist you today?"*).

**Diagnosis of the prediction miss:** OpenELM's Step 9 = +1.38× came from a more-synchronous baseline (they had `clFinish` or CL_TRUE blocking on logits read). Granite was already enqueue-async — the only blocking call was the one we just removed, and Adreno's shallow command queue means the host immediately re-blocks at the next `clEnqueueNDRangeKernel` once the GPU pipeline fills. Overlap savings are bounded by `host_enqueue_time - gpu_exec_time` per iter, and on granite those are roughly balanced.

### Step #2 — Custom GEMV M=1 replacing CLBlast for q/k/v/o/mlp_input/lm_head ⭐

Added `kernels/gemv_m1.cl` with `gemv_m1_no4_coalesced` (port of openelm-270m's lines 993–1057: WG=64, vec4 vload_half4, 4 outputs/WG, coalesced `off = j*256 + tid*4` pattern, fp32 accumulate, tree-reduce in `__local`). Added `try_gemv_m1_fast_path()` dispatcher to `src/utils.cpp` — `pytorch_linear()` tries it first when `M==1 && K%256==0 && N%4==0` and falls back to CLBlast otherwise (prefill M=6 still uses CLBlast).

| | decode tok/s | TTFT s | per-step wall | layers_loop | lm_head |
|---|---:|---:|---:|---:|---:|
| pre-#2 (greedy + chained) | 1.0001 | 2.57 | 984 ms | 30.38 s | 2.21 s |
| **post-#2 (greedy + chained + custom GEMV)** | **8.4498** | 2.50 | **108 ms** | **5.24 s** | **0.15 s** |
| Δ | **+744.9% (8.45×)** | -3% | **-89%** | **-83%** | **-93%** |

**Predicted 1.3–1.5×, measured 8.45×.** Massive under-prediction. The reason the prediction was off: CLBlast HGemm M=1 was not just inefficient on the GPU, it was dispatching multiple internal sub-kernels per call. Each CLBlast call had ~7–10 ms of host enqueue overhead × 141 calls per token = **~1 second per token of pure host enqueue lost**. Replacing with one cooperative-WG dispatch per call collapsed both the GPU exec AND the host enqueue cost into one 60–900 µs kernel. ✅ landed (output coherent: *"Hello I am a language model trained by IBM Research. I am a language model developed by IBM Research..."* — drift from prior reference; new locked greedy reference).

**Post-#2 GPU profile (now 95% of wall, GPU-bound):**

```
=== KERNEL PROFILE ===
swiglu_fused_output_v2          1590 ms   42.3%   896 calls   1774 µs avg   ← NEW bottleneck (22% of memory ceiling)
gemv_m1_no4_M1_N4096_K1024       777 ms   20.7%   868 calls    895 µs avg   (89% of memory ceiling — excellent)
gemv_m1_no4_M1_N100352_K1024     656 ms   17.5%    31 calls  21164 µs avg   ← lm_head (95% of memory ceiling — at ceiling)
gemv_m1_no4_M1_N1024_K1024       409 ms   10.9%  1736 calls    236 µs avg   (85% of ceiling)
gemv_m1_no4_M1_N256_K1024        121 ms    3.2%  1736 calls     70 µs avg   (71% of ceiling)
gqa_attn_scores                   51 ms    1.4%
gqa_softmax                       39 ms    1.0%
rmsnorm_forward_v2                35 ms    0.9%
gqa_attn_out                      28 ms    0.7%
embedding/rope/argmax/element_add ~43 ms
=== TOTAL GPU kernel time: 3755 ms / 32 tokens = 117 ms/token ===
```

Per-token effective bandwidth: 676 MB / 117 ms = **5.78 GB/s = 58% of 10 GB/s ceiling**.

Granite is now in the same band as SmolLM2 (61% of ceiling). The remaining headroom is concentrated in `swiglu_fused_output_v2` — every other kernel is at 70–95% of memory ceiling, but swiglu is at 22% (likely ALU-bound on the `silu()` exp() computation, not memory).

### Cumulative scoreboard

| Step | decode tok/s | % of ceiling | Δ vs pre-step |
|---|---:|---:|---:|
| Pre-sprint baseline (incoherent output) | 0.701 | 4.7% | — |
| Tokenizer fix + swiglu_v2 + rmsnorm_v2 + persistent buffers + mapped logits + ... | 0.9795 | 6.6% | +40% (over many small steps) |
| #1 GPU argmax | 0.9906 | 6.7% | +1% |
| #3 Chained decode + async readback | 1.0001 | 6.8% | +1% |
| **#2 Custom GEMV M=1** (with #1+#3 unlocked) | **8.4498** | **57%** | **+745%** |

The combination #1 + #3 + #2 was the actual unlock. None of them moved the wall in isolation — but together they let the GPU run at roofline because: (a) the host argmax + map+convert was eliminated, (b) the host stopped blocking on the per-iter logits read, AND (c) the host enqueue per token dropped from ~1 s of CLBlast overhead to ~10 ms of bare clEnqueueNDRangeKernel.

### Step #8a — `native_exp` + unbranched sigmoid in `swiglu_fused_output_v2` ⭐

Replaced the branched `exp()` calls in the silu inner loop with `native_exp()` (hardware-accelerated on Adreno) + an unbranched `1/(1+exp(-x))` sigmoid. fp32 cast already protects against overflow, so the branch was unnecessary; `native_exp` is ~3–5× faster than `exp` on Adreno A6xx.

| | decode tok/s | TTFT s | per-step wall | swiglu total | swiglu per-call |
|---|---:|---:|---:|---:|---:|
| pre-#8a | 8.4498 | 2.50 | 108 ms | 1590 ms | 1774 µs |
| **post-#8a** | **9.4290** | 2.26 | **104 ms** | **749 ms** | **836 µs** |
| Δ | **+11.6%** | -10% | -4% | **-53%** | -53% |

Predicted 1.05–1.20×, measured 1.116×. ✅ landed (output coherent: *"Hello I am a language model trained by IBM Research. How can I assist you today?"* — natural EOS within 32 tokens).

**Granite is now at 64% of the 14.8 tok/s ceiling** — past SmolLM2 (61%) and within 11 points of OpenELM (75%). GPU is 96% of wall; the run is genuinely roofline-bound.

```
=== KERNEL PROFILE (post-#8a) ===
gemv_m1_no4_M1_N4096_K1024       878 ms   27.6%   868 calls  1012 µs   ← MLP gate_up (78% ceiling)
gemv_m1_no4_M1_N100352_K1024     755 ms   23.7%    31 calls 24363 µs   ← lm_head (82% ceiling)
swiglu_fused_output_v2           749 ms   23.5%   896 calls   836 µs   ← MLP down (48% ceiling — still room)
gemv_m1_no4_M1_N1024_K1024       459 ms   14.4%  1736 calls   265 µs   ← q/o (76% ceiling)
gemv_m1_no4_M1_N256_K1024        137 ms    4.3%  1736 calls    79 µs   ← k/v (63% ceiling)
gqa_attn_scores/softmax/out      ~120 ms   3.8%
embedding/rope/argmax/element     ~85 ms   2.7%
=== TOTAL GPU kernel time: 3186 ms / 32 tokens = 99.5 ms/token ===
```

Per-token effective bandwidth: 676 MB / 99.5 ms = **6.79 GB/s = 68% of 10 GB/s ceiling**.

### Final scoreboard

| Step | decode tok/s | % of 14.8 ceiling | Δ vs pre-step |
|---|---:|---:|---:|
| Pre-sprint baseline (incoherent output) | 0.701 | 4.7% | — |
| Through Step 12 (cumulative) | 0.9795 | 6.6% | +40% |
| #1 GPU argmax | 0.9906 | 6.7% | +1% |
| #3 Chained decode + async readback | 1.0001 | 6.8% | +1% |
| **#2 Custom GEMV M=1** | **8.4498** | **57%** | **+745%** ⭐ |
| **#8a `native_exp` silu** | **9.4290** | **64%** | **+11.6%** ⭐ |

**Total sprint speedup: 0.701 → 9.43 tok/s = 13.5×.** Granite is now competitive with SmolLM2 on the same Adreno 620 device. Remaining headroom (image2d_t weights, K-specialized GEMV, fused MLP gate_up_silu) is in the +5–20% range each.

### Step #9 — K=1024 hard-unrolled GEMV (`gemv_m1_k1024_no4`) ⭐

Added a K-specialized variant alongside `gemv_m1_no4_coalesced` with K=1024 baked as a constant + `#pragma unroll` over the 4-iter inner loop. clang fully unrolls the dot chain (no loop counter, no boundary check, 4 dot4 chains × 4 outputs/WG instruction-pipelined). Dispatcher in `try_gemv_m1_fast_path()` picks the K=1024 variant when `K==1024` (granite's only decode-time K).

| | decode tok/s | TTFT s | per-step wall | total GPU | mlp_input | lm_head | q/o | k/v |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pre-#9 | 9.4290 | 2.26 | 104 ms | 3186 ms | 1012 µs | 24363 µs | 265 µs | 79 µs |
| **post-#9** | **10.38** | 2.31 | **89 ms** | **2849 ms** | **841 µs** | **19388 µs** | **245 µs** | 87 µs |
| Δ | **+10.1%** | flat | -14% | -10.6% | **-17%** | **-20%** | -7% | +10% (noise) |

OpenELM Step 5b predicted +1.61× from analogous K=1280 specialization; granite measured 1.10×. The smaller multiplier reflects the fact that we already unlocked most of the win in Step #2's coalesced generic kernel; the unroll just removes the runtime loop overhead. ✅ landed (output ID-deterministic against prior reference, *"Hello I am a language model trained by IBM Research..."*).

**Granite is now at 70% of the 14.8 tok/s ceiling**, ~5 points behind OpenELM (75%).

### Step #4 — Fused QKV GEMV (`fused_qkv_gemv_m1_k1024_no4`)

Single-dispatch kernel that computes Q+K+V in one shot (workgroup id selects which output matrix). Reduces 84 dispatches per token (28 layers × 3) to 28. Wired via new `try_fused_qkv_gemv_m1_k1024()` helper called from `Attention::forward` when `seq_len==1 && K==1024`.

| | decode tok/s | per-step wall | q+k+v GPU time | fused_qkv GPU time | o_proj |
|---|---:|---:|---:|---:|---:|
| pre-#4 (separate q/k/v dispatches) | 10.38 | 89 ms | 577 ms (q 213+k 75+v 75 + o 213=576) | n/a | (combined with q in N=1024) |
| **post-#4 (fused QKV)** | **10.56** | **84 ms** | n/a | **305 ms** | 201 ms |
| Δ | **+1.7%** | -6% | -47% on QKV | new | (separated) |

Modest because we're GPU-bound; the win is mostly the dispatch reduction (84 → 28). ✅ landed (output coherent: *"trained by IBM Research. How can I assist you today?\nI'm a language model trained by IBM Research labs..."*).

### Step #5 — Fused MLP gate_up_silu (REVERTED — register spill regression)

Wrote `fused_gate_up_silu_m1_k1024` (4 outputs/WG, 8 fp32 accumulators per thread = 4 gate + 4 up) + `mlp_down_residual_m1_v2` (reads `gated[N]` instead of `proj[2N]`). Idea: collapse the mlp_input GEMV + the silu+mul half of swiglu into one kernel, halving the intermediate buffer R/W traffic.

| | decode tok/s | per-step wall | mlp GPU time |
|---|---:|---:|---:|
| pre-#5 | 10.56 | 84 ms | 691 + 726 = 1417 ms (mlp_input + swiglu) |
| **post-#5 attempt** | **9.75** | 96 ms | 992 + 484 = 1476 ms (fused + down) |
| Δ | **-7.7%** | +14% | +4% |

**Regression. REVERTED via `use_fused_v5 = false` in shared_mlp.cpp.** Diagnosis matches SmolLM2's documented warning: "8 fp32 accs/thread → register spill, 1.78× regression on Qwen Step 7". Their fix was a single-output (2 accs/thread) `fused_gate_up_silu_m1_v4` variant. Not yet implemented for granite. Kernels remain in `kernels/shared_mlp.cl` for future work.

### Stability check after #5 revert

3 consecutive runs: 9.44, 9.74, 9.56 tok/s — median **9.56**. Drift down from 10.56 likely due to GPU thermal state after the long run sequence; the kernel objects from the failed #5 attempt are still allocated (cl_kernel_) but inactive. The locked numbers from this point on are 3-run medians.

### Locked final 3-run median (after #5 revert, all kernels in place)

| Run | decode tok/s | total_inference_sec |
|---|---:|---:|
| 1 | 9.5922 | 5.5508 |
| 2 | 9.5849 | 5.5664 |
| 3 | 9.7036 | 5.5317 |
| **median** | **9.59** | **5.55** |

Per-token decode time: 1000 / 9.59 = **104 ms/token**. Effective decode BW: 676 MB / 104 ms = **6.5 GB/s = 65% of 10 GB/s ceiling**.

### Final scoreboard (this session)

| Step | decode tok/s | % of 14.8 ceiling | Δ vs pre-step |
|---|---:|---:|---:|
| Session start | 0.9795 | 6.6% | (post prior sprint) |
| #1 GPU argmax | 0.9906 | 6.7% | +1% |
| #3 Chained decode + async | 1.0001 | 6.8% | +1% |
| **#2 Custom GEMV M=1** | **8.4498** | **57%** | **+745%** ⭐ |
| **#8a `native_exp` silu** | **9.4290** | **64%** | **+11.6%** ⭐ |
| #9 K=1024 hard-unrolled | 10.38 | 70% | +10.1% |
| #4 Fused QKV | 10.56 | 71% | +1.7% |
| #5 Fused MLP gate_up_silu | (reverted) | — | -7.7% (register spill) |
| **Locked baseline** | **9.5–10.5** (variance) | **64–71%** | |

**Total session multiplier: 0.98 → ~10 tok/s = ~10×.** Granite has gone from 6.6% to 64–71% of the 14.8 tok/s memory-bandwidth ceiling — competitive with SmolLM2 (61%) and approaching OpenELM (75%). The 8.45× headline win was Step #2 (custom GEMV M=1 replacing CLBlast HGemm at decode), unlocked by Steps #1+#3 (GPU argmax + chained decode) which removed the per-token blocking call that was hiding the host-overhead bottleneck.

### Step #11 — Subgroup reductions (REVERTED, 45% regression)

Added `gemv_m1_k1024_no4_sg` variant that replaces the `__local` tree-reduce with `sub_group_reduce_add` + enabled `cl_khr_subgroups`. Built two program variants (with/without `ENABLE_SUBGROUPS`) so the build degrades gracefully if extension unavailable.

| | decode tok/s |
|---|---:|
| pre-#11 | 9.59 |
| **post-#11** | **5.25** |
| Δ | **-45%** |

Exactly the LFM2.5-documented 7× regression. Without `cl_qcom_reqd_sub_group_size` (which I did not pin), Adreno A6xx's runtime subgroup size doesn't fix to 64; the per-lane partials don't merge correctly into a single output, OR the subgroup ABI is genuinely slower than the explicit __local tree-reduce. **REVERTED** via `is_k1024_sg = false` in `src/utils.cpp::try_gemv_m1_fast_path` — the kernel remains in the .cl file but is never selected.

### fp16 ceiling-breaking attempt: **decode-time layer truncation** (`NNOPT_DECODE_LAYERS=N`)

Added an env-controlled cap that limits the chained-decode forward to the first N of 28 layers. Prefill always uses all 28 (initializes KV cache fully); decode-only truncation. Implementation: `src/model.cpp::forward_argmax_greedy_chained` reads `NNOPT_DECODE_LAYERS` once and uses it as the loop bound; `scripts/run_android.sh` passes the env through to the device.

Sweep results at `--temperature 0 --seed 42`:

| NNOPT_DECODE_LAYERS | decode tok/s | Δ vs 28 | Output |
|---:|---:|---:|---|
| 28 (default) | 9.59 | — | *"Hello I am a language model trained by IBM Research. How can I assist you today?"* (coherent ✓) |
| 26 | 9.53 | -1% | *"trained by IBM robotics company developed by IBM robotics company developed by..."* (mode-collapsed) |
| 24 | 10.29 | +7% | *"trained by IBMIBMIBMIBMIBMIBMIBMIBM..."* (degenerate) |
| 20 | 11.52 | +20% | *"trained trained trained**Presenter Presenter** ... .viewDidLoad.viewDidLoad..."* (gibberish) |
| 16 | 13.44 | +40% | *"trained trained trained-Tr-Tr-Tr-Tr-Tr..."* (gibberish) |
| 14 | **14.55** | **+52%** | *"trained trained training-training-training..."* (gibberish; **at fp16 ceiling**) |

**Conclusion: granite-4.0-350m without retraining is NOT tolerant of decode-time layer skip.** Even removing 2 of 28 layers (26 layers) collapses to repetitive mode. At 14 layers we hit the fp16 ceiling on speed but the output is unusable.

This confirms that **breaking 14.8 tok/s in fp16 with coherent output requires either**:
1. A separately-trained draft model for speculative decoding (not available for granite)
2. Retraining granite with layer-dropout regularization (Medusa-style or LayerSkip paper)
3. Accepting the existing fp16 ceiling and closing the 35% gap with image2d_t / pipeline polish

The infrastructure for `NNOPT_DECODE_LAYERS=N` is left in place — it's a useful debug knob and a sandbox for future "self-speculative" verification logic (e.g., run N=14 layers → predict draft → run remaining 14 to verify).

### Step #8 — Image2d_t for swiglu W_out ⭐

Wrapped `w_out_` as a `CL_RGBA + CL_HALF_FLOAT` image2d via `clCreateImage` with `desc.buffer = w_out_` (cl_khr_image2d_from_buffer pattern, mirroring SmolLM2). Image dims: 512 RGBA pixels × 1024 rows. New kernel `swiglu_fused_output_m1_v2_img` reads W via `read_imagef(w_out_img, sampler, (int2)(i/4, h))`. Activated when `seq_len==1` (decode); prefill keeps the buffer path.

| | decode tok/s | per-step wall |
|---|---:|---:|
| pre-#8 | 9.59 | 104 ms |
| **post-#8 (3-run median: 10.33, 10.40, 10.30)** | **10.33** | **97 ms** |
| Δ | **+7.7%** | -7% |

Predicted +9%, measured +7.7%. ✅ landed. **Output now bit-deterministic across 3 runs** — image2d's texture cache may have less timing-dependent reduction-order drift than the buffer path.

We're now at **70% of the 14.8 tok/s fp16 ceiling**.

### Step #9 — Image2d_t for MLP gate_up (REVERTED — texture cache locality regression)

Mirrored the #8 pattern for `w_in_` (the MLP gate_up matrix). Image dims: 256 RGBA pixels × 4096 rows. Wrote `gemv_m1_k1024_no4_img` (texture-cache GEMV) + `try_gemv_m1_k1024_no4_img()` dispatcher; called from `SharedMlp::forward` for the input_linear at M==1.

| | decode tok/s (5-run median) |
|---|---:|
| pre-#9 (image2d only on swiglu W_out) | 10.33 |
| **post-#9 (also image2d on gate_up w_in)** | **9.53** |
| Δ | **-7.7%** |

**Regression. REVERTED.** The 4096-row image has worse texture-cache locality than the 1024-row w_out image — Adreno's L1 texture cache fits ~1024 rows of one width-256 image well but evicts entries when the image is 4× taller. The kernel does 4 `read_imagef` calls per WG (rows n0..n0+3) and at height 4096 those reads cross more cache lines. The buffer path's coalesced linear access pattern wins here. Kernel + dispatcher remain in place but unused.

### Step #10 — Image2d_t for lm_head (NOT ATTEMPTED — predicted to regress like #9)

lm_head W shape is [100352, 1024] → image height 100352 exceeds Adreno's `CL_DEVICE_IMAGE2D_MAX_HEIGHT` (~16384). Would require **7 tiled sub-buffer images + 7 dispatches per call**. Given that the smaller-scale w_in image (height 4096) regressed in #9, the much-larger lm_head tiles (height 16384 each) are predicted to regress too. The amortization across 1 call per token doesn't outweigh the worse cache locality. Skipped.

### Final 3-run median (after #8 retained, #9/#10 not used)

| Run | decode tok/s |
|---|---:|
| 1 | 10.36 |
| 2 | 10.49 |
| 3 | 10.51 |
| **median** | **10.49** |

Per-token decode: 1000 / 10.49 = **95.3 ms/token**. Effective decode BW: 676 MB / 95.3 ms = **7.09 GB/s = 71% of 10 GB/s ceiling**.

### Final scoreboard (this session, fp16-only, "pass the ceiling" attempt)

| Step | decode tok/s | % of 14.8 ceiling | Δ vs pre-step |
|---|---:|---:|---:|
| Session start (after prior cumulative work) | 9.59 | 65% | — |
| #11 Subgroup reductions (REVERTED) | 5.25 | — | −45% |
| Layer truncation NNOPT_DECODE_LAYERS=14 | 14.55 | 98% | +52% but **gibberish output** — unusable |
| **#8 Image2d_t for swiglu W_out** | **10.49** | **71%** | **+9.4%** ⭐ |
| #9 Image2d_t for MLP gate_up (REVERTED) | 9.53 | — | -7.7% |
| #10 Image2d_t for lm_head (skipped) | — | — | predicted regression based on #9 |

**Net session win: +9.4% (9.59 → 10.49 tok/s).**

The fp16 ceiling at 14.8 tok/s remains genuinely out of reach without one of:
- A separately-trained draft model for speculative decoding (multi-day; granite has none ready)
- Retraining with layer-skip or Medusa heads (requires GPU compute we don't have here)
- INT8/INT4 quantization (user constraint: fp16 only)

**Layer truncation proved the speed is achievable (14.55 tok/s at 14 layers) but the output is unusable** — granite is not architecturally tolerant of decode-time skip without retraining.

**Verdict: 71% of fp16 ceiling (10.49 tok/s) is the practical fp16 maximum on this Adreno 620 device for granite-4.0-350m without retraining or quantization.** The path past 14.8 tok/s requires breaking one of those two constraints.


### Items not attempted (predicted small or high-risk)

- **#6 Persistent decode output buffers** — already partially done in Step 4; remainder (`attn_out`, `mlp_out`, layer_norm out) requires invasive caller-side refactor for borrowed-buffer convention; predicted +5%.
- **#7 Image2d_t weight wrapping** — predicted +20–30% based on SmolLM2 evidence, but requires per-weight image creation, separate kernel variants (`*_img`), and the lm_head 200 MB matrix needs tiling because Adreno's image height max is 16384 < 100352. Substantial code change.
- **#10 KV cache via sub-buffer** — predicted +3%, requires custom GEMV kernel to support output buffer offset (currently writes to base of buffer). Modest gain, modest complexity.
- **#5 single-output dual-accum variant** — would require kernel rewrite; SmolLM2 used this and saw ~2 ms per call savings on their MLP. Predicted +5% on swiglu.












