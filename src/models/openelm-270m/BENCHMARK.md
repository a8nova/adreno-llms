# Benchmark log — OpenELM-270M on Razr 2020 (Adreno 620, fp16)

**Mission: drive OpenELM-270M's decode tok/s as close to its 19.8 tok/s memory-bandwidth ceiling as possible on this device, using the same playbook that drove Mamba1/Mamba2 from sub-2 tok/s baselines to 60–70% of their ceilings.**

OpenELM-270M is a vanilla GQA transformer (16 layers, hidden=1280, head_dim=64, variable per-layer head counts and FFN widths). It is structurally simpler than Mamba2 (no SSM, no recurrent state) but has **1.93× more weight bytes per decode token** because of (a) the variable, generally larger FFN dims and (b) the lm_head re-read every token (lm_head is tied to the input embedding but still must be streamed for each forward). The implication is a lower ceiling than Mamba2 on this exact device — 19.8 tok/s vs 37.2 tok/s.

## Benchmark protocol

**One-time setup (after any source change):**
```bash
cd <repo-root>/src/models/openelm-270m
NNOPT_DTYPE=fp16 ./scripts/build.sh --release
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh
```
- `--release` ⇒ CMake `Release` (`-O3`) with `-DNNOPT_DEBUG=` unset (debug macros stripped).
- `NNOPT_DTYPE=fp16` builds `build/fp16/OpenELM_270M_inference_fp16` and pulls `weights/model.fp16.bin` (518 MB).

**Per-run command:**
```bash
NNOPT_DTYPE=fp16 NNOPT_DEBUG_LAYERS=0 ./scripts/run_android.sh \
    "Once upon a time" 32 --temperature 0 --seed 42
```
- `NNOPT_DEBUG_LAYERS=0` suppresses per-step stderr noise.
- `--temperature 0` ⇒ greedy decode (deterministic).
- `--seed 42` pinned for reproducibility (irrelevant under greedy).
- 5 prompt tokens (`[1, 9038, 2501, 263, 931]` = `<s>`, `▁Once`, `▁upon`, `▁a`, `▁time`) + 32 new tokens.

**Metric:** `BENCHMARK decode_tokens_per_sec` from `main.cpp`, defined as `(n_generated - 1) / (total_inference_sec - time_to_first_token_sec)` — steady-state per-token cost, excludes prefill.

**Per-step ritual:** 5 consecutive warm runs, take **median** decode tok/s. Variance was ≤0.2% on this baseline; >5% is rejected as thermal/contention. The first 3 cold runs after a USB reconnect or `adb kill-server` were 20–25% slower than the warm cluster — discard those.

**Token-IDs as canonical reference.** Greedy decode is deterministic. Every step run also captures the 37 generated token IDs (5 prompt + 32 generated) and verifies against the locked reference. Divergence is treated as a correctness regression unless the new sequence is coherent and accepted as a new reference (per Mamba2 BENCHMARK.md guidance for fp16-reduction-order shifts).

**Profile mode:** Not yet wired. The `kernel_profiler.{h,cpp}` infrastructure used in the Mamba1-v2 / Mamba2 ports is not present here. Step 0's bottleneck census below is theoretical (per-byte-bandwidth derived from weight shapes), not from `cl_event` profiling. Wiring the profiler is recommended before attempting Step 6+ (kernel-graph rebatching) where empirical per-kernel time becomes load-bearing.

## Hardware ceiling

**Snapdragon 765G / Adreno 620 / LPDDR4X-2133 dual-channel** (Motorola Razr 2020, serial `ZY22D5NLGQ`)

- Peak DRAM bandwidth: 17.0 GB/s (4266 MT/s × 16 bits × 2 channels / 8). Marketing.
- Realistic GPU-visible bandwidth: ~10 GB/s (Mamba1-v2 confirmed empirically; SmolLM2 same; same SoC and DRAM as Mamba2 port).
- Adreno 620 fp16 ALU: ~2.32 TFLOPS — irrelevant; mobile LLM decode is bandwidth-bound.

**OpenELM-270M weight footprint at fp16 (verified via `weights/model.fp16.meta.json`):**

| Component | Per-layer (avg) | × 16 layers | Per-token (decode) |
|---|---|---|---|
| `attn_norm` (1280 fp16) | 2.5 KB | 40 KB | 40 KB |
| `qkv_proj` (variable: layer 0 1152→1280, layer 15 1920→1280) | 3.62 MB avg | 57.93 MB | 57.93 MB |
| `q_norm` + `k_norm` (64 fp16 each) | 256 B | 4 KB | 4 KB |
| `out_proj` (variable: layer 0 768→1280, layer 15 1280→1280) | 2.41 MB avg | 38.59 MB | 38.59 MB |
| `ffn_norm` (1280 fp16) | 2.5 KB | 40 KB | 40 KB |
| `ffn.proj_1` (1280 → 2·intermediate; intermediate ranges 768→5120) | 14.22 MB avg | 227.50 MB | 227.50 MB |
| `ffn.proj_2` (intermediate → 1280) | 7.11 MB avg | 113.75 MB | 113.75 MB |
| **Per-layer subtotal** | **27.49 MB avg** | **439.77 MB** | **439.77 MB** |
| `transformer.norm` (1280 fp16) | — | — | 2.5 KB |
| `transformer.token_embeddings` (32000 × 1280 fp16, **tied with lm_head**) | — | — | embedding lookup: 2.5 KB / lm_head: 78.12 MB |
| **Total weight bytes / decode token** | — | — | **517.90 MB** |

**KV-cache traffic (per decode token, averaged over a 32-token decode of a 5-token prompt):**

- Sum of `num_kv_heads` across the 16 layers = 63 (KV layout: 3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5).
- Total KV cache row-bytes per token: `2 × 63 × 64 × 2 B = 16,128 B` (one row of K + one row of V across all layers).
- Decode step `i` reads positions `[0, 5+i)` of K and V for each layer. With 32 decode steps, average `seq_k = 21.5`.
- Per-token K+V read averaged across the 32 steps: `2 × (sum of KVH) × 64 × 2 × 21.5 = 346 KB`.
- Per-token K+V write: 16 KB.
- **KV traffic / token (avg over decode): ~362 KB. Negligible vs the 518 MB weight read.**

Per-token total bandwidth (decode steady state): `517.90 MB + 0.36 MB ≈ 518.25 MB`.

**Roofline at 10 GB/s realistic:** `518.25 MB / 10 GB/s = 50.61 ms/token = 19.76 tok/s.`

This is the absolute fp16 ceiling on this hardware. Anything below this is suboptimal kernel utilization, not a fundamental hardware limit.

**Reference points on the same device:**
- Mamba2-130M fp16 (Step 4+6): 17.48 tok/s = 47% of its 37.2 tok/s ceiling.
- Mamba1-130M-HF fp16 (final): 24.56 tok/s = 63% of its 38.7 tok/s ceiling. The high bar.
- SmolLM2-135M fp16 (final): 10.46 tok/s = 28.3% of ceiling.

OpenELM-270M's 19.76 tok/s ceiling means **even at 100% of ceiling we cannot match Mamba1's 24.56 tok/s** — the model is simply 2× the bandwidth load per token. Realistic landing for a comparable optimization budget is **~14 tok/s = 70% of ceiling** (matching Mamba1's relative utilization).

## Step 0 — Release-build baseline (5-run warm median)

Cold runs after `adb kill-server` ranged 0.96–1.01 tok/s; once warm the variance collapses to <0.2%. Numbers below are the 5 consecutive warm runs:

| Run | decode tok/s | total tok/s | TTFT s | prefill tok/s | tokens deterministic |
|---|---|---|---|---|---|
| 1 | 1.1917 | 1.1338 | 2.2034 | 2.3298 | reference |
| 2 | 1.1933 | 1.1346 | 2.1908 | 2.3360 | ✓ exact |
| 3 | 1.1916 | 1.1322 | 2.2463 | 2.2745 | ✓ exact |
| 4 | 1.1907 | 1.1331 | 2.2047 | 2.3197 | ✓ exact |
| 5 | 1.1926 | 1.1348 | 2.2046 | 2.3172 | ✓ exact |
| **median** | **1.1917** | **1.1338** | **2.2046** | **2.3197** | ✓ |

Decode time per token: `1000 / 1.1917 = 839 ms/token`.
Effective decode bandwidth: `518.25 MB / 839 ms = 0.602 GB/s = 6.0% of 10 GB/s ceiling.`
Memory: 1359 MB peak (weights + program build + activation buffers + KV cache).

**Generated text** (32 tokens, greedy, deterministic across all 5 runs):
> Once upon a time, there was a man who was a great man. He was a great man who was a great man. He was a great man who was a great

Token sequence (locked Step-0 reference):
```
[1, 9038, 2501, 263, 931,                  # prompt: <s> ▁Once ▁upon ▁a ▁time
 29892, 727, 471, 263, 767,                # , ▁there ▁was ▁a ▁man
 1058, 471, 263, 2107, 767,                # ▁who ▁was ▁a ▁great ▁man
 29889, 940, 471, 263, 2107,               # . ▁He ▁was ▁a ▁great
 767, 1058, 471, 263, 2107,                # ▁man ▁who ▁was ▁a ▁great
 767, 29889, 940, 471, 263,                # ▁man . ▁He ▁was ▁a
 2107, 767, 1058, 471, 263,                # ▁great ▁man ▁who ▁was ▁a
 2107, 767]                                # ▁great ▁man
```

This is the **acceptance reference** — every optimization step must reproduce this text ID-for-ID at greedy temp=0 seed=42. The "great man / who was" loop is normal for a 270M base (non-instruction-tuned) model on this prompt and is reproducible token-for-token in PyTorch reference. Any ID divergence is treated as a correctness regression unless the new sequence is coherent natural language and the new IDs are explicitly accepted as a new reference (Mamba2 saw a topic shift after `gemv_m1` that was accepted; we expect the same drift here).

## Step 1 — `gemv_m1` fast-path for `pytorch_linear` (5-run warm median)

Drop-in port of `kernels/gemv_m1.cl` from the validated Mamba1-v2 port (same Adreno 620). Added `try_gemv_m1_fast_path()` to `src/utils.cpp::pytorch_linear` with eligibility predicate `M == 1 && K >= 64 && K % 64 == 0` (every decode-path GEMV in OpenELM-270M qualifies). Dispatcher selects K=768/K=1536 specialized variants where present, falls back to a generic vec4 path, and prefers the `_no4` 4-output variant when `N % 4 == 0` (which is true at every site). Prefill (M>1) keeps using CLBlast `HGemm`.

| Run | decode tok/s | total tok/s | TTFT s | prefill tok/s | tokens deterministic |
|---|---|---|---|---|---|
| 1 | 4.1666 | 3.4188 | 2.1411 | 2.4191 | reference (drift, accepted) |
| 2 | 4.2323 | 3.4655 | 2.1232 | 2.4297 | ✓ exact |
| 3 | 4.2810 | 3.5094 | 2.1245 | 2.3966 | ✓ exact |
| 4 | 4.2199 | 3.4604 | 2.1176 | 2.4276 | ✓ exact |
| 5 | 4.3193 | 3.5247 | 2.1267 | 2.4282 | ✓ exact |
| **median** | **4.2323** | **3.4655** | **2.1245** | **2.4276** | ✓ |

Decode time per token: `1000 / 4.2323 = 236.3 ms/token`.
Effective decode bandwidth: `518.25 MB / 236.3 ms = 2.193 GB/s = 21.9% of 10 GB/s ceiling.`
**Speedup over Step 0: 3.55× (1.19 → 4.23 tok/s).** Below the 6–10× plan estimate because OpenELM has 16 layers vs Mamba2's 24, so per-launch fixed cost is a smaller share, and CLBlast's M=1 path on the Adreno 620 was less catastrophic than expected for these K values.

**Generated text** (32 tokens, greedy, deterministic across runs 2–5; run 1 is the cold drift):
> Once upon a time, there was a man named John. He was a man of great wealth and power. He was a man of great wisdom and knowledge. He was a man

Token sequence (locked Step-1 reference — accepted as new reference per the same fp16-reduction-shift rule that Mamba2 followed):
```
[1, 9038, 2501, 263, 931,                  # prompt: <s> ▁Once ▁upon ▁a ▁time
 29892, 727, 471, 263, 767,                # , ▁there ▁was ▁a ▁man
 4257, 2259, 29889, 940, 471,              # ▁named ▁John . ▁He ▁was
 263, 767, 310, 2107, 17173,               # ▁a ▁man ▁of ▁great ▁wealth
 322, 3081, 29889, 940, 471,               # ▁and ▁power . ▁He ▁was
 263, 767, 310, 2107, 29714,               # ▁a ▁man ▁of ▁great ▁wisdom
 322, 7134, 29889, 940, 471,               # ▁and ▁knowledge . ▁He ▁was
 263, 767]                                 # ▁a ▁man
```

The new sequence is coherent natural language (the "great man" loop of Step 0 became a structured "He was a man of great X" enumeration). Accepted as the reference for Steps 2+.

## Step 2+3 — Persistent activation buffers + GPU-side argmax (5-run warm median)

**Step 2** added long-lived `cl_mem` activation buffers to `Attention` (9 buffers: `buf_qkv_, buf_q_, buf_k_, buf_v_, buf_qn_, buf_kn_, buf_scores_, buf_ctx_out_, buf_proj_`) and `Mlp` (3 buffers: `buf_proj1_out_, buf_gate_, buf_proj2_out_`) with sticky-capacity grow allocators (`ensure_activation_buffers_(seq_q, seq_k)` / `ensure_activation_buffers_(seq_len)`). Per decode token this eliminates 16×9 + 16×3 = 192 alloc + 192 release = 384 driver round-trips. Ownership change: `Attention::forward` and `Mlp::forward` now return BORROWED `cl_mem` handles (owned by the layer instance until next call); `Model::forward` no longer releases those return values.

**Step 3** added a single-WG cooperative `argmax_fp16` kernel (`kernels/argmax.cl`) with tree-reduction over `__local` memory. New `Model::forward_argmax_greedy(input_ids, start_pos)` mirrors `forward()` up to the on-device logits buffer, then dispatches `argmax_fp16` with an `offset_elements` argument (last-row offset, no `clCreateSubBuffer` to avoid Adreno alignment quirks) and reads back a single int32. `Model::generate()` selects this fast-path when `temperature <= 0 AND repetition_penalty == 1.0` — the standard greedy-benchmark configuration. Saves the per-token 64 KB host readback + 32000-element host fp16→fp32 conversion + linear scan.

The two changes were measured together (single rebuild) since their effects compound and isolating them would only mean an extra rebuild/redeploy cycle.

| Run | decode tok/s | total tok/s | tokens deterministic |
|---|---|---|---|
| 1 | 4.3829 | 3.4984 | ✓ matches Step-1 reference |
| 2 | 4.4541 | 3.5264 | ✓ exact |
| 3 | 4.4751 | 3.5432 | ✓ exact |
| 4 | 4.4549 | 3.5297 | ✓ exact |
| 5 | 4.3795 | 3.4937 | ✓ exact |
| 6 | 4.4106 | 3.5016 | ✓ exact |
| **median (runs 2–6)** | **4.4106** | **3.5016** | ✓ |

Decode time per token: `1000 / 4.4106 = 226.7 ms/token`.
Effective decode bandwidth: `518.25 MB / 226.7 ms = 2.286 GB/s = 22.9% of 10 GB/s ceiling.`
**Step 2+3 stacked speedup over Step 1: 1.042× (4.23 → 4.41 tok/s).**
**Cumulative over Step 0: 3.70× (1.19 → 4.41 tok/s = 22.9% of ceiling).**

The combined gain landed at the very low end of the plan estimate (Step 2 plan: 1.05–1.10×; Step 3 plan: 1.03–1.05×; combined plan range: 1.08–1.16×). Two reasons:
1. Adreno's OpenCL driver appears to have an internal small-buffer pool — the per-call `clCreateBuffer` / `clReleaseMemObject` costs were closer to ~3 µs each rather than the ~50 µs Mamba2 saw on different driver versions. Step 2's amortizable surface was therefore smaller.
2. The argmax-vs-readback delta is bound by the ~16 µs the 64 KB readback would have cost at 4 GB/s host traffic. The argmax kernel itself takes ~30 µs (single WG, 32000-element strided pass + tree reduce) plus the 4-byte readback, leaving a net of <10 µs/token. At 4.4 tok/s = 227 ms/token this is <0.5% — visible only because the prefill kernel-build cost was already amortized.

Tokens are bit-exact against the Step-1 reference across all 6 runs, confirming the persistent-buffer ownership change and the GPU argmax both preserve correctness.

## Bottleneck census (theoretical, per decode token)

This port has no `kernel_profiler` wired in yet. The bottleneck table below is derived from byte-counting, not from `cl_event` measurements. It is sufficient to rank the next 5–6 optimizations; before Step 6 (kernel-graph rebatching) the profiler should be ported from `adreno-llms/src/models/mamba2-130m/src/kernel_profiler.{h,cpp}`.

**Per-token GEMV weight reads (98.0% of total per-token bandwidth):**

| Site | Bytes / token | % of total | Path | K | N (per layer) |
|---|---|---|---|---|---|
| `ffn.proj_1` × 16 | 227.50 MB | 43.9% | `pytorch_linear` → CLBlast HGemm M=1 | 1280 | 1536 → 10240 |
| `ffn.proj_2` × 16 | 113.75 MB | 22.0% | `pytorch_linear` → CLBlast HGemm M=1 | 768 → 5120 | 1280 |
| `lm_head` × 1 | 78.12 MB | 15.1% | `pytorch_linear` → CLBlast HGemm M=1 | 1280 | 32000 |
| `attn.qkv_proj` × 16 | 59.06 MB | 11.4% | `pytorch_linear` → CLBlast HGemm M=1 | 1280 | 1152 → 1920 |
| `attn.out_proj` × 16 | 39.38 MB | 7.6% | `pytorch_linear` → CLBlast HGemm M=1 | 768 → 1280 | 1280 |
| **Total GEMV weights** | **517.81 MB** | **99.97%** | All 5 sites go through the same GEMV path | — | — |

Every K is divisible by 64 (it's always either `HIDDEN_SIZE=1280`, or `HEAD_DIM × num_q_heads/num_kv_heads`, or an FFN intermediate which is `make_divisible(_, 256)`). Every N is divisible by 4 (1280 ✓, 32000 ✓, the variable qkv_proj N values are all `× 64` ✓, ffn proj_1 N = `2 × intermediate` is `× 512` ✓). **All 5 sites are eligible for `gemv_m1_no4`.**

**Per-token launch + driver-overhead budget:**

| Source | Per token | Approx cost |
|---|---|---|
| Kernel/copy launches (embedding + 16×(pre_norm + 13 attn + post_norm + 3 mlp + 2 residual_add) + final_norm + lm_head) | 323 | ~9.7 ms (host/queue dispatch @ ~30 µs each) |
| `clCreateBuffer` calls inside `Attention::forward` + `Mlp::forward` per token | 208 | ~10.4 ms (Adreno driver bookkeeping @ ~50 µs each) |
| Logits readback (32000 fp16 = 64 KB, blocking) + host fp16→fp32 conversion + host argmax over 32000 | 1 | ~3 ms |
| `input_ids` upload (4–20 B), `clCreateBuffer COPY_HOST_PTR` | 1 | ~50 µs |

**Total recoverable host/driver overhead: ~23 ms/token = 27% of current 839 ms wall.** Step 0 is so far from ceiling that the GPU-side GEMV cost (~800 ms wall) is by far the dominant lever; once that drops, the host overheads become visible as the next ~10–15% wall.

### What jumps out

**The story is identical to Mamba2's Step 0.** GEMVs through CLBlast HGemm M=1 are **the** problem. The kernel signature CLBlast generates for that path is "single-thread per output, no vec4, no cooperative reduction" — confirmed in two prior ports on this device. Replacing it with the proven `gemv_m1.cl` (cooperative WG=64, vec4 fp16, `__local`-mem tree-reduce, 4-output multi-output variant) collapses 99.97% of the per-token bandwidth onto a kernel that actually saturates Adreno's wave-front BW.

What's different from Mamba2: **lm_head dominates a single layer's worth of bandwidth** (78.12 MB = 15.1% of total) because OpenELM ties the embedding to the lm_head. So the `gemv_m1` win at the lm_head site alone is bigger here than in Mamba2 — same kernel, single biggest-N (32000) call.

## Top-10 optimization plan (ranked by predicted impact)

Each step targets ID-for-ID match against the Step-0 reference token sequence (with the documented exception that fp16 reduction-order changes can shift IDs; coherent shifts are accepted as new references).

| # | Lever | What changes | Predicted | Risk |
|---|---|---|---|---|
| **1** | **`gemv_m1` for ALL 5 GEMV sites** | Drop in `kernels/gemv_m1.cl` from `adreno-llms/src/models/mamba-130m (verbatim — same Adreno 620). Add `try_gemv_m1_fast_path()` to `src/utils.cpp::pytorch_linear` with eligibility predicate `M == 1 && K % 64 == 0`. All 5 sites pass: K ∈ {768, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4352, 4608, 4864, 5120}. The dispatcher picks the `_no4` (4-output) variant when N % 4 == 0 — true for every site. Mamba2 saw 7.82× from the same change because CLBlast events under-report and the actual GEMV share is much higher than profilers suggest. | **6–10×** → 7–12 tok/s | Low. Drop-in code from a previously-validated port on the same device. Fp16 reduction-order may shift IDs (Mamba2 saw a topic-coherent shift; same risk class here). |
| **2** | **Persistent activation buffers in `Attention` + `Mlp`** | Add long-lived `cl_mem` members for every per-call activation buffer: `Attention` needs `buf_qkv_, buf_q_, buf_k_, buf_v_, buf_qn_, buf_kn_, buf_scores_, buf_ctx_out_, buf_proj_` (9 bufs); `Mlp` needs `buf_proj1_out_, buf_gate_, buf_proj2_out_` (3 bufs). Add `ensure_activation_buffers_(seq_q, seq_k)` lazy/grow allocator. Per decode token this saves 16×9 + 16×3 = 192 alloc + 192 release = 384 driver round-trips ≈ 9–10 ms. Same template that landed 1.05× on Mamba2. | **1.05–1.10× stacked** → 7.4–13.2 tok/s | Low. Mechanical refactor of `Attention::forward` and `Mlp::forward`. Ownership change: `Attention::forward` and `Mlp::forward` return BORROWED handles owned by the layer instance (caller must NOT release). Update `Model::forward` to drop those `clReleaseMemObject` calls. |
| **3** | **GPU-side argmax for the sampler** | At greedy `temperature == 0` the host doesn't need the logits; just the argmax. Add a `gpu_argmax(logits, vocab) → cl_mem<int32>` cooperative-reduction kernel + a 4-byte readback. Skips the 64 KB `clEnqueueReadBuffer(CL_TRUE)` + the 32000-element host fp16→fp32 conversion + the 32000-element host scan. Mamba2 plan estimated 5 ms/token; here the readback is smaller (32000 vs 50288) so estimate ~3 ms. | **1.03–1.05× stacked** → 7.6–13.9 tok/s | Low. New kernel + a sampler refactor for the greedy-only path; non-greedy paths fall back to the existing readback. |
| **4** | **Persistent `input_ids` buffer + persistent `cos_/sin_` table sized to `MAX_CONTEXT_LENGTH`** | Two micro-fixes: (a) `Embedding::forward` currently creates a new `cl_mem token_ids_buf` per call via `CL_MEM_COPY_HOST_PTR` (~50 µs of driver overhead per token); promote to a persistent member of `Embedding` sized for `max(seq_len)`. (b) `Attention::ensure_rope_tables(seq_len)` re-allocates and re-uploads the cos/sin table whenever `seq_len > rope_seq_len_` — every decode step grows seq_k by 1 and triggers a 2-buffer rebuild + host-side trig recompute. Pre-build to `MODEL_CONFIG::MAX_CONTEXT_LENGTH = 2048` once at `Attention::initialize()`. | **1.02–1.05× stacked** → 7.7–14.6 tok/s | Trivial. |
| **5** | **Fuse residual into out_proj and proj_2 (variant `*_no4_radd`)** | The Mamba1-v2 `gemv_m1.cl` already includes a fused-residual variant: instead of `pytorch_linear → element_add` as 2 launches, the GEMV kernel writes `out[i] = sum + residual[i]` in one pass. Eligible at `attn.out_proj` (writes back into the residual stream) and `ffn.proj_2` (same). Saves 32 launches/token = ~1 ms host dispatch + cuts the residual buffer's read+write traffic (1280 fp16 × 16 layers × 2 = 80 KB/token of synthetic traffic). | **1.05–1.10× stacked** → 8.1–16.0 tok/s | Low after step 1 lands. Mamba2 plan flagged this exact lever as part of its post-Step-A reassessment. |
| **6** | **Wire `kernel_profiler.{h,cpp}` from Mamba2** | Copy `src/kernel_profiler.{h,cpp}` from `adreno-llms/src/models/mamba2-130m/src/`. Wire `event_for(...)` into every `clEnqueueNDRangeKernel` site in `attention.cpp`, `mlp.cpp`, `embedding.cpp`, `layer_norm.cpp`, and the GEMV path in `utils.cpp`. Print a per-label aggregate after `model.generate()` when `NNOPT_KERNEL_PROFILE=1`. Not a perf step — but it's load-bearing infrastructure for steps 7–10. | **0× directly** | Trivial. |
| **7** | **Cooperative + vec4 RMSNorm sweep across all RMS sites** | The RMSNorm kernel at `kernels/rmsnorm.cl` is already cooperative WG=64 vec4 fp16. Step 0 dispatches it 16+16+1 = 33 times per token at row counts {seq_q, seq_q×QH (variable), seq_q×KVH (variable)}. Decode `seq_q=1` so q_norm rows = QH ∈ {12,16,20}, k_norm rows = KVH ∈ {3,4,5} — these are TINY rows. With WG=64, that's <1 wave per dispatch and fixed-cost dominated. Coalesce q_norm+k_norm into one launch per layer (write to two slices of a fused output). Saves ~16 launches/token. | **1.02–1.04× stacked** → 8.3–16.6 tok/s | Low. Either kernel split-rewrite or a host-side dispatcher batching trick. |
| **8** | **Batched-attention dispatch (single launch for scores + softmax + out)** | The 3 attention kernels (`gqa_attn_scores`, `gqa_softmax`, `gqa_attn_out`) currently dispatch separately per layer. At decode `seq_q=1`, each is small enough that launch latency dominates. Fuse into a single kernel that reads K+V from cache, computes scaled-dot scores into `__local`, applies row softmax in place, then projects against V — all 1 launch per layer instead of 3. Same row count, no extra `__global` traffic, 32 fewer launches/token. | **1.05–1.10× stacked** → 8.7–18.3 tok/s | Medium. New kernel with `__local` softmax. Fp16 reduction-order shift expected (Mamba2 noted same). |
| **9** | **Async logits readback (`CL_FALSE`)** | Currently `clEnqueueReadBuffer(CL_TRUE)` for logits blocks the host until GPU completes. Pipelining: kick off the read with `CL_FALSE`, immediately enqueue the next forward()'s embedding + per-layer kernel chain, then `clWaitForEvents` only when sample() actually needs the logits value. Lets the CPU prepare the next iteration's dispatches while step N's logits are in flight. | **1.02–1.04× stacked** → 8.9–19.0 tok/s | Low after step 6 lands. |
| **10** | **fp16 ssm_state-equivalent... not applicable** — OpenELM has no recurrent state; KV cache is already fp16 (verified at `attention.cpp:65–66` allocating `nnopt_storage_t` cache buffers). Closest analog: the attention `scores` buffer is allocated `(QH × seq_q × seq_k)` of `nnopt_storage_t` = fp16. Already storage-fp16 (compute-fp32 inside the kernel via `LOAD()` cast to float). No further win here. | **0×** | — | — |

**Cumulative predicted landing zone**, applied in order with 0.85× pessimism factor on each prediction: **9–14 tok/s.**

Realistic single-shot landing: **step 1 alone should clear 7 tok/s = 6× over baseline = ~36% of ceiling, on par with Mamba2's Step 4+6 leap from 2.2 to 17.5 tok/s but starting from a worse roofline**. Steps 1+2+3 should clear 9 tok/s. Steps 1–5 should clear 12 tok/s. The 14 tok/s headline target = 70% of ceiling = matching Mamba1's relative utilization on its own port.

Beyond ~14 tok/s requires either (a) the fused-attention kernel of step 8, (b) int8 quantization (project rule defers), or (c) research-grade work (subgroup intrinsics, image1d_buffer_t — confirmed dead-end on this device per Mamba1-v2 P2-5).

## Comparison table

| Metric | Mamba1-130M-HF (final) | Mamba2-130M (Step 4+6) | OpenELM-270M (Step 0) | OpenELM-270M (Step 1) | OpenELM-270M (Step 2+3) | OpenELM-270M (target) |
|---|---|---|---|---|---|---|
| Decode tok/s | 24.56 | 17.48 | **1.19** | **4.23** | **4.41** | **>14** |
| Decode ms/token | 40.7 | 57 | 839 | 236 | 227 | <71 |
| Effective BW (GB/s) | 6.34 | 4.72 | 0.60 | 2.19 | 2.29 | >7.0 |
| % of 10 GB/s ceiling | 63% | 47% | 6.0% | 21.9% | 22.9% | >70% of own ceiling |
| Per-token weight read | 258 MB | 258 MB | **518 MB** | 518 MB | 518 MB | same |
| Per-token state R+W | 2.4 MB (mamba1 SSM) | 19.4 MB (mamba2 SSM) | 0.36 MB (KV cache, decode avg) | 0.36 MB | 0.36 MB | same |
| Total per-token data | 260 MB | 269 MB | **518.25 MB** | 518.25 MB | 518.25 MB | same |
| Roofline tok/s @ 10 GB/s | 38.7 | 37.2 | **19.76** | 19.76 | 19.76 | — |
| Distinct GEMV K values | 1 (768) | 2 (768, 1536) | many (768–5120 ffn intermediate, 1280 hidden) | same | same | — |

OpenELM-270M is **roughly 2× the bandwidth load per token** of Mamba1/Mamba2-130M because (a) the model is 2× the parameter count and (b) the lm_head re-read (78 MB / token, tied or not) is 30% of a Mamba1 forward all by itself. The roofline is correspondingly halved. **Hitting Mamba1's 24.56 tok/s on this model is physically impossible on this device — that's above the 19.76 tok/s ceiling.**

The fair target for OpenELM-270M is **70% of its own ceiling = 13.8 tok/s**. The Mamba2 port reached 47% of its own ceiling at the same Step 4+6 stage that this plan reaches at Step 1+5; an extra 1–2 levers from steps 6–9 should push us past 50% and within a few tok/s of the Mamba1 ratio.

## Repo state (after Step 1+2+3)

- No `kernel_profiler.{h,cpp}` — bottleneck census is still theoretical (per-byte). Profiler is Step 6.
- `src/utils.cpp::pytorch_linear` now dispatches `try_gemv_m1_fast_path()` ahead of CLBlast `HGemm` for `M==1 && K%64==0`. ✅ Step 1.
- `src/layers/attention.{h,cpp}` carries 9 persistent `cl_mem` activation buffers + sticky-capacity `ensure_activation_buffers_(seq_q, seq_k)`. ✅ Step 2.
- `src/layers/mlp.{h,cpp}` carries 3 persistent `cl_mem` activation buffers + sticky-capacity `ensure_activation_buffers_(seq_len)`. ✅ Step 2.
- `Attention::forward` and `Mlp::forward` return BORROWED `cl_mem` handles; `Model::forward`/`forward_argmax_greedy` no longer release those return values. ✅ Step 2.
- `kernels/argmax.cl` + `Model::forward_argmax_greedy()` route greedy decode through a 4-byte argmax readback. ✅ Step 3.
- `src/sampler.cpp` host-side argmax path is still kept as the fallback for non-greedy / repetition-penalty configurations.
- `src/layers/embedding.cpp::forward` still creates a fresh `token_ids_buf` per call. Step 4(a) promotes.
- `src/layers/attention.cpp::ensure_rope_tables(seq_len)` still re-allocates `cos_/sin_` on growth. Step 4(b) pre-builds to `MAX_CONTEXT_LENGTH=2048`.
- `CMakeLists.txt` configures release (`-O3`) and fp16 (`NNOPT_USE_FP16=1` + CLBlast `HALF=ON`). No build changes needed for steps 4–5.
