# Benchmark log — Bonsai on Razr 2020 (Adreno 620, Q1_0)

All numbers: release build, PROCESS WALL, `scripts/run_android.sh`,
fp32 activations (the token-exact configuration; fp16 measured SLOWER).

## Benchmark protocol

**One-time setup (after any source change):**
```bash
cd <repo-root>/src/models/bonsai
./scripts/build.sh --release          # BONSAI_STORAGE=fp32 (default, token-exact)
./scripts/deploy_android.sh           # pushes binary + kernels; the .nnb once
```
- `--release` ⇒ CMake `Release` (`-O3 -ffast-math`).
- fp32 activations (default) is the **token-exact** configuration; fp16 measured *slower*.
- `BONSAI_NNB=bonsai4b.nnb ./scripts/deploy_android.sh` selects the 4B bundle.

**Per-run command:**
```bash
./scripts/run_android.sh "Tell me a fun fact about foxes." 64 chat
```
- Greedy decode (deterministic) — token IDs are the acceptance reference.
- **Metric:** `BENCHMARK decode_tokens_per_sec` (PROCESS WALL, excludes prefill).
- Gates: `reference/golden_{1,2,3}.json` (llama-server oracle, 64 greedy tokens).
- Env: `BONSAI_PROF=1` (per-op profile), `BONSAI_TOPK=1` (top-5 logits),
  `BONSAI_KERNEL_OPTS` (extra kernel build flags), `DUMP_DIR` (host layer dumps).

## Headline (2026-07-17)

| metric | value |
|---|---|
| decode, sustained | **1.95 tok/s** (0.514 s/token marginal, 64-token runs) |
| model load + weight upload (warm) | ~3.5 s |
| thermal drift over 3 consecutive 64-token runs | +0.2% (flat) |
| correctness | TOKEN-EXACT vs llama.cpp oracle, 3/3 goldens × 64 greedy tokens |
| memory | 1.16 GB weights (never dequantized) + ~600 MB KV fp32 @ ctx 2048 |

## Optimization ladder (each step re-verified token-exact)

| kernel | s/token | tok/s | note |
|---|---|---|---|
| v0 WG-per-row, byte loads, convert+dot | 4.56 | 0.22 | correctness baseline |
| v2 +split-stream, local-x, mask-AND | 10.6 | 0.09 | REGRESSION: per-WG local staging ×12288 WGs |
| v3 K-major transpose, thread-per-row | 1.53 | 0.65 | the moonshine gemv_t shape; no barriers/local |
| v4 quad-row per thread (wave-interleaved) | 0.80* | 1.25* | x L2 traffic ÷4 (*serialized profile) |
| v4 true async wall | **0.514** | **1.95** | pipelining hides launch gaps |
| v5 oct-row | regression | — | Adreno register cliff (8 float4 partials) |
| v6 v4+local-x | 9.5s/8tok prof | — | local loses to L2 broadcast (again) |
| fp16 activations | 12.6s/8tok prof | — | vload_half converts cost > bandwidth saved |
| v7 8-bit-LUT (16KB lmem, 2048 rows/WG) | 2.2× worse | — | lmem bank conflicts on per-lane random LUT reads + 3 barriers/unit |
| device-fed decode loop (no host readback) | no-op | — | token chain is GPU-serial: step i+1's gather needs step i's argmax; nothing to overlap |
| #5 batched prefill (M=8 GEMV) | 1.5s/tok prefill (REGRESSION) | — | GEMV is ISSUE-bound not memory-bound; batching only amortizes weight streaming, so same op count + worse register pressure. A true GEMM tile would win but hits the Adreno register cliff. REVERTED. |
| #6 4-bit KV cache (naive per-group symmetric) | no speed change | — | attention is ~2% of decode; KV4's value is MEMORY (604→78MB, 7.8×; enables ctx 4096 in 156MB). BUT naive quant degrades output — whitepaper's near-lossless needs their kv-mean-center calibration (footnote §4.4), which I did not implement. Env toggle BONSAI_KV4=1, experimental. |

Other wins: fused QKV (one N=6144 launch; k/v alone were 4 workgroups) and
fused gate+up (N=24576); on-device argmax (1-int readback); split-stream
weight repack at upload (aligned uint4 sign words + fp16 scale array —
still 1.125 bit/weight, just deinterleaved); logits GEMV skipped during
prefill.

## Bugs the token-exact gates caught (would have shipped silently)

1. **Adreno wraps workgroup ids at 65535/dim**: the 151669-row logits GEMV
   duplicated low-row logits into all rows ≥ 65536 — the model could never
   emit `</think>`, high-id tokens, or CJK. Fixed via K-major dispatch
   (2370 groups); any future big-N kernel must chunk.
2. Shell prompt transport mangled newlines (`$(...)` strips trailing \n) —
   `@file` prompt convention added.


## Reprofile #2 (2026-07-17b) — CORRECTED diagnosis via pure-read microbench

The earlier "latency-bound" verdict below (Reprofile #1) was WRONG — its bound
test was flawed: the "2× reads free" kernel re-read the SAME cached address (an
L2 hit, not extra bandwidth), and the "2× ALU" got dead-code-eliminated. A
clean isolation microbench (standalone lxbench2, marginal event timing, gate+up
shape N=24576/K=4096) settles it:

- Pure weight read, our K-major access pattern (no compute): **1.29 ms**
- Pure sequential read: **1.06 ms (11.9 GB/s)** — DRAM is fast, NOT the wall
- Full v4 GEMV (read + mask + accumulate): **6.1 ms**

⇒ Only ~1.3 ms is memory; **~4.8 ms (80%) is the integer mask-unpack ALU.**
The decode is **ALU-ISSUE-bound on the 1-bit unpack**, not memory/latency-bound.
Memory has ~5× headroom. Effective streaming ≈ 2.33 GB/s is an ALU ceiling, not
a bandwidth one.

**Every kernel lever re-tested cleanly against v4's 6.1 ms (microbench, same shape):**

| variant | ms | verdict |
|---|---|---|
| **q1_gemv4 (float4 int-mask, quad-row)** | **6.1** | **OPTIMUM — ships** |
| q1_gemv4xor (XOR sign, no xsum) | 6.4 | ~6% worse-to-wash |
| q1_gemv_pf (software prefetch next unit) | 6.5 | no help (compiler already schedules loads) |
| q1_gemv_duo (2 rows/thread) | 8.7 | worse — less x-reuse/ILP |
| q1_gemv9 (1 row/thread, max waves) | 14.1 | worse — 4× x-load issue, 1-way ILP |
| q1_gemv_hex (6 rows/thread) | 14.0 | worse — register spill |
| q1_gemv7 (LUT in local mem) | 10.8 | worse — lmem bank conflicts |
| q1_gemv_lut2 (LUT in global/L2, built once) | 20.5 | worse — 16 serial byte-gather latency chain |
| q1_gemv_fma (float-multiply mask) | 2508 | CLIFF — convert_float4(uint4) pathological |
| q1_gemv_h4 (native half4 fp16 mask) | 359 | CLIFF — Adreno emulates short-vector shift-mask |
| q1_gemv_h8 (half8, 8 weights/op) | 1280 | CLIFF — half8 emulated, not packed |

**Conclusion (now ironclad, not just asserted):** v4 float4 integer-mask is the
*only* fast path the Adreno 620 compiler emits for on-the-fly 1-bit unpack — it
is BOTH the local optimum (fewer rows lose reuse, more rows spill) AND the
global optimum (every fp16/wide/LUT/fma reformulation falls off a compiler
cliff). At ~1 mask-op/weight it sits near the GPU's integer-issue ceiling.
Decode = 1.06 GB/token of weights × ~1 ALU-op/weight ⇒ 2.33 GB/s effective ⇒
**~1.9–2.0 tok/s is the hard floor for a TOKEN-EXACT 8B 1-bit decode on this
GPU.** 4 tok/s needs 1.8× fewer ALU-ops/weight, which this silicon+compiler will
not give. Only stackable on-device lever left is CPU+GPU hybrid (2 A76 cores add
independent unpack ALU, ~1.3× → ~2.4 tok/s — real but still short of 4). The
SAME token-exact binary is projected to 11 tok/s on iPhone 17 (whitepaper Table
8): the path to 4+ tok/s is newer silicon or a smaller model, not a better kernel.

## CPU+GPU hybrid (2026-07-17b) — BUILT, token-exact, but REGRESSES. Env-gated.

Rationale (sound on paper): decode is ALU-issue-bound and the Adreno uses only
~2.3 of ~11 GB/s, so the idle 2×A76+6×A55 cores add INDEPENDENT 1-bit-unpack ALU
with NO bandwidth contention. Microbench confirmed the CPU is real: NEON 8-thread
1-bit GEMV (gate+up shape) = 24.3 ms = 4.14 G-mask-op/s = **25% of the GPU** →
balance math predicted ~1.25× (~2.3 tok/s).

Built it (BONSAI_HYBRID=1): CPU computes the top `frac` rows of the two fat MLP
GEMVs (gate+up→ffn_up tail, down) straight from the mmap'd Q1 weights (zero extra
RAM), same fp32 two-sum → **token-EXACT (64/64 ids identical)**. Fixed a real bug
first: the kernel's N arg is BOTH the row bound AND the transposed weight stride
`bits_t[u*N+row]`, so the GPU half must keep the true N and shrink only the launch
size (n_dispatch, 256-aligned).

Measured (razr, 64-tok decode):
| config | tok/s | vs baseline |
|---|---|---|
| baseline (pure GPU, async) | **1.85** | — |
| hybrid frac=0.22 | 1.27 | −31% |
| hybrid frac=0.08 (tiny CPU) | 1.23 | −34% |
| **readback-only probe (NO cpu/upload)** | **1.42** | **−23%** |

The readback-only probe is decisive: doing JUST the per-op blocking `clEnqueueRead`
of the activation x — no CPU work, GPU still computes the whole op — ALREADY costs
23%. **Root cause: every input activation lives on the GPU, so handing rows to the
CPU needs a host readback, and that readback DRAINS the deep async pipeline** (the
same host-runs-ahead pipelining that earned the 0.80→0.51 s/tok async win). The
drain (~2.25 ms/MLP-op × 72 ops) exceeds what the CPU offload can reclaim, at ANY
frac. Deep GPU pipelining and mid-pipeline CPU offload are MUTUALLY EXCLUSIVE on
this in-order-queue design. A single-readback/token variant (logits-only) was
estimated at ~1% — negligible. Hybrid kept OFF by default (baseline unchanged at
1.85); env toggles BONSAI_HYBRID / _FRAC / _THREADS / _READONLY retained for the
record. **1.85 tok/s stands as the floor.**

## Reprofile #1 (2026-07-17) — SUPERSEDED, flawed bound test kept for the record

Thread-per-row occupancy attempt — DEAD END (measurement caution logged):
- Initial "2× faster" was an ARTIFACT: the thread-per-row kernel got v4's
  quad-row dispatch (ceil(N/256)*64), computing only 1/4 of output rows —
  fast but garbage. The RPT probe that looked like a win was under-covering.
- With CORRECT full-row dispatch (ceil(N/64)*64), thread-per-row (v9,
  byte-identical math to v4's row-0) is token-EXACT but 0.82 tok/s — SLOWER
  than v4's 1.9. Reason: v4 reuses each activation load across 4 output rows;
  thread-per-row reloads x 4× as often. **x-reuse > occupancy on this L2.**
- v4 quad-row is the local optimum: balances x-reuse vs register pressure
  (v5 oct-row spilled; v9 thread-per-row lost x-reuse).

Texture/L1 path (image1d_buffer over weights, BONSAI_IMG=1) — TESTED, no-op:
token-exact but 1.89 tok/s (identical to buffer). The L1 texture cache
accelerates REUSE (image filtering re-reads neighbors); a streaming GEMV
reads each weight EXACTLY ONCE, so there is nothing to cache. Zero-copy
image add was free but the access pattern has no locality for it to exploit.

Speculative decoding (DSpark) — DE-RISKED AND KILLED before building:
the batched verification GEMV scales LINEARLY with draft length K
(measured: 1 tok 6.80ms, 2 tok 14.2ms, 4 tok 32.5ms — 4.8x for K=4).
Spec-decode requires verification-of-K ~= 1 step, which needs a
MEMORY-BANDWIDTH-bound decoder. This decoder is LATENCY/issue-bound
per-(token,weight), so K tokens cost K x the work and batching amortizes
NOTHING (same root cause as the batched-prefill regression). Net spec-decode
speedup A/K <= 1: no win. Drafter infrastructure NOT built — would be
negative ROI on this GPU.

DEFINITIVE: on the Adreno 620 this model's decode is fundamentally
per-token sequential-bound. Neither occupancy, texture, batching, nor
speculation amortizes it. 1.9 tok/s is the hard floor. 4-6 tok/s requires
memory bandwidth this 2020 SoC does not have (needs LPDDR5X, 3-8x); the
SAME token-exact binary is projected to 11 tok/s (iPhone 17) / 26-66 tok/s
(laptops) per the Bonsai whitepaper Table 8.

CONCLUSION: kernel levers exhausted on Adreno 620. Latency-bound with
1.06 GB/token of once-read weights; occupancy fixes lose x-reuse, texture
has no reuse to exploit, and the register budget caps waves. 1.9 tok/s is
the practical limit for this GEMV design on this 2020 GPU. The path to
4-6 tok/s is faster silicon (LPDDR5X budget phones: 3-8x bandwidth) or a
fundamentally different decode (speculative/DSpark drafter from the
whitepaper §6 — amortizes weight streaming over multiple accepted tokens).

## Ceiling analysis

- Weight stream: 1.03 GB/token ÷ ~11 GB/s DRAM ⇒ absolute ceiling ~10 tok/s.
- Measured: GEMVs are ALU-issue-bound (mask ops ≈ 4 vec-ops per 4 weights),
  ~1.7 GB/s effective streaming. The v4 op mix runs at ≈ the device's
  vector issue rate — further gains need fewer ops/weight (theoretical
  ~1.3× via cmp-mask micro-opts) or CPU+GPU hybrid, not memory tuning.
- Plan gate was ≥3 tok/s; shipped at 1.95 sustained per the plan's R5
  clause (honest numbers, no activation quantization).

## Size sweep (2026-07-22) — 1.7B added, 27B ruled out

The runtime never needed a change to take a new size: the forward pass reads
every dim from the `.nnb` header. What *was* size-specific was the CONVERTER —
`convert_to_nnb.py` hardcoded the 8B hyperparameters (hidden 4096, layers 36,
ffn 12288, vocab 151669) and `gguf_parse.py` asserted `len(tensors) == 399`.
The shipped 4B bundle could not be reproduced from the committed scripts. Both
now derive everything from the GGUF metadata block, verified to reproduce the
8B header exactly from `Bonsai-8B-Q1_0.gguf`'s own KV:

| meta field | derived from GGUF key | 8B | 4B | 1.7B |
|---|---|---:|---:|---:|
| hidden | `qwen3.embedding_length` | 4096 | 2560 | 2048 |
| layers | `qwen3.block_count` | 36 | 36 | 28 |
| heads / kv_heads | `qwen3.attention.head_count[_kv]` | 32 / 8 | 32 / 8 | 16 / 8 |
| head_dim | `qwen3.attention.key_length` | 128 | 128 | 128 |
| ffn | `qwen3.feed_forward_length` | 12288 | 9728 | 6144 |
| vocab | `token_embd.weight` ne1 | 151669 | 151669 | 151669 |
| rope_theta | `qwen3.rope.freq_base` | 1e6 | **5e6** | 1e6 |
| yarn_factor / orig_ctx | `qwen3.rope.scaling.*` | 4.0 / 16384 | 4.0 / 8192 | 4.0 / 8192 |
| tied embeddings | no `output.weight` | no | yes | yes |
| tensor count | — | 399 | 398 | 310 |

`vocab` deliberately comes from the embedding matrix's ROW COUNT, not from
`len(tokenizer.ggml.tokens)` — they can disagree, and the logits GEMV + argmax
are sized by the former.

**The 4B row is why this mattered.** Its `rope_theta` is 5e6 and its YaRN
original context is 8192 — the old hardcoded converter would have written 1e6
and 16384, i.e. wrong RoPE frequencies on every layer. The *shipped*
`bonsai4b.nnb` has the correct values (it was produced from an uncommitted local
edit), so nothing in the released bundle is wrong — but the committed scripts
could not have reproduced it, and would have silently produced a broken 4B for
anyone converting from upstream GGUF. Both shipped headers (`bonsai8b.nnb`,
`bonsai4b.nnb`) were re-derived from their upstream GGUFs and match field for
field.

### Bonsai-1.7B — MEASURED, Razr 2020 / Adreno 620

| metric | value |
|---|---|
| decode, sustained | **7.25 tok/s** (0.138 s/tok, 64-token chat run) |
| decode, 31-token run | 6.96 tok/s |
| weights | 242 MB (310 tensors, tied embeddings — no `output.weight`) |
| correctness | **TOKEN-EXACT vs the llama.cpp oracle, 3/3 goldens** |

Decode scales almost exactly inversely with parameter count across the family
(8B 1.96 → 4B 3.08 → 1.7B 7.25 tok/s), which is what an ALU-issue-bound
1-bit-unpack decode predicts: work per token ∝ weights streamed, and nothing in
the pipeline is fixed-cost-dominated at these sizes.

**Gated token-exact**, same bar as the 8B: `reference/golden_1.7b_{1,2,3}.json`
captured from a `llama-server` `/completion` oracle (greedy, `top_k=1`,
`temperature=0`, raw prompt — NOT `llama-cli`, whose conversation mode
re-applies the chat template on top of an already-wrapped prompt and silently
produces a different sequence). Device runs reproduce all three id-for-id:

```
BONSAI_GOLDEN=1.7b_ BONSAI_NNB=bonsai1.7b.nnb ./scripts/check_goldens.sh
PASS golden_1.7b_1   PASS golden_1.7b_2   PASS golden_1.7b_3
```

`scripts/check_goldens.sh` is new and works for any size — it runs the prompts
on the device and diffs the emitted ids, treating "device stopped one token
short, on the oracle's EOS" as a pass (the decode loop breaks on EOS without
forwarding it).

### Bonsai-27B — RULED OUT (architecture, not memory)

Read straight from `Bonsai-27B-Q1_0.gguf`'s own header (851 tensors, 37 KV):

```
general.architecture   = qwen35        <-- not qwen3
qwen35.block_count     = 64
qwen35.embedding_length= 5120
qwen35.feed_forward_length = 17408
qwen35.attention.head_count / _kv = 24 / 4
qwen35.attention.key_length / value_length = 256 / 256
qwen35.full_attention_interval = 4
qwen35.ssm.conv_kernel = 4   state_size = 128   group_count = 16
qwen35.ssm.time_step_rank = 48   inner_size = 6144
qwen35.rope.dimension_count = 64   dimension_sections = [11,11,10,0]
```

Tensors-per-layer alternates `14,14,14,11` across all 64 blocks. The 48 blocks
with 14 tensors are **Gated-DeltaNet linear-attention** layers:

```
blk.0.ssm_conv1d.weight   [4, 10240]     causal depthwise conv, kernel 4
blk.0.ssm_a               [48]           per-head decay
blk.0.ssm_dt.bias         [48]
blk.0.ssm_alpha.weight    [5120, 48]     gate projections
blk.0.ssm_beta.weight     [5120, 48]
blk.0.ssm_norm.weight     [128]
blk.0.ssm_out.weight      [6144, 5120]
blk.0.attn_qkv.weight     [5120, 10240]  fused q/k/v for the delta rule
blk.0.attn_gate.weight    [5120, 6144]
```

The 16 blocks with 11 tensors (`L % 4 == 3`) are quadratic attention, but still
not this port's attention: head_dim 256, 24 q / 4 kv heads, `attn_gate`, and
**partial mRoPE** (only 64 of 256 dims rotated, 3-section multimodal layout —
the repo also ships a vision `mmproj`).

What a 27B port would actually require, none of which exists here:

1. A **recurrent state** operator (delta-rule scan) replacing the KV cache on
   48/64 layers — per-layer state `[16 groups × 128 state × …]`, plus a rolling
   4-tap conv1d window. Different memory model, different kernel shape.
2. **Gated attention** kernels: output gating, head_dim 256, 24/4 GQA.
3. **Partial + sectioned mRoPE** (current `rope.cl` rotates the full head).
4. A **heterogeneous layer loop** — `model.cpp` runs one uniform block ×N.
5. A **second tokenizer**: 248320 entries vs 151669 (not the shared vocab).

Memory is a hard blocker on top of that, independent of the kernels: OpenCL
reports **3744 MB global** on this Adreno 620, and the port's device residency
is ≈ the full weight set. 27B Q1_0 is 3.8 GB of weights *before* activations,
recurrent state, or the 248k-row logits buffer — it does not fit, and the host
side would need ~2× that (device copy + mmap) against 7.6 GB of system RAM.

**Verdict:** 27B is a new port sharing only the Q1_0 unpack kernels, on a device
that cannot hold it. 1.7B was a converter fix. There is nothing in between.
