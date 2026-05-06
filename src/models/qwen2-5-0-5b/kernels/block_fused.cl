// Reference: model_info/transformers_src/modeling_qwen2.py:64-176 Qwen2Attention.forward; 14-33 Qwen2MLP.forward
// Auto-generated fused decode (M=1) kernels — Rule FUSE-DECODE-01.
//
// Validated on SmolLM2-135M / Adreno 620: 21.3× over baseline (0.396 →
// 8.43 tok/s decode), fp16, token-IDs ID-for-ID identical. Pure OpenCL
// 1.2 + cl_khr_fp16; portable to any Adreno 5xx+, Mali, Apple, PowerVR,
// Intel iGPU.
//
// What's emitted here: 3 UNIVERSAL kernels that work on every
// transformer regardless of architecture. The agent reads these as
// templates and writes architecture-specific fused kernels (qkv_gemv,
// gate_up_silu, gate_up_gelu, rope_kvwrite, c_attn_split_gemv, etc.)
// applying the same recipe — see the porting prompt's "GEMV Recipe"
// and "Fusion Opportunities" sections.
//
// Threading model (every kernel below): 1 workgroup per output column,
// 64 threads cooperate over the reduction dim, tree reduction in
// __local mem. Vec4 fp16 loads (vload_half4) for 4× memory throughput.
//
// WG_SIZE is chosen so it divides every dim uniformly — this port:
//   H        = 0
//   INTER    = 0
//   Q_DIM    = 0    (q_heads × head_dim)
//   KV_DIM   = 0   (kv_heads × head_dim)
//   VOCAB    = 0
//   WG_SIZE  = 64
// Dtype-template preamble — DO NOT EDIT. Driven by host-side -DNNOPT_USE_FP16.
#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)    vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)    ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
#endif


#define WG_SIZE 64

// Common activations (declared here so any agent-added kernel in this
// file can reuse them without re-declaring).
inline float silu_f(float x) { return x / (1.0f + exp(-x)); }
inline float gelu_f(float x) {
  // tanh approximation — matches transformers GeLU.NEW_GELU.
  const float c0 = 0.7978845608f;          // sqrt(2/pi)
  const float c1 = 0.044715f;
  return 0.5f * x * (1.0f + tanh(c0 * (x + c1 * x * x * x)));
}

// ──────────────────────────────────────────────────────────────────────
// fused_oproj_residual_m1 (UNIVERSAL — every transformer port)
//
// Replaces 2 launches at decode:
//   attn_proj = pytorch_linear(attn_out, Wo)   // CLBlast Gemm
//   residual += attn_proj                      // element_add
// with one custom GEMV that writes directly into the residual buffer.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fused_oproj_residual_m1(
    __global const storage_t* attn_out,  // [Q_DIM]
    __global const storage_t* w_o,       // [H, Q_DIM]   (PyTorch nn.Linear: [out, in])
    __global storage_t* residual,        // [H] INOUT
    const int H,
    const int Q_DIM) {
  __local float ls[WG_SIZE];
  const int c   = get_group_id(0);
  const int tid = get_local_id(0);
  if (c >= H) return;

  const int base = c * Q_DIM;
  const int dp = Q_DIM / WG_SIZE;
  const int d_start = tid * dp;

  float sum = 0.0f;
#ifdef USE_FP16
  // Vec4 fp16 + scalar tail. dp may not be a multiple of 4 for some dims
  // (e.g. Q_DIM=896 → dp=14 → 3 vec4 + 2 scalar at WG_SIZE=64).
  const int dp4 = dp >> 2;
  for (int j = 0; j < dp4; ++j) {
    const int off = d_start + j * 4;
    float4 xv = vload_half4(0, attn_out + off);
    float4 wv = vload_half4(0, w_o + base + off);
    sum += dot(xv, wv);
  }
  for (int j = dp4 * 4; j < dp; ++j) {
    sum += LOAD(attn_out, d_start + j) * LOAD(w_o, base + d_start + j);
  }
#else
  for (int j = 0; j < dp; ++j) {
    sum += LOAD(attn_out, d_start + j) * LOAD(w_o, base + d_start + j);
  }
#endif

  ls[tid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) ls[tid] += ls[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    const float r = LOAD(residual, c);
    STORE(residual, c, r + ls[0]);
  }
}

// ──────────────────────────────────────────────────────────────────────
// fused_down_residual_m1 (UNIVERSAL — every transformer/MLP port)
//
// Replaces 2 launches at decode:
//   mlp_out = pytorch_linear(mlp_in, Wdown)   // CLBlast Gemm
//   residual += mlp_out                       // element_add
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fused_down_residual_m1(
    __global const storage_t* mlp_in,    // [N]   (= INTER for SwiGLU; H for plain)
    __global const storage_t* w_down,    // [K, N]
    __global storage_t* residual,        // [K] INOUT
    const int K,
    const int N) {
  __local float ls[WG_SIZE];
  const int c   = get_group_id(0);
  const int tid = get_local_id(0);
  if (c >= K) return;

  const int base = c * N;
  const int np = N / WG_SIZE;
  const int n_start = tid * np;

  float sum = 0.0f;
#ifdef USE_FP16
  const int np4 = np >> 2;
  for (int j = 0; j < np4; ++j) {
    const int off = n_start + j * 4;
    float4 xv = vload_half4(0, mlp_in + off);
    float4 wv = vload_half4(0, w_down + base + off);
    sum += dot(xv, wv);
  }
  for (int j = np4 * 4; j < np; ++j) {
    sum += LOAD(mlp_in, n_start + j) * LOAD(w_down, base + n_start + j);
  }
#else
  for (int j = 0; j < np; ++j) {
    sum += LOAD(mlp_in, n_start + j) * LOAD(w_down, base + n_start + j);
  }
#endif

  ls[tid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) ls[tid] += ls[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    const float r = LOAD(residual, c);
    STORE(residual, c, r + ls[0]);
  }
}

// ──────────────────────────────────────────────────────────────────────
// fused_lm_head_gemv_m1 (UNIVERSAL — every port has an lm_head)
//
// Replaces CLBlast HGemm at the end of forward() when seq_len == 1:
//   logits = x @ W^T    (x: [H], W: [VOCAB, H])
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fused_lm_head_gemv_m1(
    __global const storage_t* x,        // [H]
    __global const storage_t* W,        // [VOCAB, H]
    __global storage_t* logits,         // [VOCAB]
    const int H,
    const int VOCAB) {
  __local float ls[WG_SIZE];
  const int c   = get_group_id(0);
  const int tid = get_local_id(0);
  if (c >= VOCAB) return;

  const int base = c * H;
  const int kp = H / WG_SIZE;
  const int k_start = tid * kp;

  float acc = 0.0f;
#ifdef USE_FP16
  const int kp4 = kp >> 2;
  for (int j = 0; j < kp4; ++j) {
    const int off = k_start + j * 4;
    float4 xv = vload_half4(0, x + off);
    float4 wv = vload_half4(0, W + base + off);
    acc += dot(xv, wv);
  }
  for (int j = kp4 * 4; j < kp; ++j) {
    acc += LOAD(x, k_start + j) * LOAD(W, base + k_start + j);
  }
#else
  for (int j = 0; j < kp; ++j) {
    acc += LOAD(x, k_start + j) * LOAD(W, base + k_start + j);
  }
#endif

  ls[tid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) ls[tid] += ls[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) STORE(logits, c, ls[0]);
}

// ──────────────────────────────────────────────────────────────────────
// fused_qkv_gemv_m1 (UNIVERSAL for separate Q/K/V architectures —
// Llama / Qwen / SmolLM / OpenELM / Phi / Mistral / Gemma / etc.)
//
// Replaces 3 CLBlast HGemms at decode (M=1):
//   q = pytorch_linear(x, Wq)   // [Q_DIM]
//   k = pytorch_linear(x, Wk)   // [KV_DIM]
//   v = pytorch_linear(x, Wv)   // [KV_DIM]
// with one cooperative kernel: each WG produces ONE output element,
// reading x once. Total groups = Q_DIM + 2*KV_DIM. Validated on
// SmolLM2-135M (1.94× → 2.43× when combined with vec4 + coop pattern).
//
// For GPT-2 / GPT-Neo / OPT (fused c_attn Conv1D output) the agent
// writes fused_c_attn_split_gemv_m1 instead — the layouts differ and
// this kernel does not apply directly there. See model_shapes.json
// "qkv_pattern" to know which case the current model is.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fused_qkv_gemv_m1(
    __global const storage_t* x,        // [H]
    __global const storage_t* w_q,      // [Q_DIM, H]
    __global const storage_t* w_k,      // [KV_DIM, H]
    __global const storage_t* w_v,      // [KV_DIM, H]
    __global storage_t* q_out,          // [Q_DIM]
    __global storage_t* k_out,          // [KV_DIM]
    __global storage_t* v_out,          // [KV_DIM]
    const int H,
    const int Q_DIM,
    const int KV_DIM) {
  __local float ls[WG_SIZE];

  const int gid = get_group_id(0);
  const int tid = get_local_id(0);

  // Decide which projection this workgroup is computing.
  __global const storage_t* W;
  __global storage_t* OUT;
  int row;
  if (gid < Q_DIM) {
    W = w_q;  OUT = q_out;  row = gid;
  } else if (gid < Q_DIM + KV_DIM) {
    W = w_k;  OUT = k_out;  row = gid - Q_DIM;
  } else {
    W = w_v;  OUT = v_out;  row = gid - Q_DIM - KV_DIM;
  }
  const int base = row * H;
  const int kp = H / WG_SIZE;
  const int k_start = tid * kp;

  float acc = 0.0f;
#ifdef USE_FP16
  const int kp4 = kp >> 2;
  for (int j = 0; j < kp4; ++j) {
    const int off = k_start + j * 4;
    float4 xv = vload_half4(0, x + off);
    float4 wv = vload_half4(0, W + base + off);
    acc += dot(xv, wv);
  }
  for (int j = kp4 * 4; j < kp; ++j) {
    acc += LOAD(x, k_start + j) * LOAD(W, base + k_start + j);
  }
#else
  for (int j = 0; j < kp; ++j) {
    acc += LOAD(x, k_start + j) * LOAD(W, base + k_start + j);
  }
#endif

  ls[tid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) ls[tid] += ls[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) STORE(OUT, row, ls[0]);
}

// ──────────────────────────────────────────────────────────────────────
// fused_rope_kvwrite_m1 (UNIVERSAL for HF-default RoPE — Llama / Qwen /
// SmolLM / Mistral / Gemma / Phi-3 / etc.)
//
// Replaces 3 launches at decode (seq_q == 1):
//   1) rope_apply_qk(q, k, cos_table, sin_table, start_pos)  // rotates q AND k
//   2) clEnqueueCopyBuffer(k -> k_cache @ start_pos)          // KV cache write
//   3) clEnqueueCopyBuffer(v -> v_cache @ start_pos)
// with one kernel that:
//   - rotates q in place
//   - rotates k AND writes the rotated k directly into k_cache at the
//     start_pos slot (skipping the intermediate k buffer's rotated copy)
//   - copies v directly into v_cache at the start_pos slot
//
// HF-default RoPE convention (Llama family):
//   x_rotated[i]            = x[i]            * cos[i] - x[i + half] * sin[i]
//   x_rotated[i + half]     = x[i + half]     * cos[i] + x[i]        * sin[i]
// where half = head_dim / 2 and cos/sin are duplicated across both halves
// (HF uses cat([theta, theta], dim=-1) so cos[i] == cos[i + half]).
//
// For RoPE variants that compute rotation pair-wise (rotary[2k, 2k+1])
// instead of by-half (rotary[k, k+half]) — e.g. some GPT-NeoX-style
// repos — write a separate fused_rope_kvwrite_neox_m1 from this template.
// model_shapes.json :: rope_variant tells you which case applies.
//
// Threading: 1D global, work-items concatenated:
//   [0, q_pairs)                                    -> Q rotation pair index
//   [q_pairs, q_pairs + k_pairs)                    -> K rotation+cache-write
//   [q_pairs + k_pairs, q_pairs + k_pairs + v_dim)  -> V copy element
__kernel void fused_rope_kvwrite_m1(
    __global storage_t* q,             // [Q_DIM]               INOUT — rotated in place
    __global const storage_t* k_in,    // [KV_DIM]              read-only
    __global const storage_t* v_in,    // [KV_DIM]              read-only
    __global storage_t* k_cache,       // [MAX_POS, KV_DIM]     written at start_pos slot
    __global storage_t* v_cache,       // [MAX_POS, KV_DIM]     written at start_pos slot
    __global const storage_t* cos_tab, // [seq_max, head_dim]
    __global const storage_t* sin_tab,
    const int q_heads,
    const int kv_heads,
    const int head_dim,
    const int kv_dim,
    const int start_pos) {
  const int half_dim = head_dim / 2;
  const int q_pairs = q_heads * half_dim;
  const int k_pairs = kv_heads * half_dim;
  const int v_elems = kv_dim;
  const int total = q_pairs + k_pairs + v_elems;

  const int gid = (int)get_global_id(0);
  if (gid >= total) return;

  const int cs_base = start_pos * head_dim;
  const int cache_base = start_pos * kv_dim;

  if (gid < q_pairs) {
    const int h  = gid / half_dim;
    const int hd = gid - h * half_dim;
    const int base = h * head_dim;
    const float x1 = LOAD(q, base + hd);
    const float x2 = LOAD(q, base + hd + half_dim);
    const float c  = LOAD(cos_tab, cs_base + hd);
    const float s  = LOAD(sin_tab, cs_base + hd);
    STORE(q, base + hd,            x1 * c - x2 * s);
    STORE(q, base + hd + half_dim, x2 * c + x1 * s);
    return;
  }

  if (gid < q_pairs + k_pairs) {
    const int kid = gid - q_pairs;
    const int h   = kid / half_dim;
    const int hd  = kid - h * half_dim;
    const int base = h * head_dim;
    const float x1 = LOAD(k_in, base + hd);
    const float x2 = LOAD(k_in, base + hd + half_dim);
    const float c  = LOAD(cos_tab, cs_base + hd);
    const float s  = LOAD(sin_tab, cs_base + hd);
    STORE(k_cache, cache_base + base + hd,            x1 * c - x2 * s);
    STORE(k_cache, cache_base + base + hd + half_dim, x2 * c + x1 * s);
    return;
  }

  // V copy (no rotation).
  {
    const int vid = gid - q_pairs - k_pairs;
    const float v = LOAD(v_in, vid);
    STORE(v_cache, cache_base + vid, v);
  }
}

// ──────────────────────────────────────────────────────────────────────
// AGENT-ADDED KERNELS GO BELOW. Apply the GEMV Recipe documented in the
// porting prompt's Section D ("Fused decode (M=1) kernels"):
//   1) vec4 fp16 loads (vload_half4 returns float4 directly)
//   2) workgroup-cooperative reduction (1 WG/output, WG_SIZE threads,
//      __local-mem tree reduction)
//   3) pointwise fusion into adjacent GEMV (residual-add, activation,
//      RoPE rotation — all merge into thread 0's write or the K-loop)
// Reuse WG_SIZE defined above.
//
// SCAFFOLD-EMITTED (above, ready to call from host code):
//   - fused_oproj_residual_m1      (every transformer)
//   - fused_down_residual_m1       (every MLP with down-proj)
//   - fused_lm_head_gemv_m1        (every model has lm_head)
//   - fused_qkv_gemv_m1            (separate Q/K/V — Llama family)
//   - fused_rope_kvwrite_m1        (HF-default RoPE — Llama family)
//
// AGENT-WRITTEN (architecture-specific):
//   - fused_gate_up_silu_m1        (SwiGLU MLP — silu(gate*x) * up*x)
//   - fused_gate_up_gelu_m1        (GeGLU MLP — gelu(gate*x) * up*x)
//   - fused_fc1_gelu_m1            (plain GeLU FFN — GPT-2 MLP, no gate)
//   - fused_kvwrite_m1             (KV cache write only — non-RoPE pos embeds)
//   - fused_rope_kvwrite_neox_m1   (GPT-NeoX-style pair-wise rotation,
//                                    differs from HF default by-half rotation)
//   - fused_c_attn_split_gemv_m1   (GPT-2 fused c_attn — see note below)
//
// NOTE for GPT-2 family: c_attn weight is stored as [H, 3*H] row-major
// (HF Conv1D), STRIDED for per-output GEMV reads. Two correct paths:
//   (a) Transpose c_attn at weight-load time to [3*H, H] (nn.Linear
//       layout) and then call fused_qkv_gemv_m1 directly. Simpler.
//   (b) Write fused_c_attn_split_gemv_m1 with stride-3H W reads. Slower
//       per byte but no weight munging. Apply vec4 over the H (input)
//       axis where possible.
// Path (a) is the recommended one for GPT-2 ports.
//
// See the porting prompt for full templates and per-architecture advice.
