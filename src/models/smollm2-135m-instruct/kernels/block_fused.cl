// Reference: model_info/transformers_src/modeling_llama.py:LlamaAttention.forward,LlamaMLP.forward,LlamaForCausalLM.forward
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
  // Wave-stride: all 64 threads read Q_DIM[0..63], Q_DIM[64..127], ...
  // → 64 consecutive fp16 per step = 1 cache-line per step (coalesced).
  float sum = 0.0f;
  for (int j = tid; j < Q_DIM; j += WG_SIZE) {
    sum += LOAD(attn_out, j) * LOAD(w_o, base + j);
  }

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
  // Wave-stride vec4: N=1536 = 24×WG_SIZE = 6×(4×WG_SIZE) → no tail.
  // 64 threads × 4 fp16 per step = 256 consecutive fp16 per step (coalesced).
  float sum = 0.0f;
#ifdef USE_FP16
  for (int j = tid * 4; j < N; j += WG_SIZE * 4) {
    float4 xv = vload_half4(0, mlp_in + j);
    float4 wv = vload_half4(0, w_down + base + j);
    sum += dot(xv, wv);
  }
#else
  for (int j = tid; j < N; j += WG_SIZE) {
    sum += LOAD(mlp_in, j) * LOAD(w_down, base + j);
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
  // Wave-stride: H=576 = 9×WG_SIZE → 9 exact iters, no tail, coalesced.
  float acc = 0.0f;
  for (int j = tid; j < H; j += WG_SIZE) {
    acc += LOAD(x, j) * LOAD(W, base + j);
  }

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
  // Wave-stride: H=576=9×WG_SIZE → 9 exact iters, coalesced.
  float acc = 0.0f;
  for (int j = tid; j < H; j += WG_SIZE) {
    acc += LOAD(x, j) * LOAD(W, base + j);
  }

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
// fused_gate_up_silu_m1 (SwiGLU MLP — SmolLM2/Llama family)
//
// Replaces 3 launches at decode (M=1):
//   gate = pytorch_linear(x, Wgate)  // CLBlast Gemm
//   up   = pytorch_linear(x, Wup)    // CLBlast Gemm
//   act  = silu(gate) * up           // silu_mul kernel
// Each WG computes one output: silu(Wgate[c,:].x) * (Wup[c,:].x)
// Two accumulators share the same x loads (gate + up read x simultaneously).
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fused_gate_up_silu_m1(
    __global const storage_t* x,        // [H]
    __global const storage_t* w_gate,   // [INTER, H]
    __global const storage_t* w_up,     // [INTER, H]
    __global storage_t* out,            // [INTER]   silu(gate)*up
    const int H,
    const int INTER) {
  __local float ls_gate[WG_SIZE];
  __local float ls_up[WG_SIZE];
  const int c   = get_group_id(0);
  const int tid = get_local_id(0);
  if (c >= INTER) return;

  const int base = c * H;
  // Wave-stride: H=576=9×WG_SIZE → 9 exact iters, coalesced.
  // Both gate and up accumulators read x together (shared load).
  float gate_acc = 0.0f, up_acc = 0.0f;
  for (int j = tid; j < H; j += WG_SIZE) {
    float xi = LOAD(x, j);
    gate_acc += xi * LOAD(w_gate, base + j);
    up_acc   += xi * LOAD(w_up,   base + j);
  }

  ls_gate[tid] = gate_acc;
  ls_up[tid]   = up_acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      ls_gate[tid] += ls_gate[tid + s];
      ls_up[tid]   += ls_up[tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    float g = ls_gate[0];
    STORE(out, c, (g / (1.0f + exp(-g))) * ls_up[0]);
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
// ──────────────────────────────────────────────────────────────────────
// SmolLM2-specific image2d_t-backed GEMV (Adreno texture cache path).
//
// Adreno 6xx GPUs have a dedicated texture-fetch engine with higher
// effective bandwidth than buffer reads (1.3-1.5× measured on Qwen
// port). Wrapping fp16 weights as CL_RGBA / CL_HALF_FLOAT images
// (4 fp16 per pixel) routes reads through that engine. K=576 is
// SmolLM2's hidden_size and is shared by q/k/v/o/gate/up/lm_head.
//
// no4 = 4 outputs per WG. 4 fp32 accumulators per thread share a
// single x-vector load — register-level parallelism that quarters the
// WG count. Predicate: N % 4 == 0 (true for all SmolLM2 GEMV sites).
//
// K=576 = 144 vec4-pixels = 2*64 + 16 → 2 full waves + 16-thread tail.
// Buffer-fallback variants (gemv_m1_k576_no4) preserve the same
// dispatch pattern when the W can't be wrapped as an image (oversized
// row count beyond CL_DEVICE_IMAGE2D_MAX_HEIGHT, or driver refusal).
#ifdef USE_FP16

__constant sampler_t kImgSampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

// K=576, image2d-backed W, 4 outputs per WG. N % 4 must be 0.
// Dispatch: gws = (N / 4) * WG_SIZE, lws = WG_SIZE.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k576_no4_img(
    __global const half* x,
    __read_only image2d_t W_img,
    __global half* out,
    const int N) {
  const int n_base = (int)get_group_id(0) * 4;
  const int tid   = (int)get_local_id(0);
  if (n_base >= N) return;

  __local float partial[4][WG_SIZE];
  float acc0=0.0f, acc1=0.0f, acc2=0.0f, acc3=0.0f;

  // 2 full waves of vec4 (covers 2*256 = 512 fp16 of K=576).
  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int x_off = j * (WG_SIZE * 4) + tid * 4;
    const int pix   = j *  WG_SIZE      + tid;       // pixel column (0..143)
    float4 xv = vload_half4(0, x + x_off);
    float4 w0 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }
  // Tail: 576-512 = 64 fp16 = 16 vec4 → first 16 threads.
  if (tid < 16) {
    const int x_off = 2 * (WG_SIZE * 4) + tid * 4;
    const int pix   = 2 *  WG_SIZE      + tid;       // pix in [128..143]
    float4 xv = vload_half4(0, x + x_off);
    float4 w0 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }

  partial[0][tid] = acc0; partial[1][tid] = acc1;
  partial[2][tid] = acc2; partial[3][tid] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      partial[0][tid] += partial[0][tid + s];
      partial[1][tid] += partial[1][tid + s];
      partial[2][tid] += partial[2][tid + s];
      partial[3][tid] += partial[3][tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    vstore_half(partial[0][0], 0, out + n_base + 0);
    vstore_half(partial[1][0], 0, out + n_base + 1);
    vstore_half(partial[2][0], 0, out + n_base + 2);
    vstore_half(partial[3][0], 0, out + n_base + 3);
  }
}

// K=576, vec4 fused gate+up SwiGLU with single output per WG.
// Why not no4 + image: dual-accumulator (gate + up) with no4 = 8 fp32 accs
// per thread → register spill on Adreno (Qwen Step 7 measured 1.78× regression).
// Single-output keeps 2 fp32 accs/thread (safe), vec4 turns scalar 9-iter loop
// into 2 vec4 iters + 16-thread tail. This kernel was the #1 hotspot
// (35% of GPU time) at Step 3 — lots of headroom in the inner loop.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fused_gate_up_silu_m1_v4(
    __global const half* x,        // [H]
    __global const half* w_gate,   // [INTER, H]
    __global const half* w_up,     // [INTER, H]
    __global half* out,            // [INTER]
    const int H,
    const int INTER) {
  const int c   = (int)get_group_id(0);
  const int tid = (int)get_local_id(0);
  if (c >= INTER) return;

  __local float ls_gate[WG_SIZE];
  __local float ls_up  [WG_SIZE];

  const int base = c * H;
  float gate_acc = 0.0f, up_acc = 0.0f;

  // 2 full waves of vec4 (covers 2*256 = 512 fp16 of K=576).
  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int off = j * (WG_SIZE * 4) + tid * 4;
    float4 xv  = vload_half4(0, x + off);
    float4 wgv = vload_half4(0, w_gate + base + off);
    float4 wuv = vload_half4(0, w_up   + base + off);
    gate_acc += dot(xv, wgv);
    up_acc   += dot(xv, wuv);
  }
  // Tail: 64 fp16 = 16 vec4 → first 16 threads.
  if (tid < 16) {
    const int off = 2 * (WG_SIZE * 4) + tid * 4;
    float4 xv  = vload_half4(0, x + off);
    float4 wgv = vload_half4(0, w_gate + base + off);
    float4 wuv = vload_half4(0, w_up   + base + off);
    gate_acc += dot(xv, wgv);
    up_acc   += dot(xv, wuv);
  }

  ls_gate[tid] = gate_acc;
  ls_up  [tid] = up_acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      ls_gate[tid] += ls_gate[tid + s];
      ls_up  [tid] += ls_up  [tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    float g = ls_gate[0];
    vstore_half((g / (1.0f + exp(-g))) * ls_up[0], c, out);
  }
}

// K=576, image2d-backed Wgate AND Wup, single output per WG, vec4 inner loop.
// no4 risks register spill (8 fp32 accs/thread — Qwen Step 7 regression).
// Single-output keeps 2 accs/thread; texture cache provides the BW win.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fused_gate_up_silu_m1_v4_img(
    __global const half* x,                  // [H]
    __read_only image2d_t Wg_img,            // [INTER, K_PIX]
    __read_only image2d_t Wu_img,            // [INTER, K_PIX]
    __global half* out,                      // [INTER]
    const int INTER) {
  const int c   = (int)get_group_id(0);
  const int tid = (int)get_local_id(0);
  if (c >= INTER) return;

  __local float ls_gate[WG_SIZE];
  __local float ls_up  [WG_SIZE];

  float gate_acc = 0.0f, up_acc = 0.0f;

  // K=576 = 2 vec4 waves + 16-thread tail (same layout as gemv_m1_k576_no4_img).
  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int x_off = j * (WG_SIZE * 4) + tid * 4;
    const int pix   = j *  WG_SIZE      + tid;
    float4 xv  = vload_half4(0, x + x_off);
    float4 wgv = convert_float4(read_imageh(Wg_img, kImgSampler, (int2)(pix, c)));
    float4 wuv = convert_float4(read_imageh(Wu_img, kImgSampler, (int2)(pix, c)));
    gate_acc += dot(xv, wgv);
    up_acc   += dot(xv, wuv);
  }
  if (tid < 16) {
    const int x_off = 2 * (WG_SIZE * 4) + tid * 4;
    const int pix   = 2 *  WG_SIZE      + tid;
    float4 xv  = vload_half4(0, x + x_off);
    float4 wgv = convert_float4(read_imageh(Wg_img, kImgSampler, (int2)(pix, c)));
    float4 wuv = convert_float4(read_imageh(Wu_img, kImgSampler, (int2)(pix, c)));
    gate_acc += dot(xv, wgv);
    up_acc   += dot(xv, wuv);
  }

  ls_gate[tid] = gate_acc;
  ls_up  [tid] = up_acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      ls_gate[tid] += ls_gate[tid + s];
      ls_up  [tid] += ls_up  [tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    float g = ls_gate[0];
    vstore_half((g / (1.0f + exp(-g))) * ls_up[0], c, out);
  }
}

// K=576, image2d-backed Wq/Wk/Wv, 4 outputs per WG.
// Replaces fused_qkv_gemv_m1 buffer kernel (Q_DIM=576 + 2*KV_DIM=192 = 960 outputs).
// Dispatch: gws = (Q_DIM + 2*KV_DIM) / 4 * WG_SIZE, lws = WG_SIZE.
// All boundary conditions assume Q_DIM % 4 == 0 AND KV_DIM % 4 == 0.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fused_qkv_gemv_m1_no4_img(
    __global const half* x,
    __read_only image2d_t Wq_img,
    __read_only image2d_t Wk_img,
    __read_only image2d_t Wv_img,
    __global half* q_out,
    __global half* k_out,
    __global half* v_out,
    const int Q_DIM_4,    // = Q_DIM / 4 (workgroup count for Q segment)
    const int KV_DIM_4) { // = KV_DIM / 4 (workgroup count for K and V each)
  const int gid_block = (int)get_group_id(0);
  const int tid       = (int)get_local_id(0);

  // Pick projection based on which segment this WG falls in. OpenCL forbids
  // image2d_t variables, so we replicate the inner loop per segment branch.
  const int seg = (gid_block < Q_DIM_4) ? 0
                : (gid_block < Q_DIM_4 + KV_DIM_4) ? 1
                : 2;
  const int n_base = (seg == 0) ? gid_block * 4
                  : (seg == 1) ? (gid_block - Q_DIM_4) * 4
                              : (gid_block - Q_DIM_4 - KV_DIM_4) * 4;
  __global half* OUT = (seg == 0) ? q_out
                     : (seg == 1) ? k_out
                                  : v_out;

  __local float partial[4][WG_SIZE];
  float acc0=0.0f, acc1=0.0f, acc2=0.0f, acc3=0.0f;

#define QKV_NO4_INNER(W_IMG)                                                   \
  do {                                                                         \
    _Pragma("unroll")                                                          \
    for (int j = 0; j < 2; ++j) {                                              \
      const int x_off = j * (WG_SIZE * 4) + tid * 4;                           \
      const int pix   = j *  WG_SIZE      + tid;                               \
      float4 xv = vload_half4(0, x + x_off);                                   \
      float4 w0 = convert_float4(read_imageh((W_IMG), kImgSampler, (int2)(pix, n_base + 0))); \
      float4 w1 = convert_float4(read_imageh((W_IMG), kImgSampler, (int2)(pix, n_base + 1))); \
      float4 w2 = convert_float4(read_imageh((W_IMG), kImgSampler, (int2)(pix, n_base + 2))); \
      float4 w3 = convert_float4(read_imageh((W_IMG), kImgSampler, (int2)(pix, n_base + 3))); \
      acc0 += dot(xv, w0); acc1 += dot(xv, w1);                                \
      acc2 += dot(xv, w2); acc3 += dot(xv, w3);                                \
    }                                                                          \
    if (tid < 16) {                                                            \
      const int x_off = 2 * (WG_SIZE * 4) + tid * 4;                           \
      const int pix   = 2 *  WG_SIZE      + tid;                               \
      float4 xv = vload_half4(0, x + x_off);                                   \
      float4 w0 = convert_float4(read_imageh((W_IMG), kImgSampler, (int2)(pix, n_base + 0))); \
      float4 w1 = convert_float4(read_imageh((W_IMG), kImgSampler, (int2)(pix, n_base + 1))); \
      float4 w2 = convert_float4(read_imageh((W_IMG), kImgSampler, (int2)(pix, n_base + 2))); \
      float4 w3 = convert_float4(read_imageh((W_IMG), kImgSampler, (int2)(pix, n_base + 3))); \
      acc0 += dot(xv, w0); acc1 += dot(xv, w1);                                \
      acc2 += dot(xv, w2); acc3 += dot(xv, w3);                                \
    }                                                                          \
  } while (0)

  if      (seg == 0) { QKV_NO4_INNER(Wq_img); }
  else if (seg == 1) { QKV_NO4_INNER(Wk_img); }
  else               { QKV_NO4_INNER(Wv_img); }

#undef QKV_NO4_INNER

  partial[0][tid] = acc0; partial[1][tid] = acc1;
  partial[2][tid] = acc2; partial[3][tid] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      partial[0][tid] += partial[0][tid + s];
      partial[1][tid] += partial[1][tid + s];
      partial[2][tid] += partial[2][tid + s];
      partial[3][tid] += partial[3][tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    vstore_half(partial[0][0], 0, OUT + n_base + 0);
    vstore_half(partial[1][0], 0, OUT + n_base + 1);
    vstore_half(partial[2][0], 0, OUT + n_base + 2);
    vstore_half(partial[3][0], 0, OUT + n_base + 3);
  }
}

// K=576, image2d-backed W, 4 outputs per WG, fused residual add.
// Replaces fused_oproj_residual_m1: out_proj.x → += residual.
// N must be a multiple of 4. residual is INOUT.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fused_oproj_residual_m1_no4_img(
    __global const half* attn_out,        // [Q_DIM]
    __read_only image2d_t Wo_img,         // [H, Q_DIM] as image
    __global half* residual,              // [H] INOUT
    const int H) {
  const int n_base = (int)get_group_id(0) * 4;
  const int tid    = (int)get_local_id(0);
  if (n_base >= H) return;

  __local float partial[4][WG_SIZE];
  float acc0=0.0f, acc1=0.0f, acc2=0.0f, acc3=0.0f;

  // 2 full waves of vec4 (covers 2*256 = 512 fp16 of K=576).
  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int x_off = j * (WG_SIZE * 4) + tid * 4;
    const int pix   = j *  WG_SIZE      + tid;
    float4 xv = vload_half4(0, attn_out + x_off);
    float4 w0 = convert_float4(read_imageh(Wo_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(Wo_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(Wo_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(Wo_img, kImgSampler, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }
  if (tid < 16) {
    const int x_off = 2 * (WG_SIZE * 4) + tid * 4;
    const int pix   = 2 *  WG_SIZE      + tid;
    float4 xv = vload_half4(0, attn_out + x_off);
    float4 w0 = convert_float4(read_imageh(Wo_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(Wo_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(Wo_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(Wo_img, kImgSampler, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }

  partial[0][tid] = acc0; partial[1][tid] = acc1;
  partial[2][tid] = acc2; partial[3][tid] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      partial[0][tid] += partial[0][tid + s];
      partial[1][tid] += partial[1][tid + s];
      partial[2][tid] += partial[2][tid + s];
      partial[3][tid] += partial[3][tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    float r0 = vload_half(n_base + 0, residual);
    float r1 = vload_half(n_base + 1, residual);
    float r2 = vload_half(n_base + 2, residual);
    float r3 = vload_half(n_base + 3, residual);
    vstore_half(r0 + partial[0][0], n_base + 0, residual);
    vstore_half(r1 + partial[1][0], n_base + 1, residual);
    vstore_half(r2 + partial[2][0], n_base + 2, residual);
    vstore_half(r3 + partial[3][0], n_base + 3, residual);
  }
}

// K=1536 (INTERMEDIATE_SIZE), image2d-backed W, 4 outputs per WG, fused residual add.
// Replaces fused_down_residual_m1. K_PIX = 1536/4 = 384 = 6*64 → 6 full waves, no tail.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fused_down_residual_m1_no4_img(
    __global const half* mlp_in,         // [INTER]
    __read_only image2d_t Wd_img,        // [H, INTER] as image
    __global half* residual,             // [H] INOUT
    const int H) {
  const int n_base = (int)get_group_id(0) * 4;
  const int tid    = (int)get_local_id(0);
  if (n_base >= H) return;

  __local float partial[4][WG_SIZE];
  float acc0=0.0f, acc1=0.0f, acc2=0.0f, acc3=0.0f;

  // 6 full waves of vec4 (covers 6*256 = 1536 fp16 = full K), no tail.
  #pragma unroll
  for (int j = 0; j < 6; ++j) {
    const int x_off = j * (WG_SIZE * 4) + tid * 4;
    const int pix   = j *  WG_SIZE      + tid;
    float4 xv = vload_half4(0, mlp_in + x_off);
    float4 w0 = convert_float4(read_imageh(Wd_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(Wd_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(Wd_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(Wd_img, kImgSampler, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }

  partial[0][tid] = acc0; partial[1][tid] = acc1;
  partial[2][tid] = acc2; partial[3][tid] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      partial[0][tid] += partial[0][tid + s];
      partial[1][tid] += partial[1][tid + s];
      partial[2][tid] += partial[2][tid + s];
      partial[3][tid] += partial[3][tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    float r0 = vload_half(n_base + 0, residual);
    float r1 = vload_half(n_base + 1, residual);
    float r2 = vload_half(n_base + 2, residual);
    float r3 = vload_half(n_base + 3, residual);
    vstore_half(r0 + partial[0][0], n_base + 0, residual);
    vstore_half(r1 + partial[1][0], n_base + 1, residual);
    vstore_half(r2 + partial[2][0], n_base + 2, residual);
    vstore_half(r3 + partial[3][0], n_base + 3, residual);
  }
}

#endif // USE_FP16

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

// ──────────────────────────────────────────────────────────────────────
// fused_decode_attn_m1 (GQA-compatible — Llama / SmolLM2 family)
//
// Fuses 3 separate decode dispatches into 1:
//   gqa_attn_scores   → QK^T dot products + scale
//   gqa_softmax       → stable softmax in-place
//   gqa_attn_out      → weighted V sum
//
// Threading: 1 WG per Q head (9 WGs for SmolLM2 QH=9).
// WG_SIZE=64 threads cooperate over seq_k positions and D output dims.
// Dynamic local memory: ls[seq_k + WG_SIZE] floats (scores + reduction).
//
// For seq_k ≤ 2048: ls ≤ (2048+64)×4 = 8.5 KB per WG — fits on Adreno.
// In our benchmark (32 generated tokens, 9-token prompt): seq_k ≤ 42.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fused_decode_attn_m1(
    __global const storage_t* q,        // [QH * D]
    __global const storage_t* k_cache,  // [MAX_SEQ, KV_DIM]  row-major
    __global const storage_t* v_cache,  // [MAX_SEQ, KV_DIM]  row-major
    __global storage_t* attn_out,       // [QH * D]  output
    __local float* ls,                  // dynamic: (seq_k + WG_SIZE) * sizeof(float)
    const int KV_DIM,                   // KVH * D
    const int D,                        // head_dim
    const int KVH,
    const int GRP,                      // QH / KVH  (group factor for GQA)
    const int seq_k,
    const float scale
) {
    const int qh  = get_group_id(0);   // Q head index [0, QH)
    const int tid = get_local_id(0);   // [0, 64)
    const int kvh = qh / GRP;

    __local float* ls_scores  = ls;            // [seq_k] — attention scores
    __local float* ls_partial = ls + seq_k;    // [WG_SIZE] — tree-reduce scratch

    const __global storage_t* q_ptr = q + qh * D;

    // Step 1: QK^T — thread tid handles tokens tid, tid+64, tid+128, ...
    for (int t = tid; t < seq_k; t += WG_SIZE) {
        const __global storage_t* k_t = k_cache + (size_t)t * KV_DIM + kvh * D;
        float score = 0.0f;
        for (int d = 0; d < D; d++) score += LOAD(q_ptr, d) * LOAD(k_t, d);
        ls_scores[t] = score * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 2: stable softmax — parallel max over ls_scores
    float mx = -1e30f;
    for (int t = tid; t < seq_k; t += WG_SIZE) mx = fmax(mx, ls_scores[t]);
    ls_partial[tid] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) ls_partial[tid] = fmax(ls_partial[tid], ls_partial[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float gmax = ls_partial[0];

    // exp(score - max) + sum
    float psum = 0.0f;
    for (int t = tid; t < seq_k; t += WG_SIZE) {
        ls_scores[t] = exp(ls_scores[t] - gmax);
        psum += ls_scores[t];
    }
    ls_partial[tid] = psum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) ls_partial[tid] += ls_partial[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv_sum = 1.0f / ls_partial[0];

    for (int t = tid; t < seq_k; t += WG_SIZE) ls_scores[t] *= inv_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 3: weighted V sum — each thread handles dims tid, tid+64, ...
    __global storage_t* out_ptr = attn_out + qh * D;
    for (int d = tid; d < D; d += WG_SIZE) {
        float acc = 0.0f;
        for (int t = 0; t < seq_k; t++) {
            acc += ls_scores[t] * LOAD(v_cache + (size_t)t * KV_DIM + kvh * D, d);
        }
        STORE(out_ptr, d, acc);
    }
}
