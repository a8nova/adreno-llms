// Reference: /Users/alazarshenkute/.nnopt/ref_venvs/env_stable/lib/python3.12/site-packages/transformers/models/idefics3/modeling_idefics3.py:189-265 Idefics3VisionAttention.forward
// Non-causal multi-head attention (vision encoder) for Idefics3.
// Implements:
//   attn_weights = (Q @ K^T) * scale (+ optional attention_mask)
//   attn_probs   = softmax(attn_weights, dim=-1)
//   attn_output  = attn_probs @ V
//
// Layouts (row-major contiguous):
//   Q: [B, H, T, D]
//   K: [B, H, T, D]
//   V: [B, H, T, D]
//   scores/probs: [B, H, T, T]
//   out: [B, H, T, D]
//
// This kernel file is dtype-templated via storage_t + LOAD/STORE.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)     vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)     ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
#endif

// scores[b,h,tq,tk] = dot(Q[b,h,tq,:], K[b,h,tk,:]) * scale
// Global work (OpenCL 1.2 max 3D): (B, H, TT) where TT = T*T, with tq=tk_div, tk=tk_mod.
__kernel void mha_scores(
    __global const storage_t* Q,
    __global const storage_t* K,
    __global storage_t* scores,
    const int B,
    const int H,
    const int T,
    const int D,
    const float scale) {
  const int b = (int)get_global_id(0);
  const int h = (int)get_global_id(1);
  const int tt = (int)get_global_id(2);
  if (b >= B || h >= H || tt >= T * T) return;
  const int tq = tt / T;
  const int tk = tt - tq * T;

  const long q_base = (((long)b * (long)H + (long)h) * (long)T + (long)tq) * (long)D;
  const long k_base = (((long)b * (long)H + (long)h) * (long)T + (long)tk) * (long)D;

  float acc = 0.0f;
#ifdef USE_FP16
  const int D4 = D >> 2;
  for (int d4 = 0; d4 < D4; ++d4) {
    float4 qv = convert_float4(vload_half4(0, (__global half*)Q + q_base + (long)d4 * 4));
    float4 kv = convert_float4(vload_half4(0, (__global half*)K + k_base + (long)d4 * 4));
    acc += dot(qv, kv);
  }
  for (int d = (D4 << 2); d < D; ++d) {
    acc += (float)LOAD(Q, q_base + d) * (float)LOAD(K, k_base + d);
  }
#else
  for (int d = 0; d < D; ++d) {
    acc += (float)LOAD(Q, q_base + d) * (float)LOAD(K, k_base + d);
  }
#endif

  const long idx = ((((long)b * (long)H + (long)h) * (long)T + (long)tq) * (long)T + (long)tk);
  STORE(scores, idx, (storage_t)(acc * scale));
}

// Add attention mask in-place: scores += mask
// mask: [B, 1, T, T]
// Global work (OpenCL 1.2 max 3D): (B, H, TT) where TT=T*T.
__kernel void mha_add_mask(
    __global storage_t* scores,
    __global const storage_t* mask,
    const int B,
    const int H,
    const int T) {
  const int b = (int)get_global_id(0);
  const int h = (int)get_global_id(1);
  const int tt = (int)get_global_id(2);
  if (b >= B || h >= H || tt >= T * T) return;
  const int tq = tt / T;
  const int tk = tt - tq * T;

  const long idx = ((((long)b * (long)H + (long)h) * (long)T + (long)tq) * (long)T + (long)tk);
  const long midx = (((long)b * (long)T + (long)tq) * (long)T + (long)tk);

  float v = (float)LOAD(scores, idx);
  float m = (float)LOAD(mask, midx);
  STORE(scores, idx, (storage_t)(v + m));
}

// Softmax over last dim (tk) for each (b,h,tq) row.
// Updates scores in-place to probabilities.
// Global work: (B, H, T)
__kernel void mha_softmax(
    __global storage_t* scores,
    const int B,
    const int H,
    const int T) {
  const int b = (int)get_global_id(0);
  const int h = (int)get_global_id(1);
  const int tq = (int)get_global_id(2);
  if (b >= B || h >= H || tq >= T) return;

  const long base = (((long)b * (long)H + (long)h) * (long)T + (long)tq) * (long)T;

  float maxv = -3.402823466e+38f;
  for (int tk = 0; tk < T; ++tk) {
    float v = (float)LOAD(scores, base + tk);
    if (v > maxv) maxv = v;
  }

  float sum = 0.0f;
  for (int tk = 0; tk < T; ++tk) {
    float v = (float)LOAD(scores, base + tk);
    float e = exp(v - maxv);
    sum += e;
    STORE(scores, base + tk, (storage_t)e);
  }

  const float inv = 1.0f / (sum + 1e-20f);
  for (int tk = 0; tk < T; ++tk) {
    float e = (float)LOAD(scores, base + tk);
    STORE(scores, base + tk, (storage_t)(e * inv));
  }
}

// out[b,h,tq,d] = sum_tk probs[b,h,tq,tk] * V[b,h,tk,d]
// Global work (OpenCL 1.2 max 3D): (B, H, TD4) where TD4=T*(D/4).
__kernel void mha_out(
    __global const storage_t* probs,
    __global const storage_t* V,
    __global storage_t* out,
    const int B,
    const int H,
    const int T,
    const int D) {
  const int b = (int)get_global_id(0);
  const int h = (int)get_global_id(1);
  const int td4 = (int)get_global_id(2);

  const int D4 = D >> 2;
  if (b >= B || h >= H || td4 >= T * D4) return;
  const int tq = td4 / D4;
  const int d4 = td4 - tq * D4;

  const long probs_base = (((long)b * (long)H + (long)h) * (long)T + (long)tq) * (long)T;
  const long v_base = ((long)b * (long)H + (long)h) * (long)T * (long)D;

#ifdef USE_FP16
  float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  for (int tk = 0; tk < T; ++tk) {
    float p = (float)LOAD(probs, probs_base + tk);
    float4 vv = convert_float4(vload_half4(0, (__global half*)V + v_base + (long)tk * (long)D + (long)d4 * 4));
    acc += p * vv;
  }
  const long out_idx = (((long)b * (long)H + (long)h) * (long)T + (long)tq) * (long)D + (long)d4 * 4;
  STORE(out, out_idx + 0, (storage_t)acc.s0);
  STORE(out, out_idx + 1, (storage_t)acc.s1);
  STORE(out, out_idx + 2, (storage_t)acc.s2);
  STORE(out, out_idx + 3, (storage_t)acc.s3);
#else
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (int tk = 0; tk < T; ++tk) {
    float p = (float)LOAD(probs, probs_base + tk);
    const long off = v_base + (long)tk * (long)D + (long)d4 * 4;
    a0 += p * (float)LOAD(V, off + 0);
    a1 += p * (float)LOAD(V, off + 1);
    a2 += p * (float)LOAD(V, off + 2);
    a3 += p * (float)LOAD(V, off + 3);
  }
  const long out_idx = (((long)b * (long)H + (long)h) * (long)T + (long)tq) * (long)D + (long)d4 * 4;
  STORE(out, out_idx + 0, (storage_t)a0);
  STORE(out, out_idx + 1, (storage_t)a1);
  STORE(out, out_idx + 2, (storage_t)a2);
  STORE(out, out_idx + 3, (storage_t)a3);
#endif
}
