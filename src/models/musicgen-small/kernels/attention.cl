// Reference: model_info/transformers_src/modeling_musicgen.py:179-388 MusicgenAttention.forward
//
// Baseline (non-fused) attention kernels for MusicGen.
// This file currently provides a stub kernel so the runtime can open/compile
// the program file. Actual attention math is implemented in C++/CLBlast in
// src/ops/ops_basic.cpp and/or src/ops/musicgen_decoder_full_attention_block.cpp.
//
// IMPORTANT: This must be dtype-templated (fp16/fp32) and include LOAD/STORE.

#ifdef USE_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half storage_t;
#define LOAD(p, i) vload_half((i), (p))
#define STORE(p, i, v) vstore_half((v), (i), (p))
#else
typedef float storage_t;
#define LOAD(p, i) ((p)[(i)])
#define STORE(p, i, v) ((p)[(i)] = (v))
#endif

// scores: [num_q_heads, seq_q, seq_k]
// Q, K, V are token-major [seq_len, hidden_size].
// We interpret Q/K/V as if they were already reshaped into heads:
//   q[h, t, d] = Q[t, h*head_dim + d]
// For GQA, kv heads are repeated: kv_h = h / nrep.
__kernel void gqa_attn_scores(
    __global const storage_t* q,
    __global const storage_t* k,
    __global float* scores,
    int seq_q,
    int seq_k,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    int causal) {
  int tk = (int)get_global_id(0);
  int tq = (int)get_global_id(1);
  int hq = (int)get_global_id(2);
  if (tk >= seq_k || tq >= seq_q || hq >= num_q_heads) return;

  if (causal && tk > tq) {
    scores[(hq * seq_q + tq) * seq_k + tk] = -1.0e9f;
    return;
  }

  int nrep = (num_kv_heads > 0) ? (num_q_heads / num_kv_heads) : 1;
  int hk = (nrep > 0) ? (hq / nrep) : hq;
  if (hk >= num_kv_heads) hk = num_kv_heads - 1;

  float acc = 0.0f;
  int q_base = tq * (num_q_heads * head_dim) + hq * head_dim;
  int k_base = tk * (num_kv_heads * head_dim) + hk * head_dim;
  for (int d = 0; d < head_dim; ++d) {
    float qv = LOAD(q, q_base + d);
    float kv = LOAD(k, k_base + d);
    acc += qv * kv;
  }
  scores[(hq * seq_q + tq) * seq_k + tk] = acc * scale;
}

// In-place softmax over tk for each (hq, tq)
__kernel void gqa_softmax(__global float* scores, int seq_q, int seq_k, int num_q_heads) {
  int tq = (int)get_global_id(0);
  int hq = (int)get_global_id(1);
  if (tq >= seq_q || hq >= num_q_heads) return;

  int base = (hq * seq_q + tq) * seq_k;
  float maxv = -1.0e20f;
  for (int tk = 0; tk < seq_k; ++tk) {
    float v = scores[base + tk];
    if (v > maxv) maxv = v;
  }
  float sum = 0.0f;
  for (int tk = 0; tk < seq_k; ++tk) {
    float e = exp(scores[base + tk] - maxv);
    scores[base + tk] = e;
    sum += e;
  }
  float inv = 1.0f / (sum + 1.0e-20f);
  for (int tk = 0; tk < seq_k; ++tk) {
    scores[base + tk] *= inv;
  }
}

// out_heads: [num_q_heads, seq_q, head_dim]
__kernel void gqa_attn_out(
    __global const float* scores,
    __global const storage_t* v,
    __global storage_t* out_heads,
    int seq_q,
    int seq_k,
    int num_q_heads,
    int num_kv_heads,
    int head_dim) {
  int d = (int)get_global_id(0);
  int tq = (int)get_global_id(1);
  int hq = (int)get_global_id(2);
  if (d >= head_dim || tq >= seq_q || hq >= num_q_heads) return;

  int nrep = (num_kv_heads > 0) ? (num_q_heads / num_kv_heads) : 1;
  int hk = (nrep > 0) ? (hq / nrep) : hq;
  if (hk >= num_kv_heads) hk = num_kv_heads - 1;

  float acc = 0.0f;
  int score_base = (hq * seq_q + tq) * seq_k;
  for (int tk = 0; tk < seq_k; ++tk) {
    float w = scores[score_base + tk];
    int v_idx = tk * (num_kv_heads * head_dim) + hk * head_dim + d;
    float vv = LOAD(v, v_idx);
    acc += w * vv;
  }
  int out_idx = (hq * seq_q + tq) * head_dim + d;
  STORE(out_heads, out_idx, acc);
}

// ── Fused single-query attention (decode, seq_q == 1) ───────────────────────
// One workgroup per head (global = num_heads*WG, local = WG). Computes, for head
// h: scores[t] = scale * dot(Q_h, K_h[t]) for t in [0,seq_k) (causal: all valid
// since the single query is the last position), softmax over t, then
// out[h*head_dim + d] = sum_t softmax[t] * V_h[t,d]. Collapses the 4-dispatch
// scores→softmax→out→heads_to_tokens chain into ONE dispatch. Math identical to
// the unfused kernels (fp32 accumulation, same scale, same row-major layouts).
//   q:   [hidden]                 (single query row)
//   k,v: [seq_k, hidden]          (token-major; head h occupies cols h*hd..+hd)
//   out: [hidden]                 (token-major, head-concatenated)
// WG must be >= head_dim and a power of two for the reduction; head_dim=64 → WG=64.
#define FUSED_ATTN_MAXK 256
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void fused_decode_attention(
    __global const storage_t* q,
    __global const storage_t* k,
    __global const storage_t* v,
    __global storage_t* out,
    const int seq_k,
    const int num_heads,
    const int head_dim,
    const float scale) {
  const int h = get_group_id(0);
  const int lid = get_local_id(0);
  const int WG = 64;
  if (h >= num_heads) return;
  const int hidden = num_heads * head_dim;
  const int q_base = h * head_dim;

  __local float qsh[64];          // Q_h cached in local
  __local float sc[FUSED_ATTN_MAXK];   // scores / probabilities
  __local float red[64];          // reduction scratch

  if (lid < head_dim) qsh[lid] = LOAD(q, q_base + lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  // scores[t] = scale * dot(Q_h, K_h[t]); threads stride over t.
  for (int t = lid; t < seq_k; t += WG) {
    int k_base = t * hidden + h * head_dim;
    float acc = 0.0f;
    for (int d = 0; d < head_dim; ++d) acc += qsh[d] * LOAD(k, k_base + d);
    sc[t] = acc * scale;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // max-reduce over scores (numerical stability) — parallel over WG.
  float m = -1.0e30f;
  for (int t = lid; t < seq_k; t += WG) m = fmax(m, sc[t]);
  red[lid] = m; barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = WG >> 1; o > 0; o >>= 1) { if (lid < o) red[lid] = fmax(red[lid], red[lid+o]); barrier(CLK_LOCAL_MEM_FENCE); }
  const float maxv = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  // exp + sum
  float s = 0.0f;
  for (int t = lid; t < seq_k; t += WG) { float e = exp(sc[t] - maxv); sc[t] = e; s += e; }
  red[lid] = s; barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = WG >> 1; o > 0; o >>= 1) { if (lid < o) red[lid] += red[lid+o]; barrier(CLK_LOCAL_MEM_FENCE); }
  const float inv = 1.0f / (red[0] + 1.0e-20f);
  barrier(CLK_LOCAL_MEM_FENCE);

  // out[d] = sum_t p[t] * V_h[t,d]; thread d owns output dim d.
  if (lid < head_dim) {
    float acc = 0.0f;
    for (int t = 0; t < seq_k; ++t) {
      acc += sc[t] * inv * LOAD(v, t * hidden + h * head_dim + lid);
    }
    STORE(out, h * head_dim + lid, acc);
  }
}

// out_tokens: [seq_len, num_heads*head_dim] from in_heads [num_heads, seq_len, head_dim]
__kernel void heads_to_tokens(
    __global const storage_t* in_heads,
    __global storage_t* out_tokens,
    int seq_len,
    int num_heads,
    int head_dim) {
  int gid = (int)get_global_id(0);
  int hidden = num_heads * head_dim;
  int t = gid / hidden;
  int c = gid - t * hidden;
  if (t >= seq_len) return;
  int h = c / head_dim;
  int d = c - h * head_dim;
  int in_idx = (h * seq_len + t) * head_dim + d;
  float v = LOAD(in_heads, in_idx);
  STORE(out_tokens, gid, v);
}
