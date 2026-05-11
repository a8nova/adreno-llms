// Reference: model_info/transformers_src/modeling_granitemoehybrid.py:1-90 apply_rotary_pos_emb, rotate_half
// Ported for OpenCL: provides RoPE rotation, GQA attention scores/softmax/out.

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

// Applies RoPE to q/k in-place.
// q layout: [seq_q, num_q_heads, head_dim]
// k layout: [seq_q, num_kv_heads, head_dim]
// cos, sin layout: [seq_total_k, head_dim]
__kernel void rope_apply_qk(
    __global storage_t* q,
    __global storage_t* k,
    __global const storage_t* cos,
    __global const storage_t* sin,
    int seq_q,
    int start_pos,
    int num_q_heads,
    int num_kv_heads,
    int head_dim) {
  int gid = (int)get_global_id(0);
  int total = seq_q * num_q_heads * head_dim;
  if (gid >= total) return;

  int d = gid % head_dim;
  int tmp = gid / head_dim;
  int hq = tmp % num_q_heads;
  int t = tmp / num_q_heads;

  int abs_pos = start_pos + t;
  int cs_base = abs_pos * head_dim + d;
  float c = (float)LOAD(cos, cs_base);
  float s = (float)LOAD(sin, cs_base);

  // HF apply_rotary_pos_emb uses rotate_half(x):
  //   x1 = x[..., :half]
  //   x2 = x[..., half:]
  //   rotate_half(x) = [-x2, x1]
  // and then: x * cos + rotate_half(x) * sin
  int half_dim = head_dim / 2;

  // q always has num_q_heads
  int q_base = (t * num_q_heads + hq) * head_dim;
  int q_idx = q_base + d;
  float qv = (float)LOAD(q, q_idx);
  float q_rot;
  if (d < half_dim) {
    // -x2
    q_rot = -(float)LOAD(q, q_base + d + half_dim);
  } else {
    // x1
    q_rot = (float)LOAD(q, q_base + d - half_dim);
  }
  float q_out = qv * c + q_rot * s;
  STORE(q, q_idx, q_out);

  // k has num_kv_heads; map query head -> kv head group
  int nrep = num_q_heads / num_kv_heads;
  int hk = hq / nrep;
  if (hk < num_kv_heads) {
    int k_base = (t * num_kv_heads + hk) * head_dim;
    int k_idx = k_base + d;
    float kv = (float)LOAD(k, k_idx);
    float k_rot;
    if (d < half_dim) {
      k_rot = -(float)LOAD(k, k_base + d + half_dim);
    } else {
      k_rot = (float)LOAD(k, k_base + d - half_dim);
    }
    float k_out = kv * c + k_rot * s;
    STORE(k, k_idx, k_out);
  }
}

// Compute attention scores.
// q: [seq_q, q_heads, head_dim]
// k_cache: [seq_k, kv_heads, head_dim]
// scores: [seq_q, q_heads, seq_k]
//
// Reference: model_info/transformers_src/modeling_granitemoehybrid.py eager_attention_forward
//   attn_weights = (query @ key_states.transpose(2,3)) * scaling + attention_mask
//   where key_states = repeat_kv(key, nrep)
//
// IMPORTANT: this kernel is 3D-dispatched from C++ as:
//   gws = {seq_k, q_heads, seq_q}
// so we must read get_global_id(0..2) in that SAME order.
__kernel void gqa_attn_scores(
    __global const storage_t* q,
    __global const storage_t* k_cache,
    __global storage_t* scores,
    int seq_q,
    int seq_k,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scaling) {
  int tk = (int)get_global_id(0);
  int hq = (int)get_global_id(1);
  int tq = (int)get_global_id(2);

  if (tk >= seq_k || hq >= num_q_heads || tq >= seq_q) return;

  int nrep = num_q_heads / num_kv_heads;
  int hk = hq / nrep;

  int q_base = (tq * num_q_heads + hq) * head_dim;
  int k_base = (tk * num_kv_heads + hk) * head_dim;

  float acc = 0.0f;
  for (int d = 0; d < head_dim; d++) {
    acc += (float)LOAD(q, q_base + d) * (float)LOAD(k_cache, k_base + d);
  }

  // Causal mask: this port does cached attention.
  // Query absolute position = start_pos + tq, with start_pos = seq_k - seq_q.
  // So abs_q = (seq_k - seq_q) + tq.
  int abs_q = (seq_k - seq_q) + tq;
  if (tk > abs_q) {
    acc = -INFINITY;
  } else {
    acc *= scaling;
  }

  int out_idx = (tq * num_q_heads + hq) * seq_k + tk;
  STORE(scores, out_idx, acc);
}

// In-place softmax over last dim (seq_k) for each (tq, hq)
__kernel void gqa_softmax(
    __global storage_t* scores,
    int seq_q,
    int seq_k,
    int num_q_heads) {
  int gid = (int)get_global_id(0);
  int total = seq_q * num_q_heads;
  if (gid >= total) return;

  int hq = gid % num_q_heads;
  int tq = gid / num_q_heads;

  int base = (tq * num_q_heads + hq) * seq_k;

  float maxv = -INFINITY;
  for (int i = 0; i < seq_k; i++) {
    float v = (float)LOAD(scores, base + i);
    if (v > maxv) maxv = v;
  }

  float sum = 0.0f;
  for (int i = 0; i < seq_k; i++) {
    float e = exp((float)LOAD(scores, base + i) - maxv);
    sum += e;
    STORE(scores, base + i, e);
  }

  float inv = 1.0f / sum;
  for (int i = 0; i < seq_k; i++) {
    float p = (float)LOAD(scores, base + i) * inv;
    STORE(scores, base + i, p);
  }
}

// Compute attn output.
// probs: [seq_q, q_heads, seq_k]
// v_cache: [seq_k, kv_heads, head_dim]
// out: [seq_q, q_heads, head_dim]
//
// IMPORTANT: this kernel is 3D-dispatched from C++ as:
//   gws = {head_dim, q_heads, seq_q}
// so gid0=head_dim index, gid1=head, gid2=time.
__kernel void gqa_attn_out(
    __global const storage_t* probs,
    __global const storage_t* v_cache,
    __global storage_t* out,
    int seq_q,
    int seq_k,
    int num_q_heads,
    int num_kv_heads,
    int head_dim) {
  int d = (int)get_global_id(0);
  int hq = (int)get_global_id(1);
  int tq = (int)get_global_id(2);

  if (d >= head_dim || hq >= num_q_heads || tq >= seq_q) return;

  int nrep = num_q_heads / num_kv_heads;
  int hv = hq / nrep;

  float acc = 0.0f;
  int p_base = (tq * num_q_heads + hq) * seq_k;
  int v_head_base = hv * head_dim;
  for (int tk = 0; tk < seq_k; tk++) {
    float p = (float)LOAD(probs, p_base + tk);
    int v_idx = (tk * num_kv_heads + hv) * head_dim + d;
    acc += p * (float)LOAD(v_cache, v_idx);
  }

  int out_idx = (tq * num_q_heads + hq) * head_dim + d;
  STORE(out, out_idx, acc);
}
