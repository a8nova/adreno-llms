// Reference: model_info/transformers_src/modeling_lfm2.py:185-224 eager_attention_forward (matmul+softmax+matmul), repeat_kv
// Batched-heads attention kernels (Rule HEAD-01).
// Auto-generated transformer scaffold.

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

// scores[qh, tq, tk] = scale * dot(q[tq, qh, :], k_cache[tk, kv, :])
// kv = qh / nrep (GQA), nrep = num_q_heads / num_kv_heads.
// Causal mask in absolute positions: abs_q_pos = (seq_k - seq_q) + tq.
__kernel void gqa_attn_scores(
    __global const storage_t* q,
    __global const storage_t* k_cache,
    __global storage_t* scores,
    const int seq_q,
    const int seq_k,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale) {
  const int qh = (int)get_global_id(0);
  const int tq = (int)get_global_id(1);
  const int tk = (int)get_global_id(2);

  if (qh >= num_q_heads || tq >= seq_q || tk >= seq_k) return;

  const int q_dim  = num_q_heads  * head_dim;
  const int kv_dim = num_kv_heads * head_dim;

  const int abs_q_pos = (seq_k - seq_q) + tq;
  const int score_idx = (qh * seq_q + tq) * seq_k + tk;

  if (tk > abs_q_pos) {
#ifdef USE_FP16
    // For fp16 storage: avoid overflowing half with -FLT_MAX (becomes -inf) which then
    // pollutes max-reduction and can lead to inf/NaN in later math on some drivers.
    // Use a representable large negative sentinel.
    STORE(scores, score_idx, -1.0e4f);
#else
    STORE(scores, score_idx, -3.402823466e+38f);
#endif
    return;
  }

  const int nrep = num_q_heads / num_kv_heads;
  const int kvh  = qh / nrep;

  const int q_base = tq * q_dim  + qh  * head_dim;
  const int k_base = tk * kv_dim + kvh * head_dim;

  float acc = 0.0f;
#ifdef USE_FP16
  const int D4 = head_dim >> 2;
  for (int d4 = 0; d4 < D4; ++d4) {
    float4 qv = vload_half4(0, (__global half*)q       + q_base + d4 * 4);
    float4 kv = vload_half4(0, (__global half*)k_cache + k_base + d4 * 4);
    acc += dot(qv, kv);
  }
  for (int d = D4 * 4; d < head_dim; ++d) {
    acc += LOAD(q, q_base + d) * LOAD(k_cache, k_base + d);
  }
#else
  for (int d = 0; d < head_dim; ++d) {
    acc += LOAD(q, q_base + d) * LOAD(k_cache, k_base + d);
  }
#endif

  STORE(scores, score_idx, acc * scale);
}

// Stable softmax over last dim (seq_k) for each row.
__kernel void gqa_softmax(
    __global storage_t* scores,
    const int seq_q,
    const int seq_k,
    const int total_rows) {
  const int r = (int)get_global_id(0);
  if (r >= total_rows) return;

  const int base = r * seq_k;

  float maxv = -3.402823466e+38f;
  for (int c = 0; c < seq_k; ++c) {
    float v = LOAD(scores, base + c);
    if (v > maxv) maxv = v;
  }

  float sum = 0.0f;
  for (int c = 0; c < seq_k; ++c) {
    float v = LOAD(scores, base + c);
    float e = exp(v - maxv);
    sum += e;
    STORE(scores, base + c, e);
  }

  float inv = 1.0f / (sum + 1e-20f);
  for (int c = 0; c < seq_k; ++c) {
    float e = LOAD(scores, base + c);
    STORE(scores, base + c, e * inv);
  }
}

// out[tq, qh, d] = sum_tk scores[qh, tq, tk] * v_cache[tk, kv, d]
// Work-item computes 4 contiguous d values.
__kernel void gqa_attn_out(
    __global const storage_t* scores,
    __global const storage_t* v_cache,
    __global storage_t* out,
    const int seq_q,
    const int seq_k,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim) {
  const int qh = (int)get_global_id(0);
  const int tq = (int)get_global_id(1);
  const int d4 = (int)get_global_id(2);

  const int D4 = head_dim >> 2;
  if (qh >= num_q_heads || tq >= seq_q || d4 >= D4) return;

  const int q_dim  = num_q_heads  * head_dim;
  const int kv_dim = num_kv_heads * head_dim;
  const int nrep   = num_q_heads / num_kv_heads;
  const int kvh    = qh / nrep;

  const int score_base = (qh * seq_q + tq) * seq_k;
  const int v_base_d   = kvh * head_dim + d4 * 4;

#ifdef USE_FP16
  float4 acc = (float4)(0.0f);
  for (int tk = 0; tk < seq_k; ++tk) {
    float p  = LOAD(scores, score_base + tk);
    float4 v = vload_half4(0, (__global half*)v_cache + tk * kv_dim + v_base_d);
    acc += p * v;
  }
  const int out_idx = tq * q_dim + qh * head_dim + d4 * 4;
  vstore_half4(acc, 0, (__global half*)out + out_idx);
#else
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  for (int tk = 0; tk < seq_k; ++tk) {
    float p = LOAD(scores, score_base + tk);
    int v_off = tk * kv_dim + v_base_d;
    acc0 += p * LOAD(v_cache, v_off + 0);
    acc1 += p * LOAD(v_cache, v_off + 1);
    acc2 += p * LOAD(v_cache, v_off + 2);
    acc3 += p * LOAD(v_cache, v_off + 3);
  }
  const int out_idx = tq * q_dim + qh * head_dim + d4 * 4;
  STORE(out, out_idx + 0, acc0);
  STORE(out, out_idx + 1, acc1);
  STORE(out, out_idx + 2, acc2);
  STORE(out, out_idx + 3, acc3);
#endif
}


// kv_write — single-row append into the persistent KV cache. Replaces the
// non-recordable clEnqueueCopyBuffer used in attention.cpp's decode path so
// the entire forward pass can be captured by cl_qcom_recordable_queues.
// Only NDRangeKernel is recordable per Snapdragon Programming Guide §9.1.3.
//
// One work-item per element in the row. Launch with gws=kv_dim, lws=64
// (kv_dim is a multiple of 64 in LFM2 — KVH*D = 8*64 = 512).
__kernel void kv_write(
    __global const storage_t* src,
    __global storage_t* cache,
    const int start_pos,
    const int kv_dim) {
  const int gid = (int)get_global_id(0);
  if (gid >= kv_dim) return;
  STORE(cache, start_pos * kv_dim + gid, LOAD(src, gid));
}
