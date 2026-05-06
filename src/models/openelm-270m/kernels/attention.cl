// Reference: model_info/modeling_openelm.py:189-356 (OpenELMMultiHeadCausalAttention.forward)
// Batched-heads attention kernels (Rule HEAD-01).
// Auto-generated transformer scaffold. ALL Q heads computed in single
// kernel launches — collapses the per-head CLBlast-Gemm loop pattern that
// pays 27 launch overheads per layer per token at decode M=1.
//
// Layouts (all token-major):
//   q:        [seq_q, num_q_heads, head_dim]   (NEW, rotated)
//   k_cache:  [max_seq, num_kv_heads, head_dim] (rows 0..seq_k populated)
//   v_cache:  [max_seq, num_kv_heads, head_dim]
//   scores:   [num_q_heads, seq_q, seq_k]
//   out:      [seq_q, num_q_heads, head_dim]
//
// GQA-aware: nrep = num_q_heads / num_kv_heads.
//   nrep == 1            -> Multi-Head Attention
//   nrep > 1, kv_heads>1 -> Grouped-Query Attention
//   num_kv_heads == 1    -> Multi-Query Attention
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


// scores[qh, tq, tk] = scale * dot(q[tq, qh, :], k_cache[tk, kv, :])
// where kv = qh / nrep. Causal mask in absolute positions:
// row tq corresponds to absolute query position (seq_k - seq_q + tq);
// scores[..., tk] = -inf when tk > that absolute position.
//
// Global work: (num_q_heads, seq_q, seq_k).
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
    STORE(scores, score_idx, -3.402823466e+38f);
    return;
  }

  const int nrep = num_q_heads / num_kv_heads;
  const int kvh  = qh / nrep;

  const int q_base = tq * q_dim  + qh  * head_dim;
  const int k_base = tk * kv_dim + kvh * head_dim;

  float acc = 0.0f;
#ifdef USE_FP16
  // Vec4 fp16 inner loop. head_dim is divisible by 4 in every common
  // transformer (64 / 80 / 96 / 128). Scalar tail handles oddities.
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
    float qd = LOAD(q,       q_base + d);
    float kd = LOAD(k_cache, k_base + d);
    acc += qd * kd;
  }
#endif

  STORE(scores, score_idx, acc * scale);
}

// Stable softmax over the last (seq_k) dim of scores[qh, tq, :].
// Causal mask is already applied (entries past abs position hold -inf,
// which exp's to 0), so a plain softmax over the row gives the right
// zeros after exp. fp32 accumulators for numerical stability.
//
// Global work: (num_q_heads * seq_q,) — capped via host-side gws.
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
// where kv = qh / nrep.
//
// Global work: (num_q_heads, seq_q, head_dim).
// Vec4-output variant: each thread emits 4 contiguous d-values via
// vload_half4(v_cache) + vstore_half4(out). Coalesces strided v_cache
// reads — measured +24% decode improvement on SmolLM2-135M / Adreno 620
// (Lever 1, 2026-04-29). Generic for any (head_dim % 4 == 0), which holds
// for every common transformer (head_dim ∈ {64, 80, 96, 128}).
//
// Global work: (num_q_heads, seq_q, head_dim/4).
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
  // out is laid out [seq_q, num_q_heads, head_dim] (token-major) so the
  // residual add downstream is contiguous in the hidden-dim direction.
  const int out_idx = tq * q_dim + qh * head_dim + d4 * 4;
  vstore_half4(acc, 0, (__global half*)out + out_idx);
#else
  // Fp32 fall-back: same 4-output-per-thread structure, scalar reads.
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int tk = 0; tk < seq_k; ++tk) {
    float p = LOAD(scores, score_base + tk);
    int v_off = tk * kv_dim + v_base_d;
    acc[0] += p * LOAD(v_cache, v_off + 0);
    acc[1] += p * LOAD(v_cache, v_off + 1);
    acc[2] += p * LOAD(v_cache, v_off + 2);
    acc[3] += p * LOAD(v_cache, v_off + 3);
  }
  const int out_idx = tq * q_dim + qh * head_dim + d4 * 4;
  STORE(out, out_idx + 0, acc[0]);
  STORE(out, out_idx + 1, acc[1]);
  STORE(out, out_idx + 2, acc[2]);
  STORE(out, out_idx + 3, acc[3]);
#endif
}
