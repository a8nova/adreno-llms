// Batched-heads attention kernels (GQA-aware) for Idefics3 text model.
//
// Reference (attention math): model_info/transformers_src/modeling_idefics3.py:181-201 eager_attention_forward
//   attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
//   if attention_mask is not None: attn_weights = attn_weights + attention_mask
//   attn_weights = softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
//   attn_output  = torch.matmul(attn_weights, value)
//
// NOTE: modeling_idefics3.py's Idefics3VisionAttention is non-causal (is_causal=False) and does not use KV cache.
// This repo's attention op is used for the TEXT decoder attention, which is causal and uses a KV cache.
// RoPE is applied in a separate kernel before these kernels run.
//
// Bias-add note (invariant BIAS-ADD): q_proj/k_proj/v_proj/o_proj biases (when present) are applied in the
// linear/GEMM path (src/ops/linear_0.cpp + kernels/bias_add.cl), not inside these attention kernels.

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

// Tiled prefill scores kernel. One workgroup processes a (BR × BC) tile of
// (tq × tk) pairs for one query head. Each WG cooperatively loads the BR Q-rows
// and BC K-rows into __local memory once (Adreno guide §6.5 + §7.2.2), then
// each work-item computes one score from the cached tiles.
//
// Naive `gqa_attn_scores` (below) re-reads Q + K from global per work-item;
// for seq_q=874 prefill it spends ~14 s in this op. Tiled cuts redundant
// loads by ~BC× for Q and BR× for K.
//
// Global work: (num_q_heads, seq_q_blocks * BR, seq_k_blocks * BC)
// Local:        (1, BR, BC)
#define TILED_BR 16
#define TILED_BC 16
__kernel
__attribute__((reqd_work_group_size(1, TILED_BR, TILED_BC)))
void gqa_attn_scores_tiled(
    __global const storage_t* q,        // [seq_q, num_q_heads, head_dim]
    __global const storage_t* k_cache,  // [seq_k, num_kv_heads, head_dim]
    __global storage_t* scores,         // [num_q_heads, seq_q, seq_k]
    const int seq_q,
    const int seq_k,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale) {
  const int qh       = (int)get_group_id(0);
  const int tq_block = (int)get_group_id(1);
  const int tk_block = (int)get_group_id(2);
  const int lr = (int)get_local_id(1);
  const int lc = (int)get_local_id(2);

  if (qh >= num_q_heads) return;

  const int q_dim  = num_q_heads * head_dim;
  const int kv_dim = num_kv_heads * head_dim;
  const int nrep   = num_q_heads / num_kv_heads;
  const int kvh    = qh / nrep;

  const int tq = tq_block * TILED_BR + lr;
  const int tk = tk_block * TILED_BC + lc;

  // Local tiles, fp16 storage to match global. head_dim assumed 64 here
  // (guarded by `use_tiled` runtime check in the dispatcher).
  __local half q_tile[TILED_BR][64];
  __local half k_tile[TILED_BC][64];

  // Cooperative load of Q tile (BR*head_dim halfs = 1024 halfs = 256 half4
  // loads) and K tile (same). With 256 threads, each thread does exactly
  // one vec4 load for Q and one for K.
  const int tid = lr * TILED_BC + lc;
  {
    int rr = tid / 16;       // 64/4 = 16 vec4s per row
    int d4 = tid - rr * 16;
    int abs_tq = tq_block * TILED_BR + rr;
    float4 v_f;
    if (abs_tq < seq_q) {
#ifdef USE_FP16
      v_f = convert_float4(vload_half4(0, (__global half*)q + abs_tq * q_dim + qh * head_dim + d4 * 4));
#else
      v_f = (float4)((float)LOAD(q, abs_tq * q_dim + qh * head_dim + d4 * 4 + 0),
                     (float)LOAD(q, abs_tq * q_dim + qh * head_dim + d4 * 4 + 1),
                     (float)LOAD(q, abs_tq * q_dim + qh * head_dim + d4 * 4 + 2),
                     (float)LOAD(q, abs_tq * q_dim + qh * head_dim + d4 * 4 + 3));
#endif
    } else {
      v_f = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
    half4 v_h = convert_half4(v_f);
    vstore4(v_h, 0, &q_tile[rr][d4 * 4]);
  }
  {
    int cc = tid / 16;
    int d4 = tid - cc * 16;
    int abs_tk = tk_block * TILED_BC + cc;
    float4 v_f;
    if (abs_tk < seq_k) {
#ifdef USE_FP16
      v_f = convert_float4(vload_half4(0, (__global half*)k_cache + abs_tk * kv_dim + kvh * head_dim + d4 * 4));
#else
      v_f = (float4)((float)LOAD(k_cache, abs_tk * kv_dim + kvh * head_dim + d4 * 4 + 0),
                     (float)LOAD(k_cache, abs_tk * kv_dim + kvh * head_dim + d4 * 4 + 1),
                     (float)LOAD(k_cache, abs_tk * kv_dim + kvh * head_dim + d4 * 4 + 2),
                     (float)LOAD(k_cache, abs_tk * kv_dim + kvh * head_dim + d4 * 4 + 3));
#endif
    } else {
      v_f = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
    half4 v_h = convert_half4(v_f);
    vstore4(v_h, 0, &k_tile[cc][d4 * 4]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (tq >= seq_q || tk >= seq_k) return;

  const int abs_q_pos = (seq_k - seq_q) + tq;
  const int score_idx = (qh * seq_q + tq) * seq_k + tk;
  if (tk > abs_q_pos) {
    STORE(scores, score_idx, (storage_t)(-3.402823466e+38f));
    return;
  }

  // 64-element dot product from local mem as 16 vec4 ops.
  float acc = 0.0f;
  #pragma unroll
  for (int d4 = 0; d4 < 16; ++d4) {
    float4 qv = convert_float4(vload4(0, &q_tile[lr][d4 * 4]));
    float4 kv = convert_float4(vload4(0, &k_tile[lc][d4 * 4]));
    acc += dot(qv, kv);
  }
  STORE(scores, score_idx, (storage_t)(acc * scale));
}

// scores[qh, tq, tk] = scale * dot(q[tq, qh, :], k_cache[tk, kvh, :])
// where kvh = qh / nrep and nrep = num_q_heads / num_kv_heads.
//
// Causal mask in absolute positions (decoder KV cache convention):
//   abs_q_pos = (seq_k - seq_q) + tq
//   if tk > abs_q_pos => score = -3.402823466e+38f
//
// Expected layouts (row-major contiguous):
//   q:       [seq_q, num_q_heads, head_dim]
//   k_cache: [seq_k, num_kv_heads, head_dim]
//   scores:  [num_q_heads, seq_q, seq_k]
//
// Global work: (num_q_heads, seq_q, seq_k)
// Image-backed scores kernel: reads K via texture cache through `read_imageh`.
// K image is [kv_dim/4, seq_k] in CL_RGBA half (4 fp16 per pixel along head_dim).
// One WI per (qh, tq, tk) — same dispatch shape as the non-tiled gqa_attn_scores.
__kernel void gqa_attn_scores_image(
    __global const storage_t* q,
    __read_only image2d_t k_cache_img,
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

  const int q_dim = num_q_heads * head_dim;
  const int abs_q_pos = (seq_k - seq_q) + tq;
  const int score_idx = (qh * seq_q + tq) * seq_k + tk;

  if (tk > abs_q_pos) {
    STORE(scores, score_idx, (storage_t)(-3.402823466e+38f));
    return;
  }

  const int nrep = num_q_heads / num_kv_heads;
  const int kvh = qh / nrep;
  const int D4 = head_dim >> 2;
  const int kv_x_base = kvh * D4;
  const int q_base = tq * q_dim + qh * head_dim;

  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
  float acc = 0.0f;
#ifdef USE_FP16
  for (int d4 = 0; d4 < D4; ++d4) {
    float4 qv = convert_float4(vload_half4(0, (__global half*)q + q_base + d4 * 4));
    float4 kv = convert_float4(read_imageh(k_cache_img, smp, (int2)(kv_x_base + d4, tk)));
    acc += dot(qv, kv);
  }
#else
  // fp32 path falls through to the buffer kernel.
#endif
  STORE(scores, score_idx, (storage_t)(acc * scale));
}

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

  const int q_dim = num_q_heads * head_dim;
  const int kv_dim = num_kv_heads * head_dim;

  const int abs_q_pos = (seq_k - seq_q) + tq;
  const int score_idx = (qh * seq_q + tq) * seq_k + tk;

  if (tk > abs_q_pos) {
    STORE(scores, score_idx, (storage_t)(-3.402823466e+38f));
    return;
  }

  const int nrep = num_q_heads / num_kv_heads;
  const int kvh = qh / nrep;

  const int q_base = tq * q_dim + qh * head_dim;
  const int k_base = tk * kv_dim + kvh * head_dim;

  // Dot product accumulates in fp32 even on fp16 storage.
  float acc = 0.0f;

#ifdef USE_FP16
  // Vec4 fp16 inner loop. head_dim is divisible by 4 for this model (head_dim=64).
  const int D4 = head_dim >> 2;
  for (int d4 = 0; d4 < D4; ++d4) {
    // Use vload_half4 for coalesced reads; convert to float4 for fp32 accumulation.
    float4 qv = convert_float4(vload_half4(0, (__global half*)q + q_base + d4 * 4));
    float4 kv = convert_float4(vload_half4(0, (__global half*)k_cache + k_base + d4 * 4));
    acc += dot(qv, kv);
  }
  // Scalar tail (should be empty for head_dim multiple of 4).
  for (int d = (D4 << 2); d < head_dim; ++d) {
    float qd = (float)LOAD(q, q_base + d);
    float kd = (float)LOAD(k_cache, k_base + d);
    acc += qd * kd;
  }
#else
  for (int d = 0; d < head_dim; ++d) {
    float qd = (float)LOAD(q, q_base + d);
    float kd = (float)LOAD(k_cache, k_base + d);
    acc += qd * kd;
  }
#endif

  STORE(scores, score_idx, (storage_t)(acc * scale));
}

// Stable row-softmax over last dimension (seq_k).
// One workgroup per row; uses cl_khr_subgroups for max/sum reductions —
// no __local memory, no barriers (Adreno guide §6.5 + §9.2).
// We require sub_group_size == WG_SIZE so a single subgroup spans the WG.
//
// Global work: (total_rows * 64), local: 64
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define SM_WG 64
__kernel
__attribute__((reqd_work_group_size(SM_WG, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("full")))
void gqa_softmax(
    __global storage_t* scores,
    const int seq_q,
    const int seq_k,
    const int total_rows) {
  (void)seq_q;
  const int r = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  if (r >= total_rows) return;

  const int base = r * seq_k;

  // Pass 1: per-lane local max → subgroup_reduce_max.
  float lm = -3.402823466e+38f;
  for (int c = lid; c < seq_k; c += SM_WG) {
    float v = (float)LOAD(scores, base + c);
    if (v > lm) lm = v;
  }
  float row_max = sub_group_reduce_max(lm);

  // Pass 2: exp + write back + per-lane partial sum → subgroup_reduce_add.
  float ls = 0.0f;
  for (int c = lid; c < seq_k; c += SM_WG) {
    float v = (float)LOAD(scores, base + c);
    float e = native_exp(v - row_max);
    STORE(scores, base + c, (storage_t)e);
    ls += e;
  }
  float row_sum = sub_group_reduce_add(ls);

  // Pass 3: normalize.
  const float inv = native_recip(row_sum + 1e-20f);
  for (int c = lid; c < seq_k; c += SM_WG) {
    float e = (float)LOAD(scores, base + c);
    STORE(scores, base + c, (storage_t)(e * inv));
  }
}

// Decode-only variant of gqa_attn_out (seq_q==1). One workgroup per (qh, d4);
// 64 lanes parallel-reduce the sum over seq_k. Eliminates the 144-workitem
// occupancy bottleneck at decode time. Prefill keeps the simple kernel below.
//
// Global work: (num_q_heads, head_dim/4 * 64), local: (1, 64)
#define AO_WG 64
__kernel
__attribute__((reqd_work_group_size(1, AO_WG, 1)))
void gqa_attn_out_decode(
    __global const storage_t* scores,
    __global const storage_t* v_cache,
    __global storage_t* out,
    const int seq_k,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim) {
  const int qh = (int)get_group_id(0);
  const int d4 = (int)get_group_id(1);
  const int lid = (int)get_local_id(1);
  const int D4 = head_dim >> 2;
  if (qh >= num_q_heads || d4 >= D4) return;

  const int q_dim = num_q_heads * head_dim;
  const int kv_dim = num_kv_heads * head_dim;
  const int nrep = num_q_heads / num_kv_heads;
  const int kvh = qh / nrep;
  const int score_base = qh * seq_k;
  const int v_base_d = kvh * head_dim + d4 * 4;

  __local float r0[AO_WG], r1[AO_WG], r2[AO_WG], r3[AO_WG];
#ifdef USE_FP16
  float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  for (int tk = lid; tk < seq_k; tk += AO_WG) {
    float p = (float)LOAD(scores, score_base + tk);
    float4 vv = convert_float4(vload_half4(0, (__global half*)v_cache + tk * kv_dim + v_base_d));
    acc += p * vv;
  }
  r0[lid] = acc.s0; r1[lid] = acc.s1; r2[lid] = acc.s2; r3[lid] = acc.s3;
#else
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (int tk = lid; tk < seq_k; tk += AO_WG) {
    float p = (float)LOAD(scores, score_base + tk);
    const int v_off = tk * kv_dim + v_base_d;
    a0 += p * (float)LOAD(v_cache, v_off + 0);
    a1 += p * (float)LOAD(v_cache, v_off + 1);
    a2 += p * (float)LOAD(v_cache, v_off + 2);
    a3 += p * (float)LOAD(v_cache, v_off + 3);
  }
  r0[lid] = a0; r1[lid] = a1; r2[lid] = a2; r3[lid] = a3;
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = AO_WG >> 1; s > 0; s >>= 1) {
    if (lid < s) {
      r0[lid] += r0[lid + s];
      r1[lid] += r1[lid + s];
      r2[lid] += r2[lid + s];
      r3[lid] += r3[lid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    const int out_idx = qh * head_dim + d4 * 4;
    STORE(out, out_idx + 0, (storage_t)r0[0]);
    STORE(out, out_idx + 1, (storage_t)r1[0]);
    STORE(out, out_idx + 2, (storage_t)r2[0]);
    STORE(out, out_idx + 3, (storage_t)r3[0]);
  }
}

// Tiled prefill attn_out. One WG handles (qh, tq_block, d4): AO_BR consecutive
// query tokens, single d4 output slot. All AO_BR threads read the SAME
// v[tk, kvh, d4*4..+3] each step → cooperative load shares the V read across
// AO_BR lanes. Process tk in chunks of AO_BC (loaded once into __local).
//
// Global work: (num_q_heads, seq_q_blocks * AO_BR, head_dim/4)
// Local:        (1, AO_BR, 1)
#define AO_BR 64
#define AO_BC 16
__kernel
__attribute__((reqd_work_group_size(1, AO_BR, 1)))
void gqa_attn_out_tiled(
    __global const storage_t* scores,
    __global const storage_t* v_cache,
    __global storage_t* out,
    const int seq_q,
    const int seq_k,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim) {
  const int qh       = (int)get_group_id(0);
  const int tq_block = (int)get_group_id(1);
  const int d4       = (int)get_group_id(2);
  const int lr       = (int)get_local_id(1);
  const int tq       = tq_block * AO_BR + lr;

  if (qh >= num_q_heads) return;
  const int q_dim = num_q_heads * head_dim;
  const int kv_dim = num_kv_heads * head_dim;
  const int nrep = num_q_heads / num_kv_heads;
  const int kvh = qh / nrep;

  __local float v_chunk[AO_BC][4];
  float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

  for (int tk_block = 0; tk_block < seq_k; tk_block += AO_BC) {
    // Cooperative V load: lanes [0, AO_BC) load one V slice each.
    if (lr < AO_BC) {
      int tk = tk_block + lr;
      float4 vv;
      if (tk < seq_k) {
#ifdef USE_FP16
        vv = convert_float4(vload_half4(0, (__global half*)v_cache + tk * kv_dim + kvh * head_dim + d4 * 4));
#else
        vv = (float4)((float)LOAD(v_cache, tk * kv_dim + kvh * head_dim + d4 * 4 + 0),
                      (float)LOAD(v_cache, tk * kv_dim + kvh * head_dim + d4 * 4 + 1),
                      (float)LOAD(v_cache, tk * kv_dim + kvh * head_dim + d4 * 4 + 2),
                      (float)LOAD(v_cache, tk * kv_dim + kvh * head_dim + d4 * 4 + 3));
#endif
      } else {
        vv = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
      }
      v_chunk[lr][0] = vv.s0;
      v_chunk[lr][1] = vv.s1;
      v_chunk[lr][2] = vv.s2;
      v_chunk[lr][3] = vv.s3;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tq < seq_q) {
      const int score_base = (qh * seq_q + tq) * seq_k;
      #pragma unroll
      for (int c = 0; c < AO_BC; ++c) {
        int tk = tk_block + c;
        if (tk >= seq_k) break;
        float p = (float)LOAD(scores, score_base + tk);
        acc.s0 += p * v_chunk[c][0];
        acc.s1 += p * v_chunk[c][1];
        acc.s2 += p * v_chunk[c][2];
        acc.s3 += p * v_chunk[c][3];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tq < seq_q) {
    const int out_idx = tq * q_dim + qh * head_dim + d4 * 4;
    STORE(out, out_idx + 0, (storage_t)acc.s0);
    STORE(out, out_idx + 1, (storage_t)acc.s1);
    STORE(out, out_idx + 2, (storage_t)acc.s2);
    STORE(out, out_idx + 3, (storage_t)acc.s3);
  }
}

// out[tq, qh, d] = sum_tk scores[qh, tq, tk] * v_cache[tk, kvh, d]
// where kvh = qh / nrep.
//
// Expected layouts (row-major contiguous):
//   scores:  [num_q_heads, seq_q, seq_k]
//   v_cache: [seq_k, num_kv_heads, head_dim]
//   out:     [seq_q, num_q_heads, head_dim]
//
// Image-backed prefill attn_out — reads v_cache via read_imageh through the
// L1 texture cache instead of buffer + L2. v_cache image is [kv_dim/4, seq_k]
// with CL_RGBA half (4 fp16 per pixel along head_dim).
//
// Global work: (num_q_heads, seq_q, head_dim/4)
__kernel void gqa_attn_out_image(
    __global const storage_t* scores,
    __read_only image2d_t v_cache_img,
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

  const int q_dim = num_q_heads * head_dim;
  const int kv_d4 = (num_kv_heads * head_dim) >> 2;
  const int nrep = num_q_heads / num_kv_heads;
  const int kvh = qh / nrep;

  const int score_base = (qh * seq_q + tq) * seq_k;
  // Image x-coord for this (kvh, d4): kvh*D4 + d4 — pixels within the row.
  const int v_x = kvh * D4 + d4;
  (void)kv_d4;

  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;

#ifdef USE_FP16
  float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  for (int tk = 0; tk < seq_k; ++tk) {
    float p = (float)LOAD(scores, score_base + tk);
    float4 vv = convert_float4(read_imageh(v_cache_img, smp, (int2)(v_x, tk)));
    acc += p * vv;
  }
  const int out_idx = tq * q_dim + qh * head_dim + d4 * 4;
  STORE(out, out_idx + 0, (storage_t)acc.s0);
  STORE(out, out_idx + 1, (storage_t)acc.s1);
  STORE(out, out_idx + 2, (storage_t)acc.s2);
  STORE(out, out_idx + 3, (storage_t)acc.s3);
#else
  // fp32 path not currently used for prefill; fall through to the buffer kernel.
#endif
}

// Global work: (num_q_heads, seq_q, head_dim/4)
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

  const int q_dim = num_q_heads * head_dim;
  const int kv_dim = num_kv_heads * head_dim;

  const int nrep = num_q_heads / num_kv_heads;
  const int kvh = qh / nrep;

  const int score_base = (qh * seq_q + tq) * seq_k;
  const int v_base_d = kvh * head_dim + d4 * 4;

#ifdef USE_FP16
  float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  for (int tk = 0; tk < seq_k; ++tk) {
    float p = (float)LOAD(scores, score_base + tk);
    float4 vv = convert_float4(vload_half4(0, (__global half*)v_cache + tk * kv_dim + v_base_d));
    acc += p * vv;
  }

  const int out_idx = tq * q_dim + qh * head_dim + d4 * 4;
  STORE(out, out_idx + 0, (storage_t)acc.s0);
  STORE(out, out_idx + 1, (storage_t)acc.s1);
  STORE(out, out_idx + 2, (storage_t)acc.s2);
  STORE(out, out_idx + 3, (storage_t)acc.s3);
#else
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  for (int tk = 0; tk < seq_k; ++tk) {
    float p = (float)LOAD(scores, score_base + tk);
    const int v_off = tk * kv_dim + v_base_d;
    acc0 += p * (float)LOAD(v_cache, v_off + 0);
    acc1 += p * (float)LOAD(v_cache, v_off + 1);
    acc2 += p * (float)LOAD(v_cache, v_off + 2);
    acc3 += p * (float)LOAD(v_cache, v_off + 3);
  }

  const int out_idx = tq * q_dim + qh * head_dim + d4 * 4;
  STORE(out, out_idx + 0, (storage_t)acc0);
  STORE(out, out_idx + 1, (storage_t)acc1);
  STORE(out, out_idx + 2, (storage_t)acc2);
  STORE(out, out_idx + 3, (storage_t)acc3);
#endif
}
