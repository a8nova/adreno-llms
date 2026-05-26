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

// Tiled scores: each WG handles a BR×BC block of (tq, tk) for one (b, h).
// Each WG cooperatively loads Q[tq_block, :] + K[tk_block, :] into __local,
// then each thread computes one score from the tiles. Cuts global-memory reads
// by BR× and BC× respectively (D-dim is fully reused inside the tile).
//
// fp16, D divisible by 4. For vision: B=1, H=12, T=1024, D=64.
//   BR=16, BC=16 → local mem 16*64*2 + 16*64*2 = 4 KB (well under 32 KB)
//   gws = (H, T/BC, T) — split: head, key-block-index, query-row
//   lws = (1, 1, BR) — 16 threads per WG, each one query row of the BR
//   per thread: 64-D dot product → ~64 FP muls/adds, all from local mem
#ifdef USE_FP16
// Tiled mha_scores. Each WG handles a BR×BC tile (BR query rows × BC key rows)
// for one (b, h). Cooperatively loads Q tile + K tile into __local, then each
// of BR threads computes BC scores per row.
//
// NDRange:
//   gws = (B*H, T, T/BC)   — total ((B*H), (T/BR groups × BR threads), (T/BC groups × 1 thread))
//   lws = (1, BR, 1)       — BR threads per WG, one per query row
//   WG count: B*H × (T/BR) × (T/BC)
//   For vision (B=1, H=12, T=1024, BR=BC=16): 12 × 64 × 64 = 49,152 WGs × 16 threads = 786K work-items.
//   Each thread reads only its row of Q (cooperative) + computes BC dot products from the K tile.
//   Memory traffic per layer: (BR + BC) × T × D × (T/BR) reads ≈ 32 × 1024 × 64 × 64 = 134 MB
//                              vs naive 2 × T × T × D = 128 MB — same total, but accesses
//                              are reused inside the workgroup (L1/local hits), so effective.
#define VIS_BR 16
#define VIS_BC 16
__kernel
__attribute__((reqd_work_group_size(1, VIS_BR, 1)))
void mha_scores_tiled(
    __global const half* Q,
    __global const half* K,
    __global half* scores,
    const int B,
    const int H,
    const int T,
    const int D,
    const float scale) {
  const int bh = (int)get_group_id(0);
  const int qg = (int)get_group_id(1);          // query block index
  const int kc = (int)get_group_id(2);          // key block index
  const int br = (int)get_local_id(1);          // 0..VIS_BR-1

  const int b = bh / H;
  const int h = bh - b * H;
  if (b >= B || h >= H) return;

  const int tq = qg * VIS_BR + br;
  const int tk_base = kc * VIS_BC;
  if (tq >= T || tk_base >= T) return;

  const long bh_off = ((long)b * (long)H + (long)h) * (long)T * (long)D;
  __global const half* Qp = Q + bh_off;
  __global const half* Kp = K + bh_off;

  __local half Q_tile[VIS_BR * 64];
  __local half K_tile[VIS_BC * 64];

  // Cooperatively load Q_tile[br, :D] from global. BR threads × D elements,
  // each thread loads D/BR elements. Assuming D is a multiple of BR (D=64, BR=16 → 4 each).
  const int per_thread = D / VIS_BR;
  for (int i = 0; i < per_thread; ++i) {
    int d = i * VIS_BR + br;
    if (d < D) {
      Q_tile[br * D + d] = Qp[(long)tq * (long)D + d];
    }
  }
  // K_tile[br, :D] = K[tk_base + br, :D]
  const int tk_my = tk_base + br;
  if (tk_my < T) {
    for (int i = 0; i < per_thread; ++i) {
      int d = i * VIS_BR + br;
      if (d < D) {
        K_tile[br * D + d] = Kp[(long)tk_my * (long)D + d];
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Now each thread `br` computes scores[tq, tk_base..tk_base+BC-1].
  const long s_base = (((long)b * (long)H + (long)h) * (long)T + (long)tq) * (long)T;
  const int D4 = D >> 2;
  for (int j = 0; j < VIS_BC; ++j) {
    int tk = tk_base + j;
    if (tk >= T) break;
    float acc = 0.0f;
    for (int d4 = 0; d4 < D4; ++d4) {
      float4 qv = convert_float4(vload_half4(0, (__local half*)(Q_tile + br * D) + d4 * 4));
      float4 kv = convert_float4(vload_half4(0, (__local half*)(K_tile + j  * D) + d4 * 4));
      acc += dot(qv, kv);
    }
    scores[s_base + tk] = (half)(acc * scale);
  }
}
#endif

// Coalesced scores: 1 WG per (b, h, tq) row. All 64 lanes cooperate on a
// single tk dot product per iteration — every lane reads K[tk, lid] (one
// coalesced 128-B cacheline per WG per iter) and sub_group_reduce_add the
// partial product. T=1024 iters per WG; 12×1024 WGs total.
//
// Win over mha_scores_par (which has 64 lanes reading 64 DIFFERENT K rows
// simultaneously — strided across 64 cachelines): 64× fewer memory
// transactions, near-perfect L2 reuse of K across query rows.
#ifdef USE_FP16
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define VIS_SC_WG_CO 64
__kernel
__attribute__((reqd_work_group_size(VIS_SC_WG_CO, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("full")))
void mha_scores_coalesced(
    __global const half* Q,
    __global const half* K,
    __global half* scores,
    const int B,
    const int H,
    const int T,
    const int D,
    const float scale) {
  const int lid = (int)get_local_id(0);
  const int b   = (int)get_group_id(0);
  const int h   = (int)get_group_id(1);
  const int tq  = (int)get_group_id(2);
  if (b >= B || h >= H || tq >= T) return;

  const long bh_off = ((long)b * (long)H + (long)h) * (long)T * (long)D;

  // Load Q[tq, :D] into __local once. D=64 expected; lane lid loads lane lid.
  __local half Q_local[64];
  if (lid < D) Q_local[lid] = Q[bh_off + (long)tq * (long)D + lid];
  barrier(CLK_LOCAL_MEM_FENCE);

  const float q_elt = (float)Q_local[lid];

  const long s_base = (((long)b * (long)H + (long)h) * (long)T + (long)tq) * (long)T;

  // Each tk iter: all 64 lanes read K[tk, lid] (ONE cacheline / WG / iter).
  // sub_group_reduce_add collapses to row dot product. Lane 0 writes scores.
  for (int tk = 0; tk < T; ++tk) {
    const float k_elt = (float)K[bh_off + (long)tk * (long)D + lid];
    const float prod = q_elt * k_elt;
    const float dot_v = sub_group_reduce_add(prod);
    if (lid == 0) {
      scores[s_base + tk] = (half)(dot_v * scale);
    }
  }
}
#endif

// Coalesced mha_out: 1 WG per (b, h, tq), 64 lanes cooperate. Lane lid holds
// the running accumulator for output element [tq, lid]. Each tk iter all 64
// lanes do ONE coalesced cacheline read of V[tk, 0..63], multiply by the
// shared probs[tq, tk], and accumulate.
//
// Win over mha_out (one WI per output, 16 D4 strided readers across the row):
// 16× fewer V transactions, perfect cacheline-aligned reads.
#ifdef USE_FP16
#define VIS_OUT_WG_CO 64
__kernel
__attribute__((reqd_work_group_size(VIS_OUT_WG_CO, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("full")))
void mha_out_coalesced(
    __global const half* probs,
    __global const half* V,
    __global half* out,
    const int B,
    const int H,
    const int T,
    const int D) {
  const int lid = (int)get_local_id(0);
  const int b   = (int)get_group_id(0);
  const int h   = (int)get_group_id(1);
  const int tq  = (int)get_group_id(2);
  if (b >= B || h >= H || tq >= T) return;

  const long bh_off = ((long)b * (long)H + (long)h) * (long)T * (long)D;
  const long probs_base = (((long)b * (long)H + (long)h) * (long)T + (long)tq) * (long)T;

  float acc = 0.0f;
  for (int tk = 0; tk < T; ++tk) {
    const float v_elt = (float)V[bh_off + (long)tk * (long)D + lid];
    const float p = (float)probs[probs_base + tk];
    acc += p * v_elt;
  }

  if (lid < D) {
    out[bh_off + (long)tq * (long)D + lid] = (half)acc;
  }
}
#endif

// Parallel scores: one WG per (b, h, tq) row, 64 threads cooperatively produce
// all T scores in that row by striding over tk. Q[tq, :] is loaded into __local
// once per WG, K is read globally by lanes.
#ifdef USE_FP16
#define VIS_SC_WG 64
__kernel
__attribute__((reqd_work_group_size(VIS_SC_WG, 1, 1)))
void mha_scores_par(
    __global const half* Q,
    __global const half* K,
    __global half* scores,
    const int B,
    const int H,
    const int T,
    const int D,
    const float scale) {
  const int lid = (int)get_local_id(0);
  const int b   = (int)get_group_id(0);
  const int h   = (int)get_group_id(1);
  const int tq  = (int)get_group_id(2);
  if (b >= B || h >= H || tq >= T) return;

  const long bh_off = ((long)b * (long)H + (long)h) * (long)T * (long)D;

  // Load Q[tq, :D] once per WG into __local. D=64 expected.
  __local half Q_local[64];
  if (lid < D) {
    Q_local[lid] = Q[bh_off + (long)tq * (long)D + lid];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const long s_base = (((long)b * (long)H + (long)h) * (long)T + (long)tq) * (long)T;
  const int D4 = D >> 2;
  for (int tk = lid; tk < T; tk += VIS_SC_WG) {
    float acc = 0.0f;
    for (int d4 = 0; d4 < D4; ++d4) {
      float4 qv = convert_float4(vload_half4(0, (__local half*)Q_local + d4 * 4));
      float4 kv = convert_float4(vload_half4(0, (__global half*)K + bh_off + (long)tk * (long)D + d4 * 4));
      acc += dot(qv, kv);
    }
    scores[s_base + tk] = (half)(acc * scale);
  }
}
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

// Parallel softmax: one WG per (b, h, tq) row, 64 threads cooperatively reduce
// max + sum across the T-wide row. Cuts the per-row serialization of the naive
// kernel (3 sequential passes of length T per WI). Subgroup reduce gives us
// max/sum in O(log64) instead of O(T).
#ifdef USE_FP16
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define VIS_SM_WG 64
__kernel
__attribute__((reqd_work_group_size(VIS_SM_WG, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("full")))
void mha_softmax_parallel(
    __global half* scores,
    const int B,
    const int H,
    const int T) {
  const int lid = (int)get_local_id(0);
  const int b   = (int)get_group_id(0);
  const int h   = (int)get_group_id(1);
  const int tq  = (int)get_group_id(2);
  if (b >= B || h >= H || tq >= T) return;

  const long base = (((long)b * (long)H + (long)h) * (long)T + (long)tq) * (long)T;

  // Pass 1: find row max via subgroup reduce.
  float local_max = -3.402823466e+38f;
  for (int tk = lid; tk < T; tk += VIS_SM_WG) {
    float v = (float)scores[base + tk];
    if (v > local_max) local_max = v;
  }
  const float row_max = sub_group_reduce_max(local_max);

  // Pass 2: exp(v - max) and accumulate sum.
  float local_sum = 0.0f;
  for (int tk = lid; tk < T; tk += VIS_SM_WG) {
    float e = native_exp((float)scores[base + tk] - row_max);
    scores[base + tk] = (half)e;
    local_sum += e;
  }
  const float row_sum = sub_group_reduce_add(local_sum);
  const float inv = 1.0f / (row_sum + 1e-20f);

  // Pass 3: normalize.
  for (int tk = lid; tk < T; tk += VIS_SM_WG) {
    scores[base + tk] = (half)((float)scores[base + tk] * inv);
  }
}
#endif

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

// Parallel mha_out: one WG per (b, h, tq), 64 lanes stride over tk, each lane
// accumulates one float4 partial of out[d4_slot], then lanes hand off via
// __local memory so every lane writes its assigned d4 outputs.
// Layout: lane lid is the only writer for d4 in {lid % D4} aggregated across
// lanes via tree reduce of the per-lane partials.
//
// Simpler: each WG has 64 lanes; we structure it so lane lid computes one d4
// slot of the output (4 fp16 lanes), reducing over tk in stride 64.
// Requires WG_SIZE >= D4 (vision: D=64 → D4=16, plenty).
#ifdef USE_FP16
#define VIS_OUT_WG 64
__kernel
__attribute__((reqd_work_group_size(VIS_OUT_WG, 1, 1)))
void mha_out_par(
    __global const half* probs,
    __global const half* V,
    __global half* out,
    const int B,
    const int H,
    const int T,
    const int D) {
  const int lid = (int)get_local_id(0);
  const int b   = (int)get_group_id(0);
  const int h   = (int)get_group_id(1);
  const int tq  = (int)get_group_id(2);
  if (b >= B || h >= H || tq >= T) return;

  const int D4 = D >> 2;

  const long probs_base = (((long)b * (long)H + (long)h) * (long)T + (long)tq) * (long)T;
  const long v_base = ((long)b * (long)H + (long)h) * (long)T * (long)D;

  // Cooperatively load probs[tq, :T] into __local (T up to 2048 ≈ 4 KB fp16).
  __local half probs_local[2048];
  for (int t = lid; t < T; t += VIS_OUT_WG) {
    probs_local[t] = probs[probs_base + t];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Each of the first D4 lanes computes one output (b, h, tq, d4*4..+3) by
  // striding tk through __local probs and __global V.
  if (lid < D4) {
    const int d4 = lid;
    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (int tk = 0; tk < T; ++tk) {
      float p = (float)probs_local[tk];
      float4 vv = convert_float4(vload_half4(0, (__global half*)V + v_base + (long)tk * (long)D + (long)d4 * 4));
      acc += p * vv;
    }
    const long out_idx = (((long)b * (long)H + (long)h) * (long)T + (long)tq) * (long)D + (long)d4 * 4;
    out[out_idx + 0] = (half)acc.s0;
    out[out_idx + 1] = (half)acc.s1;
    out[out_idx + 2] = (half)acc.s2;
    out[out_idx + 3] = (half)acc.s3;
  }
}
#endif

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

// ── Flash Attention (Round 8) ─────────────────────────────────────────────
// Fused scores+softmax+mha_out for non-causal MHA. Never materializes the
// [B,H,T,T] scores matrix; instead iterates over K/V in blocks, maintaining
// online softmax statistics (running max + sum) per query row.
//
// One WG per (b, h, tq) row. 64 lanes cooperate via a coalesced cooperative
// load of FA_BC rows of K and V into __local each block, then 64 lanes also
// participate in the P·V accumulation (4 d4 slots each via tk-stride 16),
// finally reducing across lanes via __local tree to produce the 16 outputs.
//
// All running state (m_run, s_run, acc) is fp32 for numerical stability.
// Q/K/V/out remain fp16 in global memory.
//
// Local memory: 128 (Q) + 8192 (K_tile) + 8192 (V_tile) + 256 (scores) +
//               4096 (pv_partials) ≈ 21 KB. Adreno 620 has 32 KB local.
#ifdef USE_FP16
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define FA_WG 64
#define FA_BC 64

__kernel
__attribute__((reqd_work_group_size(FA_WG, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("full")))
void mha_flash_attn(
    __global const half* Q,
    __global const half* K,
    __global const half* V,
    __global half* out,
    const int B,
    const int H,
    const int T,
    const int D,
    const float scale) {
  const int lid = (int)get_local_id(0);
  const int b   = (int)get_group_id(0);
  const int h   = (int)get_group_id(1);
  const int tq  = (int)get_group_id(2);
  if (b >= B || h >= H || tq >= T) return;

  const int D4 = D >> 2;
  const long bh_off = ((long)b * (long)H + (long)h) * (long)T * (long)D;
  __global const half* Qp = Q + bh_off;
  __global const half* Kp = K + bh_off;
  __global const half* Vp = V + bh_off;

  // __local layout:
  //   Q_local[D]                  : query row (64 fp16 = 128 B)
  //   K_tile[FA_BC * D]           : key tile (64×64 fp16 = 8 KB)
  //   V_tile[FA_BC * D]           : value tile (64×64 fp16 = 8 KB)
  //   scores_tile[FA_BC]          : per-row scores (64 fp32 = 256 B)
  __local half  Q_local[64];                  // D=64
  __local half  K_tile[FA_BC * 64];           // 8 KB
  __local half  V_tile[FA_BC * 64];           // 8 KB
  __local float scores_tile[FA_BC];
  __local float4 pv_partials[FA_WG];          // 1 KB

  if (lid < D) Q_local[lid] = Qp[(long)tq * (long)D + lid];

  float m_run = -3.402823466e+38f;
  float s_run = 0.0f;
  // Per-lane partial output: lane lid handles d4 = (lid % D4).
  // Accumulate across tk-stride D4 = 16 within the block.
  float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int tk0 = 0; tk0 < T; tk0 += FA_BC) {
    // Cooperative load of K_tile[FA_BC, D] and V_tile[FA_BC, D] from global.
    // Coalesced pattern: lane lid loads K[tk0+r, lid] for r=0..FA_BC-1.
    // 64 lanes hit consecutive elements within ONE K row each iteration ↦
    // 128 B = 2 cachelines per iter, 64 iters per block ↦ 8 KB / WG.
    for (int r = 0; r < FA_BC; ++r) {
      const int tk = tk0 + r;
      // lid is 0..D-1 (D=64=FA_WG); reads K[tk, lid] and V[tk, lid].
      if (tk < T && lid < D) {
        K_tile[r * D + lid] = Kp[(long)tk * (long)D + lid];
        V_tile[r * D + lid] = Vp[(long)tk * (long)D + lid];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Score computation: lane lid (lid < FA_BC) computes its score.
    // For FA_BC == FA_WG, every lane participates.
    float my_score = -3.402823466e+38f;
    if (lid < FA_BC) {
      const int tk = tk0 + lid;
      if (tk < T) {
        float s = 0.0f;
        for (int d4 = 0; d4 < D4; ++d4) {
          float4 qv = convert_float4(vload_half4(0, (__local half*)Q_local + d4 * 4));
          float4 kv = convert_float4(vload_half4(0, (__local half*)(K_tile + lid * D) + d4 * 4));
          s += dot(qv, kv);
        }
        my_score = s * scale;
      }
    }
    const float block_max = sub_group_reduce_max(my_score);
    const float m_new = fmax(m_run, block_max);
    const float alpha = native_exp(m_run - m_new);

    float my_p = 0.0f;
    if (lid < FA_BC) {
      my_p = native_exp(my_score - m_new);
      scores_tile[lid] = my_p;
    }
    const float block_sum = sub_group_reduce_add(my_p);
    barrier(CLK_LOCAL_MEM_FENCE);

    // P·V accumulation — all 64 lanes participate.
    // Lane lid handles d4 slot = (lid % D4), with stride D4 over the FA_BC
    // dimension. Each lane visits FA_BC/D4 = 4 (j, d4) combinations.
    // Partial sums per (j_offset, d4) ↦ reduce later across the 4 lanes
    // covering the same d4 slot.
    const int d4 = lid % D4;            // 0..15
    const int j0 = lid / D4;            // 0..3
    float4 lane_pv = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j = j0; j < FA_BC; j += (FA_WG / D4)) {
      const int tk = tk0 + j;
      if (tk >= T) break;
      float p = scores_tile[j];
      float4 vv = convert_float4(vload_half4(0, (__local half*)(V_tile + j * D) + d4 * 4));
      lane_pv += p * vv;
    }
    // Reduce 4 lanes that share the same d4 slot. Stash partials in __local
    // scratch, then have lanes 0..D4-1 read and sum.
    pv_partials[lid] = lane_pv;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < D4) {
      // Lane lid reads partials at strides D4: indices lid, lid+D4, lid+2*D4, lid+3*D4.
      float4 sum_pv = pv_partials[lid] + pv_partials[lid + D4] +
                      pv_partials[lid + 2 * D4] + pv_partials[lid + 3 * D4];
      acc = alpha * acc + sum_pv;
    }

    s_run = alpha * s_run + block_sum;
    m_run = m_new;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Final write: lanes 0..D4-1 write their normalized acc to out[tq, d4*4..+3].
  if (lid < D4) {
    const float inv_s = 1.0f / s_run;
    const float4 result = acc * inv_s;
    const long out_off = bh_off + (long)tq * (long)D + (long)lid * 4;
    out[out_off + 0] = (half)result.s0;
    out[out_off + 1] = (half)result.s1;
    out[out_off + 2] = (half)result.s2;
    out[out_off + 3] = (half)result.s3;
  }
}
#endif

// ─── IMAGE2D K/V variants attempted (in git history) ─────────────────────
// Tried mha_scores_par_image + mha_out_image reading K/V via image2d_t L1
// texture cache. Per-op profile showed clear wins (vis_scores −21%,
// vis_mha_out −37%) but wall-clock TTFT was flat: the vision pipeline is
// CPU-dispatch-bound at ~360 dispatches/inference, so making GPU ops faster
// doesn't move the wall while the CPU keeps queuing. Recordable queues
// (cl_qcom_recordable_queues) would attack this directly — deferred.
