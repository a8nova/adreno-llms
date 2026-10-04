// Decode-step (M=1) helper kernels for GPUs without a usable texture path
// (PowerVR Rogue). Shared verbatim by smollm2-135m-instruct and
// smolvlm-256m-instruct (same text decoder: D=64, GQA 9 q / 3 kv heads).
//
// On the GE8320 a minimal kernel costs ~0.11 ms end to end, so a decode step
// pays for every launch; these kernels exist to remove launches:
//   attn_rope_gqa: RoPE(q,k) + KV-cache append + attention, one launch/layer
//                  (was rope_kvwrite + decode_attn), and it reads each K/V
//                  cache row once per KV head instead of once per Q head.
//   embed_row:     token-embedding gather with one lane per element (the
//                  ports' embedding kernels copy a whole row on one lane).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define DH 64          // head_dim; one lane per dimension
#define MAXGRP 4       // q heads per kv head supported (SmolLM2/SmolVLM: 3)

// out[t*H + i] = wte[ids[t]*H + i]
__kernel void embed_row(__global const int* ids, __global const half* wte,
                        __global half* out, const int H, const int n) {
  const int g = (int)get_global_id(0);
  if (g >= n * H) return;
  const int t = g / H, i = g - t * H;
  out[g] = wte[(size_t)ids[t] * H + i];
}

// out = half(x * rsqrt(mean(x^2) + eps) * gamma), one work-group of 64 lanes.
// Same float4-per-lane + 64-wide tree order as the ports' rmsnorm kernels;
// `precise` selects rsqrt (SmolVLM's kernel) or native_rsqrt (SmolLM2's).
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void rms_row(__global const half* x, __global const half* gamma, __global half* out,
             const int K, const float eps, const int precise) {
  const int lid = (int)get_local_id(0);
  __local float red[64];
  float ss = 0.0f;
  for (int c = lid; c < (K >> 2); c += 64) {
    const float4 v = vload_half4(c, x);
    ss += dot(v, v);
  }
  red[lid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int w = 32; w > 0; w >>= 1) {
    if (lid < w) red[lid] += red[lid + w];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float m = red[0] / (float)K + eps;
  const float inv = precise ? rsqrt(m) : native_rsqrt(m);
  for (int c = lid; c < (K >> 2); c += 64)
    vstore_half4(vload_half4(c, x) * inv * vload_half4(c, gamma), c, out);
}

// Shared attention core over cache positions [lo, hi) for the GRP query heads
// in ql (rotated, fp32, __local). Online softmax in 64-position chunks with
// three barriers per chunk (was ~14 with tree reductions): lane t writes the
// scores of position lo+t to __local, every lane scans them for the chunk max,
// lane t turns its own score into p_t, and every lane sums the p's itself. Lane d owns output dimension d.
// Returns per-head running max m, sum l and unnormalised acc (dimension d).
inline void attend_range(__local float (*ql)[DH], __local float (*sc)[DH],
                         __global const half* k_cache, __global const half* v_cache,
                         const int KV_DIM, const int kvh, const int GRP,
                         const int lo, const int hi, const float scale, const int d,
                         float* m, float* l, float* acc) {
  for (int g = 0; g < MAXGRP; ++g) { m[g] = -1e30f; l[g] = 0.0f; acc[g] = 0.0f; }
  for (int base = lo; base < hi; base += DH) {
    const int t = base + d;
    const int n = min(DH, hi - base);
    if (t < hi) {
      __global const uint4* kr = (__global const uint4*)(k_cache + (size_t)t * KV_DIM + kvh * DH);
      float dt[MAXGRP] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int cc = 0; cc < DH / 8; ++cc) {
        const float8 kv = convert_float8(as_half8(kr[cc]));
        for (int g = 0; g < GRP; ++g) {
          const float8 qv = vload8(cc, &ql[g][0]);
          dt[g] += dot(qv.lo, kv.lo) + dot(qv.hi, kv.hi);
        }
      }
      for (int g = 0; g < GRP; ++g) sc[g][d] = dt[g] * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // chunk max: every lane scans the 64 scores (no tree, no barriers), then
    // lane t turns ITS score into p_t in place -- one exp per lane per head.
    float mn[MAXGRP];
    for (int g = 0; g < GRP; ++g) {
      float cm = sc[g][0];
      for (int j = 1; j < n; ++j) cm = fmax(cm, sc[g][j]);
      mn[g] = fmax(m[g], cm);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (d < n)
      for (int g = 0; g < GRP; ++g) sc[g][d] = native_exp(sc[g][d] - mn[g]);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int g = 0; g < GRP; ++g) {
      const float corr = native_exp(m[g] - mn[g]);   // 0 on the first chunk
      l[g] *= corr;
      acc[g] *= corr;
      m[g] = mn[g];
    }
    __global const half* vp = v_cache + (size_t)base * KV_DIM + kvh * DH + d;
    for (int j = 0; j < n; ++j) {
      const float vv = vload_half(0, vp + (size_t)j * KV_DIM);
      for (int g = 0; g < GRP; ++g) {
        const float p = sc[g][j];
        l[g] += p;
        acc[g] += p * vv;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);   // sc is rewritten by the next chunk
  }
}

// RoPE on the GRP query heads of KV head kvh into ql, and (if write_row) RoPE
// on k + append of k, v at row start_pos.
inline void rope_and_append(__global const half* q, __global const half* k_in,
                            __global const half* v_in, __global half* k_cache,
                            __global half* v_cache, __global const half* cos_t,
                            __global const half* sin_t, __local float (*ql)[DH],
                            const int GRP, const int KV_DIM, const int kvh,
                            const int start_pos, const int d, const int write_row) {
  const int half_d = DH / 2;
  const int pd = d < half_d ? d + half_d : d - half_d;   // rotate-half partner
  const int ci = start_pos * DH + (d < half_d ? d : d - half_d);
  const float c = vload_half(ci, cos_t), s = vload_half(ci, sin_t);
  const float sgn = d < half_d ? -1.0f : 1.0f;            // x*c + sgn*partner*s
  for (int g = 0; g < GRP; ++g) {
    const int qb = (kvh * GRP + g) * DH;
    const float x = vload_half(qb + d, q), xp = vload_half(qb + pd, q);
    ql[g][d] = x * c + sgn * xp * s;
  }
  if (write_row) {
    const int kb = kvh * DH;
    const float x = vload_half(kb + d, k_in), xp = vload_half(kb + pd, k_in);
    const size_t row = (size_t)start_pos * KV_DIM + kb + d;
    vstore_half(x * c + sgn * xp * s, 0, k_cache + row);
    v_cache[row] = v_in[kb + d];
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

// One work-group (DH lanes) per KV head; it serves the GRP query heads that
// share that KV head (GQA), so each K/V cache row is read once per KV head.
//   q:      [QH*DH]   un-rotated query of the new token
//   k_in/v_in: [KVH*DH] un-rotated key / value of the new token
//   k_cache/v_cache: [MAX_SEQ, KVH*DH]; row start_pos is written here
//   cos/sin: [MAX_POS, DH] HF rotate-half tables (first DH/2 entries used)
//   out:    [QH*DH]
// __local use is fixed (2 x MAXGRP x DH floats = 2 KB), so context length is
// limited only by the cache, not by the 4 KB of local memory on the GE8320.
__kernel __attribute__((reqd_work_group_size(DH, 1, 1)))
void attn_rope_gqa(__global const half* q, __global const half* k_in,
                   __global const half* v_in,
                   __global half* k_cache, __global half* v_cache,
                   __global const half* cos_t, __global const half* sin_t,
                   __global half* out,
                   const int GRP, const int KVH, const int start_pos,
                   const float scale) {
  const int kvh = (int)get_group_id(0);
  const int d   = (int)get_local_id(0);
  const int KV_DIM = KVH * DH;
  __local float ql[MAXGRP][DH];
  __local float sc[MAXGRP][DH];
  rope_and_append(q, k_in, v_in, k_cache, v_cache, cos_t, sin_t, ql, GRP, KV_DIM, kvh,
                  start_pos, d, 1);
  float m[MAXGRP], l[MAXGRP], acc[MAXGRP];
  attend_range(ql, sc, k_cache, v_cache, KV_DIM, kvh, GRP, 0, start_pos + 1, scale, d, m, l, acc);
  for (int g = 0; g < GRP; ++g)
    vstore_half(acc[g] / l[g], (kvh * GRP + g) * DH + d, out);
}

// ── Split-context variant (flash-decoding) ──────────────────────────────────
// Each KV head's positions are split into S ranges, one work-group each
// (KVH*S groups); every group writes its partial (max, sum, unnormalised acc)
// per q head and attn_merge (KVH groups) folds them. The group owning
// position start_pos appends the new K/V row.
//   part: [KVH*S*GRP] m, [KVH*S*GRP] l, [KVH*S*GRP*DH] acc   (floats)
__kernel __attribute__((reqd_work_group_size(DH, 1, 1)))
void attn_split(__global const half* q, __global const half* k_in,
                __global const half* v_in,
                __global half* k_cache, __global half* v_cache,
                __global const half* cos_t, __global const half* sin_t,
                __global float* part,
                const int GRP, const int KVH, const int start_pos,
                const float scale, const int S) {
  const int kvh = (int)get_group_id(0) / S;
  const int sp  = (int)get_group_id(0) - kvh * S;
  const int d   = (int)get_local_id(0);
  const int KV_DIM = KVH * DH;
  const int seq_k = start_pos + 1;
  const int span = (seq_k + S - 1) / S;
  const int lo = sp * span, hi = min(lo + span, seq_k);
  __local float ql[MAXGRP][DH];
  __local float sc[MAXGRP][DH];
  rope_and_append(q, k_in, v_in, k_cache, v_cache, cos_t, sin_t, ql, GRP, KV_DIM, kvh,
                  start_pos, d, start_pos >= lo && start_pos < hi);
  float m[MAXGRP], l[MAXGRP], acc[MAXGRP];
  attend_range(ql, sc, k_cache, v_cache, KV_DIM, kvh, GRP, lo, hi, scale, d, m, l, acc);
  const int NP = KVH * S * GRP;
  for (int g = 0; g < GRP; ++g) {
    const int idx = (kvh * S + sp) * GRP + g;
    if (d == 0) { part[idx] = m[g]; part[NP + idx] = l[g]; }
    part[2 * NP + idx * DH + d] = acc[g];
  }
}

__kernel __attribute__((reqd_work_group_size(DH, 1, 1)))
void attn_merge(__global const float* part, __global half* out,
                const int GRP, const int KVH, const int S) {
  const int kvh = (int)get_group_id(0);
  const int d = (int)get_local_id(0);
  const int NP = KVH * S * GRP;
  for (int g = 0; g < GRP; ++g) {
    float M = -1e30f;
    for (int sp = 0; sp < S; ++sp) M = fmax(M, part[(kvh * S + sp) * GRP + g]);
    float L = 0.0f, A = 0.0f;
    for (int sp = 0; sp < S; ++sp) {
      const int idx = (kvh * S + sp) * GRP + g;
      const float w = native_exp(part[idx] - M);
      L += part[NP + idx] * w;
      A += part[2 * NP + idx * DH + d] * w;
    }
    vstore_half(A / L, (kvh * GRP + g) * DH + d, out);
  }
}

// Previous single-pass kernel (tree reductions, ~14 barriers per chunk), kept
// for on-device A/B against attn_rope_gqa (NNOPT_ATTN_TREE=1 selects it).
__kernel __attribute__((reqd_work_group_size(DH, 1, 1)))
void attn_rope_gqa_tree(__global const half* q, __global const half* k_in,
                        __global const half* v_in,
                        __global half* k_cache, __global half* v_cache,
                        __global const half* cos_t, __global const half* sin_t,
                        __global half* out,
                        const int GRP, const int KVH, const int start_pos,
                        const float scale) {
  const int kvh = (int)get_group_id(0);
  const int d   = (int)get_local_id(0);
  const int KV_DIM = KVH * DH;
  __local float ql[MAXGRP][DH];
  __local float pr[MAXGRP][DH];
  __local float red[MAXGRP][DH];
  rope_and_append(q, k_in, v_in, k_cache, v_cache, cos_t, sin_t, ql, GRP, KV_DIM, kvh,
                  start_pos, d, 1);
  const int seq_k = start_pos + 1;
  float m[MAXGRP], l[MAXGRP], acc[MAXGRP];
  for (int g = 0; g < MAXGRP; ++g) { m[g] = -1e30f; l[g] = 0.0f; acc[g] = 0.0f; }
  for (int base = 0; base < seq_k; base += DH) {
    const int t = base + d;
    float sc[MAXGRP];
    for (int g = 0; g < MAXGRP; ++g) sc[g] = -1e30f;
    if (t < seq_k) {
      __global const half* kr = k_cache + (size_t)t * KV_DIM + kvh * DH;
      float dot_[MAXGRP] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int cc = 0; cc < DH / 8; ++cc) {
        const float8 kv = vload_half8(cc, kr);
        for (int g = 0; g < GRP; ++g) {
          const float8 qv = vload8(cc, &ql[g][0]);
          dot_[g] += dot(qv.lo, kv.lo) + dot(qv.hi, kv.hi);
        }
      }
      for (int g = 0; g < GRP; ++g) sc[g] = dot_[g] * scale;
    }
    for (int g = 0; g < GRP; ++g) red[g][d] = sc[g];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int w = DH >> 1; w > 0; w >>= 1) {
      if (d < w)
        for (int g = 0; g < GRP; ++g) red[g][d] = fmax(red[g][d], red[g][d + w]);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mn[MAXGRP];
    for (int g = 0; g < GRP; ++g) mn[g] = fmax(m[g], red[g][0]);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int g = 0; g < GRP; ++g) {
      const float p = (t < seq_k) ? native_exp(sc[g] - mn[g]) : 0.0f;
      pr[g][d] = p;
      red[g][d] = p;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int w = DH >> 1; w > 0; w >>= 1) {
      if (d < w)
        for (int g = 0; g < GRP; ++g) red[g][d] += red[g][d + w];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const int n = min(DH, seq_k - base);
    __global const half* vp = v_cache + (size_t)base * KV_DIM + kvh * DH + d;
    for (int g = 0; g < GRP; ++g) {
      const float corr = native_exp(m[g] - mn[g]);
      l[g] = l[g] * corr + red[g][0];
      acc[g] *= corr;
      m[g] = mn[g];
    }
    for (int j = 0; j < n; ++j) {
      const float vv = vload_half(0, vp + (size_t)j * KV_DIM);
      for (int g = 0; g < GRP; ++g) acc[g] += pr[g][j] * vv;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int g = 0; g < GRP; ++g)
    vstore_half(acc[g] / l[g], (kvh * GRP + g) * DH + d, out);
}
