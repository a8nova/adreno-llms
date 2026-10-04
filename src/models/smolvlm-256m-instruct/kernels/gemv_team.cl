// fp16 M=1 GEMV family for GPUs without a usable texture path (PowerVR Rogue).
//
// The block_fused.cl buffer kernels give each output row a whole 64-lane
// work-group, read 2-8 bytes per lane and finish with a 6-level barrier tree.
// That is the Adreno pattern. On a 1-CU Rogue (GE8320) it streams ~0.3 GB/s:
// the NNOPT_STREAM_PAT probe measured 1.45 GB/s at 4-byte lane loads and
// 3.41 GB/s at 16-byte loads, and showed that the number of lanes in flight,
// not coalescing, is what gates streaming on this part.
//
// Here a TEAM of LPR lanes shares a row: lane `sub` reads 16-byte chunks
// (8 halves) sub, sub+LPR, ... so the team reads LPR*16 contiguous bytes per
// step, and each team owns RPL rows (RPL independent FMA chains that reuse
// one activation load). The team then folds its partial sums with a
// log2(LPR)-level tree in __local (RPL*WG*4 bytes; 1 KB at RPL=4, WG=64).
// LPR=1 is row-per-lane: no __local, no barriers.
//
// Geometry is a build option: -DWG=.. -DLPR=.. -DRPL=.. -DLOADMODE=.. (WG % LPR
// == 0). K must be a multiple of 8 and rows 16-byte aligned. Accumulation is fp32.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef WG
#define WG 64
#endif
#ifndef LPR
#define LPR 8
#endif
#ifndef RPL
#define RPL 1
#endif
#define TEAMS (WG / LPR)

// How a 16-byte chunk of 8 halves is loaded and multiplied:
//   0: vload_half8 -> float8 (portable; the Rogue compiler may split it into
//      eight 2-byte loads, the slowest pattern on that part)
//   1: one native uint4 (16-byte) load, reinterpreted as half8, widened to fp32
//   2: as 1, but the 8 products are formed in fp16 (Rogue fp16 ALU measured
//      4.8x its fp32 rate) and only their sum is widened
#ifndef LOADMODE
#define LOADMODE 1
#endif
#if LOADMODE == 0
#define LD8F(c, p) vload_half8((c), (p))
#else
#define LD8H(c, p) as_half8(((__global const uint4*)(p))[c])
#define LD8F(c, p) convert_float8(LD8H(c, p))
#endif
// FUSE_NORM: x is the raw residual stream and the kernel applies the decoder
// layer's RMSNorm itself (gamma, eps extra args), replacing a separate 1-WG
// rmsnorm launch per norm. Every work-group recomputes sum(x^2) over K (a 1 KB
// read), and x is formed exactly as rmsnorm.cl stores it -- half(x*inv*gamma)
// -- so the GEMV sees the same fp16 activations.
// Match the port's own rmsnorm kernel: SmolLM2 uses native_rsqrt, SmolVLM rsqrt.
#ifdef NORM_PRECISE_RSQRT
#define NORM_RSQRT rsqrt
#else
#define NORM_RSQRT native_rsqrt
#endif
#ifdef FUSE_NORM
#define NORM_ARGS , __global const half* gamma, const float eps
#define NORM_PROLOGUE                                                         \
  __local float nred[WG];                                                     \
  {                                                                           \
    float ss = 0.0f;                                                          \
    /* float4 per lane, stride WG, then a WG tree: the same summation */     \
    /* order as the ports' rmsnorm kernels (at WG=64), so inv_rms matches */  \
    for (int c = lid; c < (K >> 2); c += WG) {                                \
      const float4 v = vload_half4(c, x);                                     \
      ss += dot(v, v);                                                        \
    }                                                                         \
    nred[lid] = ss;                                                           \
    barrier(CLK_LOCAL_MEM_FENCE);                                             \
    for (int s = WG >> 1; s > 0; s >>= 1) {                                   \
      if (lid < s) nred[lid] += nred[lid + s];                                \
      barrier(CLK_LOCAL_MEM_FENCE);                                           \
    }                                                                         \
  }                                                                           \
  const float inv_rms = NORM_RSQRT(nred[0] / (float)K + eps);
#define NORMED_X(c) convert_half8(LD8F(c, x) * inv_rms * LD8F(c, gamma))
#else
#define NORM_ARGS
#define NORM_PROLOGUE
#endif

// XLOCAL=n (n >= K): stage the activation vector -- already normalised when
// FUSE_NORM -- in __local once per work-group (n halves), instead of every
// lane re-reading it from global memory for every 16-byte weight chunk (at
// RPL=2 that is 50% extra load traffic, and the RMSNorm math redone per chunk).
// Bit-identical to the unstaged kernel: the same half8 values are formed.
#ifdef XLOCAL
#ifdef FUSE_NORM
#define XSRC(c) NORMED_X(c)
#else
#define XSRC(c) vload8((c), x)
#endif
#define XSTAGE                                                                \
  __local half xl[XLOCAL];                                                    \
  for (int c = lid; c < (K >> 3); c += WG) vstore8(XSRC(c), c, xl);           \
  barrier(CLK_LOCAL_MEM_FENCE);
#else
#define XSTAGE
#endif

#if defined(XLOCAL) && LOADMODE == 2
#define XCHUNK(c) const half8 xv = vload8((c), xl);
#elif defined(XLOCAL)
#define XCHUNK(c) const float8 xv = convert_float8(vload8((c), xl));
#endif

#if LOADMODE == 2
// x chunk kept as half8; returns the fp32 sum of the 8 fp16 products.
#if defined(XLOCAL)
#elif defined(FUSE_NORM)
#define XCHUNK(c) const half8 xv = NORMED_X(c);
#else
#define XCHUNK(c) const half8 xv = LD8H(c, x);
#endif
inline float dot8h(half8 a, half8 b) {
  const half8 p = a * b;
  return convert_float((p.s0 + p.s1) + (p.s2 + p.s3)) + convert_float((p.s4 + p.s5) + (p.s6 + p.s7));
}
#define DOT8(c, row) dot8h(xv, LD8H(c, row))
#else
#if defined(XLOCAL)
#elif defined(FUSE_NORM)
#define XCHUNK(c) const float8 xv = convert_float8(NORMED_X(c));
#else
#define XCHUNK(c) const float8 xv = LD8F(c, x);
#endif
#define DOT8(c, row) dot8f(xv, LD8F(c, row))
#endif
inline float dot8f(float8 a, float8 b) { return dot(a.lo, b.lo) + dot(a.hi, b.hi); }

inline float silu_t(float v) { return v / (1.0f + native_exp(-v)); }

// acc[r] += dot(row r of the team's rows, x) over this lane's K-chunks.
// rows[r] points at the start of row r (already clamped to a valid row).
// UNR=2: two chunks per iteration, so each lane has 2*RPL 16-byte weight
// loads in flight instead of RPL (same products, same summation order).
#ifndef UNR
#define UNR 1
#endif
#if UNR == 2
#define TEAM_DOT(rows, acc)                                                   \
  {                                                                           \
    int c = sub;                                                              \
    for (; c + LPR < (K >> 3); c += 2 * LPR) {                                \
      float t0[RPL], t1[RPL];                                                 \
      {                                                                       \
        XCHUNK(c)                                                             \
        _Pragma("unroll")                                                     \
        for (int r = 0; r < RPL; ++r) t0[r] = DOT8(c, rows[r]);               \
      }                                                                       \
      {                                                                       \
        XCHUNK(c + LPR)                                                       \
        _Pragma("unroll")                                                     \
        for (int r = 0; r < RPL; ++r) t1[r] = DOT8(c + LPR, rows[r]);         \
      }                                                                       \
      _Pragma("unroll")                                                       \
      for (int r = 0; r < RPL; ++r) { acc[r] += t0[r]; acc[r] += t1[r]; }     \
    }                                                                         \
    if (c < (K >> 3)) {                                                       \
      XCHUNK(c)                                                               \
      _Pragma("unroll")                                                       \
      for (int r = 0; r < RPL; ++r) acc[r] += DOT8(c, rows[r]);               \
    }                                                                         \
  }
#else
#define TEAM_DOT(rows, acc)                                                   \
  for (int c = sub; c < (K >> 3); c += LPR) {                                 \
    XCHUNK(c)                                                                 \
    _Pragma("unroll")                                                         \
    for (int r = 0; r < RPL; ++r) acc[r] += DOT8(c, rows[r]);                 \
  }
#endif

// Fold acc[0..RPL) across the LPR lanes of each team. Every lane must call it
// (barriers); afterwards lane sub==0 holds the team's totals in acc[].
#if LPR > 1
#define TEAM_REDUCE(red, acc)                                                 \
  {                                                                           \
    _Pragma("unroll")                                                         \
    for (int r = 0; r < RPL; ++r) red[r * WG + lid] = acc[r];                 \
    barrier(CLK_LOCAL_MEM_FENCE);                                             \
    for (int s = LPR >> 1; s > 0; s >>= 1) {                                  \
      if (sub < s) {                                                          \
        _Pragma("unroll")                                                     \
        for (int r = 0; r < RPL; ++r)                                         \
          red[r * WG + lid] += red[r * WG + lid + s];                         \
      }                                                                       \
      barrier(CLK_LOCAL_MEM_FENCE);                                           \
    }                                                                         \
    _Pragma("unroll")                                                         \
    for (int r = 0; r < RPL; ++r) acc[r] = red[r * WG + lid];                 \
  }
#else
#define TEAM_REDUCE(red, acc)
#endif

#define TEAM_PROLOGUE                                                         \
  const int lid  = (int)get_local_id(0);                                      \
  const int sub  = lid % LPR;                                                 \
  const int team = lid / LPR;                                                 \
  const int n0   = ((int)get_group_id(0) * TEAMS + team) * RPL;

// out[n] = W[n,:] . x
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void gemv_t(__global const half* x, __global const half* W,
            __global half* out, const int N, const int K NORM_ARGS) {
  TEAM_PROLOGUE
  NORM_PROLOGUE
  XSTAGE
#if LPR > 1
  __local float red[RPL * WG];
#endif
  __global const half* rows[RPL];
  float acc[RPL];
  _Pragma("unroll")
  for (int r = 0; r < RPL; ++r) {
    rows[r] = W + (size_t)min(n0 + r, N - 1) * K;
    acc[r] = 0.0f;
  }
  TEAM_DOT(rows, acc)
  TEAM_REDUCE(red, acc)
  if (sub == 0) {
    _Pragma("unroll")
    for (int r = 0; r < RPL; ++r)
      if (n0 + r < N) vstore_half(acc[r], n0 + r, out);
  }
}

// res[n] += W[n,:] . x      (o_proj / down_proj fused with the residual add)
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void gemv_res_t(__global const half* x, __global const half* W,
                __global half* res, const int N, const int K NORM_ARGS) {
  TEAM_PROLOGUE
  NORM_PROLOGUE
  XSTAGE
#if LPR > 1
  __local float red[RPL * WG];
#endif
  __global const half* rows[RPL];
  float acc[RPL];
  _Pragma("unroll")
  for (int r = 0; r < RPL; ++r) {
    rows[r] = W + (size_t)min(n0 + r, N - 1) * K;
    acc[r] = 0.0f;
  }
  TEAM_DOT(rows, acc)
  TEAM_REDUCE(red, acc)
  if (sub == 0) {
    _Pragma("unroll")
    for (int r = 0; r < RPL; ++r)
      if (n0 + r < N) vstore_half(vload_half(n0 + r, res) + acc[r], n0 + r, res);
  }
}

// out[n] = res[n] + W[n,:] . x   (out-of-place variant of gemv_res_t)
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void gemv_addres_t(__global const half* x, __global const half* W,
                   __global const half* res, __global half* out,
                   const int N, const int K NORM_ARGS) {
  TEAM_PROLOGUE
  NORM_PROLOGUE
  XSTAGE
#if LPR > 1
  __local float red[RPL * WG];
#endif
  __global const half* rows[RPL];
  float acc[RPL];
  _Pragma("unroll")
  for (int r = 0; r < RPL; ++r) {
    rows[r] = W + (size_t)min(n0 + r, N - 1) * K;
    acc[r] = 0.0f;
  }
  TEAM_DOT(rows, acc)
  TEAM_REDUCE(red, acc)
  if (sub == 0) {
    _Pragma("unroll")
    for (int r = 0; r < RPL; ++r)
      if (n0 + r < N) vstore_half(vload_half(n0 + r, res) + acc[r], n0 + r, out);
  }
}

// out[n] = silu(Wg[n,:] . x) * (Wu[n,:] . x)
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void gemv_swiglu_t(__global const half* x, __global const half* Wg,
                   __global const half* Wu, __global half* out,
                   const int N, const int K NORM_ARGS) {
  TEAM_PROLOGUE
  NORM_PROLOGUE
  XSTAGE
#if LPR > 1
  __local float red[RPL * WG];
#endif
  __global const half* rg[RPL];
  __global const half* ru[RPL];
  float ag[RPL], au[RPL];
  _Pragma("unroll")
  for (int r = 0; r < RPL; ++r) {
    const size_t off = (size_t)min(n0 + r, N - 1) * K;
    rg[r] = Wg + off; ru[r] = Wu + off;
    ag[r] = 0.0f; au[r] = 0.0f;
  }
#if UNR == 2
  int c = sub;
  for (; c + LPR < (K >> 3); c += 2 * LPR) {
    float g0[RPL], u0[RPL], g1[RPL], u1[RPL];
    {
      XCHUNK(c)
      _Pragma("unroll")
      for (int r = 0; r < RPL; ++r) { g0[r] = DOT8(c, rg[r]); u0[r] = DOT8(c, ru[r]); }
    }
    {
      XCHUNK(c + LPR)
      _Pragma("unroll")
      for (int r = 0; r < RPL; ++r) { g1[r] = DOT8(c + LPR, rg[r]); u1[r] = DOT8(c + LPR, ru[r]); }
    }
    _Pragma("unroll")
    for (int r = 0; r < RPL; ++r) { ag[r] += g0[r]; ag[r] += g1[r]; au[r] += u0[r]; au[r] += u1[r]; }
  }
  if (c < (K >> 3)) {
    XCHUNK(c)
    _Pragma("unroll")
    for (int r = 0; r < RPL; ++r) { ag[r] += DOT8(c, rg[r]); au[r] += DOT8(c, ru[r]); }
  }
#else
  for (int c = sub; c < (K >> 3); c += LPR) {
    XCHUNK(c)
    _Pragma("unroll")
    for (int r = 0; r < RPL; ++r) {
      ag[r] += DOT8(c, rg[r]);
      au[r] += DOT8(c, ru[r]);
    }
  }
#endif
  TEAM_REDUCE(red, ag)
  TEAM_REDUCE(red, au)
  if (sub == 0) {
    _Pragma("unroll")
    for (int r = 0; r < RPL; ++r)
      if (n0 + r < N) vstore_half(silu_t(ag[r]) * au[r], n0 + r, out);
  }
}

// q|k|v = [Wq;Wk;Wv] . x   — one launch over N = QN + 2*KVN rows.
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void gemv_qkv_t(__global const half* x,
                __global const half* Wq, __global const half* Wk,
                __global const half* Wv,
                __global half* q, __global half* k, __global half* v,
                const int QN, const int KVN, const int K NORM_ARGS) {
  TEAM_PROLOGUE
  NORM_PROLOGUE
  XSTAGE
  const int N = QN + 2 * KVN;
#if LPR > 1
  __local float red[RPL * WG];
#endif
  __global const half* rows[RPL];
  float acc[RPL];
  _Pragma("unroll")
  for (int r = 0; r < RPL; ++r) {
    const int n = min(n0 + r, N - 1);
    rows[r] = n < QN        ? Wq + (size_t)n * K
            : n < QN + KVN  ? Wk + (size_t)(n - QN) * K
                            : Wv + (size_t)(n - QN - KVN) * K;
    acc[r] = 0.0f;
  }
  TEAM_DOT(rows, acc)
  TEAM_REDUCE(red, acc)
  if (sub == 0) {
    _Pragma("unroll")
    for (int r = 0; r < RPL; ++r) {
      const int n = n0 + r;
      if (n < QN)                vstore_half(acc[r], n, q);
      else if (n < QN + KVN)     vstore_half(acc[r], n - QN, k);
      else if (n < N)            vstore_half(acc[r], n - QN - KVN, v);
    }
  }
}
