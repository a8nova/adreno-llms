// SAMPLE-AND-WRITE-GRID KERNEL (Fix-12 Stage 3, hybrid) — on-GPU CFG blend +
// temperature/top-k sampling that writes 4 sampled int32 ids straight into the
// GPU-resident delay-pattern grid. Keeps the fast CLBlast lm_heads GEMM (the
// fused-GEMV variant regressed ~12% — see BENCHMARKS), and only moves the
// blend+sample+grid-write on-GPU. This is what enables Stage 4's zero-per-step-
// sync decode loop (no 8192-fp16 logits readback, no host sample, no per-step
// id upload — the host only enqueues steps and reads the whole grid ONCE).
//
// Input logits2 = [2, num_codebooks*vocab] (row0=cond, row1=uncond), exactly the
// CLBlast lm_heads output. ONE workgroup per codebook (4 WGs, 256 threads).
//   blended[v] = uncond + g*(cond-uncond) for v ∈ [0,vocab).
//   force_argmax → argmax(blended) (greedy guard 66/1534/1513/1801 still applies).
//   else → temperature/top-k=250 softmax + inverse-CDF xorshift sample.
// AGENT_DIRECTIVE_FP32_ACCUM: blend/softmax in fp32 (logits load fp16).

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)     vload_half((i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)     ((p)[(i)])
#endif

#define SG_WG    256
#define SG_VOCAB 2048

// Counter-based uniform draw. The previous xorshift32 single-round over the
// (seed ^ step*PHI ^ cb*C2) lattice had NO avalanche: the 4 codebooks' draws
// within a step (and neighboring steps') landed at correlated CDF quantiles,
// audibly degrading sampled output (dark/mushy clips; reproduced 1:1 in a
// PyTorch monkeypatch sim, 2026-06-05). splitmix32 finalizer = full avalanche,
// independent r per (seed, step, cb). Greedy/force_argmax paths never read r,
// so the 66/1534/1513/1801 guard and tf gates are untouched by this change.
static inline float sg_uniform_at(uint seed, uint step, uint cb) {
  uint x = seed ^ (step * 0x9E3779B9u) ^ (cb * 0x85EBCA6Bu);
  x ^= x >> 16; x *= 0x7FEB352Du;
  x ^= x >> 15; x *= 0x846CA68Bu;
  x ^= x >> 16;
  return (x >> 8) * (1.0f / 16777216.0f);
}

static inline void sg_emit(__global int* out_ids, __global int* grid, int write_grid,
                           int cb, int step, int steps1, int bos, int id) {
  out_ids[cb] = id;
  if (write_grid && grid) {
    int col = step + 1;
    if (col < steps1) grid[cb * steps1 + col] = (col <= cb) ? bos : id;
  }
}

__kernel __attribute__((reqd_work_group_size(SG_WG, 1, 1)))
void sample_grid(
    __global const storage_t* logits2,   // [2, num_codebooks*vocab]
    __global int* out_ids,               // [num_codebooks]
    const float guidance,
    const float temperature,
    const int   top_k,
    const uint  seed,
    __global const int* sp,             // step-params: step = sp[0]. Buffered
                                        // (not a literal arg) so recordable-
                                        // queue replays need no arg override.
    const int   force_argmax,
    const int   vocab,
    __global int* grid,                  // [num_codebooks, steps1] (or null)
    const int   write_grid,
    const int   steps1,
    const int   bos,
    const int   single) {                // 1 = CFG-early single-row decode: row1
                                         // of logits2 is STALE; use cond row
                                         // directly (exact — replaces the old
                                         // row0→row1 logits dup CopyBuffer,
                                         // which the recordable-queue replay
                                         // path cannot record)

  const int cb  = get_group_id(0);
  const int lid = get_local_id(0);
  const int N   = get_num_groups(0) * vocab;   // num_codebooks*vocab (row stride)
  const int step = sp[0];

  __local float logit[SG_VOCAB];   // 8KB
  __local float red[SG_WG];
  __local int   ired[SG_WG];

  // blended logits for codebook cb: cond at logits2[cb*vocab+v], uncond at +N.
  const int base = cb * vocab;
  for (int v = lid; v < vocab; v += SG_WG) {
    float c = LOAD(logits2, base + v);
    if (single) { logit[v] = c; }
    else {
      float u = LOAD(logits2, N + base + v);
      logit[v] = u + guidance * (c - u);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (force_argmax) {
    float bestv = -1e30f; int besti = 0;
    for (int v = lid; v < vocab; v += SG_WG) if (logit[v] > bestv) { bestv = logit[v]; besti = v; }
    red[lid] = bestv; ired[lid] = besti; barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = SG_WG >> 1; o > 0; o >>= 1) {
      if (lid < o) { if (red[lid+o] > red[lid]) { red[lid] = red[lid+o]; ired[lid] = ired[lid+o]; } }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) sg_emit(out_ids, grid, write_grid, cb, step, steps1, bos, ired[0]);
    return;
  }

  if (lid == 0) {
    int keep = (top_k > 0 && top_k < vocab) ? top_k : vocab;
    float maxv = -1e30f, minv = 1e30f;
    for (int v = 0; v < vocab; ++v) { float l = logit[v]; if (l > maxv) maxv = l; if (l < minv) minv = l; }
    float invT = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;
    // bisection threshold so count(logit>=thr) ~ keep (lo side keeps >= keep).
    float lo = minv, hi = maxv;
    for (int it = 0; it < 48; ++it) {
      float mid = 0.5f * (lo + hi);
      int cnt = 0;
      for (int v = 0; v < vocab; ++v) if (logit[v] >= mid) ++cnt;
      if (cnt > keep) lo = mid; else hi = mid;
    }
    float thr = lo;
    float sum = 0.0f;
    for (int v = 0; v < vocab; ++v) if (logit[v] >= thr) sum += exp((logit[v] - maxv) * invT);
    float r = sg_uniform_at(seed, (uint)step, (uint)cb) * sum;
    float acc = 0.0f; int picked = -1;
    for (int v = 0; v < vocab; ++v) {
      if (logit[v] >= thr) { acc += exp((logit[v] - maxv) * invT); if (acc >= r) { picked = v; break; } }
    }
    if (picked < 0) { float bv = -1e30f; int bi = 0; for (int v = 0; v < vocab; ++v) if (logit[v] > bv) { bv = logit[v]; bi = v; } picked = bi; }
    sg_emit(out_ids, grid, write_grid, cb, step, steps1, bos, picked);
  }
}
