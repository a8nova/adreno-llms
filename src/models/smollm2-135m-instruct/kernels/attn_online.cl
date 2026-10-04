// Decode attention (M=1, GQA) with an online softmax and CONSTANT __local use.
//
// fused_decode_attn_m1 (block_fused.cl) keeps every score of the row in
// __local: (seq_k + 64) * 4 bytes. PowerVR GE8320 has 4 KB of local memory, so
// past ~960 positions of context that kernel cannot launch at all and decode
// fails. This kernel walks the cache in 64-position chunks, keeping a running
// max / sum per head (flash-decoding style), so it needs 512 bytes at any
// length. The host uses it only when the original's scratch would not fit.
//
// One work-group of 64 lanes per Q head; lane `tid` scores position
// base+tid of each chunk, then owns output dimension `tid` (requires D == 64).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WGA 64

__kernel __attribute__((reqd_work_group_size(WGA, 1, 1)))
void decode_attn_online(__global const half* q,        // [QH * D]
                        __global const half* k_cache,  // [MAX_SEQ, KV_DIM]
                        __global const half* v_cache,  // [MAX_SEQ, KV_DIM]
                        __global half* out,            // [QH * D]
                        const int KV_DIM, const int D, const int GRP,
                        const int seq_k, const float scale) {
  const int qh  = (int)get_group_id(0);
  const int tid = (int)get_local_id(0);
  const int kvh = qh / GRP;
  __local float p_chunk[WGA];
  __local float red[WGA];

  __global const half* qp = q + qh * D;
  float m = -INFINITY, l = 0.0f, acc = 0.0f;

  for (int base = 0; base < seq_k; base += WGA) {
    const int t = base + tid;
    float s = -INFINITY;
    if (t < seq_k) {
      __global const half* kp = k_cache + (size_t)t * KV_DIM + kvh * D;
      float d = 0.0f;
      for (int c = 0; c < (D >> 3); ++c) {
        const float8 a = vload_half8(c, qp), b = vload_half8(c, kp);
        d += dot(a.lo, b.lo) + dot(a.hi, b.hi);
      }
      s = d * scale;
    }
    // chunk max
    red[tid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int w = WGA >> 1; w > 0; w >>= 1) {
      if (tid < w) red[tid] = fmax(red[tid], red[tid + w]);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float m_new = fmax(m, red[0]);
    barrier(CLK_LOCAL_MEM_FENCE);
    const float p = (t < seq_k) ? native_exp(s - m_new) : 0.0f;
    p_chunk[tid] = p;
    red[tid] = p;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int w = WGA >> 1; w > 0; w >>= 1) {
      if (tid < w) red[tid] += red[tid + w];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float corr = native_exp(m - m_new);   // 0 on the first chunk (m = -inf)
    l = l * corr + red[0];
    acc *= corr;
    const int n = min(WGA, seq_k - base);
    __global const half* vp = v_cache + (size_t)base * KV_DIM + kvh * D + tid;
    for (int j = 0; j < n; ++j) acc += p_chunk[j] * vload_half(0, vp + (size_t)j * KV_DIM);
    m = m_new;
    barrier(CLK_LOCAL_MEM_FENCE);   // p_chunk / red reused next chunk
  }
  if (tid < D) vstore_half(acc / l, qh * D + tid, out);
}
