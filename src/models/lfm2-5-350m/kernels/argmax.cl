// GPU-side argmax for the lm_head logits row (greedy decode hot path).
//
// Input:  logits[V] (fp16), V = vocab_size = 65536
// Output: out_idx[1] (int32), the argmax index
//
// Two-pass reduction:
//   Pass 1 (argmax_block): each WG of 64 threads handles a contiguous chunk
//     of CHUNK=1024 fp16 logits, reduces to per-WG (max_val, max_idx) and
//     writes into partials[2 * num_wg]. For V=65536 → 64 WGs.
//   Pass 2 (argmax_final): single WG of 64 threads reduces num_wg pairs to a
//     single (max_val, max_idx) and writes the index to out_idx[0].
//
// Replaces the per-decode-token sequence:
//   1) clEnqueueReadBuffer(logits, 65536 * 2 = 128 KB) — DMA + sync
//   2) for i in [0, 65536): logits_f32[i] = nnopt_f16_to_f32(...)
//   3) std::max_element over 65536 elements
// All three combined cost ~5 ms/tok host-side. Replaced with two tiny GPU
// dispatches (~50 µs) + a 4-byte readback.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ARGMAX_WG    64
#define ARGMAX_CHUNK 1024  // fp16 elts per WG; CHUNK must be % (ARGMAX_WG * 4) == 0

// Pass 1 — block-wise argmax across the vocab.
//
// Dispatch: gws = num_wg * ARGMAX_WG, lws = ARGMAX_WG, where
//           num_wg = ceil_div(V, ARGMAX_CHUNK).
__kernel __attribute__((reqd_work_group_size(ARGMAX_WG, 1, 1)))
void argmax_block(
    __global const half* logits,   // [V]
    __global float* partials_val,  // [num_wg]
    __global int*   partials_idx,  // [num_wg]
    const int V) {
  __local float lv[ARGMAX_WG];
  __local int   li[ARGMAX_WG];

  const int wg  = get_group_id(0);
  const int tid = get_local_id(0);
  const int chunk_start = wg * ARGMAX_CHUNK;

  float my_max = -FLT_MAX;
  int   my_idx = -1;

  // Each thread strides through the chunk in vec4 chunks for coalesced reads.
  const int chunk_end = min(chunk_start + ARGMAX_CHUNK, V);
  // Stride: WG threads × 4 fp16 = 256 contiguous fp16 per iteration.
  const int chunk_size = chunk_end - chunk_start;
  const int chunk_v4   = chunk_size >> 2;

  for (int i4 = tid; i4 < chunk_v4; i4 += ARGMAX_WG) {
    float4 v = vload_half4(i4, logits + chunk_start);
    int    base_idx = chunk_start + i4 * 4;
    if (v.s0 > my_max) { my_max = v.s0; my_idx = base_idx + 0; }
    if (v.s1 > my_max) { my_max = v.s1; my_idx = base_idx + 1; }
    if (v.s2 > my_max) { my_max = v.s2; my_idx = base_idx + 2; }
    if (v.s3 > my_max) { my_max = v.s3; my_idx = base_idx + 3; }
  }
  // Tail (V not divisible by 4 — V=65536 always is, but kept for safety).
  for (int i = chunk_start + chunk_v4 * 4 + tid; i < chunk_end; i += ARGMAX_WG) {
    float v = vload_half(0, logits + i);
    if (v > my_max) { my_max = v; my_idx = i; }
  }

  lv[tid] = my_max;
  li[tid] = my_idx;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Tree reduce — argmax (NOT sum). At each step: take the (val, idx) pair
  // with the larger val.
  for (int s = ARGMAX_WG >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      if (lv[tid + s] > lv[tid]) {
        lv[tid] = lv[tid + s];
        li[tid] = li[tid + s];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    partials_val[wg] = lv[0];
    partials_idx[wg] = li[0];
  }
}

// Pass 2 — final reduction over the per-WG partials.
// Dispatch: gws = ARGMAX_WG, lws = ARGMAX_WG. Caller guarantees num_wg <= ARGMAX_WG.
// For V=65536, num_wg=64 = ARGMAX_WG exactly.
__kernel __attribute__((reqd_work_group_size(ARGMAX_WG, 1, 1)))
void argmax_final(
    __global const float* partials_val,  // [num_wg]
    __global const int*   partials_idx,  // [num_wg]
    __global int*         out_idx,       // [1]
    const int num_wg) {
  __local float lv[ARGMAX_WG];
  __local int   li[ARGMAX_WG];
  const int tid = get_local_id(0);

  if (tid < num_wg) {
    lv[tid] = partials_val[tid];
    li[tid] = partials_idx[tid];
  } else {
    lv[tid] = -FLT_MAX;
    li[tid] = -1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = ARGMAX_WG >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      if (lv[tid + s] > lv[tid]) {
        lv[tid] = lv[tid + s];
        li[tid] = li[tid + s];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) out_idx[0] = li[0];
}
