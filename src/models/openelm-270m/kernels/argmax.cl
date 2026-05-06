// GPU greedy-argmax over a [VOCAB] logits buffer.
//
// Step 3 (BENCHMARK.md): replaces the per-token 100 KB host-side readback
// (vocab × fp16) + linear scan with a single tree-reduced argmax that
// returns one int32 to the host. Device→host traffic shrinks ~25,000×.
//
// Single workgroup of WG_SIZE threads runs a strided pass over the vocab
// (each thread tracks a private (best_val, best_idx) pair), then a tree-
// reduction in __local memory selects the global winner. Result is written
// as a single int32 to out_idx[0].

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if defined(NNOPT_USE_FP16)
typedef half storage_t;
#else
typedef float storage_t;
#endif

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void argmax_fp16(
    __global const storage_t* logits,
    __global int* out_idx,
    const int vocab,
    const int offset_elements) {
  __local float ls_val[WG_SIZE];
  __local int   ls_idx[WG_SIZE];

  const int tid = get_local_id(0);
  __global const storage_t* row = logits + offset_elements;

  float best_val = -INFINITY;
  int   best_idx = 0;

  for (int v = tid; v < vocab; v += WG_SIZE) {
    const float x = (float)row[v];
    if (x > best_val) {
      best_val = x;
      best_idx = v;
    }
  }
  ls_val[tid] = best_val;
  ls_idx[tid] = best_idx;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float a = ls_val[tid];
      const float b = ls_val[tid + stride];
      if (b > a) {
        ls_val[tid] = b;
        ls_idx[tid] = ls_idx[tid + stride];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) {
    out_idx[0] = ls_idx[0];
  }
}
