// GPU-side argmax over a logits vector. Single workgroup processes the full
// vocab via strided scan + tree-reduce. Output is one int32 (the token id).
//
// Used by the greedy decode fast path to eliminate the ~100 KB readback +
// host-side std::max_element each decode token. The host instead reads back
// just one int32 — under 1 µs over USB and no fp16→fp32 conversion.
//
// Threading: 1 WG with WG_SIZE_AM threads. Each thread strides through the
// vocab finding its local (max, idx). Tree-reduce in __local memory picks
// the global argmax. WG_SIZE_AM=256 → each thread examines vocab/256 ≈ 197
// elements for vocab=50280.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i) vload_half((i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i) ((p)[(i)])
#endif

#ifndef WG_SIZE_AM
#define WG_SIZE_AM 256
#endif

// `offset` is the start index (in storage_t units, NOT bytes) of the slice
// inside `logits` to scan. For prefill the caller passes (seq_len-1)*vocab
// to point at the last-token row; for decode the caller passes 0. Avoids
// clCreateSubBuffer which has 128-byte (CL_DEVICE_MEM_BASE_ADDR_ALIGN)
// origin alignment that doesn't always hold for arbitrary seq_len*vocab.
__kernel __attribute__((reqd_work_group_size(WG_SIZE_AM, 1, 1)))
void argmax_logits(
    __global const storage_t* logits,   // [seq_len, vocab_size]
    __global int* out_idx,              // [1] — single token id written here
    const int vocab_size,
    const int offset) {
  __local float ls_val[WG_SIZE_AM];
  __local int   ls_idx[WG_SIZE_AM];

  const int tid = get_local_id(0);

  float my_max = -INFINITY;
  int   my_idx = -1;

  // Each thread strides through vocab.
  for (int i = tid; i < vocab_size; i += WG_SIZE_AM) {
    float v = LOAD(logits, offset + i);
    if (v > my_max) {
      my_max = v;
      my_idx = i;
    }
  }

  ls_val[tid] = my_max;
  ls_idx[tid] = my_idx;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Tree-reduce: max-with-index across the WG.
  for (int s = WG_SIZE_AM >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      if (ls_val[tid + s] > ls_val[tid]) {
        ls_val[tid] = ls_val[tid + s];
        ls_idx[tid] = ls_idx[tid + s];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) {
    out_idx[0] = ls_idx[0];
  }
}
