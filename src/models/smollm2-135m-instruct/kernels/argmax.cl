// GPU argmax over a contiguous logits row.
//
// Single-WG cooperative reduce. Each of WG_SIZE threads strides through
// the logits, tracking a running (best_val, best_idx); a __local tree
// reduce produces the final argmax. One int32 written to out_idx[0].
//
// Used by Model::generate's greedy fast path (Lever 1). Eliminates the
// 100 KB padded-vocab readback + host fp16→fp32 + host argmax — at
// vocab=50288 that's ~5 ms/tok of host stall.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i) vload_half((i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i) ((p)[(i)])
#endif

#ifndef WG_SIZE
#define WG_SIZE 64
#endif

#ifndef NUM_WG
#define NUM_WG 32
#endif

// Pass 1: each WG scans an interleaved subset of [0, valid_n) and writes
// per-WG (best_val, best_idx) into scratch buffers.
//   gws = NUM_WG * WG_SIZE, lws = WG_SIZE
//   scratch_v: float[NUM_WG], scratch_i: int[NUM_WG]
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void argmax_partial(__global const storage_t* logits,
                    __global float* scratch_v,
                    __global int*   scratch_i,
                    const int n,
                    const int valid_n,
                    const int row_off) {
  __local float ls_v[WG_SIZE];
  __local int   ls_i[WG_SIZE];

  const int tid = get_local_id(0);
  const int wg  = get_group_id(0);
  const int total_threads = WG_SIZE * NUM_WG;
  const int gid = wg * WG_SIZE + tid;
  const int limit = (valid_n > 0 && valid_n < n) ? valid_n : n;

  float best_v = -INFINITY;
  int   best_i = 0;

  for (int j = gid; j < limit; j += total_threads) {
    float v = LOAD(logits, row_off + j);
    if (v > best_v) { best_v = v; best_i = j; }
  }

  ls_v[tid] = best_v;
  ls_i[tid] = best_i;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      if (ls_v[tid + s] > ls_v[tid]) {
        ls_v[tid] = ls_v[tid + s];
        ls_i[tid] = ls_i[tid + s];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) {
    scratch_v[wg] = ls_v[0];
    scratch_i[wg] = ls_i[0];
  }
}

// Pass 2: 1 WG of NUM_WG threads (or padded) reduces NUM_WG candidates,
// writes the final argmax to BOTH out_idx[write_off] (history) AND
// cur_token_buf[0] (next-iter input). Folds the copy step.
__kernel __attribute__((reqd_work_group_size(NUM_WG, 1, 1)))
void argmax_final(__global const float* scratch_v,
                  __global const int*   scratch_i,
                  __global int* out_idx,
                  __global int* cur_token,
                  const int write_off) {
  __local float ls_v[NUM_WG];
  __local int   ls_i[NUM_WG];
  const int tid = get_local_id(0);

  ls_v[tid] = scratch_v[tid];
  ls_i[tid] = scratch_i[tid];
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = NUM_WG >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      if (ls_v[tid + s] > ls_v[tid]) {
        ls_v[tid] = ls_v[tid + s];
        ls_i[tid] = ls_i[tid + s];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) {
    out_idx[write_off] = ls_i[0];
    cur_token[0] = ls_i[0];
  }
}

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void argmax_row(__global const storage_t* logits,
                __global int* out_idx,
                const int n,
                const int valid_n,
                const int row_off,
                const int write_off) {
  // n = padded vocab (length of one logits row).
  // valid_n = real vocab; positions [valid_n, n) are pad and must be ignored.
  // row_off = element offset into `logits` of the row we want
  //           (= (seq_len - 1) * n). Saves a clCreateSubBuffer per decode.
  __local float ls_v[WG_SIZE];
  __local int   ls_i[WG_SIZE];

  const int tid = get_local_id(0);
  const int limit = (valid_n > 0 && valid_n < n) ? valid_n : n;

  float best_v = -INFINITY;
  int   best_i = 0;

  for (int j = tid; j < limit; j += WG_SIZE) {
    float v = LOAD(logits, row_off + j);
    if (v > best_v) { best_v = v; best_i = j; }
  }

  ls_v[tid] = best_v;
  ls_i[tid] = best_i;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      if (ls_v[tid + s] > ls_v[tid]) {
        ls_v[tid] = ls_v[tid + s];
        ls_i[tid] = ls_i[tid + s];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) out_idx[write_off] = ls_i[0];
}
