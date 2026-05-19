// Reference: model_info/transformers_src/modeling_vits.py (tensor permute patterns around Conv1d)
//
// Minimal transpose helpers for VITS Conv1d glue.
// Layout conversions:
//   BTC [B, T, C]  <->  NCL [B, C, T]

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)   vload_half((i),(p))
  #define STORE(p,i,v) vstore_half((v),(i),(p))
#else
  typedef float storage_t;
  #define LOAD(p,i)   ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
#endif

__kernel void transpose_btc_to_ncl(
    __global const storage_t* in_btc,
    __global storage_t* out_ncl,
    const int B,
    const int T,
    const int C) {
  const int gid = get_global_id(0);
  const int total = B * T * C;
  if (gid >= total) return;

  const int b = gid / (T * C);
  const int rem = gid - b * (T * C);
  const int t = rem / C;
  const int c = rem - t * C;

  const int in_idx = (b * T + t) * C + c;
  const int out_idx = (b * C + c) * T + t;
  STORE(out_ncl, out_idx, (float)LOAD(in_btc, in_idx));
}

__kernel void transpose_ncl_to_btc(
    __global const storage_t* in_ncl,
    __global storage_t* out_btc,
    const int B,
    const int T,
    const int C) {
  const int gid = get_global_id(0);
  const int total = B * T * C;
  if (gid >= total) return;

  const int b = gid / (T * C);
  const int rem = gid - b * (T * C);
  const int t = rem / C;
  const int c = rem - t * C;

  const int out_idx = (b * T + t) * C + c;
  const int in_idx = (b * C + c) * T + t;
  STORE(out_btc, out_idx, (float)LOAD(in_ncl, in_idx));
}
