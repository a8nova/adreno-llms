// Reference: model_info/transformers_src/modeling_qwen2.py:187-248 Qwen2Attention (Linear projections with bias)
// Bias-add kernel: out[m, n] += bias[n].

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)    vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)    ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
#endif

// Vec4 fp16 path: 1 thread emits 4 contiguous fp16, dispatched as
// gws = (rows*cols)/4. Caller must guarantee cols % 4 == 0 (Qwen
// Q_DIM=896, KV_DIM=128 — both divisible by 4).
__kernel void bias_add_rowmajor(
    __global storage_t* out,
    __global const storage_t* bias,
    const int rows,
    const int cols) {
  int gid = (int)get_global_id(0);
#ifdef USE_FP16
  const int total4 = (rows * cols) >> 2;
  if (gid >= total4) return;

  __global half* oh = (__global half*)out;
  __global const half* bh = (__global const half*)bias;

  // Linear vec4 index → row-major (row, col4) where col4 is the vec4 column.
  const int cols4    = cols >> 2;
  const int row      = gid / cols4;
  const int col4     = gid - row * cols4;
  (void)row;  // bias is replicated per row, so we only need col4.

  float4 v = vload_half4(gid, oh);
  float4 b = vload_half4(col4, bh);
  vstore_half4(v + b, gid, oh);
#else
  const int total = rows * cols;
  if (gid >= total) return;
  const int col = gid - (gid / cols) * cols;
  out[gid] = out[gid] + bias[col];
#endif
}
