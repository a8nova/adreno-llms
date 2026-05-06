// Reference: model_info/transformers_src/modeling_lfm2.py (Lfm2RMSNorm uses RMSNorm, not LayerNorm);
//            this kernel is a generic LayerNorm implementation kept for scaffold completeness.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)     vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)     ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
#endif

#define WG_SIZE 64

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void layernorm_forward(
    __global const storage_t* x,
    __global const storage_t* weight,
    __global const storage_t* bias,
    __global storage_t* out,
    const int rows,
    const int cols,
    const float eps) {
  const int row = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  if (row >= rows) return;

  const int base = row * cols;
  __local float partial[WG_SIZE];
  __local float mean_bcast;
  __local float inv_std_bcast;

  // Pass 1: mean
  float sum = 0.0f;
  for (int c = lid; c < cols; c += WG_SIZE) sum += (float)LOAD(x, base + c);
  partial[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) partial[lid] += partial[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) mean_bcast = partial[0] / (float)cols;
  barrier(CLK_LOCAL_MEM_FENCE);
  const float mean = mean_bcast;

  // Pass 2: variance
  float ss = 0.0f;
  for (int c = lid; c < cols; c += WG_SIZE) {
    float d = (float)LOAD(x, base + c) - mean;
    ss += d * d;
  }
  partial[lid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) partial[lid] += partial[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) inv_std_bcast = native_rsqrt((partial[0] / (float)cols) + eps);
  barrier(CLK_LOCAL_MEM_FENCE);
  const float inv_std = inv_std_bcast;

  // Pass 3: normalize + affine
  for (int c = lid; c < cols; c += WG_SIZE) {
    float v = (float)LOAD(x, base + c);
    float w = (float)LOAD(weight, c);
    float b = (float)LOAD(bias, c);
    STORE(out, base + c, (v - mean) * inv_std * w + b);
  }
}
