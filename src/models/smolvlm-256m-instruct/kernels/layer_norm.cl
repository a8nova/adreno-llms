// Cooperative-workgroup LayerNorm forward kernel.
// Implements: y = ((x - mean) / sqrt(var + eps)) * weight + bias
// mean = sum(x)/cols, var = sum((x-mean)^2)/cols

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

#ifdef USE_FP16
  const int C4 = cols >> 2;

  __global const half* xh = (__global const half*)x + base;
  __global const half* wh = (__global const half*)weight;
  __global const half* bh = (__global const half*)bias;

  // Pass 1: sum for mean (fp32 accumulation).
  float sum = 0.0f;
  for (int c4 = lid; c4 < C4; c4 += WG_SIZE) {
    const float4 v = vload_half4(c4, xh);
    sum += v.s0 + v.s1 + v.s2 + v.s3;
  }
  for (int c = (C4 << 2) + lid; c < cols; c += WG_SIZE) {
    sum += (float)LOAD(x, base + c);
  }

  partial[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) partial[lid] += partial[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) mean_bcast = partial[0] / (float)cols;
  barrier(CLK_LOCAL_MEM_FENCE);
  const float mean = mean_bcast;

  // Pass 2: sum of squared deviations for variance (fp32 accumulation).
  float ss = 0.0f;
  for (int c4 = lid; c4 < C4; c4 += WG_SIZE) {
    const float4 v = vload_half4(c4, xh);
    const float4 d = v - (float4)(mean);
    ss += dot(d, d);
  }
  for (int c = (C4 << 2) + lid; c < cols; c += WG_SIZE) {
    const float d = (float)LOAD(x, base + c) - mean;
    ss += d * d;
  }

  partial[lid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) partial[lid] += partial[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    const float var = partial[0] / (float)cols;
    inv_std_bcast = native_rsqrt(var + eps);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float inv_std = inv_std_bcast;

  // Pass 3: normalize + affine.
  __global half* oh = (__global half*)out + base;
  for (int c4 = lid; c4 < C4; c4 += WG_SIZE) {
    const float4 v = vload_half4(c4, xh);
    const float4 w = vload_half4(c4, wh);
    const float4 b = vload_half4(c4, bh);
    vstore_half4((v - (float4)(mean)) * inv_std * w + b, c4, oh);
  }
  for (int c = (C4 << 2) + lid; c < cols; c += WG_SIZE) {
    const float v = (float)LOAD(x, base + c);
    const float w = (float)LOAD(weight, c);
    const float b = (float)LOAD(bias, c);
    STORE(out, base + c, (storage_t)((v - mean) * inv_std * w + b));
  }
#else
  // FP32 path.
  float sum = 0.0f;
  for (int c = lid; c < cols; c += WG_SIZE) {
    sum += (float)LOAD(x, base + c);
  }

  partial[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) partial[lid] += partial[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) mean_bcast = partial[0] / (float)cols;
  barrier(CLK_LOCAL_MEM_FENCE);
  const float mean = mean_bcast;

  float ss = 0.0f;
  for (int c = lid; c < cols; c += WG_SIZE) {
    const float d = (float)LOAD(x, base + c) - mean;
    ss += d * d;
  }

  partial[lid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) partial[lid] += partial[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    const float var = partial[0] / (float)cols;
    inv_std_bcast = native_rsqrt(var + eps);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float inv_std = inv_std_bcast;

  for (int c = lid; c < cols; c += WG_SIZE) {
    const float v = (float)LOAD(x, base + c);
    const float w = (float)LOAD(weight, c);
    const float b = (float)LOAD(bias, c);
    STORE(out, base + c, (storage_t)((v - mean) * inv_std * w + b));
  }
#endif
}
