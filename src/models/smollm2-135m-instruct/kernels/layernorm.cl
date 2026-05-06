// Reference: model_info/transformers_src/modeling_llama.py:LlamaRMSNorm.forward (for normalization pattern; LayerNorm is standard torch.nn.LayerNorm)
// Implements LayerNorm: y = ((x - mean) / sqrt(var + eps)) * weight + bias
// where mean = (1/cols) * sum(x) and var = (1/cols) * sum((x - mean)^2).
//
// Same cooperative-WG + vec4 + tree-reduce skeleton as rmsnorm_forward,
// with TWO reductions (sum then sum-of-squared-deviations) and a bias
// add in the write pass.

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
    __global const storage_t* weight,    // [cols] gamma
    __global const storage_t* bias,      // [cols] beta
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

  // Pass 1: partial sum for mean.
  float sum = 0.0f;
  for (int c4 = lid; c4 < C4; c4 += WG_SIZE) {
    float4 v = vload_half4(c4, xh);
    sum += v.s0 + v.s1 + v.s2 + v.s3;
  }
  for (int c = C4 * 4 + lid; c < cols; c += WG_SIZE) {
    sum += LOAD(x, base + c);
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

  // Pass 2: partial sum-of-squared-deviations for variance.
  float ss = 0.0f;
  for (int c4 = lid; c4 < C4; c4 += WG_SIZE) {
    float4 v = vload_half4(c4, xh);
    float4 d = v - (float4)(mean);
    ss += dot(d, d);
  }
  for (int c = C4 * 4 + lid; c < cols; c += WG_SIZE) {
    float d = LOAD(x, base + c) - mean;
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

  // Pass 3: vec4 normalize + scale + bias write.
  __global half* oh = (__global half*)out + base;
  for (int c4 = lid; c4 < C4; c4 += WG_SIZE) {
    float4 v = vload_half4(c4, xh);
    float4 w = vload_half4(c4, wh);
    float4 b = vload_half4(c4, bh);
    vstore_half4((v - (float4)(mean)) * inv_std * w + b, c4, oh);
  }
  for (int c = C4 * 4 + lid; c < cols; c += WG_SIZE) {
    float v = LOAD(x, base + c);
    float w = LOAD(weight, c);
    float b = LOAD(bias, c);
    STORE(out, base + c, (v - mean) * inv_std * w + b);
  }
#else
  // Fp32 fall-back path.
  float sum = 0.0f;
  for (int c = lid; c < cols; c += WG_SIZE) sum += LOAD(x, base + c);
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
    float d = LOAD(x, base + c) - mean;
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

  for (int c = lid; c < cols; c += WG_SIZE) {
    float v = LOAD(x, base + c);
    float w = LOAD(weight, c);
    float b = LOAD(bias, c);
    STORE(out, base + c, (v - mean) * inv_std * w + b);
  }
#endif
}
