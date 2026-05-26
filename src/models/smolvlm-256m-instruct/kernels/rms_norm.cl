// Reference: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/modeling_llama.py LlamaRMSNorm.forward
// (See class LlamaRMSNorm: hidden_states->fp32, variance=pow(2).mean(-1,keepdim=True), hidden_states*=rsqrt(variance+eps), return weight*hidden_states)

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

#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define WG_SIZE 64

// Fused residual-add + RMSNorm. Reads x[row, :] and residual[row, :], writes
// (x + residual) back into x in-place (so callers reuse x as the running
// residual stream), and writes rmsnorm(x + residual) into out. One workgroup
// per row.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("full")))
void rms_norm_residual_forward(
    __global storage_t* x,             // [rows, cols] — mutated to x + residual
    __global const storage_t* residual,
    __global const storage_t* weight,
    __global storage_t* out,
    const int rows,
    const int cols,
    const float eps) {
  const int row = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  if (row >= rows) return;
  const int base = row * cols;

  // Pass 1: add residual into x in-place AND accumulate sumsq for that row.
  float ss = 0.0f;
#ifdef USE_FP16
  const int C4 = cols >> 2;
  __global half* xh = (__global half*)x + base;
  __global const half* rh = (__global const half*)residual + base;
  for (int c4 = lid; c4 < C4; c4 += WG_SIZE) {
    float4 xv = vload_half4(c4, xh);
    float4 rv = vload_half4(c4, rh);
    float4 s = xv + rv;
    vstore_half4(s, c4, xh);
    ss += dot(s, s);
  }
  for (int c = (C4 << 2) + lid; c < cols; c += WG_SIZE) {
    float v = (float)LOAD(x, base + c) + (float)LOAD(residual, base + c);
    STORE(x, base + c, (storage_t)v);
    ss += v * v;
  }
#else
  for (int c = lid; c < cols; c += WG_SIZE) {
    float v = (float)LOAD(x, base + c) + (float)LOAD(residual, base + c);
    STORE(x, base + c, (storage_t)v);
    ss += v * v;
  }
#endif

  // Opt #6: subgroup reduce replaces __local + barrier tree reduce.
  const float total_ss = sub_group_reduce_add(ss);
  const float mean_ss = total_ss / (float)cols;
  const float inv_rms = rsqrt(mean_ss + eps);

#ifdef USE_FP16
  __global const half* wh = (__global const half*)weight;
  __global half* oh = (__global half*)out + base;
  for (int c4 = lid; c4 < C4; c4 += WG_SIZE) {
    float4 xv = vload_half4(c4, xh);
    float4 wv = vload_half4(c4, wh);
    vstore_half4(xv * inv_rms * wv, c4, oh);
  }
  for (int c = (C4 << 2) + lid; c < cols; c += WG_SIZE) {
    float xv = (float)LOAD(x, base + c);
    float wv = (float)LOAD(weight, c);
    STORE(out, base + c, (storage_t)(xv * inv_rms * wv));
  }
#else
  for (int c = lid; c < cols; c += WG_SIZE) {
    float xv = (float)LOAD(x, base + c);
    float wv = (float)LOAD(weight, c);
    STORE(out, base + c, (storage_t)(xv * inv_rms * wv));
  }
#endif
}

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("full")))
void rms_norm_forward(
    __global const storage_t* x,
    __global const storage_t* weight,
    __global storage_t* out,
    const int rows,
    const int cols,
    const float eps) {
  const int row = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  if (row >= rows) return;

  const int base = row * cols;

  // sumsq
  float ss = 0.0f;
#ifdef USE_FP16
  const int C4 = cols >> 2;
  __global const half* xh = (__global const half*)x + base;
  for (int c4 = lid; c4 < C4; c4 += WG_SIZE) {
    float4 v = vload_half4(c4, xh);
    ss += dot(v, v);
  }
  for (int c = (C4 << 2) + lid; c < cols; c += WG_SIZE) {
    float v = (float)LOAD(x, base + c);
    ss += v * v;
  }
#else
  for (int c = lid; c < cols; c += WG_SIZE) {
    float v = (float)LOAD(x, base + c);
    ss += v * v;
  }
#endif

  // Opt #6: subgroup reduce replaces __local + barrier tree reduce.
  const float total_ss = sub_group_reduce_add(ss);
  const float mean_ss = total_ss / (float)cols;
  const float inv_rms = rsqrt(mean_ss + eps);

  // affine
#ifdef USE_FP16
  __global const half* wh = (__global const half*)weight;
  __global half* oh = (__global half*)out + base;
  const int C4b = cols >> 2;
  for (int c4 = lid; c4 < C4b; c4 += WG_SIZE) {
    float4 xv = vload_half4(c4, xh);
    float4 wv = vload_half4(c4, wh);
    vstore_half4(xv * inv_rms * wv, c4, oh);
  }
  for (int c = (C4b << 2) + lid; c < cols; c += WG_SIZE) {
    float xv = (float)LOAD(x, base + c);
    float wv = (float)LOAD(weight, c);
    STORE(out, base + c, (storage_t)(xv * inv_rms * wv));
  }
#else
  for (int c = lid; c < cols; c += WG_SIZE) {
    float xv = (float)LOAD(x, base + c);
    float wv = (float)LOAD(weight, c);
    STORE(out, base + c, (storage_t)(xv * inv_rms * wv));
  }
#endif
}
