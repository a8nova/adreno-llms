// Reference: model_info/transformers_src/modeling_lfm2.py:32-63 Lfm2RMSNorm.forward
// Implements RMSNorm: y = weight * (x * rsqrt(mean(x^2) + eps))
//
// Cooperative-WG variant: one workgroup per row, WG=64 threads cooperate
// over the column dimension via vec4 fp16 loads + __local-mem tree
// reduction. Host launches with gws=rows*64, lws=64.
//
// NOTE: this kernel REQUIRES lws == WG_SIZE for correct get_group_id behavior
// on Adreno (reqd_work_group_size).

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

__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__kernel void rmsnorm_forward(
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
  __local float partial[WG_SIZE];

#ifdef USE_FP16
  const int C4 = cols >> 2;
  __global const half* xh = (__global const half*)x + base;
  __global const half* wh = (__global const half*)weight;

  float acc = 0.0f;
  for (int c4 = lid; c4 < C4; c4 += WG_SIZE) {
    float4 v = vload_half4(c4, xh);
    acc += dot(v, v);
  }
  for (int c = C4 * 4 + lid; c < cols; c += WG_SIZE) {
    float v = LOAD(x, base + c);
    acc += v * v;
  }
  partial[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) partial[lid] += partial[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv_rms = native_rsqrt((partial[0] / (float)cols) + eps);

  __global half* oh = (__global half*)out + base;
  for (int c4 = lid; c4 < C4; c4 += WG_SIZE) {
    float4 v = vload_half4(c4, xh);
    float4 w = vload_half4(c4, wh);
    vstore_half4(v * inv_rms * w, c4, oh);
  }
  for (int c = C4 * 4 + lid; c < cols; c += WG_SIZE) {
    float v = LOAD(x, base + c);
    float w = LOAD(weight, c);
    STORE(out, base + c, (v * inv_rms) * w);
  }
#else
  float acc = 0.0f;
  for (int c = lid; c < cols; c += WG_SIZE) {
    float v = LOAD(x, base + c);
    acc += v * v;
  }
  partial[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) partial[lid] += partial[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv_rms = native_rsqrt((partial[0] / (float)cols) + eps);
  for (int c = lid; c < cols; c += WG_SIZE) {
    float v = LOAD(x, base + c);
    float w = LOAD(weight, c);
    STORE(out, base + c, (v * inv_rms) * w);
  }
#endif
}
