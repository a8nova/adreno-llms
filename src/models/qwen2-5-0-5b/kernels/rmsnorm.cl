// Reference: model_info/transformers_src/modeling_qwen2.py:177-198 Qwen2RMSNorm.forward
// Reference: HF transformers — LlamaRMSNorm.forward
// Implements RMSNorm: y = weight * (x * rsqrt(mean(x^2) + eps))
//
// Cooperative-WG variant: one workgroup per row, WG=64 threads cooperate
// over the column dimension via vec4 fp16 loads + __local-mem tree
// reduction. Host launches with gws=rows*64, lws=64. Bumping WG requires
// kernel rewrite (the kp / kp4 split below assumes WG=64).

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
void rmsnorm_forward(
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
  // Vec4 fp16 path — cols expected to be multiple of 4 (head_dim ⌃
  // hidden_size for every common transformer). Scalar tail handles
  // any leftover (unusual configs).
  const int C4 = cols >> 2;
  __global const half* xh = (__global const half*)x + base;
  __global const half* wh = (__global const half*)weight;

  // Pass 1: partial sum-of-squares over this thread's strided vec4 cols.
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
  // Match torch.rsqrt semantics more closely than native_rsqrt.
  const float inv_rms = 1.0f / sqrt((partial[0] / (float)cols) + eps);

  // Pass 2: vec4 normalize+weight write.
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
  // Fp32 fall-back path — same cooperative structure, scalar reads.
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
  // Match torch.rsqrt semantics more closely than native_rsqrt.
  const float inv_rms = 1.0f / sqrt((partial[0] / (float)cols) + eps);
  for (int c = lid; c < cols; c += WG_SIZE) {
    float v = LOAD(x, base + c);
    float w = LOAD(weight, c);
    STORE(out, base + c, (v * inv_rms) * w);
  }
#endif
}
