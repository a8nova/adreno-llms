// Reference: model_info/transformers_src/modeling_qwen2.py:170-193 Qwen2RMSNorm.forward
// Implements RMSNorm (no mean subtraction, no bias):
//   variance = mean(x^2)
//   y = x * rsqrt(variance + eps) * weight
//
// Dtype-template preamble — copy from kernels/utils.cl.
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

__kernel void rmsnorm_forward(
    __global const storage_t* x,
    __global storage_t* y,
    __global const storage_t* weight,
    int hidden_size,
    float eps) {
  const int row = (int)get_global_id(0);
  const int base = row * hidden_size;

  float ss = 0.0f;
  for (int i = 0; i < hidden_size; ++i) {
    const float v = (float)LOAD(x, base + i);
    ss += v * v;
  }
  const float inv_rms = rsqrt(ss / (float)hidden_size + eps);

  for (int i = 0; i < hidden_size; ++i) {
    const float v = (float)LOAD(x, base + i);
    const float w = (float)LOAD(weight, i);
    STORE(y, base + i, v * inv_rms * w);
  }
}
