// Reference: model_info/transformers_src/modeling_mamba2.py (RMSNorm path: rms_norm=True)
// Simple RMSNorm kernel used for Mamba2 mixer.norm (d_inner) and backbone norms.
// Dtype-template preamble — DO NOT EDIT. Driven by host-side -DNNOPT_USE_FP16.
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

__kernel void rms_norm(
    __global const storage_t* x,      // [rows, cols]
    __global const storage_t* weight, // [cols]
    __global storage_t* y,            // [rows, cols]
    const int cols,
    const float eps) {
  const int row = (int)get_global_id(0);
  const int base = row * cols;
  float ss = 0.0f;
  for (int i = 0; i < cols; ++i) {
    float v = LOAD(x, base + i);
    ss += v * v;
  }
  float inv_rms = rsqrt(ss / (float)cols + eps);
  for (int i = 0; i < cols; ++i) {
    float v = LOAD(x, base + i);
    float w = LOAD(weight, i);
    STORE(y, base + i, v * inv_rms * w);
  }
}
