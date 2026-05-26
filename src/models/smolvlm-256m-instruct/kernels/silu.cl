// Reference: /Users/alazarshenkute/.nnopt/ref_venvs/env_stable/lib/python3.12/site-packages/torch/nn/modules/activation.py:358-397 SiLU.forward
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

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

__kernel void silu_forward(
    __global const storage_t* x,
    __global storage_t* y,
    const int n_elements) {
  const int i = (int)get_global_id(0);
  if (i >= n_elements) return;
  const float v = (float)LOAD(x, i);
  const float s = 1.0f / (1.0f + exp(-v));
  STORE(y, i, (storage_t)(v * s));
}
