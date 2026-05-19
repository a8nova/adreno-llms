// Reference: /Users/alazarshenkute/.nnopt/ref_venvs/env_stable/lib/python3.12/site-packages/torch/nn/modules/conv.py:439-460 Conv2d.forward (bias add after matmul)

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

// Adds bias[col] to each row element.
// x: [rows, cols], bias: [cols]
__kernel void bias_add_2d(__global storage_t* x,
                          __global const storage_t* bias,
                          const int rows,
                          const int cols) {
  const int r = (int)get_global_id(0);
  const int c = (int)get_global_id(1);
  if (r >= rows || c >= cols) return;

  const long idx = (long)r * (long)cols + (long)c;
  float v = (float)LOAD(x, idx);
  float b = (float)LOAD(bias, c);
  STORE(x, idx, (storage_t)(v + b));
}
