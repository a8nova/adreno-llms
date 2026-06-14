// Reference: model_info/transformers_src/activations.py (GELU) + GPT2-style tanh approx
// Implements tanh-based GELU used by HF PytorchGELUTanh.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)   vload_half((i),(p))
  #define STORE(p,i,v) vstore_half((v),(i),(p))
#else
  typedef float storage_t;
  #define LOAD(p,i)   ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
#endif

__kernel void gelu_tanh(__global const storage_t* in,
                        __global storage_t* out,
                        int n) {
  int gid = (int)get_global_id(0);
  if (gid >= n) return;
  float x = (float)LOAD(in, gid);
  // tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
  const float k0 = 0.7978845608028654f; // sqrt(2/pi)
  float x3 = x * x * x;
  float t = k0 * (x + 0.044715f * x3);
  float y = 0.5f * x * (1.0f + tanh(t));
  STORE(out, gid, y);
}
