// Reference: model_info/modeling_openelm.py (OpenELMFeedForward / ffn_with_glu), activation_fn_name='swish'
// Implements SwiGLU: y = silu(gate) * up  (a.k.a swish)
// Dtype-template preamble — mirrors kernels/utils.cl conventions.
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

inline float sigmoid_f(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float silu_f(float x) { return x * sigmoid_f(x); }

// gate_up is [rows, 2*intermediate] in row-major (gate then up)
// out is [rows, intermediate]
__kernel void swiglu_forward(
    __global const storage_t* gate_up,
    __global storage_t* out,
    int intermediate) {
  const int gid = (int)get_global_id(0);
  const int col = gid % intermediate;
  const int row = gid / intermediate;
  const int base = row * (2 * intermediate);
  float g = (float)LOAD(gate_up, base + col);
  float u = (float)LOAD(gate_up, base + intermediate + col);
  float y = silu_f(g) * u;
  STORE(out, gid, y);
}

__kernel void bias_add(
    __global storage_t* x,
    __global const storage_t* bias,
    int cols,
    int total) {
  const int gid = (int)get_global_id(0);
  if (gid >= total) return;
  const int c = gid % cols;
  float v = (float)LOAD(x, gid) + (float)LOAD(bias, c);
  STORE(x, gid, v);
}
