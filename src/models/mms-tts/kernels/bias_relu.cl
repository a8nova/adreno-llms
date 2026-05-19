// Reference: model_info/transformers_src/modeling_vits.py (ReLU activation usage)
// Kernel: y = relu(x + bias) with optional per-channel bias.
// For flat tensors, pass has_bias=0 and bias may be null.

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

__kernel void bias_relu(__global const storage_t* x,
                        __global const storage_t* bias,
                        __global storage_t* out,
                        const int has_bias,
                        const int N,
                        const int C) {
  int gid = get_global_id(0);
  if (gid >= N) return;
  float v = LOAD(x, gid);
  if (has_bias && C > 0) {
    int c = gid % C;
    v += LOAD(bias, c);
  }
  if (v < 0.0f) v = 0.0f;
  STORE(out, gid, v);
}
