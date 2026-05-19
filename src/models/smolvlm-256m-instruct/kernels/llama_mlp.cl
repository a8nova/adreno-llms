// Reference: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/modeling_llama.py LlamaMLP.forward
// Kernel: SwiGLU pointwise: gate = silu(gate) * up

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

inline float silu_f(float x) {
  return x / (1.0f + exp(-x));
}

__kernel void swiglu(__global storage_t* gate,
                    __global const storage_t* up,
                    const int total) {
  int gid = get_global_id(0);
  if (gid >= total) return;
  float g = LOAD(gate, gid);
  float u = LOAD(up, gid);
  float y = silu_f(g) * u;
  STORE(gate, gid, y);
}
