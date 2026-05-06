// Reference: model_info/transformers_src/modeling_llama.py:171-186 LlamaMLP.forward
// Implements: down_proj(silu(gate_proj(x)) * up_proj(x))

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

inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + exp(-x));
}

__kernel void silu_mul(__global const storage_t* gate,
                      __global const storage_t* up,
                      __global storage_t* out,
                      int total) {
    int gid = (int)get_global_id(0);
    if (gid >= total) return;

    float g = (float)LOAD(gate, gid);
    float u = (float)LOAD(up, gid);

    // SiLU: x * sigmoid(x)
    float silu = g * sigmoid_f(g);
    float y = silu * u;

    STORE(out, gid, (storage_t)y);
}
