// GELU kernel (erf formulation).
// Reference: model_info/transformers_src/modeling_whisper.py ACT2FN["gelu"] usage (GELUActivation)

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

inline float gelu_erf_f(float x) {
    const float inv_sqrt2 = 0.7071067811865475f;
    return 0.5f * x * (1.0f + erf(x * inv_sqrt2));
}

__kernel void gelu_forward(__global const storage_t* x,
                          __global storage_t* y,
                          const int n) {
    int gid = (int)get_global_id(0);
    if (gid >= n) return;
    float v = (float)LOAD(x, gid);
    STORE(y, gid, gelu_erf_f(v));
}
