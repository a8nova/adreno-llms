// Shared preamble — concatenated FIRST by GpuOps::init ahead of every other
// kernel file. Defines the fp16/fp32 storage type + LOAD/STORE and the shared
// activation helper. Driven by host-side -DNNOPT_USE_FP16. DO NOT define
// kernels here; this is the common header for all kernels/*.cl files.
#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)    vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
  // 128-bit (8×fp16) vectorized load -> float8 (converts fp16->fp32). base must be a
  // multiple of 8. Reads p[base .. base+7].
  #define LOAD8(p, base) vload_half8((base) >> 3, (p))
  #define STORE8(p, base, v) vstore_half8((v), (base) >> 3, (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)    ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
  #define LOAD8(p, base) vload8((base) >> 3, (p))
  #define STORE8(p, base, v) vstore8((v), (base) >> 3, (p))
#endif

// fp32 dot of two float8 (two float4 dots).
static inline float dot8(float8 a, float8 b) { return dot(a.lo, b.lo) + dot(a.hi, b.hi); }

// Activation codes: 0 gelu, 1 swish, 2 relu, 3 tanh, 4 sigmoid, 5 lrelu0.1, 6 lrelu0.01.
static inline float act_apply(float x, int code) {
    if (code == 0) return 0.5f * x * (1.0f + erf(x * 0.70710678f));
    if (code == 1) return x / (1.0f + native_exp(-x));      // swish
    if (code == 2) return x > 0.0f ? x : 0.0f;
    if (code == 3) return tanh(x);
    if (code == 4) return 1.0f / (1.0f + native_exp(-x));   // sigmoid
    if (code == 5) return x >= 0.0f ? x : x * 0.1f;
    if (code == 6) return x >= 0.0f ? x : x * 0.01f;
    return x;
}
