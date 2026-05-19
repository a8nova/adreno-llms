// Reference: model_info/transformers_src/modeling_vits.py: VitsElementwiseAffine.forward (reverse=True)
// Affine inverse: y = (x - m) * exp(-logs)

// NOTE: OpenCL programs are built from a single source string (no include paths).
// Inline the shared preamble instead of using #include "utils.cl".
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


__kernel void flow_inverse_affine(
    __global const storage_t* z,     // [B*C*T]
    __global const storage_t* m,     // [C]
    __global const storage_t* logs,  // [C]
    __global storage_t* out,         // [B*C*T]
    const int B,
    const int C,
    const int T) {
  const int gid = (int)get_global_id(0);
  const int n = B * C * T;
  if (gid >= n) return;

  const int t = gid % T;
  const int tmp = gid / T;
  const int c = tmp % C;

  const float x = LOAD(z, gid);
  const float bias = LOAD(m, c);
  const float l = LOAD(logs, c);
  const float y = (x - bias) * exp(-l);
  STORE(out, gid, y);
}
