// Reference: model_info/transformers_src/modeling_qwen2.py:20-33 Qwen2MLP.forward
// Implements SwiGLU MLP: down_proj( silu(gate_proj(x)) * up_proj(x) )
// Dtype-template preamble — storage_t + LOAD/STORE.
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

// gate[i] = silu(gate[i]) * up[i]   (in-place on gate)
//
// Vec4 fp16 path: each thread processes 4 contiguous fp16 via vload_half4
// + vstore_half4. seq_len * INTERMEDIATE_SIZE is the total element count;
// caller dispatches with gws = total / 4 (Qwen INTERMEDIATE_SIZE=4864
// guarantees divisibility by 4 — caller is responsible).
//
// Dispatch: gws = total/4 (one thread per vec4 group), no required lws.
__kernel void swiglu_inplace(
    __global storage_t* gate,
    __global const storage_t* up,
    int n) {
  int gid = (int)get_global_id(0);
#ifdef USE_FP16
  const int n4 = n >> 2;
  if (gid >= n4) return;
  __global half* gh = (__global half*)gate;
  __global const half* uh = (__global const half*)up;
  float4 g = vload_half4(gid, gh);
  float4 u = vload_half4(gid, uh);
  float4 o;
  o.x = silu_f(g.x) * u.x;
  o.y = silu_f(g.y) * u.y;
  o.z = silu_f(g.z) * u.z;
  o.w = silu_f(g.w) * u.w;
  vstore_half4(o, gid, gh);
#else
  if (gid >= n) return;
  float g = (float)LOAD(gate, gid);
  float u = (float)LOAD(up,   gid);
  STORE(gate, gid, silu_f(g) * u);
#endif
}
