// scale_float.cl — simple float buffer scaling helper
// Reference: model_info/transformers_src/modeling_whisper.py:241-359 WhisperAttention.forward (q_proj scaling)

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)    vload_half((i),(p))
  #define STORE(p,i,v) vstore_half((v),(i),(p))
#else
  typedef float storage_t;
  #define LOAD(p,i)    ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
#endif

// In WhisperAttention.forward, query_states are computed as:
//   query_states = (q_proj(hidden_states) * scaling)
// where q_proj output is in the model's dtype (fp16 storage here).
// So we must scale a storage_t buffer, not float.
__kernel void scale_storage_inplace(__global storage_t* x, float scale, int n) {
  int gid = (int)get_global_id(0);
  if (gid >= n) return;
  float v = (float)LOAD(x, gid);
  v *= scale;
  STORE(x, gid, v);
}
