// Reference: model_info/transformers_src/modeling_lfm2.py:133-154 Lfm2MLP.forward
// Implements SwiGLU MLP: out = w2( silu(w1(x)) * w3(x) )

#include "utils.cl"

__kernel void silu_mul(
    __global const storage_t* a,      // [n]
    __global const storage_t* b,      // [n]
    __global storage_t* out,          // [n]
    const int n) {
  const int gid = (int)get_global_id(0);
  if (gid >= n) return;
  const float x = LOAD(a, gid);
  const float y = LOAD(b, gid);
  const float sig = 1.0f / (1.0f + exp(-x));
  const float silu = x * sig;
  STORE(out, gid, silu * y);
}
