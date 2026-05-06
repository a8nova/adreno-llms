// Reference: model_info/transformers_src/modeling_lfm2.py:41-66 Lfm2RMSNorm.forward
// RMSNorm over last dimension (hidden_size): y = x * rsqrt(mean(x^2) + eps) * weight

#include "utils.cl"

__kernel void operator_rmsnorm_forward(
    __global const storage_t* x,
    __global const storage_t* weight,
    __global storage_t* y,
    const int rows,
    const int hidden_size,
    const float eps) {
  const int r = get_global_id(0);
  if (r >= rows) return;

  const int base = r * hidden_size;
  float ss = 0.0f;
  for (int i = 0; i < hidden_size; ++i) {
    const float v = LOAD(x, base + i);
    ss += v * v;
  }
  const float mean_sq = ss / (float)hidden_size;
  const float inv_rms = rsqrt(mean_sq + eps);

  for (int i = 0; i < hidden_size; ++i) {
    const float v = LOAD(x, base + i);
    const float w = LOAD(weight, i);
    STORE(y, base + i, v * inv_rms * w);
  }
}
