// Reference: model_info/transformers_src/modeling_mamba2.py (RMSNorm implementation used by Mamba2 blocks)
// Note: scaffold names this layer_type "layer_norm", but Mamba2 config has rms_norm=true.
// We implement RMSNorm here: y = x * rsqrt(mean(x^2) + eps) * weight
// Dtype-template preamble — copy of kernels/utils.cl pattern.
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

#ifndef WG_SIZE
#define WG_SIZE 64
#endif

// Cooperative + vec4 rms_norm. 1 WG per row, WG_SIZE threads cooperate over
// hidden via __local-mem tree reduction. Same template as mamba1 v2 Step 2.
// hidden=768 → 12 fp16 / thread = 3 vec4. fp32 sum_sq accumulator.
//
// Replaces 1-thread-per-row scalar 2-pass version that was 14% of GPU at Step 0.
//
// Host dispatch: gws = rows * WG_SIZE, lws = WG_SIZE.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void rms_norm(
    __global const storage_t* x,      // [rows, hidden]
    __global const storage_t* weight, // [hidden]
    __global storage_t* y,            // [rows, hidden]
    const int hidden,
    const float eps) {
  __local float ls[WG_SIZE];

  const int row = get_group_id(0);
  const int tid = get_local_id(0);
  const int off = row * hidden;

  const int hp = hidden / WG_SIZE;       // 12 fp16 / thread at hidden=768
  const int h_start = tid * hp;

  // Pass 1: sum of squares.
  float ss = 0.0f;
#ifdef USE_FP16
  int j = 0;
  for (; j + 3 < hp; j += 4) {
    const int idx = h_start + j;
    float4 v = vload_half4(0, x + off + idx);
    ss += dot(v, v);
  }
  for (; j < hp; ++j) {
    const float v = LOAD(x, off + h_start + j);
    ss += v * v;
  }
#else
  for (int jj = 0; jj < hp; ++jj) {
    const float v = LOAD(x, off + h_start + jj);
    ss += v * v;
  }
#endif

  ls[tid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) ls[tid] += ls[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv_rms = rsqrt(ls[0] / (float)hidden + eps);

  // Pass 2: y = weight * x * inv_rms.
#ifdef USE_FP16
  int j2 = 0;
  for (; j2 + 3 < hp; j2 += 4) {
    const int idx = h_start + j2;
    float4 v = vload_half4(0, x      + off + idx);
    float4 w = vload_half4(0, weight + idx);
    float4 yv = v * inv_rms * w;
    vstore_half4(yv, 0, y + off + idx);
  }
  for (; j2 < hp; ++j2) {
    const int idx = h_start + j2;
    const float v = LOAD(x, off + idx);
    const float w = LOAD(weight, idx);
    STORE(y, off + idx, v * inv_rms * w);
  }
#else
  for (int jj = 0; jj < hp; ++jj) {
    const int idx = h_start + jj;
    const float v = LOAD(x, off + idx);
    const float w = LOAD(weight, idx);
    STORE(y, off + idx, v * inv_rms * w);
  }
#endif
}

__kernel void rms_norm_gated(
    __global const storage_t* x,      // [rows, hidden]
    __global const storage_t* gate,   // [rows, hidden]
    __global const storage_t* weight, // [hidden]
    __global storage_t* y,            // [rows, hidden]
    const int hidden,
    const float eps) {
  const int row = (int)get_global_id(0);
  const int base = row * hidden;

  // Apply gating then RMS norm.
  float ss = 0.0f;
  for (int i = 0; i < hidden; ++i) {
    const float xv = LOAD(x, base + i);
    const float gv = LOAD(gate, base + i);
    const float gated = xv * (gv / (1.0f + exp(-gv))); // silu(g) = g * sigmoid(g)
    ss += gated * gated;
  }
  const float inv_rms = rsqrt(ss / (float)hidden + eps);

  for (int i = 0; i < hidden; ++i) {
    const float xv = LOAD(x, base + i);
    const float gv = LOAD(gate, base + i);
    const float gated = xv * (gv / (1.0f + exp(-gv)));
    const float w = LOAD(weight, i);
    STORE(y, base + i, gated * inv_rms * w);
  }
}
