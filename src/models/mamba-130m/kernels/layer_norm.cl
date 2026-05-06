// Reference: model_info/transformers_src/modeling_mamba.py:65-97 MambaRMSNorm.forward
// Implements RMSNorm for Mamba (no mean subtraction, no bias).

// NOTE: OpenCL on Android does not support relative includes by default.
// Use the canonical dtype/template preamble inline instead of #include "utils.cl".
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

// rms_norm: cooperative vec4 — 1 WG per row, WG_SIZE threads cooperate over
// hidden via __local-mem tree reduction. Replaces the 1-thread-per-row scalar
// version that left the GPU at WG=1 for 25 launches per decode token.
//
// Host dispatch:
//   gws = rows * WG_SIZE
//   lws = WG_SIZE
//
// Caller guarantees hidden_size % WG_SIZE == 0 (768 / 64 = 12 fp16 / thread,
// = 3 vec4). For non-conforming sizes the scalar tail in the inner loop
// handles the remainder safely.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void rms_norm(
    __global const storage_t* input,   // [rows, hidden]
    __global const storage_t* gamma,   // [hidden]
    __global storage_t* output,        // [rows, hidden]
    const int hidden_size,
    const float eps) {
  __local float ls[WG_SIZE];

  const int row = get_group_id(0);
  const int tid = get_local_id(0);
  const int off = row * hidden_size;

  const int hp = hidden_size / WG_SIZE;     // hidden per thread (768/64 = 12)
  const int h_start = tid * hp;

  // Pass 1 — sum of squares, vec4 + scalar tail.
  float ss = 0.0f;
#ifdef USE_FP16
  int j = 0;
  for (; j + 3 < hp; j += 4) {
    const int idx = h_start + j;
    float4 v = vload_half4(0, input + off + idx);
    ss += dot(v, v);
  }
  for (; j < hp; ++j) {
    const float v = LOAD(input, off + h_start + j);
    ss += v * v;
  }
#else
  for (int j = 0; j < hp; ++j) {
    const float v = LOAD(input, off + h_start + j);
    ss += v * v;
  }
#endif

  ls[tid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) ls[tid] += ls[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv_rms = rsqrt(ls[0] / (float)hidden_size + eps);

  // Pass 2 — apply gain, vec4 store.
#ifdef USE_FP16
  int j2 = 0;
  for (; j2 + 3 < hp; j2 += 4) {
    const int idx = h_start + j2;
    float4 x = vload_half4(0, input + off + idx);
    float4 g = vload_half4(0, gamma + idx);
    float4 y = x * inv_rms * g;
    vstore_half4(y, 0, output + off + idx);
  }
  for (; j2 < hp; ++j2) {
    const int idx = h_start + j2;
    const float x = LOAD(input, off + idx);
    const float g = LOAD(gamma, idx);
    STORE(output, off + idx, x * inv_rms * g);
  }
#else
  for (int j2 = 0; j2 < hp; ++j2) {
    const int idx = h_start + j2;
    const float x = LOAD(input, off + idx);
    const float g = LOAD(gamma, idx);
    STORE(output, off + idx, x * inv_rms * g);
  }
#endif
}
