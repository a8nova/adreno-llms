// Reference: model_info/transformers_src/modeling_mamba2.py:103-120 MambaRMSNormGated.forward

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

#ifndef WG_SIZE
#define WG_SIZE 64
#endif

inline float silu_f(float x) {
  return x / (1.0f + exp(-x));
}

// Cooperative + vec4 mamba_rms_norm_gated.
//
// Implements (unchanged math):
//   gated     = x * silu(gate)
//   variance  = mean(gated^2)
//   inv_rms   = rsqrt(variance + eps)
//   y         = weight * gated * inv_rms
//
// 1 WG per row. WG_SIZE=64 threads cooperate over `cols` via __local-mem
// tree reduction. cols/WG_SIZE elements per thread (24 fp16 = 6 vec4 at
// cols=1536). vec4 fp16 loads via vload_half4. Recurrent fp32 sum_sq
// accumulator avoids fp16 saturation across the 1536 sum.
//
// Host dispatch:
//   gws = rows * WG_SIZE, lws = WG_SIZE.
//
// Replaces the 1-thread-per-row scalar 2-pass version that was 40.7% of
// GPU time at Step 0.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void mamba_rms_norm_gated(__global const storage_t* x,
                         __global const storage_t* gate,
                         __global const storage_t* weight,
                         __global storage_t* y,
                         int rows,
                         int cols,
                         float eps) {
  __local float ls[WG_SIZE];
  const int row = get_group_id(0);
  const int tid = get_local_id(0);
  if (row >= rows) return;

  const int off = row * cols;
  const int hp = cols / WG_SIZE;        // 24 fp16 / thread at cols=1536
  const int h_start = tid * hp;

  // Pass 1: sum of squares of gated = (x * silu(gate)).
  float ss = 0.0f;
#ifdef USE_FP16
  int j = 0;
  for (; j + 3 < hp; j += 4) {
    const int idx = h_start + j;
    float4 xv = vload_half4(0, x    + off + idx);
    float4 gv = vload_half4(0, gate + off + idx);
    float4 sg;
    sg.x = silu_f(gv.x);
    sg.y = silu_f(gv.y);
    sg.z = silu_f(gv.z);
    sg.w = silu_f(gv.w);
    float4 gx = xv * sg;
    ss += dot(gx, gx);
  }
  for (; j < hp; ++j) {
    const float xv = (float)LOAD(x, off + h_start + j);
    const float gv = (float)LOAD(gate, off + h_start + j);
    const float gx = xv * silu_f(gv);
    ss += gx * gx;
  }
#else
  for (int jj = 0; jj < hp; ++jj) {
    const float xv = LOAD(x, off + h_start + jj);
    const float gv = LOAD(gate, off + h_start + jj);
    const float gx = xv * silu_f(gv);
    ss += gx * gx;
  }
#endif

  ls[tid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) ls[tid] += ls[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv_rms = rsqrt(ls[0] / (float)cols + eps);

  // Pass 2: y = weight * gated * inv_rms.
#ifdef USE_FP16
  int j2 = 0;
  for (; j2 + 3 < hp; j2 += 4) {
    const int idx = h_start + j2;
    float4 xv = vload_half4(0, x      + off + idx);
    float4 gv = vload_half4(0, gate   + off + idx);
    float4 wv = vload_half4(0, weight + idx);
    float4 sg;
    sg.x = silu_f(gv.x);
    sg.y = silu_f(gv.y);
    sg.z = silu_f(gv.z);
    sg.w = silu_f(gv.w);
    float4 yv = wv * (xv * sg * inv_rms);
    vstore_half4(yv, 0, y + off + idx);
  }
  for (; j2 < hp; ++j2) {
    const int idx = h_start + j2;
    const float xv = (float)LOAD(x, off + idx);
    const float gv = (float)LOAD(gate, off + idx);
    const float wv = (float)LOAD(weight, idx);
    STORE(y, off + idx, wv * (xv * silu_f(gv) * inv_rms));
  }
#else
  for (int jj = 0; jj < hp; ++jj) {
    const int idx = h_start + jj;
    const float xv = LOAD(x, off + idx);
    const float gv = LOAD(gate, off + idx);
    const float wv = LOAD(weight, idx);
    STORE(y, off + idx, wv * (xv * silu_f(gv) * inv_rms));
  }
#endif
}
