// Reference: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/modeling_llama.py LlamaRMSNorm.forward
// (See class LlamaRMSNorm: hidden_states->fp32, variance=pow(2).mean(-1,keepdim=True), hidden_states*=rsqrt(variance+eps), return weight*hidden_states)

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)     vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)     ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
#endif


// ── Work-group reduction: subgroup fast path + portable fallback ────────────
// sub_group_reduce_* needs cl_khr_subgroups, and it is only CORRECT when one
// subgroup spans the whole work-group — which only qcom_reqd_sub_group_size
// ("full") guarantees. MEASURED on PowerVR Rogue GE8320 (Vivo Y21), whose driver
// has neither: every program using these builtins failed to build with
//   "candidate unavailable as it requires OpenCL extension 'cl_khr_subgroups'"
// their kernels were therefore absent, the dispatches were skipped, and the model
// emitted confident garbage at an impossible 11 tok/s. The host compile-probes
// BOTH the builtin and the attribute and defines NNOPT_SUBGROUP_REDUCE=1 only
// when both genuinely work (see opencl_context.cpp).
#if NNOPT_SUBGROUP_REDUCE
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define NNOPT_WAVE_ATTR __attribute__((qcom_reqd_sub_group_size("full")))
#define NNOPT_DECL_SCRATCH(name, n)
#define NNOPT_REDUCE_ADD(name, lid, n, v) sub_group_reduce_add(v)
#define NNOPT_REDUCE_MAX(name, lid, n, v) sub_group_reduce_max(v)
#else
#define NNOPT_WAVE_ATTR
#define NNOPT_DECL_SCRATCH(name, n) __local float name[n]
// Barrier tree reduce over the whole work-group. Safe at every call site here:
// they are all lane-uniform (early-outs branch on get_group_id(), which is
// uniform), so every lane reaches every barrier.
static inline float nnopt_wg_reduce_add(__local float* s, int lid, int n, float v) {
  s[lid] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = n >> 1; off > 0; off >>= 1) {
    if (lid < off) s[lid] += s[lid + off];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float r = s[0];
  barrier(CLK_LOCAL_MEM_FENCE);   // scratch reusable after this point
  return r;
}
static inline float nnopt_wg_reduce_max(__local float* s, int lid, int n, float v) {
  s[lid] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = n >> 1; off > 0; off >>= 1) {
    if (lid < off) s[lid] = fmax(s[lid], s[lid + off]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float r = s[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return r;
}
#define NNOPT_REDUCE_ADD(name, lid, n, v) nnopt_wg_reduce_add(name, lid, n, v)
#define NNOPT_REDUCE_MAX(name, lid, n, v) nnopt_wg_reduce_max(name, lid, n, v)
#endif
#define WG_SIZE 64

// Fused residual-add + RMSNorm. Reads x[row, :] and residual[row, :], writes
// (x + residual) back into x in-place (so callers reuse x as the running
// residual stream), and writes rmsnorm(x + residual) into out. One workgroup
// per row.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
NNOPT_WAVE_ATTR
void rms_norm_residual_forward(
    __global storage_t* x,             // [rows, cols] — mutated to x + residual
    __global const storage_t* residual,
    __global const storage_t* weight,
    __global storage_t* out,
    const int rows,
    const int cols,
    const float eps) {
  // Scratch for the non-subgroup reduction fallback; expands to
  // nothing when NNOPT_SUBGROUP_REDUCE is on.
  NNOPT_DECL_SCRATCH(nnopt_red, WG_SIZE);
  const int row = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  if (row >= rows) return;
  const int base = row * cols;

  // Pass 1: add residual into x in-place AND accumulate sumsq for that row.
  float ss = 0.0f;
#ifdef USE_FP16
  const int C4 = cols >> 2;
  __global half* xh = (__global half*)x + base;
  __global const half* rh = (__global const half*)residual + base;
  for (int c4 = lid; c4 < C4; c4 += WG_SIZE) {
    float4 xv = vload_half4(c4, xh);
    float4 rv = vload_half4(c4, rh);
    float4 s = xv + rv;
    vstore_half4(s, c4, xh);
    ss += dot(s, s);
  }
  for (int c = (C4 << 2) + lid; c < cols; c += WG_SIZE) {
    float v = (float)LOAD(x, base + c) + (float)LOAD(residual, base + c);
    STORE(x, base + c, v);
    ss += v * v;
  }
#else
  for (int c = lid; c < cols; c += WG_SIZE) {
    float v = (float)LOAD(x, base + c) + (float)LOAD(residual, base + c);
    STORE(x, base + c, v);
    ss += v * v;
  }
#endif

  const float total_ss = NNOPT_REDUCE_ADD(nnopt_red, lid, WG_SIZE, ss);
  const float mean_ss = total_ss / (float)cols;
  const float inv_rms = rsqrt(mean_ss + eps);

#ifdef USE_FP16
  __global const half* wh = (__global const half*)weight;
  __global half* oh = (__global half*)out + base;
  for (int c4 = lid; c4 < C4; c4 += WG_SIZE) {
    float4 xv = vload_half4(c4, xh);
    float4 wv = vload_half4(c4, wh);
    vstore_half4(xv * inv_rms * wv, c4, oh);
  }
  for (int c = (C4 << 2) + lid; c < cols; c += WG_SIZE) {
    float xv = (float)LOAD(x, base + c);
    float wv = (float)LOAD(weight, c);
    STORE(out, base + c, (xv * inv_rms * wv));
  }
#else
  for (int c = lid; c < cols; c += WG_SIZE) {
    float xv = (float)LOAD(x, base + c);
    float wv = (float)LOAD(weight, c);
    STORE(out, base + c, (xv * inv_rms * wv));
  }
#endif
}

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
NNOPT_WAVE_ATTR
void rms_norm_forward(
    __global const storage_t* x,
    __global const storage_t* weight,
    __global storage_t* out,
    const int rows,
    const int cols,
    const float eps) {
  // Scratch for the non-subgroup reduction fallback; expands to
  // nothing when NNOPT_SUBGROUP_REDUCE is on.
  NNOPT_DECL_SCRATCH(nnopt_red, WG_SIZE);
  const int row = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  if (row >= rows) return;

  const int base = row * cols;

  // sumsq
  float ss = 0.0f;
#ifdef USE_FP16
  const int C4 = cols >> 2;
  __global const half* xh = (__global const half*)x + base;
  for (int c4 = lid; c4 < C4; c4 += WG_SIZE) {
    float4 v = vload_half4(c4, xh);
    ss += dot(v, v);
  }
  for (int c = (C4 << 2) + lid; c < cols; c += WG_SIZE) {
    float v = (float)LOAD(x, base + c);
    ss += v * v;
  }
#else
  for (int c = lid; c < cols; c += WG_SIZE) {
    float v = (float)LOAD(x, base + c);
    ss += v * v;
  }
#endif

  const float total_ss = NNOPT_REDUCE_ADD(nnopt_red, lid, WG_SIZE, ss);
  const float mean_ss = total_ss / (float)cols;
  const float inv_rms = rsqrt(mean_ss + eps);

  // affine
#ifdef USE_FP16
  __global const half* wh = (__global const half*)weight;
  __global half* oh = (__global half*)out + base;
  const int C4b = cols >> 2;
  for (int c4 = lid; c4 < C4b; c4 += WG_SIZE) {
    float4 xv = vload_half4(c4, xh);
    float4 wv = vload_half4(c4, wh);
    vstore_half4(xv * inv_rms * wv, c4, oh);
  }
  for (int c = (C4b << 2) + lid; c < cols; c += WG_SIZE) {
    float xv = (float)LOAD(x, base + c);
    float wv = (float)LOAD(weight, c);
    STORE(out, base + c, (xv * inv_rms * wv));
  }
#else
  for (int c = lid; c < cols; c += WG_SIZE) {
    float xv = (float)LOAD(x, base + c);
    float wv = (float)LOAD(weight, c);
    STORE(out, base + c, (xv * inv_rms * wv));
  }
#endif
}
