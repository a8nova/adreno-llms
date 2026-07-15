// Depth-Anything-V2 (DINOv2 backbone + DPT neck/head) OpenCL kernels.
// One FILE = one cl_program on purpose: Adreno applies a program-wide
// worst-case register footprint, so a fat kernel in a shared program cuts
// wave occupancy for every kernel in it (measured 2026-07-07: frame
// 6.6s -> 10.5s from two never-dispatched kernels). Keep op families
// isolated. All data buffers are storage_t (fp16 or fp32); accumulators
// are float.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)   vload_half((i),(p))
  #define STORE(p,i,v) vstore_half((v),(i),(p))
  #define LOAD4(p,i)   vload_half4(0, (p) + (i))
  #define STORE4(p,i,v) vstore_half4((v), 0, (p) + (i))
#else
  typedef float storage_t;
  #define LOAD(p,i)   ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
  #define LOAD4(p,i)   vload4(0, (p) + (i))
  #define STORE4(p,i,v) vstore4((v), 0, (p) + (i))
#endif


// ── gelu (exact erf) in-place-style: out[i]=gelu(in[i]) ──

__kernel void gelu(__global const storage_t* input, __global storage_t* output, const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = (float)LOAD(input, i);
    float y = 0.5f * x * (1.0f + erf(x * 0.70710678118654752440f));
    STORE(output, i, y);
}

// ── relu ──

__kernel void relu(__global const storage_t* input, __global storage_t* output, const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = (float)LOAD(input, i);
    STORE(output, i, x > 0.0f ? x : 0.0f);
}

// ── layer_scale: out[r,d] = in[r,d] * lambda[d]  (row-major [rows,D]) ──

__kernel void layer_scale(
    __global const storage_t* input,
    __global const storage_t* lambda,
    __global storage_t* output,
    const int rows, const int D)
{
    int idx = get_global_id(0);
    int total = rows * D;
    if (idx >= total) return;
    int d = idx % D;
    float v = (float)LOAD(input, idx) * (float)LOAD(lambda, d);
    STORE(output, idx, v);
}

// ── add_bias_rows: out[r,d] = in[r,d] + bias[d] (for linear layers done via GEMM) ──

__kernel void add_bias_rows(
    __global storage_t* buf,
    __global const storage_t* bias,
    const int rows, const int D)
{
    int idx = get_global_id(0);
    int total = rows * D;
    if (idx >= total) return;
    int d = idx % D;
    float v = (float)LOAD(buf, idx) + (float)LOAD(bias, d);
    STORE(buf, idx, v);
}

// ── add: out[i] = a[i] + b[i] (residual connections) ──
__kernel void add(__global const storage_t* a, __global const storage_t* b,
                  __global storage_t* out, const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    STORE(out, i, (float)LOAD(a, i) + (float)LOAD(b, i));
}
