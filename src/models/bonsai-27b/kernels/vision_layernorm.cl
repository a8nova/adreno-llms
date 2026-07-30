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


// ── layernorm over last dim D, row-major [rows, D] ──
// one workgroup per row is overkill; use one thread per row (D small=384).

__kernel void layernorm_rows(
    __global const storage_t* input,
    __global const storage_t* gamma,
    __global const storage_t* beta,
    __global storage_t* output,
    const int rows, const int D, const float eps)
{
    int r = get_global_id(0);
    if (r >= rows) return;
    int base = r * D;
    float mean = 0.0f;
    for (int i = 0; i < D; i++) mean += (float)LOAD(input, base + i);
    mean /= (float)D;
    float var = 0.0f;
    for (int i = 0; i < D; i++) {
        float d = (float)LOAD(input, base + i) - mean;
        var += d * d;
    }
    var /= (float)D;
    float inv = rsqrt(var + eps);
    for (int i = 0; i < D; i++) {
        float xn = ((float)LOAD(input, base + i) - mean) * inv;
        float g = (float)LOAD(gamma, i);
        float b = (float)LOAD(beta, i);
        STORE(output, base + i, xn * g + b);
    }
}
