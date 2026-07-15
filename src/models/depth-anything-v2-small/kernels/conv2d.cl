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


// ── conv2d: general NCHW convolution, batch=1 ──
// input:  [Cin, Hin, Win]  (row-major)
// weight: [Cout, Cin, KH, KW]
// bias:   [Cout] or null (use has_bias flag)
// output: [Cout, Hout, Wout]
// one thread per output element.

__kernel void conv2d(
    __global const storage_t* input,
    __global const storage_t* weight,
    __global const storage_t* bias,
    __global storage_t* output,
    const int Cin, const int Hin, const int Win,
    const int Cout, const int KH, const int KW,
    const int stride, const int pad,
    const int Hout, const int Wout,
    const int has_bias)
{
    int idx = get_global_id(0);
    int total = Cout * Hout * Wout;
    if (idx >= total) return;

    int ow = idx % Wout;
    int oh = (idx / Wout) % Hout;
    int oc = idx / (Wout * Hout);

    float acc = has_bias ? (float)LOAD(bias, oc) : 0.0f;

    for (int ic = 0; ic < Cin; ic++) {
        int in_base = ic * Hin * Win;
        int w_base = ((oc * Cin) + ic) * KH * KW;
        for (int kh = 0; kh < KH; kh++) {
            int ih = oh * stride - pad + kh;
            if (ih < 0 || ih >= Hin) continue;
            for (int kw = 0; kw < KW; kw++) {
                int iw = ow * stride - pad + kw;
                if (iw < 0 || iw >= Win) continue;
                float iv = (float)LOAD(input, in_base + ih * Win + iw);
                float wv = (float)LOAD(weight, w_base + kh * KW + kw);
                acc += iv * wv;
            }
        }
    }
    STORE(output, idx, acc);
}
