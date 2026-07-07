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


// ── conv_transpose2d: NCHW transposed conv, batch=1, stride==kernel, pad=0 ──
// Used by reassemble resize with factor>1: kernel_size=factor, stride=factor.
// input:  [Cin, Hin, Win]
// weight: [Cin, Cout, KH, KW]   (PyTorch ConvTranspose2d layout)
// bias:   [Cout]
// output: [Cout, Hout, Wout]  with Hout=(Hin-1)*stride+KH, Wout=(Win-1)*stride+KW
// one thread per output element; gather formulation.

__kernel void conv_transpose2d(
    __global const storage_t* input,
    __global const storage_t* weight,
    __global const storage_t* bias,
    __global storage_t* output,
    const int Cin, const int Hin, const int Win,
    const int Cout, const int KH, const int KW,
    const int stride,
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

    // output pixel (oh,ow) receives from input (ih,iw) where
    //   oh = ih*stride + kh  =>  kh = oh - ih*stride, need 0<=kh<KH
    for (int ic = 0; ic < Cin; ic++) {
        int in_base = ic * Hin * Win;
        int w_base = ((ic * Cout) + oc) * KH * KW;
        for (int kh = 0; kh < KH; kh++) {
            int ih_num = oh - kh;
            if (ih_num < 0 || (ih_num % stride) != 0) continue;
            int ih = ih_num / stride;
            if (ih < 0 || ih >= Hin) continue;
            for (int kw = 0; kw < KW; kw++) {
                int iw_num = ow - kw;
                if (iw_num < 0 || (iw_num % stride) != 0) continue;
                int iw = iw_num / stride;
                if (iw < 0 || iw >= Win) continue;
                float iv = (float)LOAD(input, in_base + ih * Win + iw);
                float wv = (float)LOAD(weight, w_base + kh * KW + kw);
                acc += iv * wv;
            }
        }
    }
    STORE(output, idx, acc);
}

// ── conv_transpose2d_sk: stride==kernel, pad=0 specialization ──
// With S==KH==KW and no padding, output (oh,ow) receives from EXACTLY ONE
// input pixel and ONE weight tap: ih=oh/S, kh=oh%S, iw=ow/S, kw=ow%S.
// The generic gather kernel walked Cin*KH*KW with per-iteration div/mod
// guards — 16× more iterations than needed at K=4 (745 ms/frame for the
// two reassemble-resize calls). This is a plain Cin-dot per output.
// Each thread computes 4 consecutive ow (same oh → same kh, contiguous iw
// pattern) to amortize the weight walk.

__kernel void conv_transpose2d_sk(
    __global const storage_t* input,
    __global const storage_t* weight,
    __global const storage_t* bias,
    __global storage_t* output,
    const int Cin, const int Hin, const int Win,
    const int Cout, const int S,
    const int Hout, const int Wout,
    const int has_bias)
{
    const int idx = get_global_id(0);              // over Cout*Hout*ceil(Wout/4)
    const int wq = (Wout + 3) / 4;
    const int total = Cout * Hout * wq;
    if (idx >= total) return;

    const int ow0 = (idx % wq) * 4;
    const int oh  = (idx / wq) % Hout;
    const int oc  = idx / (wq * Hout);

    const int ih = oh / S, kh = oh % S;
    const float b = has_bias ? (float)LOAD(bias, oc) : 0.0f;
    float acc0 = b, acc1 = b, acc2 = b, acc3 = b;

    const int iw0 = (ow0    ) / S, kw0 = (ow0    ) % S;
    const int iw1 = (ow0 + 1) / S, kw1 = (ow0 + 1) % S;
    const int iw2 = (ow0 + 2) / S, kw2 = (ow0 + 2) % S;
    const int iw3 = (ow0 + 3) / S, kw3 = (ow0 + 3) % S;

    for (int ic = 0; ic < Cin; ic++) {
        const int in_row = (ic * Hin + ih) * Win;
        const int w_row  = (((ic * Cout) + oc) * S + kh) * S;
        const float i0 = (float)LOAD(input, in_row + iw0);
        const float i1 = (float)LOAD(input, in_row + iw1);
        const float i2 = (float)LOAD(input, in_row + iw2);
        const float i3 = (float)LOAD(input, in_row + iw3);
        acc0 += i0 * (float)LOAD(weight, w_row + kw0);
        acc1 += i1 * (float)LOAD(weight, w_row + kw1);
        acc2 += i2 * (float)LOAD(weight, w_row + kw2);
        acc3 += i3 * (float)LOAD(weight, w_row + kw3);
    }

    const int out_base = (oc * Hout + oh) * Wout + ow0;
    STORE(output, out_base, acc0);
    if (ow0 + 1 < Wout) STORE(output, out_base + 1, acc1);
    if (ow0 + 2 < Wout) STORE(output, out_base + 2, acc2);
    if (ow0 + 3 < Wout) STORE(output, out_base + 3, acc3);
}
