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


// ── im2col for conv-as-GEMM (optimization campaign 2026-07-05).
// Writes col[K, ncols] for output positions p = n0 .. n0+ncols-1, where
// K = Cin*KH*KW. Zero-fills out-of-bounds taps (padding). One WI per col
// element; consecutive gids walk consecutive output positions (coalesced
// stores, coalesced input rows for fixed kidx).

__kernel void im2col2d(
    __global const storage_t* input,   // [Cin, Hin, Win]
    __global storage_t* col,           // [K, ncols]
    const int Cin, const int Hin, const int Win,
    const int KH, const int KW,
    const int stride, const int pad,
    const int Wout,
    const int n0, const int ncols)
{
    int gid = get_global_id(0);
    int K = Cin * KH * KW;
    if (gid >= K * ncols) return;
    int j    = gid % ncols;
    int kidx = gid / ncols;
    int c  = kidx / (KH * KW);
    int r  = kidx % (KH * KW);
    int kh = r / KW;
    int kw = r % KW;
    int p  = n0 + j;
    int oy = p / Wout;
    int ox = p % Wout;
    int iy = oy * stride - pad + kh;
    int ix = ox * stride - pad + kw;
    float v = 0.0f;
    if (iy >= 0 && iy < Hin && ix >= 0 && ix < Win)
        v = LOAD(input, c * (Hin * Win) + iy * Win + ix);
    STORE(col, gid, v);
}

// ── im2col2d_v4: 4 output positions per WI (campaign lever 4).
// The scalar kernel spent most of its time on per-element address math
// (790 ms/frame across 26 calls). Fixed kidx + 4 consecutive j share all
// the kidx decomposition; when the 4 taps sit in one input row, in-bounds,
// stride==1, they collapse to a single vload4. Stores are one vstore4
// (col rows are ncols-contiguous). Tail (ncols%4) handled by scalar guard.

__kernel void im2col2d_v4(
    __global const storage_t* input,   // [Cin, Hin, Win]
    __global storage_t* col,           // [K, ncols]
    const int Cin, const int Hin, const int Win,
    const int KH, const int KW,
    const int stride, const int pad,
    const int Wout,
    const int n0, const int ncols)
{
    const int gid = get_global_id(0);
    const int jq  = (ncols + 3) / 4;
    const int K   = Cin * KH * KW;
    if (gid >= K * jq) return;

    const int j0   = (gid % jq) * 4;
    const int kidx = gid / jq;
    const int c  = kidx / (KH * KW);
    const int r  = kidx % (KH * KW);
    const int kh = r / KW;
    const int kw = r % KW;

    const int p0 = n0 + j0;
    const int oy0 = p0 / Wout, ox0 = p0 % Wout;
    const int iy0 = oy0 * stride - pad + kh;
    const int ix0 = ox0 * stride - pad + kw;
    const int in_row = c * (Hin * Win) + iy0 * Win;
    const int out_base = kidx * ncols + j0;

    // Fast path: 4 full lanes, one output row, stride 1, all taps interior.
    if (stride == 1 && j0 + 3 < ncols && ox0 + 3 < Wout &&
        iy0 >= 0 && iy0 < Hin && ix0 >= 0 && ix0 + 3 < Win) {
        float4 v = LOAD4(input, in_row + ix0);
        STORE4(col, out_base, v);
        return;
    }

    // Scalar path (edges, padding, stride>1, tail).
    for (int l = 0; l < 4; l++) {
        const int j = j0 + l;
        if (j >= ncols) return;
        const int p  = n0 + j;
        const int oy = p / Wout;
        const int ox = p % Wout;
        const int iy = oy * stride - pad + kh;
        const int ix = ox * stride - pad + kw;
        float v = 0.0f;
        if (iy >= 0 && iy < Hin && ix >= 0 && ix < Win)
            v = LOAD(input, c * (Hin * Win) + iy * Win + ix);
        STORE(col, kidx * ncols + j, v);
    }
}

// ── Per-channel bias add for GEMM-shaped conv output [Cout, HW].

__kernel void bias_per_row(
    __global storage_t* out,          // [Cout, HW]
    __global const storage_t* bias,   // [Cout]
    const int HW, const int total)
{
    int gid = get_global_id(0);
    if (gid >= total) return;
    int row = gid / HW;
    STORE(out, gid, LOAD(out, gid) + LOAD(bias, row));
}
