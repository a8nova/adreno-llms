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

// Reference: /Users/alazarshenkute/.nnopt/ref_venvs/env_stable/lib/python3.12/site-packages/torch/nn/modules/conv.py:439-460 Conv2d.forward
// im2col for NCHW input.
// Produces col[M, K] where:
//   M = N * Hout * Wout
//   K = Cin * Kh * Kw
// Global size: (M, K)

__kernel void im2col_nchw(
    __global const storage_t* input, // [N, Cin, Hin, Win]
    __global storage_t* col,         // [M, K]
    const int N,
    const int Cin,
    const int Hin,
    const int Win,
    const int Hout,
    const int Wout,
    const int Kh,
    const int Kw,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w) {
  const int m = (int)get_global_id(0);
  const int k = (int)get_global_id(1);

  const int M = N * Hout * Wout;
  const int K = Cin * Kh * Kw;
  if (m >= M || k >= K) return;

  const int n = m / (Hout * Wout);
  const int rem = m - n * (Hout * Wout);
  const int out_y = rem / Wout;
  const int out_x = rem - out_y * Wout;

  const int c = k / (Kh * Kw);
  const int krem = k - c * (Kh * Kw);
  const int ky = krem / Kw;
  const int kx = krem - ky * Kw;

  const int in_y = out_y * stride_h + ky - pad_h;
  const int in_x = out_x * stride_w + kx - pad_w;

  float v = 0.0f;
  if (in_y >= 0 && in_y < Hin && in_x >= 0 && in_x < Win) {
    const long in_idx = (((long)n * (long)Cin + (long)c) * (long)Hin + (long)in_y) * (long)Win + (long)in_x;
    v = (float)LOAD(input, in_idx);
  }

  const long col_idx = (long)m * (long)K + (long)k;
  STORE(col, col_idx, (storage_t)v);
}
