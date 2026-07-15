// Conv1d kernels: naive fallback + im2col (OPT-7). Owned by src/ops/Conv1d.cpp.
// Split from the former kernels/moonshine.cl monolith (whisper-parity layout);
// kernel bodies are byte-identical to the monolith at the time of the split.
// Dtype-template preamble — DO NOT EDIT. Driven by host-side -DNNOPT_USE_FP16.
// Dtype-template preamble — DO NOT EDIT. Driven by host-side -DNNOPT_USE_FP16.
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

// ── Conv1d ──────────────────────────────────────────────────────────────
// PyTorch Conv1d weight layout [out_channels, in_channels, kernel_size].
// input  : [Cin, Lin]  (row-major, channel-major)
// output : [Cout, Lout] (row-major, channel-major)
// Lout = (Lin - K) / stride + 1  (no padding)
// One work item per (out_channel, out_pos).
__kernel void conv1d(
    __global const storage_t* input,   // [Cin, Lin]
    __global const storage_t* weight,  // [Cout, Cin, K]
    __global const storage_t* bias,    // [Cout] or null (has_bias flag)
    __global storage_t* output,        // [Cout, Lout]
    const int Cin,
    const int Lin,
    const int Cout,
    const int K,
    const int stride,
    const int Lout,
    const int has_bias) {
    int gid = get_global_id(0);
    const int total = Cout * Lout;
    if (gid >= total) return;
    int oc = gid / Lout;
    int op = gid - oc * Lout;
    int in_start = op * stride;
    float acc = 0.0f;
    for (int ic = 0; ic < Cin; ++ic) {
        int w_base = (oc * Cin + ic) * K;
        int x_base = ic * Lin + in_start;
        for (int k = 0; k < K; ++k) {
            acc += LOAD(weight, w_base + k) * LOAD(input, x_base + k);
        }
    }
    if (has_bias) acc += LOAD(bias, oc);
    STORE(output, oc * Lout + op, acc);
}

// ── im2col for 1-D conv (OPT-7): col[l, ic*K+k] = input[ic, l*stride+k] ──
// Turns the conv into a GEMM: col[Lout, Cin*K] @ W[Cout, Cin*K]^T, which
// runs on linear_gemm_t (the naive one-MAC-per-work-item conv kernel above
// managed <1% ALU utilization — 165ms for ~320 MFLOP).
__kernel void im2col_1d(
    __global const storage_t* input,   // [Cin, Lin]
    __global storage_t* col,           // [Lout, Cin*K]
    const int Cin,
    const int Lin,
    const int K,
    const int stride,
    const int Lout) {
    int gid = get_global_id(0);
    const int row = Cin * K;
    const int total = Lout * row;
    if (gid >= total) return;
    int l = gid / row;
    int r = gid - l * row;
    int ic = r / K;
    int k = r - ic * K;
    STORE(col, gid, LOAD(input, ic * Lin + l * stride + k));
}
