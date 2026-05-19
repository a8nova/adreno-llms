// Reference: model_info/transformers_src/modeling_vits.py (see uses of torch.nn.Conv1d in
// VitsWaveNet, VitsFeedForward, VitsDurationPredictor, VitsPosteriorEncoder, VitsHifiGan)
//
// conv_1d — dilation-aware 1D convolution, NCL layout (batch=1 implicit).
//
// Inputs:
//   in       — [C_in,  L_in]   storage_t
//   weight   — [C_out, C_in, K] storage_t   (groups=1; see conv_1d_grouped for groups>1)
//   bias     — [C_out]          storage_t   (NULL-able; if null, no bias)
// Output:
//   out      — [C_out, L_out]  storage_t,  L_out = (L_in + 2*pad - dilation*(K-1) - 1)/stride + 1
//
// One work-item per output element (C_out * L_out total). Naive reduction
// over C_in * K. Adequate for first-port correctness; OptimizeKernel can
// later swap in a tiled/vectorized variant.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)   vload_half((i),(p))
  #define STORE(p,i,v) vstore_half((v),(i),(p))
#else
  typedef float storage_t;
  #define LOAD(p,i)   ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
#endif

// Step C.c: bias promoted to __constant memory (Adreno guide §6.4 p.46),
// opencl_unroll_hint on the K loop (§8.4 p.65). conv_1d is used for conv_pre
// (192→512 K=7) and conv_post (32→1 K=7) — both have small fixed K, ideal for
// unrolling. The IC loop is variable so left to the compiler's heuristic.
__kernel void conv_1d(
    __global const storage_t* in,
    __global const storage_t* weight,
    __constant     storage_t* bias         // [C_out] — may be NULL via has_bias flag
        __attribute__((max_constant_size(2048))),
    __global       storage_t* out,
    const int C_in,
    const int C_out,
    const int L_in,
    const int L_out,
    const int K,
    const int stride,
    const int padding,
    const int dilation,
    const int has_bias) {

    const int gid = get_global_id(0);
    const int total = C_out * L_out;
    if (gid >= total) return;

    const int oc = gid / L_out;
    const int ol = gid % L_out;

    float acc = 0.0f;
    for (int ic = 0; ic < C_in; ++ic) {
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            const int il = ol * stride + k * dilation - padding;
            if (il < 0 || il >= L_in) continue;
            const float xv = (float)LOAD(in, ic * L_in + il);
            const float wv = (float)LOAD(weight, oc * (C_in * K) + ic * K + k);
            acc += xv * wv;
        }
    }
    if (has_bias) acc += (float)LOAD(bias, oc);
    STORE(out, gid, acc);
}
