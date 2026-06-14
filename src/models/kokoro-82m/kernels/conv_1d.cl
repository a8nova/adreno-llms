// Reference: /Users/alazarshenkute/.nnopt/repos/kokoro/kokoro/istftnet.py: AdaINResBlock1/AdaINResBlk1d and weight_norm Conv1d usage
//
// conv_1d — dilation-aware 1D convolution, NCL layout (batch=1 implicit).
//
// Dtype-template preamble (LOAD/STORE) is REQUIRED for fp16 correctness on Adreno.
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

__kernel void conv_1d(
    __global const storage_t* in,
    __global const storage_t* weight,
    __global const storage_t* bias,        // may be NULL via has_bias flag
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
        for (int k = 0; k < K; ++k) {
            const int il = ol * stride + k * dilation - padding;
            if (il < 0 || il >= L_in) continue;
            const float xv = LOAD(in, ic * L_in + il);
            const float wv = LOAD(weight, oc * (C_in * K) + ic * K + k);
            acc += xv * wv;
        }
    }
    if (has_bias) acc += LOAD(bias, oc);
    STORE(out, gid, acc);
}
