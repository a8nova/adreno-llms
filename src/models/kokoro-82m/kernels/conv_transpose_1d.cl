// Reference: /Users/alazarshenkute/.nnopt/repos/kokoro/kokoro/istftnet.py: Generator.ups ConvTranspose1d usage
//
// conv_transpose_1d — transposed (fractionally-strided) 1D convolution.
//
// Used by iSTFTNet upsampling blocks.
// PyTorch semantics (output_padding=0, groups=1):
//   L_out = (L_in - 1) * stride - 2*padding + dilation*(K - 1) + 1
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

__kernel void conv_transpose_1d(
    __global const storage_t* in,           // [C_in,  L_in]
    __global const storage_t* weight,       // [C_in,  C_out, K]   (transposed-conv layout)
    __global const storage_t* bias,         // [C_out] or NULL
    __global       storage_t* out,          // [C_out, L_out]
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
    for (int k = 0; k < K; ++k) {
        const int num = ol + padding - k * dilation;
        if (num < 0) continue;
        if ((num % stride) != 0) continue;
        const int il = num / stride;
        if (il < 0 || il >= L_in) continue;
        for (int ic = 0; ic < C_in; ++ic) {
            const float xv = LOAD(in, ic * L_in + il);
            // PyTorch's ConvTranspose1d weight is [C_in, C_out, K]
            const float wv = LOAD(weight, ic * (C_out * K) + oc * K + k);
            acc += xv * wv;
        }
    }
    if (has_bias) acc += LOAD(bias, oc);
    STORE(out, gid, acc);
}
