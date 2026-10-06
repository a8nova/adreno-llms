// conv_transpose_1d — transposed (fractionally-strided) 1D convolution.
//
// Used by HiFi-GAN upsampling blocks in VITS / Kokoro / Piper vocoders.
// PyTorch semantics (output_padding=0, groups=1):
//
//   L_out = (L_in - 1) * stride - 2*padding + dilation*(K - 1) + 1
//
// Implementation: each output position 'ol' "gathers" the input positions
// 'il' whose forward-conv kernel taps would have landed on ol. Concretely:
//
//   il*stride - padding + k*dilation == ol      (forward sense)
// ⇒ il = (ol + padding - k*dilation) / stride   (integer; must be exact)
//
// We loop over k, check the integer-divisibility condition, accumulate.
// Layout matches conv_1d for consistency.
//
// One work-item per output element. C_out * L_out total.

__kernel void conv_transpose_1d(
    __global const nnopt_storage_t* in,           // [C_in,  L_in]
    __global const nnopt_storage_t* weight,       // [C_in,  C_out, K]   (transposed-conv layout)
    __global const nnopt_storage_t* bias,         // [C_out] or NULL
    __global       nnopt_storage_t* out,          // [C_out, L_out]
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
            const float xv = (float)in[ic * L_in + il];
            // PyTorch's ConvTranspose1d weight is [C_in, C_out, K]
            const float wv = (float)weight[ic * (C_out * K) + oc * K + k];
            acc += xv * wv;
        }
    }
    if (has_bias) acc += (float)bias[oc];
    out[gid] = (nnopt_storage_t)acc;
}
