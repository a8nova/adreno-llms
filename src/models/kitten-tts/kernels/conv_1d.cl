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

__kernel void conv_1d(
    __global const nnopt_storage_t* in,
    __global const nnopt_storage_t* weight,
    __global const nnopt_storage_t* bias,        // may be NULL via has_bias flag
    __global       nnopt_storage_t* out,
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
            const float xv = (float)in[ic * L_in + il];
            const float wv = (float)weight[oc * (C_in * K) + ic * K + k];
            acc += xv * wv;
        }
    }
    if (has_bias) acc += (float)bias[oc];
    out[gid] = (nnopt_storage_t)acc;
}
