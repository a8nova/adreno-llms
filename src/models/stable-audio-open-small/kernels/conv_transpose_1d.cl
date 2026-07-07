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
// Reference: stable_audio_tools/models/blocks.py WNConvTranspose1d (used by
//   OobleckDecoder DecoderBlock upsample).

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

// conv_transpose_1d_t4x4 — register-tiled transpose conv (dilation==1).
// Key identity: outputs at ol0, ol0+stride, ol0+2*stride, ol0+3*stride share
// the SAME valid kernel taps (same phase), and their gathered inputs are
// CONSECUTIVE (il, il+1, il+2, il+3). Each WI computes 4 oc x 4 same-phase ol
// with the (at most 2) valid taps for that phase: 4x+ arithmetic intensity
// over the scalar gather kernel.
// Dispatch: 2D gws = (stride * ceil(ceil(L_out/stride)/4), ceil(C_out/4)).
__kernel void conv_transpose_1d_t4x4(
    __global const storage_t* in,           // [C_in,  L_in]
    __global const storage_t* weight,       // [C_in,  C_out, K]
    __global const storage_t* bias,         // [C_out]
    __global       storage_t* out,          // [C_out, L_out]
    const int C_in,
    const int C_out,
    const int L_in,
    const int L_out,
    const int K,
    const int stride,
    const int padding,
    const int has_bias) {

    const int blocks_per_phase = ((L_out + stride - 1) / stride + 3) / 4;
    const int g0 = get_global_id(0);
    if (g0 >= stride * blocks_per_phase) return;
    const int ophase = g0 % stride;
    const int oblk   = g0 / stride;
    const int oc0 = get_global_id(1) * 4;
    if (oc0 >= C_out) return;

    // Valid taps for this phase: k with (ophase + padding - k) % stride == 0.
    const int r = (ophase + padding) % stride;      // smallest valid k
    // il_j = oblk*4 + j + d, where d = (ophase + padding - k) / stride.
    const int base = oblk * 4;

    float4 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int wrow = C_out * K;

    for (int tap = 0; tap < 2; ++tap) {
        const int k = r + tap * stride;
        if (k >= K) break;
        const int d = (ophase + padding - k) / stride;
        const int il0 = base + d;
        for (int ic = 0; ic < C_in; ++ic) {
            const int ib = ic * L_in;
            float4 x;
            x.s0 = (il0     >= 0 && il0     < L_in) ? LOAD(in, ib + il0)     : 0.0f;
            x.s1 = (il0 + 1 >= 0 && il0 + 1 < L_in) ? LOAD(in, ib + il0 + 1) : 0.0f;
            x.s2 = (il0 + 2 >= 0 && il0 + 2 < L_in) ? LOAD(in, ib + il0 + 2) : 0.0f;
            x.s3 = (il0 + 3 >= 0 && il0 + 3 < L_in) ? LOAD(in, ib + il0 + 3) : 0.0f;

            const int wb = ic * wrow + k;
            const float w0 = LOAD(weight, wb + (oc0    ) * K);
            const float w1 = (oc0 + 1 < C_out) ? LOAD(weight, wb + (oc0 + 1) * K) : 0.0f;
            const float w2 = (oc0 + 2 < C_out) ? LOAD(weight, wb + (oc0 + 2) * K) : 0.0f;
            const float w3 = (oc0 + 3 < C_out) ? LOAD(weight, wb + (oc0 + 3) * K) : 0.0f;

            acc0 = mad(x, (float4)(w0), acc0);
            acc1 = mad(x, (float4)(w1), acc1);
            acc2 = mad(x, (float4)(w2), acc2);
            acc3 = mad(x, (float4)(w3), acc3);
        }
    }

    if (has_bias) {
        acc0 += LOAD(bias, oc0);
        if (oc0 + 1 < C_out) acc1 += LOAD(bias, oc0 + 1);
        if (oc0 + 2 < C_out) acc2 += LOAD(bias, oc0 + 2);
        if (oc0 + 3 < C_out) acc3 += LOAD(bias, oc0 + 3);
    }

    // Store 4 same-phase outputs: ol_j = (base + j) * stride + ophase.
    #define STORE_ROW_T(oc, accv) do { \
        if ((oc) < C_out) { \
            const size_t ob = (size_t)(oc) * L_out; \
            const float vals[4] = { (accv).s0, (accv).s1, (accv).s2, (accv).s3 }; \
            for (int j = 0; j < 4; ++j) { \
                const int ol = (base + j) * stride + ophase; \
                if (ol < L_out) STORE(out, ob + ol, vals[j]); \
            } \
        } \
    } while (0)
    STORE_ROW_T(oc0,     acc0);
    STORE_ROW_T(oc0 + 1, acc1);
    STORE_ROW_T(oc0 + 2, acc2);
    STORE_ROW_T(oc0 + 3, acc3);
    #undef STORE_ROW_T
}
