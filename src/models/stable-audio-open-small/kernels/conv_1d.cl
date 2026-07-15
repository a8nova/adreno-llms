// conv_1d — dilation-aware 1D convolution, NCL layout (batch=1 implicit).
// Reference: stable_audio_tools/models/blocks.py WNConv1d (nn.Conv1d) — used by
//   ResidualUnit + OobleckDecoder in the pretransform autoencoder decode path.
//
// Inputs:
//   in       — [C_in,  L_in]   storage_t
//   weight   — [C_out, C_in, K] storage_t   (groups=1)
//   bias     — [C_out]          storage_t   (may be null via has_bias flag)
// Output:
//   out      — [C_out, L_out]  storage_t,  L_out = (L_in + 2*pad - dilation*(K-1) - 1)/stride + 1
//
// One work-item per output element (C_out * L_out). fp32 accumulator.

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
    __global const storage_t* bias,
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

// conv_1d_t4x4 — register-tiled conv for stride==1 (all decoder resunit convs).
// Each work-item computes a 4 output-channel x 4 output-position tile:
// 16 fp32 accumulators, 8 loads per (ic,k) instead of 32 -> 4x arithmetic
// intensity over the naive kernel. Kokoro conv1d_f32_t4x4 lineage.
// Dispatch: 2D gws = (ceil(L_out/4), ceil(C_out/4)); host falls back to
// conv_1d when stride != 1 or NNOPT_CONV_T4X4=0.
__kernel void conv_1d_t4x4(
    __global const storage_t* in,      // [C_in, L_in]
    __global const storage_t* weight,  // [C_out, C_in, K]
    __global const storage_t* bias,    // [C_out]
    __global       storage_t* out,     // [C_out, L_out]
    const int C_in,
    const int C_out,
    const int L_in,
    const int L_out,
    const int K,
    const int padding,
    const int dilation,
    const int has_bias) {

    const int ol0 = get_global_id(0) * 4;
    const int oc0 = get_global_id(1) * 4;
    if (ol0 >= L_out || oc0 >= C_out) return;

    float4 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int oc1 = oc0 + 1, oc2 = oc0 + 2, oc3 = oc0 + 3;
    const int wstride = C_in * K;

    for (int ic = 0; ic < C_in; ++ic) {
        const int in_base = ic * L_in;
        const int w_base  = ic * K;
        for (int k = 0; k < K; ++k) {
            const int il = ol0 + k * dilation - padding;
            // 4 consecutive input positions (stride==1). Bounds-guarded loads;
            // interior tiles never take the zero branch.
            float4 x;
            x.s0 = (il     >= 0 && il     < L_in) ? LOAD(in, in_base + il)     : 0.0f;
            x.s1 = (il + 1 >= 0 && il + 1 < L_in) ? LOAD(in, in_base + il + 1) : 0.0f;
            x.s2 = (il + 2 >= 0 && il + 2 < L_in) ? LOAD(in, in_base + il + 2) : 0.0f;
            x.s3 = (il + 3 >= 0 && il + 3 < L_in) ? LOAD(in, in_base + il + 3) : 0.0f;

            const int wi = w_base + k;
            const float w0 = LOAD(weight, (size_t)oc0 * wstride + wi);
            const float w1 = (oc1 < C_out) ? LOAD(weight, (size_t)oc1 * wstride + wi) : 0.0f;
            const float w2 = (oc2 < C_out) ? LOAD(weight, (size_t)oc2 * wstride + wi) : 0.0f;
            const float w3 = (oc3 < C_out) ? LOAD(weight, (size_t)oc3 * wstride + wi) : 0.0f;

            acc0 = mad(x, (float4)(w0), acc0);
            acc1 = mad(x, (float4)(w1), acc1);
            acc2 = mad(x, (float4)(w2), acc2);
            acc3 = mad(x, (float4)(w3), acc3);
        }
    }

    if (has_bias) {
        acc0 += LOAD(bias, oc0);
        if (oc1 < C_out) acc1 += LOAD(bias, oc1);
        if (oc2 < C_out) acc2 += LOAD(bias, oc2);
        if (oc3 < C_out) acc3 += LOAD(bias, oc3);
    }

    const int lend = min(4, L_out - ol0);
    #define STORE_ROW(oc, accv) do { \
        if ((oc) < C_out) { \
            const size_t ob = (size_t)(oc) * L_out + ol0; \
            if (lend > 0) STORE(out, ob + 0, (accv).s0); \
            if (lend > 1) STORE(out, ob + 1, (accv).s1); \
            if (lend > 2) STORE(out, ob + 2, (accv).s2); \
            if (lend > 3) STORE(out, ob + 3, (accv).s3); \
        } \
    } while (0)
    STORE_ROW(oc0, acc0);
    STORE_ROW(oc1, acc1);
    STORE_ROW(oc2, acc2);
    STORE_ROW(oc3, acc3);
    #undef STORE_ROW
}


#ifdef USE_FP16
  #define WLOAD4(p, off4) vload_half4((off4), (p))
#else
  #define WLOAD4(p, off4) vload4((off4), (p))
#endif

// conv_1d_t4x4v2 — v2 of the register-tiled conv (stride==1):
//   * weight REPACKED oc-tile-blocked [Cout/4][Cin*K][4]: sequential vec4
//     stream per work-item (one cache line feeds 8 iterations)
//   * interior fast path: tiles whose taps are fully in-bounds skip all
//     per-element bounds checks (only edge tiles branch)
// Requires C_out % 4 == 0 (host guarantees; falls back to v1 otherwise).
__kernel void conv_1d_t4x4v2(
    __global const storage_t* in,      // [C_in, L_in]
    __global const storage_t* weight,  // [C_out/4][C_in*K][4]  (oc-tile-blocked)
    __global const storage_t* bias,    // [C_out]
    __global       storage_t* out,     // [C_out, L_out]
    const int C_in,
    const int C_out,
    const int L_in,
    const int L_out,
    const int K,
    const int padding,
    const int dilation,
    const int has_bias) {

    const int ol0 = get_global_id(0) * 4;
    const int oc0 = get_global_id(1) * 4;
    if (ol0 >= L_out || oc0 >= C_out) return;

    float4 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int il_lo = ol0 - padding;                          // k=0 tap
    const int il_hi = ol0 + 3 + (K - 1) * dilation - padding; // last tap, last lane
    const int interior = (il_lo >= 0) && (il_hi < L_in);
    const int ickn = C_in * K;
    const int wtile = (oc0 >> 2) * ickn;   // vec4 base index for this oc-tile

    if (interior) {
        for (int ic = 0; ic < C_in; ++ic) {
            const int in_base = ic * L_in;
            const int wrow0 = ic * K;
            for (int k = 0; k < K; ++k) {
                const int il = ol0 + k * dilation - padding;
                const float4 x = (float4)(LOAD(in, in_base + il),
                                          LOAD(in, in_base + il + 1),
                                          LOAD(in, in_base + il + 2),
                                          LOAD(in, in_base + il + 3));
                const float4 wv = convert_float4(WLOAD4(weight, wtile + wrow0 + k));
                acc0 = mad(x, (float4)(wv.s0), acc0);
                acc1 = mad(x, (float4)(wv.s1), acc1);
                acc2 = mad(x, (float4)(wv.s2), acc2);
                acc3 = mad(x, (float4)(wv.s3), acc3);
            }
        }
    } else {
        for (int ic = 0; ic < C_in; ++ic) {
            const int in_base = ic * L_in;
            const int wrow0 = ic * K;
            for (int k = 0; k < K; ++k) {
                const int il = ol0 + k * dilation - padding;
                float4 x;
                x.s0 = (il     >= 0 && il     < L_in) ? LOAD(in, in_base + il)     : 0.0f;
                x.s1 = (il + 1 >= 0 && il + 1 < L_in) ? LOAD(in, in_base + il + 1) : 0.0f;
                x.s2 = (il + 2 >= 0 && il + 2 < L_in) ? LOAD(in, in_base + il + 2) : 0.0f;
                x.s3 = (il + 3 >= 0 && il + 3 < L_in) ? LOAD(in, in_base + il + 3) : 0.0f;
                const float4 wv = convert_float4(WLOAD4(weight, wtile + wrow0 + k));
                acc0 = mad(x, (float4)(wv.s0), acc0);
                acc1 = mad(x, (float4)(wv.s1), acc1);
                acc2 = mad(x, (float4)(wv.s2), acc2);
                acc3 = mad(x, (float4)(wv.s3), acc3);
            }
        }
    }

    if (has_bias) {
        acc0 += LOAD(bias, oc0);
        acc1 += LOAD(bias, oc0 + 1);
        acc2 += LOAD(bias, oc0 + 2);
        acc3 += LOAD(bias, oc0 + 3);
    }
    const int lend = min(4, L_out - ol0);
    #define V2_STORE_ROW(oc, accv) do { \
        const size_t ob = (size_t)(oc) * L_out + ol0; \
        if (lend > 0) STORE(out, ob + 0, (accv).s0); \
        if (lend > 1) STORE(out, ob + 1, (accv).s1); \
        if (lend > 2) STORE(out, ob + 2, (accv).s2); \
        if (lend > 3) STORE(out, ob + 3, (accv).s3); \
    } while (0)
    V2_STORE_ROW(oc0,     acc0);
    V2_STORE_ROW(oc0 + 1, acc1);
    V2_STORE_ROW(oc0 + 2, acc2);
    V2_STORE_ROW(oc0 + 3, acc3);
    #undef V2_STORE_ROW
}


// im2col_1d — unroll a length-chunk of [C_in, L_in] into col [C_in*K, Lc]
// for the GEMM conv path (stride==1). Chunking (l0, Lc) lets long stages run
// through GEMM with a bounded scratch instead of bailing to the slower tiled
// kernel: col[(ic*K + k) * Lc + (ol - l0)] = in[ic, ol + k*dil - pad]
// (zero outside), ol in [l0, l0+Lc). Unchunked call: l0=0, Lc=L_out.
// One WI per 4 consecutive chunk-local outputs of one (ic,k) row.
__kernel void im2col_1d(
    __global const storage_t* in,   // [C_in, L_in]
    __global       storage_t* col,  // [C_in*K, Lc]
    const int C_in, const int L_in, const int L_out,
    const int K, const int padding, const int dilation,
    const int l0, const int Lc) {
    const int row = get_global_id(1);              // ic*K + k
    if (row >= C_in * K) return;
    const int o0 = get_global_id(0) * 4;           // chunk-local
    if (o0 >= Lc) return;
    const int ic = row / K;
    const int k  = row % K;
    const int in_base = ic * L_in;
    const int d = k * dilation - padding;
    const size_t ob = (size_t)row * Lc;
    const int lend = min(min(4, Lc - o0), L_out - l0 - o0);
    for (int j = 0; j < lend; j++) {
        const int il = l0 + o0 + j + d;
        const float v = (il >= 0 && il < L_in) ? LOAD(in, in_base + il) : 0.0f;
        STORE(col, ob + o0 + j, v);
    }
}

// B7: 8-wide im2col — interior (in-bounds, full-lane) chunks move as one
// 128-bit load + store; borders fall back to the guarded scalar loop.
__kernel void im2col_1d_v8(
    __global const storage_t* in,   // [C_in, L_in]
    __global       storage_t* col,  // [C_in*K, Lc]
    const int C_in, const int L_in, const int L_out,
    const int K, const int padding, const int dilation,
    const int l0, const int Lc) {
    const int row = get_global_id(1);              // ic*K + k
    if (row >= C_in * K) return;
    const int o0 = get_global_id(0) * 8;           // chunk-local
    if (o0 >= Lc) return;
    const int ic = row / K;
    const int k  = row % K;
    const int in_base = ic * L_in;
    const int d = k * dilation - padding;
    const size_t ob = (size_t)row * Lc;
    const int il0 = l0 + o0 + d;
    const int lend = min(min(8, Lc - o0), L_out - l0 - o0);
#ifdef USE_FP16
    if (lend == 8 && il0 >= 0 && il0 + 8 <= L_in) {
        vstore_half8(vload_half8(0, in + in_base + il0), 0, col + ob + o0);
        return;
    }
#else
    if (lend == 8 && il0 >= 0 && il0 + 8 <= L_in) {
        vstore8(vload8(0, in + in_base + il0), 0, col + ob + o0);
        return;
    }
#endif
    for (int j = 0; j < lend; j++) {
        const int il = il0 + j;
        const float v = (il >= 0 && il < L_in) ? LOAD(in, in_base + il) : 0.0f;
        STORE(col, ob + o0 + j, v);
    }
}
