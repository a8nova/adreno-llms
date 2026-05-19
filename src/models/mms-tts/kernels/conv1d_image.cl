// Image2d-based conv1d for HiFi-GAN vocoder.
//
// Adreno 620 has a dedicated texture engine + L1 image cache. Per the
// Qualcomm OpenCL guide §6.2, small read-mostly operands (conv weights here)
// hit ~2-3× higher effective bandwidth as image2d_t versus buffers. We pack
// the weight tensor [C_out, C_in*K] into an RGBA half image of dimensions
// (C_in*K, C_out/4) so each `read_imageh` returns 4 output channels at once.
//
// Each workitem computes 4 output channels for one output time step, so
// the kernel emits (L_out × C_out/4) workitems instead of (L_out × C_out × C_in × K)
// individual MACs per pixel.
//
// Stride is always 1 (the upsample path zero-stuffs first). Dilation and
// padding are runtime parameters so kernel-3 dilated ResBlock convs (dilations
// 1/3/5) and the K=7 conv_pre / K=7 conv_post all share this kernel.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

const sampler_t SMP = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

// One-shot weight packer: copy [C_out, CinK] half buffer into a half-RGBA image
// where pixel(x, y) == { W[4y+0, x], W[4y+1, x], W[4y+2, x], W[4y+3, x] }.
__kernel void pack_weight_h4(
    __global const half* w_buf,           // [C_out, CinK] row-major fp16
    __write_only image2d_t w_img,         // (CinK, ceil(C_out/4)) CL_RGBA half
    const int C_out,
    const int CinK) {

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int oc4_count = (C_out + 3) / 4;
    if (x >= CinK || y >= oc4_count) return;

    const int row0 = y * 4;
    half4 v;
    v.x =                              vload_half((row0 + 0) * CinK + x, w_buf);
    v.y = (row0 + 1 < C_out) ?         vload_half((row0 + 1) * CinK + x, w_buf) : (half)0.0h;
    v.z = (row0 + 2 < C_out) ?         vload_half((row0 + 2) * CinK + x, w_buf) : (half)0.0h;
    v.w = (row0 + 3 < C_out) ?         vload_half((row0 + 3) * CinK + x, w_buf) : (half)0.0h;
    write_imageh(w_img, (int2)(x, y), v);
}

// Fused conv1d (stride=1) using image-resident weights.
//   global_size = (L_out, ceil(C_out / 4))
// Each workitem computes 4 output channels at one output position.
//
// `leaky_in` is applied to the input value before MAC (HiFi-GAN slope=0.1).
// `has_bias` adds the per-channel bias.
// `has_resid` adds an elementwise residual at the end (used for ResBlock final).
__kernel void conv1d_image_h4(
    __global const half* in,              // [C_in, L_in]
    __read_only image2d_t w_img,          // (C_in*K, ceil(C_out/4)) RGBA half
    __global const half* bias,            // [C_out]; ignored if has_bias=0
    __global const half* resid,           // [C_out, L_out]; ignored if has_resid=0
    __global half* out,                   // [C_out, L_out]
    const int C_in,
    const int C_out,
    const int L_in,
    const int L_out,
    const int K,
    const int padding,
    const int dilation,
    const int has_bias,
    const int has_resid,
    const int leaky_in) {

    const int ol  = get_global_id(0);
    const int oc4 = get_global_id(1);
    if (ol >= L_out) return;
    const int oc = oc4 * 4;
    if (oc >= C_out) return;

    float4 acc = (float4)(0.0f);

    for (int k = 0; k < K; ++k) {
        const int src = ol + k * dilation - padding;
        if (src < 0 || src >= L_in) continue;
        for (int ic = 0; ic < C_in; ++ic) {
            float xv = (float)vload_half(ic * L_in + src, in);
            if (leaky_in && xv < 0.0f) xv *= 0.1f;
            const int wx = ic * K + k;
            half4 wv = read_imageh(w_img, SMP, (int2)(wx, oc4));
            acc.x += xv * (float)wv.x;
            acc.y += xv * (float)wv.y;
            acc.z += xv * (float)wv.z;
            acc.w += xv * (float)wv.w;
        }
    }

    if (has_bias) {
        acc.x +=                       (float)vload_half(oc + 0, bias);
        if (oc + 1 < C_out) acc.y +=  (float)vload_half(oc + 1, bias);
        if (oc + 2 < C_out) acc.z +=  (float)vload_half(oc + 2, bias);
        if (oc + 3 < C_out) acc.w +=  (float)vload_half(oc + 3, bias);
    }
    if (has_resid) {
        acc.x +=                       (float)vload_half((oc + 0) * L_out + ol, resid);
        if (oc + 1 < C_out) acc.y +=  (float)vload_half((oc + 1) * L_out + ol, resid);
        if (oc + 2 < C_out) acc.z +=  (float)vload_half((oc + 2) * L_out + ol, resid);
        if (oc + 3 < C_out) acc.w +=  (float)vload_half((oc + 3) * L_out + ol, resid);
    }

    vstore_half(acc.x, (oc + 0) * L_out + ol, out);
    if (oc + 1 < C_out) vstore_half(acc.y, (oc + 1) * L_out + ol, out);
    if (oc + 2 < C_out) vstore_half(acc.z, (oc + 2) * L_out + ol, out);
    if (oc + 3 < C_out) vstore_half(acc.w, (oc + 3) * L_out + ol, out);
}

// ─── Tiled variant — 4 output channels × 8 output timesteps per workitem ───
//
// Each workitem holds 32 fp32 accumulators (4×8) in registers. For every
// (ic, k) we issue ONE image fetch (4 OCs worth of weight as half4) and reuse
// it across all 8 timesteps. That's 8× fewer image fetches than the naive
// kernel and brings work granularity per workitem up high enough to hide
// dispatch/setup, mimicking what CLBlast HGemm does for the GEMM path.
//
// Global size: (ceil(L_out / 8), ceil(C_out / 4))
//
// Stride is always 1 (the upsample path zero-stuffs first). Dilation, padding,
// has_bias, has_resid, leaky_in are runtime parameters so the same kernel
// covers every conv shape in the vocoder.

#define NNOPT_TN 4

__kernel void conv1d_image_tiled_h4(
    __global const half* in,
    __read_only image2d_t w_img,
    __global const half* bias,
    __global const half* resid,
    __global half* out,
    const int C_in,
    const int C_out,
    const int L_in,
    const int L_out,
    const int K,
    const int padding,
    const int dilation,
    const int has_bias,
    const int has_resid,
    const int leaky_in) {

    const int ol_base = get_global_id(0) * NNOPT_TN;
    const int oc4     = get_global_id(1);
    const int oc      = oc4 * 4;
    if (oc >= C_out) return;
    if (ol_base >= L_out) return;

    // 32 register-resident accumulators.
    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);
    float4 acc4 = (float4)(0.0f);
    float4 acc5 = (float4)(0.0f);
    float4 acc6 = (float4)(0.0f);
    float4 acc7 = (float4)(0.0f);

    for (int k = 0; k < K; ++k) {
        // For fixed k, src positions for the 8 outputs are consecutive:
        // src_t = (ol_base + t) + k*dilation - padding.
        const int src_base = ol_base + k * dilation - padding;
        for (int ic = 0; ic < C_in; ++ic) {
            const int wx = ic * K + k;
            const float4 wvf = convert_float4(read_imageh(w_img, SMP, (int2)(wx, oc4)));

            const int row_off = ic * L_in;
            // Branch-free path when whole tile is in-bounds. Otherwise per-t check.
            if (src_base >= 0 && src_base + NNOPT_TN <= L_in) {
                float x0 = (float)vload_half(row_off + src_base + 0, in);
                float x1 = (float)vload_half(row_off + src_base + 1, in);
                float x2 = (float)vload_half(row_off + src_base + 2, in);
                float x3 = (float)vload_half(row_off + src_base + 3, in);
                float x4 = (float)vload_half(row_off + src_base + 4, in);
                float x5 = (float)vload_half(row_off + src_base + 5, in);
                float x6 = (float)vload_half(row_off + src_base + 6, in);
                float x7 = (float)vload_half(row_off + src_base + 7, in);
                if (leaky_in) {
                    if (x0 < 0.0f) x0 *= 0.1f;
                    if (x1 < 0.0f) x1 *= 0.1f;
                    if (x2 < 0.0f) x2 *= 0.1f;
                    if (x3 < 0.0f) x3 *= 0.1f;
                    if (x4 < 0.0f) x4 *= 0.1f;
                    if (x5 < 0.0f) x5 *= 0.1f;
                    if (x6 < 0.0f) x6 *= 0.1f;
                    if (x7 < 0.0f) x7 *= 0.1f;
                }
                acc0 += x0 * wvf;
                acc1 += x1 * wvf;
                acc2 += x2 * wvf;
                acc3 += x3 * wvf;
                acc4 += x4 * wvf;
                acc5 += x5 * wvf;
                acc6 += x6 * wvf;
                acc7 += x7 * wvf;
            } else {
                #define NNOPT_LOAD_X(t, accv) {                              \
                    const int s = src_base + (t);                             \
                    float xv = 0.0f;                                          \
                    if (s >= 0 && s < L_in) {                                 \
                        xv = (float)vload_half(row_off + s, in);              \
                        if (leaky_in && xv < 0.0f) xv *= 0.1f;                \
                    }                                                         \
                    accv += xv * wvf;                                         \
                }
                NNOPT_LOAD_X(0, acc0);
                NNOPT_LOAD_X(1, acc1);
                NNOPT_LOAD_X(2, acc2);
                NNOPT_LOAD_X(3, acc3);
                NNOPT_LOAD_X(4, acc4);
                NNOPT_LOAD_X(5, acc5);
                NNOPT_LOAD_X(6, acc6);
                NNOPT_LOAD_X(7, acc7);
                #undef NNOPT_LOAD_X
            }
        }
    }

    // Load bias once (shared across all 8 timesteps).
    float4 biasv = (float4)(0.0f);
    if (has_bias) {
        biasv.x =                       (float)vload_half(oc + 0, bias);
        if (oc + 1 < C_out) biasv.y =  (float)vload_half(oc + 1, bias);
        if (oc + 2 < C_out) biasv.z =  (float)vload_half(oc + 2, bias);
        if (oc + 3 < C_out) biasv.w =  (float)vload_half(oc + 3, bias);
    }

    #define NNOPT_STORE_T(t, accv) {                                          \
        const int ol = ol_base + (t);                                          \
        if (ol < L_out) {                                                      \
            float4 v = accv + biasv;                                           \
            if (has_resid) {                                                   \
                v.x +=                       (float)vload_half((oc + 0) * L_out + ol, resid); \
                if (oc + 1 < C_out) v.y +=  (float)vload_half((oc + 1) * L_out + ol, resid); \
                if (oc + 2 < C_out) v.z +=  (float)vload_half((oc + 2) * L_out + ol, resid); \
                if (oc + 3 < C_out) v.w +=  (float)vload_half((oc + 3) * L_out + ol, resid); \
            }                                                                  \
            vstore_half(v.x, (oc + 0) * L_out + ol, out);                      \
            if (oc + 1 < C_out) vstore_half(v.y, (oc + 1) * L_out + ol, out);  \
            if (oc + 2 < C_out) vstore_half(v.z, (oc + 2) * L_out + ol, out);  \
            if (oc + 3 < C_out) vstore_half(v.w, (oc + 3) * L_out + ol, out);  \
        }                                                                      \
    }
    NNOPT_STORE_T(0, acc0);
    NNOPT_STORE_T(1, acc1);
    NNOPT_STORE_T(2, acc2);
    NNOPT_STORE_T(3, acc3);
    NNOPT_STORE_T(4, acc4);
    NNOPT_STORE_T(5, acc5);
    NNOPT_STORE_T(6, acc6);
    NNOPT_STORE_T(7, acc7);
    #undef NNOPT_STORE_T
}
