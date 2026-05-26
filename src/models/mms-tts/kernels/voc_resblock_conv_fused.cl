// voc_resblock_conv_fused.cl — tiled fused 1D conv for HiFi-GAN resblocks.
// Replaces im2col + CLBlast HGEMM with a single dispatch that computes
// im2col on-the-fly inside a tiled GEMM loop.
//
// The key insight: conv1d(h, w, K, dil, pad) = GEMM(w_flat, im2col(h))
// where im2col(h)[ci*K + k, t] = h[ci, t + k*dil - pad].
// Instead of materializing im2col in global memory, we gather h values
// on-the-fly inside the GEMM tile loop.
//
// Tiling: each workgroup computes a TILE_CO × TILE_T tile of output.
// The reduction dimension is CI*K (input channels × kernel width).
// We tile over CI in chunks of TILE_CI, and unroll over K.
//
// For C=256, K=11: reduction dim = 256*11 = 2816. TILE_CI=16 → 176 tiles.
// Each tile loads TILE_CI × K values from h (with leaky ReLU on load)
// and TILE_CO × TILE_CI × K values from w. All from global mem —
// no __local needed for small tile sizes.

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

#define VOC_LEAKY_SLOPE 0.1f

// Each workitem computes one output element (co, t).
// The inner loop walks over ci in steps of 4 (NEON-style vectorization
// via vload_half4), accumulating K taps per ci.
// Weight layout: w[co, ci, K] row-major (standard PyTorch layout).
__kernel void voc_conv1d_resblock_fused(
    __global const storage_t* h_in,     // [C, L]
    __global const storage_t* w,        // [C_out, C_in, K] (standard layout)
    __global const storage_t* b,        // [C_out]
    __global const storage_t* resid,    // [C, L] (or dummy)
    __global storage_t*       h_out,    // [C, L]
    const int C,
    const int L,
    const int K,
    const int dilation,
    const int pad,
    const int leaky_in,
    const int has_resid)
{
    const int co = get_global_id(0);
    const int t  = get_global_id(1);
    if (co >= C || t >= L) return;

    float acc = (float)LOAD(b, co);

    // w offset for this output channel: w[co, :, :] starts at co * C * K
    const int w_base = co * C * K;

    // Inner loop: iterate over input channels and kernel positions.
    // For each (ci, k): acc += w[co, ci, k] * leaky(h[ci, t + k*dil - pad])
    // Unroll K in the inner loop since K ∈ {3, 7, 11}.
    for (int ci = 0; ci < C; ++ci) {
        const int w_ci = w_base + ci * K;
        const int h_ci_base = ci * L;
        for (int k = 0; k < K; ++k) {
            const int tt = t + k * dilation - pad;
            float hv = 0.0f;
            if (tt >= 0 && tt < L) {
                hv = (float)LOAD(h_in, h_ci_base + tt);
                if (leaky_in) hv = hv < 0.0f ? VOC_LEAKY_SLOPE * hv : hv;
            }
            acc += (float)LOAD(w, w_ci + k) * hv;
        }
    }

    if (has_resid) {
        acc += (float)LOAD(resid, co * L + t);
    }

    STORE(h_out, co * L + t, acc);
}
