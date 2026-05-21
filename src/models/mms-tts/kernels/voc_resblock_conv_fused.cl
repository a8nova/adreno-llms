// voc_resblock_conv_fused.cl — single-dispatch fused 1D conv for HiFi-GAN
// ResBlocks in the MMS-TTS vocoder. Replaces the 2-dispatch sequence
// (im2col_1d_fused_bias + CLBlast HGemm) with one custom kernel, mirroring
// the flow_wn_fused.cl template that brought flow_inverse from 6.79s → 2.75s.
//
// Why this exists
// ---------------
// Vocoder wall is 95% driver QUEUED→SUBMIT gap (75-226ms per dispatch on
// Adreno 620). 144 resblock conv calls × 2 dispatches each = 288 dispatches.
// Collapsing each conv to 1 dispatch saves ~144 × 75ms = ~10.8s of pure
// scheduling overhead. CLBlast HGemm is FAST per kernel (the 5% GPU-busy
// figure includes its compute), but the gap dominates. The fused kernel
// only needs to match CLBlast's compute, not beat it.
//
// Per-call work
// -------------
//   acc[c, t] = bias[c] + sum_k sum_ci w[c, k, ci] * leaky_in_opt(h[ci, t - pad + k*d])
//   if has_resid: acc[c, t] += resid[c, t]
//   h_out[c, t] = acc[c, t]
//
// Layout (channels-first, B=1):
//   h_in       : [C, L]        global, read-only
//   w          : [C, K, C]     reordered from [C_out, C_in, K] → [C_out, K, C_in]
//                              (one-shot host transpose, cached)
//   b          : [C]
//   resid      : [C, L]        global, read-only (ignored when has_resid=0)
//   h_out      : [C, L]        global, write
//
// Workgroup model:
//   gws = (C_padded, L)       local = (C_padded, 1)
//   One WG per output time-step. WI index c (within WG) handles channel c.
//   We round C up to the next multiple of WG_C_QUANTUM to keep the
//   cooperative load lockstep; out-of-range WIs early-return after the load.
//
// Local memory budget:
//   h_tile = K * C floats = at most 11 * 192 * 4 = 8.4KB on Adreno (32KB cap).
//   acts/scratch = 0.
//
// Notes
// -----
// * Adreno's vload_half4 returns float4 (hardware does the half→float on
//   load), so accumulation stays in fp32. Inner ci-loop is 4-wide.
// * leaky_in is applied AT LOAD TIME — a leaky_relu_in followed by conv is
//   exactly conv(leaky(h)), so we pre-leaky the local tile once and reuse
//   it across K kernel positions. This is the same trick im2col_1d_fused
//   used pre-fusion.

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

// MMS-TTS UPSAMPLE_INITIAL_CHANNEL=512 → 256 after stage-1 upsample → 128 → 64 → 32.
// The pre-upsample conv_pre at C=512 does NOT go through this fused path (it
// runs once outside the resblock loop). So max C entering resblocks is 256.
#ifndef VOC_C_MAX
#define VOC_C_MAX 256
#endif
#ifndef VOC_K_MAX
#define VOC_K_MAX 11
#endif

// LeakyReLU slope used everywhere in the vocoder.
#define VOC_LEAKY_SLOPE 0.1f

__kernel void voc_conv1d_resblock_fused(
    __global const storage_t* h_in,     // [C, L]
    __global const storage_t* w,        // [C, K, C] (reordered ic↔k for stride-1 ci load)
    __global const storage_t* b,        // [C]
    __global const storage_t* resid,    // [C, L] (or dummy when has_resid=0)
    __global storage_t*       h_out,    // [C, L]
    const int C,
    const int L,
    const int K,
    const int dilation,
    const int pad,
    const int leaky_in,
    const int has_resid)
{
    const int c = get_local_id(0);
    const int t = get_group_id(1);

    // Out-of-range WIs participate in the local load barrier (they own no
    // channel rows so the load loop below skips them) but skip compute.
    const bool active = (c < C);

    // ── Step 1: cooperative load of the K-wide input tile into __local memory.
    //   Each active WI loads K timesteps for its own channel row ci=c.
    //   h_tile layout = [k][ci] row-major (ci stride-1 — matches reordered w).
    //   leaky_in is applied here so downstream MACs see pre-activation values.
    __local float h_tile[VOC_K_MAX * VOC_C_MAX];
    if (active) {
        #pragma unroll
        for (int k = 0; k < VOC_K_MAX; ++k) {
            if (k >= K) break;
            const int tk = t + k * dilation - pad;
            float v = 0.0f;
            if (tk >= 0 && tk < L) {
                v = (float)LOAD(h_in, c * L + tk);
                if (leaky_in) v = v < 0.0f ? VOC_LEAKY_SLOPE * v : v;
            }
            h_tile[k * C + c] = v;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (!active) return;

    // ── Step 2: convolve. w[c, k, ci] indexes as c*K*C + k*C + ci.
    //   Inner ci loop is 4-wide via vload_half4 (returns float4) → dot(float4).
    //   C is always a multiple of 4 in the vocoder (192/96/48/24).
    float acc = (float)LOAD(b, c);
    #pragma unroll
    for (int k = 0; k < VOC_K_MAX; ++k) {
        if (k >= K) break;
        const int h_off = k * C;
        const int w_off = c * K * C + k * C;
#ifdef USE_FP16
        for (int ci4 = 0; ci4 < C; ci4 += 4) {
            const float4 wv4 = vload_half4(0, (__global const half*)w + w_off + ci4);
            const float4 hv4 = (float4)(h_tile[h_off + ci4 + 0],
                                        h_tile[h_off + ci4 + 1],
                                        h_tile[h_off + ci4 + 2],
                                        h_tile[h_off + ci4 + 3]);
            acc += dot(wv4, hv4);
        }
#else
        for (int ci = 0; ci < C; ++ci) {
            acc += (float)LOAD(w, w_off + ci) * h_tile[h_off + ci];
        }
#endif
    }

    if (has_resid) {
        acc += (float)LOAD(resid, c * L + t);
    }

    STORE(h_out, c * L + t, acc);
}
