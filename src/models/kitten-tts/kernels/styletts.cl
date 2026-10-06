// Shared StyleTTS2 primitives for kitten-tts-nano-0.1 (ONNX int8-dynamic source).
//
// These are the portable ops the whole model is built from — every remaining
// stage (acoustic TextEncoder, DurationEncoder, F0/N predictor, decoder
// encode/decode, HiFiGAN generator) is a composition of the kernels below.
//
// INVARIANTS ENCODED HERE (each was verified against reference/layers/ oracle
// dumps; see PORT_JOURNAL.md):
//   * Converted `*_quantized` weights are ALREADY dequantized floats. No kernel
//     here applies a weight scale.
//   * DynamicQuantizeLinear is part of the MODEL SEMANTICS, not an optimization.
//     st_dql_scale + st_dql_apply replicate it exactly (round-half-to-even,
//     per-tensor, xmin/xmax clamped against 0, scale computed in fp32).
//   * InstanceNormalization uses BIASED variance (/T) and eps=1e-5, with LEARNED
//     norm.weight/norm.bias, then AdaIN combines as (1+gamma)*n + beta.
//   * All reductions accumulate in fp32 even under fp16 storage — per-channel
//     sums of squares reach ~1.2e7, far over fp16's 65504 ceiling.
//
// Layout convention: activations are channel-first [C, T] row-major, so channel
// c occupies the contiguous span [c*T, (c+1)*T). Concat on axis=1 is therefore a
// contiguous block copy (done host-side with clEnqueueCopyBuffer, no kernel).

// Dtype-template preamble — DO NOT EDIT. Driven by host-side -DNNOPT_USE_FP16.
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

#define ST_LRELU_ALPHA 0.20000000298023224f
#define ST_IN_EPS      9.999999747378752e-06f
#define ST_INV_SQRT2   0.7071067690849304f

// ─────────────────────────────────────────────────────────────────────────
// DynamicQuantizeLinear
// ─────────────────────────────────────────────────────────────────────────
// Pass 1: single work-group reduces the whole tensor to (scale, zero_point).
// params is a float2 buffer (fp32 ALWAYS — never storage_t; the scale is ~1e-3
// and quantizing it to fp16 would itself corrupt the result).
//   xmin = min(0, min x) ; xmax = max(0, max x)
//   scale = (xmax - xmin) / 255            (computed in float)
//   zp    = clamp(rint(-xmin / scale), 0, 255)   [round-half-to-EVEN]
// Launch with ONE work-group: global == local == ST_RED (256).
#define ST_RED 256
__kernel void st_dql_scale(
    __global const storage_t* x,
    __global float*           params,   // [0]=scale, [1]=zero_point
    const int                 n) {
    __local float lmin[ST_RED];
    __local float lmax[ST_RED];
    int lid = get_local_id(0);
    float mn = 0.0f, mx = 0.0f;          // clamped against 0 per ONNX spec
    for (int i = lid; i < n; i += ST_RED) {
        float v = LOAD(x, i);
        mn = fmin(mn, v);
        mx = fmax(mx, v);
    }
    lmin[lid] = mn; lmax[lid] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = ST_RED / 2; s > 0; s >>= 1) {
        if (lid < s) {
            lmin[lid] = fmin(lmin[lid], lmin[lid + s]);
            lmax[lid] = fmax(lmax[lid], lmax[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        float xmin = lmin[0], xmax = lmax[0];
        float scale = (xmax - xmin) / 255.0f;
        float zp = 0.0f;
        if (scale > 0.0f) {
            zp = rint(-xmin / scale);          // rint == round-half-to-even
            zp = clamp(zp, 0.0f, 255.0f);
        }
        params[0] = scale;
        params[1] = zp;
    }
}

// Pass 2: quantize-then-dequantize in one elementwise sweep.
//   q   = clamp(rint(x/scale) + zp, 0, 255)
//   out = (q - zp) * scale
// scale == 0 (constant tensor) passes zeros through, matching ORT.
__kernel void st_dql_apply(
    __global const storage_t* x,
    __global const float*     params,
    __global storage_t*       out,
    const int                 n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float scale = params[0];
    if (scale <= 0.0f) { STORE(out, i, 0.0f); return; }
    float zp = params[1];
    float q = rint((float)LOAD(x, i) / scale) + zp;
    q = clamp(q, 0.0f, 255.0f);
    STORE(out, i, (q - zp) * scale);
}

// ─────────────────────────────────────────────────────────────────────────
// Conv1d — general: stride, symmetric pad, groups (group==Cin==Cout ⇒ depthwise)
// in [Cin, T_in]; w [Cout, Cin/groups, K]; bias [Cout] (may be null); out [Cout, T_out]
// T_out = (T_in + 2*pad - K)/stride + 1
// ─────────────────────────────────────────────────────────────────────────
__kernel void st_conv1d(
    __global const storage_t* in,
    __global const storage_t* w,
    __global const storage_t* bias,
    __global storage_t*       out,
    const int Cin, const int Cout, const int T_in, const int T_out,
    const int K, const int stride, const int pad, const int groups,
    const int has_bias, const int dilation) {
    int gid = get_global_id(0);
    if (gid >= Cout * T_out) return;
    int co = gid / T_out;
    int t  = gid - co * T_out;

    int cin_per_g = Cin / groups;
    int co_per_g  = Cout / groups;
    int g         = co / co_per_g;
    int ci_base   = g * cin_per_g;

    float acc = has_bias ? (float)LOAD(bias, co) : 0.0f;
    int t0 = t * stride - pad;
    for (int ci = 0; ci < cin_per_g; ci++) {
        int wbase = (co * cin_per_g + ci) * K;
        int ibase = (ci_base + ci) * T_in;
        for (int k = 0; k < K; k++) {
            int ti = t0 + k * dilation;
            if (ti < 0 || ti >= T_in) continue;   // zero pad
            acc += (float)LOAD(w, wbase + k) * (float)LOAD(in, ibase + ti);
        }
    }
    STORE(out, gid, acc);
}

// ─────────────────────────────────────────────────────────────────────────
// Conv1d — TILED, groups==1 fast path (OPT #1, razr Adreno620v2).
// Each workgroup computes a [ST_LTIME output-times × ST_CO_TILE output-chans]
// block. Input channels are looped in ST_CIN_CHUNK chunks; each chunk's input
// time-span for this workgroup's ST_LTIME outputs is cooperatively staged into
// LDS ONCE and reused across all ST_CO_TILE output channels. This kills the
// naive kernel's Cout× global re-read of the input (the 190× excess-traffic).
//
// Guide grounding (Qualcomm 80-NB295-11 Rev C):
//   §7.1.2 windowed-reuse: WIs access the same data >2× with overlap → stage in
//     LDS, every WI participates in the cooperative load, one barrier, no
//     async_work_group_copy.
//   LDS tile is fp32 (guide §7.2.4 hybrid: load-as-fp16 / accumulate-fp32).
//   Reductions/accums stay fp32 (journal: conv accums reach 5e5, over fp16 max).
// Constraints: groups==1; host guarantees span ≤ ST_SPAN_MAX. Dispatch 2-D with
//   local = (ST_LTIME, ST_CO_TILE); grid dims stay < 65535 (razr wg-id wrap).
// Each WI computes ST_TREG consecutive output TIMES for 1 output channel, so each
// weight load w[co,ci,k] is reused across ST_TREG outputs (weight global traffic
// /ST_TREG) and each staged input across ST_CO_TILE channels. ST_TREG=4 = the
// Adreno accumulator sweet spot (repo: >4 accs regress). tile column for the r-th
// output time of WI lt, tap k: j = (lt*ST_TREG+r)*stride + k*dilation ∈ [0,span).
#define ST_LTIME     16      // WIs along time; each does ST_TREG outputs
#define ST_TREG      4
#define ST_CO_TILE   16      // input tile reused across 16 output channels
#define ST_CIN_CHUNK 16      // LDS tile = 16*128*4 = 8 KB → up to 4 workgroups/SP
#define ST_SPAN_MAX  128     // covers stride-1 K≤11 dil≤5; host routes larger spans to naive
__kernel void st_conv1d_tiled(
    __global const storage_t* in,
    __global const storage_t* w,      // [Cout, Cin, K]  (groups==1)
    __global const storage_t* bias,
    __global storage_t*       out,
    const int Cin, const int Cout, const int T_in, const int T_out,
    const int K, const int stride, const int pad, const int has_bias,
    const int dilation) {
    __local float tile[ST_CIN_CHUNK][ST_SPAN_MAX];
    const int lt  = get_local_id(0);            // 0..ST_LTIME-1
    const int lco = get_local_id(1);            // 0..ST_CO_TILE-1
    const int co  = get_global_id(1);           // output channel
    const int t_base = get_group_id(0) * (ST_LTIME * ST_TREG);
    const int t0_wi  = t_base + lt * ST_TREG;   // first of this WI's ST_TREG outputs
    const int in_t0  = t_base * stride - pad;    // input time mapped to tile[*][0]
    const int span   = (ST_LTIME * ST_TREG - 1) * stride + (K - 1) * dilation + 1;
    const int nthreads = ST_LTIME * ST_CO_TILE;
    const int tid = lco * ST_LTIME + lt;
    const int co_active = (co < Cout);

    float bias_v = (has_bias && co_active) ? (float)LOAD(bias, co) : 0.0f;
    float acc0 = bias_v, acc1 = bias_v, acc2 = bias_v, acc3 = bias_v;

    for (int ci0 = 0; ci0 < Cin; ci0 += ST_CIN_CHUNK) {
        // Cooperative strided load of tile[cc][j] = in[ci0+cc, in_t0+j] (zero-pad OOB).
        // (Vectorizing this load MEASURED slower — conv1d's compute vload4 dominates,
        // and the interior/edge branch adds divergence. Kept scalar.)
        const int total = ST_CIN_CHUNK * span;
        for (int idx = tid; idx < total; idx += nthreads) {
            int cc = idx / span;
            int j  = idx - cc * span;
            int ci = ci0 + cc;
            int ti = in_t0 + j;
            float v = 0.0f;
            if (ci < Cin && ti >= 0 && ti < T_in) v = (float)LOAD(in, ci * T_in + ti);
            tile[cc][j] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (co_active) {
            int nci = min(ST_CIN_CHUNK, Cin - ci0);
            if (stride == 1) {
                // The 4 time outputs read 4 CONSECUTIVE LDS elements → one vload4
                // instead of 4 scalar LDS loads (this kernel is LDS-load bound).
                int base0 = lt * ST_TREG;
                for (int cc = 0; cc < nci; cc++) {
                    int wbase = (co * Cin + (ci0 + cc)) * K;
                    __local const float* trow = tile[cc];
                    for (int k = 0; k < K; k++) {
                        float wv = (float)LOAD(w, wbase + k);
                        float4 tv = vload4(0, trow + base0 + k * dilation);
                        acc0 = mad(wv, tv.x, acc0);
                        acc1 = mad(wv, tv.y, acc1);
                        acc2 = mad(wv, tv.z, acc2);
                        acc3 = mad(wv, tv.w, acc3);
                    }
                }
            } else {
                int base0 = (lt * ST_TREG) * stride;   // tile col for r=0,k=0
                for (int cc = 0; cc < nci; cc++) {
                    int wbase = (co * Cin + (ci0 + cc)) * K;
                    __local const float* trow = tile[cc];
                    for (int k = 0; k < K; k++) {
                        float wv = (float)LOAD(w, wbase + k);
                        int j = base0 + k * dilation;
                        acc0 = mad(wv, trow[j],              acc0);
                        acc1 = mad(wv, trow[j + stride],     acc1);
                        acc2 = mad(wv, trow[j + 2 * stride], acc2);
                        acc3 = mad(wv, trow[j + 3 * stride], acc3);
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (co_active) {
        int ob = co * T_out;
        if (t0_wi     < T_out) STORE(out, ob + t0_wi,     acc0);
        if (t0_wi + 1 < T_out) STORE(out, ob + t0_wi + 1, acc1);
        if (t0_wi + 2 < T_out) STORE(out, ob + t0_wi + 2, acc2);
        if (t0_wi + 3 < T_out) STORE(out, ob + t0_wi + 3, acc3);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// ConvTranspose1d, DEPTHWISE (groups == Cin == Cout), stride 2, pad 1,
// output_padding 1, K=3 ⇒ T_out == 2*T_in exactly.
// w [C, 1, K]; bias [C].
// Accumulation form (verified vs oracle, max err 4.6e-3 vs the fp16 reference):
//   out[c,o] = bias[c] + Σ_k w[c,0,k] * x[c, (o+pad-k)/stride]
//              for (o+pad-k) divisible by stride and index in range
// ─────────────────────────────────────────────────────────────────────────
__kernel void st_convtranspose1d_dw(
    __global const storage_t* in,
    __global const storage_t* w,
    __global const storage_t* bias,
    __global storage_t*       out,
    const int C, const int T_in, const int T_out,
    const int K, const int stride, const int pad, const int has_bias) {
    int gid = get_global_id(0);
    if (gid >= C * T_out) return;
    int c = gid / T_out;
    int o = gid - c * T_out;
    float acc = has_bias ? (float)LOAD(bias, c) : 0.0f;
    for (int k = 0; k < K; k++) {
        int num = o + pad - k;
        if (num < 0) continue;
        if (num % stride != 0) continue;
        int ti = num / stride;
        if (ti < 0 || ti >= T_in) continue;
        acc += (float)LOAD(w, c * K + k) * (float)LOAD(in, c * T_in + ti);
    }
    STORE(out, gid, acc);
}

// ─────────────────────────────────────────────────────────────────────────
// InstanceNormalization over T for each channel, with learned affine.
// BIASED variance. fp32 accumulation (sums reach ~1.2e7).
// in/out [C, T]; norm_w/norm_b [C]. One work-item per channel.
// ─────────────────────────────────────────────────────────────────────────
__kernel void st_instancenorm(
    __global const storage_t* in,
    __global const storage_t* norm_w,
    __global const storage_t* norm_b,
    __global storage_t*       out,
    const int C, const int T) {
    int c = get_global_id(0);
    if (c >= C) return;
    int base = c * T;
    float mean = 0.0f;
    for (int t = 0; t < T; t++) mean += (float)LOAD(in, base + t);
    mean /= (float)T;
    float var = 0.0f;
    for (int t = 0; t < T; t++) {
        float d = (float)LOAD(in, base + t) - mean;
        var += d * d;
    }
    var /= (float)T;                      // BIASED — /T, not /(T-1)
    float inv = rsqrt(var + ST_IN_EPS);
    float gw = LOAD(norm_w, c);
    float gb = LOAD(norm_b, c);
    for (int t = 0; t < T; t++) {
        float v = ((float)LOAD(in, base + t) - mean) * inv;
        STORE(out, base + t, v * gw + gb);
    }
}

// InstanceNorm — OPT #7: ONE WORKGROUP per channel (was 1 work-item per channel,
// = catastrophic occupancy: C=64 → 64 WIs total). IN_WG work-items cooperatively
// reduce over T via LDS tree, keeping the VALIDATED two-pass numerics (mean, then
// centered biased variance) and fp32 accumulation (sums reach ~1.2e7). Dispatch
// global=C*IN_WG, local=IN_WG → C workgroups, IN_WG waves of work each.
#define IN_WG 128
__kernel void st_instancenorm_wg(
    __global const storage_t* in,
    __global const storage_t* norm_w,
    __global const storage_t* norm_b,
    __global storage_t*       out,
    const int C, const int T) {
    __local float red[IN_WG];
    const int c   = get_group_id(0);
    const int lid = get_local_id(0);
    if (c >= C) return;
    const int base = c * T;

    // Pass 1: mean.
    float s = 0.0f;
    for (int t = lid; t < T; t += IN_WG) s += (float)LOAD(in, base + t);
    red[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = IN_WG / 2; st > 0; st >>= 1) {
        if (lid < st) red[lid] += red[lid + st];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = red[0] / (float)T;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Pass 2: centered biased variance.
    float v = 0.0f;
    for (int t = lid; t < T; t += IN_WG) { float d = (float)LOAD(in, base + t) - mean; v += d * d; }
    red[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = IN_WG / 2; st > 0; st >>= 1) {
        if (lid < st) red[lid] += red[lid + st];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = rsqrt(red[0] / (float)T + ST_IN_EPS);

    // Pass 3: normalize + affine.
    float gw = LOAD(norm_w, c), gb = LOAD(norm_b, c);
    for (int t = lid; t < T; t += IN_WG)
        STORE(out, base + t, ((float)LOAD(in, base + t) - mean) * inv * gw + gb);
}

// AdaIN combine: out[c,t] = (1 + gamma[c]) * n[c,t] + beta[c]
// gamma/beta are slices of the style fc output: fc_out[0:C] and fc_out[C:2C].
__kernel void st_adain_combine(
    __global const storage_t* n,
    __global const storage_t* fc_out,   // [2C]
    __global storage_t*       out,
    const int C, const int T) {
    int gid = get_global_id(0);
    if (gid >= C * T) return;
    int c = gid / T;
    float gamma = LOAD(fc_out, c);
    float beta  = LOAD(fc_out, C + c);
    STORE(out, gid, (1.0f + gamma) * (float)LOAD(n, gid) + beta);
}

// ─────────────────────────────────────────────────────────────────────────
// LayerNormalization over the LAST axis of a [rows, cols] row-major tensor.
// BIASED variance. Used by the acoustic TextEncoder CNN blocks (axis=-1, C=128).
// ─────────────────────────────────────────────────────────────────────────
__kernel void st_layernorm_rows(
    __global const storage_t* in,
    __global const storage_t* gamma,
    __global const storage_t* beta,
    __global storage_t*       out,
    const int rows, const int cols, const float eps) {
    int r = get_global_id(0);
    if (r >= rows) return;
    int base = r * cols;
    float mean = 0.0f;
    for (int i = 0; i < cols; i++) mean += (float)LOAD(in, base + i);
    mean /= (float)cols;
    float var = 0.0f;
    for (int i = 0; i < cols; i++) {
        float d = (float)LOAD(in, base + i) - mean;
        var += d * d;
    }
    var /= (float)cols;
    float inv = rsqrt(var + eps);
    for (int i = 0; i < cols; i++) {
        float v = ((float)LOAD(in, base + i) - mean) * inv;
        STORE(out, base + i, v * (float)LOAD(gamma, i) + (float)LOAD(beta, i));
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Elementwise / shape utilities
// ─────────────────────────────────────────────────────────────────────────
__kernel void st_leaky_relu(
    __global const storage_t* in, __global storage_t* out, const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = LOAD(in, i);
    STORE(out, i, x >= 0.0f ? x : ST_LRELU_ALPHA * x);
}

// out = (a + b) * scale   — the AdainResBlk1d residual join, scale = 1/sqrt(2)
__kernel void st_add_scaled(
    __global const storage_t* a, __global const storage_t* b,
    __global storage_t* out, const int n, const float scale) {
    int i = get_global_id(0);
    if (i >= n) return;
    STORE(out, i, ((float)LOAD(a, i) + (float)LOAD(b, i)) * scale);
}

__kernel void st_add(
    __global const storage_t* a, __global const storage_t* b,
    __global storage_t* out, const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    STORE(out, i, (float)LOAD(a, i) + (float)LOAD(b, i));
}

// Broadcast a per-channel bias over [C, T]
__kernel void st_add_bias_ct(
    __global const storage_t* in, __global const storage_t* bias,
    __global storage_t* out, const int C, const int T) {
    int gid = get_global_id(0);
    if (gid >= C * T) return;
    STORE(out, gid, (float)LOAD(in, gid) + (float)LOAD(bias, gid / T));
}

// Broadcast a per-column bias over [rows, cols]
__kernel void st_add_bias_rows(
    __global const storage_t* in, __global const storage_t* bias,
    __global storage_t* out, const int rows, const int cols) {
    int gid = get_global_id(0);
    if (gid >= rows * cols) return;
    STORE(out, gid, (float)LOAD(in, gid) + (float)LOAD(bias, gid % cols));
}

// Transpose [A, B] -> [B, A] row-major
__kernel void st_transpose(
    __global const storage_t* in, __global storage_t* out,
    const int A, const int B) {
    int gid = get_global_id(0);
    if (gid >= A * B) return;
    int a = gid / B;
    int b = gid - a * B;
    STORE(out, b * A + a, LOAD(in, gid));
}

// Resize nearest, integer upscale on the T axis, asymmetric + floor:
//   out[c, t] = in[c, t / factor]
__kernel void st_resize_nearest_t(
    __global const storage_t* in, __global storage_t* out,
    const int C, const int T_in, const int factor) {
    int gid = get_global_id(0);
    int T_out = T_in * factor;
    if (gid >= C * T_out) return;
    int c = gid / T_out;
    int t = gid - c * T_out;
    STORE(out, gid, LOAD(in, c * T_in + t / factor));
}

// Embedding gather producing [T, C] row-major: out[t, c] = emb[ids[t], c]
__kernel void st_embed_tc(
    __global const int* ids, __global const storage_t* emb,
    __global storage_t* out, const int T, const int C) {
    int gid = get_global_id(0);
    if (gid >= T * C) return;
    int t = gid / C;
    int c = gid - t * C;
    STORE(out, gid, LOAD(emb, ids[t] * C + c));
}

// Alignment expansion: out[c, f] = in[c, phoneme_of_frame[f]]
// phoneme_of_frame is the flattened one-hot alignment (int32 [T_frames]),
// which is how length regulation is applied without materializing [T, F].
__kernel void st_align_expand(
    __global const storage_t* in, __global const int* frame_src,
    __global storage_t* out, const int C, const int T_in, const int F) {
    int gid = get_global_id(0);
    if (gid >= C * F) return;
    int c = gid / F;
    int f = gid - c * F;
    STORE(out, gid, LOAD(in, c * T_in + frame_src[f]));
}

// AdaLayerNorm combine for a [rows, cols] row-major tensor (feature axis last):
//   out[r,c] = (1 + gamma[c]) * n[r,c] + beta[c]
// gamma/beta are the two halves of the style fc output [2*cols].
__kernel void st_adaln_rows(
    __global const storage_t* n,
    __global const storage_t* fc_out,   // [2*cols]
    __global storage_t*       out,
    const int rows, const int cols) {
    int gid = get_global_id(0);
    if (gid >= rows * cols) return;
    int c = gid % cols;
    float gamma = LOAD(fc_out, c);
    float beta  = LOAD(fc_out, cols + c);
    STORE(out, gid, (1.0f + gamma) * (float)LOAD(n, gid) + beta);
}

// Re-concat the style vector onto each timestep's features:
//   out[t, 0:C1]      = a[t, :]
//   out[t, C1:C1+C2]  = s[:]
__kernel void st_pack_style(
    __global const storage_t* a,
    __global const storage_t* s,
    __global storage_t*       out,
    const int T, const int C1, const int C2) {
    int gid = get_global_id(0);
    int W = C1 + C2;
    if (gid >= T * W) return;
    int t = gid / W;
    int c = gid - t * W;
    STORE(out, gid, c < C1 ? LOAD(a, t * C1 + c) : LOAD(s, c - C1));
}

// ─────────────────────────────────────────────────────────────────────────
// Generator primitives
// ─────────────────────────────────────────────────────────────────────────

// Snake activation:  out = y + (1/alpha[c]) * sin(alpha[c]*y)^2
// `inv_alpha` is the PRE-COMPUTED reciprocal shipped in the converted weights
// (decoder.generator.<block>.Reciprocal*_output_0) — do not divide at runtime.
__kernel void st_snake(
    __global const storage_t* in,
    __global const storage_t* alpha,      // [C]
    __global const storage_t* inv_alpha,  // [C]
    __global storage_t*       out,
    const int C, const int T) {
    int gid = get_global_id(0);
    if (gid >= C * T) return;
    int c = gid / T;
    float a  = LOAD(alpha, c);
    float ia = LOAD(inv_alpha, c);
    float y  = LOAD(in, gid);
    float s  = sin(a * y);
    STORE(out, gid, y + ia * s * s);
}

// General (non-depthwise) ConvTranspose1d.
// ONNX weight layout for ConvTranspose is [in_C, out_C, K] — TRANSPOSED vs Conv.
//   out[co,o] = bias[co] + Σ_ci Σ_k w[ci][co][k] * x[ci][(o+pad-k)/stride]
//               for (o+pad-k) divisible by stride and the index in range
__kernel void st_convtranspose1d(
    __global const storage_t* in,
    __global const storage_t* w,
    __global const storage_t* bias,
    __global storage_t*       out,
    const int Cin, const int Cout, const int T_in, const int T_out,
    const int K, const int stride, const int pad, const int has_bias) {
    int gid = get_global_id(0);
    if (gid >= Cout * T_out) return;
    int co = gid / T_out;
    int o  = gid - co * T_out;
    float acc = has_bias ? (float)LOAD(bias, co) : 0.0f;
    for (int k = 0; k < K; k++) {
        int num = o + pad - k;
        if (num < 0 || (num % stride) != 0) continue;
        int ti = num / stride;
        if (ti < 0 || ti >= T_in) continue;
        for (int ci = 0; ci < Cin; ci++)
            acc += (float)LOAD(w, (ci * Cout + co) * K + k) * (float)LOAD(in, ci * T_in + ti);
    }
    STORE(out, gid, acc);
}

// ─────────────────────────────────────────────────────────────────────────
// ConvTranspose1d — TILED, general (OPT #2, razr Adreno620v2).
// Workgroup computes [CT_OT output-times × CT_CO output-chans]. Input channels
// loop in CT_CIN_CHUNK chunks; each chunk's input span for the tile's outputs is
// staged into LDS once and reused across all CT_CO output channels (kills the
// naive kernel's Cin×-per-output global re-read). Branch-free tap stepping: for a
// given output o the valid taps are exactly k ≡ (o+pad) mod stride, so we step k
// by `stride` from k0 instead of testing (o+pad-k)%stride (guide §8.5: prefer
// arithmetic over divergent control flow). w layout [Cin, Cout, K].
// Host guarantees span_ti = ti_hi-ti_lo+1 ≤ CT_SPAN_MAX, with
//   ti_lo = floor((o_base+pad-(K-1))/stride), ti_hi = floor((o_base+CT_OT-1+pad)/stride).
// T_REG=4: each WI computes 4 consecutive output times for 1 channel. For a fixed
// input ti, those 4 outputs multiply 4 CONSECUTIVE weights w[kb..kb+3] (kb =
// o0+pad-ti*stride) → one vload4 replaces 4 scalar global weight loads (this
// kernel is weight-load bound). Same input staged in LDS, reused across CT_CO chans.
#define CT_OT        32      // output-times per workgroup (for ti_lo/span calc)
#define CT_OTWI      8       // WIs along time; each does CT_TREG outputs
#define CT_TREG      4
#define CT_CO        8
#define CT_CIN_CHUNK 16
#define CT_SPAN_MAX  48
static inline int ct_fdiv(int a, int b) {        // floor division, b>0
    return (a >= 0) ? (a / b) : -(((-a) + b - 1) / b);
}
__kernel void st_convtranspose1d_tiled(
    __global const storage_t* in,
    __global const storage_t* w,      // [Cin, Cout, K]
    __global const storage_t* bias,
    __global storage_t*       out,
    const int Cin, const int Cout, const int T_in, const int T_out,
    const int K, const int stride, const int pad, const int has_bias) {
    __local float tile[CT_CIN_CHUNK][CT_SPAN_MAX];
    const int lt  = get_local_id(0);            // 0..CT_OTWI-1
    const int lco = get_local_id(1);            // 0..CT_CO-1
    const int co  = get_global_id(1);           // output channel
    const int o_base = get_group_id(0) * CT_OT;
    const int o0 = o_base + lt * CT_TREG;        // first of this WI's CT_TREG outputs
    const int ti_lo   = ct_fdiv(o_base + pad - (K - 1), stride);   // tile[*][0] maps to this ti
    const int span_ti = ct_fdiv(o_base + CT_OT - 1 + pad, stride) - ti_lo + 1;
    const int nthreads = CT_OTWI * CT_CO;
    const int tid = lco * CT_OTWI + lt;
    const int co_active = (co < Cout);

    float bv = (has_bias && co_active) ? (float)LOAD(bias, co) : 0.0f;
    float acc0 = bv, acc1 = bv, acc2 = bv, acc3 = bv;

    for (int ci0 = 0; ci0 < Cin; ci0 += CT_CIN_CHUNK) {
        const int total = CT_CIN_CHUNK * span_ti;
        for (int idx = tid; idx < total; idx += nthreads) {
            int cc = idx / span_ti;
            int m  = idx - cc * span_ti;
            int ci = ci0 + cc;
            int ti = ti_lo + m;
            float v = 0.0f;
            if (ci < Cin && ti >= 0 && ti < T_in) v = (float)LOAD(in, ci * T_in + ti);
            tile[cc][m] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (co_active) {
            int nci = min(CT_CIN_CHUNK, Cin - ci0);
            // ti that can feed outputs o0..o0+3 (kb+r ∈ [0,K)); clamp to the tile.
            int ti_a = ct_fdiv(o0 + pad - (K - 1), stride);
            int ti_b = ct_fdiv(o0 + CT_TREG - 1 + pad, stride);
            if (ti_a < ti_lo) ti_a = ti_lo;
            if (ti_b > ti_lo + span_ti - 1) ti_b = ti_lo + span_ti - 1;
            for (int cc = 0; cc < nci; cc++) {
                int wrow = ((ci0 + cc) * Cout + co) * K;
                __local const float* trow = tile[cc];
                for (int ti = ti_a; ti <= ti_b; ti++) {
                    float x = trow[ti - ti_lo];
                    int kb = o0 + pad - ti * stride;    // weight idx for output o0
#ifndef USE_FP16
                    if (kb >= 0 && kb + 3 < K) {
                        float4 wv = vload4(0, w + wrow + kb);
                        acc0 = mad(x, wv.x, acc0);
                        acc1 = mad(x, wv.y, acc1);
                        acc2 = mad(x, wv.z, acc2);
                        acc3 = mad(x, wv.w, acc3);
                    } else
#endif
                    {
                        if (kb + 0 >= 0 && kb + 0 < K) acc0 = mad(x, (float)LOAD(w, wrow + kb + 0), acc0);
                        if (kb + 1 >= 0 && kb + 1 < K) acc1 = mad(x, (float)LOAD(w, wrow + kb + 1), acc1);
                        if (kb + 2 >= 0 && kb + 2 < K) acc2 = mad(x, (float)LOAD(w, wrow + kb + 2), acc2);
                        if (kb + 3 >= 0 && kb + 3 < K) acc3 = mad(x, (float)LOAD(w, wrow + kb + 3), acc3);
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (co_active) {
        int ob = co * T_out;
        if (o0 + 0 < T_out) STORE(out, ob + o0 + 0, acc0);
        if (o0 + 1 < T_out) STORE(out, ob + o0 + 1, acc1);
        if (o0 + 2 < T_out) STORE(out, ob + o0 + 2, acc2);
        if (o0 + 3 < T_out) STORE(out, ob + o0 + 3, acc3);
    }
}

// iSTFT overlap-add, fused. Replaces the two ConvTransposes and the Sub:
//   y[t] = Σ_{f: 0 <= t-hop*f < N} Σ_k  ( REAL[k,f]*basis_r[k, t-hop*f]
//                                       - IMAG[k,f]*basis_i[k, t-hop*f] )
// basis_r/basis_i are the stored backward bases [K, 1, N] which already fold in
// the 1/N factor and the periodic Hann window. There is NO window-sum
// normalization and NO 2x Hermitian factor — this is an unnormalized
// half-spectrum overlap-add, which is what the network was trained against.
__kernel void st_istft_ola(
    __global const storage_t* re,       // [K, Fr]
    __global const storage_t* im,       // [K, Fr]
    __global const storage_t* basis_r,  // [K, N]
    __global const storage_t* basis_i,  // [K, N]
    __global storage_t*       out,      // [T_out]
    const int K, const int Fr, const int N, const int hop, const int T_out) {
    int t = get_global_id(0);
    if (t >= T_out) return;
    float acc = 0.0f;
    int f_lo = (t - N + 1 + hop - 1) / hop;   // ceil((t-N+1)/hop)
    if (f_lo < 0) f_lo = 0;
    int f_hi = t / hop;
    if (f_hi >= Fr) f_hi = Fr - 1;
    for (int f = f_lo; f <= f_hi; f++) {
        int n = t - hop * f;
        if (n < 0 || n >= N) continue;
        for (int k = 0; k < K; k++)
            acc += (float)LOAD(re, k * Fr + f) * (float)LOAD(basis_r, k * N + n)
                 - (float)LOAD(im, k * Fr + f) * (float)LOAD(basis_i, k * N + n);
    }
    STORE(out, t, acc);
}

// magnitude/phase for the source STFT front-end.
// The graph hand-expands atan2 with the tie-break (IMAG==0 && REAL<0) -> +pi,
// which differs from library atan2f's -pi for atan2(-0.0, -x). Reproduced here.
__kernel void st_mag_phase(
    __global const storage_t* re,
    __global const storage_t* im,
    __global storage_t*       mag,
    __global storage_t*       phase,
    const int n, const float eps) {
    int i = get_global_id(0);
    if (i >= n) return;
    float r = LOAD(re, i), m = LOAD(im, i);
    STORE(mag, i, sqrt(r * r + m * m + eps));
    float a = atan(m / r);
    float w = (m > 0.0f) ? (a + M_PI_F) : (a - M_PI_F);
    float w1 = (r < 0.0f) ? w : a;
    float w2 = (m == 0.0f && r < 0.0f) ? M_PI_F : w1;
    STORE(phase, i, w2);
}

// out = exp(in)
__kernel void st_exp(__global const storage_t* in, __global storage_t* out, const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    STORE(out, i, exp((float)LOAD(in, i)));
}

// mag*cos(sin(raw)) and mag*sin(sin(raw)) — note the phase is sin(raw), so
// there are two nested sines on that branch. Collapsing them is a real trap.
__kernel void st_polar_to_cart(
    __global const storage_t* mag,
    __global const storage_t* raw,
    __global storage_t*       re,
    __global storage_t*       im,
    const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float m = LOAD(mag, i);
    float p = sin((float)LOAD(raw, i));
    STORE(re, i, m * cos(p));
    STORE(im, i, m * sin(p));
}

// out = in * scale (elementwise, scalar scale)
__kernel void st_scale(__global const storage_t* in, __global storage_t* out,
                       const int n, const float scale) {
    int i = get_global_id(0);
    if (i >= n) return;
    STORE(out, i, (float)LOAD(in, i) * scale);
}

// LeakyReLU with a caller-supplied alpha (the generator uses 0.1 on the trunk
// and 0.01 immediately before conv_post — not the 0.2 used everywhere else).
__kernel void st_leaky_relu_a(
    __global const storage_t* in, __global storage_t* out,
    const int n, const float alpha) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = LOAD(in, i);
    STORE(out, i, x >= 0.0f ? x : alpha * x);
}

// Copy [C, T_in] into [C, T_out] at time-offset `off`, replicating nothing
// (used for the 1-sample reflect pad at the FRONT of the time axis).
__kernel void st_pad_front_reflect(
    __global const storage_t* in, __global storage_t* out,
    const int C, const int T_in, const int pad_front) {
    int gid = get_global_id(0);
    int T_out = T_in + pad_front;
    if (gid >= C * T_out) return;
    int c = gid / T_out;
    int t = gid - c * T_out;
    int src = t - pad_front;
    if (src < 0) src = -src;            // reflect about index 0
    if (src >= T_in) src = T_in - 1;
    STORE(out, gid, LOAD(in, c * T_in + src));
}

// ── microbenchmarks (device peak characterization) ───────────────────────
// Pure streaming bandwidth: 1 read + 1 write per element, no reuse, coalesced.
__kernel void bench_copy(__global const float* in, __global float* out, const int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = in[i];
}
// Vectorized streaming bandwidth (float4).
__kernel void bench_copy4(__global const float4* in, __global float4* out, const int n4) {
    int i = get_global_id(0);
    if (i < n4) out[i] = in[i];
}
// Compute peak: long dependent-free FMA chain, negligible memory traffic.
__kernel void bench_fma(__global float* out, const int iters) {
    // 8 INDEPENDENT chains: with only 4 the kernel is FMA-latency-bound and
    // measures latency, not throughput. float4 ops give 8 lanes x 4 = 32 flops
    // per loop iteration.
    int i = get_global_id(0);
    float4 a0 = (float4)(i, i+1, i+2, i+3), a1 = a0 + 1.0f;
    float4 a2 = a0 + 2.0f, a3 = a0 + 3.0f;
    float4 a4 = a0 + 4.0f, a5 = a0 + 5.0f;
    float4 a6 = a0 + 6.0f, a7 = a0 + 7.0f;
    float4 b = (float4)(1.0000001f), c = (float4)(0.0000001f);
    for (int k = 0; k < iters; k++) {
        a0 = fma(a0,b,c); a1 = fma(a1,b,c); a2 = fma(a2,b,c); a3 = fma(a3,b,c);
        a4 = fma(a4,b,c); a5 = fma(a5,b,c); a6 = fma(a6,b,c); a7 = fma(a7,b,c);
    }
    if (i == -1) { float4 s = a0+a1+a2+a3+a4+a5+a6+a7; out[0] = s.x+s.y+s.z+s.w; }
}
