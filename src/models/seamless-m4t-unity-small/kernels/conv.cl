// 1-D convolutions: time-major grouped (encoder), channel-major (vocoder),
// and channel-major transpose (vocoder upsample).

// Encoder convs. in[T, Cin], weight[Cout, Cin/groups, K], out[Tout, Cout].
// Handles grouped (pos_conv g=16), strided (conv_pool s=8), depthwise (g=Cout).
__kernel void conv1d_tc(__global const storage_t* in,
                        __global const storage_t* w,
                        __global const storage_t* bias,
                        __global storage_t* out,
                        const int T, const int Cin, const int Cout, const int Tout,
                        const int K, const int stride, const int pad, const int dil,
                        const int groups, const int has_bias) {
    int gid = get_global_id(0);
    if (gid >= Tout * Cout) return;
    int tp = gid / Cout, o = gid - tp * Cout;
    int inpg = Cin / groups;
    int opg = Cout / groups;
    int base = (o / opg) * inpg;
    float acc = has_bias ? (float)LOAD(bias, o) : 0.0f;
    for (int il = 0; il < inpg; ++il) {
        int wrow = (o * inpg + il) * K;
        for (int k = 0; k < K; ++k) {
            int ti = tp * stride - pad + k * dil;
            if (ti >= 0 && ti < T) acc += (float)LOAD(w, wrow + k) * (float)LOAD(in, ti * Cin + (base + il));
        }
    }
    STORE(out, gid, acc);
}

// im2col for a groups=1 time-major conv: gathers the conv windows into a dense
// [Tout, Cin*K] matrix so the conv becomes a GEMM (out = col @ W^T, W=[Cout,Cin*K]).
// out[to, il*K + k] = in[(to*stride - pad + k*dil)*Cin + il]  (0 if out of range).
// Turns the adaptor conv_pool (stride=K=8) from a 0.2 GFLOP/s scalar conv into the
// tiled GEMM (~25×). global = Tout * Cin * K.
__kernel void im2col_tc(__global const storage_t* in, __global storage_t* out,
                        const int T, const int Cin, const int Tout,
                        const int K, const int stride, const int pad, const int dil) {
    int gid = get_global_id(0);
    int KK = Cin * K;
    if (gid >= Tout * KK) return;
    int to = gid / KK, r = gid - to * KK;
    int il = r / K, k = r - il * K;
    int ti = to * stride - pad + k * dil;
    float v = (ti >= 0 && ti < T) ? (float)LOAD(in, ti * Cin + il) : 0.0f;
    STORE(out, gid, v);
}

// Channel-major im2col for the vocoder (groups=1): col[(ci*K+k), t] = in[ci, t-pad+k*dil].
// Then out[Cout,T] = W[Cout,Cin*K] @ col[Cin*K,T] via CLBlast — STAYS channel-major.
// 3D launch (t, k, ci) — NO integer divisions per element. The old 1D version did TWO
// div/element (gid/T and row/K); on Adreno int-div is many cycles, and with tens of millions
// of elements that dominated this otherwise-trivial copy. global = (T, K, Cin).
__kernel void im2col_ct(__global const storage_t* in, __global storage_t* col,
                        const int Cin, const int T, const int K, const int dil, const int pad,
                        const int pre_act) {
    int t  = get_global_id(0);
    int k  = get_global_id(1);
    int ci = get_global_id(2);
    if (t >= T || k >= K || ci >= Cin) return;
    int ti = t - pad + k * dil;
    float v = (ti >= 0 && ti < T) ? (float)LOAD(in, ci * T + ti) : 0.0f;
    // FUSED pre-activation (e.g. vocoder lrelu): apply while gathering — the im2col already
    // reads every input element, so this eliminates a separate full-tensor act pass + launch.
    if (pre_act >= 0) v = act_apply(v, pre_act);
    STORE(col, ((ci * K + k) * T) + t, v);
}

// Vocoder convs. in[Cin, T], weight[Cout, Cin, K] (groups=1), out[Cout, T].
__kernel void conv1d_ct(__global const storage_t* in,
                        __global const storage_t* w,
                        __global const storage_t* bias,
                        __global storage_t* out,
                        const int Cin, const int T, const int Cout,
                        const int K, const int dil, const int pad, const int has_bias) {
    int gid = get_global_id(0);
    if (gid >= Cout * T) return;
    int co = gid / T, t = gid - co * T;
    float acc = has_bias ? (float)LOAD(bias, co) : 0.0f;
    for (int ci = 0; ci < Cin; ++ci) {
        int wrow = (co * Cin + ci) * K;
        int irow = ci * T;
        for (int k = 0; k < K; ++k) {
            int ti = t - pad + k * dil;
            if (ti >= 0 && ti < T) acc += (float)LOAD(w, wrow + k) * (float)LOAD(in, irow + ti);
        }
    }
    STORE(out, gid, acc);
}

// Vocoder conv, 4 output channels per work-item. Each input sample in[ci,ti] is
// loaded ONCE and reused across 4 output channels (4 independent fp32 accumulator
// chains → ILP + 4× less input bandwidth). Requires Cout % 4 == 0. global=(Cout/4)*T.
__kernel void conv1d_ct4(__global const storage_t* in,
                         __global const storage_t* w,
                         __global const storage_t* bias,
                         __global storage_t* out,
                         const int Cin, const int T, const int Cout,
                         const int K, const int dil, const int pad, const int has_bias,
                         const int pre_act) {
    int gid = get_global_id(0);
    int blk = gid / T, t = gid - blk * T;
    int co0 = blk << 2;
    if (co0 >= Cout) return;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (has_bias) { a0 = (float)LOAD(bias, co0); a1 = (float)LOAD(bias, co0 + 1);
                    a2 = (float)LOAD(bias, co0 + 2); a3 = (float)LOAD(bias, co0 + 3); }
    int wr0 = co0 * Cin * K;
    int CinK = Cin * K;
    for (int ci = 0; ci < Cin; ++ci) {
        int irow = ci * T;
        int wb = wr0 + ci * K;
        for (int k = 0; k < K; ++k) {
            int ti = t - pad + k * dil;
            if (ti >= 0 && ti < T) {
                float iv = (float)LOAD(in, irow + ti);
                if (pre_act >= 0) iv = act_apply(iv, pre_act);  // fuse lrelu (late-stage resblock)
                a0 += (float)LOAD(w, wb + k) * iv;
                a1 += (float)LOAD(w, wb + CinK + k) * iv;
                a2 += (float)LOAD(w, wb + 2 * CinK + k) * iv;
                a3 += (float)LOAD(w, wb + 3 * CinK + k) * iv;
            }
        }
    }
    STORE(out, co0 * T + t, a0); STORE(out, (co0 + 1) * T + t, a1);
    STORE(out, (co0 + 2) * T + t, a2); STORE(out, (co0 + 3) * T + t, a3);
}

// Vocoder conv, 4 output channels × 4 time-steps per work-item. Extends conv1d_ct4
// by ALSO tiling time: the 4-channel weight column w[co0..co0+3][ci][k] is loaded
// ONCE and reused across 4 consecutive output times (4× fewer weight reads — the
// conv was weight-bandwidth-bound, re-reading the full Cin×K weight per output t).
// Requires Cout % 4 == 0. global = (Cout/4) * ceil(T/4).
__kernel void conv1d_ct44(__global const storage_t* in,
                          __global const storage_t* w,
                          __global const storage_t* bias,
                          __global storage_t* out,
                          const int Cin, const int T, const int Cout,
                          const int K, const int dil, const int pad, const int has_bias) {
    int nT4 = (T + 3) >> 2;
    int gid = get_global_id(0);
    int blk = gid / nT4, tb = (gid - blk * nT4) << 2;
    int co0 = blk << 2;
    if (co0 >= Cout) return;
    float b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    if (has_bias) { b0 = (float)LOAD(bias, co0); b1 = (float)LOAD(bias, co0 + 1); b2 = (float)LOAD(bias, co0 + 2); b3 = (float)LOAD(bias, co0 + 3); }
    float a[4][4];   // [ch][tt]
    #pragma unroll
    for (int ch = 0; ch < 4; ++ch) { a[ch][0] = a[ch][1] = a[ch][2] = a[ch][3] = 0.0f; }
    int CinK = Cin * K, wbc = co0 * Cin * K;
    for (int ci = 0; ci < Cin; ++ci) {
        int irow = ci * T, wb = wbc + ci * K;
        for (int k = 0; k < K; ++k) {
            float w0 = (float)LOAD(w, wb + k), w1 = (float)LOAD(w, wb + CinK + k);
            float w2 = (float)LOAD(w, wb + 2 * CinK + k), w3 = (float)LOAD(w, wb + 3 * CinK + k);
            #pragma unroll
            for (int tt = 0; tt < 4; ++tt) {
                int t = tb + tt; if (t >= T) continue;
                int ti = t - pad + k * dil;
                if (ti >= 0 && ti < T) {
                    float iv = (float)LOAD(in, irow + ti);
                    a[0][tt] += w0 * iv; a[1][tt] += w1 * iv; a[2][tt] += w2 * iv; a[3][tt] += w3 * iv;
                }
            }
        }
    }
    #pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
        int t = tb + tt; if (t >= T) continue;
        STORE(out, co0 * T + t, a[0][tt] + b0); STORE(out, (co0 + 1) * T + t, a[1][tt] + b1);
        STORE(out, (co0 + 2) * T + t, a[2][tt] + b2); STORE(out, (co0 + 3) * T + t, a[3][tt] + b3);
    }
}

// col2im fold for transposed conv: out[co,j] = bias[co] + sum_k tmp[co*K+k, t] where
// t=(j+pad-k)/stride (when divisible & in range). tmp[Cout*K,T] = Wr[Cout*K,Cin] @ in
// (a CLBlast GEMM). Replaces the scalar conv_transpose1d_ct4 (1.68 s). global = Cout*Tout.
__kernel void col2im_transpose(__global const storage_t* tmp, __global const storage_t* bias,
                               __global storage_t* out, const int Cout, const int K, const int T,
                               const int Tout, const int stride, const int pad, const int has_bias) {
    int gid = get_global_id(0);
    if (gid >= Cout * Tout) return;
    int co = gid / Tout, j = gid - co * Tout;
    float acc = has_bias ? (float)LOAD(bias, co) : 0.0f;
    int base = co * K * T;
    for (int k = 0; k < K; ++k) {
        int pos = j + pad - k;
        if (pos % stride == 0) { int t = pos / stride; if (t >= 0 && t < T) acc += (float)LOAD(tmp, base + k * T + t); }
    }
    STORE(out, gid, acc);
}

// Vocoder upsample. in[Cin, T], weight[Cin, Cout, K], out[Cout, Tout],
// Tout = (T-1)*stride - 2*pad + K. Gather form.
__kernel void conv_transpose1d_ct(__global const storage_t* in,
                                  __global const storage_t* w,
                                  __global const storage_t* bias,
                                  __global storage_t* out,
                                  const int Cin, const int T, const int Cout,
                                  const int Tout, const int K, const int stride,
                                  const int pad, const int has_bias) {
    int gid = get_global_id(0);
    if (gid >= Cout * Tout) return;
    int co = gid / Tout, j = gid - co * Tout;
    float acc = has_bias ? (float)LOAD(bias, co) : 0.0f;
    for (int ci = 0; ci < Cin; ++ci) {
        int wbase = (ci * Cout + co) * K;
        int irow = ci * T;
        for (int k = 0; k < K; ++k) {
            int pos = j + pad - k;
            if (pos % stride == 0) {
                int t = pos / stride;
                if (t >= 0 && t < T) acc += (float)LOAD(w, wbase + k) * (float)LOAD(in, irow + t);
            }
        }
    }
    STORE(out, gid, acc);
}

// Transpose conv, 4 output channels per work-item: input sample in[ci,t] loaded
// once, reused across 4 output channels (weight w is [Cin,Cout,K] so the 4 co's
// are stride-K apart within a ci block). Requires Cout % 4 == 0. global=(Cout/4)*Tout.
__kernel void conv_transpose1d_ct4(__global const storage_t* in,
                                   __global const storage_t* w,
                                   __global const storage_t* bias,
                                   __global storage_t* out,
                                   const int Cin, const int T, const int Cout,
                                   const int Tout, const int K, const int stride,
                                   const int pad, const int has_bias) {
    int gid = get_global_id(0);
    int blk = gid / Tout, j = gid - blk * Tout;
    int co0 = blk << 2;
    if (co0 >= Cout) return;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (has_bias) { a0 = (float)LOAD(bias, co0); a1 = (float)LOAD(bias, co0 + 1);
                    a2 = (float)LOAD(bias, co0 + 2); a3 = (float)LOAD(bias, co0 + 3); }
    for (int ci = 0; ci < Cin; ++ci) {
        int wb = (ci * Cout + co0) * K;
        int irow = ci * T;
        for (int k = 0; k < K; ++k) {
            int pos = j + pad - k;
            if (pos % stride == 0) {
                int t = pos / stride;
                if (t >= 0 && t < T) {
                    float iv = (float)LOAD(in, irow + t);
                    a0 += (float)LOAD(w, wb + k) * iv;
                    a1 += (float)LOAD(w, wb + K + k) * iv;
                    a2 += (float)LOAD(w, wb + 2 * K + k) * iv;
                    a3 += (float)LOAD(w, wb + 3 * K + k) * iv;
                }
            }
        }
    }
    STORE(out, co0 * Tout + j, a0); STORE(out, (co0 + 1) * Tout + j, a1);
    STORE(out, (co0 + 2) * Tout + j, a2); STORE(out, (co0 + 3) * Tout + j, a3);
}

__kernel void conv1d_dw(__global const storage_t* in,
                        __global const storage_t* w,
                        __global const storage_t* bias,
                        __global storage_t* out,
                        const int T, const int Cin, const int Cout, const int Tout,
                        const int K, const int stride, const int pad, const int dil,
                        const int groups, const int has_bias) {
    int gid = get_global_id(0);
    if (gid >= Tout * Cout) return;
    int tp = gid / Cout, o = gid - tp * Cout;
    int inpg = Cin / groups;
    int opg = Cout / groups;
    int base = (o / opg) * inpg;
    float acc = has_bias ? (float)LOAD(bias, o) : 0.0f;
    for (int il = 0; il < inpg; ++il) {
        int wrow = (o * inpg + il) * K;
        for (int k = 0; k < K; ++k) {
            int ti = tp * stride - pad + k * dil;
            if (ti >= 0 && ti < T) acc += (float)LOAD(w, wrow + k) * (float)LOAD(in, ti * Cin + (base + il));
        }
    }
    STORE(out, gid, acc);
}

// Vocoder input construction on GPU (was a 378K-element CPU build + 25MB dict page-in =
// ~580ms of pure GPU idle at the start of the vocoder stage). Channel-major xin[Cin,T0]:
//   c in [0,LE)        -> lang[lang_id, c]              (per-channel constant over t)
//   c in [LE,LE+CE)    -> dict[code[t], c-LE]           (per-token gather)
//   c in [LE+CE,Cin)   -> spkr[spkr_id, c-LE-CE]
// global = (T0, Cin). code[] is the (wrapped) unit ids on device.
__kernel void vocoder_input_gather(__global const storage_t* lang, __global const storage_t* dict,
                                   __global const storage_t* spkr, __global const int* code,
                                   __global storage_t* xin, const int LE, const int CE, const int SE,
                                   const int T0, const int lang_id, const int spkr_id) {
    int t = get_global_id(0), c = get_global_id(1);
    const int Cin = LE + CE + SE;
    if (t >= T0 || c >= Cin) return;
    float v;
    if (c < LE)            v = (float)LOAD(lang, lang_id * LE + c);
    else if (c < LE + CE)  v = (float)LOAD(dict, code[t] * CE + (c - LE));
    else                   v = (float)LOAD(spkr, spkr_id * SE + (c - LE - CE));
    STORE(xin, (size_t)c * T0 + t, v);
}
