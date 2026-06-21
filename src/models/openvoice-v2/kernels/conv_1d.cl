// OpenVoice V2 ToneColorConverter — OpenCL kernels (split per functional unit).
// Built by concatenation (Engine reads _preamble.cl FIRST). DO NOT define macros here.

// ── conv1d ─────────────────────────────────────────────────────────────────
// in:  [Cin, Tin]   weight: [Cout, Cin, K]   bias: [Cout] (or null)
// out: [Cout, Tout]   Tout = (Tin + 2*pad - dilation*(K-1) - 1)/stride + 1
// One work-item per (cout, tout).
__kernel void conv1d(
    __global const REAL*  in,    __global const half* w,  __global const half* bias,
    __global REAL*        out,
    const int Cin, const int Tin, const int Cout, const int Tout,
    const int K, const int stride, const int dilation, const int pad,
    const int has_bias)
{
    int co = get_global_id(0);   // out channel
    int to = get_global_id(1);   // out time
    if (co >= Cout || to >= Tout) return;
    float acc = has_bias ? WLOAD(co, bias) : 0.0f;
    int t0 = to * stride - pad;
    for (int ci = 0; ci < Cin; ++ci) {
        __global const REAL* in_c = in + (size_t)ci * Tin;
        __global const half* w_c  = w + ((size_t)co * Cin + ci) * K;
        for (int k = 0; k < K; ++k) {
            int ti = t0 + k * dilation;
            if (ti < 0 || ti >= Tin) continue;
            acc += LOAD(ti, in_c) * WLOAD(k, w_c);
        }
    }
    STORE(acc, (size_t)co * Tout + to, out);
}

// ── conv1d_opt — register-tiled (Qualcomm guide §10.3.4, §6.3, §8.4) ─────────
// Each work-item computes TILE_T consecutive output time-steps for one out
// channel, so every weight element is loaded ONCE and reused TILE_T times
// (naive reloads weights per output). dim0=Cout, dim1=ceil(Tout/TILE_T).
#ifndef TILE_T
#define TILE_T 8
#endif
__kernel void conv1d_opt(
    __global const REAL*  in,    __global const half* w,  __global const half* bias,
    __global REAL*        out,
    const int Cin, const int Tin, const int Cout, const int Tout,
    const int K, const int stride, const int dilation, const int pad,
    const int has_bias)
{
    int co  = get_global_id(0);
    int to0 = get_global_id(1) * TILE_T;
    if (co >= Cout || to0 >= Tout) return;
    float b = has_bias ? WLOAD(co, bias) : 0.0f;

#if (TILE_T==8) && !defined(STORE_FP32)
    // Vectorized fast path (§6.3): TILE_T=8 contiguous outputs read 8 contiguous
    // inputs per tap (stride=1 ⇒ spacing is 1 regardless of dilation; dilation only
    // shifts the base) → one half8 vload instead of 8 scalar loads. Interior tiles
    // only (fully in-bounds); edges fall to the scalar path.
    if (stride==1 && (to0+8)<=Tout && (to0-pad)>=0 && (to0+7+(K-1)*dilation-pad)<Tin) {
        float8 a = (float8)(b);
        for (int ci=0; ci<Cin; ++ci) {
            __global const half* in_c = in + (size_t)ci*Tin;
            __global const half* w_c  = w + ((size_t)co*Cin + ci)*K;
            for (int k=0;k<K;++k) {
                float wv = WLOAD(k, w_c);
                half8 iv = vload8(0, in_c + (to0 + k*dilation - pad));
                a += convert_float8(iv) * wv;
            }
        }
        vstore8(convert_half8(a), 0, out + (size_t)co*Tout + to0);
        return;
    }
#endif
    float acc[TILE_T];
    #pragma unroll
    for (int t=0;t<TILE_T;++t) acc[t]=b;
    for (int ci=0; ci<Cin; ++ci) {
        __global const REAL* in_c = in + (size_t)ci*Tin;
        __global const half* w_c  = w + ((size_t)co*Cin + ci)*K;
        for (int k=0;k<K;++k) {
            float wv = WLOAD(k, w_c);
            int base = k*dilation - pad;
            #pragma unroll
            for (int t=0;t<TILE_T;++t) {
                int ti = (to0+t)*stride + base;
                if (ti>=0 && ti<Tin) acc[t] += LOAD(ti, in_c) * wv;
            }
        }
    }
    #pragma unroll
    for (int t=0;t<TILE_T;++t) {
        int to=to0+t;
        if (to<Tout) STORE(acc[t], (size_t)co*Tout + to, out);
    }
}

// ── conv1d_2d — tile NC output channels × TILE_T times (#3 2D, §6.3) ──────────
// The half8 input load is shared across all output channels, so it's loaded ONCE
// per (ci,k) and reused across NC channels → ~NC× less input bandwidth (helps the
// bandwidth-bound low-channel convs). dim0=ceil(Cout/NC), dim1=ceil(Tout/TILE_T).
#ifndef NC
#define NC 2
#endif
#if (TILE_T==8) && !defined(STORE_FP32)
__kernel void conv1d_2d(
    __global const REAL*  in,    __global const half* w,  __global const half* bias,
    __global REAL*        out,
    const int Cin, const int Tin, const int Cout, const int Tout,
    const int K, const int stride, const int dilation, const int pad,
    const int has_bias)
{
    int co0 = get_global_id(0) * NC;
    int to0 = get_global_id(1) * TILE_T;
    if (co0 >= Cout || to0 >= Tout) return;
    bool interior = (stride==1 && (to0+8)<=Tout && (to0-pad)>=0 && (to0+7+(K-1)*dilation-pad)<Tin);
    float8 a[NC];
    #pragma unroll
    for (int n=0;n<NC;++n) a[n] = (float8)((has_bias && co0+n<Cout)? WLOAD(co0+n,bias):0.0f);
    if (interior) {
        for (int ci=0; ci<Cin; ++ci) {
            __global const half* in_c = in + (size_t)ci*Tin;
            for (int k=0;k<K;++k) {
                float8 ivf = convert_float8(vload8(0, in_c + (to0 + k*dilation - pad)));
                #pragma unroll
                for (int n=0;n<NC;++n) a[n] += ivf * (float)WLOAD(k, w + ((size_t)(co0+n)*Cin+ci)*K);
            }
        }
        #pragma unroll
        for (int n=0;n<NC;++n) if(co0+n<Cout) vstore8(convert_half8(a[n]), 0, out + (size_t)(co0+n)*Tout + to0);
        return;
    }
    // edge fallback: scalar over the (up to) TILE_T outputs for each of NC channels
    float acc[NC][TILE_T];
    #pragma unroll
    for(int n=0;n<NC;++n){ float b=(has_bias&&co0+n<Cout)?WLOAD(co0+n,bias):0.0f; for(int t=0;t<TILE_T;++t) acc[n][t]=b; }
    for (int ci=0; ci<Cin; ++ci) {
        __global const REAL* in_c = in + (size_t)ci*Tin;
        for (int k=0;k<K;++k) {
            int base=k*dilation-pad;
            #pragma unroll
            for(int n=0;n<NC;++n){ float wv=WLOAD(k, w + ((size_t)(co0+n)*Cin+ci)*K);
                for(int t=0;t<TILE_T;++t){ int ti=(to0+t)*stride+base; if(ti>=0&&ti<Tin) acc[n][t]+=LOAD(ti,in_c)*wv; } }
        }
    }
    #pragma unroll
    for(int n=0;n<NC;++n) if(co0+n<Cout) for(int t=0;t<TILE_T;++t){ int to=to0+t; if(to<Tout) STORE(acc[n][t],(size_t)(co0+n)*Tout+to,out); }
}
#endif

// ── conv1d_2dh — fp16-ACCUMULATE variant of conv1d_2d (2× ALU roof) ───────────
// Accumulates in half8 with native half FMA (no float convert on load/store).
// Adreno does fp16 at ~2× fp32. Precision: fp16 accumulation over Cin*K terms —
// gated on cosine in the microbench. dim0=ceil(Cout/NC), dim1=ceil(Tout/TILE_T).
#if (TILE_T==8) && !defined(STORE_FP32)
__kernel void conv1d_2dh(
    __global const half*  in,    __global const half* w,  __global const half* bias,
    __global half*        out,
    const int Cin, const int Tin, const int Cout, const int Tout,
    const int K, const int stride, const int dilation, const int pad,
    const int has_bias)
{
    int co0 = get_global_id(0) * NC;
    int to0 = get_global_id(1) * TILE_T;
    if (co0 >= Cout || to0 >= Tout) return;
    bool interior = (stride==1 && (to0+8)<=Tout && (to0-pad)>=0 && (to0+7+(K-1)*dilation-pad)<Tin);
    half8 a[NC];
    #pragma unroll
    for (int n=0;n<NC;++n) a[n] = (half8)((has_bias && co0+n<Cout)? (half)WLOAD(co0+n,bias):(half)0);
    if (interior) {
        for (int ci=0; ci<Cin; ++ci) {
            __global const half* in_c = in + (size_t)ci*Tin;
            for (int k=0;k<K;++k) {
                half8 iv = vload8(0, in_c + (to0 + k*dilation - pad));
                #pragma unroll
                for (int n=0;n<NC;++n) a[n] += iv * w[((size_t)(co0+n)*Cin+ci)*K + k];
            }
        }
        #pragma unroll
        for (int n=0;n<NC;++n) if(co0+n<Cout) vstore8(a[n], 0, out + (size_t)(co0+n)*Tout + to0);
        return;
    }
    half acc[NC][TILE_T];
    #pragma unroll
    for(int n=0;n<NC;++n){ half b=(has_bias&&co0+n<Cout)?(half)WLOAD(co0+n,bias):(half)0; for(int t=0;t<TILE_T;++t) acc[n][t]=b; }
    for (int ci=0; ci<Cin; ++ci) {
        __global const half* in_c = in + (size_t)ci*Tin;
        for (int k=0;k<K;++k) {
            int base=k*dilation-pad;
            #pragma unroll
            for(int n=0;n<NC;++n){ half wv=w[((size_t)(co0+n)*Cin+ci)*K + k];
                for(int t=0;t<TILE_T;++t){ int ti=(to0+t)*stride+base; if(ti>=0&&ti<Tin) acc[n][t]+=vload_half(ti,in_c)*wv; } }
        }
    }
    #pragma unroll
    for(int n=0;n<NC;++n) if(co0+n<Cout) for(int t=0;t<TILE_T;++t){ int to=to0+t; if(to<Tout) vstore_half(acc[n][t],(size_t)(co0+n)*Tout+to,out); }
}
#endif

// ── conv1d_local — like conv1d_opt but weights staged in __local (#4/#6) ──────
// Work-group = LCH channels × LT time-tiles. The LCH channels' weights (contiguous
// in [Cout,Cin,K]) are loaded ONCE into __local by the whole group, then reused by
// all LT time-tile threads (removes the LT× redundant global weight loads). All
// threads must reach the barrier, so the out-of-range guard comes AFTER it.
// Requires local size {LCH, LT}; wl sized LCH*Cin*K halfs via a __local arg.
__kernel void conv1d_local(
    __global const REAL*  in,    __global const half* w,  __global const half* bias,
    __global REAL*        out,   __local half* wl,
    const int Cin, const int Tin, const int Cout, const int Tout,
    const int K, const int stride, const int dilation, const int pad,
    const int has_bias)
{
    int lid0 = get_local_id(0), lid1 = get_local_id(1);
    int LCH  = get_local_size(0), LT = get_local_size(1);
    int co_base = get_group_id(0) * LCH;
    int tid = lid1*LCH + lid0, nthreads = LCH*LT;
    int wcount = LCH*Cin*K;
    // cooperative load of this group's LCH channels' weights (contiguous block)
    size_t wbase = (size_t)co_base*Cin*K;
    for (int i=tid; i<wcount; i+=nthreads) wl[i] = w[wbase + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    int co  = co_base + lid0;
    int to0 = get_global_id(1) * TILE_T;
    if (co >= Cout || to0 >= Tout) return;
    __local const half* wco = wl + (size_t)lid0*Cin*K;     // this thread's channel weights
    float b = has_bias ? WLOAD(co, bias) : 0.0f;
#if (TILE_T==8) && !defined(STORE_FP32)
    if (stride==1 && (to0+8)<=Tout && (to0-pad)>=0 && (to0+7+(K-1)*dilation-pad)<Tin) {
        float8 a = (float8)(b);
        for (int ci=0; ci<Cin; ++ci) {
            __global const half* in_c = in + (size_t)ci*Tin;
            __local  const half* w_c  = wco + (size_t)ci*K;
            for (int k=0;k<K;++k) {
                float wv = vload_half(k, w_c);
                half8 iv = vload8(0, in_c + (to0 + k*dilation - pad));
                a += convert_float8(iv) * wv;
            }
        }
        vstore8(convert_half8(a), 0, out + (size_t)co*Tout + to0);
        return;
    }
#endif
    float acc[TILE_T];
    #pragma unroll
    for (int t=0;t<TILE_T;++t) acc[t]=b;
    for (int ci=0; ci<Cin; ++ci) {
        __global const REAL* in_c = in + (size_t)ci*Tin;
        __local  const half* w_c  = wco + (size_t)ci*K;
        for (int k=0;k<K;++k) {
            float wv = vload_half(k, w_c);
            int base = k*dilation - pad;
            #pragma unroll
            for (int t=0;t<TILE_T;++t) { int ti=(to0+t)*stride+base; if(ti>=0&&ti<Tin) acc[t]+=LOAD(ti,in_c)*wv; }
        }
    }
    #pragma unroll
    for (int t=0;t<TILE_T;++t){ int to=to0+t; if(to<Tout) STORE(acc[t],(size_t)co*Tout+to,out); }
}
