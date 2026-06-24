// OpenVoice V2 ToneColorConverter — OpenCL kernels (split per functional unit).
// Built by concatenation (Engine reads _preamble.cl FIRST). DO NOT define macros here.

// ── conv_transpose1d (upsample) ─────────────────────────────────────────────
// in: [Cin, Tin]  weight: [Cin, Cout, K]  bias:[Cout]|null
// out: [Cout, Tout]  Tout = (Tin-1)*stride - 2*pad + K   (output_padding=0)
// One work-item per (cout, tout); gather contributions.
__kernel void conv_transpose1d(
    __global const REAL* in, __global const half* w, __global const half* bias,
    __global REAL* out,
    const int Cin, const int Tin, const int Cout, const int Tout,
    const int K, const int stride, const int pad, const int has_bias)
{
    int co = get_global_id(0);
    int to = get_global_id(1);
    if (co >= Cout || to >= Tout) return;
    float acc = has_bias ? WLOAD(co, bias) : 0.0f;
    // out[to] gets in[ti] * w[ci,co,k] where to + pad = ti*stride + k
    for (int k = 0; k < K; ++k) {
        int num = to + pad - k;
        if (num < 0 || (num % stride) != 0) continue;
        int ti = num / stride;
        if (ti < 0 || ti >= Tin) continue;
        for (int ci = 0; ci < Cin; ++ci) {
            __global const REAL* in_c = in + (size_t)ci * Tin;
            __global const half* w_e  = w + ((size_t)ci * Cout + co) * K + k;
            acc += LOAD(ti, in_c) * WLOAD(0, w_e);
        }
    }
    STORE(acc, (size_t)co * Tout + to, out);
}

// ── conv_transpose1d_opt — register-tiled over OUTPUT channels (#15, §10.3.4) ─
// Each work-item computes TILE_CO output channels for one output time. The input
// element in[ci,ti] is shared across all output channels, so it's loaded ONCE and
// reused TILE_CO times (naive reloads it per output channel). All TILE_CO outputs
// share the same time `to` → same phase → identical num%stride branch (no extra
// divergence). dim0=ceil(Cout/TILE_CO), dim1=Tout.
#ifndef TILE_CO
#define TILE_CO 8
#endif
__kernel void conv_transpose1d_opt(
    __global const REAL* in, __global const half* w, __global const half* bias,
    __global REAL* out,
    const int Cin, const int Tin, const int Cout, const int Tout,
    const int K, const int stride, const int pad, const int has_bias)
{
    int co0 = get_global_id(0) * TILE_CO;
    int to  = get_global_id(1);
    if (co0 >= Cout || to >= Tout) return;
    float acc[TILE_CO];
    #pragma unroll
    for (int t=0;t<TILE_CO;++t) acc[t] = (has_bias && co0+t<Cout) ? WLOAD(co0+t, bias) : 0.0f;
    // Phase-stepping: only k ≡ (to+pad) (mod stride) contribute. Start at that phase
    // and step by `stride` → visits K/stride taps instead of K (no wasted modulo).
    int kphase = (to + pad) % stride;
    for (int k = kphase; k < K; k += stride) {
        int num = to + pad - k;
        if (num < 0) break;                  // larger k only more negative
        int ti = num / stride;               // exact (num divisible by stride by construction)
        if (ti >= Tin) continue;             // smaller k ⇒ larger ti; later k brings it in range
        for (int ci = 0; ci < Cin; ++ci) {
            float iv = LOAD((size_t)ci*Tin + ti, in);            // loaded once, reused TILE_CO×
            __global const half* wb = w + ((size_t)ci*Cout + co0)*K + k;
            #pragma unroll
            for (int t=0;t<TILE_CO;++t) acc[t] += iv * WLOAD((size_t)t*K, wb);  // w[ci, co0+t, k]
        }
    }
    #pragma unroll
    for (int t=0;t<TILE_CO;++t){ int co=co0+t; if(co<Cout) STORE(acc[t], (size_t)co*Tout + to, out); }
}
