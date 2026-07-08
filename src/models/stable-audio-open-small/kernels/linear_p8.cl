// linear_p8.cl — B13 bespoke panel-packed linear (QUARANTINED PROGRAM).
// linear_p8_mt4 spills registers (4 float8 accumulators); on Adreno a
// register-heavy kernel degrades EVERY kernel compiled into the same
// program (measured: its presence in dit.cl took attn_fused's program from
// DiT 32 s → 44 s without ever being dispatched). It therefore lives alone
// in this file, built only when NNOPT_BESPOKE_LINEAR=1.
// VERDICT so far: 32 → 152 s as the linear path — kept only as the baseline
// for a future texture-path (image2d) variant. See BENCHMARK.md round 3.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)    vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
  #define LOAD8(p, i) convert_float8(vload_half8(0, (p) + (i)))
#else
  typedef float storage_t;
  #define LOAD(p, i)    ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
  #define LOAD8(p, i) vload8(0, (p) + (i))
#endif

inline float sum8(float8 v) {
    float4 a = v.lo + v.hi;
    float2 b = a.lo + a.hi;
    return b.x + b.y;
}

#define BLIN_MT 4
__kernel void linear_p8_mt4(__global const storage_t* x,    // [M, K]
                            __global const storage_t* w8,   // [K/8, N, 8]
                            __global storage_t* out,        // [M, N]
                            const int M, const int N, const int K) {
    const int n  = get_global_id(0);
    if (n >= N) return;
    const int m0 = get_group_id(1) * BLIN_MT;
    const int mt = min(BLIN_MT, M - m0);
    const int K8 = K >> 3;

    float8 acc0 = (float8)(0.0f), acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f), acc3 = (float8)(0.0f);
    for (int k8 = 0; k8 < K8; k8++) {
        const float8 wv = LOAD8(w8, ((size_t)k8 * N + n) * 8);
        const size_t xb = (size_t)m0 * K + k8 * 8;
        acc0 = mad(LOAD8(x, xb), wv, acc0);
        if (mt > 1) acc1 = mad(LOAD8(x, xb + K), wv, acc1);
        if (mt > 2) acc2 = mad(LOAD8(x, xb + 2 * K), wv, acc2);
        if (mt > 3) acc3 = mad(LOAD8(x, xb + 3 * K), wv, acc3);
    }
    STORE(out, (size_t)m0 * N + n, sum8(acc0));
    if (mt > 1) STORE(out, (size_t)(m0 + 1) * N + n, sum8(acc1));
    if (mt > 2) STORE(out, (size_t)(m0 + 2) * N + n, sum8(acc2));
    if (mt > 3) STORE(out, (size_t)(m0 + 3) * N + n, sum8(acc3));
}

// One-time repack W[N,K] → W8[K/8][N][8] (feeds linear_p8_mt4).
__kernel void linear_pack_p8(__global const storage_t* w,   // [N, K]
                             __global storage_t* w8,        // [K/8, N, 8]
                             const int N, const int K) {
    const int gid = get_global_id(0);        // n * K8 + k8
    const int K8 = K >> 3;
    if (gid >= N * K8) return;
    const int n  = gid / K8;
    const int k8 = gid % K8;
    const float8 v = LOAD8(w, (size_t)n * K + k8 * 8);
    #define P8B (((size_t)k8 * N + n) * 8)
    STORE(w8, P8B + 0, v.s0); STORE(w8, P8B + 1, v.s1);
    STORE(w8, P8B + 2, v.s2); STORE(w8, P8B + 3, v.s3);
    STORE(w8, P8B + 4, v.s4); STORE(w8, P8B + 5, v.s5);
    STORE(w8, P8B + 6, v.s6); STORE(w8, P8B + 7, v.s7);
    #undef P8B
}
