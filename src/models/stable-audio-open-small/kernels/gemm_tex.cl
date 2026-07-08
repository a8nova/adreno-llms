// gemm_tex.cl — round-4 bespoke linear candidates: W via the Adreno L1
// TEXTURE cache (guide §6.2). Buffer-based W tiling is a measured dead end on
// this CU (register spill at MT=4×float8; local staging kills occupancy;
// see BENCHMARK.md). The texture path is the one weight-reuse mechanism not
// yet rejected: reads have no coalescing constraints and hit a dedicated L1.
//
// Layout: W[N, K] fp16 → image2d RGBA/HALF_FLOAT, width K/4 texels, height N.
// Texel (k4, n) = W[n, 4k4 .. 4k4+3]. Lane n reads column-adjacent texels to
// its neighbors — a 64-tall texel strip per wave step, ideal 2D locality.
//
// out[M,N] = x[M,K] @ W[N,K]^T; x stays a buffer (all lanes read the same
// x[m, k4] — uniform/broadcast load, L1-friendly). Accumulators are float4
// (the playbook's ≤4-accumulator register budget).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half storage_t;

// MT=4 rows per lane: 4 float4 accs — at the register budget, full mad rate.
__kernel void tex_linear_mt4(__global const storage_t* x,   // [M, K]
                             __read_only image2d_t W,       // K/4 x N texels
                             __global storage_t* out,       // [M, N]
                             const int M, const int N, const int K) {
    const int n = get_global_id(0);
    if (n >= N) return;
    const int m0 = get_group_id(1) * 4;
    const int mt = min(4, M - m0);
    const int K4 = K >> 2;

    float4 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int k4 = 0; k4 < K4; k4++) {
        const float4 w = convert_float4(read_imageh(W, (int2)(k4, n)));
        const size_t xb = (size_t)m0 * K + (k4 << 2);
        acc0 = mad(convert_float4(vload_half4(0, x + xb)), w, acc0);
        if (mt > 1) acc1 = mad(convert_float4(vload_half4(0, x + xb + K)), w, acc1);
        if (mt > 2) acc2 = mad(convert_float4(vload_half4(0, x + xb + 2 * K)), w, acc2);
        if (mt > 3) acc3 = mad(convert_float4(vload_half4(0, x + xb + 3 * K)), w, acc3);
    }
    #define SUM4(v) ((v).x + (v).y + (v).z + (v).w)
    vstore_half(SUM4(acc0), (size_t)m0 * N + n, out);
    if (mt > 1) vstore_half(SUM4(acc1), (size_t)(m0 + 1) * N + n, out);
    if (mt > 2) vstore_half(SUM4(acc2), (size_t)(m0 + 2) * N + n, out);
    if (mt > 3) vstore_half(SUM4(acc3), (size_t)(m0 + 3) * N + n, out);
}

// MT=2: halves the register pressure; W re-read 2× more (M/2 sweeps).
__kernel void tex_linear_mt2(__global const storage_t* x,
                             __read_only image2d_t W,
                             __global storage_t* out,
                             const int M, const int N, const int K) {
    const int n = get_global_id(0);
    if (n >= N) return;
    const int m0 = get_group_id(1) * 2;
    const int mt = min(2, M - m0);
    const int K4 = K >> 2;
    float4 acc0 = 0.0f, acc1 = 0.0f;
    for (int k4 = 0; k4 < K4; k4++) {
        const float4 w = convert_float4(read_imageh(W, (int2)(k4, n)));
        const size_t xb = (size_t)m0 * K + (k4 << 2);
        acc0 = mad(convert_float4(vload_half4(0, x + xb)), w, acc0);
        if (mt > 1) acc1 = mad(convert_float4(vload_half4(0, x + xb + K)), w, acc1);
    }
    vstore_half(SUM4(acc0), (size_t)m0 * N + n, out);
    if (mt > 1) vstore_half(SUM4(acc1), (size_t)(m0 + 1) * N + n, out);
}

// MT=8 with SCALAR accumulators (8 floats): W re-read only M/8 sweeps.
// The dot() reduction per step costs extra ALU vs mad4 — this variant tests
// whether W-traffic or ALU is the binding constraint at MT=8.
__kernel void tex_linear_mt8s(__global const storage_t* x,
                              __read_only image2d_t W,
                              __global storage_t* out,
                              const int M, const int N, const int K) {
    const int n = get_global_id(0);
    if (n >= N) return;
    const int m0 = get_group_id(1) * 8;
    const int mt = min(8, M - m0);
    const int K4 = K >> 2;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (int k4 = 0; k4 < K4; k4++) {
        const float4 w = convert_float4(read_imageh(W, (int2)(k4, n)));
        const size_t xb = (size_t)m0 * K + (k4 << 2);
        for (int r = 0; r < mt; r++)
            acc[r] += dot(convert_float4(vload_half4(0, x + xb + (size_t)r * K)), w);
    }
    for (int r = 0; r < mt; r++)
        vstore_half(acc[r], (size_t)(m0 + r) * N + n, out);
}

// ── v2: small local x-tile (2-4 KB — CLBlast's winning SB-tile size, far
// below the 12 KB that collapsed occupancy in the attn r8 experiment) with
// EXPLICITLY UNROLLED scalar accumulators (acc arrays indexed by a loop var
// go to scratch — measured 0.6 GFLOPS in v1's mt8s). W re-read M/16 times
// instead of M/4.
#define KC 64

#define STAGE_X(MT_) \
    for (int i = lid; i < MT_ * KC; i += 64) { \
        const int r = i / KC, kk = i % KC; \
        const int m = m0 + r; \
        xl[r * KC + kk] = (m < M) ? (float)vload_half((size_t)m * K + kc + kk, x) : 0.0f; \
    } \
    barrier(CLK_LOCAL_MEM_FENCE);

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void tex_local_m16(__global const storage_t* x,
                   __read_only image2d_t W,
                   __global storage_t* out,
                   const int M, const int N, const int K) {
    const int lid = get_local_id(0);
    const int n = get_group_id(0) * 64 + lid;
    const int m0 = get_group_id(1) * 16;
    __local float xl[16 * KC];
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    float a8=0,a9=0,a10=0,a11=0,a12=0,a13=0,a14=0,a15=0;
    for (int kc = 0; kc < K; kc += KC) {
        STAGE_X(16)
        if (n < N) {
            for (int k4 = 0; k4 < KC / 4; k4++) {
                const float4 w = convert_float4(read_imageh(W, (int2)(((kc >> 2) + k4), n)));
                const int xb = k4 << 2;
                a0  += dot(vload4(0, xl + 0*KC  + xb), w);
                a1  += dot(vload4(0, xl + 1*KC  + xb), w);
                a2  += dot(vload4(0, xl + 2*KC  + xb), w);
                a3  += dot(vload4(0, xl + 3*KC  + xb), w);
                a4  += dot(vload4(0, xl + 4*KC  + xb), w);
                a5  += dot(vload4(0, xl + 5*KC  + xb), w);
                a6  += dot(vload4(0, xl + 6*KC  + xb), w);
                a7  += dot(vload4(0, xl + 7*KC  + xb), w);
                a8  += dot(vload4(0, xl + 8*KC  + xb), w);
                a9  += dot(vload4(0, xl + 9*KC  + xb), w);
                a10 += dot(vload4(0, xl + 10*KC + xb), w);
                a11 += dot(vload4(0, xl + 11*KC + xb), w);
                a12 += dot(vload4(0, xl + 12*KC + xb), w);
                a13 += dot(vload4(0, xl + 13*KC + xb), w);
                a14 += dot(vload4(0, xl + 14*KC + xb), w);
                a15 += dot(vload4(0, xl + 15*KC + xb), w);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (n >= N) return;
    const float acc[16] = {a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15};
    for (int r = 0; r < 16 && m0 + r < M; r++)
        vstore_half(acc[r], (size_t)(m0 + r) * N + n, out);
}

// Same tiling, W from a packed BUFFER W4[k4][n][4] (coalesced across lanes) —
// isolates "texture cache" from "reuse tiling" as the active ingredient.
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void p4_local_m16(__global const storage_t* x,
                  __global const storage_t* W4,   // [K/4, N, 4]
                  __global storage_t* out,
                  const int M, const int N, const int K) {
    const int lid = get_local_id(0);
    const int n = get_group_id(0) * 64 + lid;
    const int m0 = get_group_id(1) * 16;
    __local float xl[16 * KC];
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    float a8=0,a9=0,a10=0,a11=0,a12=0,a13=0,a14=0,a15=0;
    for (int kc = 0; kc < K; kc += KC) {
        STAGE_X(16)
        if (n < N) {
            for (int k4 = 0; k4 < KC / 4; k4++) {
                const float4 w = convert_float4(
                    vload_half4(0, W4 + ((size_t)((kc >> 2) + k4) * N + n) * 4));
                const int xb = k4 << 2;
                a0  += dot(vload4(0, xl + 0*KC  + xb), w);
                a1  += dot(vload4(0, xl + 1*KC  + xb), w);
                a2  += dot(vload4(0, xl + 2*KC  + xb), w);
                a3  += dot(vload4(0, xl + 3*KC  + xb), w);
                a4  += dot(vload4(0, xl + 4*KC  + xb), w);
                a5  += dot(vload4(0, xl + 5*KC  + xb), w);
                a6  += dot(vload4(0, xl + 6*KC  + xb), w);
                a7  += dot(vload4(0, xl + 7*KC  + xb), w);
                a8  += dot(vload4(0, xl + 8*KC  + xb), w);
                a9  += dot(vload4(0, xl + 9*KC  + xb), w);
                a10 += dot(vload4(0, xl + 10*KC + xb), w);
                a11 += dot(vload4(0, xl + 11*KC + xb), w);
                a12 += dot(vload4(0, xl + 12*KC + xb), w);
                a13 += dot(vload4(0, xl + 13*KC + xb), w);
                a14 += dot(vload4(0, xl + 14*KC + xb), w);
                a15 += dot(vload4(0, xl + 15*KC + xb), w);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (n >= N) return;
    const float acc[16] = {a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15};
    for (int r = 0; r < 16 && m0 + r < M; r++)
        vstore_half(acc[r], (size_t)(m0 + r) * N + n, out);
}

// ── v3: 2D lane split — 64 lanes = 16 n-columns × 4 row-groups; each lane
// owns (n, 4 rows) with 4 float4 accumulators → full mad4 rate (v2's scalar
// dot() reduction ran at half rate), while the WG still amortizes W over a
// 16-row x-tile. The 4 lanes sharing one n read the same texel per step —
// adjacent-lane duplicate reads are exactly what the texture L1 absorbs.
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void tex_local_16x16(__global const storage_t* x,
                     __read_only image2d_t W,
                     __global storage_t* out,
                     const int M, const int N, const int K) {
    const int lid = get_local_id(0);
    const int nl  = lid & 15;                    // n within the WG tile
    const int mg  = lid >> 4;                    // row-group 0..3
    const int n   = get_group_id(0) * 16 + nl;
    const int m0  = get_group_id(1) * 16;
    const int mr  = m0 + mg * 4;                 // this lane's first row
    __local float xl[16 * KC];

    float4 c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    for (int kc = 0; kc < K; kc += KC) {
        STAGE_X(16)
        if (n < N) {
            for (int k4 = 0; k4 < KC / 4; k4++) {
                const float4 w = convert_float4(read_imageh(W, (int2)(((kc >> 2) + k4), n)));
                const int xb = (mg * 4) * KC + (k4 << 2);
                c0 = mad(vload4(0, xl + xb), w, c0);
                c1 = mad(vload4(0, xl + xb + KC), w, c1);
                c2 = mad(vload4(0, xl + xb + 2 * KC), w, c2);
                c3 = mad(vload4(0, xl + xb + 3 * KC), w, c3);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (n >= N) return;
    if (mr     < M) vstore_half(SUM4(c0), (size_t)(mr    ) * N + n, out);
    if (mr + 1 < M) vstore_half(SUM4(c1), (size_t)(mr + 1) * N + n, out);
    if (mr + 2 < M) vstore_half(SUM4(c2), (size_t)(mr + 2) * N + n, out);
    if (mr + 3 < M) vstore_half(SUM4(c3), (size_t)(mr + 3) * N + n, out);
}
