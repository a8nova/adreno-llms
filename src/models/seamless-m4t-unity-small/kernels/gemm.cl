// Dense ops: linear (matmul+bias), layernorm, GLU, inference batchnorm.

// Cooperative M=1 GEMV for the single-token decode hot path (text/unit decoders).
// out[o] = bias[o] + sum_k x[k] * w[o, k]   (PyTorch Linear weight [N, K]).
//
// The register-blocked `linear_fast` launches only ceil(N/4) work-items at M=1
// (e.g. 192 for a 768-wide proj), each serially reducing K — far too few to hide
// DRAM latency on the single-CU Adreno 620, so decode runs at ~4% of the memory
// roofline. This kernel instead assigns ONE work-group (64 threads = exactly one
// A6xx wave) per output row: the 64 threads stride over K with coalesced 128-bit
// (half8) loads, accumulate in fp32, then tree-reduce in local memory. That puts
// N×64 work-items in flight (49 152 for N=768, 640 960 for the 10 015-wide
// lm_head) — enough occupancy to saturate the bus. The reduce barriers are
// intra-wave on WG=64, so they compile to near-no-ops. Proven on Adreno 620
// (adreno-llms/src/models/lfm2-5-350m gemv_m1.cl: ~5% → ~30% of roofline).
#ifndef COOP_WG
#define COOP_WG 64
#endif
// Tiled-GEMM block sizes (shared by linear_gemm_tiled + linear_gemm_tiled_int8).
#define TG_BM 32
#define TG_BN 32
#define TG_BK 16
__kernel __attribute__((reqd_work_group_size(COOP_WG, 1, 1)))
void gemv_coop(__global const storage_t* x,     // [K]
               __global const storage_t* w,     // [N, K] row-major
               __global const storage_t* bias,  // [N] (or dummy when !has_bias)
               __global storage_t* out,         // [N]
               const int M, const int K, const int N, const int has_bias) {
    __local float ls[COOP_WG];
    const int row = get_group_id(0);
    const int tid = get_local_id(0);
    if (row >= N) return;
    const int wb = row * K;
    const int nv = K >> 3;            // # of half8 chunks
    float acc = 0.0f;
    for (int c = tid; c < nv; c += COOP_WG) {
        int k = c << 3;
        acc += dot8(LOAD8(x, k), LOAD8(w, wb + k));
    }
    int kt = nv << 3;                 // scalar tail (K not a multiple of 8)
    if (tid == 0) for (int k = kt; k < K; ++k) acc += (float)LOAD(x, k) * (float)LOAD(w, wb + k);
    ls[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = COOP_WG >> 1; s > 0; s >>= 1) {
        if (tid < s) ls[tid] += ls[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        float v = ls[0];
        if (has_bias) v += (float)LOAD(bias, row);
        STORE(out, row, v);
    }
}

// 4-outputs-per-WG cooperative GEMV. One WG (64 threads) computes 4 consecutive
// output rows, loading each x half8 once and feeding 4 independent fp32 dot
// chains (register-level parallelism, x amortized 4×, 4× fewer WGs + reduces than
// gemv_coop). Proven faster on Adreno 620 for narrow K where the single-output
// reduce dominates. Requires N % 4 == 0 (every proj/lm_head N here is).
__kernel __attribute__((reqd_work_group_size(COOP_WG, 1, 1)))
void gemv_coop4(__global const storage_t* x,     // [K]
                __global const storage_t* w,     // [N, K] row-major
                __global const storage_t* bias,  // [N] (or dummy)
                __global storage_t* out,         // [N]
                const int M, const int K, const int N, const int has_bias) {
    __local float ls[COOP_WG * 4];
    const int n0  = get_group_id(0) * 4;
    const int tid = get_local_id(0);
    if (n0 >= N) return;
    const int wb0 = n0 * K;
    const int nv = K >> 3;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int c = tid; c < nv; c += COOP_WG) {
        int k = c << 3;
        float8 xv = LOAD8(x, k);
        a0 += dot8(xv, LOAD8(w, wb0 + k));
        a1 += dot8(xv, LOAD8(w, wb0 + K + k));
        a2 += dot8(xv, LOAD8(w, wb0 + 2 * K + k));
        a3 += dot8(xv, LOAD8(w, wb0 + 3 * K + k));
    }
    ls[tid] = a0; ls[tid + COOP_WG] = a1; ls[tid + 2 * COOP_WG] = a2; ls[tid + 3 * COOP_WG] = a3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = COOP_WG >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            ls[tid] += ls[tid + s];
            ls[tid + COOP_WG] += ls[tid + COOP_WG + s];
            ls[tid + 2 * COOP_WG] += ls[tid + 2 * COOP_WG + s];
            ls[tid + 3 * COOP_WG] += ls[tid + 3 * COOP_WG + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid < 4 && (n0 + tid) < N) {
        float v = ls[tid * COOP_WG];
        if (has_bias) v += (float)LOAD(bias, n0 + tid);
        STORE(out, n0 + tid, v);
    }
}

// ===== int8 weight variants (per-row symmetric quant: w_i8[N,K] + scale[N]). =====
// Halve weight DRAM traffic (1 byte vs 2) → up to 2× on the memory-bound decode
// GEMVs and encoder FFN. Activations stay fp16; accumulate fp32; multiply the
// per-output-row scale once at the end. char8 vload + convert_float8 keeps the
// 128-bit coalesced read pattern.
__kernel __attribute__((reqd_work_group_size(COOP_WG, 1, 1)))
void gemv_coop_int8(__global const storage_t* x, __global const char* w,
                    __global const storage_t* scale, __global const storage_t* bias,
                    __global storage_t* out, const int M, const int K, const int N, const int has_bias) {
    __local float ls[COOP_WG];
    const int row = get_group_id(0), tid = get_local_id(0);
    if (row >= N) return;
    const int wb = row * K, nv = K >> 3;
    float acc = 0.0f;
    for (int c = tid; c < nv; c += COOP_WG)
        acc += dot8(LOAD8(x, c << 3), convert_float8(vload8(c, w + wb)));
    int kt = nv << 3;
    if (tid == 0) for (int k = kt; k < K; ++k) acc += (float)LOAD(x, k) * (float)(w[wb + k]);
    ls[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = COOP_WG >> 1; s > 0; s >>= 1) { if (tid < s) ls[tid] += ls[tid + s]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (tid == 0) { float v = ls[0] * (float)LOAD(scale, row); if (has_bias) v += (float)LOAD(bias, row); STORE(out, row, v); }
}

__kernel __attribute__((reqd_work_group_size(COOP_WG, 1, 1)))
void gemv_coop4_int8(__global const storage_t* x, __global const char* w,
                     __global const storage_t* scale, __global const storage_t* bias,
                     __global storage_t* out, const int M, const int K, const int N, const int has_bias) {
    __local float ls[COOP_WG * 4];
    const int n0 = get_group_id(0) * 4, tid = get_local_id(0);
    if (n0 >= N) return;
    const int nv = K >> 3;
    const int wb0 = n0 * K;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int c = tid; c < nv; c += COOP_WG) {
        float8 xv = LOAD8(x, c << 3);
        a0 += dot8(xv, convert_float8(vload8(c, w + wb0)));
        a1 += dot8(xv, convert_float8(vload8(c, w + wb0 + K)));
        a2 += dot8(xv, convert_float8(vload8(c, w + wb0 + 2 * K)));
        a3 += dot8(xv, convert_float8(vload8(c, w + wb0 + 3 * K)));
    }
    ls[tid] = a0; ls[tid + COOP_WG] = a1; ls[tid + 2 * COOP_WG] = a2; ls[tid + 3 * COOP_WG] = a3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = COOP_WG >> 1; s > 0; s >>= 1) {
        if (tid < s) { ls[tid] += ls[tid + s]; ls[tid + COOP_WG] += ls[tid + COOP_WG + s];
                       ls[tid + 2 * COOP_WG] += ls[tid + 2 * COOP_WG + s]; ls[tid + 3 * COOP_WG] += ls[tid + 3 * COOP_WG + s]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid < 4 && (n0 + tid) < N) {
        float v = ls[tid * COOP_WG] * (float)LOAD(scale, n0 + tid);
        if (has_bias) v += (float)LOAD(bias, n0 + tid);
        STORE(out, n0 + tid, v);
    }
}

// int8 tiled GEMM (encoder M>=32). Same tiling as linear_gemm_tiled; weight strip
// loaded from int8 (Bs holds dequant-free int8-as-float), per-row scale at store.
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void linear_gemm_tiled_int8(__global const storage_t* x, __global const char* w,
                            __global const storage_t* scale, __global const storage_t* bias,
                            __global storage_t* out, const int M, const int K, const int N, const int has_bias) {
    __local float As[TG_BM][TG_BK];
    __local float Bs[TG_BK][TG_BN];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bn0 = get_group_id(0) * TG_BN, bm0 = get_group_id(1) * TG_BM;
    const int tid = ty * 8 + tx;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TG_BK) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = tid + i * 64;
            int r = idx >> 4, kk = idx & 15;
            int gm = bm0 + r;
            As[r][kk] = (gm < M) ? (float)LOAD(x, gm * K + k0 + kk) : 0.0f;
            int kk2 = idx >> 5, c = idx & 31;
            int gn = bn0 + c;
            Bs[kk2][c] = (gn < N) ? (float)(w[gn * K + k0 + kk2]) : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int kk = 0; kk < TG_BK; ++kk) {
            float av[4], bv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) av[i] = As[ty * 4 + i][kk];
            #pragma unroll
            for (int j = 0; j < 4; ++j) bv[j] = Bs[kk][tx * 4 + j];
            #pragma unroll
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m = bm0 + ty * 4 + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int o = bn0 + tx * 4 + j;
            if (o < N) { float v = acc[i][j] * (float)LOAD(scale, o);
                         if (has_bias) v += (float)LOAD(bias, o); STORE(out, m * N + o, v); }
        }
    }
}

// Broadcast bias add: out[m, n] += bias[n]. Used after CLBlast HGemm (which has no
// bias) for the encoder linears. global = M*N.
__kernel void bias_add(__global storage_t* out, __global const storage_t* bias,
                       const int M, const int N) {
    int gid = get_global_id(0);
    if (gid >= M * N) return;
    int n = gid - (gid / N) * N;
    STORE(out, gid, (float)LOAD(out, gid) + (float)LOAD(bias, n));
}

// Channel-major broadcast bias: out[c, t] += bias[c]  (vocoder convs after CLBlast).
__kernel void bias_add_ct(__global storage_t* out, __global const storage_t* bias,
                          const int C, const int T) {
    int gid = get_global_id(0);
    if (gid >= C * T) return;
    int c = gid / T;
    STORE(out, gid, (float)LOAD(out, gid) + (float)LOAD(bias, c));
}

// out[t, o] = bias[o] + sum_i x[t, i] * w[o, i]   (PyTorch Linear weight [Dout, Din])
__kernel void linear_forward(__global const storage_t* x,
                             __global const storage_t* w,
                             __global const storage_t* bias,
                             __global storage_t* out,
                             const int T, const int Din, const int Dout,
                             const int has_bias) {
    int gid = get_global_id(0);
    if (gid >= T * Dout) return;
    int t = gid / Dout;
    int o = gid - t * Dout;
    float acc = has_bias ? (float)LOAD(bias, o) : 0.0f;
    int xb = t * Din, wb = o * Din;
    for (int i = 0; i < Din; ++i) acc += (float)LOAD(x, xb + i) * (float)LOAD(w, wb + i);
    STORE(out, gid, acc);
}

// Vectorized, N-register-blocked linear: out[m,o] = bias[o] + sum_k x[m,k]*w[o,k].
// Each work-item computes LIN_TN consecutive outputs for one row m, loading the
// activation chunk x8 once (128-bit fp16) and reusing it across the LIN_TN weight
// rows. K loop is half8-vectorized (Adreno: 2× fp16 ALU + 128-bit bandwidth).
// global = (ceil(N/LIN_TN), M). Weight is row-major [N, K]; K is a multiple of 8 for
// every layer in this model (a scalar tail handles any leftover for safety).
#define LIN_TN 4
__kernel void linear_fast(__global const storage_t* x,
                          __global const storage_t* w,
                          __global const storage_t* bias,
                          __global storage_t* out,
                          const int M, const int K, const int N, const int has_bias) {
    int m = get_global_id(1);
    if (m >= M) return;
    int o0 = get_global_id(0) * LIN_TN;
    float acc[LIN_TN];
    #pragma unroll
    for (int i = 0; i < LIN_TN; ++i) acc[i] = (has_bias && (o0 + i) < N) ? (float)LOAD(bias, o0 + i) : 0.0f;
    int xb = m * K;
    int kv = K & ~7;
    for (int k = 0; k < kv; k += 8) {
        float8 xv = LOAD8(x, xb + k);
        #pragma unroll
        for (int i = 0; i < LIN_TN; ++i) {
            int o = o0 + i;
            if (o < N) acc[i] += dot8(xv, LOAD8(w, o * K + k));
        }
    }
    for (int k = kv; k < K; ++k) {
        float xvv = (float)LOAD(x, xb + k);
        #pragma unroll
        for (int i = 0; i < LIN_TN; ++i) { int o = o0 + i; if (o < N) acc[i] += xvv * (float)LOAD(w, o * K + k); }
    }
    #pragma unroll
    for (int i = 0; i < LIN_TN; ++i) { int o = o0 + i; if (o < N) STORE(out, m * N + o, acc[i]); }
}

// Cooperative small-M GEMM (text beam, M=B=5). One work-group (64 threads = one
// wave) per output channel n: the 64 threads cooperate over K, each holding M fp32
// accumulators (one per activation row), so the weight half8 is loaded ONCE and
// reused across all M rows. N×64 work-items → far better occupancy than the 4×4
// register kernel at M=5 (which launched only ~N/4×2 work-items and wasted 3/4 of
// each M-tile). M<=GM_MAXM. out[m,n] = bias[n] + sum_k x[m,k]*w[n,k].
#define GM_MAXM 8
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gemm_coop_m(__global const storage_t* x, __global const storage_t* w,
                 __global const storage_t* bias, __global storage_t* out,
                 const int M, const int K, const int N, const int has_bias) {
    __local float ls[64 * GM_MAXM];
    const int n = get_group_id(0), tid = get_local_id(0);
    if (n >= N) return;
    const int wb = n * K, nv = K >> 3;
    float acc[GM_MAXM];
    #pragma unroll
    for (int m = 0; m < GM_MAXM; ++m) acc[m] = 0.0f;
    for (int c = tid; c < nv; c += 64) {
        float8 wv = LOAD8(w, wb + (c << 3));
        for (int m = 0; m < M; ++m) acc[m] += dot8(LOAD8(x, m * K + (c << 3)), wv);
    }
    for (int m = 0; m < M; ++m) ls[tid + m * 64] = acc[m];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int p = 32; p > 0; p >>= 1) {
        if (tid < p) for (int m = 0; m < M; ++m) ls[tid + m * 64] += ls[tid + p + m * 64];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        float bv = has_bias ? (float)LOAD(bias, n) : 0.0f;
        for (int m = 0; m < M; ++m) STORE(out, m * N + n, ls[m * 64] + bv);
    }
}

// M×N register-blocked GEMM for the encoder/teacher-forced forwards (M>1). Each
// work-item computes a GEMM_TM×GEMM_TN output tile: per K-chunk it loads GEMM_TM
// activation rows (x8) and GEMM_TN weight rows (w8), doing TM*TN fma's — so each
// weight chunk is reused across TM rows and each activation chunk across TN outputs.
// global = (ceil(N/GEMM_TN), ceil(M/GEMM_TM)). Weight row-major [N,K], K multiple of 8.
#define GEMM_TM 4
#define GEMM_TN 4

// BW-optimal small-M GEMM (M in [2..GEMVM_MAX]): ONE work-item per output channel o.
// Loads each weight row w[o,:] EXACTLY ONCE into a register (float8) and reuses it across
// all M activation rows — vs linear_gemm which reads the weight M/TM× and linear_fast M×.
// On a memory-bandwidth-bound device the weight is the dominant traffic (e.g. lm_head
// N=20005×K=768 = 30 MB), so reading it once is the win. x[m,:] is tiny and stays hot in
// L1/texture across the N work-items. out is [M, N] row-major: out[m*N + o].
// Conv GEMM (vocoder): out[Cout,T] = W[Cout,CinK] @ col[CinK,T] — both row-major, NO transpose
// (C=A@B, unlike linear_gemm's A@B^T). Replaces the CLBlast conv call, which measured ~30 ms of
// HOST overhead per call (CONVPROF) for ~1 ms of GPU work — ~2.75 s of vocoder idle across ~91 convs.
// Register tile CG_TM output-channels × CG_TN time-steps; weight A[m,k] reused across TN, col B[k,n]
// reused across TM. bias folded in. global = (ceil(T/CG_TN), ceil(Cout/CG_TM)).
#define CG_TM 4
#define CG_TN 4
__kernel void conv_gemm_ct(__global const storage_t* w,    // W[Cout, CinK]
                           __global const storage_t* col,  // col[CinK, T]
                           __global const storage_t* bias, // [Cout]
                           __global storage_t* out,        // out[Cout, T]
                           const int Cout, const int CinK, const int T, const int has_bias) {
    int co0 = get_global_id(1) * CG_TM;
    int t0  = get_global_id(0) * CG_TN;
    float acc[CG_TM][CG_TN];
    #pragma unroll
    for (int i = 0; i < CG_TM; ++i)
        #pragma unroll
        for (int j = 0; j < CG_TN; ++j)
            acc[i][j] = (has_bias && (co0 + i) < Cout) ? (float)LOAD(bias, co0 + i) : 0.0f;
    for (int k = 0; k < CinK; ++k) {
        float a[CG_TM];
        #pragma unroll
        for (int i = 0; i < CG_TM; ++i) { int co = co0 + i; a[i] = (co < Cout) ? (float)LOAD(w, co * CinK + k) : 0.0f; }
        const int cb = k * T;
        #pragma unroll
        for (int j = 0; j < CG_TN; ++j) {
            int t = t0 + j;
            if (t < T) { float b = (float)LOAD(col, cb + t);
                #pragma unroll
                for (int i = 0; i < CG_TM; ++i) acc[i][j] += a[i] * b; }
        }
    }
    #pragma unroll
    for (int i = 0; i < CG_TM; ++i) { int co = co0 + i; if (co < Cout)
        #pragma unroll
        for (int j = 0; j < CG_TN; ++j) { int t = t0 + j; if (t < T) STORE(out, co * T + t, acc[i][j]); } }
}

// M-exact small-M GEMM (M in [2..8]): ONE work-item computes ALL M rows × LGM_TN output
// channels. The weight row w[o,:] is loaded ONCE per WI and reused across all M rows (vs
// linear_gemm's TM=4 which reads weight M/4× = 2× for M=5 AND wastes the half-empty 2nd
// M-tile). x[m,:] (8 vals) reused across LGM_TN channels. acc[8][TN] in registers. Unlike
// gemv_m (TN=1, no x-reuse → no MLP, regressed), this keeps TN=4 for register reuse.
// global = ceil(N / LGM_TN). out is [M,N] row-major.
#define LGM_TN 4
__kernel void linear_gemm_m(__global const storage_t* x, __global const storage_t* w,
                            __global const storage_t* bias, __global storage_t* out,
                            const int M, const int K, const int N, const int has_bias) {
    int o0 = get_global_id(0) * LGM_TN;
    float acc[8][LGM_TN];
    #pragma unroll
    for (int m = 0; m < 8; ++m)
        #pragma unroll
        for (int tn = 0; tn < LGM_TN; ++tn)
            acc[m][tn] = (has_bias && (o0 + tn) < N) ? (float)LOAD(bias, o0 + tn) : 0.0f;
    int kv = K & ~7;
    for (int k = 0; k < kv; k += 8) {
        float8 wv[LGM_TN];
        #pragma unroll
        for (int tn = 0; tn < LGM_TN; ++tn) { int o = o0 + tn; wv[tn] = (o < N) ? LOAD8(w, o * K + k) : (float8)(0.0f); }
        for (int m = 0; m < M; ++m) {
            float8 xv = LOAD8(x, m * K + k);
            #pragma unroll
            for (int tn = 0; tn < LGM_TN; ++tn) acc[m][tn] += dot8(xv, wv[tn]);
        }
    }
    for (int k = kv; k < K; ++k) {
        float wvv[LGM_TN];
        #pragma unroll
        for (int tn = 0; tn < LGM_TN; ++tn) { int o = o0 + tn; wvv[tn] = (o < N) ? (float)LOAD(w, o * K + k) : 0.0f; }
        for (int m = 0; m < M; ++m) {
            float xv = (float)LOAD(x, m * K + k);
            #pragma unroll
            for (int tn = 0; tn < LGM_TN; ++tn) acc[m][tn] += xv * wvv[tn];
        }
    }
    for (int m = 0; m < M; ++m)
        #pragma unroll
        for (int tn = 0; tn < LGM_TN; ++tn) { int o = o0 + tn; if (o < N) STORE(out, m * N + o, acc[m][tn]); }
}

#define GEMVM_MAX 8
__kernel void gemv_m(__global const storage_t* x, __global const storage_t* w,
                     __global const storage_t* bias, __global storage_t* out,
                     const int M, const int K, const int N, const int has_bias) {
    int o = get_global_id(0);
    if (o >= N) return;
    float acc[GEMVM_MAX];
    #pragma unroll
    for (int m = 0; m < GEMVM_MAX; ++m) acc[m] = 0.0f;
    const int wb = o * K, kv = K & ~7;
    for (int k = 0; k < kv; k += 8) {
        float8 wv = LOAD8(w, wb + k);          // weight chunk read ONCE, reused across all M
        for (int m = 0; m < M; ++m) acc[m] += dot8(LOAD8(x, m * K + k), wv);
    }
    for (int k = kv; k < K; ++k) {
        float wvv = (float)LOAD(w, wb + k);
        for (int m = 0; m < M; ++m) acc[m] += (float)LOAD(x, m * K + k) * wvv;
    }
    float b = has_bias ? (float)LOAD(bias, o) : 0.0f;
    for (int m = 0; m < M; ++m) STORE(out, m * N + o, acc[m] + b);
}

__kernel void linear_gemm(__global const storage_t* x,
                          __global const storage_t* w,
                          __global const storage_t* bias,
                          __global storage_t* out,
                          const int M, const int K, const int N, const int has_bias) {
    int row0 = get_global_id(1) * GEMM_TM;
    int o0 = get_global_id(0) * GEMM_TN;
    float acc[GEMM_TM][GEMM_TN];
    #pragma unroll
    for (int tm = 0; tm < GEMM_TM; ++tm)
        #pragma unroll
        for (int tn = 0; tn < GEMM_TN; ++tn)
            acc[tm][tn] = (has_bias && (o0 + tn) < N) ? (float)LOAD(bias, o0 + tn) : 0.0f;
    int kv = K & ~7;
    for (int k = 0; k < kv; k += 8) {
        float8 xv[GEMM_TM];
        #pragma unroll
        for (int tm = 0; tm < GEMM_TM; ++tm) { int m = row0 + tm; xv[tm] = (m < M) ? LOAD8(x, m * K + k) : (float8)(0.0f); }
        #pragma unroll
        for (int tn = 0; tn < GEMM_TN; ++tn) {
            int o = o0 + tn;
            if (o < N) {
                float8 wv = LOAD8(w, o * K + k);
                #pragma unroll
                for (int tm = 0; tm < GEMM_TM; ++tm) acc[tm][tn] += dot8(xv[tm], wv);
            }
        }
    }
    for (int k = kv; k < K; ++k) {
        #pragma unroll
        for (int tn = 0; tn < GEMM_TN; ++tn) {
            int o = o0 + tn;
            if (o < N) { float wv = (float)LOAD(w, o * K + k);
                #pragma unroll
                for (int tm = 0; tm < GEMM_TM; ++tm) { int m = row0 + tm; if (m < M) acc[tm][tn] += (float)LOAD(x, m * K + k) * wv; } }
        }
    }
    #pragma unroll
    for (int tm = 0; tm < GEMM_TM; ++tm) { int m = row0 + tm; if (m < M)
        #pragma unroll
        for (int tn = 0; tn < GEMM_TN; ++tn) { int o = o0 + tn; if (o < N) STORE(out, m * N + o, acc[tm][tn]); } }
}

// Barrier-free register GEMM, 4×8 tile: each of the 4 x-rows is loaded once per
// K-chunk (half8) and reused across 8 output columns → 2× the x-reuse of linear_gemm
// without any __local (guide §6.1.4: barrier-free → max workgroup → more resident
// waves; §6.5: no local → higher occupancy). Used for the encoder (M>=32).
// global = (ceil(N/8), ceil(M/4)).
__kernel void linear_gemm8(__global const storage_t* x, __global const storage_t* w,
                           __global const storage_t* bias, __global storage_t* out,
                           const int M, const int K, const int N, const int has_bias) {
    int row0 = get_global_id(1) * 4;
    int o0 = get_global_id(0) * 8;
    float acc[4][8];
    #pragma unroll
    for (int tm = 0; tm < 4; ++tm)
        #pragma unroll
        for (int tn = 0; tn < 8; ++tn) acc[tm][tn] = (has_bias && (o0 + tn) < N) ? (float)LOAD(bias, o0 + tn) : 0.0f;
    int kv = K & ~7;
    for (int k = 0; k < kv; k += 8) {
        float8 xv0 = (row0 + 0 < M) ? LOAD8(x, (row0 + 0) * K + k) : (float8)(0.0f);
        float8 xv1 = (row0 + 1 < M) ? LOAD8(x, (row0 + 1) * K + k) : (float8)(0.0f);
        float8 xv2 = (row0 + 2 < M) ? LOAD8(x, (row0 + 2) * K + k) : (float8)(0.0f);
        float8 xv3 = (row0 + 3 < M) ? LOAD8(x, (row0 + 3) * K + k) : (float8)(0.0f);
        #pragma unroll
        for (int tn = 0; tn < 8; ++tn) {
            int o = o0 + tn;
            if (o < N) {
                float8 wv = LOAD8(w, o * K + k);
                acc[0][tn] += dot8(xv0, wv); acc[1][tn] += dot8(xv1, wv);
                acc[2][tn] += dot8(xv2, wv); acc[3][tn] += dot8(xv3, wv);
            }
        }
    }
    #pragma unroll
    for (int tm = 0; tm < 4; ++tm) { int m = row0 + tm; if (m < M)
        #pragma unroll
        for (int tn = 0; tn < 8; ++tn) { int o = o0 + tn; if (o < N) STORE(out, m * N + o, acc[tm][tn]); } }
}

// int8 weight variant of linear_gemm8 (per-row symmetric quant: w_i8[N,K]+scale[N]).
// The barrier-free register GEMM re-reads weights from DRAM per M-tile (no local
// reuse) → it IS weight-bandwidth-bound, so halving weight bytes (int8) gives a real
// speedup here. char8 vload + convert_float8 keeps the 128-bit coalesced read.
__kernel void linear_gemm8_int8(__global const storage_t* x, __global const char* w,
                                __global const storage_t* scale, __global const storage_t* bias,
                                __global storage_t* out,
                                const int M, const int K, const int N, const int has_bias) {
    int row0 = get_global_id(1) * 4;
    int o0 = get_global_id(0) * 8;
    float acc[4][8];
    #pragma unroll
    for (int tm = 0; tm < 4; ++tm)
        #pragma unroll
        for (int tn = 0; tn < 8; ++tn) acc[tm][tn] = 0.0f;
    int kv = K & ~7;
    for (int k = 0; k < kv; k += 8) {
        float8 xv0 = (row0 + 0 < M) ? LOAD8(x, (row0 + 0) * K + k) : (float8)(0.0f);
        float8 xv1 = (row0 + 1 < M) ? LOAD8(x, (row0 + 1) * K + k) : (float8)(0.0f);
        float8 xv2 = (row0 + 2 < M) ? LOAD8(x, (row0 + 2) * K + k) : (float8)(0.0f);
        float8 xv3 = (row0 + 3 < M) ? LOAD8(x, (row0 + 3) * K + k) : (float8)(0.0f);
        #pragma unroll
        for (int tn = 0; tn < 8; ++tn) {
            int o = o0 + tn;
            if (o < N) {
                float8 wv = convert_float8(vload8((o * K + k) >> 3, w));
                acc[0][tn] += dot8(xv0, wv); acc[1][tn] += dot8(xv1, wv);
                acc[2][tn] += dot8(xv2, wv); acc[3][tn] += dot8(xv3, wv);
            }
        }
    }
    #pragma unroll
    for (int tm = 0; tm < 4; ++tm) { int m = row0 + tm; if (m < M)
        #pragma unroll
        for (int tn = 0; tn < 8; ++tn) { int o = o0 + tn; if (o < N) {
            float v = acc[tm][tn] * (float)LOAD(scale, o);
            if (has_bias) v += (float)LOAD(bias, o);
            STORE(out, m * N + o, v); } } }
}

// ── Local-memory tiled GEMM for the encoder (M>=32, e.g. T=299 conformer). ──
// out[m,o] = bias[o] + sum_k x[m,k]*w[o,k];  x:[M,K], w:[N,K] row-major.
// The register-tiled `linear_gemm` is memory-bound (~2 MACs per fp16 loaded — every
// x/w chunk re-fetched from DRAM). This stages a 32×16 strip of x and a 16×32 strip
// of w into local memory per K-step; the 64 threads (= one Adreno wave) then reuse
// them for 16 MACs/element loaded (~8× arithmetic intensity). Each thread owns a
// 4×4 micro-tile. WG = 8×8 = 64. K is a multiple of 16 for every encoder GEMM
// (768, 1536, 3072, 4096); M/N guards handle the ragged T=299 / adaptor T=38 edges.
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void linear_gemm_tiled(__global const storage_t* x,
                       __global const storage_t* w,
                       __global const storage_t* bias,
                       __global storage_t* out,
                       const int M, const int K, const int N, const int has_bias) {
    // BK=16. (BK=8 to double occupancy was tried — flat; image-weight reads and
    // int8 also fail to help. The M=299 encoder GEMM is at its practical floor on
    // this single-CU GPU: compute/occupancy-bound, ~4.5 GFLOP/s.)
    #define LGT_BK 16
    __local float As[TG_BM][LGT_BK];   // x strip  [32][16]
    __local float Bs[LGT_BK][TG_BN];   // w strip  [16][32]
    const int tx = get_local_id(0);   // 0..7  -> N micro-col group
    const int ty = get_local_id(1);   // 0..7  -> M micro-row group
    const int bn0 = get_group_id(0) * TG_BN;
    const int bm0 = get_group_id(1) * TG_BM;
    const int tid = ty * 8 + tx;      // 0..63 flat

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += LGT_BK) {
        // Cooperative load: 512 As (32×16) + 512 Bs (16×32) = 8 each per 64 threads.
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = tid + i * 64;            // 0..511
            int r = idx >> 4, kk = idx & 15;   // As[32][16]
            int gm = bm0 + r;
            As[r][kk] = (gm < M) ? (float)LOAD(x, gm * K + k0 + kk) : 0.0f;
            int kk2 = idx >> 5, c = idx & 31;  // Bs[16][32]
            int gn = bn0 + c;
            Bs[kk2][c] = (gn < N) ? (float)LOAD(w, gn * K + k0 + kk2) : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int kk = 0; kk < LGT_BK; ++kk) {
            float av[4], bv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) av[i] = As[ty * 4 + i][kk];
            #pragma unroll
            for (int j = 0; j < 4; ++j) bv[j] = Bs[kk][tx * 4 + j];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m = bm0 + ty * 4 + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int o = bn0 + tx * 4 + j;
            if (o < N) {
                float v = acc[i][j];
                if (has_bias) v += (float)LOAD(bias, o);
                STORE(out, m * N + o, v);
            }
        }
    }
}

// Cooperative LayerNorm: one work-group (64 threads = one wave) per row, instead
// of 1 work-item per row (which left decode's T=1 norms running on a SINGLE thread
// — 8.6 s / 1759 calls in the GPU profile). Two-pass (mean, var) with vectorized
// half8 loads + fp32 tree-reduce. D multiple of 8 (768/160 here).
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void layernorm_coop(__global const storage_t* x, __global const storage_t* g,
                    __global const storage_t* b, __global storage_t* out,
                    const int T, const int D, const float eps) {
    __local float ls[64];
    const int row = get_group_id(0), tid = get_local_id(0);
    if (row >= T) return;
    const int base = row * D, nv = D >> 3;
    float s = 0.0f;
    for (int c = tid; c < nv; c += 64) { float8 v = LOAD8(x, base + (c << 3)); s += v.s0+v.s1+v.s2+v.s3+v.s4+v.s5+v.s6+v.s7; }
    ls[tid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int p = 32; p > 0; p >>= 1) { if (tid < p) ls[tid] += ls[tid + p]; barrier(CLK_LOCAL_MEM_FENCE); }
    float mean = ls[0] / (float)D;
    barrier(CLK_LOCAL_MEM_FENCE);
    float sq = 0.0f;
    for (int c = tid; c < nv; c += 64) { float8 v = LOAD8(x, base + (c << 3)) - mean; v = v * v; sq += v.s0+v.s1+v.s2+v.s3+v.s4+v.s5+v.s6+v.s7; }
    ls[tid] = sq;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int p = 32; p > 0; p >>= 1) { if (tid < p) ls[tid] += ls[tid + p]; barrier(CLK_LOCAL_MEM_FENCE); }
    float inv = 1.0f / sqrt(ls[0] / (float)D + eps);
    for (int c = tid; c < nv; c += 64) {
        int o = base + (c << 3);
        float8 v = (LOAD8(x, o) - mean) * inv * LOAD8(g, c << 3) + LOAD8(b, c << 3);
        STORE8(out, o, v);
    }
}

__kernel void layernorm_forward(__global const storage_t* x,
                                __global const storage_t* g,
                                __global const storage_t* b,
                                __global storage_t* out,
                                const int T, const int D, const float eps) {
    int t = get_global_id(0);
    if (t >= T) return;
    int base = t * D;
    float mean = 0.0f;
    for (int i = 0; i < D; ++i) mean += (float)LOAD(x, base + i);
    mean /= (float)D;
    float var = 0.0f;
    for (int i = 0; i < D; ++i) { float d = (float)LOAD(x, base + i) - mean; var += d * d; }
    var /= (float)D;
    float inv = 1.0f / sqrt(var + eps);
    for (int i = 0; i < D; ++i) {
        float v = ((float)LOAD(x, base + i) - mean) * inv * (float)LOAD(g, i) + (float)LOAD(b, i);
        STORE(out, base + i, v);
    }
}

// Time-major chunk(2,dim=-1) GLU: out[t,c] = in[t,c] * sigmoid(in[t, C+c]). in is [T, 2C].
__kernel void glu_tc(__global const storage_t* in, __global storage_t* out,
                     const int T, const int C) {
    int gid = get_global_id(0);
    if (gid >= T * C) return;
    int t = gid / C, c = gid - t * C;
    float a = (float)LOAD(in, t * 2 * C + c);
    float bg = (float)LOAD(in, t * 2 * C + C + c);
    STORE(out, gid, a * (1.0f / (1.0f + native_exp(-bg))));
}

// Time-major [T, C] inference batchnorm: out = (in-mean)/sqrt(var+eps)*g + b.
__kernel void batchnorm_tc(__global const storage_t* in, __global storage_t* out,
                           __global const storage_t* mean, __global const storage_t* var,
                           __global const storage_t* g, __global const storage_t* b,
                           const int T, const int C, const float eps) {
    int gid = get_global_id(0);
    if (gid >= T * C) return;
    int c = gid - (gid / C) * C;
    float v = ((float)LOAD(in, gid) - (float)LOAD(mean, c)) / sqrt((float)LOAD(var, c) + eps)
              * (float)LOAD(g, c) + (float)LOAD(b, c);
    STORE(out, gid, v);
}
