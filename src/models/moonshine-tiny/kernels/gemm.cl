// GEMM/GEMV kernel family (OPT-4/9/10/11) + one-time weight transpose.
// Owned by src/ops/Linear.cpp (linear_gemm_t4 also used by Conv1d.cpp im2col path).
// Split from the former kernels/moonshine.cl monolith (whisper-parity layout);
// kernel bodies are byte-identical to the monolith at the time of the split.
// Dtype-template preamble — DO NOT EDIT. Driven by host-side -DNNOPT_USE_FP16.
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

// ── linear GEMV (M=1 decode path): out[N] = W[N,K] · x[K] ───────────────
// nn.Linear layout (W rows are output features). One workgroup per output
// row; lanes stride the row in 4-element chunks (coalesced across lanes),
// fp32 accumulate via dot(), local-mem tree reduction, lane 0 stores.
// Replaces CLBlast Gemm at M=1, which ran at ~1% of DRAM bandwidth on
// Adreno 620 (920µs for a 288×288 row-scan — r3 profile).
// One-time transpose W[N,K] → WT[K,N] (feeds linear_gemv_t below).
// Tiled 16×16 through local memory: reads AND writes are coalesced (the
// naive 1-elem/work-item scatter ran at ~0.5 GB/s — 196ms of TTFT across
// the 100 weight tensors).
#define TR_TILE 16
__kernel void mat_transpose(
    __global const storage_t* W,      // [N, K]
    __global storage_t* WT,           // [K, N]
    const int N,
    const int K,
    __local float* tile) {            // [TR_TILE * (TR_TILE+1)]
    const int lk = get_local_id(0);   // 0..15 along K (input cols)
    const int ln = get_local_id(1);   // 0..15 along N (input rows)
    const int k0 = get_group_id(0) * TR_TILE;
    const int n0 = get_group_id(1) * TR_TILE;
    const int in_k = k0 + lk;
    const int in_n = n0 + ln;
    if (in_k < K && in_n < N)
        tile[ln * (TR_TILE + 1) + lk] = LOAD(W, (size_t)in_n * K + in_k);
    barrier(CLK_LOCAL_MEM_FENCE);
    const int out_n = n0 + lk;        // swapped roles → coalesced store
    const int out_k = k0 + ln;
    if (out_n < N && out_k < K)
        STORE(WT, (size_t)out_k * N + out_n, tile[lk * (TR_TILE + 1) + ln]);
}

// ── linear GEMV, transposed-weight variant (huge N, e.g. proj_out) ──────
// out[N] = WT[K,N]^T-view · x[K]. Column-major access: each lane owns 4
// consecutive outputs, so every k-iteration's loads are fully coalesced
// across the wave (the row-scan variant above hits 32768 barriers there).
// x is staged in local memory once per WG.
__kernel void linear_gemv_t(
    __global const storage_t* x,       // [K]
    __global const storage_t* WT,      // [K, N]
    __global storage_t* out,           // [N]
    const int N,
    const int K,
    __local float* lx) {               // [K]
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    for (int k = lid; k < K; k += lsz) lx[k] = LOAD(x, k);
    barrier(CLK_LOCAL_MEM_FENCE);
    const int n4 = get_global_id(0);   // 4-wide output tile index
    const int N4 = N >> 2;
    if (n4 >= N4) return;
    float4 acc = (float4)(0.0f);
    for (int k = 0; k < K; ++k) {
#ifdef USE_FP16
        acc += lx[k] * vload_half4((size_t)k * N4 + n4, WT);
#else
        acc += lx[k] * vload4((size_t)k * N4 + n4, WT);
#endif
    }
#ifdef USE_FP16
    vstore_half4(acc, n4, out);
#else
    vstore4(acc, n4, out);
#endif
}

// ── K-split GEMV (decode, small N): partials + reduce (OPT-11, r10) ──────
// For N=288 the K-major GEMV runs only N/4=72 lanes (2 workgroups) — the
// GPU is starved. Split K into S chunks across get_group_id(1); each chunk
// writes fp32 partials, a second tiny kernel reduces. fp32 partial buffer
// keeps the accumulation precision of the single-pass kernel.
__kernel void linear_gemv_ks(
    __global const storage_t* x,       // [K]
    __global const storage_t* WT,      // [K, N]
    __global float* partial,           // [S, N] fp32
    const int N,
    const int K,
    const int S) {
    const int n4 = get_global_id(0);
    const int s = get_group_id(1);
    const int N4 = N >> 2;
    if (n4 >= N4) return;
    const int chunk = (K + S - 1) / S;
    const int k0 = s * chunk;
    const int k1 = min(k0 + chunk, K);
    float4 acc = (float4)(0.0f);
    for (int k = k0; k < k1; ++k) {
#ifdef USE_FP16
        acc += LOAD(x, k) * vload_half4((size_t)k * N4 + n4, WT);
#else
        acc += LOAD(x, k) * vload4((size_t)k * N4 + n4, WT);
#endif
    }
    vstore4(acc, s * N4 + n4, partial);
}

__kernel void gemv_ks_reduce(
    __global const float* partial,     // [S, N]
    __global storage_t* out,           // [N]
    const int N,
    const int S) {
    const int n = get_global_id(0);
    if (n >= N) return;
    float acc = 0.0f;
    for (int s = 0; s < S; ++s) acc += partial[s * N + n];
    STORE(out, n, acc);
}

// ── linear GEMM, transposed-weight, 4-row-tiled (encoder / prefill M>1) ──
// out[M,N] = x[M,K] @ WT[K,N]. Each workgroup owns a (4-row, 256-col) output
// tile: x rows staged in local per 256-deep k-chunk, each lane carries 4
// float4 accumulators (playbook: ≤4 accs — more spills on Adreno), so WT is
// read once per 4 rows instead of once per row. The untiled variant re-read
// conv2's 2.3MB weight 210× (487MB → 52ms/call).
#define GEMM_MT 4
#define GEMM_KT 256
__kernel void linear_gemm_t4(
    __global const storage_t* x,       // [M, K]
    __global const storage_t* WT,      // [K, N]
    __global storage_t* out,           // [M, N]
    const int M,
    const int N,
    const int K,
    __local float* lx) {               // [GEMM_MT * GEMM_KT]
    const int n4 = get_global_id(0);
    const int N4 = N >> 2;
    const int m0 = get_group_id(1) * GEMM_MT;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int mt_n = min(GEMM_MT, M - m0);
    float4 acc0 = (float4)(0.0f), acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f), acc3 = (float4)(0.0f);
    for (int kc = 0; kc < K; kc += GEMM_KT) {
        const int kl = min(GEMM_KT, K - kc);
        for (int i = lid; i < mt_n * kl; i += lsz) {
            int mt = i / kl;
            int kk = i - mt * kl;
            lx[mt * GEMM_KT + kk] = LOAD(x, (size_t)(m0 + mt) * K + kc + kk);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (n4 < N4) {
            for (int k = 0; k < kl; ++k) {
#ifdef USE_FP16
                float4 wv = vload_half4((size_t)(kc + k) * N4 + n4, WT);
#else
                float4 wv = vload4((size_t)(kc + k) * N4 + n4, WT);
#endif
                acc0 += lx[k] * wv;
                if (mt_n > 1) acc1 += lx[GEMM_KT + k] * wv;
                if (mt_n > 2) acc2 += lx[2 * GEMM_KT + k] * wv;
                if (mt_n > 3) acc3 += lx[3 * GEMM_KT + k] * wv;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (n4 >= N4) return;
#ifdef USE_FP16
    vstore_half4(acc0, (size_t)m0 * N4 + n4, out);
    if (mt_n > 1) vstore_half4(acc1, (size_t)(m0 + 1) * N4 + n4, out);
    if (mt_n > 2) vstore_half4(acc2, (size_t)(m0 + 2) * N4 + n4, out);
    if (mt_n > 3) vstore_half4(acc3, (size_t)(m0 + 3) * N4 + n4, out);
#else
    vstore4(acc0, (size_t)m0 * N4 + n4, out);
    if (mt_n > 1) vstore4(acc1, (size_t)(m0 + 1) * N4 + n4, out);
    if (mt_n > 2) vstore4(acc2, (size_t)(m0 + 2) * N4 + n4, out);
    if (mt_n > 3) vstore4(acc3, (size_t)(m0 + 3) * N4 + n4, out);
#endif
}

// ── linear GEMM, transposed-weight (encoder / prefill M>1) ──────────────
// out[M,N] = x[M,K] @ WT[K,N]. Same coalesced K-major walk as linear_gemv_t
// with one workgroup-row per (m, 64-lane n-tile); x row staged in local.
// Superseded by linear_gemm_t4 for the hot paths; kept as fallback.
__kernel void linear_gemm_t(
    __global const storage_t* x,       // [M, K]
    __global const storage_t* WT,      // [K, N]
    __global storage_t* out,           // [M, N]
    const int M,
    const int N,
    const int K,
    __local float* lx) {               // [K]
    const int m = get_group_id(1);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    for (int k = lid; k < K; k += lsz) lx[k] = LOAD(x, m * K + k);
    barrier(CLK_LOCAL_MEM_FENCE);
    const int n4 = get_global_id(0);
    const int N4 = N >> 2;
    if (n4 >= N4) return;
    float4 acc = (float4)(0.0f);
    for (int k = 0; k < K; ++k) {
#ifdef USE_FP16
        acc += lx[k] * vload_half4((size_t)k * N4 + n4, WT);
#else
        acc += lx[k] * vload4((size_t)k * N4 + n4, WT);
#endif
    }
#ifdef USE_FP16
    vstore_half4(acc, (size_t)m * N4 + n4, out);
#else
    vstore4(acc, (size_t)m * N4 + n4, out);
#endif
}

__kernel void linear_gemv(
    __global const storage_t* x,       // [K]
    __global const storage_t* W,       // [N, K]
    __global storage_t* out,           // [N]
    const int N,
    const int K,
    __local float* lacc) {
    const int row = get_group_id(0);
    if (row >= N) return;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    __global const storage_t* wrow = W + (size_t)row * K;
    float acc = 0.0f;
    const int K4 = K >> 2;
    for (int i = lid; i < K4; i += lsz) {
#ifdef USE_FP16
        acc += dot(vload_half4(i, x), vload_half4(i, wrow));
#else
        acc += dot(vload4(i, x), vload4(i, wrow));
#endif
    }
    for (int k = (K4 << 2) + lid; k < K; k += lsz)   // tail when K % 4 != 0
        acc += LOAD(x, k) * LOAD(wrow, k);
    lacc[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) lacc[lid] += lacc[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) STORE(out, row, lacc[0]);
}
