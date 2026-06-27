// Reference: model_info/transformers_src/modeling_gemma3.py — Gemma3 kernels
// (RMSNorm with (1+w) offset, scaled embedding, gelu-tanh, RoPE, GQA attention)
// Dtype-template preamble — copied verbatim from kernels/utils.cl.
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

// ── Custom nn.Linear GEMM: out[M,N] = x[M,K] @ W[N,K]^T ──
// Replaces CLBlast HGemm to eliminate its ~3.6ms/call HOST overhead. fp16
// storage, fp32 accumulation — same accumulation precision as CLBlast HGemm,
// so the near-tie argmax behaviour is preserved.
//
// COALESCING: one WORKGROUP per output (m,n); the GEMM_WG threads cooperatively
// reduce over K. Adjacent threads (lid, lid+1) read adjacent weights W[n*K+lid]
// and W[n*K+lid+1] → fully coalesced global loads (a one-work-item-per-output
// kernel read W with stride K across lanes → ~10-20x bandwidth loss). x[m,:] is
// broadcast across the workgroup (L2-hot). Launch: gws={N*GEMM_WG, M}, lws={GEMM_WG,1}.
#define GEMM_WG 64
// One workgroup (GEMM_WG threads) per output, reducing over K with a 4-way
// unrolled loop for memory-level parallelism. fp16 storage, fp32 accumulation
// (matches CLBlast HGemm precision → preserves the near-tie argmax). Adjacent
// lanes read adjacent W[n*K+k] → coalesced. On Adreno 620 this hits the GPU's
// fp16 matvec throughput ceiling (~2 G elem/s); transpose, int8 weights, output
// blocking, and float-x pre-convert were all measured and did NOT beat it.
__kernel void gemma3_gemm_linear(
    __global const storage_t* x,
    __global const storage_t* W,
    __global storage_t* out,
    const int M,
    const int N,
    const int K) {
    int n   = get_group_id(0);
    int m   = get_global_id(1);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    if (n >= N || m >= M) return;
    int xb = m * K;
    int wb = n * K;   // n*K <= 262143*640 ≈ 1.7e8 < INT_MAX; int math is faster
    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
    int k = lid;
    int s2 = lsz * 2, s3 = lsz * 3, s4 = lsz * 4;
    for (; k + s3 < K; k += s4) {
        p0 += (float)LOAD(x, xb + k)       * (float)LOAD(W, wb + k);
        p1 += (float)LOAD(x, xb + k + lsz) * (float)LOAD(W, wb + k + lsz);
        p2 += (float)LOAD(x, xb + k + s2)  * (float)LOAD(W, wb + k + s2);
        p3 += (float)LOAD(x, xb + k + s3)  * (float)LOAD(W, wb + k + s3);
    }
    for (; k < K; k += lsz)
        p0 += (float)LOAD(x, xb + k) * (float)LOAD(W, wb + k);
    float partial = (p0 + p1) + (p2 + p3);
    __local float scratch[128];   // sized for the largest swept reduction width
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) STORE(out, m * N + n, scratch[0]);
}

// ── GPU argmax over the logits (greedy decode) ──
// Avoids reading 262144 fp16 logits to host + a host f16→f32 convert loop + a
// host argmax loop every token (all with the GPU idle). One workgroup scans the
// vocab in a grid-stride, tracking (max,idx) per lane, then reduces in local mem.
// Reads back a single int. Decode is autoregressive so a sync is unavoidable,
// but this shrinks the host-side post-process from ~2×262144 ops to ~nothing.
#define ARGMAX_WG 256
__kernel void gemma3_argmax(
    __global const storage_t* logits,
    __global int* out_idx,
    const int n) {
    int lid = get_local_id(0);
    float best = -INFINITY;
    int   besti = 0;
    for (int i = lid; i < n; i += ARGMAX_WG) {
        float v = (float)LOAD(logits, i);
        if (v > best) { best = v; besti = i; }
    }
    __local float lv[ARGMAX_WG];
    __local int   li[ARGMAX_WG];
    lv[lid] = best; li[lid] = besti;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = ARGMAX_WG >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            // Tie-break on lower index to match a stable host argmax (first max).
            if (lv[lid + s] > lv[lid] || (lv[lid + s] == lv[lid] && li[lid + s] < li[lid])) {
                lv[lid] = lv[lid + s]; li[lid] = li[lid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) out_idx[0] = li[0];
}

// ── Fused multi-output GEMV: one dispatch produces up to 3 projections ──
// q/k/v (and gate/up) all read the SAME x but write separate outputs. Doing them
// as one GEMV over a concatenated weight [N0+N1+N2, K] collapses 3 (or 2) kernel
// launches into 1 — decode is dispatch-bound on the small projections (~480µs GPU
// launch overhead each), so this directly removes that overhead. Same coalesced
// workgroup-per-output reduction; each output routes to its buffer by N-boundary.
// Set N2=0 (and pass o1 for o2) for the 2-output (gate/up) case.
__kernel void gemma3_gemm_fused3(
    __global const storage_t* x,
    __global const storage_t* W,        // concatenated [N0+N1+N2, K], row-major
    __global storage_t* o0,
    __global storage_t* o1,
    __global storage_t* o2,
    const int M,
    const int N0,
    const int N1,
    const int N2,
    const int K) {
    int n   = get_group_id(0);
    int m   = get_global_id(1);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    int Ntot = N0 + N1 + N2;
    if (n >= Ntot || m >= M) return;
    int xb = m * K;
    int wb = n * K;
    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
    int k = lid;
    int s2 = lsz * 2, s3 = lsz * 3, s4 = lsz * 4;
    for (; k + s3 < K; k += s4) {
        p0 += (float)LOAD(x, xb + k)       * (float)LOAD(W, wb + k);
        p1 += (float)LOAD(x, xb + k + lsz) * (float)LOAD(W, wb + k + lsz);
        p2 += (float)LOAD(x, xb + k + s2)  * (float)LOAD(W, wb + k + s2);
        p3 += (float)LOAD(x, xb + k + s3)  * (float)LOAD(W, wb + k + s3);
    }
    for (; k < K; k += lsz)
        p0 += (float)LOAD(x, xb + k) * (float)LOAD(W, wb + k);
    float partial = (p0 + p1) + (p2 + p3);
    __local float scratch[128];
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        float v = scratch[0];
        if (n < N0)            STORE(o0, m * N0 + n,            v);
        else if (n < N0 + N1)  STORE(o1, m * N1 + (n - N0),     v);
        else                   STORE(o2, m * N2 + (n - N0 - N1), v);
    }
}

// Globally-coalesced transposed GEMV (like gemma3_gemv_t) but routes outputs to
// up to 3 buffers by N-boundary. Wt is the TRANSPOSED concatenated weight
// [K, N0+N1+N2]. x cached in local. Launch lws=256, grid = pad(Ntot, lws).
__kernel void gemma3_gemv_t_fused3(
    __global const storage_t* x,
    __global const storage_t* Wt,
    __global storage_t* o0,
    __global storage_t* o1,
    __global storage_t* o2,
    const int N0,
    const int N1,
    const int N2,
    const int K) {
    int n   = get_global_id(0);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    int Ntot = N0 + N1 + N2;
    __local float xs[2048];
    for (int k = lid; k < K; k += lsz) xs[k] = (float)LOAD(x, k);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (n >= Ntot) return;
    float acc = 0.0f;
    int kn = n;
    for (int k = 0; k < K; ++k, kn += Ntot) acc += xs[k] * (float)LOAD(Wt, kn);
    if (n < N0)            STORE(o0, n,            acc);
    else if (n < N0 + N1)  STORE(o1, n - N0,       acc);
    else                   STORE(o2, n - N0 - N1,  acc);
}

// ── Split-K transposed fused GEMV (proj: coalesced AND occupied at small N) ──
// The plain transposed GEMV (one thread/output) under-occupies for small proj N.
// Here a workgroup = TSK_TN(64) consecutive outputs × P K-splits: lanes 0..63 of
// each P-group share a k and read contiguous Wt[k*Ntot + n..] → 64-wide coalesced
// (like the bandwidth microbench), while P splits multiply work-items P× for
// occupancy and give each P× more loads/thread (better latency hiding than the
// WG=64 reduction's ~10 loads). A tiny local reduce combines the P partials.
// Wt is the TRANSPOSED concat weight [K, Ntot]. Launch lws=64*P, gws=ceil(Ntot/64)*lws.
#define TSK_TN 64
__kernel void gemma3_gemv_tsk_fused3(
    __global const storage_t* x,
    __global const storage_t* Wt,
    __global storage_t* o0,
    __global storage_t* o1,
    __global storage_t* o2,
    const int N0,
    const int N1,
    const int N2,
    const int K,
    const int P) {
    int Ntot = N0 + N1 + N2;
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    int n_idx = lid % TSK_TN;            // output within the tile
    int p     = lid / TSK_TN;            // which K-split
    int n     = get_group_id(0) * TSK_TN + n_idx;
    __local float xs[2048];              // x cached once (max K = 2048)
    for (int kk = lid; kk < K; kk += lsz) xs[kk] = (float)LOAD(x, kk);
    barrier(CLK_LOCAL_MEM_FENCE);
    float acc = 0.0f;
    if (n < Ntot) {
        int kper   = (K + P - 1) / P;
        int kstart = p * kper;
        int kend   = kstart + kper; if (kend > K) kend = K;
        long kn = (long)kstart * Ntot + n;
        for (int k = kstart; k < kend; ++k, kn += Ntot)
            acc += xs[k] * (float)LOAD(Wt, kn);
    }
    __local float part[TSK_TN][16];      // [output][split], P ≤ 16
    part[n_idx][p] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (p == 0 && n < Ntot) {
        float t = 0.0f;
        for (int pp = 0; pp < P; ++pp) t += part[n_idx][pp];
        if (n < N0)            STORE(o0, n,            t);
        else if (n < N0 + N1)  STORE(o1, n - N0,       t);
        else                   STORE(o2, n - N0 - N1,  t);
    }
}

// ── Subgroup-reduction GEMV (M==1 decode) ──
// Same coalesced workgroup-per-output structure as gemma3_gemm_linear, but the
// K-reduction uses the hardware sub_group_reduce_add (OpenCL 2.0, cl_khr_subgroups
// — confirmed on Adreno 620) instead of a 6-round local-memory barrier tree. The
// barriers were the tax pinning the reduction GEMV at ~4 GB/s vs the device's
// ~9 GB/s streaming ceiling. One barrier remains only to combine partial-subgroup
// sums when a workgroup spans >1 subgroup. fp32 accumulation preserves near-tie argmax.
// DISABLED: Adreno 620 advertises cl_khr_subgroups but its compiler rejects the
// sub_group_reduce_* built-ins ("OpenCL 2.0 built-in is not supported", -11) — and
// the driver defaults to CL2.0 kernel language, so an __OPENCL_C_VERSION__ guard
// does NOT exclude this; it broke the whole program build. Gated behind a macro we
// never define, so it stays dormant. (cl_qcom_subgroup_shuffle exists but a manual
// shuffle reduction would need unverified qcom intrinsic names — not worth the risk.)
#ifdef ENABLE_SG_GEMV
__kernel void gemma3_gemm_sg(
    __global const storage_t* x,
    __global const storage_t* W,
    __global storage_t* out,
    const int M,
    const int N,
    const int K) {
    int n   = get_group_id(0);
    int m   = get_global_id(1);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    if (n >= N || m >= M) return;
    int xb = m * K;
    int wb = n * K;
    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
    int k = lid;
    int s2 = lsz * 2, s3 = lsz * 3, s4 = lsz * 4;
    for (; k + s3 < K; k += s4) {
        p0 += (float)LOAD(x, xb + k)       * (float)LOAD(W, wb + k);
        p1 += (float)LOAD(x, xb + k + lsz) * (float)LOAD(W, wb + k + lsz);
        p2 += (float)LOAD(x, xb + k + s2)  * (float)LOAD(W, wb + k + s2);
        p3 += (float)LOAD(x, xb + k + s3)  * (float)LOAD(W, wb + k + s3);
    }
    for (; k < K; k += lsz)
        p0 += (float)LOAD(x, xb + k) * (float)LOAD(W, wb + k);
    float partial = (p0 + p1) + (p2 + p3);
    float sg = sub_group_reduce_add(partial);          // barrier-free hw reduction
    int nsg = get_num_sub_groups();
    if (nsg <= 1) {                                     // common case: 1 subgroup/WG
        if (lid == 0) STORE(out, m * N + n, sg);
        return;
    }
    __local float sgs[16];                              // combine across subgroups
    if (get_sub_group_local_id() == 0) sgs[get_sub_group_id()] = sg;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        float t = 0.0f;
        for (int i = 0; i < nsg; ++i) t += sgs[i];
        STORE(out, m * N + n, t);
    }
}
#endif

// ── Warp-synchronous GEMV (barrier-free reduction, no subgroup intrinsics) ──
// Same coalesced workgroup-per-output structure, but the K-reduction drops all
// 6 local-memory barriers: with lws==wavefront (64 on Adreno), the workgroup is
// a single SIMT wavefront executing in lockstep, so the reduction tree needs no
// barrier — `volatile` local memory prevents the compiler caching the partials.
// This is the standard CUDA warp-synchronous trick; it attacks the barrier tax
// that pins the reduction GEMV at ~4 GB/s vs the device's ~9 GB/s streaming
// ceiling, WITHOUT the subgroup built-ins this driver rejects. Launch lws=64.
__kernel void gemma3_gemm_warp(
    __global const storage_t* x,
    __global const storage_t* W,
    __global storage_t* out,
    const int M,
    const int N,
    const int K) {
    int n   = get_group_id(0);
    int m   = get_global_id(1);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    if (n >= N || m >= M) return;
    int xb = m * K, wb = n * K;
    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
    int k = lid;
    int s2 = lsz * 2, s3 = lsz * 3, s4 = lsz * 4;
    for (; k + s3 < K; k += s4) {
        p0 += (float)LOAD(x, xb + k)       * (float)LOAD(W, wb + k);
        p1 += (float)LOAD(x, xb + k + lsz) * (float)LOAD(W, wb + k + lsz);
        p2 += (float)LOAD(x, xb + k + s2)  * (float)LOAD(W, wb + k + s2);
        p3 += (float)LOAD(x, xb + k + s3)  * (float)LOAD(W, wb + k + s3);
    }
    for (; k < K; k += lsz)
        p0 += (float)LOAD(x, xb + k) * (float)LOAD(W, wb + k);
    __local volatile float s[64];
    s[lid] = (p0 + p1) + (p2 + p3);
    barrier(CLK_LOCAL_MEM_FENCE);        // single barrier to publish partials
    // Warp-synchronous tail: lanes 0..31 reduce 64→1 with no barriers (lockstep).
    if (lid < 32) {
        s[lid] += s[lid + 32];
        s[lid] += s[lid + 16];
        s[lid] += s[lid + 8];
        s[lid] += s[lid + 4];
        s[lid] += s[lid + 2];
        s[lid] += s[lid + 1];
    }
    if (lid == 0) STORE(out, m * N + n, s[0]);
}

// ── Shuffle-reduction GEMV (barrier-free K-reduction) ──
// Gated behind -D ENABLE_SHFL so a compile failure on this driver's subgroup
// intrinsics can't break the default program build. Replaces the 6 local-memory
// barrier rounds with a subgroup shuffle butterfly — the barrier tax is what
// pins the reduction GEMV at ~4 GB/s vs the device's ~9 GB/s streaming ceiling.
#ifdef ENABLE_SHFL
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
__kernel void gemma3_gemm_shfl(
    __global const storage_t* x,
    __global const storage_t* W,
    __global storage_t* out,
    const int M,
    const int N,
    const int K) {
    int n   = get_group_id(0);
    int m   = get_global_id(1);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    if (n >= N || m >= M) return;
    int xb = m * K, wb = n * K;
    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
    int k = lid;
    int s2 = lsz * 2, s3 = lsz * 3, s4 = lsz * 4;
    for (; k + s3 < K; k += s4) {
        p0 += (float)LOAD(x, xb + k)       * (float)LOAD(W, wb + k);
        p1 += (float)LOAD(x, xb + k + lsz) * (float)LOAD(W, wb + k + lsz);
        p2 += (float)LOAD(x, xb + k + s2)  * (float)LOAD(W, wb + k + s2);
        p3 += (float)LOAD(x, xb + k + s3)  * (float)LOAD(W, wb + k + s3);
    }
    for (; k < K; k += lsz)
        p0 += (float)LOAD(x, xb + k) * (float)LOAD(W, wb + k);
    float v = (p0 + p1) + (p2 + p3);
    uint sgs = get_sub_group_size();
    for (uint o = sgs >> 1; o > 0; o >>= 1) v += sub_group_shuffle_down(v, o);
    // v on each subgroup's lane 0 holds that subgroup's sum. Combine across
    // subgroups (workgroup may span >1) via a tiny local-mem step.
    __local float sgsum[8];
    if (get_sub_group_local_id() == 0) sgsum[get_sub_group_id()] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        float t = 0.0f;
        uint nsg = get_num_sub_groups();
        for (uint i = 0; i < nsg; ++i) t += sgsum[i];
        STORE(out, m * N + n, t);
    }
}
#endif

// ── Streaming memory-bandwidth microbenchmark (NNOPT_BWTEST) ──
// Reads n_elems fp16 weights once with the coalesced stride pattern (consecutive
// work-items read consecutive addresses each step) and sums them. Minimal
// compute → measures the device's achievable fp16 streaming bandwidth, i.e. the
// hard floor for any matvec (which must read every weight once).
__kernel void gemma3_bw_test(
    __global const storage_t* W,
    __global float* out,
    const int n_elems) {
    int gid = get_global_id(0);
    int gsz = get_global_size(0);
    float s = 0.0f;
    for (int i = gid; i < n_elems; i += gsz) s += (float)LOAD(W, i);
    out[gid] = s;
}

// ── Tiled transpose [N,K] row-major → [K,N] (one-time, for the transposed GEMV) ──
// Naive transpose has scattered writes (stride N) → ~0.5 GB/s, which put the
// 335MB lm_head transpose (~6s) into TTFT. Tiled via local memory makes BOTH the
// read and the write coalesced. Launch lws={16,16}, gws=pad(K,16)×pad(N,16).
#define TR_TILE 16
__kernel void gemma3_transpose(
    __global const storage_t* W,
    __global storage_t* Wt,
    const int N,
    const int K) {
    __local float tile[TR_TILE][TR_TILE + 1];   // +1 pads away bank conflicts
    int bx = get_group_id(0) * TR_TILE;         // along K
    int by = get_group_id(1) * TR_TILE;         // along N
    int tx = get_local_id(0), ty = get_local_id(1);
    int n = by + ty, k = bx + tx;               // read W[n,k]
    if (n < N && k < K) tile[ty][tx] = (float)LOAD(W, (long)n * K + k);
    barrier(CLK_LOCAL_MEM_FENCE);
    int kt = bx + ty, nt = by + tx;             // write Wt[kt,nt] = W[nt,kt]
    if (kt < K && nt < N) STORE(Wt, (long)kt * N + nt, tile[tx][ty]);
}

// ── Transposed GEMV: one work-item per output, no reduction/barriers ──
// Wt is [K,N]. out[n] = sum_k x[k] * Wt[k*N+n]. Adjacent work-items (n, n+1)
// read adjacent addresses Wt[k*N+n], Wt[k*N+n+1] → fully coalesced, like the
// bandwidth microbench. For huge N (lm_head, N=262144 ≈ the grid sweet spot)
// this captures the streaming bandwidth the workgroup-reduction GEMV cannot
// (its 6 barrier rounds cap it at ~2 G MAC/s vs the ~4.5 G MAC/s device floor).
// fp32 accumulation preserves the near-tie argmax.
__kernel void gemma3_gemv_t(
    __global const storage_t* x,
    __global const storage_t* Wt,
    __global storage_t* out,
    const int M,
    const int N,
    const int K) {
    // x cached in local once per workgroup (removes the per-thread global
    // broadcast). Then in-flight wavefronts read Wt row-by-row: at step k all
    // active lanes read contiguous Wt[k*N + n..] → globally coalesced like the
    // bandwidth microbench, which the per-output-workgroup reduction GEMV can't.
    int n   = get_global_id(0);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    __local float xs[2048];             // max K (down_proj interm = 2048)
    for (int k = lid; k < K; k += lsz) xs[k] = (float)LOAD(x, k);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (n >= N) return;
    float acc = 0.0f;
    int kn = n;                         // k*N + n, incremented by N each step
    for (int k = 0; k < K; ++k, kn += N)
        acc += xs[k] * (float)LOAD(Wt, kn);
    STORE(out, n, acc);
}

// ── Persistent-workgroup GEMV (M==1 decode) ──
// Instead of one workgroup per output (262144 workgroups for lm_head → heavy
// scheduling overhead, and x re-read from L2 per output), launch a FIXED small
// number of workgroups that grid-stride over outputs. x[0,:] is cached once in
// local memory and reused for every output the workgroup handles. Same 64-wide
// coalesced reduction over K, fp32 accumulation. Launch: gws={G*PWG}, lws={PWG}.
#define PWG 64
__kernel void gemma3_gemv_persist(
    __global const storage_t* x,
    __global const storage_t* W,
    __global storage_t* out,
    const int M,
    const int N,
    const int K) {
    int lid = get_local_id(0);
    int wg  = get_group_id(0);
    int ngroups = get_num_groups(0);
    __local float xs[2048];      // max K (down_proj interm = 2048)
    for (int k = lid; k < K; k += PWG) xs[k] = (float)LOAD(x, k);
    barrier(CLK_LOCAL_MEM_FENCE);
    __local float scratch[PWG];
    for (int n = wg; n < N; n += ngroups) {
        int wb = n * K;
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
        int k = lid;
        for (; k + PWG * 3 < K; k += PWG * 4) {
            p0 += xs[k]           * (float)LOAD(W, wb + k);
            p1 += xs[k + PWG]     * (float)LOAD(W, wb + k + PWG);
            p2 += xs[k + PWG * 2] * (float)LOAD(W, wb + k + PWG * 2);
            p3 += xs[k + PWG * 3] * (float)LOAD(W, wb + k + PWG * 3);
        }
        for (; k < K; k += PWG) p0 += xs[k] * (float)LOAD(W, wb + k);
        scratch[lid] = (p0 + p1) + (p2 + p3);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = PWG >> 1; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) STORE(out, n, scratch[0]);
        barrier(CLK_LOCAL_MEM_FENCE);   // protect scratch before next n
    }
}

// ── Gemma3RMSNorm ──
// _norm(x) = x * rsqrt(mean(x^2) + eps); out = _norm(x) * (1.0 + weight)
// Compute in fp32. weight stored as raw fp16/fp32.
//
// One WORKGROUP (RMS_WG threads) per row, cooperatively reducing over cols.
// At decode M=1 there is one row, so a one-work-item-per-row kernel ran the
// whole 640-element reduction on a single lane (memory-latency bound, ~515us).
// The workgroup reduction parallelises the reduction and coalesces the loads.
// Launch: gws = rows*RMS_WG, lws = RMS_WG.
#define RMS_WG 64
__kernel void gemma3_rmsnorm(
    __global const storage_t* x,
    __global const storage_t* weight,
    __global storage_t* out,
    const int cols,
    const float eps) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    int base = row * cols;
    float p = 0.0f;
    for (int i = lid; i < cols; i += lsz) { float v = LOAD(x, base + i); p += v * v; }
    __local float sc[RMS_WG];
    sc[lid] = p;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) sc[lid] += sc[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = rsqrt(sc[0] / (float)cols + eps);
    for (int i = lid; i < cols; i += lsz) {
        float v = LOAD(x, base + i);
        float w = LOAD(weight, i);
        STORE(out, base + i, (v * inv) * (1.0f + w));
    }
}

// ── Gemma3RMSNorm with fp32 RESIDUAL INPUT ──
// Identical math to gemma3_rmsnorm, but the input `x` is a raw fp32 buffer
// (the residual stream), NOT storage_t. Gemma3's residual stream grows past
// fp16 max (~65504) by mid-stack, so it MUST be kept in fp32; storing it as
// fp16 overflows to inf and the next norm produces NaN. Output is storage_t
// (the normalized values are small, well within fp16 range).
__kernel void gemma3_rmsnorm_f32in(
    __global const float* x,
    __global const storage_t* weight,
    __global storage_t* out,
    const int cols,
    const float eps) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    int base = row * cols;
    float p = 0.0f;
    for (int i = lid; i < cols; i += lsz) { float v = x[base + i]; p += v * v; }
    __local float sc[RMS_WG];
    sc[lid] = p;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) sc[lid] += sc[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = rsqrt(sc[0] / (float)cols + eps);
    for (int i = lid; i < cols; i += lsz) {
        float v = x[base + i];
        float w = LOAD(weight, i);
        STORE(out, base + i, (v * inv) * (1.0f + w));
    }
}

// ── Gemma3RMSNorm with fp32 OUTPUT (q/k norm precision path) ──
// Same math as gemma3_rmsnorm, but writes a raw fp32 buffer so the
// QK-norm result keeps full precision before RoPE and the QK^T dot.
// Reference computes _norm(x.float()) in fp32; storing the result fp16
// then reading it back in the attention dot loses ~3 decimal digits and
// flips the argmax on near-tie logits (15th vs 10th). Keeping q/k fp32
// through attention recovers it.
__kernel void gemma3_rmsnorm_f32out(
    __global const storage_t* x,
    __global const storage_t* weight,
    __global float* out,
    const int cols,
    const float eps) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    int base = row * cols;
    float p = 0.0f;
    for (int i = lid; i < cols; i += lsz) { float v = LOAD(x, base + i); p += v * v; }
    __local float sc[RMS_WG];
    sc[lid] = p;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) sc[lid] += sc[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = rsqrt(sc[0] / (float)cols + eps);
    for (int i = lid; i < cols; i += lsz) {
        float v = LOAD(x, base + i);
        float w = LOAD(weight, i);
        out[base + i] = (v * inv) * (1.0f + w);
    }
}

// ── Residual add into fp32 accumulator: out_f32[i] = a_f32[i] + b_f16[i] ──
// a is the fp32 residual stream, b is the (small) fp16 sublayer output. The
// result stays fp32 so the residual never overflows fp16 across the stack.
__kernel void gemma3_add_f32(
    __global const float* a,
    __global const storage_t* b,
    __global float* out,
    const int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    out[gid] = a[gid] + LOAD(b, gid);
}

// ── Convert fp16 storage → fp32 (seed the residual stream from embedding) ──
__kernel void gemma3_f16_to_f32(
    __global const storage_t* in,
    __global float* out,
    const int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    out[gid] = LOAD(in, gid);
}

// ── Scaled word embedding ──
// out[t, :] = embed_table[ids[t], :] * embed_scale
__kernel void gemma3_embed(
    __global const int* ids,
    __global const storage_t* table,
    __global storage_t* out,
    const int hidden,
    const float embed_scale) {
    int gid = get_global_id(0);          // index over seq*hidden
    int total = get_global_size(0);
    if (gid >= total) return;
    int t = gid / hidden;
    int d = gid - t * hidden;
    int tok = ids[t];
    float v = LOAD(table, (long)tok * hidden + d);
    STORE(out, gid, v * embed_scale);
}

// ── GELU (tanh approximation) ──
// gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
__kernel void gemma3_gelu_tanh(
    __global const storage_t* x,
    __global storage_t* out,
    const int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float v = LOAD(x, gid);
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    float inner = c * (v + 0.044715f * v * v * v);
    float g = 0.5f * v * (1.0f + tanh(inner));
    STORE(out, gid, g);
}

// ── elementwise multiply: out[i] = a[i]*b[i] ──
__kernel void gemma3_mul(
    __global const storage_t* a,
    __global const storage_t* b,
    __global storage_t* out,
    const int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    STORE(out, gid, LOAD(a, gid) * LOAD(b, gid));
}

// ── Apply RoPE in-place to a [seq, heads, head_dim] tensor (head-major within row) ──
// Layout: buf[t * (heads*head_dim) + h*head_dim + d]
// rotate_half: x1 = x[:hd/2], x2 = x[hd/2:]; out = x*cos + rotate_half(x)*sin
// where rotate_half(x) = cat(-x2, x1). cos/sin tables are [seq, head_dim].
__kernel void gemma3_rope(
    __global storage_t* buf,
    __global const storage_t* cos_tab,
    __global const storage_t* sin_tab,
    const int seq_len,
    const int heads,
    const int head_dim,
    const int start_pos) {
    int gid = get_global_id(0);          // over seq*heads*head_dim
    int total = seq_len * heads * head_dim;
    if (gid >= total) return;
    int d = gid % head_dim;
    int rest = gid / head_dim;
    int h = rest % heads;
    int t = rest / heads;
    (void)h;
    int half_hd = head_dim / 2;
    int pos = t + start_pos;
    float cs = LOAD(cos_tab, pos * head_dim + d);
    float sn = LOAD(sin_tab, pos * head_dim + d);
    float xv = LOAD(buf, gid);
    // rotate_half value at position d:
    float rot;
    if (d < half_hd) {
        // -x2 where x2 = buf[... d+half_hd]
        rot = -LOAD(buf, gid - d + (d + half_hd));
    } else {
        // x1 where x1 = buf[... d-half_hd]
        rot = LOAD(buf, gid - d + (d - half_hd));
    }
    STORE(buf, gid, xv * cs + rot * sn);
}

// ── Apply RoPE in-place to an fp32 [seq, heads, head_dim] tensor ──
// Identical math to gemma3_rope but operates on a raw fp32 buffer (the
// fp32 q/k from gemma3_rmsnorm_f32out). cos/sin tables are ALSO raw fp32
// (nnopt_get_rotary_tables_f32) — the reference computes RoPE entirely in
// fp32; storing the tables fp16 introduces ~2.4e-4 per-element error that
// accumulates through the short-wavelength sliding-window RoPE layers.
__kernel void gemma3_rope_f32(
    __global float* buf,
    __global const float* cos_tab,
    __global const float* sin_tab,
    const int seq_len,
    const int heads,
    const int head_dim,
    const int start_pos) {
    int gid = get_global_id(0);
    int total = seq_len * heads * head_dim;
    if (gid >= total) return;
    int d = gid % head_dim;
    int rest = gid / head_dim;
    int t = rest / heads;
    int half_hd = head_dim / 2;
    int pos = t + start_pos;
    float cs = cos_tab[pos * head_dim + d];
    float sn = sin_tab[pos * head_dim + d];
    float xv = buf[gid];
    float rot;
    if (d < half_hd) {
        rot = -buf[gid - d + (d + half_hd)];
    } else {
        rot = buf[gid - d + (d - half_hd)];
    }
    buf[gid] = xv * cs + rot * sn;
}

// ── Convert v projection (fp16 storage) → raw fp32 for the precision attn path ──
__kernel void gemma3_v_to_f32(
    __global const storage_t* v,
    __global float* out,
    const int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    out[gid] = LOAD(v, gid);
}

// ── GQA attention with FULLY fp32 q/k/v (precision path) ──
// q,k,v are raw fp32 buffers. The reference runs the whole attention in fp32
// (torch default dtype). Keeping q/k/v + dot/softmax/acc all fp32 and only
// narrowing to storage_t at the final STORE eliminates the per-layer drift
// that flips the argmax on near-tie logits.
//
// PARALLELISM: one WORKGROUP (ATTN_WG threads) per (query t, head h). The old
// kernel ran one work-item per (t,h) — at decode that's only seq_q*num_q_heads
// = 4 work-items doing the whole online-softmax serially (~1.7ms, latency-bound,
// no occupancy). Here ATTN_WG threads split head_dim for the QK dot (local
// reduction) and the V-accumulate. Online-softmax scalars are computed
// identically by every thread. Each thread owns head_dim/ATTN_WG output dims.
// Launch: gws = seq_q*num_q_heads*ATTN_WG, lws = ATTN_WG.
#define ATTN_WG 64
__kernel void gemma3_gqa_attn_f32qk(
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global storage_t* out,
    const int seq_q,
    const int seq_k,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scaling,
    const int start_pos,
    const int sliding_window) {
    int grp = get_group_id(0);
    if (grp >= seq_q * num_q_heads) return;
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    int h = grp % num_q_heads;
    int t = grp / num_q_heads;
    int nrep = num_q_heads / num_kv_heads;
    int kvh = h / nrep;
    int q_base = (t * num_q_heads + h) * head_dim;
    int q_abs = t + start_pos;

    // Cache q[head_dim] in local (read once per key otherwise).
    __local float qsh[256];      // head_dim <= 256 for this family
    for (int d = lid; d < head_dim; d += lsz) qsh[d] = q[q_base + d];
    barrier(CLK_LOCAL_MEM_FENCE);

    __local float red[ATTN_WG];
    float maxs = -INFINITY;
    float denom = 0.0f;
    float acc[8];                // head_dim/lsz partial outputs (<=256/64 → 4)
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

    for (int j = 0; j < seq_k; ++j) {
        if (j > q_abs) continue;                                   // causal
        if (sliding_window > 0 && (q_abs - j) >= sliding_window) continue;
        int k_base = (j * num_kv_heads + kvh) * head_dim;
        // Parallel dot(q, K[j]) over head_dim → local reduction → scalar.
        float pd = 0.0f;
        for (int d = lid; d < head_dim; d += lsz) pd += qsh[d] * k[k_base + d];
        red[lid] = pd;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = lsz >> 1; s > 0; s >>= 1) {
            if (lid < s) red[lid] += red[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float dot = red[0] * scaling;
        // Online softmax scalars — identical across all threads.
        float new_max = fmax(maxs, dot);
        float corr = exp(maxs - new_max);
        float p = exp(dot - new_max);
        denom = denom * corr + p;
        int v_base = (j * num_kv_heads + kvh) * head_dim;
        for (int i = 0, d = lid; d < head_dim; d += lsz, ++i)
            acc[i] = acc[i] * corr + p * v[v_base + d];
        maxs = new_max;
        barrier(CLK_LOCAL_MEM_FENCE);   // before red[] is reused next key
    }
    float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    int out_base = (t * num_q_heads + h) * head_dim;
    for (int i = 0, d = lid; d < head_dim; d += lsz, ++i)
        STORE(out, out_base + d, acc[i] * inv);
}

// ── GQA attention scores + softmax + weighted-sum ──
// q:   [seq, num_q_heads, head_dim]   (token-major, head within row)
// k,v: [seq_k, num_kv_heads, head_dim]
// out: [seq, num_q_heads, head_dim]
// One work-item per (query_row t, q_head h). Loops over k positions.
// scaling applied to scores; causal mask (and sliding window) applied.
__kernel void gemma3_gqa_attn(
    __global const storage_t* q,
    __global const storage_t* k,
    __global const storage_t* v,
    __global storage_t* out,
    const int seq_q,
    const int seq_k,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scaling,
    const int start_pos,
    const int sliding_window) {   // sliding_window<=0 means disabled
    int gid = get_global_id(0);          // over seq_q * num_q_heads
    int total = seq_q * num_q_heads;
    if (gid >= total) return;
    int h = gid % num_q_heads;
    int t = gid / num_q_heads;
    int nrep = num_q_heads / num_kv_heads;
    int kvh = h / nrep;
    int q_base = (t * num_q_heads + h) * head_dim;
    int q_abs = t + start_pos;           // absolute query position

    // Online softmax accumulation (single pass over keys). head_dim<=256.
    float maxs = -INFINITY;
    float denom = 0.0f;
    float acc[256];
    for (int d = 0; d < head_dim; ++d) acc[d] = 0.0f;

    for (int j = 0; j < seq_k; ++j) {
        if (j > q_abs) continue;         // causal
        if (sliding_window > 0 && (q_abs - j) >= sliding_window) continue;
        int k_base = (j * num_kv_heads + kvh) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot += LOAD(q, q_base + d) * LOAD(k, k_base + d);
        dot *= scaling;
        float new_max = fmax(maxs, dot);
        float corr = exp(maxs - new_max);
        float p = exp(dot - new_max);
        denom = denom * corr + p;
        int v_base = (j * num_kv_heads + kvh) * head_dim;
        for (int d = 0; d < head_dim; ++d)
            acc[d] = acc[d] * corr + p * LOAD(v, v_base + d);
        maxs = new_max;
    }
    float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    int out_base = (t * num_q_heads + h) * head_dim;
    for (int d = 0; d < head_dim; ++d)
        STORE(out, out_base + d, acc[d] * inv);
}
