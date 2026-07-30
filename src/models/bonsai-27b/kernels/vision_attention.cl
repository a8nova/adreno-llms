// Depth-Anything-V2 (DINOv2 backbone + DPT neck/head) OpenCL kernels.
// One FILE = one cl_program on purpose: Adreno applies a program-wide
// worst-case register footprint, so a fat kernel in a shared program cuts
// wave occupancy for every kernel in it (measured 2026-07-07: frame
// 6.6s -> 10.5s from two never-dispatched kernels). Keep op families
// isolated. All data buffers are storage_t (fp16 or fp32); accumulators
// are float.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)   vload_half((i),(p))
  #define STORE(p,i,v) vstore_half((v),(i),(p))
  #define LOAD4(p,i)   vload_half4(0, (p) + (i))
  #define STORE4(p,i,v) vstore_half4((v), 0, (p) + (i))
#else
  typedef float storage_t;
  #define LOAD(p,i)   ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
  #define LOAD4(p,i)   vload4(0, (p) + (i))
  #define STORE4(p,i,v) vstore4((v), 0, (p) + (i))
#endif


// ── attention scores: Q[H,M,hd] x K[H,M,hd]^T -> scores[H,M,M], scaled ──
// q,k row-major head-major [H, M, hd]. one thread per (h,i,j).

__kernel void attn_scores(
    __global const storage_t* q,
    __global const storage_t* k,
    __global storage_t* scores,
    const int H, const int M, const int hd, const float scale)
{
    int idx = get_global_id(0);
    int total = H * M * M;
    if (idx >= total) return;
    int j = idx % M;
    int i = (idx / M) % M;
    int h = idx / (M * M);
    int qb = (h * M + i) * hd;
    int kb = (h * M + j) * hd;
    float acc = 0.0f;
    for (int d = 0; d < hd; d++) acc += (float)LOAD(q, qb + d) * (float)LOAD(k, kb + d);
    STORE(scores, idx, acc * scale);
}

// ── attention output: probs[H,M,M] x V[H,M,hd] -> ctx token-major [M, H*hd] ──
// PyTorch reshapes context (transpose(1,2)) to [M, H*hd]. Write directly to
// token-major so out_proj GEMM reads contiguous [M, D].

__kernel void attn_context(
    __global const storage_t* probs,
    __global const storage_t* v,
    __global storage_t* ctx,
    const int H, const int M, const int hd)
{
    int idx = get_global_id(0);
    int total = M * H * hd;   // token-major [M, H, hd]
    if (idx >= total) return;
    int d = idx % hd;
    int h = (idx / hd) % H;
    int i = idx / (hd * H);
    int pb = (h * M + i) * M;   // probs[h,i,:]
    float acc = 0.0f;
    for (int j = 0; j < M; j++) {
        float p = (float)LOAD(probs, pb + j);
        float vv = (float)LOAD(v, (h * M + j) * hd + d);
        acc += p * vv;
    }
    // ctx token-major: [i, h, d] -> i*(H*hd) + h*hd + d
    STORE(ctx, i * (H * hd) + h * hd + d, acc);
}

// ── transpose token-major [M, H, hd] to head-major [H, M, hd] ──
// input q/k/v come out of GEMM as [M, D] with D=H*hd (token-major). scores
// kernel wants head-major. one thread per element.

__kernel void to_head_major(
    __global const storage_t* in_tm,   // [M, H*hd]
    __global storage_t* out_hm,        // [H, M, hd]
    const int M, const int H, const int hd)
{
    int idx = get_global_id(0);
    int total = M * H * hd;
    if (idx >= total) return;
    int d = idx % hd;
    int h = (idx / hd) % H;
    int i = idx / (hd * H);
    float v = (float)LOAD(in_tm, i * (H * hd) + h * hd + d);
    STORE(out_hm, (h * M + i) * hd + d, v);
}

// ── fused QKV split + head-major transpose ────────────────────────────────
// The vision tower's attn_qkv is ONE Linear emitting [M][3*D] with q at 0, k at D, v at 2D — so q/k/v
// are strided, not contiguous, and to_head_major above (which assumes a contiguous [M, H*hd]) cannot
// be used on them directly. Doing the split and the transpose in one pass avoids materialising three
// intermediate token-major copies.
// Apply rope_vision to the qkv buffer BEFORE this (row_stride = 3*D, head_off = 0 for q, D for k).
__kernel void qkv_to_head_major(__global const storage_t* qkv,   // [M, 3*D]
                                __global storage_t* q_hm,        // [H, M, hd]
                                __global storage_t* k_hm,
                                __global storage_t* v_hm,
                                const int M, const int H, const int hd)
{
    const int idx = get_global_id(0);
    const int D = H * hd;
    if (idx >= M * D) return;
    const int d = idx % hd;
    const int h = (idx / hd) % H;
    const int i = idx / D;
    const size_t row = (size_t)i * 3 * D + (size_t)h * hd + d;
    const size_t o = ((size_t)h * M + i) * hd + d;
    STORE(q_hm, o, (float)LOAD(qkv, row));
    STORE(k_hm, o, (float)LOAD(qkv, row + D));
    STORE(v_hm, o, (float)LOAD(qkv, row + 2 * D));
}

// ── TILED attention: bounded memory for long patch sequences ───────────────
// The untiled path allocates scores[H][M][M]. For one full-resolution photo M is ~3840 patches
// (before the 2x2 merge), so that is 16*3840*3840*2 = 472 MB — which, on top of the resident text
// tower, exceeded the device budget and took the phone down with a system OOM.
// Here only QT query rows are in flight, so the buffer is H*QT*M and memory no longer grows with M^2.
__kernel void attn_scores_tile(__global const storage_t* q,   // [H, M, hd]
                               __global const storage_t* k,
                               __global storage_t* scores,    // [H, qn, M]
                               const int H, const int M, const int hd,
                               const float scale, const int q0, const int qn)
{
    const int idx = get_global_id(0);
    if (idx >= H * qn * M) return;
    const int j = idx % M;
    const int r = (idx / M) % qn;
    const int h = idx / (qn * M);
    const size_t qb = ((size_t)h * M + (q0 + r)) * hd;
    const size_t kb = ((size_t)h * M + j) * hd;
    float acc = 0.0f;
    for (int d = 0; d < hd; ++d) acc += (float)LOAD(q, qb + d) * (float)LOAD(k, kb + d);
    STORE(scores, (size_t)(h * qn + r) * M + j, acc * scale);
}

// One row of length M per work-item. Non-causal: no mask, every patch sees every patch.
__kernel void attn_softmax_tile(__global storage_t* scores, const int H, const int M, const int qn)
{
    const int row = get_global_id(0);
    if (row >= H * qn) return;
    const size_t b = (size_t)row * M;
    float mx = -1e30f;
    for (int j = 0; j < M; ++j) mx = fmax(mx, (float)LOAD(scores, b + j));
    float sum = 0.0f;
    for (int j = 0; j < M; ++j) { const float e = exp((float)LOAD(scores, b + j) - mx);
                                  STORE(scores, b + j, e); sum += e; }
    const float inv = 1.0f / sum;
    for (int j = 0; j < M; ++j) STORE(scores, b + j, (float)LOAD(scores, b + j) * inv);
}

// ── FUSED attention: scores + softmax + context in one workgroup ───────────
// Replaces the attn_scores_tile / attn_softmax_tile / attn_context_tile trio, which cost two full
// round-trips of the [H][qn][M] scores buffer (218 MB per tile at 3040 patches) and, worse, ran the
// softmax as ONE work-item per row doing three serial passes over M — 9120 dependent global loads
// with only H*qn work-items to hide the latency (Qualcomm 80-NB295-11 §3.2.3, §6.5).
//
// Here one workgroup owns one (head, query) row: the whole probability row lives in LOCAL memory, so
// it is written and read at on-chip speed and never reaches DRAM. Local use is (M + W + hd) floats,
// passed in at dispatch so a small image does not pay for a large one; the host falls back to the
// three-kernel path when that would exceed the device's local memory.
__kernel void attn_fused_tile(__global const storage_t* q,    // [H, M, hd]
                              __global const storage_t* k,
                              __global const storage_t* v,
                              __global storage_t* ctx,        // [M, H*hd]
                              __local float* p,               // M   — the probability row
                              __local float* red,             // W   — reduction workspace
                              __local float* qs,              // hd  — this row's query, shared
                              const int H, const int M, const int hd,
                              const float scale, const int q0, const int qn)
{
    const int g = get_group_id(0);
    const int t = get_local_id(0);
    const int W = get_local_size(0);
    const int r = g % qn;
    const int h = g / qn;
    if (h >= H) return;
    const size_t qb = ((size_t)h * M + (q0 + r)) * hd;

    // Every work-item uses the SAME query row, so load it once into local instead of re-reading it
    // from global on every one of the M dot products.
    for (int d = t; d < hd; d += W) qs[d] = (float)LOAD(q, qb + d);
    barrier(CLK_LOCAL_MEM_FENCE);

    float mx = -INFINITY;
    for (int j = t; j < M; j += W) {
        const size_t kb = ((size_t)h * M + j) * hd;
        float acc = 0.0f;
        for (int d = 0; d < hd; ++d) acc += qs[d] * (float)LOAD(k, kb + d);
        acc *= scale;
        p[j] = acc;
        mx = fmax(mx, acc);
    }
    red[t] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = W >> 1; s > 0; s >>= 1) {
        if (t < s) red[t] = fmax(red[t], red[t + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    mx = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0.0f;
    for (int j = t; j < M; j += W) { const float e = native_exp(p[j] - mx); p[j] = e; sum += e; }
    red[t] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = W >> 1; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = 1.0f / red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Context. Work-item t owns output dim t, so at a fixed j the group reads v[h][j][0..hd) as one
    // contiguous run — coalesced, which the old kernel's per-item j loop was not.
    for (int d = t; d < hd; d += W) {
        float acc = 0.0f;
        for (int j = 0; j < M; ++j) acc += p[j] * (float)LOAD(v, ((size_t)h * M + j) * hd + d);
        STORE(ctx, (size_t)(q0 + r) * (H * hd) + h * hd + d, acc * inv);
    }
}

// Writes token-major [M, H*hd] at absolute row q0+r, so the tiles compose into one output.
__kernel void attn_context_tile(__global const storage_t* probs,  // [H, qn, M]
                                __global const storage_t* v,      // [H, M, hd]
                                __global storage_t* ctx,          // [M, H*hd]
                                const int H, const int M, const int hd,
                                const int q0, const int qn)
{
    const int idx = get_global_id(0);
    if (idx >= qn * H * hd) return;
    const int d = idx % hd;
    const int h = (idx / hd) % H;
    const int r = idx / (hd * H);
    const size_t pb = (size_t)(h * qn + r) * M;
    float acc = 0.0f;
    for (int j = 0; j < M; ++j)
        acc += (float)LOAD(probs, pb + j) * (float)LOAD(v, ((size_t)h * M + j) * hd + d);
    STORE(ctx, (size_t)(q0 + r) * (H * hd) + h * hd + d, acc);
}
