// LayerNorm (OPT-5 WG-per-row) + GroupNorm. Owned by src/ops/LayerNorm.cpp / GroupNorm.cpp.
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

// ── GroupNorm (num_groups=1 → normalize over ALL channels per position) ──
// Moonshine: GroupNorm(num_groups=1, num_channels=Cin). Input is [Cin, L]
// (channel-major, from conv output). For num_groups=1 the stats are computed
// over the whole [Cin, L] feature map for each batch (batch=1 here), i.e.
// over Cin*L elements. weight/bias are per-channel [Cin].
// One work item per (channel, pos) but stats are global — so we compute in
// two passes via a single kernel that recomputes mean/var (cheap for L~1249).
// Simpler: dispatch ONE workgroup, compute global mean/var into local, then
// normalize. We instead do a single-thread reduction per channel-independent?
// GroupNorm(1) shares stats across channels — must be global. Use a 2-stage:
// stage 1 computes sum & sumsq via kernel groupnorm_stats; stage 2 normalizes.
__kernel void groupnorm_stats(
    __global const storage_t* input,  // [C, L]
    __global float* partial_sum,       // [nwg]
    __global float* partial_sumsq,     // [nwg]
    const int total,
    __local float* lsum,
    __local float* lsumsq) {
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    int gid = get_global_id(0);
    int gsize = get_global_size(0);
    float s = 0.0f, ss = 0.0f;
    for (int i = gid; i < total; i += gsize) {
        float v = LOAD(input, i);
        s += v; ss += v * v;
    }
    lsum[lid] = s; lsumsq[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsize / 2; off > 0; off >>= 1) {
        if (lid < off) { lsum[lid] += lsum[lid + off]; lsumsq[lid] += lsumsq[lid + off]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) { partial_sum[get_group_id(0)] = lsum[0]; partial_sumsq[get_group_id(0)] = lsumsq[0]; }
}

__kernel void groupnorm_apply(
    __global const storage_t* input,   // [C, L]
    __global const storage_t* weight,  // [C]
    __global const storage_t* bias,    // [C]
    __global storage_t* output,        // [C, L]
    const float mean,
    const float inv_std,
    const int C,
    const int L) {
    int gid = get_global_id(0);
    const int total = C * L;
    if (gid >= total) return;
    int c = gid / L;
    float x = LOAD(input, gid);
    float norm = (x - mean) * inv_std;
    float y = norm * LOAD(weight, c) + LOAD(bias, c);
    STORE(output, gid, y);
}

// ── LayerNorm (bias=False for all Moonshine LayerNorms) ─────────────────
// input [rows, cols], normalize over last dim (cols). weight [cols], no bias.
// eps = 1e-5. One work item per row.
// One WORKGROUP per row (OPT-5, r5): lanes stride cols, two tree reductions
// (sum, then centered sum-of-squares — same two-pass formula as PyTorch).
// The old kernel ran 1 work-item per row: a serial 288-wide reduction that
// cost 300µs/call and 78ms/clip at decode where rows==1 (r3 profile).
// gws MUST be rows*lws with lws==local_size (host: LayerNorm.cpp).
__kernel void layernorm_nobias(
    __global const storage_t* input,   // [rows, cols]
    __global const storage_t* weight,  // [cols]
    __global storage_t* output,        // [rows, cols]
    const int rows,
    const int cols,
    const float eps,
    __local float* lred) {             // [local_size]
    const int row = get_group_id(0);
    if (row >= rows) return;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int base = row * cols;
    float s = 0.0f;
    for (int c = lid; c < cols; c += lsz) s += LOAD(input, base + c);
    lred[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) lred[lid] += lred[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float mean = lred[0] / (float)cols;
    barrier(CLK_LOCAL_MEM_FENCE);      // lred reused below
    float vs = 0.0f;
    for (int c = lid; c < cols; c += lsz) {
        float d = LOAD(input, base + c) - mean;
        vs += d * d;
    }
    lred[lid] = vs;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) lred[lid] += lred[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv_std = rsqrt(lred[0] / (float)cols + eps);
    for (int c = lid; c < cols; c += lsz) {
        float norm = (LOAD(input, base + c) - mean) * inv_std;
        STORE(output, base + c, norm * LOAD(weight, c));
    }
}
