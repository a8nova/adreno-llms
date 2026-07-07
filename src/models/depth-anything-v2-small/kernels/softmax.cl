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


// ── softmax over last dim (row length M) of scores[H,M,M] ──
// one thread per (h,i) row.

__kernel void attn_softmax(__global storage_t* scores, const int H, const int M) {
    int row = get_global_id(0);
    int total_rows = H * M;
    if (row >= total_rows) return;
    int base = row * M;
    float mx = -1e30f;
    for (int j = 0; j < M; j++) { float v = (float)LOAD(scores, base + j); if (v > mx) mx = v; }
    float sum = 0.0f;
    for (int j = 0; j < M; j++) { float e = exp((float)LOAD(scores, base + j) - mx); STORE(scores, base + j, e); sum += e; }
    float inv = 1.0f / sum;
    for (int j = 0; j < M; j++) { float e = (float)LOAD(scores, base + j) * inv; STORE(scores, base + j, e); }
}

// ── Softmax, workgroup-per-row (optimization campaign 2026-07-05).
// The 1-WI-per-row version serialized 3×M global accesses per row with
// IEEE exp (271 ms/call). Here: 128 WIs cooperate per row with local-mem
// reductions and native_exp. Dispatch: gws = rows*128, lws = 128.

__kernel void attn_softmax_wg(__global storage_t* scores, const int M) {
    const int row = get_group_id(0);
    const int lid = get_local_id(0);
    const int base = row * M;
    __local float red[128];
    float mx = -1e30f;
    for (int j = lid; j < M; j += 128) mx = fmax(mx, (float)LOAD(scores, base + j));
    red[lid] = mx; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 64; s > 0; s >>= 1) {
        if (lid < s) red[lid] = fmax(red[lid], red[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    mx = red[0]; barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.0f;
    for (int j = lid; j < M; j += 128) {
        float e = native_exp((float)LOAD(scores, base + j) - mx);
        STORE(scores, base + j, e);
        sum += e;
    }
    red[lid] = sum; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 64; s > 0; s >>= 1) {
        if (lid < s) red[lid] += red[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = native_recip(red[0]);
    for (int j = lid; j < M; j += 128)
        STORE(scores, base + j, (float)LOAD(scores, base + j) * inv);
}
