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
