// pack.cl — layout transforms for Whisper attention
// Reference: model_info/transformers_src/modeling_whisper.py:241-359 WhisperAttention.forward
// This file implements a simple packing kernel to reinterpret a row-major
// [T, H*D] tensor (where hidden=H*D) into an explicit [H, T, D] contiguous
// layout expected by kernels/attn.cl.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)    vload_half((i),(p))
  #define STORE(p,i,v) vstore_half((v),(i),(p))
#else
  typedef float storage_t;
  #define LOAD(p,i)    ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
#endif

// out[h,t,d] = in[t, h*D + d]
__kernel void pack_th_to_htd(
    __global const storage_t* in_th,
    __global storage_t* out_htd,
    int T,
    int H,
    int D)
{
    int gid = (int)get_global_id(0);
    int total = H * T * D;
    if (gid >= total) return;

    int tmp = gid;
    int h = tmp / (T * D);
    tmp -= h * (T * D);
    int t = tmp / D;
    int d = tmp - t * D;

    int src = t * (H * D) + h * D + d;
    int dst = (h * T + t) * D + d;
    STORE(out_htd, dst, (float)LOAD(in_th, src));
}

// Pack the T new rows of in_th directly into a persistent KV cache laid out as
// [H, CAP, D] (per-head capacity CAP), at column offset col0 (opt #12). Lets the
// decoder self-attn append a token's K/V to its cache in one launch — no extra
// copy. out_cache[h, col0+t, d] = in_th[t, h*D + d].
__kernel void pack_th_to_htd_cache(
    __global const storage_t* in_th,
    __global storage_t* out_cache,
    int T,
    int H,
    int D,
    int CAP,
    int col0)
{
    int gid = (int)get_global_id(0);
    int total = H * T * D;
    if (gid >= total) return;

    int tmp = gid;
    int h = tmp / (T * D);
    tmp -= h * (T * D);
    int t = tmp / D;
    int d = tmp - t * D;

    int src = t * (H * D) + h * D + d;
    int dst = (h * CAP + (col0 + t)) * D + d;
    STORE(out_cache, dst, (float)LOAD(in_th, src));
}
