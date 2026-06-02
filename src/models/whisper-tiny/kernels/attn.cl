// Scaled dot-product attention kernels (Whisper MHA).
// Reference: model_info/transformers_src/modeling_whisper.py:~110-175 eager_attention_forward
//            model_info/transformers_src/modeling_whisper.py:~175-270 WhisperAttention.forward

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

// The scores/probs buffer is storage_t (fp16 under USE_FP16) so it can be the
// output of a CLBlast fp16 GEMM (opt #1/#2) and the input of the probs@V GEMM.
// Masked entries use a large FINITE negative (-1e9 overflows fp16) so that
// exp(masked - max) underflows cleanly to 0.
#define ATTN_MASK_NEG (-30000.0f)

// scores[h, tq, tk] = sum_d q[h,tq,d] * k[h,tk,d]
// Kept for the CAUSAL (decoder self-attn) path; the non-causal encoder/cross
// path uses CLBlast GemmStridedBatched instead (see utils.cpp attn_scores_batched).
__kernel void attn_scores(
    __global const storage_t* q,
    __global const storage_t* k,
    __global storage_t* scores,
    int Tq,
    int Tk,
    int H,
    int D,
    int causal)
{
    int gid = (int)get_global_id(0);
    int total = H * Tq * Tk;
    if (gid >= total) return;

    int tmp = gid;
    int h = tmp / (Tq * Tk);
    tmp -= h * (Tq * Tk);
    int tq = tmp / Tk;
    int tk = tmp - tq * Tk;

    if (causal && tk > tq) {
        STORE(scores, gid, ATTN_MASK_NEG);
        return;
    }

    // Accumulate in float, but use fma for numerical parity.
    float acc = 0.0f;
    int q_base = (h * Tq + tq) * D;
    int k_base = (h * Tk + tk) * D;
    for (int d = 0; d < D; ++d) {
        float qv = (float)LOAD(q, q_base + d);
        float kv = (float)LOAD(k, k_base + d);
        acc = fma(qv, kv, acc);
    }
    STORE(scores, gid, acc);
}

// Row-wise softmax over the Tk axis (opt #3). One WORKGROUP per (h,tq) row;
// the Tk elements are split across the work-items, which cooperate via a
// __local reduction for the row max and the exp-sum — replacing the old
// one-thread-per-row serial triple-loop over Tk=1500. native_exp (Adreno
// fast-math transcendental, guideline G10) for the exponential.
//
// Launch contract: global_size = H*Tq*SOFTMAX_WG, local_size = SOFTMAX_WG.
// get_group_id(0) selects the row; gid_row == h*Tq+tq, so base = row*Tk.
#define SOFTMAX_WG 128

__kernel __attribute__((reqd_work_group_size(SOFTMAX_WG, 1, 1)))
void attn_softmax(
    __global storage_t* scores,
    int Tq,
    int Tk,
    int H)
{
    const int row = (int)get_group_id(0);
    const int total_rows = H * Tq;
    if (row >= total_rows) return;

    const int lid = (int)get_local_id(0);
    const int lsz = (int)get_local_size(0);
    const int base = row * Tk;

    __local float red[SOFTMAX_WG];

    // 1) row max — strided load + tree reduction in local memory.
    float m = -3.4e38f;
    for (int j = lid; j < Tk; j += lsz) {
        m = fmax(m, (float)LOAD(scores, base + j));
    }
    red[lid] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) red[lid] = fmax(red[lid], red[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float maxv = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2) exp(x - max) written back in place, plus the running sum.
    float sum = 0.0f;
    for (int j = lid; j < Tk; j += lsz) {
        float e = native_exp((float)LOAD(scores, base + j) - maxv);
        STORE(scores, base + j, e);
        sum += e;
    }
    red[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) red[lid] += red[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = 1.0f / red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3) normalize in place.
    for (int j = lid; j < Tk; j += lsz) {
        STORE(scores, base + j, (float)LOAD(scores, base + j) * inv);
    }
}

// out[h, tq, d] = sum_tk probs[h,tq,tk] * v[h,tk,d]
// Kept as a fallback; the live path uses CLBlast GemmStridedBatched (opt #2,
// utils.cpp attn_context_batched). probs is now storage_t (fp16 under USE_FP16).
__kernel void attn_weighted_sum(
    __global const storage_t* probs,
    __global const storage_t* v,
    __global storage_t* out,
    int Tq,
    int Tk,
    int H,
    int D)
{
    int gid = (int)get_global_id(0);
    int total = H * Tq * D;
    if (gid >= total) return;

    int tmp = gid;
    int h = tmp / (Tq * D);
    tmp -= h * (Tq * D);
    int tq = tmp / D;
    int d = tmp - tq * D;

    int p_base = (h * Tq + tq) * Tk;
    float acc = 0.0f;
    for (int tk = 0; tk < Tk; ++tk) {
        float p = (float)LOAD(probs, p_base + tk);
        acc += p * (float)LOAD(v, (h * Tk + tk) * D + d);
    }
    STORE(out, (h * Tq + tq) * D + d, acc);
}

// transpose [H,Tq,D] -> [Tq,H,D]
__kernel void attn_transpose_htd_to_thd(
    __global const storage_t* x_htd,
    __global storage_t* y_thd,
    int Tq,
    int H,
    int D)
{
    int gid = (int)get_global_id(0);
    int total = Tq * H * D;
    if (gid >= total) return;

    int tmp = gid;
    int tq = tmp / (H * D);
    tmp -= tq * (H * D);
    int h = tmp / D;
    int d = tmp - h * D;

    int src = (h * Tq + tq) * D + d;
    int dst = (tq * H + h) * D + d;
    STORE(y_thd, dst, (float)LOAD(x_htd, src));
}
