// Moonshine attention kernel (OPT-2). Owned by src/ops/MoonshineAttention.cpp.
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

// ── attention scores + softmax + weighted value ────────────────────────
// Q [H, Tq, D], K [H, Tk, D], V [H, Tk, D] -> out [H, Tq, D]
// scores[h,i,j] = scale * sum_d Q[h,i,d]*K[h,j,d]  (+ causal mask if causal)
// softmax over j, then out[h,i,d] = sum_j p[h,i,j]*V[h,j,d].
// One work item per (h, i). Loops over j (Tk small for decoder self-attn;
// Tk=207 for cross-attn — acceptable). Uses fp32 accumulators.
__kernel void attention(
    __global const storage_t* Q,   // [H, Tq, D]
    __global const storage_t* K,   // [H, kv_stride, D]
    __global const storage_t* V,   // [H, kv_stride, D]
    __global storage_t* out,       // [H, Tq, D]
    const int H,
    const int Tq,
    const int Tk,
    const int D,
    const float scale,
    const int causal,          // 1 => mask j>i (aligned to end for start_pos)
    const int q_pos_offset,    // absolute pos of query row 0 (for causal)
    const int kv_stride) {     // row capacity of K/V per head
    // OPT-2 (2026-07-02): one WORKGROUP per (h, i) query row; scores computed
    // ONCE into local memory (the old kernel used one work-item per row and
    // recomputed every QK dot 3x — 96.6% of all GPU time). local size = LWS.
    #define ATTN_LWS 64
    #define ATTN_MAX_TK 1024  // >= enc_T (16000/384 rows/s => ~25s audio) and KV_CAP=194
    __local float scores[ATTN_MAX_TK];
    __local float red[ATTN_LWS];
    __local float q_row[64];  // D<=64 (D=36 here); fp32 copy of the query row

    const int grp = get_group_id(0);
    const int lid = get_local_id(0);
    const int h = grp / Tq;
    const int i = grp - h * Tq;
    const int q_base = (h * Tq + i) * D;
    const int abs_i = i + q_pos_offset;
    const int jmax = (causal && (abs_i + 1) < Tk) ? (abs_i + 1) : Tk;  // rows attended

    // stage the query row once (fp32) — every lane reuses it Tk times.
    for (int d = lid; d < D; d += ATTN_LWS) q_row[d] = (float)LOAD(Q, q_base + d);
    barrier(CLK_LOCAL_MEM_FENCE);

    // pass 1: each lane computes score(j) ONCE for its strided j's.
    float lmax = -1e30f;
    for (int j = lid; j < jmax; j += ATTN_LWS) {
        const int k_base = (h * kv_stride + j) * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += q_row[d] * (float)LOAD(K, k_base + d);
        dot *= scale;
        scores[j] = dot;
        if (dot > lmax) lmax = dot;
    }
    // workgroup max-reduce
    red[lid] = lmax; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s2 = ATTN_LWS >> 1; s2 > 0; s2 >>= 1) {
        if (lid < s2 && red[lid + s2] > red[lid]) red[lid] = red[lid + s2];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float gmax = red[0]; barrier(CLK_LOCAL_MEM_FENCE);

    // pass 2: exp + sum-reduce (scores[] becomes probabilities pre-normalize)
    float lsum = 0.0f;
    for (int j = lid; j < jmax; j += ATTN_LWS) {
        const float e = exp(scores[j] - gmax);
        scores[j] = e;
        lsum += e;
    }
    red[lid] = lsum; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s2 = ATTN_LWS >> 1; s2 > 0; s2 >>= 1) {
        if (lid < s2) red[lid] += red[lid + s2];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv_sum = 1.0f / red[0]; barrier(CLK_LOCAL_MEM_FENCE);

    // pass 3: lanes own output dims d (strided); loop j over staged scores.
    for (int d = lid; d < D; d += ATTN_LWS) {
        float acc = 0.0f;
        for (int j = 0; j < jmax; ++j) {
            const int k_base = (h * kv_stride + j) * D;
            acc += scores[j] * (float)LOAD(V, k_base + d);
        }
        STORE(out, q_base + d, acc * inv_sum);
    }
    #undef ATTN_LWS
    #undef ATTN_MAX_TK
}
