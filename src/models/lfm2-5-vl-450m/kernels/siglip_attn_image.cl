// Image2d-backed SigLIP bidirectional flash attention.
// K and V are wrapped as image2d_t via cl_khr_image2d_from_buffer:
//   layout: CL_RGBA / CL_HALF_FLOAT
//   width  = head_dim / 4 = 16 pixels
//   height = kv_heads * k_rows  (max 16384 on Adreno 620 covers k_rows=1024)
// Texture L1 cache amplifies the K/V reuse across query tiles.
//
// IMPORTANT: separate cl_program from lfm2_ops.cl. Adreno compiler does global
// register allocation across all kernels in a program; mixing image2d kernels
// with buffer-path kernels causes register spills (ARTICLE.md: 13x regression).
//
// Same algorithm as siglip_flash_attn_prefill in lfm2_ops.cl, only the K/V
// read path changes.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t kSiglipKVSampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

#define SFA_BQ 4
#define SFA_BK 64
#define SFA_WG 64
#define SFA_HD 64
#define SFA_WI_PER_ROW (SFA_WG / SFA_BQ)

__kernel __attribute__((reqd_work_group_size(SFA_WG, 1, 1)))
void siglip_flash_attn_prefill_img(
    __global const half* Q,
    __read_only image2d_t K_img,
    __read_only image2d_t V_img,
    __global const int*  pad_mask,
    __global half*       out_heads,
    const int q_rows,
    const int k_rows,
    const int q_heads,
    const int kv_heads,
    const int head_dim,
    const float scale) {
    const int num_q_tiles = (q_rows + SFA_BQ - 1) / SFA_BQ;
    const int gid = get_group_id(0);
    const int qh = gid / num_q_tiles;
    const int qt = gid % num_q_tiles;
    const int lid = get_local_id(0);
    if (qh >= q_heads) return;

    const int kvh = qh / (q_heads / kv_heads);
    const int q_start = qt * SFA_BQ;
    const int bq = min(SFA_BQ, q_rows - q_start);

    const int my_row = lid / SFA_WI_PER_ROW;
    const int lane   = lid % SFA_WI_PER_ROW;

    __local half  Q_loc[SFA_BQ * SFA_HD];
    __local half  KV_loc[SFA_BK * SFA_HD];
    __local float S_loc[SFA_BQ * SFA_BK];
    __local float O_loc[SFA_BQ * SFA_HD];
    __local float m_loc[SFA_BQ];
    __local float l_loc[SFA_BQ];
    __local float red[SFA_WG];

    for (int e = lid; e < SFA_BQ * SFA_HD; e += SFA_WG) O_loc[e] = 0.0f;
    if (lid < SFA_BQ) { m_loc[lid] = -INFINITY; l_loc[lid] = 0.0f; }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Load Q tile (still buffer-backed, Q is small per tile = 4 * 64 half = 512 B)
    {
        __global const half* Qbase = Q + (size_t)qh * q_rows * SFA_HD + (size_t)q_start * SFA_HD;
        int qelems = bq * SFA_HD;
        __global const half4* Qsrc = (__global const half4*)Qbase;
        __local  half4*       Qdst = (__local  half4*)Q_loc;
        int q4 = qelems >> 2;
        for (int i = lid; i < q4; i += SFA_WG) Qdst[i] = Qsrc[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int kv_base_row = kvh * k_rows;

    for (int kk = 0; kk < k_rows; kk += SFA_BK) {
        const int bk = min(SFA_BK, k_rows - kk);

        // Load K tile via image2d
        {
            __local half4* Kdst = (__local half4*)KV_loc;
            int k4 = (bk * SFA_HD) >> 2;
            for (int i = lid; i < k4; i += SFA_WG) {
                int j = i / (SFA_HD >> 2);
                int d = i & ((SFA_HD >> 2) - 1);
                Kdst[i] = read_imageh(K_img, kSiglipKVSampler, (int2)(d, kv_base_row + kk + j));
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // S = Q @ K^T (scaled, pad-masked)
        int stotal = bq * bk;
        for (int s = lid; s < stotal; s += SFA_WG) {
            int qi = s / bk;
            int kj = s - qi * bk;
            __local const half4* qv = (__local const half4*)(Q_loc + qi * SFA_HD);
            __local const half4* kvv = (__local const half4*)(KV_loc + kj * SFA_HD);
            float4 a4 = (float4)(0.0f);
            for (int d = 0; d < (SFA_HD >> 2); ++d)
                a4 += convert_float4(qv[d]) * convert_float4(kvv[d]);
            float dot = a4.s0 + a4.s1 + a4.s2 + a4.s3;
            int abs_k = kk + kj;
            int valid = pad_mask[abs_k];
            S_loc[qi * SFA_BK + kj] = (valid == 0) ? -INFINITY : dot * scale;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Load V tile via image2d (reuse KV_loc)
        {
            __local half4* Vdst = (__local half4*)KV_loc;
            int v4 = (bk * SFA_HD) >> 2;
            for (int i = lid; i < v4; i += SFA_WG) {
                int j = i / (SFA_HD >> 2);
                int d = i & ((SFA_HD >> 2) - 1);
                Vdst[i] = read_imageh(V_img, kSiglipKVSampler, (int2)(d, kv_base_row + kk + j));
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Online softmax (16 WIs per Q row, 4 rows in parallel)
        if (my_row < bq) {
            const int qi = my_row;
            const int rbase = qi * SFA_WI_PER_ROW;
            float lmax = -INFINITY;
            for (int j = lane; j < bk; j += SFA_WI_PER_ROW)
                lmax = fmax(lmax, S_loc[qi * SFA_BK + j]);
            red[lid] = lmax;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 8) red[rbase + lane] = fmax(red[rbase + lane], red[rbase + lane + 8]);
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 4) red[rbase + lane] = fmax(red[rbase + lane], red[rbase + lane + 4]);
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 2) red[rbase + lane] = fmax(red[rbase + lane], red[rbase + lane + 2]);
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 1) red[rbase + lane] = fmax(red[rbase + lane], red[rbase + lane + 1]);
            barrier(CLK_LOCAL_MEM_FENCE);

            float new_m = red[rbase];
            float old_m = m_loc[qi];
            float comb_m = fmax(old_m, new_m);
            float alpha = (comb_m == -INFINITY) ? 1.0f : native_exp(old_m - comb_m);

            for (int d = lane; d < SFA_HD; d += SFA_WI_PER_ROW)
                O_loc[qi * SFA_HD + d] *= alpha;

            float pl = 0.0f;
            for (int j = lane; j < bk; j += SFA_WI_PER_ROW) {
                float sval = S_loc[qi * SFA_BK + j];
                float ev = (sval == -INFINITY || comb_m == -INFINITY) ? 0.0f : native_exp(sval - comb_m);
                S_loc[qi * SFA_BK + j] = ev;
                pl += ev;
            }
            red[lid] = pl;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 8) red[rbase + lane] += red[rbase + lane + 8];
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 4) red[rbase + lane] += red[rbase + lane + 4];
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 2) red[rbase + lane] += red[rbase + lane + 2];
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 1) red[rbase + lane] += red[rbase + lane + 1];
            barrier(CLK_LOCAL_MEM_FENCE);

            if (lane == 0) {
                m_loc[qi] = comb_m;
                l_loc[qi] = l_loc[qi] * alpha + red[rbase];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // O += P @ V
        for (int e = lid; e < bq * SFA_HD; e += SFA_WG) {
            int qi = e / SFA_HD;
            int d  = e - qi * SFA_HD;
            float o_acc = 0.0f;
            for (int j = 0; j < bk; ++j)
                o_acc += S_loc[qi * SFA_BK + j] * convert_float(KV_loc[j * SFA_HD + d]);
            O_loc[e] += o_acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int e = lid; e < bq * SFA_HD; e += SFA_WG) {
        int qi = e / SFA_HD;
        int d  = e - qi * SFA_HD;
        float inv_l = 1.0f / fmax(l_loc[qi], 1.0e-20f);
        int gidx = qh * q_rows * SFA_HD + (q_start + qi) * SFA_HD + d;
        vstore_half(O_loc[e] * inv_l, gidx, out_heads);
    }
}
