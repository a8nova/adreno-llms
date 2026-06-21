// Attention + position embeddings: standard MHA, conformer rel-pos MHA,
// rel-pos sinusoid table, decoder token+position embedding.

// Online-softmax MHA. q[Tq,Dm], k/v[Tk,Dm], out[Tq,Dm]. One work-item per (h,i).
__kernel void attention_context(__global const storage_t* q,
                                __global const storage_t* k,
                                __global const storage_t* v,
                                __global storage_t* out,
                                const int Tq, const int Tk, const int H,
                                const int Dk, const int causal) {
    int gid = get_global_id(0);
    if (gid >= H * Tq) return;
    int h = gid / Tq, i = gid - h * Tq;
    int Dm = H * Dk;
    int lim = causal ? (i + 1) : Tk;
    float scale = 1.0f / sqrt((float)Dk);
    int qb = i * Dm + h * Dk;
    int nv = Dk >> 3;  // half8 chunks per head (6 for Dk=48)
    float m = -1e30f, l = 0.0f;
    float8 acc[8];
    for (int d = 0; d < nv; ++d) acc[d] = (float8)(0.0f);
    for (int j = 0; j < lim; ++j) {
        int kb = j * Dm + h * Dk;
        float s = 0.0f;
        for (int d = 0; d < nv; ++d) s += dot8(LOAD8(q, qb + d * 8), LOAD8(k, kb + d * 8));
        s *= scale;
        float nm = fmax(m, s);
        float corr = native_exp(m - nm), e = native_exp(s - nm);
        l = l * corr + e;
        for (int d = 0; d < nv; ++d) acc[d] = acc[d] * corr + e * LOAD8(v, kb + d * 8);
        m = nm;
    }
    int ob = i * Dm + h * Dk;
    for (int d = 0; d < nv; ++d) STORE8(out, ob + d * 8, acc[d] / l);
}

// Transformer-XL conformer attention. q/k/v[T,Dm], p[L,Dm] projected pos emb,
// bu/bv[Dm]. sc[jj] = ((q+bu).k_jj + (q+bv).p_{(T-1)-i+jj}) * scale.
__kernel void relpos_attention_context(__global const storage_t* q,
                                       __global const storage_t* k,
                                       __global const storage_t* v,
                                       __global const storage_t* p,
                                       __global const storage_t* bu,
                                       __global const storage_t* bv,
                                       __global storage_t* out,
                                       const int T, const int L, const int H, const int Dk) {
    int gid = get_global_id(0);
    if (gid >= H * T) return;
    int h = gid / T, i = gid - h * T;
    int Dm = H * Dk;
    float scale = 1.0f / sqrt((float)Dk);
    int qb = i * Dm + h * Dk, hb = h * Dk;
    int nv = Dk >> 3;
    float m = -1e30f, l = 0.0f;
    float8 acc[8];
    // Hoist q+pos_bias_u / q+pos_bias_v into registers (invariant across the T-key
    // loop). A low-register variant (load per key) was tried and REGRESSED encoder
    // 5.5→6.2 s — the per-key cache re-reads cost more than the register pressure saved.
    float8 qu[8], qv[8];
    for (int d = 0; d < nv; ++d) {
        float8 qd = LOAD8(q, qb + d * 8);
        qu[d] = qd + LOAD8(bu, hb + d * 8);
        qv[d] = qd + LOAD8(bv, hb + d * 8);
        acc[d] = (float8)(0.0f);
    }
    for (int jj = 0; jj < T; ++jj) {
        int kb = jj * Dm + h * Dk;
        int pidx = (T - 1) - i + jj;
        int pb = pidx * Dm + h * Dk;
        float ac = 0.0f, bd = 0.0f;
        for (int d = 0; d < nv; ++d) {
            ac += dot8(qu[d], LOAD8(k, kb + d * 8));
            bd += dot8(qv[d], LOAD8(p, pb + d * 8));
        }
        float s = (ac + bd) * scale;
        float nm = fmax(m, s);
        float corr = native_exp(m - nm), e = native_exp(s - nm);
        l = l * corr + e;
        for (int d = 0; d < nv; ++d) acc[d] = acc[d] * corr + e * LOAD8(v, kb + d * 8);
        m = nm;
    }
    int ob = i * Dm + h * Dk;
    for (int d = 0; d < nv; ++d) STORE8(out, ob + d * 8, acc[d] / l);
}

// Cooperative rel-pos attention: ATT_G threads per (head,query) split the T keys
// and flash-merge. The 1-WI-per-query version above is serial over ~299 keys with
// only H*T work-items; this puts H*T*ATT_G in flight and cuts each thread's key
// loop ATT_G×. Online-softmax partials (m,l,acc) merged by lane 0 in local memory.
#define ATT_G 8                       // key-groups (threads) per query
#define ATT_QPB 8                      // queries per work-group (ATT_G*ATT_QPB=64=wave)
#define ATT_NV 8                       // max half8 chunks per head (Dk/8)
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void relpos_attention_coop(__global const storage_t* q, __global const storage_t* k,
                           __global const storage_t* v, __global const storage_t* p,
                           __global const storage_t* bu, __global const storage_t* bv,
                           __global storage_t* out,
                           const int T, const int L, const int H, const int Dk) {
    __local float lm[64], ll[64], lacc[64 * (ATT_NV * 8)];
    int lid = get_local_id(0);
    int ql = lid >> 3;                 // which query in this block (0..7)
    int g = lid & 7;                   // key-group / thread within query (0..7)
    int qglob = get_group_id(0) * ATT_QPB + ql;   // flat (head,query) index; H*T % 8 == 0
    int h = qglob / T, i = qglob - h * T;
    int Dm = H * Dk, nv = Dk >> 3, hb = h * Dk, qb = i * Dm + hb;
    float scale = 1.0f / sqrt((float)Dk);
    float8 qu[ATT_NV], qv[ATT_NV], acc[ATT_NV];
    for (int d = 0; d < nv; ++d) {
        float8 qd = LOAD8(q, qb + d * 8);
        qu[d] = qd + LOAD8(bu, hb + d * 8);
        qv[d] = qd + LOAD8(bv, hb + d * 8);
        acc[d] = (float8)(0.0f);
    }
    float m = -1e30f, l = 0.0f;
    for (int jj = g; jj < T; jj += ATT_G) {
        int kb = jj * Dm + hb, pb = ((T - 1) - i + jj) * Dm + hb;
        float ac = 0.0f, bd = 0.0f;
        for (int d = 0; d < nv; ++d) { ac += dot8(qu[d], LOAD8(k, kb + d * 8)); bd += dot8(qv[d], LOAD8(p, pb + d * 8)); }
        float s = (ac + bd) * scale, nm = fmax(m, s);
        float corr = native_exp(m - nm), e = native_exp(s - nm);
        l = l * corr + e;
        for (int d = 0; d < nv; ++d) acc[d] = acc[d] * corr + e * LOAD8(v, kb + d * 8);
        m = nm;
    }
    lm[lid] = m; ll[lid] = l;
    for (int d = 0; d < nv; ++d) vstore8(acc[d], d, &lacc[lid * (ATT_NV * 8)]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (g == 0) {
        int base = ql * 8;
        float M = lm[base];
        for (int gg = 1; gg < ATT_G; ++gg) M = fmax(M, lm[base + gg]);
        float Lsum = 0.0f;
        float8 A[ATT_NV];
        for (int d = 0; d < nv; ++d) A[d] = (float8)(0.0f);
        for (int gg = 0; gg < ATT_G; ++gg) {
            float wgt = native_exp(lm[base + gg] - M);
            Lsum += ll[base + gg] * wgt;
            for (int d = 0; d < nv; ++d) A[d] += wgt * vload8(d, &lacc[(base + gg) * (ATT_NV * 8)]);
        }
        int ob = i * Dm + hb;
        for (int d = 0; d < nv; ++d) STORE8(out, ob + d * 8, A[d] / Lsum);
    }
}

// Conformer rel-pos sinusoid table [L, Dm], L = 2*ET-1, interleaved sin/cos.
__kernel void relpos_pos_emb(__global storage_t* pe, const int L, const int Dm, const int ET) {
    int gid = get_global_id(0);
    int hd = Dm / 2;
    if (gid >= L * hd) return;
    int row = gid / hd, i = gid - row * hd;
    float pos = (float)((ET - 1) - row);
    float div = exp((float)(2 * i) * -(log(10000.0f) / (float)Dm));
    STORE(pe, row * Dm + 2 * i,     sin(pos * div));
    STORE(pe, row * Dm + 2 * i + 1, cos(pos * div));
}

// Decoder embedding + sinusoidal positions (fairseq, block sin|cos, pos from start).
// out[t,i] = embed_scale*emb[tok_t,i] + sinusoid. One work-item per (t, i<Dm/2).
// pos_stride: 1 when rows are consecutive sequence positions (teacher-forced /
// single-token), 0 when rows are different beams at the SAME position `start`.
__kernel void embed_scale_pos(__global const int* token_ids,
                              __global const storage_t* emb,
                              __global storage_t* out,
                              const int T, const int Dm, const float embed_scale,
                              const int start, const int pos_stride) {
    int gid = get_global_id(0);
    int hd = Dm / 2;
    if (gid >= T * hd) return;
    int t = gid / hd, i = gid - t * hd;
    int tok = token_ids[t];
    int eb = tok * Dm, ob = t * Dm;
    float ec = log(10000.0f) / (float)(hd - 1);
    float pos = (float)(start + t * pos_stride);
    float f = pos * exp(-(float)i * ec);
    STORE(out, ob + i,      embed_scale * (float)LOAD(emb, eb + i)      + sin(f));
    STORE(out, ob + hd + i, embed_scale * (float)LOAD(emb, eb + hd + i) + cos(f));
}
