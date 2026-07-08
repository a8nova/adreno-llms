// DiT (DiffusionTransformer) kernels for Stable Audio Open Small.
// Reference: stable_audio_tools/models/transformer.py (Attention, FeedForward,
//   LayerNorm, RotaryEmbedding, apply_rotary_pos_emb) and
//   stable_audio_tools/models/blocks.py (FourierFeatures) and
//   stable_audio_tools/models/dit.py (DiffusionTransformer._forward).
//
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

// ── SiLU: x * sigmoid(x) ──  (nn.SiLU)
__kernel void silu_inplace(__global storage_t* x, const int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float v = LOAD(x, gid);
    STORE(x, gid, v / (1.0f + exp(-v)));
}

// ── add bias per-column: x[row, c] += bias[c] ──
__kernel void add_bias(__global storage_t* x, __global const storage_t* bias,
                       const int rows, const int cols) {
    int gid = get_global_id(0);
    if (gid >= rows * cols) return;
    int c = gid % cols;
    float v = LOAD(x, gid) + LOAD(bias, c);
    STORE(x, gid, v);
}

// ── LayerNorm over last dim, affine (gamma, beta) ──
// F.layer_norm(x, [D], weight=gamma, bias=beta, eps). One work-item per row.
__kernel void layernorm(__global const storage_t* x,
                        __global const storage_t* gamma,
                        __global const storage_t* beta,
                        __global storage_t* out,
                        const int rows, const int D, const float eps) {
    int row = get_global_id(0);
    if (row >= rows) return;
    int base = row * D;
    float mean = 0.0f;
    for (int i = 0; i < D; i++) mean += LOAD(x, base + i);
    mean /= (float)D;
    float var = 0.0f;
    for (int i = 0; i < D; i++) { float d = LOAD(x, base + i) - mean; var += d * d; }
    var /= (float)D;
    float inv = rsqrt(var + eps);
    for (int i = 0; i < D; i++) {
        float g = LOAD(gamma, i);
        float b = LOAD(beta, i);
        float v = (LOAD(x, base + i) - mean) * inv * g + b;
        STORE(out, base + i, v);
    }
}

// ── FourierFeatures.forward ──
// input t is [rows,1]; weight W is [half_out,1]. f = 2*pi * t * W^T -> [rows,half_out]
// out = cat[cos(f), sin(f)] -> [rows, 2*half_out].
// one work-item per (row, j) with j in [0, half_out).
__kernel void fourier_features(__global const storage_t* t,
                               __global const storage_t* W,
                               __global storage_t* out,
                               const int rows, const int half_out) {
    int gid = get_global_id(0);
    if (gid >= rows * half_out) return;
    int row = gid / half_out;
    int j = gid % half_out;
    float tv = LOAD(t, row);
    float wv = LOAD(W, j);
    float f = 2.0f * 3.14159265358979323846f * tv * wv;
    int out_cols = 2 * half_out;
    STORE(out, row * out_cols + j,            cos(f));
    STORE(out, row * out_cols + half_out + j, sin(f));
}

// ── qk LayerNorm applied per-head over head_dim, then reshape to [H, N, Dh] ──
// Input q_flat is row-major [N, H*Dh] (token-major after linear projection).
// This kernel: for each (head h, token n): take the Dh-slice, layernorm(affine),
// write into head-major q_hnd[h, n, d] = q_hnd[h*N*Dh + n*Dh + d].
__kernel void qk_norm_to_heads(__global const storage_t* q_flat,   // [N, src_stride]
                               __global const storage_t* gamma,    // [Dh]
                               __global const storage_t* beta,     // [Dh]
                               __global storage_t* q_hnd,          // [H, N, Dh]
                               const int N, const int H, const int Dh,
                               const float eps,
                               const int src_stride, const int src_off) {
    int gid = get_global_id(0);   // one per (h, n)
    if (gid >= H * N) return;
    int h = gid / N;
    int n = gid % N;
    int in_base = n * src_stride + src_off + h * Dh;
    float mean = 0.0f;
    for (int d = 0; d < Dh; d++) mean += LOAD(q_flat, in_base + d);
    mean /= (float)Dh;
    float var = 0.0f;
    for (int d = 0; d < Dh; d++) { float dd = LOAD(q_flat, in_base + d) - mean; var += dd * dd; }
    var /= (float)Dh;
    float inv = rsqrt(var + eps);
    int out_base = (h * N + n) * Dh;
    for (int d = 0; d < Dh; d++) {
        float g = LOAD(gamma, d);
        float b = LOAD(beta, d);
        STORE(q_hnd, out_base + d, (LOAD(q_flat, in_base + d) - mean) * inv * g + b);
    }
}

// ── reshape [N, H*Dh] -> head-major [H, N, Dh] WITHOUT norm (for v) ──
__kernel void split_heads(__global const storage_t* x_flat, // [N, src_stride]
                          __global storage_t* x_hnd,        // [H, N, Dh]
                          const int N, const int H, const int Dh,
                          const int src_stride, const int src_off) {
    int gid = get_global_id(0);
    if (gid >= H * N * Dh) return;
    int d = gid % Dh;
    int n = (gid / Dh) % N;
    int h = gid / (Dh * N);
    int in_idx = n * src_stride + src_off + h * Dh + d;
    STORE(x_hnd, (h * N + n) * Dh + d, LOAD(x_flat, in_idx));
}

// ── merge head-major [H, N, Dh] -> token-major [N, H*Dh] ──
__kernel void merge_heads(__global const storage_t* x_hnd, // [H, N, Dh]
                          __global storage_t* x_flat,      // [N, H*Dh]
                          const int N, const int H, const int Dh) {
    int gid = get_global_id(0);
    if (gid >= H * N * Dh) return;
    int d = gid % Dh;
    int n = (gid / Dh) % N;
    int h = gid / (Dh * N);
    STORE(x_flat, n * (H * Dh) + h * Dh + d, LOAD(x_hnd, (h * N + n) * Dh + d));
}

// ── partial RoPE applied to head-major q/k [H, N, Dh] over first rot_dim dims ──
// freqs precomputed as [N, rot_dim] where freqs[n] = cat(pos*inv_freq, pos*inv_freq)
// (rot_dim = 2*len(inv_freq)). apply_rotary_pos_emb:
//   t_rot[..,i]        = t[..,i]*cos(f[i])       + (-t[..,i+half])*sin(f[i])   for i<half
//   t_rot[..,i+half]   = t[..,i+half]*cos(f[..]) + ( t[..,i]      )*sin(f[..]) for i in first half
// where rotate_half([a,b]) = [-b, a] with a=first half, b=second half of rot_dim.
// dims >= rot_dim are passthrough. one work-item per (h, n).
__kernel void rope_apply(__global storage_t* x_hnd,        // [H,N,Dh] in/out
                         __global const storage_t* freqs,  // [N, rot_dim]
                         const int H, const int N, const int Dh,
                         const int rot_dim) {
    int gid = get_global_id(0);
    if (gid >= H * N) return;
    int h = gid / N;
    int n = gid % N;
    int base = (h * N + n) * Dh;
    int half_dim = rot_dim / 2;
    // buffer first-half rotated values before overwrite
    for (int i = 0; i < half_dim; i++) {
        float a = LOAD(x_hnd, base + i);          // first half
        float b = LOAD(x_hnd, base + i + half_dim);   // second half
        float cf = cos(LOAD(freqs, n * rot_dim + i));       // freqs first half == cos(pos*inv_freq[i])
        float sf = sin(LOAD(freqs, n * rot_dim + i));
        // rotate_half: [-b, a]; t*cos + rotate_half(t)*sin
        float new_a = a * cf + (-b) * sf;
        float new_b = b * cf + ( a) * sf;
        STORE(x_hnd, base + i,        new_a);
        STORE(x_hnd, base + i + half_dim, new_b);
    }
    // dims [rot_dim, Dh) passthrough (unchanged)
}

// ── attention scores: scores[h, i, j] = scale * dot(q[h,i,:], k[h,j,:]) ──
// q_hnd [H, Nq, Dh], k_hnd [H, Nk, Dh]. scale = 1/sqrt(Dh) (PyTorch SDPA default).
__kernel void attn_scores(__global const storage_t* q, // [H,Nq,Dh]
                          __global const storage_t* k, // [H,Nk,Dh]
                          __global storage_t* scores,  // [H,Nq,Nk]
                          const int H, const int Nq, const int Nk, const int Dh,
                          const float scale) {
    int gid = get_global_id(0);
    if (gid >= H * Nq * Nk) return;
    int j = gid % Nk;
    int i = (gid / Nk) % Nq;
    int h = gid / (Nk * Nq);
    int qb = (h * Nq + i) * Dh;
    int kb = (h * Nk + j) * Dh;
    float acc = 0.0f;
    for (int d = 0; d < Dh; d++) acc += LOAD(q, qb + d) * LOAD(k, kb + d);
    STORE(scores, gid, acc * scale);
}

// ── softmax over last dim (Nk) of scores [H, Nq, Nk], row = (h,i) ──
__kernel void softmax_rows(__global storage_t* scores,
                           const int rows, const int Nk) {
    int row = get_global_id(0);
    if (row >= rows) return;
    int base = row * Nk;
    float mx = -1e30f;
    for (int j = 0; j < Nk; j++) { float v = LOAD(scores, base + j); if (v > mx) mx = v; }
    float sum = 0.0f;
    for (int j = 0; j < Nk; j++) { float e = exp(LOAD(scores, base + j) - mx); STORE(scores, base + j, e); sum += e; }
    float inv = 1.0f / sum;
    for (int j = 0; j < Nk; j++) STORE(scores, base + j, LOAD(scores, base + j) * inv);
}

// ── attention output: out[h,i,:] = sum_j scores[h,i,j] * v[h,j,:] ──
__kernel void attn_av(__global const storage_t* scores, // [H,Nq,Nk]
                      __global const storage_t* v,       // [H,Nk,Dh]
                      __global storage_t* out,           // [H,Nq,Dh]
                      const int H, const int Nq, const int Nk, const int Dh) {
    int gid = get_global_id(0);
    if (gid >= H * Nq * Dh) return;
    int d = gid % Dh;
    int i = (gid / Dh) % Nq;
    int h = gid / (Dh * Nq);
    float acc = 0.0f;
    int sb = (h * Nq + i) * Nk;
    for (int j = 0; j < Nk; j++) acc += LOAD(scores, sb + j) * LOAD(v, (h * Nk + j) * Dh + d);
    STORE(out, (h * Nq + i) * Dh + d, acc);
}

// ── SwiGLU: given proj [N, 2*inner], out[N, inner] = value * silu(gate) ──
// GLU.forward: x,gate = proj.chunk(2,-1); return x * act(gate). act = SiLU.
// value = first half, gate = second half.
__kernel void swiglu(__global const storage_t* proj, // [N, 2*inner]
                     __global storage_t* out,         // [N, inner]
                     const int N, const int inner) {
    int gid = get_global_id(0);
    if (gid >= N * inner) return;
    int n = gid / inner;
    int c = gid % inner;
    int base = n * (2 * inner);
    float value = LOAD(proj, base + c);
    float gate  = LOAD(proj, base + inner + c);
    float sg = gate / (1.0f + exp(-gate));
    STORE(out, gid, value * sg);
}

// ── prepend one row (global token) in front of x: out[0,:]=g; out[1..N]=x ──
__kernel void prepend_row(__global const storage_t* g,   // [D]
                          __global const storage_t* x,    // [N, D]
                          __global storage_t* out,        // [N+1, D]
                          const int N, const int D) {
    int gid = get_global_id(0);
    if (gid >= (N + 1) * D) return;
    int row = gid / D;
    int c = gid % D;
    if (row == 0) STORE(out, gid, LOAD(g, c));
    else STORE(out, gid, LOAD(x, (row - 1) * D + c));
}

// ── drop first `drop` rows: out[i,:] = x[i+drop,:] ──
__kernel void drop_rows(__global const storage_t* x,  // [N, D]
                        __global storage_t* out,       // [N-drop, D]
                        const int out_rows, const int D, const int drop) {
    int gid = get_global_id(0);
    if (gid >= out_rows * D) return;
    int row = gid / D;
    int c = gid % D;
    STORE(out, gid, LOAD(x, (row + drop) * D + c));
}

// ── transpose [C, T] <-> [T, C] (rearrange b c t <-> b t c for batch=1) ──
__kernel void transpose_ct(__global const storage_t* in, // [C, T]
                           __global storage_t* out,       // [T, C]
                           const int C, const int T) {
    int gid = get_global_id(0);
    if (gid >= C * T) return;
    int c = gid / T;
    int t = gid % T;
    STORE(out, t * C + c, LOAD(in, c * T + t));
}

// ═════════════════════ Optimization-campaign kernels (2026-07-08) ═══════════
// OPT-2/OPT-3 live in this same program — dit.cl is a lean elementwise/reshape
// program, so adding these does not trip the Adreno program-wide register-
// footprint trap (depth-anything BENCHMARK.md: quarantine register-heavy
// kernels; these use ≤16 registers/lane and tiny local arrays).

#ifdef USE_FP16
  #define LOAD8(p, i) convert_float8(vload_half8(0, (p) + (i)))
#else
  #define LOAD8(p, i) vload8(0, (p) + (i))
#endif

inline float sum8(float8 v) {
    float4 a = v.lo + v.hi;
    float2 b = a.lo + a.hi;
    return b.x + b.y;
}

// ── OPT-2: fused attention — one 64-lane workgroup per (h, i) query row. ──
// Replaces attn_scores → softmax_rows → attn_av → merge_heads (4 kernels, a
// [H,Nq,Nk] fp16 global scores tensor, and a serial per-row softmax) with a
// single dispatch: scores live in __local, softmax is a workgroup tree
// reduction with native_exp, and the output is written TOKEN-MAJOR [Nq, H*Dh]
// so merge_heads disappears. q/k dots are 128-bit vectorized (vload_half8,
// guide §6.3). Requires Dh % 8 == 0 (Dh=128 here). NNOPT_ATTN_FUSED=0 reverts.
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void attn_fused(__global const storage_t* q,   // [H,Nq,Dh] head-major
                __global const storage_t* k,   // [H,Nk,Dh]
                __global const storage_t* v,   // [H,Nk,Dh]
                __global storage_t* out,       // [Nq, H*Dh] TOKEN-major
                const int H, const int Nq, const int Nk, const int Dh,
                const float scale,
                __local float* scores) {       // [Nk]
    const int lane = get_local_id(0);
    const int wg   = get_group_id(0);          // h*Nq + i
    const int i = wg % Nq;
    const int h = wg / Nq;
    const int qb  = (h * Nq + i) * Dh;
    const int kb0 = h * Nk * Dh;
    __local float red[64];

    // 1) scores[j] = scale * q·k_j  (lanes stride over j, 8-wide dot over Dh)
    for (int j = lane; j < Nk; j += 64) {
        const int kb = kb0 + j * Dh;
        float8 acc = (float8)(0.0f);
        for (int d = 0; d < Dh; d += 8)
            acc = mad(LOAD8(q, qb + d), LOAD8(k, kb + d), acc);
        scores[j] = sum8(acc) * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2) row max (tree reduction)
    float m = -1e30f;
    for (int j = lane; j < Nk; j += 64) m = fmax(m, scores[j]);
    red[lane] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 32; s > 0; s >>= 1) {
        if (lane < s) red[lane] = fmax(red[lane], red[lane + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float rowmax = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3) exp + sum (tree reduction)
    float ssum = 0.0f;
    for (int j = lane; j < Nk; j += 64) {
        float e = native_exp(scores[j] - rowmax);
        scores[j] = e;
        ssum += e;
    }
    red[lane] = ssum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 32; s > 0; s >>= 1) {
        if (lane < s) red[lane] += red[lane + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = 1.0f / red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 4) out[i, h*Dh + d] = inv * Σ_j scores[j]·v[h,j,d]
    //    Lanes stride over d → consecutive lanes hit consecutive v addresses
    //    (hardware-coalesced); Dh=128 → exactly 2 iterations per lane.
    for (int d = lane; d < Dh; d += 64) {
        float acc = 0.0f;
        for (int j = 0; j < Nk; j++)
            acc = mad(scores[j], (float)LOAD(v, kb0 + j * Dh + d), acc);
        STORE(out, (size_t)i * (H * Dh) + h * Dh + d, acc * inv);
    }
}

// ── OPT-3: workgroup LayerNorm — one 64-lane WG per row, 8-wide loads. ──
// The scalar `layernorm` above runs ONE work-item per row (257 rows → 257
// work items on a 1-CU GPU, each walking D=1024 serially three times).
// Moonshine's identical rewrite (r5) + the guide's workgroup-reduction
// pattern. Requires D % 8 == 0. NNOPT_LN_WG=0 reverts.
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void layernorm_wg(__global const storage_t* x,
                  __global const storage_t* gamma,
                  __global const storage_t* beta,
                  __global storage_t* out,
                  const int rows, const int D, const float eps) {
    const int row  = get_group_id(0);
    const int lane = get_local_id(0);
    const int base = row * D;
    __local float red[64];

    // mean
    float s = 0.0f;
    for (int d = lane * 8; d < D; d += 64 * 8) s += sum8(LOAD8(x, base + d));
    red[lane] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int t = 32; t > 0; t >>= 1) {
        if (lane < t) red[lane] += red[lane + t];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float mean = red[0] / (float)D;
    barrier(CLK_LOCAL_MEM_FENCE);

    // variance
    float vs = 0.0f;
    for (int d = lane * 8; d < D; d += 64 * 8) {
        float8 v8 = LOAD8(x, base + d) - (float8)(mean);
        vs += sum8(v8 * v8);
    }
    red[lane] = vs;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int t = 32; t > 0; t >>= 1) {
        if (lane < t) red[lane] += red[lane + t];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = rsqrt(red[0] / (float)D + eps);

    // normalize + affine
    for (int d = lane; d < D; d += 64) {
        float g = LOAD(gamma, d);
        float b = LOAD(beta, d);
        float val = ((float)LOAD(x, base + d) - mean) * inv * g + b;
        STORE(out, base + d, val);
    }
}

// ── B2: workgroup qk-norm — one 64-lane WG per (h, n) row. ──
// The scalar qk_norm_to_heads above runs ONE work item per row (H*N items,
// each walking Dh=128 serially three times → 6.4 ms/call, 512 calls/clip).
// Same WG-reduction shape as layernorm_wg. NNOPT_QK_WG=0 reverts.
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void qk_norm_to_heads_wg(__global const storage_t* q_flat,   // [N, src_stride]
                         __global const storage_t* gamma,    // [Dh]
                         __global const storage_t* beta,     // [Dh]
                         __global storage_t* q_hnd,          // [H, N, Dh]
                         const int N, const int H, const int Dh,
                         const float eps,
                         const int src_stride, const int src_off) {
    const int lane = get_local_id(0);
    const int wg   = get_group_id(0);        // h*N + n
    const int h = wg / N;
    const int n = wg % N;
    const int in_base = n * src_stride + src_off + h * Dh;
    __local float red[64];

    float s = 0.0f;
    for (int d = lane; d < Dh; d += 64) s += (float)LOAD(q_flat, in_base + d);
    red[lane] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int t = 32; t > 0; t >>= 1) {
        if (lane < t) red[lane] += red[lane + t];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float mean = red[0] / (float)Dh;
    barrier(CLK_LOCAL_MEM_FENCE);

    float vs = 0.0f;
    for (int d = lane; d < Dh; d += 64) {
        const float dd = (float)LOAD(q_flat, in_base + d) - mean;
        vs += dd * dd;
    }
    red[lane] = vs;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int t = 32; t > 0; t >>= 1) {
        if (lane < t) red[lane] += red[lane + t];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = rsqrt(red[0] / (float)Dh + eps);

    const int out_base = (h * N + n) * Dh;
    for (int d = lane; d < Dh; d += 64) {
        const float g = LOAD(gamma, d);
        const float b = LOAD(beta, d);
        STORE(q_hnd, out_base + d, ((float)LOAD(q_flat, in_base + d) - mean) * inv * g + b);
    }
}

// ── B4: fused pingpong update — x = (1-sn)*(x - sc*v) + sn*noise, in-place. ──
// One dispatch replaces: blocking v download + two host scalar loops + a
// noise-file read + blocking x re-upload, per denoise step.
__kernel void pingpong_step(__global storage_t* x,
                            __global const storage_t* v,
                            __global const storage_t* noise,
                            const float sc, const float sn, const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    const float d = (float)LOAD(x, i) - sc * (float)LOAD(v, i);
    STORE(x, i, (1.0f - sn) * d + sn * (float)LOAD(noise, i));
}

// ── B6: multi-row fused attention — 8 query rows per workgroup. ──
// attn_fused (above) re-reads the full K and V panels for EVERY query row:
// 130 KB × H×Nq WGs ≈ 267 MB per attention call → 68 GB per clip, squarely
// memory-bound (measured 6.3 s). Amortizing K/V across ATTN_ROWS=8 rows cuts
// that traffic 8×. q rows are staged in __local as float; K is loaded 8-wide
// once per (lane, j) and dotted against all 8 rows; V is loaded once per
// (lane-d, j) and applied to all 8 rows. NNOPT_ATTN_ROWS=1 falls back to
// attn_fused. Local: scores 8×Nk×4B (Nk=257 → 8.2 KB) + q 4 KB + red 256 B.
#define ATTN_ROWS 4
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void attn_fused_r8(__global const storage_t* q,   // [H,Nq,Dh] head-major
                   __global const storage_t* k,   // [H,Nk,Dh]
                   __global const storage_t* v,   // [H,Nk,Dh]
                   __global storage_t* out,       // [Nq, H*Dh] TOKEN-major
                   const int H, const int Nq, const int Nk, const int Dh,
                   const float scale,
                   __local float* scores) {       // [ATTN_ROWS * Nk]
    const int lane = get_local_id(0);
    const int wg   = get_group_id(0);              // h*ceil(Nq/R) + block
    const int nblk = (Nq + ATTN_ROWS - 1) / ATTN_ROWS;
    const int h  = wg / nblk;
    const int i0 = (wg % nblk) * ATTN_ROWS;        // first query row
    const int nr = min(ATTN_ROWS, Nq - i0);        // rows in this block
    const int kb0 = h * Nk * Dh;

    __local float8 qs8[ATTN_ROWS * 16];            // Dh/8 <= 16, vector lanes
    __local float red[64];

    // stage q rows (cooperative, 8-wide vector stores)
    for (int t = lane; t < nr * (Dh / 8); t += 64) {
        const int r = t / (Dh / 8);
        const int d8 = t % (Dh / 8);
        qs8[r * 16 + d8] = LOAD8(q, (size_t)(h * Nq + i0 + r) * Dh + d8 * 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 1) scores[r][j] — K loaded once per (lane, j), dotted vs all rows.
    //    q comes from __local as whole float8s (one vector read per mad-8).
    for (int j = lane; j < Nk; j += 64) {
        const int kb = kb0 + j * Dh;
        float acc[ATTN_ROWS];
        for (int r = 0; r < ATTN_ROWS; r++) acc[r] = 0.0f;
        for (int d = 0; d < Dh; d += 8) {
            const float8 kv = LOAD8(k, kb + d);
            const int q8 = d / 8;
            for (int r = 0; r < nr; r++)
                acc[r] += sum8(qs8[r * 16 + q8] * kv);
        }
        for (int r = 0; r < nr; r++) scores[r * Nk + j] = acc[r] * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2) per-row softmax (max + exp-sum tree reductions)
    float inv[ATTN_ROWS];
    for (int r = 0; r < nr; r++) {
        __local float* srow = scores + r * Nk;
        float m = -1e30f;
        for (int j = lane; j < Nk; j += 64) m = fmax(m, srow[j]);
        red[lane] = m;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = 32; s > 0; s >>= 1) {
            if (lane < s) red[lane] = fmax(red[lane], red[lane + s]);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const float rowmax = red[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        float ssum = 0.0f;
        for (int j = lane; j < Nk; j += 64) {
            const float e = native_exp(srow[j] - rowmax);
            srow[j] = e;
            ssum += e;
        }
        red[lane] = ssum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = 32; s > 0; s >>= 1) {
            if (lane < s) red[lane] += red[lane + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        inv[r] = 1.0f / red[0];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 3) AV — V loaded once per (lane-d, j), applied to all rows
    for (int d = lane; d < Dh; d += 64) {
        float acc[ATTN_ROWS];
        for (int r = 0; r < ATTN_ROWS; r++) acc[r] = 0.0f;
        for (int j = 0; j < Nk; j++) {
            const float vv = (float)LOAD(v, kb0 + j * Dh + d);
            for (int r = 0; r < nr; r++)
                acc[r] = mad(scores[r * Nk + j], vv, acc[r]);
        }
        for (int r = 0; r < nr; r++)
            STORE(out, (size_t)(i0 + r) * (H * Dh) + h * Dh + d, acc[r] * inv[r]);
    }
}

