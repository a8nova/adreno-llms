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
