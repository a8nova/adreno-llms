// ops.cl — core compute kernels for the pocket-tts hand port.
// fp16 storage, fp32 arithmetic. See .nnport/PORT_SPEC.md for the math.
// Dtype preamble driven by host-side -DUSE_FP16 (matches utils.cl convention).
#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)     vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
  #define LOAD8(p, k)    vload_half8(0, (p) + (k))   // 8 halves → float8
  #define LOAD4(p, k)    vload_half4(0, (p) + (k))   // 4 halves → float4
  #define STORE8(p, k, v) vstore_half8((v), 0, (p) + (k))
#else
  typedef float storage_t;
  #define LOAD(p, i)     ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
  #define LOAD8(p, k)    vload8(0, (p) + (k))        // 8 floats → float8
  #define LOAD4(p, k)    vload4(0, (p) + (k))        // 4 floats → float4
  #define STORE8(p, k, v) vstore8((v), 0, (p) + (k))
#endif

// ── Thin GEMM for nn.Linear at small M (recordable, mimi transformer Tq=16) ──
// out[M,N] = x[M,K] @ W[N,K]^T (+bias[N]). One thread per output COLUMN n: loads
// W[n,:] ONCE (vectorized) and reuses it across all M rows (M private accumulators).
// vs one-thread-per-element (which re-reads W M times), this cuts W traffic M×. x is
// small ([M,K]) so re-reading it per column stays in cache. Coalesced output writes.
#define THIN_MMAX 16
__kernel void gemm_thin_nk(__global const storage_t* x,    // [M,K] row-major
                           __global const storage_t* W,    // [N,K] row-major (nn.Linear)
                           __global const storage_t* bias, // [N] or dummy
                           __global storage_t* out,        // [M,N] row-major
                           const int M, const int N, const int K, const int has_bias) {
    int n = get_global_id(0);
    if (n >= N) return;
    const __global storage_t* wr = W + (size_t)n * K;
    float acc[THIN_MMAX];
    float b = has_bias ? (float)LOAD(bias, n) : 0.0f;
    for (int m = 0; m < M; ++m) acc[m] = b;
    int k = 0, K8 = K & ~7;
    for (; k < K8; k += 8) {
        float8 wv = LOAD8(wr, k);
        for (int m = 0; m < M; ++m) {
            float8 xv = LOAD8(x + (size_t)m * K, k);
            acc[m] += dot(wv.lo, xv.lo) + dot(wv.hi, xv.hi);
        }
    }
    for (; k < K; ++k) {
        float w = (float)LOAD(wr, k);
        for (int m = 0; m < M; ++m) acc[m] += w * (float)LOAD(x + (size_t)m * K, k);
    }
    for (int m = 0; m < M; ++m) STORE(out, (size_t)m * N + n, acc[m]);
}

// linear copy of n elements (recordable; used to pad/extract aligned GEMM buffers).
__kernel void copy_n(__global const storage_t* src, __global storage_t* dst, const int n) {
    int i = get_global_id(0);
    if (i < n) dst[i] = src[i];
}

// ── Column-blocked GEMM for nn.Linear (recordable, register-blocked) ─────────
// out[M,N] = x[M,K] @ W[N,K]^T (+bias). One thread per (m, 8-column block): loads
// x[m,k:k+8] ONCE and reuses across 8 output columns (8 register accumulators), each
// W row streamed once vectorized. W traffic = N*K (optimal); x stays in cache. 8 accs
// (not 16) avoids the register spill that sank gemm_thin_nk. 1D dispatch: gid = m*NB+nb.
#define CB_N 8
__kernel void gemm_nk_c8(__global const storage_t* x,    // [M,K]
                         __global const storage_t* W,    // [N,K] (nn.Linear)
                         __global const storage_t* bias, // [N] or dummy
                         __global storage_t* out,        // [M,N]
                         const int M, const int N, const int K, const int has_bias) {
    int NB = (N + CB_N - 1) / CB_N;
    int gid = get_global_id(0);
    if (gid >= M * NB) return;
    int m = gid / NB, nb = gid - m * NB, n0 = nb * CB_N;
    const __global storage_t* xr = x + (size_t)m * K;
    float acc[CB_N];
    #pragma unroll
    for (int j = 0; j < CB_N; ++j) acc[j] = 0.0f;
    int k = 0, K8 = K & ~7;
    for (; k < K8; k += 8) {
        float8 xv = LOAD8(xr, k);
        #pragma unroll
        for (int j = 0; j < CB_N; ++j) {
            int n = n0 + j; if (n >= N) break;
            float8 wv = LOAD8(W + (size_t)n * K, k);
            acc[j] += dot(xv.lo, wv.lo) + dot(xv.hi, wv.hi);
        }
    }
    for (; k < K; ++k) {
        float xk = (float)LOAD(xr, k);
        for (int j = 0; j < CB_N; ++j) { int n = n0 + j; if (n < N) acc[j] += xk * (float)LOAD(W + (size_t)n * K, k); }
    }
    for (int j = 0; j < CB_N; ++j) {
        int n = n0 + j; if (n >= N) break;
        float b = has_bias ? (float)LOAD(bias, n) : 0.0f;
        STORE(out, (size_t)m * N + n, acc[j] + b);
    }
}

// ── Tiled GEMM (local-memory, recordable) ───────────────────────────────────
// C[M,N] = op(A)[M,K] @ op(B)[K,N] (+ bias[N]). transA/transB pick the on-disk
// layout: A is [M,K] (transA=0) or [K,M] (transA=1); B is [K,N] (transB=0) or
// [N,K] (transB=1, e.g. nn.Linear weight). 16×16 tiles staged in local memory so
// each operand element is read once per tile and reused TS times — the naive
// one-thread-per-output kernels were ~130× slower (strided, no reuse). Recordable
// (pure NDRange) unlike CLBlast's indirect GEMM.
#define GEMM_TS 16
__kernel void gemm_tiled(__global const storage_t* A,
                         __global const storage_t* B,
                         __global storage_t* C,
                         const int M, const int N, const int K,
                         const int transA, const int transB,
                         __global const storage_t* bias, const int has_bias) {
    __local float As[GEMM_TS][GEMM_TS];
    __local float Bs[GEMM_TS][GEMM_TS];
    int lc = get_local_id(0), lr = get_local_id(1);
    int col = get_group_id(0) * GEMM_TS + lc;   // N
    int row = get_group_id(1) * GEMM_TS + lr;   // M
    float acc = 0.0f;
    for (int t = 0; t < K; t += GEMM_TS) {
        int ak = t + lc, am = row;              // A[row, ak]
        As[lr][lc] = (am < M && ak < K)
            ? (float)(transA ? LOAD(A, (size_t)ak * M + am) : LOAD(A, (size_t)am * K + ak)) : 0.0f;
        int bk = t + lr, bn = col;              // B[bk, col]
        Bs[lr][lc] = (bn < N && bk < K)
            ? (float)(transB ? LOAD(B, (size_t)bn * K + bk) : LOAD(B, (size_t)bk * N + bn)) : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < GEMM_TS; ++k) acc += As[lr][k] * Bs[k][lc];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) {
        if (has_bias) acc += (float)LOAD(bias, col);
        STORE(C, (size_t)row * N + col, acc);
    }
}

// ── Recordable GEMM (conv1d layout) ─────────────────────────────────────────
// out[M,N] = x[M,K] @ W[K,N]   (HF Conv1D weight layout [K,N] = [in,out]).
// Pure NDRange (one work-item per output element) so it can be captured by a
// cl_qcom recordable queue — CLBlast's indirect GEMM emits an internal pad/copy
// that recording rejects (CL_INVALID_OPERATION). K is small (≤192) here so a
// naive K-loop with fp32 accumulate is cheap and matches CLBlast's accumulation.
__kernel void gemm_mk_kn(__global const storage_t* x,   // [M,K] row-major
                         __global const storage_t* W,   // [K,N] row-major
                         __global storage_t* out,       // [M,N] row-major
                         const int M, const int N, const int K) {
    int idx = get_global_id(0);               // flattened over M*N (1D dispatch)
    if (idx >= M * N) return;
    int m = idx / N, n = idx - m * N;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += (float)LOAD(x, (size_t)m * K + k) * (float)LOAD(W, (size_t)k * N + n);
    STORE(out, (size_t)idx, acc);
}

// out[M,N] = A[Kc,M]^T @ B[Kc,N]   (convtranspose GEMM-map; recordable variant of
// gemm_AtB). Same recordability rationale as gemm_mk_kn.
__kernel void gemm_atb(__global const storage_t* A,    // [Kc,M] row-major
                       __global const storage_t* B,    // [Kc,N] row-major
                       __global storage_t* C,          // [M,N] row-major
                       const int M, const int N, const int Kc) {
    int idx = get_global_id(0);
    if (idx >= M * N) return;
    int m = idx / N, n = idx - m * N;
    float acc = 0.0f;
    for (int k = 0; k < Kc; ++k)
        acc += (float)LOAD(A, (size_t)k * M + m) * (float)LOAD(B, (size_t)k * N + n);
    STORE(C, (size_t)idx, acc);
}

// out[M,N] = x[M,K] @ W[N,K]^T (+ bias[N])   (nn.Linear; recordable variant of
// pytorch_linear). Bias fused. Same recordability rationale as gemm_mk_kn.
__kernel void gemm_mk_nk(__global const storage_t* x,    // [M,K] row-major
                         __global const storage_t* W,    // [N,K] row-major (nn.Linear)
                         __global const storage_t* bias, // [N] or dummy
                         __global storage_t* out,        // [M,N] row-major
                         const int M, const int N, const int K, const int has_bias) {
    int idx = get_global_id(0);
    if (idx >= M * N) return;
    int m = idx / N, n = idx - m * N;
    const __global storage_t* xr = x + (size_t)m * K;
    const __global storage_t* wr = W + (size_t)n * K;   // [n,:] contiguous in k
    float acc = has_bias ? (float)LOAD(bias, n) : 0.0f;
    int k = 0, K8 = K & ~7;
    for (; k < K8; k += 8) {                              // 8-wide K reduction, fp32 accumulate
        float8 xv = LOAD8(xr, k), wv = LOAD8(wr, k);
        acc += dot(xv.lo, wv.lo) + dot(xv.hi, wv.hi);
    }
    for (; k < K; ++k) acc += (float)LOAD(xr, k) * (float)LOAD(wr, k);
    STORE(out, (size_t)idx, acc);
}

// ── GEMV (M=1 decode path) ──────────────────────────────────────────────────
// y[N] = W[N,K] @ x[K]   (no bias — caller applies add_bias separately, as the
// CLBlast path does). One work-item per output row n; vectorized 8-wide loads
// over K with fp32 accumulate (matches CLBlast Hgemm's fp32 accumulation).
// Replaces clblast::Gemm at M=1, where GEMM badly under-utilizes the ALU.
__kernel void gemv_fp16(__global const storage_t* W,   // [N,K] row-major
                        __global const storage_t* x,   // [K]
                        __global const storage_t* bias,// [N] (or dummy)
                        __global storage_t* y,         // [N]
                        const int N, const int K, const int has_bias) {
    int n = get_global_id(0);
    if (n >= N) return;
    const __global storage_t* wrow = W + (size_t)n * K;
    float acc = has_bias ? (float)LOAD(bias, n) : 0.0f;
    int k = 0;
    int K8 = K & ~7;
    for (; k < K8; k += 8) {
        float8 wv = LOAD8(wrow, k);
        float8 xv = LOAD8(x, k);
        acc += dot(wv.lo, xv.lo) + dot(wv.hi, xv.hi);
    }
    for (; k < K; ++k) acc += LOAD(wrow, k) * LOAD(x, k);
    STORE(y, n, acc);
}

// Workgroup-reduction GEMV for large K (decode). One WORKGROUP per output row n;
// the LSZ threads cooperatively reduce over K reading CONSECUTIVE 8-wide chunks
// (coalesced across the wavefront — the per-row kernel above is uncoalesced:
// adjacent threads there read addresses K apart). Requires K % 8 == 0, LSZ power
// of two. gws = N*LSZ, lws = LSZ.
__kernel void gemv_fp16_wg(__global const storage_t* W, __global const storage_t* x,
                           __global const storage_t* bias, __global storage_t* y,
                           const int N, const int K, const int has_bias) {
    int n   = get_group_id(0);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    const __global storage_t* wrow = W + (size_t)n * K;
    float partial = 0.0f;
    for (int k = lid * 8; k < K; k += lsz * 8) {     // thread lid → chunk [k,k+8); coalesced
        float8 w8 = LOAD8(wrow, k);
        float8 x8 = LOAD8(x, k);
        partial += dot(w8.lo, x8.lo) + dot(w8.hi, x8.hi);
    }
    __local float red[256];
    red[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s) red[lid] += red[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) STORE(y, n, red[0] + (has_bias ? (float)LOAD(bias, n) : 0.0f));
}

// ── elementwise ────────────────────────────────────────────────────────────

// out[i] = in[i]  (copy / Identity)
__kernel void copy_buf(__global const storage_t* in, __global storage_t* out, const int n) {
    int g = get_global_id(0);
    if (g < n) STORE(out, g, LOAD(in, g));
}

// a[i] += b[i]   (in-place residual; fp32 accumulate)
__kernel void add_inplace(__global storage_t* a, __global const storage_t* b, const int n) {
    int g = get_global_id(0);
    if (g < n) STORE(a, g, LOAD(a, g) + LOAD(b, g));
}

// out[i] = a[i] + b[i]
__kernel void add_buf(__global const storage_t* a, __global const storage_t* b,
                      __global storage_t* out, const int n) {
    int g = get_global_id(0);
    if (g < n) STORE(out, g, LOAD(a, g) + LOAD(b, g));
}

// out[i] = a[i] * b[i]   (elementwise buffer multiply; ResBlock gate*h)
__kernel void mul_buf(__global const storage_t* a, __global const storage_t* b,
                      __global storage_t* out, const int n) {
    int g = get_global_id(0);
    if (g < n) STORE(out, g, LOAD(a, g) * LOAD(b, g));
}

// out[r*C + c] = in[r*C + c] + bias[c]   (broadcast bias over rows)
__kernel void add_bias(__global storage_t* x, __global const storage_t* bias,
                       const int rows, const int C) {
    int g = get_global_id(0);
    if (g >= rows * C) return;
    int c = g % C;
    STORE(x, g, LOAD(x, g) + LOAD(bias, c));
}

// out[i] = in[i] * s   (scalar scale)
__kernel void scale_buf(__global const storage_t* in, __global storage_t* out,
                        const float s, const int n) {
    int g = get_global_id(0);
    if (g < n) STORE(out, g, LOAD(in, g) * s);
}

// per-channel scale: out[r*C+c] = in[r*C+c] * scale[c]   (LayerScale)
__kernel void mul_channel(__global const storage_t* in, __global const storage_t* scalev,
                          __global storage_t* out, const int rows, const int C) {
    int g = get_global_id(0);
    if (g >= rows * C) return;
    int c = g % C;
    STORE(out, g, LOAD(in, g) * LOAD(scalev, c));
}

// ── activations ────────────────────────────────────────────────────────────

// ELU(alpha=1): x>0 ? x : exp(x)-1
__kernel void elu(__global const storage_t* in, __global storage_t* out, const int n) {
    int g = get_global_id(0);
    if (g >= n) return;
    float x = LOAD(in, g);
    STORE(out, g, x > 0.0f ? x : (exp(x) - 1.0f));
}

// SiLU: x * sigmoid(x)
__kernel void silu(__global const storage_t* in, __global storage_t* out, const int n) {
    int g = get_global_id(0);
    if (g >= n) return;
    float x = LOAD(in, g);
    STORE(out, g, x / (1.0f + exp(-x)));
}

// GELU exact (erf form): 0.5*x*(1+erf(x/sqrt(2)))
__kernel void gelu(__global const storage_t* in, __global storage_t* out, const int n) {
    int g = get_global_id(0);
    if (g >= n) return;
    float x = LOAD(in, g);
    STORE(out, g, 0.5f * x * (1.0f + erf(x * 0.70710678118654752440f)));
}

// ── normalization (one work-item per row; row length C) ─────────────────────

// LayerNorm over last dim C. affine: out = (x-mean)/sqrt(var+eps)*w + b.
// affine==0 ⇒ w/b ignored (norm_final). var is population (biased).
__kernel void layernorm(__global const storage_t* in, __global storage_t* out,
                        __global const storage_t* w, __global const storage_t* b,
                        const int rows, const int C, const float eps, const int affine) {
    int r = get_global_id(0);
    if (r >= rows) return;
    int base = r * C;
    float mean = 0.0f;
    for (int c = 0; c < C; ++c) mean += LOAD(in, base + c);
    mean /= (float)C;
    float var = 0.0f;
    for (int c = 0; c < C; ++c) { float d = LOAD(in, base + c) - mean; var += d * d; }
    var /= (float)C;
    float inv = rsqrt(var + eps);
    for (int c = 0; c < C; ++c) {
        float v = (LOAD(in, base + c) - mean) * inv;
        if (affine) v = v * (float)LOAD(w, c) + (float)LOAD(b, c);
        STORE(out, base + c, v);
    }
}

// Workgroup-reduction LayerNorm: one WORKGROUP per row, threads coalesced over C
// (the per-row kernel above runs ONE thread per row → for decode rows=1 that's a
// single thread reducing C). gws = rows*LSZ, lws = LSZ (power of two). Same math.
__kernel void layernorm_wg(__global const storage_t* in, __global storage_t* out,
                           __global const storage_t* w, __global const storage_t* b,
                           const int rows, const int C, const float eps, const int affine) {
    int r = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    int base = r * C;
    __local float red[256];
    float s = 0.0f;
    for (int c = lid; c < C; c += lsz) s += LOAD(in, base + c);
    red[lid] = s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = lsz >> 1; o > 0; o >>= 1) { if (lid < o) red[lid] += red[lid + o]; barrier(CLK_LOCAL_MEM_FENCE); }
    float mean = red[0] / (float)C;
    barrier(CLK_LOCAL_MEM_FENCE);
    float v = 0.0f;
    for (int c = lid; c < C; c += lsz) { float d = LOAD(in, base + c) - mean; v += d * d; }
    red[lid] = v; barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = lsz >> 1; o > 0; o >>= 1) { if (lid < o) red[lid] += red[lid + o]; barrier(CLK_LOCAL_MEM_FENCE); }
    float inv = rsqrt(red[0] / (float)C + eps);
    for (int c = lid; c < C; c += lsz) {
        float val = (LOAD(in, base + c) - mean) * inv;
        if (affine) val = val * (float)LOAD(w, c) + (float)LOAD(b, c);
        STORE(out, base + c, val);
    }
}

// RMSNorm (mlp.py _rms_norm): var = eps + population-var(x); out = x*alpha*rsqrt(var).
// NOTE: variance (mean-subtracted), eps INSIDE the var, per TimestepEmbedder.mlp[3].
__kernel void rmsnorm_var(__global const storage_t* in, __global storage_t* out,
                          __global const storage_t* alpha,
                          const int rows, const int C, const float eps) {
    int r = get_global_id(0);
    if (r >= rows) return;
    int base = r * C;
    float mean = 0.0f;
    for (int c = 0; c < C; ++c) mean += LOAD(in, base + c);
    mean /= (float)C;
    float var = 0.0f;
    for (int c = 0; c < C; ++c) { float d = LOAD(in, base + c) - mean; var += d * d; }
    var = eps + var / (float)C;
    float inv = rsqrt(var);
    for (int c = 0; c < C; ++c)
        STORE(out, base + c, LOAD(in, base + c) * (float)LOAD(alpha, c) * inv);
}

// adaLN modulate: out = x*(1+scale) + shift, with scale/shift broadcast per row.
// x:[rows,C], scale/shift:[rows,C] (same shape; one (shift,scale) vector per row).
// modulate reading shift/scale from a PACKED buffer (e.g. adaLN mod[3C]) at column
// offsets — saves the separate slice_cols enqueues. rows=1. out = x*(1+scale)+shift.
__kernel void modulate_packed(__global const storage_t* x, __global const storage_t* mod,
                              __global storage_t* out, const int C,
                              const int shiftOff, const int scaleOff) {
    int c = get_global_id(0);
    if (c >= C) return;
    float sh = LOAD(mod, shiftOff + c), sc = LOAD(mod, scaleOff + c);
    STORE(out, c, LOAD(x, c) * (1.0f + sc) + sh);
}

// set a 1-int device buffer (the record/replay offset). val is captured at enqueue
// time (clSetKernelArg copies it), so per-frame enqueues are race-free under replay.
__kernel void set_int(__global int* buf, const int val) { if (get_global_id(0) == 0) buf[0] = val; }

// out[c] = a[aOff + c] * b[c]   (gate*h with gate packed in `a` at offset aOff)
__kernel void mul_off(__global const storage_t* a, __global const storage_t* b,
                      __global storage_t* out, const int aOff, const int C) {
    int c = get_global_id(0);
    if (c >= C) return;
    STORE(out, c, LOAD(a, aOff + c) * LOAD(b, c));
}

__kernel void modulate(__global const storage_t* x, __global const storage_t* shift,
                       __global const storage_t* scale, __global storage_t* out,
                       const int rows, const int C) {
    int g = get_global_id(0);
    if (g >= rows * C) return;
    int c = g % C;
    int rr = g / C;
    float sh = LOAD(shift, rr * C + c);
    float sc = LOAD(scale, rr * C + c);
    STORE(out, g, LOAD(x, g) * (1.0f + sc) + sh);
}

// de-normalize latent: out[t*C+c] = x[t*C+c]*std[c] + mean[c]   (C=ldim=32)
__kernel void denorm_latent(__global const storage_t* x, __global const storage_t* stdv,
                            __global const storage_t* meanv, __global storage_t* out,
                            const int rows, const int C) {
    int g = get_global_id(0);
    if (g >= rows * C) return;
    int c = g % C;
    STORE(out, g, LOAD(x, g) * (float)LOAD(stdv, c) + (float)LOAD(meanv, c));
}

// ── convolutions (channels-first [C,T]) ────────────────────────────────────

// Causal Conv1d (stride 1), left-padded with (K-1)*dil zeros (pad_mode constant,
// first frame / no streaming state). weight [Cout,Cin,K]. out length == T.
// out[co,t] = bias[co] + sum_ci sum_kk w[co,ci,kk] * in[ci, t-(K-1)*dil + kk*dil]
__kernel void conv1d_causal(__global const storage_t* in, __global const storage_t* w,
                            __global const storage_t* bias, __global storage_t* out,
                            const int Cin, const int Cout, const int T,
                            const int K, const int dil, const int has_bias) {
    int g = get_global_id(0);
    if (g >= Cout * T) return;
    int co = g / T, t = g - co * T;
    float acc = has_bias ? (float)LOAD(bias, co) : 0.0f;
    for (int ci = 0; ci < Cin; ++ci) {
        int wbase = (co * Cin + ci) * K;
        int ibase = ci * T;
        for (int kk = 0; kk < K; ++kk) {
            int src = t - (K - 1) * dil + kk * dil;
            if (src >= 0 && src < T) acc += (float)LOAD(w, wbase + kk) * (float)LOAD(in, ibase + src);
        }
    }
    STORE(out, co * T + t, acc);
}

// Streaming causal Conv1d: left-pad with `leftctx` [Cin, P] (prev frame tail,
// P=(K-1)*dil) instead of zeros. out length == T. Carries continuity across frames.
__kernel void conv1d_streaming(__global const storage_t* in, __global const storage_t* leftctx,
                               __global const storage_t* w, __global const storage_t* bias,
                               __global storage_t* out, const int Cin, const int Cout,
                               const int T, const int K, const int dil, const int P,
                               const int has_bias) {
    int g = get_global_id(0);
    if (g >= Cout * T) return;
    int co = g / T, t = g - co * T;
    float acc = has_bias ? (float)LOAD(bias, co) : 0.0f;
    for (int ci = 0; ci < Cin; ++ci) {
        int wbase = (co * Cin + ci) * K;
        for (int kk = 0; kk < K; ++kk) {
            int j = t - (K - 1) * dil + kk * dil;
            float xv;
            if (j >= 0) xv = (float)LOAD(in, ci * T + j);
            else { int li = P + j; xv = (li >= 0) ? (float)LOAD(leftctx, ci * P + li) : 0.0f; }
            acc += (float)LOAD(w, wbase + kk) * xv;
        }
    }
    STORE(out, co * T + t, acc);
}

// extract the last P columns of [C,T] → [C,P]  (next frame's left-context)
__kernel void extract_tail(__global const storage_t* in, __global storage_t* out,
                           const int C, const int T, const int P) {
    int g = get_global_id(0);
    if (g >= C * P) return;
    int c = g / P, p = g - c * P;
    STORE(out, c * P + p, LOAD(in, c * T + (T - P) + p));
}

// ConvTranspose1d producing the FULL (untrimmed) output, length Tfull=(T-1)*S+K.
__kernel void convtranspose1d_full(__global const storage_t* in, __global const storage_t* w,
                                   __global const storage_t* bias, __global storage_t* out,
                                   const int Cin, const int Cout, const int T,
                                   const int K, const int S, const int Tfull, const int has_bias) {
    int g = get_global_id(0);
    if (g >= Cout * Tfull) return;
    int co = g / Tfull, to = g - co * Tfull;
    float acc = has_bias ? (float)LOAD(bias, co) : 0.0f;
    for (int kk = 0; kk < K; ++kk) {
        int num = to - kk;
        if (num < 0 || num % S != 0) continue;
        int ti = num / S;
        if (ti >= T) continue;
        for (int ci = 0; ci < Cin; ++ci)
            acc += (float)LOAD(w, (ci * Cout + co) * K + kk) * (float)LOAD(in, ci * T + ti);
    }
    STORE(out, co * Tfull + to, acc);
}

// im2col for GEMM-mapped causal Conv1d (streaming). Builds xcol[Cin*K, T] where
// xcol[(ci*K+kk), t] = in_padded[ci, t-(K-1)*dil+kk*dil] (left-context for j<0).
// Then out[Cout,T] = W[Cout,Cin*K] @ xcol via pytorch_conv1d. Same math as
// conv1d_streaming, but routes the Cin*K reduction through CLBlast.
__kernel void conv1d_im2col(__global const storage_t* in, __global const storage_t* leftctx,
                            __global storage_t* xcol, const int Cin, const int T,
                            const int K, const int dil, const int P) {
    int g = get_global_id(0);
    if (g >= Cin * K * T) return;
    int row = g / T, t = g - row * T;
    int ci = row / K, kk = row - ci * K;
    int j = t - (K - 1) * dil + kk * dil;
    float v;
    if (j >= 0) v = (float)LOAD(in, ci * T + j);
    else { int li = P + j; v = (li >= 0) ? (float)LOAD(leftctx, ci * P + li) : 0.0f; }
    STORE(xcol, g, v);
}

// out[co*T+t] += bias[co]  (per-row bias for GEMM-mapped convs; channel-first [Cout,T])
__kernel void add_bias_rows(__global storage_t* x, __global const storage_t* bias,
                            const int rows, const int T) {
    int g = get_global_id(0);
    if (g >= rows * T) return;
    STORE(x, g, LOAD(x, g) + LOAD(bias, g / T));
}

// col2im scatter for GEMM-mapped ConvTranspose1d. cols[Cout*K, T] holds
// cols[(co*K+kk), ti] = sum_ci W[ci,co,kk]*in[ci,ti] (computed by gemm_AtB).
// full[co, to] = bias[co] + sum over (ti,kk) with to = ti*S+kk of cols[(co*K+kk),ti].
// One work-item per output element; iterates only the K/S contributing taps.
__kernel void convtr_col2im(__global const storage_t* cols, __global const storage_t* bias,
                            __global storage_t* full, const int Cout, const int K,
                            const int S, const int T, const int Tfull, const int has_bias) {
    int g = get_global_id(0);
    if (g >= Cout * Tfull) return;
    int co = g / Tfull, to = g - co * Tfull;
    float acc = has_bias ? (float)LOAD(bias, co) : 0.0f;
    int ti = to / S; if (ti > T - 1) ti = T - 1;
    for (; ti >= 0; --ti) {
        int kk = to - ti * S;          // grows by S as ti decreases
        if (kk >= K) break;
        acc += (float)LOAD(cols, (co * K + kk) * T + ti);
    }
    STORE(full, co * Tfull + to, acc);
}

// Depthwise ConvTranspose1d, FULL output length Tfull=(T-1)*S+K. weight [C,1,K].
__kernel void convtranspose1d_dw_full(__global const storage_t* in, __global const storage_t* w,
                                      __global storage_t* out, const int C, const int T,
                                      const int K, const int S, const int Tfull) {
    int g = get_global_id(0);
    if (g >= C * Tfull) return;
    int c = g / Tfull, to = g - c * Tfull;
    float acc = 0.0f;
    for (int kk = 0; kk < K; ++kk) {
        int num = to - kk;
        if (num < 0 || num % S != 0) continue;
        int ti = num / S;
        if (ti >= T) continue;
        acc += (float)LOAD(w, c * K + kk) * (float)LOAD(in, c * T + ti);
    }
    STORE(out, c * Tfull + to, acc);
}

// out[c, 0:Padd] += partial[c, 0:Padd]   (overlap-add prefix; out is [C, Tout])
__kernel void overlap_add(__global storage_t* out, __global const storage_t* partial,
                          const int C, const int Tout, const int Padd) {
    int g = get_global_id(0);
    if (g >= C * Padd) return;
    int c = g / Padd, p = g - c * Padd;
    STORE(out, c * Tout + p, LOAD(out, c * Tout + p) + LOAD(partial, c * Padd + p));
}

// extract columns [c0:c0+w) of [C,full] → [C,w], optionally subtracting bias[c].
__kernel void extract_cols_sub_bias(__global const storage_t* in, __global const storage_t* bias,
                                    __global storage_t* out, const int C, const int full,
                                    const int c0, const int w, const int has_bias) {
    int g = get_global_id(0);
    if (g >= C * w) return;
    int c = g / w, p = g - c * w;
    float v = (float)LOAD(in, c * full + c0 + p);
    if (has_bias) v -= (float)LOAD(bias, c);
    STORE(out, c * w + p, v);
}

// ConvTranspose1d, stride S, weight [Cin,Cout,K]. Streaming first-frame output
// length Tout = T*S (the (K-S) overlap tail is dropped / carried as state).
// out[co,to] = bias[co] + sum_kk (ti=(to-kk)/S integer, 0<=ti<T) sum_ci w[ci,co,kk]*in[ci,ti]
__kernel void convtranspose1d(__global const storage_t* in, __global const storage_t* w,
                              __global const storage_t* bias, __global storage_t* out,
                              const int Cin, const int Cout, const int T,
                              const int K, const int S, const int Tout, const int has_bias) {
    int g = get_global_id(0);
    if (g >= Cout * Tout) return;
    int co = g / Tout, to = g - co * Tout;
    float acc = has_bias ? (float)LOAD(bias, co) : 0.0f;
    for (int kk = 0; kk < K; ++kk) {
        int num = to - kk;
        if (num < 0) continue;
        if (num % S != 0) continue;
        int ti = num / S;
        if (ti >= T) continue;
        for (int ci = 0; ci < Cin; ++ci)
            acc += (float)LOAD(w, (ci * Cout + co) * K + kk) * (float)LOAD(in, ci * T + ti);
    }
    STORE(out, co * Tout + to, acc);
}

// Depthwise (groups==channels) ConvTranspose1d, stride S, weight [C,1,K].
// Used by mimi.upsample (groups=512). out[c,to] = sum_kk(ti) w[c,0,kk]*in[c,ti].
__kernel void convtranspose1d_depthwise(__global const storage_t* in, __global const storage_t* w,
                                        __global storage_t* out,
                                        const int C, const int T,
                                        const int K, const int S, const int Tout) {
    int g = get_global_id(0);
    if (g >= C * Tout) return;
    int c = g / Tout, to = g - c * Tout;
    float acc = 0.0f;
    for (int kk = 0; kk < K; ++kk) {
        int num = to - kk;
        if (num < 0) continue;
        if (num % S != 0) continue;
        int ti = num / S;
        if (ti >= T) continue;
        acc += (float)LOAD(w, c * K + kk) * (float)LOAD(in, c * T + ti);
    }
    STORE(out, c * Tout + to, acc);
}

// transpose a row-major [A,B] tensor → [B,A].
__kernel void transpose2d(__global const storage_t* in, __global storage_t* out,
                          const int A, const int B) {
    int g = get_global_id(0);
    if (g >= A * B) return;
    int a = g / B, b = g - a * B;
    STORE(out, b * A + a, LOAD(in, a * B + b));
}

// slice columns [c0:c0+w) from a row-major [T,full] tensor → [T,w].
__kernel void slice_cols(__global const storage_t* in, __global storage_t* out,
                         const int T, const int full, const int c0, const int w) {
    int g = get_global_id(0);
    if (g >= T * w) return;
    int t = g / w, c = g - t * w;
    STORE(out, g, LOAD(in, t * full + c0 + c));
}

// ── attention (RoPE + scaled-dot-product, streaming KV) ─────────────────────

// RoPE, interleaved-pair convention, applied to x of shape [T,H,D] in place.
// freqs[j]=exp(-log(max_period)*2j/D), pos=offset+t. Pair (2j,2j+1):
//   x0' = x0*cos - x1*sin ; x1' = x0*sin + x1*cos.  Computed in fp32.
__kernel void rope_inplace(__global storage_t* x, const int T, const int H, const int D,
                           const int offset, const float max_period) {
    int g = get_global_id(0);            // one item per (t,h,j), j in [0,D/2)
    int HD2 = H * (D / 2);
    if (g >= T * HD2) return;
    int t = g / HD2; int rem = g - t * HD2;
    int h = rem / (D / 2); int j = rem - h * (D / 2);
    float freq = exp(-log(max_period) * (2.0f * j) / (float)D);
    float ang = freq * (float)(offset + t);
    float cs = cos(ang), sn = sin(ang);
    int base = (t * H + h) * D + 2 * j;
    float x0 = LOAD(x, base), x1 = LOAD(x, base + 1);
    STORE(x, base,     x0 * cs - x1 * sn);
    STORE(x, base + 1, x0 * sn + x1 * cs);
}

// Fused qkv-split + RoPE + KV-cache write. Replaces 3×slice_cols + 2×rope_inplace
// + 2×cache_append (7 launches) per attention with ONE kernel. qkv:[T,3d], row t =
// [q(d) | k(d) | v(d)]. Applies RoPE to q,k; writes roped q→q_out[T,d]; writes roped
// k and v straight into kc/vc[cap,d] at row (offset+t). One item per (t,h,j), j<Dh/2.
__kernel void qkv_rope_cache(__global const storage_t* qkv, __global storage_t* q_out,
                             __global storage_t* kc, __global storage_t* vc,
                             const int T, const int d, const int H, const int Dh,
                             __global const int* offbuf, const float max_period) {
    int offset = offbuf[0];   // device-buffer offset (record/replay-safe; not a kernel-arg override)
    int HD2 = H * (Dh / 2);
    int g = get_global_id(0);
    if (g >= T * HD2) return;
    int t = g / HD2; int rem = g - t * HD2;
    int h = rem / (Dh / 2); int j = rem - h * (Dh / 2);
    float freq = exp(-log(max_period) * (2.0f * j) / (float)Dh);
    float ang = freq * (float)(offset + t);
    float cs = cos(ang), sn = sin(ang);
    int hd = h * Dh + 2 * j;
    int qb = t * 3 * d + hd, kb = t * 3 * d + d + hd, vb = t * 3 * d + 2 * d + hd;
    int outd = t * d + hd, cache = (offset + t) * d + hd;
    float q0 = LOAD(qkv, qb), q1 = LOAD(qkv, qb + 1);
    STORE(q_out, outd,     q0 * cs - q1 * sn);
    STORE(q_out, outd + 1, q0 * sn + q1 * cs);
    float k0 = LOAD(qkv, kb), k1 = LOAD(qkv, kb + 1);
    STORE(kc, cache,     k0 * cs - k1 * sn);
    STORE(kc, cache + 1, k0 * sn + k1 * cs);
    STORE(vc, cache,     LOAD(qkv, vb));
    STORE(vc, cache + 1, LOAD(qkv, vb + 1));
}

// Scaled-dot-product attention over a streaming KV cache. q:[Tq,H,D],
// kc/vc:[Tkv,H,D] (absolute positions 0..Tkv-1). out:[Tq,H,D]. Causal:
// key tk allowed iff tk <= offset+tq; sliding window: also tk > offset+tq-context
// (context<=0 ⇒ unbounded). scale = 1/sqrt(D). fp32 accumulation.
// Single-pass online (flash-style) softmax attention, vectorized over D (vload_half8,
// fp32 accumulate). Requires D % 8 == 0 (D=64 here). One work-item per (tq,h) — keeps
// the high prefill thread count; for decode (Tq=1) the win is single-pass + 8-wide D
// vs the old 2-pass scalar kernel. Numerically identical (running-max rescale).
__kernel void attention(__global const storage_t* q, __global const storage_t* kc,
                        __global const storage_t* vc, __global storage_t* out,
                        const int Tq, const int Tkv, const int H, const int D,
                        const int offset, const int context) {
    int g = get_global_id(0);
    if (g >= Tq * H) return;
    int tq = g / H, h = g - tq * H;
    float scale = 1.0f / sqrt((float)D);
    int qpos = offset + tq;
    int qbase = (tq * H + h) * D;
    int lo = 0, hi = qpos;               // inclusive key-position range
    if (hi > Tkv - 1) hi = Tkv - 1;
    if (context > 0 && lo < qpos - context + 1) lo = qpos - context + 1;
    if (lo < 0) lo = 0;
    int D8 = D >> 3;
    float m = -1e30f, denom = 0.0f;
    float8 acc[16];                      // D<=128 → D8<=16
    for (int i = 0; i < D8; ++i) acc[i] = (float8)(0.0f);
    for (int tk = lo; tk <= hi; ++tk) {
        int kbase = (tk * H + h) * D;
        float s = 0.0f;
        for (int i = 0; i < D8; ++i) {
            float8 q8 = LOAD8(q, qbase + (i << 3));
            float8 k8 = LOAD8(kc, kbase + (i << 3));
            s += dot(q8.lo, k8.lo) + dot(q8.hi, k8.hi);
        }
        s *= scale;
        float m_new = fmax(m, s);
        float corr = exp(m - m_new);     // rescale prior accumulation when max grows
        float w    = exp(s - m_new);
        denom = denom * corr + w;
        for (int i = 0; i < D8; ++i) {
            float8 v8 = LOAD8(vc, kbase + (i << 3));
            acc[i] = acc[i] * corr + w * v8;
        }
        m = m_new;
    }
    float inv = 1.0f / (denom + 1e-20f);
    for (int i = 0; i < D8; ++i) STORE8(out, qbase + (i << 3), acc[i] * inv);
}

// Workgroup attention: one WORKGROUP per (tq,h), LSZ threads parallelize over the
// KV positions (the per-item kernel above runs ONE work-item per (tq,h) → only
// Tq*H = 16 work-items at decode). 3 phases via local mem: (1) scores+max,
// (2) exp+denom, (3) output parallel over D. Requires D%8==0, nk<=320, LSZ pow2.
__kernel void attention_wg(__global const storage_t* q, __global const storage_t* kc,
                           __global const storage_t* vc, __global storage_t* out,
                           const int Tq, const int H, const int D,
                           __global const int* offbuf, const int context) {
    int offset = offbuf[0];        // device-buffer offset (record/replay-safe)
    int Tkv = offset + Tq;         // attend to all positions up to offset+Tq
    int wg = get_group_id(0);
    int tq = wg / H, h = wg - tq * H;
    int lid = get_local_id(0), lsz = get_local_size(0);
    float scale = 1.0f / sqrt((float)D);
    int qpos = offset + tq;
    int qbase = (tq * H + h) * D;
    int lo = 0, hi = qpos; if (hi > Tkv - 1) hi = Tkv - 1;
    if (context > 0 && lo < qpos - context + 1) lo = qpos - context + 1;
    if (lo < 0) lo = 0;
    int nk = hi - lo + 1;
    int D8 = D >> 3;
    __local float scores[320];
    __local float red[256];
    // phase 1: scores[i] = scale * q·k(lo+i), and per-thread max
    float lmax = -1e30f;
    for (int i = lid; i < nk; i += lsz) {
        int kbase = ((lo + i) * H + h) * D;
        float s = 0.0f;
        for (int j = 0; j < D8; ++j) {
            float8 q8 = LOAD8(q, qbase + (j << 3));
            float8 k8 = LOAD8(kc, kbase + (j << 3));
            s += dot(q8.lo, k8.lo) + dot(q8.hi, k8.hi);
        }
        s *= scale; scores[i] = s; if (s > lmax) lmax = s;
    }
    red[lid] = lmax; barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = lsz >> 1; o > 0; o >>= 1) { if (lid < o) red[lid] = fmax(red[lid], red[lid + o]); barrier(CLK_LOCAL_MEM_FENCE); }
    float m = red[0]; barrier(CLK_LOCAL_MEM_FENCE);
    // phase 2: scores[i] = exp(scores[i]-m); per-thread denom
    float ldenom = 0.0f;
    for (int i = lid; i < nk; i += lsz) { float e = exp(scores[i] - m); scores[i] = e; ldenom += e; }
    red[lid] = ldenom; barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = lsz >> 1; o > 0; o >>= 1) { if (lid < o) red[lid] += red[lid + o]; barrier(CLK_LOCAL_MEM_FENCE); }
    float inv = 1.0f / (red[0] + 1e-20f); barrier(CLK_LOCAL_MEM_FENCE);
    // phase 3: out[d] = inv * sum_i scores[i] * v(lo+i, d), parallel over d
    for (int d = lid; d < D; d += lsz) {
        float acc = 0.0f;
        for (int i = 0; i < nk; ++i) acc += scores[i] * (float)LOAD(vc, ((lo + i) * H + h) * D + d);
        STORE(out, qbase + d, acc * inv);
    }
}

// embedding gather: out[t*dim+c] = table[ids[t]*dim + c]
__kernel void embedding_gather(__global const storage_t* table, __global const int* ids,
                               __global storage_t* out, const int n_tok, const int dim) {
    int g = get_global_id(0);
    if (g >= n_tok * dim) return;
    int t = g / dim;
    int c = g - t * dim;
    int row = ids[t];
    STORE(out, g, LOAD(table, row * dim + c));
}
