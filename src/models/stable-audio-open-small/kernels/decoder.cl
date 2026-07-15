// decoder.cl — Oobleck autoencoder decoder activations.
// Reference: stable_audio_tools/models/blocks.py snake_beta / SnakeBeta.forward
//   snake_beta(x, alpha, beta) = x + (1/(beta+1e-9)) * sin(x*alpha)^2
//   alpha,beta are per-channel; alpha_logscale=True so exp(alpha), exp(beta).
// Layout: x is [C, L] (channel-major, batch=1). alpha/beta are [C].

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

// SnakeBeta with alpha_logscale=True: a=exp(alpha[c]), b=exp(beta[c]).
// Opt: one work-item processes SNAKE_CHUNK consecutive L positions of one
// channel — exp(alpha)/exp(beta)/reciprocal computed once per chunk instead
// of once per element; native_ transcendentals.
#define SNAKE_CHUNK 8
__kernel void snake_beta(__global storage_t* x,           // [C, L] in/out
                         __global const storage_t* alpha,  // [C]
                         __global const storage_t* beta,   // [C]
                         const int C, const int L) {
    const int chunks_per_ch = (L + SNAKE_CHUNK - 1) / SNAKE_CHUNK;
    const int gid = get_global_id(0);
    if (gid >= C * chunks_per_ch) return;
    const int c  = gid / chunks_per_ch;
    const int l0 = (gid % chunks_per_ch) * SNAKE_CHUNK;
    const float a  = native_exp(LOAD(alpha, c));
    const float rb = 1.0f / (native_exp(LOAD(beta, c)) + 1.0e-9f);
    const int base = c * L;
    const int lend = min(l0 + SNAKE_CHUNK, L);
    for (int l = l0; l < lend; l++) {
        const float v = LOAD(x, base + l);
        const float s = native_sin(v * a);
        STORE(x, base + l, v + rb * (s * s));
    }
}

// out[i] = a[i] + b[i]   (residual add, [C,L])
__kernel void add_cl(__global const storage_t* a,
                     __global const storage_t* b,
                     __global storage_t* out,
                     const int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    STORE(out, gid, LOAD(a, gid) + LOAD(b, gid));
}


// out[c, l] += bias[c]  — epilogue for the GEMM conv path.
__kernel void bias_add_rows(__global storage_t* x,          // [C, L]
                            __global const storage_t* bias,  // [C]
                            const int C, const int L) {
    const int gid = get_global_id(0);
    if (gid >= C * L) return;
    const int c = gid / L;
    STORE(x, gid, LOAD(x, gid) + LOAD(bias, c));
}

// ═══ B7 (2026-07 campaign): 128-bit vectorized elementwise variants ═══
// Guide §6.3/§7.2: scalar vload_half is a 16-bit load path; these move the
// hot decoder elementwise ops to half8 (128-bit) transactions. Hosts fall
// back to the scalar kernels when n/L isn't 8-aligned or NNOPT_VEC_KERNELS=0.
#ifdef USE_FP16
  #define LOAD8V(p, i)     convert_float8(vload_half8(0, (p) + (i)))
  #define STORE8V(p, i, v) vstore_half8((v), 0, (p) + (i))
#else
  #define LOAD8V(p, i)     vload8(0, (p) + (i))
  #define STORE8V(p, i, v) vstore8((v), 0, (p) + (i))
#endif

__kernel void snake_beta_v8(__global storage_t* x,           // [C, L], L % 8 == 0
                            __global const storage_t* alpha, // [C]
                            __global const storage_t* beta,  // [C]
                            const int C, const int L) {
    const int Lv = L >> 3;
    const int gid = get_global_id(0);
    if (gid >= C * Lv) return;
    const int c = gid / Lv;
    const float a  = native_exp((float)LOAD(alpha, c));
    const float rb = 1.0f / (native_exp((float)LOAD(beta, c)) + 1.0e-9f);
    const size_t base = (size_t)gid * 8;
    const float8 v = LOAD8V(x, base);
    const float8 s = native_sin(v * a);
    STORE8V(x, base, v + rb * (s * s));
}

__kernel void add_cl_v8(__global const storage_t* a,
                        __global const storage_t* b,
                        __global storage_t* out,
                        const int n) {              // n % 8 == 0
    const int gid = get_global_id(0);
    if (gid >= (n >> 3)) return;
    const size_t off = (size_t)gid * 8;
    STORE8V(out, off, LOAD8V(a, off) + LOAD8V(b, off));
}

__kernel void bias_add_rows_v8(__global storage_t* x,          // [C, L], L % 8 == 0
                               __global const storage_t* bias, // [C]
                               const int C, const int L) {
    const int Lv = L >> 3;
    const int gid = get_global_id(0);
    if (gid >= C * Lv) return;
    const int c = gid / Lv;
    const size_t off = (size_t)gid * 8;
    STORE8V(x, off, LOAD8V(x, off) + (float8)((float)LOAD(bias, c)));
}
