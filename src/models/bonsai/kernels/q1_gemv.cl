// BONSAI_NO_FMA: disable fp contraction for bit-consistent fp32 across
// GPUs (Adreno mad-fusion flips near-tie logits vs the CPU oracle).
#ifdef BONSAI_NO_FMA
#pragma OPENCL FP_CONTRACT OFF
#endif
// 1-bit GEMV for Bonsai-8B Q1_0 decode path — adapted from the proven
// qwen2.5 gemv_m1 shape (WG=64 cooperative per output row, fp32 tree
// reduce), inner loop rewritten for 18-byte Q1 units:
//   unit = [fp16 scale s][16 bytes = 128 sign bits, LSB-first]
//   dot(unit, x) = s * (2 * sum_{bit=1} x[i] - sum_i x[i])
// The per-unit activation sums (sum_i x[i] over each 64-lane half-unit) are
// precomputed ONCE per token by q1_xsum and shared across all N rows.
//
// Thread mapping: 2 threads per unit — thread t owns unit u = base + (t>>1),
// half h = t&1 (bytes [2+h*8, 2+h*8+8), x slice [u*128 + h*64, +64)).
// K % 128 == 0 always (4096, 1024, 12288). Dispatch: gws = N*64, lws = 64.

// Dtype template: activations/outputs are storage_t (fp32 first for the
// P2 token-exact gate; -DUSE_FP16 is the P3 A/B). Weight-unit scales are
// ALWAYS fp16 (that is the packed format), independent of storage_t.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef USE_FP16
  typedef half storage_t;
  #define LOAD4(p, i)     vload_half4((i), (__global const half*)(p))
  #define STORE1(v, p, i) vstore_half((v), (i), (__global half*)(p))
  #define LOAD1(p, i)     vload_half((i), (__global const half*)(p))
#else
  typedef float storage_t;
  #define LOAD4(p, i)     vload4((i), (__global const float*)(p))
  #define STORE1(v, p, i) (((__global float*)(p))[(i)] = (v))
  #define LOAD1(p, i)     (((__global const float*)(p))[(i)])
#endif

#define WG_SIZE 64

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_xsum(__global const storage_t* x,  // [K]
             __global float* xsum,          // [K/64]
             const int K) {
    const int i = (int)get_global_id(0);    // one half-unit (64 lanes) per WI
    if (i >= K / 64) return;
    const __global storage_t* xs = x + i * 64;
    float acc = 0.0f;
    for (int j = 0; j < 16; ++j) {
        const float4 v = LOAD4(xs, j);
        acc += v.x + v.y + v.z + v.w;
    }
    xsum[i] = acc;
}

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemv(__global const uchar* W,     // Q1 units, row-major [N][K/128]
             __global const storage_t* x, // [K]
             __global const float* xsum,  // [K/64] from q1_xsum
             __global storage_t* out,     // [N] (written at out_off + n)
             const int N,
             const int K,
             const int out_off,
             const int n_base) {
    // n_base: Adreno wraps workgroup ids at 65535 per grid dimension —
    // the host chunks any N > 32768 into multiple launches (the vocab
    // logits GEMV is 151669 rows; without this, rows >= 65536 silently
    // duplicate low rows and every high-id token gets a garbage logit).
    const int n   = (int)get_group_id(0) + n_base;
    const int tid = (int)get_local_id(0);
    const int U   = K >> 7;               // units per row
    const __global uchar* row = W + ((long)n * U) * 18;

    float acc = 0.0f;
    // 2 threads per unit; stride the WG across units 32 at a time
    for (int base = 0; base < U; base += WG_SIZE / 2) {
        const int u = base + (tid >> 1);
        if (u >= U) break;
        const int h = tid & 1;
        const __global uchar* unit = row + (long)u * 18;
        const float s = vload_half(0, (__global const half*)unit);
        const __global uchar* bits = unit + 2 + h * 8;
        const __global storage_t* xs = x + (u << 7) + (h << 6);
        float pos = 0.0f;
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            const int w = bits[b];
            const float4 xa = LOAD4(xs, b * 2);
            const float4 xb = LOAD4(xs, b * 2 + 1);
            const int4 ma = ((int4)(w) >> (int4)(0, 1, 2, 3)) & (int4)(1);
            const int4 mb = ((int4)(w) >> (int4)(4, 5, 6, 7)) & (int4)(1);
            pos += dot(xa, convert_float4(ma));
            pos += dot(xb, convert_float4(mb));
        }
        acc += s * (2.0f * pos - xsum[(u << 1) + h]);
    }

    __local float partial[WG_SIZE];
    partial[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s2 = WG_SIZE >> 1; s2 > 0; s2 >>= 1) {
        if (tid < s2) partial[tid] += partial[tid + s2];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) STORE1(partial[0], out, out_off + n);
}

// 1-bit embedding row gather: decode ONE token's row to fp16 activations.
__kernel
void q1_row_gather(__global const uchar* W,   // token_embd units [V][K/128]
                   __global storage_t* out,   // [K]
                   const int token,
                   const int K) {
    const int i = (int)get_global_id(0);      // one weight element per WI
    if (i >= K) return;
    const int U = K >> 7;
    const int u = i >> 7, r = i & 127;
    const __global uchar* unit = W + ((long)token * U + u) * 18;
    const float s = vload_half(0, (__global const half*)unit);
    const int bit = (unit[2 + (r >> 3)] >> (r & 7)) & 1;
    STORE1(bit ? s : -s, out, i);
}

// Greedy argmax over fp16 logits — readback is ONE int, never 151669 floats.
__kernel
__attribute__((reqd_work_group_size(256, 1, 1)))
void q1_argmax(__global const storage_t* logits, __global int* out, const int N) {
    const int tid = (int)get_local_id(0);
    __local float best[256];
    __local int bidx[256];
    float b = -1e30f;
    int bi = 0;
    for (int i = tid; i < N; i += 256) {
        const float v = LOAD1(logits, i);
        if (v > b || (v == b && i < bi)) { b = v; bi = i; }
    }
    best[tid] = b; bidx[tid] = bi;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (best[tid + s] > best[tid] ||
                (best[tid + s] == best[tid] && bidx[tid + s] < bidx[tid])) {
                best[tid] = best[tid + s];
                bidx[tid] = bidx[tid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) out[0] = bidx[0];
}

// ── v2: split-stream GEMV (P3 instruction diet) ──────────────────────────
// Weights repacked at UPLOAD (still 1-bit; just deinterleaved for the GPU):
//   bits:   [N][K/64] uint2 — 4-byte aligned sign words (LSB-first)
//   scales: [N][K/128] half — one per 128-weight unit
// Per thread-half (64 weights): ONE aligned uint2 load + 16 mask-AND float4
// accumulates from __local-staged x. No converts, no dots, no byte loads.
#define XCHUNK 4096

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemv2(__global const uint2* bits,     // [N][K/64]
              __global const half* scales,    // [N][K/128]
              __global const storage_t* x,    // [K]
              __global const float* xsum,     // [K/64]
              __global storage_t* out,
              const int N,
              const int K,
              const int out_off,
              const int n_base) {
    const int n   = (int)get_group_id(0) + n_base;
    const int tid = (int)get_local_id(0);
    const int U   = K >> 7;                   // 128-weight units per row
    __local float xl[XCHUNK];                 // staged x slice (fp32 in local)

    float acc = 0.0f;
    for (int kb = 0; kb < K; kb += XCHUNK) {
        const int kn = min(XCHUNK, K - kb);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = tid * 4; i < kn; i += WG_SIZE * 4) {
            const float4 v = LOAD4(x + kb, i >> 2);
            vstore4(v, i >> 2, xl);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int u0 = kb >> 7, nu = kn >> 7;
        // 2 threads per unit: thread-half h owns 64 weights (one uint2)
        for (int ub = 0; ub < nu; ub += WG_SIZE / 2) {
            const int ul = ub + (tid >> 1);
            if (ul >= nu) break;
            const int u = u0 + ul;
            const int h = tid & 1;
            const uint2 bw = bits[((long)n * U + u) * 2 + h];
            const float s = vload_half((long)n * U + u, scales);
            const __local float* xs = xl + (ul << 7) + (h << 6);
            float4 pos4 = (float4)(0.0f);
            uint w = bw.x;
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                const int4 m = ((int4)(w) << (int4)(31, 30, 29, 28)) >> (int4)(31);
                const float4 xv = vload4(g, xs);
                pos4 += as_float4(as_int4(xv) & m);
                w >>= 4;
            }
            w = bw.y;
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                const int4 m = ((int4)(w) << (int4)(31, 30, 29, 28)) >> (int4)(31);
                const float4 xv = vload4(8 + g, xs);
                pos4 += as_float4(as_int4(xv) & m);
                w >>= 4;
            }
            const float pos = pos4.x + pos4.y + pos4.z + pos4.w;
            acc += s * (2.0f * pos - xsum[(u << 1) + h]);
        }
    }

    __local float partial[WG_SIZE];
    partial[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s2 = WG_SIZE >> 1; s2 > 0; s2 >>= 1) {
        if (tid < s2) partial[tid] += partial[tid + s2];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) STORE1(partial[0], out, out_off + n);
}

// split-stream embedding gather
__kernel
void q1_row_gather2(__global const uint2* bits,
                    __global const half* scales,
                    __global storage_t* out,
                    const int token,
                    const int K) {
    const int i = (int)get_global_id(0);
    if (i >= K) return;
    const int U = K >> 7;
    const int u = i >> 7, r = i & 127;
    const uint2 bw = bits[((long)token * U + u) * 2 + (r >> 6)];
    const uint w = (r & 63) < 32 ? bw.x : bw.y;
    const int bit = (w >> (r & 31)) & 1;
    const float s = vload_half((long)token * U + u, scales);
    STORE1(bit ? s : -s, out, i);
}

// ── v3: K-major wave-stride GEMV (the moonshine linear_gemv_t shape) ─────
// Upload-time transpose: bits_T [U][N] uint4 (one unit's 128 sign bits per
// (u,n)), scales_T [U][N] half. Thread = one FULL output row; adjacent
// lanes read adjacent rows' words -> fully coalesced weight streaming; x
// comes through L1/L2 (identical slice for all lanes, broadcast-friendly).
// No barriers, no local memory, no reduction tree. Groups = ceil(N/64):
// always < 65535 -> no workgroup-id wrap chunking needed.
// Per-row accumulation order = sequential over units, matching the host
// P1 reference loop.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemv3(__global const uint4* bits_t,   // [U][N]
              __global const half* scales_t,  // [U][N]
              __global const storage_t* x,    // [K]
              __global const float* xsum,     // [K/64] (pairs summed here)
              __global storage_t* out,
              const int N,
              const int K,
              const int out_off) {
    const int n = (int)get_global_id(0);
    if (n >= N) return;
    const int U = K >> 7;
    float acc = 0.0f;
    for (int u = 0; u < U; ++u) {
        const uint4 bw = bits_t[(long)u * N + n];
        const float s = vload_half((long)u * N + n, scales_t);
        const __global storage_t* xs = x + (u << 7);
        float4 pos4 = (float4)(0.0f);
        uint w;
        #define Q3_WORD(WSEL, XOFF)                                        \
            w = WSEL;                                                      \
            _Pragma("unroll")                                              \
            for (int g = 0; g < 8; ++g) {                                  \
                const int4 m = ((int4)(w) << (int4)(31,30,29,28)) >> (int4)(31); \
                const float4 xv = LOAD4(xs, XOFF + g);                     \
                pos4 += as_float4(as_int4(xv) & m);                        \
                w >>= 4;                                                   \
            }
        Q3_WORD(bw.x, 0)
        Q3_WORD(bw.y, 8)
        Q3_WORD(bw.z, 16)
        Q3_WORD(bw.w, 24)
        #undef Q3_WORD
        const float pos = pos4.x + pos4.y + pos4.z + pos4.w;
        acc += s * (2.0f * pos - (xsum[u << 1] + xsum[(u << 1) + 1]));
    }
    STORE1(acc, out, out_off + n);
}

// gather for the K-major transposed layout
__kernel
void q1_row_gather3(__global const uint4* bits_t,   // [U][V]
                    __global const half* scales_t,  // [U][V]
                    __global storage_t* out,
                    const int token,
                    const int V,
                    const int K) {
    const int i = (int)get_global_id(0);
    if (i >= K) return;
    const int u = i >> 7, r = i & 127;
    const uint4 bw = bits_t[(long)u * V + token];
    const uint w = (r < 32) ? bw.x : (r < 64) ? bw.y : (r < 96) ? bw.z : bw.w;
    const int bit = (w >> (r & 31)) & 1;
    const float s = vload_half((long)u * V + token, scales_t);
    STORE1(bit ? s : -s, out, i);
}

// ── v4: quad-row K-major GEMV — x loads amortized over 4 rows ────────────
// v3's instruction profile: 32 float4 x-loads per 18B of weights = x
// dominates issue slots. Each thread now accumulates 4 rows, WAVE-
// INTERLEAVED (thread t owns rows g*256 + t + j*64, j=0..3) so lane loads
// stay contiguous across the wave for every j. x loaded once per unit,
// applied to 4 rows' masks.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemv4(__global const uint4* bits_t,   // [U][N]
              __global const half* scales_t,  // [U][N]
              __global const storage_t* x,
              __global const float* xsum,
              __global storage_t* out,
              const int N,
              const int K,
              const int out_off) {
    const int g = (int)get_group_id(0);
    const int t = (int)get_local_id(0);
    const int r0 = g * (WG_SIZE * 4) + t;      // rows r0 + j*64
    const int U = K >> 7;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int u = 0; u < U; ++u) {
        const long ub = (long)u * N;
        const float xs128 = xsum[u << 1] + xsum[(u << 1) + 1];
        const __global storage_t* xp = x + (u << 7);
        // stage the unit's 128 x values into registers as 8 float4 pairs?
        // Registers: reuse via loop — load once per g-index inside the
        // row-quad loop below by hoisting: process 32 nibble-groups, each
        // loading xv once and applying 4 masks.
        uint4 b0 = (uint4)(0), b1 = (uint4)(0), b2 = (uint4)(0), b3 = (uint4)(0);
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        const int n0 = r0, n1 = r0 + 64, n2 = r0 + 128, n3 = r0 + 192;
        if (n0 < N) { b0 = bits_t[ub + n0]; s0 = vload_half(ub + n0, scales_t); }
        if (n1 < N) { b1 = bits_t[ub + n1]; s1 = vload_half(ub + n1, scales_t); }
        if (n2 < N) { b2 = bits_t[ub + n2]; s2 = vload_half(ub + n2, scales_t); }
        if (n3 < N) { b3 = bits_t[ub + n3]; s3 = vload_half(ub + n3, scales_t); }
        float4 p0 = (float4)(0.0f), p1 = (float4)(0.0f),
               p2 = (float4)(0.0f), p3 = (float4)(0.0f);
        #pragma unroll
        for (int q = 0; q < 4; ++q) {          // uint4 component
            uint w0 = ((const uint*)&b0)[q], w1 = ((const uint*)&b1)[q],
                 w2 = ((const uint*)&b2)[q], w3 = ((const uint*)&b3)[q];
            #pragma unroll
            for (int gg = 0; gg < 8; ++gg) {
                const float4 xv = LOAD4(xp, q * 8 + gg);
                const int4 iv = as_int4(xv);
                p0 += as_float4(iv & (((int4)(w0) << (int4)(31,30,29,28)) >> (int4)(31)));
                p1 += as_float4(iv & (((int4)(w1) << (int4)(31,30,29,28)) >> (int4)(31)));
                p2 += as_float4(iv & (((int4)(w2) << (int4)(31,30,29,28)) >> (int4)(31)));
                p3 += as_float4(iv & (((int4)(w3) << (int4)(31,30,29,28)) >> (int4)(31)));
                w0 >>= 4; w1 >>= 4; w2 >>= 4; w3 >>= 4;
            }
        }
        acc0 += s0 * (2.0f * (p0.x + p0.y + p0.z + p0.w) - xs128);
        acc1 += s1 * (2.0f * (p1.x + p1.y + p1.z + p1.w) - xs128);
        acc2 += s2 * (2.0f * (p2.x + p2.y + p2.z + p2.w) - xs128);
        acc3 += s3 * (2.0f * (p3.x + p3.y + p3.z + p3.w) - xs128);
    }
    if (r0 < N)       STORE1(acc0, out, out_off + r0);
    if (r0 + 64 < N)  STORE1(acc1, out, out_off + r0 + 64);
    if (r0 + 128 < N) STORE1(acc2, out, out_off + r0 + 128);
    if (r0 + 192 < N) STORE1(acc3, out, out_off + r0 + 192);
}

// ── v5: oct-row K-major GEMV — x L2 traffic halved again vs v4 ───────────
// x re-reads dominate (N/rows_per_thread threads × K×4B through 64KB L2).
// 8 rows/thread: registers = 8 uint4 bits + 8 scalar accs — at the edge of
// the Adreno register cliff; A/B'd against v4 before shipping.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemv5(__global const uint4* bits_t,
              __global const half* scales_t,
              __global const storage_t* x,
              __global const float* xsum,
              __global storage_t* out,
              const int N,
              const int K,
              const int out_off) {
    const int g = (int)get_group_id(0);
    const int t = (int)get_local_id(0);
    const int r0 = g * (WG_SIZE * 8) + t;      // rows r0 + j*64, j = 0..7
    const int U = K >> 7;
    float acc[8] = {0,0,0,0,0,0,0,0};

    for (int u = 0; u < U; ++u) {
        const long ub = (long)u * N;
        const float xs128 = xsum[u << 1] + xsum[(u << 1) + 1];
        const __global storage_t* xp = x + (u << 7);
        uint4 bw[8];
        float sc[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int n = r0 + j * 64;
            if (n < N) { bw[j] = bits_t[ub + n]; sc[j] = vload_half(ub + n, scales_t); }
            else       { bw[j] = (uint4)(0);    sc[j] = 0.0f; }
        }
        float4 p[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) p[j] = (float4)(0.0f);
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            #pragma unroll
            for (int gg = 0; gg < 8; ++gg) {
                const float4 xv = LOAD4(xp, q * 8 + gg);
                const int4 iv = as_int4(xv);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint w = ((const uint*)&bw[j])[q] >> (gg * 4);
                    p[j] += as_float4(iv & (((int4)(w) << (int4)(31,30,29,28)) >> (int4)(31)));
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[j] += sc[j] * (2.0f * (p[j].x + p[j].y + p[j].z + p[j].w) - xs128);
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int n = r0 + j * 64;
        if (n < N) STORE1(acc[j], out, out_off + n);
    }
}

// ── v6: v4 quad-row + chunked __local x ──────────────────────────────────
// At WG-per-row (v2) local staging lost: 12288 stages of 16KB. At v4's
// 256-rows-per-WG it wins: gu = 96 WGs x 16KB staged vs ~100MB of L2 x
// re-reads. K chunked at 4096 floats (16KB local); 2 barriers per chunk.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemv6(__global const uint4* bits_t,
              __global const half* scales_t,
              __global const storage_t* x,
              __global const float* xsum,
              __global storage_t* out,
              const int N,
              const int K,
              const int out_off) {
    const int g = (int)get_group_id(0);
    const int t = (int)get_local_id(0);
    const int r0 = g * (WG_SIZE * 4) + t;
    const int U = K >> 7;
    __local float xl[XCHUNK];
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int kb = 0; kb < K; kb += XCHUNK) {
        const int kn = min(XCHUNK, K - kb);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = t * 4; i < kn; i += WG_SIZE * 4) {
            const float4 v = LOAD4(x + kb, i >> 2);
            vstore4(v, i >> 2, xl);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int u0 = kb >> 7, nu = kn >> 7;
        for (int ul = 0; ul < nu; ++ul) {
            const int u = u0 + ul;
            const long ub = (long)u * N;
            const float xs128 = xsum[u << 1] + xsum[(u << 1) + 1];
            const __local float* xp = xl + (ul << 7);
            uint4 b0 = (uint4)(0), b1 = (uint4)(0), b2 = (uint4)(0), b3 = (uint4)(0);
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            const int n0 = r0, n1 = r0 + 64, n2 = r0 + 128, n3 = r0 + 192;
            if (n0 < N) { b0 = bits_t[ub + n0]; s0 = vload_half(ub + n0, scales_t); }
            if (n1 < N) { b1 = bits_t[ub + n1]; s1 = vload_half(ub + n1, scales_t); }
            if (n2 < N) { b2 = bits_t[ub + n2]; s2 = vload_half(ub + n2, scales_t); }
            if (n3 < N) { b3 = bits_t[ub + n3]; s3 = vload_half(ub + n3, scales_t); }
            float4 p0 = (float4)(0.0f), p1 = (float4)(0.0f),
                   p2 = (float4)(0.0f), p3 = (float4)(0.0f);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                uint w0 = ((const uint*)&b0)[q], w1 = ((const uint*)&b1)[q],
                     w2 = ((const uint*)&b2)[q], w3 = ((const uint*)&b3)[q];
                #pragma unroll
                for (int gg = 0; gg < 8; ++gg) {
                    const float4 xv = vload4(q * 8 + gg, xp);
                    const int4 iv = as_int4(xv);
                    p0 += as_float4(iv & (((int4)(w0) << (int4)(31,30,29,28)) >> (int4)(31)));
                    p1 += as_float4(iv & (((int4)(w1) << (int4)(31,30,29,28)) >> (int4)(31)));
                    p2 += as_float4(iv & (((int4)(w2) << (int4)(31,30,29,28)) >> (int4)(31)));
                    p3 += as_float4(iv & (((int4)(w3) << (int4)(31,30,29,28)) >> (int4)(31)));
                    w0 >>= 4; w1 >>= 4; w2 >>= 4; w3 >>= 4;
                }
            }
            acc0 += s0 * (2.0f * (p0.x + p0.y + p0.z + p0.w) - xs128);
            acc1 += s1 * (2.0f * (p1.x + p1.y + p1.z + p1.w) - xs128);
            acc2 += s2 * (2.0f * (p2.x + p2.y + p2.z + p2.w) - xs128);
            acc3 += s3 * (2.0f * (p3.x + p3.y + p3.z + p3.w) - xs128);
        }
    }
    if (r0 < N)       STORE1(acc0, out, out_off + r0);
    if (r0 + 64 < N)  STORE1(acc1, out, out_off + r0 + 64);
    if (r0 + 128 < N) STORE1(acc2, out, out_off + r0 + 128);
    if (r0 + 192 < N) STORE1(acc3, out, out_off + r0 + 192);
}

// ── v7: 8-bit-LUT GEMV for FAT weights (N >= 16384) ──────────────────────
// 1-bit-specific: LUT[seg][byte] = sum of the byte's set-bit x values
// (256 entries per 8-activation segment). Inner loop = 1 byte extract +
// 1 LUT read + 1 add per 8 WEIGHTS (vs ~8 vec-ops). LUT build (16 segs x
// 256 entries, ~4 adds each) is amortized over 2048 rows per WG — the
// break-even needs ~1000+; narrow GEMVs stay on v4.
#define WG7 128
#define ROWS7 16   // rows per thread; WG covers 2048
__kernel
__attribute__((reqd_work_group_size(WG7, 1, 1)))
void q1_gemv7(__global const uint4* bits_t,   // [U][N]
              __global const half* scales_t,  // [U][N]
              __global const storage_t* x,
              __global const float* xsum,
              __global storage_t* out,
              const int N,
              const int K,
              const int out_off) {
    const int g = (int)get_group_id(0);
    const int t = (int)get_local_id(0);
    const int r0 = g * (WG7 * ROWS7) + t;     // rows r0 + j*128
    const int U = K >> 7;
    __local float xu[128];
    __local float lut[16][256];
    float acc[ROWS7];
    #pragma unroll
    for (int j = 0; j < ROWS7; ++j) acc[j] = 0.0f;

    for (int u = 0; u < U; ++u) {
        const long ub = (long)u * N;
        const float xs128 = xsum[u << 1] + xsum[(u << 1) + 1];
        barrier(CLK_LOCAL_MEM_FENCE);
        // stage this unit's 128 x values
        if (t < 32) {
            const float4 v = LOAD4(x + (u << 7), t);
            vstore4(v, t, xu);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // build 16 LUTs: thread handles 32 entries (seg = t>>3, base byte)
        {
            const int seg = t >> 3;           // 0..15
            const int b0 = (t & 7) * 32;      // 32 entries per thread
            const __local float* xs = xu + seg * 8;
            for (int b = b0; b < b0 + 32; ++b) {
                float s = 0.0f;
                if (b & 1)   s += xs[0];
                if (b & 2)   s += xs[1];
                if (b & 4)   s += xs[2];
                if (b & 8)   s += xs[3];
                if (b & 16)  s += xs[4];
                if (b & 32)  s += xs[5];
                if (b & 64)  s += xs[6];
                if (b & 128) s += xs[7];
                lut[seg][b] = s;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // 16 rows per thread, sequential (one uint4 live at a time)
        #pragma unroll 4
        for (int j = 0; j < ROWS7; ++j) {
            const int n = r0 + j * WG7;
            if (n >= N) break;
            const uint4 bw = bits_t[ub + n];
            const float sc = vload_half(ub + n, scales_t);
            float pos = 0.0f;
            uint w;
            w = bw.x;
            pos += lut[0][w & 255]; pos += lut[1][(w >> 8) & 255];
            pos += lut[2][(w >> 16) & 255]; pos += lut[3][w >> 24];
            w = bw.y;
            pos += lut[4][w & 255]; pos += lut[5][(w >> 8) & 255];
            pos += lut[6][(w >> 16) & 255]; pos += lut[7][w >> 24];
            w = bw.z;
            pos += lut[8][w & 255]; pos += lut[9][(w >> 8) & 255];
            pos += lut[10][(w >> 16) & 255]; pos += lut[11][w >> 24];
            w = bw.w;
            pos += lut[12][w & 255]; pos += lut[13][(w >> 8) & 255];
            pos += lut[14][(w >> 16) & 255]; pos += lut[15][w >> 24];
            acc[j] += sc * (2.0f * pos - xs128);
        }
    }
    #pragma unroll
    for (int j = 0; j < ROWS7; ++j) {
        const int n = r0 + j * WG7;
        if (n < N) STORE1(acc[j], out, out_off + n);
    }
}

// gather with the token id read from a device buffer (the argmax output) —
// removes the per-token blocking readback; host reads tokens 1 step behind.
__kernel
void q1_row_gather3_dev(__global const uint4* bits_t,
                        __global const half* scales_t,
                        __global storage_t* out,
                        __global const int* tok_buf,
                        const int V,
                        const int K) {
    const int i = (int)get_global_id(0);
    if (i >= K) return;
    const int token = tok_buf[0];
    const int u = i >> 7, r = i & 127;
    const uint4 bw = bits_t[(long)u * V + token];
    const uint w = (r < 32) ? bw.x : (r < 64) ? bw.y : (r < 96) ? bw.z : bw.w;
    const int bit = (w >> (r & 31)) & 1;
    const float s = vload_half((long)u * V + token, scales_t);
    STORE1(bit ? s : -s, out, i);
}

// ── batched-M GEMV for prefill (weights streamed ONCE per M-token chunk) ─
// Slice-aware: reads rows [row_off, row_off+N) of a FUSED tensor with total
// width Ntot (so prefill reuses the fused QKV / gate+up uploads without a
// second copy). Per-row unit math is byte-identical to v4's, so prefill
// numerics (and the KV entries they produce) keep the token-exact gate.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemv_b(__global const uint4* bits_t,   // [U][Ntot]
               __global const half* scales_t,  // [U][Ntot]
               __global const storage_t* x,    // [M][K]
               __global const float* xsum,     // [M][K/64]
               __global storage_t* out,        // [M][N]
               const int N,
               const int K,
               const int Ntot,
               const int row_off,
               const int M) {
#ifndef MB
#define MB 8
#endif
    const int n = (int)get_global_id(0);
    if (n >= N) return;
    const int U = K >> 7;
    const int xs_stride = K >> 6;
    float acc[MB];
    #pragma unroll
    for (int m = 0; m < MB; ++m) acc[m] = 0.0f;

    for (int u = 0; u < U; ++u) {
        const uint4 bw = bits_t[(long)u * Ntot + row_off + n];
        const float s = vload_half((long)u * Ntot + row_off + n, scales_t);
        #pragma unroll
        for (int m = 0; m < MB; ++m) {
            if (m >= M) break;
            const __global storage_t* xp = x + (long)m * K + (u << 7);
            const float xs128 = xsum[m * xs_stride + (u << 1)] +
                                xsum[m * xs_stride + (u << 1) + 1];
            float4 p = (float4)(0.0f);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                uint w = ((const uint*)&bw)[q];
                #pragma unroll
                for (int gg = 0; gg < 8; ++gg) {
                    const float4 xv = LOAD4(xp, q * 8 + gg);
                    p += as_float4(as_int4(xv) &
                                   (((int4)(w) << (int4)(31,30,29,28)) >> (int4)(31)));
                    w >>= 4;
                }
            }
            acc[m] += s * (2.0f * (p.x + p.y + p.z + p.w) - xs128);
        }
    }
    #pragma unroll
    for (int m = 0; m < MB; ++m)
        if (m < M) STORE1(acc[m], out, (long)m * N + n);
}

// batched xsum: one half-unit sum per (i, m)
__kernel
void q1_xsum_b(__global const storage_t* x,   // [M][K]
               __global float* xsum,          // [M][K/64]
               const int K,
               const int M) {
    const int i = (int)get_global_id(0);
    const int m = (int)get_global_id(1);
    if (i >= K / 64 || m >= M) return;
    const __global storage_t* xs = x + (long)m * K + i * 64;
    float acc = 0.0f;
    for (int j = 0; j < 16; ++j) {
        const float4 v = LOAD4(xs, j);
        acc += v.x + v.y + v.z + v.w;
    }
    xsum[m * (K / 64) + i] = acc;
}

// batched embedding gather: M tokens -> x[M][K]
__kernel
void q1_row_gather_b(__global const uint4* bits_t,
                     __global const half* scales_t,
                     __global storage_t* out,       // [M][K]
                     __global const int* tokens,    // [M]
                     const int V,
                     const int K,
                     const int M) {
    const int i = (int)get_global_id(0);
    const int m = (int)get_global_id(1);
    if (i >= K || m >= M) return;
    const int token = tokens[m];
    const int u = i >> 7, r = i & 127;
    const uint4 bw = bits_t[(long)u * V + token];
    const uint w = (r < 32) ? bw.x : (r < 64) ? bw.y : (r < 96) ? bw.z : bw.w;
    const int bit = (w >> (r & 31)) & 1;
    const float s = vload_half((long)u * V + token, scales_t);
    STORE1(bit ? s : -s, out, (long)m * K + i);
}

// ── PROBES (QC guide bound test) ── clean, macro-based mask ──────────────
#define MASK4(w,gg) (((int4)((w)>>((gg)*4)) << (int4)(31,30,29,28)) >> (int4)(31,31,31,31))
__kernel __attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void q1_gemv4_2c(__global const uint4* bits_t,__global const half* scales_t,
                 __global const storage_t* x,__global const float* xsum,
                 __global storage_t* out,const int N,const int K,const int out_off){
    const int g=get_group_id(0),t=get_local_id(0),r0=g*(WG_SIZE*4)+t,U=K>>7;
    float a0=0,a1=0,a2=0,a3=0,sink=0;
    for(int u=0;u<U;++u){const long ub=(long)u*N;const float xs=xsum[u<<1]+xsum[(u<<1)+1];
        const __global storage_t* xp=x+(u<<7);
        uint4 b0=(uint4)(0),b1=(uint4)(0),b2=(uint4)(0),b3=(uint4)(0);float s0=0,s1=0,s2=0,s3=0;
        int n0=r0,n1=r0+64,n2=r0+128,n3=r0+192;
        if(n0<N){b0=bits_t[ub+n0];s0=vload_half(ub+n0,scales_t);}
        if(n1<N){b1=bits_t[ub+n1];s1=vload_half(ub+n1,scales_t);}
        if(n2<N){b2=bits_t[ub+n2];s2=vload_half(ub+n2,scales_t);}
        if(n3<N){b3=bits_t[ub+n3];s3=vload_half(ub+n3,scales_t);}
        float4 p0=(float4)(0),p1=(float4)(0),p2=(float4)(0),p3=(float4)(0),qq=(float4)(0);
        #pragma unroll
        for(int q=0;q<4;++q){uint w0=((uint*)&b0)[q],w1=((uint*)&b1)[q],w2=((uint*)&b2)[q],w3=((uint*)&b3)[q];
          #pragma unroll
          for(int gg=0;gg<8;++gg){float4 xv=LOAD4(xp,q*8+gg);int4 iv=as_int4(xv);
            p0+=as_float4(iv&MASK4(w0,gg));p1+=as_float4(iv&MASK4(w1,gg));
            p2+=as_float4(iv&MASK4(w2,gg));p3+=as_float4(iv&MASK4(w3,gg));
            qq+=as_float4(iv&MASK4(w0,gg));}}
        a0+=s0*(2*(p0.x+p0.y+p0.z+p0.w)-xs);a1+=s1*(2*(p1.x+p1.y+p1.z+p1.w)-xs);
        a2+=s2*(2*(p2.x+p2.y+p2.z+p2.w)-xs);a3+=s3*(2*(p3.x+p3.y+p3.z+p3.w)-xs);sink+=qq.x;}
    if(sink==1e30f)a0+=sink;
    if(r0<N)STORE1(a0,out,out_off+r0);if(r0+64<N)STORE1(a1,out,out_off+r0+64);
    if(r0+128<N)STORE1(a2,out,out_off+r0+128);if(r0+192<N)STORE1(a3,out,out_off+r0+192);}
__kernel __attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void q1_gemv4_2m(__global const uint4* bits_t,__global const half* scales_t,
                 __global const storage_t* x,__global const float* xsum,
                 __global storage_t* out,const int N,const int K,const int out_off){
    const int g=get_group_id(0),t=get_local_id(0),r0=g*(WG_SIZE*4)+t,U=K>>7;
    float a0=0,a1=0,a2=0,a3=0;uint sink=0;
    for(int u=0;u<U;++u){const long ub=(long)u*N;const float xs=xsum[u<<1]+xsum[(u<<1)+1];
        const __global storage_t* xp=x+(u<<7);
        uint4 b0=(uint4)(0),b1=(uint4)(0),b2=(uint4)(0),b3=(uint4)(0);float s0=0,s1=0,s2=0,s3=0;
        int n0=r0,n1=r0+64,n2=r0+128,n3=r0+192;
        if(n0<N){b0=bits_t[ub+n0];sink^=bits_t[ub+n0].x;s0=vload_half(ub+n0,scales_t);}
        if(n1<N){b1=bits_t[ub+n1];sink^=bits_t[ub+n1].x;s1=vload_half(ub+n1,scales_t);}
        if(n2<N){b2=bits_t[ub+n2];sink^=bits_t[ub+n2].x;s2=vload_half(ub+n2,scales_t);}
        if(n3<N){b3=bits_t[ub+n3];sink^=bits_t[ub+n3].x;s3=vload_half(ub+n3,scales_t);}
        float4 p0=(float4)(0),p1=(float4)(0),p2=(float4)(0),p3=(float4)(0);
        #pragma unroll
        for(int q=0;q<4;++q){uint w0=((uint*)&b0)[q],w1=((uint*)&b1)[q],w2=((uint*)&b2)[q],w3=((uint*)&b3)[q];
          #pragma unroll
          for(int gg=0;gg<8;++gg){float4 xv=LOAD4(xp,q*8+gg);int4 iv=as_int4(xv);
            p0+=as_float4(iv&MASK4(w0,gg));p1+=as_float4(iv&MASK4(w1,gg));
            p2+=as_float4(iv&MASK4(w2,gg));p3+=as_float4(iv&MASK4(w3,gg));}}
        a0+=s0*(2*(p0.x+p0.y+p0.z+p0.w)-xs);a1+=s1*(2*(p1.x+p1.y+p1.z+p1.w)-xs);
        a2+=s2*(2*(p2.x+p2.y+p2.z+p2.w)-xs);a3+=s3*(2*(p3.x+p3.y+p3.z+p3.w)-xs);}
    if(sink==0xdeadbeef)a0+=1;
    if(r0<N)STORE1(a0,out,out_off+r0);if(r0+64<N)STORE1(a1,out,out_off+r0+64);
    if(r0+128<N)STORE1(a2,out,out_off+r0+128);if(r0+192<N)STORE1(a3,out,out_off+r0+192);}

// occupancy probes: v4 with fewer rows/thread => more threads => more waves
#define GEN_V4R(NAME, RPT, RSHIFT) \
__kernel __attribute__((reqd_work_group_size(WG_SIZE,1,1))) \
void NAME(__global const uint4* bits_t,__global const half* scales_t, \
          __global const storage_t* x,__global const float* xsum, \
          __global storage_t* out,const int N,const int K,const int out_off){ \
    const int g=get_group_id(0),t=get_local_id(0),r0=g*(WG_SIZE*RPT)+t,U=K>>7; \
    float acc[RPT]; for(int j=0;j<RPT;++j)acc[j]=0.0f; \
    for(int u=0;u<U;++u){const long ub=(long)u*N;const float xs=xsum[u<<1]+xsum[(u<<1)+1]; \
        const __global storage_t* xp=x+(u<<7); \
        for(int j=0;j<RPT;++j){const int n=r0+j*64; if(n>=N)continue; \
            const uint4 bw=bits_t[ub+n];const float s=vload_half(ub+n,scales_t); \
            float4 p=(float4)(0.0f); \
            for(int q=0;q<4;++q){uint w=((const uint*)&bw)[q]; \
                for(int gg=0;gg<8;++gg){const float4 xv=LOAD4(xp,q*8+gg); \
                    p+=as_float4(as_int4(xv)&(((int4)(w)<<(int4)(31,30,29,28))>>(int4)(31,31,31,31)));w>>=4;}} \
            acc[j]+=s*(2.0f*(p.x+p.y+p.z+p.w)-xs);}} \
    for(int j=0;j<RPT;++j){const int n=r0+j*64;if(n<N)STORE1(acc[j],out,out_off+n);}}
GEN_V4R(q1_gemv_r1, 1, 0)
GEN_V4R(q1_gemv_r2, 2, 0)

// ── v9: v4's EXACT row-0 path, thread-per-row (occupancy fix, LATENCY bound)
// Byte-identical math to q1_gemv4's acc0 — cloned line-for-line, just one
// row per thread instead of four. The bound test proved thread-per-row is
// ~2x faster (more waves hide memory latency). Dispatch: ceil(N/64)*64, lws 64.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemv9(__global const uint4* bits_t,
              __global const half* scales_t,
              __global const storage_t* x,
              __global const float* xsum,
              __global storage_t* out,
              const int N, const int K, const int out_off) {
    const int n = (int)get_group_id(0) * WG_SIZE + (int)get_local_id(0);
    const int U = K >> 7;
    float acc0 = 0.0f;
    for (int u = 0; u < U; ++u) {
        const long ub = (long)u * N;
        const float xs128 = xsum[u << 1] + xsum[(u << 1) + 1];
        const __global storage_t* xp = x + (u << 7);
        uint4 b0 = (uint4)(0);
        float s0 = 0.f;
        if (n < N) { b0 = bits_t[ub + n]; s0 = vload_half(ub + n, scales_t); }
        float4 p0 = (float4)(0.0f);
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint w0 = ((const uint*)&b0)[q];
            #pragma unroll
            for (int gg = 0; gg < 8; ++gg) {
                const float4 xv = LOAD4(xp, q * 8 + gg);
                const int4 iv = as_int4(xv);
                p0 += as_float4(iv & (((int4)(w0) << (int4)(31,30,29,28)) >> (int4)(31)));
                w0 >>= 4;
            }
        }
        acc0 += s0 * (2.0f * (p0.x + p0.y + p0.z + p0.w) - xs128);
    }
    if (n < N) STORE1(acc0, out, out_off + n);
}

// ── v4img: v4 quad-row with WEIGHTS via the L1 TEXTURE path ───────────────
// The bound test showed we're latency-bound on weight loads. This reads the
// 1-bit weight units through image1d_buffer (L1 texture cache) instead of
// the L2 buffer path — QC guide §6.2/§7.1.5.4's top recommendation, and the
// one lever that hides load latency without touching activation traffic.
// Each texel = uint4 = one 128-bit unit. Zero-copy image over the same
// bytes. Math byte-identical to q1_gemv4.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemv4_img(__read_only image1d_buffer_t wimg,   // uint4 units [U*N]
                  __global const half* scales_t,
                  __global const storage_t* x,
                  __global const float* xsum,
                  __global storage_t* out,
                  const int N, const int K, const int out_off) {
    const int g = (int)get_group_id(0);
    const int t = (int)get_local_id(0);
    const int r0 = g * (WG_SIZE * 4) + t;
    const int U = K >> 7;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int u = 0; u < U; ++u) {
        const long ub = (long)u * N;
        const float xs128 = xsum[u << 1] + xsum[(u << 1) + 1];
        const __global storage_t* xp = x + (u << 7);
        uint4 b0 = (uint4)(0), b1 = (uint4)(0), b2 = (uint4)(0), b3 = (uint4)(0);
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        const int n0 = r0, n1 = r0 + 64, n2 = r0 + 128, n3 = r0 + 192;
        if (n0 < N) { b0 = read_imageui(wimg, (int)(ub + n0)); s0 = vload_half(ub + n0, scales_t); }
        if (n1 < N) { b1 = read_imageui(wimg, (int)(ub + n1)); s1 = vload_half(ub + n1, scales_t); }
        if (n2 < N) { b2 = read_imageui(wimg, (int)(ub + n2)); s2 = vload_half(ub + n2, scales_t); }
        if (n3 < N) { b3 = read_imageui(wimg, (int)(ub + n3)); s3 = vload_half(ub + n3, scales_t); }
        float4 p0 = (float4)(0.0f), p1 = (float4)(0.0f),
               p2 = (float4)(0.0f), p3 = (float4)(0.0f);
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint w0 = ((const uint*)&b0)[q], w1 = ((const uint*)&b1)[q],
                 w2 = ((const uint*)&b2)[q], w3 = ((const uint*)&b3)[q];
            #pragma unroll
            for (int gg = 0; gg < 8; ++gg) {
                const float4 xv = LOAD4(xp, q * 8 + gg);
                const int4 iv = as_int4(xv);
                p0 += as_float4(iv & (((int4)(w0) << (int4)(31,30,29,28)) >> (int4)(31)));
                p1 += as_float4(iv & (((int4)(w1) << (int4)(31,30,29,28)) >> (int4)(31)));
                p2 += as_float4(iv & (((int4)(w2) << (int4)(31,30,29,28)) >> (int4)(31)));
                p3 += as_float4(iv & (((int4)(w3) << (int4)(31,30,29,28)) >> (int4)(31)));
                w0 >>= 4; w1 >>= 4; w2 >>= 4; w3 >>= 4;
            }
        }
        acc0 += s0 * (2.0f * (p0.x + p0.y + p0.z + p0.w) - xs128);
        acc1 += s1 * (2.0f * (p1.x + p1.y + p1.z + p1.w) - xs128);
        acc2 += s2 * (2.0f * (p2.x + p2.y + p2.z + p2.w) - xs128);
        acc3 += s3 * (2.0f * (p3.x + p3.y + p3.z + p3.w) - xs128);
    }
    if (r0 < N)       STORE1(acc0, out, out_off + r0);
    if (r0 + 64 < N)  STORE1(acc1, out, out_off + r0 + 64);
    if (r0 + 128 < N) STORE1(acc2, out, out_off + r0 + 128);
    if (r0 + 192 < N) STORE1(acc3, out, out_off + r0 + 192);
}

// ── verification-batch GEMV: MV tokens, compile-time unrolled, no spill ──
// De-risks speculative decoding: if this costs ~1x a single-token GEMV
// (weight loaded once, applied to MV activation vectors, extra compute
// hidden in the latency-bound weight-load shadow), spec-decode wins.
// MV set via -DMV=N. Quad-row (x-reuse) preserved per token.
#ifndef MV
#define MV 4
#endif
__kernel __attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void q1_gemv_ver(__global const uint4* bits_t,__global const half* scales_t,
                 __global const storage_t* x,   // [MV][K]
                 __global const float* xsum,     // [MV][K/64]
                 __global storage_t* out,        // [MV][N]
                 const int N,const int K,const int out_off){
    const int g=get_group_id(0),t=get_local_id(0),r0=g*(WG_SIZE*4)+t,U=K>>7;
    const int xstride=K, xss=K>>6;
    float acc[MV][4];
    #pragma unroll
    for(int m=0;m<MV;++m){acc[m][0]=0;acc[m][1]=0;acc[m][2]=0;acc[m][3]=0;}
    for(int u=0;u<U;++u){const long ub=(long)u*N;
        uint4 b0=(uint4)(0),b1=(uint4)(0),b2=(uint4)(0),b3=(uint4)(0);float s0=0,s1=0,s2=0,s3=0;
        int n0=r0,n1=r0+64,n2=r0+128,n3=r0+192;
        if(n0<N){b0=bits_t[ub+n0];s0=vload_half(ub+n0,scales_t);}
        if(n1<N){b1=bits_t[ub+n1];s1=vload_half(ub+n1,scales_t);}
        if(n2<N){b2=bits_t[ub+n2];s2=vload_half(ub+n2,scales_t);}
        if(n3<N){b3=bits_t[ub+n3];s3=vload_half(ub+n3,scales_t);}
        #pragma unroll
        for(int m=0;m<MV;++m){
            const __global storage_t* xp=x+(long)m*xstride+(u<<7);
            const float xs=xsum[m*xss+(u<<1)]+xsum[m*xss+(u<<1)+1];
            float4 p0=(float4)(0),p1=(float4)(0),p2=(float4)(0),p3=(float4)(0);
            #pragma unroll
            for(int q=0;q<4;++q){uint w0=((uint*)&b0)[q],w1=((uint*)&b1)[q],w2=((uint*)&b2)[q],w3=((uint*)&b3)[q];
              #pragma unroll
              for(int gg=0;gg<8;++gg){float4 xv=LOAD4(xp,q*8+gg);int4 iv=as_int4(xv);
                p0+=as_float4(iv&(((int4)(w0)<<(int4)(31,30,29,28))>>(int4)(31)));
                p1+=as_float4(iv&(((int4)(w1)<<(int4)(31,30,29,28))>>(int4)(31)));
                p2+=as_float4(iv&(((int4)(w2)<<(int4)(31,30,29,28))>>(int4)(31)));
                p3+=as_float4(iv&(((int4)(w3)<<(int4)(31,30,29,28))>>(int4)(31)));
                w0>>=4;w1>>=4;w2>>=4;w3>>=4;}}
            acc[m][0]+=s0*(2.0f*(p0.x+p0.y+p0.z+p0.w)-xs);
            acc[m][1]+=s1*(2.0f*(p1.x+p1.y+p1.z+p1.w)-xs);
            acc[m][2]+=s2*(2.0f*(p2.x+p2.y+p2.z+p2.w)-xs);
            acc[m][3]+=s3*(2.0f*(p3.x+p3.y+p3.z+p3.w)-xs);}}
    #pragma unroll
    for(int m=0;m<MV;++m){const long ob=(long)m*N+out_off;
        if(r0<N)STORE1(acc[m][0],out,ob+r0);if(r0+64<N)STORE1(acc[m][1],out,ob+r0+64);
        if(r0+128<N)STORE1(acc[m][2],out,ob+r0+128);if(r0+192<N)STORE1(acc[m][3],out,ob+r0+192);}}

// ── v4xor: v4 quad-row WITHOUT the separate xsum kernel ───────────────────
// The two-sum identity (2*pos - xsum) needs a pre-pass computing sum(x) per
// block (a whole extra kernel, ~180 launches/token). This folds the sign
// directly: dot = sum_i (bit_i ? x_i : -x_i), via XOR of x's sign bit with
// (~bit << 31). No xsum kernel, no xsum buffer read. NOTE: different fp
// summation order than two-sum → not bit-exact (may flip near-ties), but
// removes real GPU work. Speed/coherence A/B.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemv4xor(__global const uint4* bits_t,
                 __global const half* scales_t,
                 __global const storage_t* x,
                 __global const float* xsum_unused,
                 __global storage_t* out,
                 const int N, const int K, const int out_off) {
    const int g = (int)get_group_id(0);
    const int t = (int)get_local_id(0);
    const int r0 = g * (WG_SIZE * 4) + t;
    const int U = K >> 7;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int u = 0; u < U; ++u) {
        const long ub = (long)u * N;
        const __global storage_t* xp = x + (u << 7);
        uint4 b0 = (uint4)(0), b1 = (uint4)(0), b2 = (uint4)(0), b3 = (uint4)(0);
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        const int n0 = r0, n1 = r0 + 64, n2 = r0 + 128, n3 = r0 + 192;
        if (n0 < N) { b0 = bits_t[ub + n0]; s0 = vload_half(ub + n0, scales_t); }
        if (n1 < N) { b1 = bits_t[ub + n1]; s1 = vload_half(ub + n1, scales_t); }
        if (n2 < N) { b2 = bits_t[ub + n2]; s2 = vload_half(ub + n2, scales_t); }
        if (n3 < N) { b3 = bits_t[ub + n3]; s3 = vload_half(ub + n3, scales_t); }
        float4 p0 = (float4)(0.0f), p1 = (float4)(0.0f),
               p2 = (float4)(0.0f), p3 = (float4)(0.0f);
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint w0 = ((const uint*)&b0)[q], w1 = ((const uint*)&b1)[q],
                 w2 = ((const uint*)&b2)[q], w3 = ((const uint*)&b3)[q];
            #pragma unroll
            for (int gg = 0; gg < 8; ++gg) {
                const int4 iv = as_int4(LOAD4(xp, q * 8 + gg));
                // bit=1 -> +x; bit=0 -> -x via XOR of the sign bit.
                // ((~w)<<(31-lane)) & 0x80000000 = sign-flip mask per lane.
                p0 += as_float4(iv ^ (((~(int4)(w0)) << (int4)(31,30,29,28)) & (int4)(0x80000000)));
                p1 += as_float4(iv ^ (((~(int4)(w1)) << (int4)(31,30,29,28)) & (int4)(0x80000000)));
                p2 += as_float4(iv ^ (((~(int4)(w2)) << (int4)(31,30,29,28)) & (int4)(0x80000000)));
                p3 += as_float4(iv ^ (((~(int4)(w3)) << (int4)(31,30,29,28)) & (int4)(0x80000000)));
                w0 >>= 4; w1 >>= 4; w2 >>= 4; w3 >>= 4;
            }
        }
        acc0 += s0 * (p0.x + p0.y + p0.z + p0.w);
        acc1 += s1 * (p1.x + p1.y + p1.z + p1.w);
        acc2 += s2 * (p2.x + p2.y + p2.z + p2.w);
        acc3 += s3 * (p3.x + p3.y + p3.z + p3.w);
    }
    if (r0 < N)       STORE1(acc0, out, out_off + r0);
    if (r0 + 64 < N)  STORE1(acc1, out, out_off + r0 + 64);
    if (r0 + 128 < N) STORE1(acc2, out, out_off + r0 + 128);
    if (r0 + 192 < N) STORE1(acc3, out, out_off + r0 + 192);
}

// ── vLX: thread-per-row + LOCAL-staged activations (latency-bound fix) ────
// Bound test: latency-bound (compute & bandwidth both free). Thread-per-row
// maximizes waves (min registers → best latency hiding) but v9 lost by
// reloading x from L2 per unit. Here x is staged ONCE into local memory per
// K-chunk, so every thread reads activations from on-chip local (fast) with
// zero repeated DRAM/L2 traffic — occupancy AND cheap x. Weights: each row's
// units streamed once (coalesced across the wave, K-major layout).
#define LXWG 64
#define LXCHUNK 4096         // x floats per local chunk (16KB fp32) — fits 32KB
__kernel __attribute__((reqd_work_group_size(LXWG, 1, 1)))
void q1_gemv_lx1(__global const uint4* bits_t,
                 __global const half* scales_t,
                 __global const storage_t* x,
                 __global const float* xsum,
                 __global storage_t* out,
                 const int N, const int K, const int out_off) {
    const int tid = get_local_id(0);
    const int n = get_group_id(0) * LXWG + tid;
    const int U = K >> 7;
    __local float xl[LXCHUNK];
    float acc = 0.0f;

    for (int kb = 0; kb < K; kb += LXCHUNK) {
        const int kn = min(LXCHUNK, K - kb);
        barrier(CLK_LOCAL_MEM_FENCE);
        // cooperative load of this x-chunk into local (vec4, coalesced)
        for (int i = tid * 4; i < kn; i += LXWG * 4) {
            const float4 v = LOAD4(x + kb, i >> 2);
            vstore4(v, i >> 2, xl);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (n < N) {
            const int u0 = kb >> 7, nu = kn >> 7;
            for (int uu = 0; uu < nu; ++uu) {
                const int u = u0 + uu;
                const uint4 bw = bits_t[(long)u * N + n];
                const float s = vload_half((long)u * N + n, scales_t);
                const __local float* xp = xl + (uu << 7);
                float4 p = (float4)(0.0f);
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    uint w = ((const uint*)&bw)[q];
                    #pragma unroll
                    for (int gg = 0; gg < 8; ++gg) {
                        const float4 xv = vload4(q * 8 + gg, xp);
                        p += as_float4(as_int4(xv) &
                             (((int4)(w) << (int4)(31,30,29,28)) >> (int4)(31,31,31,31)));
                        w >>= 4;
                    }
                }
                acc += s * (2.0f * (p.x + p.y + p.z + p.w) -
                            (xsum[u << 1] + xsum[(u << 1) + 1]));
            }
        }
    }
    if (n < N) STORE1(acc, out, out_off + n);
}

// ── vPF: v4 quad-row + SOFTWARE PREFETCH (latency-bound fix) ──────────────
// Bound test: latency-bound — the GPU stalls waiting for each weight load
// because compute can't fill the gap. Fix: issue unit u+1's loads BEFORE
// computing unit u, so the load latency overlaps with u's compute (classic
// software pipeline). Doubles weight registers (current+next) but keeps the
// proven quad-row x-reuse. Math identical to q1_gemv4.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemv_pf(__global const uint4* bits_t, __global const half* scales_t,
                __global const storage_t* x, __global const float* xsum,
                __global storage_t* out, const int N, const int K, const int out_off) {
    const int g = get_group_id(0), t = get_local_id(0), r0 = g*(WG_SIZE*4)+t, U = K>>7;
    const int n0=r0, n1=r0+64, n2=r0+128, n3=r0+192;
    float acc0=0, acc1=0, acc2=0, acc3=0;
    // prime: load unit 0
    uint4 b0=(uint4)0,b1=(uint4)0,b2=(uint4)0,b3=(uint4)0; float s0=0,s1=0,s2=0,s3=0;
    if(n0<N){b0=bits_t[n0];s0=vload_half(n0,scales_t);}
    if(n1<N){b1=bits_t[n1];s1=vload_half(n1,scales_t);}
    if(n2<N){b2=bits_t[n2];s2=vload_half(n2,scales_t);}
    if(n3<N){b3=bits_t[n3];s3=vload_half(n3,scales_t);}
    for(int u=0;u<U;++u){
        // prefetch u+1 (issued now, overlaps with the compute below)
        uint4 nb0=(uint4)0,nb1=(uint4)0,nb2=(uint4)0,nb3=(uint4)0; float ns0=0,ns1=0,ns2=0,ns3=0;
        if(u+1<U){const long vb=(long)(u+1)*N;
            if(n0<N){nb0=bits_t[vb+n0];ns0=vload_half(vb+n0,scales_t);}
            if(n1<N){nb1=bits_t[vb+n1];ns1=vload_half(vb+n1,scales_t);}
            if(n2<N){nb2=bits_t[vb+n2];ns2=vload_half(vb+n2,scales_t);}
            if(n3<N){nb3=bits_t[vb+n3];ns3=vload_half(vb+n3,scales_t);}}
        const float xs=xsum[u<<1]+xsum[(u<<1)+1];
        const __global storage_t* xp=x+(u<<7);
        float4 p0=(float4)0,p1=(float4)0,p2=(float4)0,p3=(float4)0;
        #pragma unroll
        for(int q=0;q<4;++q){uint w0=((uint*)&b0)[q],w1=((uint*)&b1)[q],w2=((uint*)&b2)[q],w3=((uint*)&b3)[q];
          #pragma unroll
          for(int gg=0;gg<8;++gg){const float4 xv=LOAD4(xp,q*8+gg);const int4 iv=as_int4(xv);
            p0+=as_float4(iv&(((int4)(w0)<<(int4)(31,30,29,28))>>(int4)(31)));
            p1+=as_float4(iv&(((int4)(w1)<<(int4)(31,30,29,28))>>(int4)(31)));
            p2+=as_float4(iv&(((int4)(w2)<<(int4)(31,30,29,28))>>(int4)(31)));
            p3+=as_float4(iv&(((int4)(w3)<<(int4)(31,30,29,28))>>(int4)(31)));
            w0>>=4;w1>>=4;w2>>=4;w3>>=4;}}
        acc0+=s0*(2.0f*(p0.x+p0.y+p0.z+p0.w)-xs);acc1+=s1*(2.0f*(p1.x+p1.y+p1.z+p1.w)-xs);
        acc2+=s2*(2.0f*(p2.x+p2.y+p2.z+p2.w)-xs);acc3+=s3*(2.0f*(p3.x+p3.y+p3.z+p3.w)-xs);
        b0=nb0;b1=nb1;b2=nb2;b3=nb3;s0=ns0;s1=ns1;s2=ns2;s3=ns3;
    }
    if(n0<N)STORE1(acc0,out,out_off+n0);if(n1<N)STORE1(acc1,out,out_off+n1);
    if(n2<N)STORE1(acc2,out,out_off+n2);if(n3<N)STORE1(acc3,out,out_off+n3);
}

// pure memory-floor probes (no mask compute) — reveals if we're bandwidth-bound
__kernel __attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void q1_read_model(__global const uint4* bits_t,__global const half* scales_t,
                   __global const storage_t* x,__global const float* xsum,
                   __global storage_t* out,const int N,const int K,const int out_off){
    const int g=get_group_id(0),t=get_local_id(0),r0=g*(WG_SIZE*4)+t,U=K>>7;
    uint4 acc=(uint4)(0);
    for(int u=0;u<U;++u){const long ub=(long)u*N;
        if(r0<N)acc^=bits_t[ub+r0]; if(r0+64<N)acc^=bits_t[ub+r0+64];
        if(r0+128<N)acc^=bits_t[ub+r0+128]; if(r0+192<N)acc^=bits_t[ub+r0+192];}
    if(acc.x==0xdeadbeef&&r0<N)STORE1((float)acc.y,out,out_off+r0);
}
__kernel void q1_read_seq(__global const uint4* bits_t,__global storage_t* out,const int total){
    const int i=get_global_id(0); const int gsz=get_global_size(0);
    uint4 acc=(uint4)(0);
    for(int j=i;j<total;j+=gsz) acc^=bits_t[j];
    if(acc.x==0xdeadbeef)STORE1((float)acc.y,out,i);
}

// duo: 2 rows/thread — middle ground between v4 quad (x-reuse) and v9 (max waves)
__kernel __attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void q1_gemv_duo(__global const uint4* bits_t,__global const half* scales_t,
                 __global const storage_t* x,__global const float* xsum,
                 __global storage_t* out,const int N,const int K,const int out_off){
    const int g=get_group_id(0),t=get_local_id(0),r0=g*(WG_SIZE*2)+t,U=K>>7;
    const int n0=r0,n1=r0+64; float acc0=0,acc1=0;
    for(int u=0;u<U;++u){const long ub=(long)u*N;
        uint4 b0=(uint4)0,b1=(uint4)0; float s0=0,s1=0;
        if(n0<N){b0=bits_t[ub+n0];s0=vload_half(ub+n0,scales_t);}
        if(n1<N){b1=bits_t[ub+n1];s1=vload_half(ub+n1,scales_t);}
        const float xs=xsum[u<<1]+xsum[(u<<1)+1]; const __global storage_t* xp=x+(u<<7);
        float4 p0=(float4)0,p1=(float4)0;
        #pragma unroll
        for(int qq=0;qq<4;++qq){uint w0=((uint*)&b0)[qq],w1=((uint*)&b1)[qq];
          #pragma unroll
          for(int gg=0;gg<8;++gg){const float4 xv=LOAD4(xp,qq*8+gg);const int4 iv=as_int4(xv);
            p0+=as_float4(iv&(((int4)(w0)<<(int4)(31,30,29,28))>>(int4)(31)));
            p1+=as_float4(iv&(((int4)(w1)<<(int4)(31,30,29,28))>>(int4)(31)));
            w0>>=4;w1>>=4;}}
        acc0+=s0*(2.0f*(p0.x+p0.y+p0.z+p0.w)-xs);acc1+=s1*(2.0f*(p1.x+p1.y+p1.z+p1.w)-xs);}
    if(n0<N)STORE1(acc0,out,out_off+n0);if(n1<N)STORE1(acc1,out,out_off+n1);
}

// ── GLOBAL-LUT GEMV: fewer ALU ops/weight, LUT in L2 not local (v7's death) ──
// Two-sum needs pos = Σ_{bit=1} x. Split the 128-weight unit into 16 bytes of
// 8 x-values. Precompute, per (unit,byte-segment), all 256 subset-sums ONCE per
// token (build_xlut). Then GEMV replaces 128 mask-adds/row with 16 L2 lookups.
// LUT per unit = 16 segs × 256 × 4B = 16KB → L2-resident, reused across all N
// rows of that unit. No local mem → no bank conflicts (what killed v7).
__kernel void build_xlut(__global const storage_t* x, __global float* xlut, const int K){
    const int gid=get_global_id(0);            // one thread per (u,seg,byteval)
    const int U=K>>7; if(gid >= U*16*256) return;
    const int b   = gid & 255;
    const int seg = (gid>>8) & 15;
    const int u   = gid>>12;
    const int base= u*128 + seg*8;
    float s=0.0f;
    #pragma unroll
    for(int i=0;i<8;++i) if(b&(1<<i)) s += (float)x[base+i];
    xlut[gid]=s;
}
// quad-row GEMV reading the precomputed L2 LUT
__kernel __attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void q1_gemv_lut2(__global const uint4* bits_t,__global const half* scales_t,
                  __global const float* xlut,__global const float* xsum,
                  __global storage_t* out,const int N,const int K,const int out_off){
    const int g=get_group_id(0),t=get_local_id(0),r0=g*(WG_SIZE*4)+t,U=K>>7;
    const int n0=r0,n1=r0+64,n2=r0+128,n3=r0+192;
    float acc0=0,acc1=0,acc2=0,acc3=0;
    for(int u=0;u<U;++u){const long ub=(long)u*N; __global const float* L=xlut+u*4096;
        const float xs=xsum[u<<1]+xsum[(u<<1)+1];
        float p0=0,p1=0,p2=0,p3=0;
        uint4 b0=(n0<N)?bits_t[ub+n0]:(uint4)0, b1=(n1<N)?bits_t[ub+n1]:(uint4)0,
              b2=(n2<N)?bits_t[ub+n2]:(uint4)0, b3=(n3<N)?bits_t[ub+n3]:(uint4)0;
        #pragma unroll
        for(int seg=0;seg<16;++seg){ __global const float* Ls=L+seg*256;
            p0+=Ls[((uchar*)&b0)[seg]]; p1+=Ls[((uchar*)&b1)[seg]];
            p2+=Ls[((uchar*)&b2)[seg]]; p3+=Ls[((uchar*)&b3)[seg]]; }
        const float s0=(n0<N)?vload_half(ub+n0,scales_t):0, s1=(n1<N)?vload_half(ub+n1,scales_t):0,
                    s2=(n2<N)?vload_half(ub+n2,scales_t):0, s3=(n3<N)?vload_half(ub+n3,scales_t):0;
        acc0+=s0*(2.0f*p0-xs);acc1+=s1*(2.0f*p1-xs);acc2+=s2*(2.0f*p2-xs);acc3+=s3*(2.0f*p3-xs);}
    if(n0<N)STORE1(acc0,out,out_off+n0);if(n1<N)STORE1(acc1,out,out_off+n1);
    if(n2<N)STORE1(acc2,out,out_off+n2);if(n3<N)STORE1(acc3,out,out_off+n3);
}

// fma: float-multiply masking. If Adreno int-shifts are slower than float FMA,
// pos=Σ bit·x via convert+fma (2 int + 2 float ops) beats the 2-shift int mask
// (3 int + 1 float). Quad-row, math identical to q1_gemv4.
__kernel __attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void q1_gemv_fma(__global const uint4* bits_t,__global const half* scales_t,
                 __global const storage_t* x,__global const float* xsum,
                 __global storage_t* out,const int N,const int K,const int out_off){
    const int g=get_group_id(0),t=get_local_id(0),r0=g*(WG_SIZE*4)+t,U=K>>7;
    const int n0=r0,n1=r0+64,n2=r0+128,n3=r0+192;
    float acc0=0,acc1=0,acc2=0,acc3=0;
    for(int u=0;u<U;++u){const long ub=(long)u*N;
        uint4 b0=(n0<N)?bits_t[ub+n0]:(uint4)0,b1=(n1<N)?bits_t[ub+n1]:(uint4)0,
              b2=(n2<N)?bits_t[ub+n2]:(uint4)0,b3=(n3<N)?bits_t[ub+n3]:(uint4)0;
        float s0=(n0<N)?vload_half(ub+n0,scales_t):0,s1=(n1<N)?vload_half(ub+n1,scales_t):0,
              s2=(n2<N)?vload_half(ub+n2,scales_t):0,s3=(n3<N)?vload_half(ub+n3,scales_t):0;
        const float xs=xsum[u<<1]+xsum[(u<<1)+1]; const __global storage_t* xp=x+(u<<7);
        float4 p0=(float4)0,p1=(float4)0,p2=(float4)0,p3=(float4)0;
        #pragma unroll
        for(int qq=0;qq<4;++qq){uint w0=((uint*)&b0)[qq],w1=((uint*)&b1)[qq],w2=((uint*)&b2)[qq],w3=((uint*)&b3)[qq];
          #pragma unroll
          for(int gg=0;gg<8;++gg){const float4 xv=LOAD4(xp,qq*8+gg);
            p0=fma(convert_float4(((uint4)(w0)>>(uint4)(0,1,2,3))&(uint4)(1)),xv,p0);
            p1=fma(convert_float4(((uint4)(w1)>>(uint4)(0,1,2,3))&(uint4)(1)),xv,p1);
            p2=fma(convert_float4(((uint4)(w2)>>(uint4)(0,1,2,3))&(uint4)(1)),xv,p2);
            p3=fma(convert_float4(((uint4)(w3)>>(uint4)(0,1,2,3))&(uint4)(1)),xv,p3);
            w0>>=4;w1>>=4;w2>>=4;w3>>=4;}}
        acc0+=s0*(2.0f*(p0.x+p0.y+p0.z+p0.w)-xs);acc1+=s1*(2.0f*(p1.x+p1.y+p1.z+p1.w)-xs);
        acc2+=s2*(2.0f*(p2.x+p2.y+p2.z+p2.w)-xs);acc3+=s3*(2.0f*(p3.x+p3.y+p3.z+p3.w)-xs);}
    if(n0<N)STORE1(acc0,out,out_off+n0);if(n1<N)STORE1(acc1,out,out_off+n1);
    if(n2<N)STORE1(acc2,out,out_off+n2);if(n3<N)STORE1(acc3,out,out_off+n3);
}

// h8: fp16 half8 masking — 8 weights/op at Adreno packed-2×fp16 rate.
// x stored fp16. Per byte(8 bits)->short8 mask, AND with x bits, accumulate
// half8. Halves mask-op count vs v4's float4 AND may run 2× (packed fp16).
// Separate kernel + bench; activations are __global const half*.
__kernel __attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void q1_gemv_h8(__global const uint4* bits_t,__global const half* scales_t,
                __global const half* x,__global const float* xsum,
                __global float* out,const int N,const int K,const int out_off){
    const int g=get_group_id(0),t=get_local_id(0),r0=g*(WG_SIZE*4)+t,U=K>>7;
    const int n0=r0,n1=r0+64,n2=r0+128,n3=r0+192;
    float acc0=0,acc1=0,acc2=0,acc3=0;
    const short8 sh=(short8)(15,14,13,12,11,10,9,8);
    for(int u=0;u<U;++u){const long ub=(long)u*N;
        uint4 b0=(n0<N)?bits_t[ub+n0]:(uint4)0,b1=(n1<N)?bits_t[ub+n1]:(uint4)0,
              b2=(n2<N)?bits_t[ub+n2]:(uint4)0,b3=(n3<N)?bits_t[ub+n3]:(uint4)0;
        float s0=(n0<N)?vload_half(ub+n0,scales_t):0,s1=(n1<N)?vload_half(ub+n1,scales_t):0,
              s2=(n2<N)?vload_half(ub+n2,scales_t):0,s3=(n3<N)?vload_half(ub+n3,scales_t):0;
        const float xs=xsum[u<<1]+xsum[(u<<1)+1]; __global const half* xp=x+(u<<7);
        half8 p0=(half8)0,p1=(half8)0,p2=(half8)0,p3=(half8)0;
        #pragma unroll
        for(int by=0;by<16;++by){                       // 16 bytes = 128 weights
            const half8 xv=vload8(by,xp); const short8 iv=as_short8(xv);
            const short w0=(short)(((uchar*)&b0)[by]),w1=(short)(((uchar*)&b1)[by]),
                        w2=(short)(((uchar*)&b2)[by]),w3=(short)(((uchar*)&b3)[by]);
            p0+=as_half8(iv&(((short8)(w0)<<sh)>>(short8)15));
            p1+=as_half8(iv&(((short8)(w1)<<sh)>>(short8)15));
            p2+=as_half8(iv&(((short8)(w2)<<sh)>>(short8)15));
            p3+=as_half8(iv&(((short8)(w3)<<sh)>>(short8)15));
        }
        float4 lo0=convert_float4(p0.lo)+convert_float4(p0.hi);
        float4 lo1=convert_float4(p1.lo)+convert_float4(p1.hi);
        float4 lo2=convert_float4(p2.lo)+convert_float4(p2.hi);
        float4 lo3=convert_float4(p3.lo)+convert_float4(p3.hi);
        acc0+=s0*(2.0f*(lo0.x+lo0.y+lo0.z+lo0.w)-xs);acc1+=s1*(2.0f*(lo1.x+lo1.y+lo1.z+lo1.w)-xs);
        acc2+=s2*(2.0f*(lo2.x+lo2.y+lo2.z+lo2.w)-xs);acc3+=s3*(2.0f*(lo3.x+lo3.y+lo3.z+lo3.w)-xs);}
    if(n0<N)out[out_off+n0]=acc0;if(n1<N)out[out_off+n1]=acc1;
    if(n2<N)out[out_off+n2]=acc2;if(n3<N)out[out_off+n3]=acc3;
}

// h4: native half4 masking (fp16 throughout, no convert). Same op count as v4;
// wins ONLY if Adreno 620 runs packed-2× fp16. Direct fp16 analog of q1_gemv4.
__kernel __attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void q1_gemv_h4(__global const uint4* bits_t,__global const half* scales_t,
                __global const half* x,__global const float* xsum,
                __global float* out,const int N,const int K,const int out_off){
    const int g=get_group_id(0),t=get_local_id(0),r0=g*(WG_SIZE*4)+t,U=K>>7;
    const int n0=r0,n1=r0+64,n2=r0+128,n3=r0+192;
    float acc0=0,acc1=0,acc2=0,acc3=0;
    const short4 sh=(short4)(15,14,13,12);
    for(int u=0;u<U;++u){const long ub=(long)u*N;
        uint4 b0=(n0<N)?bits_t[ub+n0]:(uint4)0,b1=(n1<N)?bits_t[ub+n1]:(uint4)0,
              b2=(n2<N)?bits_t[ub+n2]:(uint4)0,b3=(n3<N)?bits_t[ub+n3]:(uint4)0;
        float s0=(n0<N)?vload_half(ub+n0,scales_t):0,s1=(n1<N)?vload_half(ub+n1,scales_t):0,
              s2=(n2<N)?vload_half(ub+n2,scales_t):0,s3=(n3<N)?vload_half(ub+n3,scales_t):0;
        const float xs=xsum[u<<1]+xsum[(u<<1)+1]; __global const half* xp=x+(u<<7);
        half4 p0=(half4)0,p1=(half4)0,p2=(half4)0,p3=(half4)0;
        #pragma unroll
        for(int qq=0;qq<4;++qq){uint w0=((uint*)&b0)[qq],w1=((uint*)&b1)[qq],w2=((uint*)&b2)[qq],w3=((uint*)&b3)[qq];
          #pragma unroll
          for(int gg=0;gg<8;++gg){const half4 xv=vload4(qq*8+gg,xp); const short4 iv=as_short4(xv);
            p0+=as_half4(iv&(((short4)((short)w0)<<sh)>>(short4)15));
            p1+=as_half4(iv&(((short4)((short)w1)<<sh)>>(short4)15));
            p2+=as_half4(iv&(((short4)((short)w2)<<sh)>>(short4)15));
            p3+=as_half4(iv&(((short4)((short)w3)<<sh)>>(short4)15));
            w0>>=4;w1>>=4;w2>>=4;w3>>=4;}}
        acc0+=s0*(2.0f*((float)p0.x+p0.y+p0.z+p0.w)-xs);acc1+=s1*(2.0f*((float)p1.x+p1.y+p1.z+p1.w)-xs);
        acc2+=s2*(2.0f*((float)p2.x+p2.y+p2.z+p2.w)-xs);acc3+=s3*(2.0f*((float)p3.x+p3.y+p3.z+p3.w)-xs);}
    if(n0<N)out[out_off+n0]=acc0;if(n1<N)out[out_off+n1]=acc1;
    if(n2<N)out[out_off+n2]=acc2;if(n3<N)out[out_off+n3]=acc3;
}

// hex: 6 rows/thread — one step past v4 quad toward more x-reuse/ILP, before
// the oct-row register cliff. Same integer-mask fast path.
__kernel __attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void q1_gemv_hex(__global const uint4* bits_t,__global const half* scales_t,
                 __global const storage_t* x,__global const float* xsum,
                 __global storage_t* out,const int N,const int K,const int out_off){
    const int g=get_group_id(0),t=get_local_id(0),r0=g*(WG_SIZE*6)+t,U=K>>7;
    int n[6]; float acc[6];
    #pragma unroll
    for(int r=0;r<6;++r){n[r]=r0+r*64;acc[r]=0;}
    for(int u=0;u<U;++u){const long ub=(long)u*N;
        uint4 b[6]; float s[6];
        #pragma unroll
        for(int r=0;r<6;++r){ if(n[r]<N){b[r]=bits_t[ub+n[r]];s[r]=vload_half(ub+n[r],scales_t);} else {b[r]=(uint4)0;s[r]=0;} }
        const float xs=xsum[u<<1]+xsum[(u<<1)+1]; const __global storage_t* xp=x+(u<<7);
        float4 p[6];
        #pragma unroll
        for(int r=0;r<6;++r)p[r]=(float4)0;
        #pragma unroll
        for(int qq=0;qq<4;++qq){uint w[6];
          #pragma unroll
          for(int r=0;r<6;++r)w[r]=((uint*)&b[r])[qq];
          #pragma unroll
          for(int gg=0;gg<8;++gg){const float4 xv=LOAD4(xp,qq*8+gg);const int4 iv=as_int4(xv);
            #pragma unroll
            for(int r=0;r<6;++r){p[r]+=as_float4(iv&(((int4)(w[r])<<(int4)(31,30,29,28))>>(int4)(31)));w[r]>>=4;}}}
        #pragma unroll
        for(int r=0;r<6;++r)acc[r]+=s[r]*(2.0f*(p[r].x+p[r].y+p[r].z+p[r].w)-xs);}
    #pragma unroll
    for(int r=0;r<6;++r)if(n[r]<N)STORE1(acc[r],out,out_off+n[r]);
}
