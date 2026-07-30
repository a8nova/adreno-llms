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

// Overridable so a -DWG_SIZE=N build can sweep workgroup size (see DeviceModel::run_bench).
// Was a bare #define, which SILENTLY beat any -D on the command line: the kernel kept
// reqd_work_group_size(64,1,1) while the host dispatched the requested size, so every
// non-64 configuration died with CL_INVALID_WORK_GROUP_SIZE and looked like a hardware limit.
#ifndef WG_SIZE
#define WG_SIZE 64
#endif

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
    const int r0 = g * (WG_SIZE * 4) + t;      // rows r0 + j*WG_SIZE
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
        const int n0 = r0, n1 = r0 + WG_SIZE, n2 = r0 + 2 * WG_SIZE, n3 = r0 + 3 * WG_SIZE;
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
    // WG_SIZE-relative, NOT hardcoded: with r0 = g*(WG_SIZE*4)+t, a fixed +64/+128/+192 only covers
    // the whole row block when WG_SIZE==64. At 128 it silently skipped 192 rows of every 512.
    if (r0 < N)                   STORE1(acc0, out, out_off + r0);
    if (r0 + WG_SIZE < N)         STORE1(acc1, out, out_off + r0 + WG_SIZE);
    if (r0 + 2 * WG_SIZE < N)     STORE1(acc2, out, out_off + r0 + 2 * WG_SIZE);
    if (r0 + 3 * WG_SIZE < N)     STORE1(acc3, out, out_off + r0 + 3 * WG_SIZE);
}

// NOTE: the non-split texture GEMV (q1_gemv4_img) already exists further down in this file —
// it was written when the bound test showed the GEMV is weight-load latency bound, but was
// never wired up host-side. Only the SPLIT-K twin was missing, and split-K is the path most
// of this model's projections actually take, so the texture win never reached them.
// ── Split-K with the reduction FUSED in (no second dispatch) ───────────────
// MEASURED on an Adreno 840, per decode token: sk_reduce ran 496 times for 8.3 ms — 27% of the
// token — and an EMPTY kernel launch on that device costs 16.8 us against sk_reduce's 16.7 us.
// The reduce pass was doing nothing but paying for its own dispatch. Worse, split-K makes the GEMV
// about 2x faster but was handing all of that back: gemv_sk and sk_reduce each cost ~16.7 us, so
// the pair cost exactly what one unsplit GEMV would have.
//
// So the last workgroup to finish a row-block reduces it in place. `flags` holds one counter per
// row-block; each group bumps it after publishing its partials, and whoever sees splits-1 knows
// every slice is written and does the sum. That group also resets the counter to 0, so the buffer
// is self-cleaning and needs no zeroing dispatch (which would have put the launch straight back).
//
// The release/acquire pair is load-bearing, not decoration: the reducing group reads partials that
// OTHER workgroups wrote, and without the ordering there is no guarantee those stores are visible.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemv4_sk_img_fused(__read_only image1d_buffer_t bits_img,
                           __global const half* scales_t,
                           __global const storage_t* x,
                           __global const float* xsum,
                           __global float* partial,        // [splits][N]
                           __global volatile int* flags,   // [ng] arrival counters, self-resetting (atomic_int)
                           __global storage_t* out,
                           const int N,
                           const int K,
                           const int splits,
                           const int out_off) {
    const int U  = K >> 7;
    // #4: NO integer divide or modulo. This used to derive (row-block, split) from a flat group id
    // with `gid % ng` and `gid / ng`; guide 8.12 calls integer divide "very expensive" on Adreno and
    // says to avoid the modulo operator outright. A 2D dispatch carries both indices in hardware, so
    // the arithmetic disappears rather than getting cheaper.
    const int g = (int)get_group_id(0);
    const int sp = (int)get_group_id(1);
    const int t = (int)get_local_id(0);
    const int r0 = g * (WG_SIZE * 4) + t;
    // IDENTICAL slicing to q1_gemv4_sk. Both a ceil-based and a proportional split partition [0,U)
    // correctly, but they group the per-unit sums differently, so the two kernels would disagree in
    // the last bits — and only for shapes where splits does not divide U (here: attn_q, splits=7,
    // U=40). Matching the reference exactly is what makes "bit-exact" a usable test rather than an
    // approximate one that hides real bugs behind expected noise.
    const int u0 = (int)(((long)U * sp) / splits);
    const int u1 = (int)(((long)U * (sp + 1)) / splits);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int u = u0; u < u1; ++u) {
        const int ub = u * N;
        const float xs128 = xsum[u << 1] + xsum[(u << 1) + 1];
        const __global storage_t* xp = x + (u << 7);
        // #5: branch-free. Guide 8.5 — an if/else invokes the control-flow hardware and diverges the
        // wave; select() is a plain ALU op. The index is CLAMPED so the (always in-bounds) image read
        // stays legal for the tail lanes, and the scale is zeroed instead, which nulls their
        // contribution without a branch.
        const int n0 = r0, n1 = r0 + WG_SIZE, n2 = r0 + 2 * WG_SIZE, n3 = r0 + 3 * WG_SIZE;
        const int c0 = min(n0, N - 1), c1 = min(n1, N - 1);
        const int c2 = min(n2, N - 1), c3 = min(n3, N - 1);
        const uint4 b0 = read_imageui(bits_img, ub + c0);
        const uint4 b1 = read_imageui(bits_img, ub + c1);
        const uint4 b2 = read_imageui(bits_img, ub + c2);
        const uint4 b3 = read_imageui(bits_img, ub + c3);
        const float s0 = select(0.0f, vload_half(ub + c0, scales_t), (int)(n0 < N));
        const float s1 = select(0.0f, vload_half(ub + c1, scales_t), (int)(n1 < N));
        const float s2 = select(0.0f, vload_half(ub + c2, scales_t), (int)(n2 < N));
        const float s3 = select(0.0f, vload_half(ub + c3, scales_t), (int)(n3 < N));
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
    const long pb = (long)sp * N;
    if (r0 < N)               partial[pb + r0] = acc0;
    if (r0 + WG_SIZE < N)     partial[pb + r0 + WG_SIZE] = acc1;
    if (r0 + 2 * WG_SIZE < N) partial[pb + r0 + 2 * WG_SIZE] = acc2;
    if (r0 + 3 * WG_SIZE < N) partial[pb + r0 + 3 * WG_SIZE] = acc3;

    // DEVICE-SCOPE release/acquire, not write_mem_fence. The 1.x fences order accesses within a
    // single work-item and say nothing across workgroups, so the reducing group could legally see
    // partials that another group had not yet published — which showed up as one shape out of six
    // returning 3.5e13 while the other five were bit-exact. A race that only bites at some
    // (splits, group-count) combination is exactly the kind that survives a casual test and
    // corrupts output in the field.
    barrier(CLK_GLOBAL_MEM_FENCE);
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);

    __local int last;
    if (t == 0) {
        const int prev = atomic_fetch_add_explicit(
            (__global volatile atomic_int*)&flags[g], 1, memory_order_acq_rel, memory_scope_device);
        last = (prev == splits - 1) ? 1 : 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!last) return;

    // Sole survivor for this row-block: acquire makes every other group's partials visible here.
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
    if (t == 0) flags[g] = 0;                      // self-cleaning: no zeroing dispatch needed
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int n = r0 + j * WG_SIZE;
        if (n < N) {
            float s = 0.0f;
            for (int k = 0; k < splits; ++k) s += partial[(long)k * N + n];
            STORE1(s, out, out_off + n);
        }
    }
}


// Split-K twin of the above — same reason run_gemv() needs one: a narrow matrix cannot fill the GPU
// from N alone, and that is the shape most of this model's projections have.
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
        const int n0 = r0, n1 = r0 + WG_SIZE, n2 = r0 + 2 * WG_SIZE, n3 = r0 + 3 * WG_SIZE;
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
    // WG_SIZE-relative, NOT hardcoded: with r0 = g*(WG_SIZE*4)+t, a fixed +64/+128/+192 only covers
    // the whole row block when WG_SIZE==64. At 128 it silently skipped 192 rows of every 512.
    if (r0 < N)                   STORE1(acc0, out, out_off + r0);
    if (r0 + WG_SIZE < N)         STORE1(acc1, out, out_off + r0 + WG_SIZE);
    if (r0 + 2 * WG_SIZE < N)     STORE1(acc2, out, out_off + r0 + 2 * WG_SIZE);
    if (r0 + 3 * WG_SIZE < N)     STORE1(acc3, out, out_off + r0 + 3 * WG_SIZE);
}

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemv4_sk_img(__read_only image1d_buffer_t bits_img,
                     __global const half* scales_t,
                     __global const storage_t* x,
                     __global const float* xsum,
                     __global float* partial,       // [splits][N]
                     const int N,
                     const int K,
                     const int splits) {
    const int U = K >> 7;
    const int ng = (N + WG_SIZE * 4 - 1) / (WG_SIZE * 4);
    const int gid = (int)get_group_id(0);
    const int g = gid % ng, sp = gid / ng;
    const int t = (int)get_local_id(0);
    const int r0 = g * (WG_SIZE * 4) + t;
    const int per = (U + splits - 1) / splits;
    const int u0 = sp * per, u1 = min(u0 + per, U);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int u = u0; u < u1; ++u) {
        const int ub = u * N;
        const float xs128 = xsum[u << 1] + xsum[(u << 1) + 1];
        const __global storage_t* xp = x + (u << 7);
        uint4 b0 = (uint4)(0), b1 = (uint4)(0), b2 = (uint4)(0), b3 = (uint4)(0);
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        const int n0 = r0, n1 = r0 + WG_SIZE, n2 = r0 + 2 * WG_SIZE, n3 = r0 + 3 * WG_SIZE;
        if (n0 < N) { b0 = read_imageui(bits_img, ub + n0); s0 = vload_half(ub + n0, scales_t); }
        if (n1 < N) { b1 = read_imageui(bits_img, ub + n1); s1 = vload_half(ub + n1, scales_t); }
        if (n2 < N) { b2 = read_imageui(bits_img, ub + n2); s2 = vload_half(ub + n2, scales_t); }
        if (n3 < N) { b3 = read_imageui(bits_img, ub + n3); s3 = vload_half(ub + n3, scales_t); }
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
    if (r0 < N)               partial[(long)sp * N + r0] = acc0;
    if (r0 + WG_SIZE < N)     partial[(long)sp * N + r0 + WG_SIZE] = acc1;
    if (r0 + 2 * WG_SIZE < N) partial[(long)sp * N + r0 + 2 * WG_SIZE] = acc2;
    if (r0 + 3 * WG_SIZE < N) partial[(long)sp * N + r0 + 3 * WG_SIZE] = acc3;
}

// ── q1_gemm_r1: ONE row per work-item, GEMM_MT tokens, activations STAGED ──
// MEASURED on an Adreno 840, ffn_gate N=17408 K=5120, 32 tokens:
//     4-row tile, staged     180.79 ms   0.6 GB/s   private 2944 B/wi
//     1-row,      staged      14.47 ms   6.9 GB/s   private  672 B/wi
//     1-row,      NOT staged  36.90 ms   2.7 GB/s   private 1004 B/wi
//
// One row instead of four is 11.8x: the 4-row tile kept acc[4][MT] + 4 uint4 + 4 float4 live across
// a 128-deep unrolled body and spilled to private memory, which on Adreno is global memory.
//
// The local staging STAYS. Removing it looked right — the GEMV reads x straight from global and hits
// ~40 GB/s, so L2 ought to broadcast — but that reasoning does not carry over: a GEMV work-item needs
// 128 floats per unit while an r1 work-item needs GEMM_MT slices of 128, so dropping the staging
// turned 512 cooperative loads per workgroup into 65536 individual ones and cost 2.5x. The staged
// version loads x once per workgroup and every work-item reads it from on-chip memory.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemm_r1(__global const uint4* bits_t,    // [U][N]
                __global const half* scales_t,   // [U][N]
                __global const storage_t* x,     // [M][K]
                __global const float* xsum,      // [M][K/64]
                __global storage_t* out,         // [M][N]
                const int N,      // output width (rows of the SLICE this call computes)
                const int K,
                const int M,
                const int NW,     // rows per u in the WEIGHT — differs from N on a fused matrix
                const int n0) {   // row offset of this slice within the weight
    const int U   = K >> 7;
    const int per = K >> 6;
    const int ng  = (N + WG_SIZE - 1) / WG_SIZE;
    const int g   = (int)get_group_id(0);
    const int gn  = g % ng;
    const int mt  = g / ng;
    const int t   = (int)get_local_id(0);
    const int n   = gn * WG_SIZE + t;
    const int m0  = mt * GEMM_MT;

    __local float xt[GEMM_MT][128];
    float acc[GEMM_MT];
    #pragma unroll
    for (int i = 0; i < GEMM_MT; ++i) acc[i] = 0.0f;

    for (int u = 0; u < U; ++u) {
        for (int idx = t; idx < GEMM_MT * 128; idx += WG_SIZE) {
            const int mm = idx >> 7, kk = idx & 127;
            const int m = m0 + mm;
            xt[mm][kk] = (m < M) ? (float)LOAD1(x + (long)m * K, (u << 7) + kk) : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Weight row is (n0 + n) in a matrix NW wide; the OUTPUT row is n in a matrix N wide. The
        // two coincide only when the matrix is not fused, which is why they are separate arguments —
        // reusing N for both silently reads the wrong rows once gate and up share one allocation.
        const long ub = (long)u * NW;
        uint4 b = (uint4)(0);
        float s = 0.f;
        if (n < N) { b = bits_t[ub + n0 + n]; s = vload_half(ub + n0 + n, scales_t); }

        #pragma unroll
        for (int mm = 0; mm < GEMM_MT; ++mm) {
            float4 p = (float4)(0.0f);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                uint w = ((const uint*)&b)[q];
                #pragma unroll
                for (int gg = 0; gg < 8; ++gg) {
                    const int4 iv = as_int4(vload4(0, &xt[mm][q * 32 + gg * 4]));
                    p += as_float4(iv & (((int4)(w) << (int4)(31,30,29,28)) >> (int4)(31)));
                    w >>= 4;
                }
            }
            const int msafe = min(m0 + mm, M - 1);
            const long xb = (long)msafe * per + (u << 1);
            acc[mm] += s * (2.0f * (p.x + p.y + p.z + p.w) - (xsum[xb] + xsum[xb + 1]));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (n >= N) return;
    #pragma unroll
    for (int mm = 0; mm < GEMM_MT; ++mm) {
        if (m0 + mm >= M) continue;
        STORE1(acc[mm], out, (long)(m0 + mm) * N + n);
    }
}


// ── lm_head GEMV with FP32 output, and its argmax ─────────────────────────
// The output head is the ONE place fp16 activations must not follow through. Its logits reach
// magnitudes in the hundreds, fp16 has ~3 decimal digits there, and the very next thing that happens
// is an ARGMAX over 248320 of them. Two near-tied candidates quantise to the same half and the wrong
// token wins — output that matches the reference for one or two tokens and then diverges into
// plausible nonsense. This is a repeat offender across ports in this repo, so the head keeps fp32
// storage regardless of what the rest of the network uses.
//
// Under fp32 storage these are byte-identical to the storage_t versions and cost nothing.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemv4_img_f32(__read_only image1d_buffer_t bits_img,
                      __global const half* scales_t,
                      __global const storage_t* x,
                      __global const float* xsum,
                      __global float* out,            // FP32, always
                      const int N, const int K, const int out_off) {
    const int g = (int)get_group_id(0);
    const int t = (int)get_local_id(0);
    const int r0 = g * (WG_SIZE * 4) + t;
    const int U = K >> 7;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int u = 0; u < U; ++u) {
        const int ub = u * N;
        const float xs128 = xsum[u << 1] + xsum[(u << 1) + 1];
        const __global storage_t* xp = x + (u << 7);
        const int n0 = r0, n1 = r0 + WG_SIZE, n2 = r0 + 2 * WG_SIZE, n3 = r0 + 3 * WG_SIZE;
        const int c0 = min(n0, N - 1), c1 = min(n1, N - 1);
        const int c2 = min(n2, N - 1), c3 = min(n3, N - 1);
        const uint4 b0 = read_imageui(bits_img, ub + c0);
        const uint4 b1 = read_imageui(bits_img, ub + c1);
        const uint4 b2 = read_imageui(bits_img, ub + c2);
        const uint4 b3 = read_imageui(bits_img, ub + c3);
        const float s0 = select(0.0f, vload_half(ub + c0, scales_t), (int)(n0 < N));
        const float s1 = select(0.0f, vload_half(ub + c1, scales_t), (int)(n1 < N));
        const float s2 = select(0.0f, vload_half(ub + c2, scales_t), (int)(n2 < N));
        const float s3 = select(0.0f, vload_half(ub + c3, scales_t), (int)(n3 < N));
        float4 p0 = (float4)(0.0f), p1 = (float4)(0.0f),
               p2 = (float4)(0.0f), p3 = (float4)(0.0f);
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint w0 = ((const uint*)&b0)[q], w1 = ((const uint*)&b1)[q],
                 w2 = ((const uint*)&b2)[q], w3 = ((const uint*)&b3)[q];
            #pragma unroll
            for (int gg = 0; gg < 8; ++gg) {
                const int4 iv = as_int4(LOAD4(xp, q * 8 + gg));
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
    if (r0 < N)               out[out_off + r0] = acc0;
    if (r0 + WG_SIZE < N)     out[out_off + r0 + WG_SIZE] = acc1;
    if (r0 + 2 * WG_SIZE < N) out[out_off + r0 + 2 * WG_SIZE] = acc2;
    if (r0 + 3 * WG_SIZE < N) out[out_off + r0 + 3 * WG_SIZE] = acc3;
}

__kernel
__attribute__((reqd_work_group_size(256, 1, 1)))
void q1_argmax_f32(__global const float* logits, __global int* out, const int N) {
    const int tid = (int)get_local_id(0);
    __local float best[256];
    __local int bidx[256];
    float b = -1e30f;
    int bi = 0;
    for (int i = tid; i < N; i += 256) {
        const float v = logits[i];
        if (v > b || (v == b && i < bi)) { b = v; bi = i; }
    }
    best[tid] = b; bidx[tid] = bi;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (best[tid + s] > best[tid] || (best[tid + s] == best[tid] && bidx[tid + s] < bidx[tid])) {
                best[tid] = best[tid + s]; bidx[tid] = bidx[tid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) out[0] = bidx[0];
}

// ══ BATCHED PREFILL ═══════════════════════════════════════════════════════
// Prefill currently runs one token per forward pass, so it re-reads all 3617 MB of weights for EVERY
// prompt token — which is why prefill and decode measure the same tok/s. For text that costs ~9 s of
// TTFT; for vision it is fatal, since one full-resolution image is ~980 tokens (Qwen3-VL max_pixels
// 1003520, 32x32 px per token after the 2x2 merge) = ~125 s before the first word.
//
// A GEMM amortises the weight read across a tile of GEMM_MT tokens: weight traffic drops by a factor
// of GEMM_MT. The activation tile is staged in LOCAL memory once per 128-weight unit and reused by
// every work-item in the group, so the expensive direction (weights) is read once per tile and the
// cheap direction (x) is read once per group.
//
// Layout: x and out are ROW-MAJOR [M][K] / [M][N]; weights stay K-major [U][N] exactly as decode
// uses them, so no repacking and no second copy of the 3.6 GB.
#ifndef GEMM_MT
#define GEMM_MT 4           // TOKENS per work-item, on top of the 4 rows it already owns
#endif

// Each work-item owns 4 output rows (exactly as q1_gemv4, which measured optimal on this GPU) AND
// GEMM_MT tokens. Both tilings matter, for different reasons:
//   - 4 ROWS amortise the ACTIVATION reads: one x load feeds 4 masks. A first version used 1 row x 16
//     tokens and ran 0.7x — SLOWER than looping the GEMV — because it did 32 activation loads per
//     output where the GEMV does 8. Weight amortisation is worthless if you pay it back in x traffic.
//   - GEMM_MT TOKENS amortise the WEIGHT reads: 18 B of weights serve GEMM_MT tokens, so a prompt of
//     M tokens sweeps the 3617 MB ceil(M/GEMM_MT) times instead of M times.
// Accumulators = 4 * GEMM_MT, so GEMM_MT is bounded by registers; local use is GEMM_MT*512 B.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
// Texture-path twin of q1_gemm. The load-time tuner on an Adreno 840 picked the SEQUENTIAL
// path over the batched one, and the reason was that the comparison was rigged: run_gemv()
// reads weights through image1d_buffer AND splits K, while this kernel did neither. Batching
// cuts weight traffic by GEMM_MT, but that is worth nothing if each byte is fetched on the
// slower path — the texture L1 alone was worth +58% on the GEMV.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemm_img(__read_only image1d_buffer_t bits_img,   // [U][N] RGBA/UINT32
             __global const half* scales_t,   // [U][N]
             __global const storage_t* x,     // [M][K]
             __global const float* xsum,      // [M][K/64]
             __global storage_t* out,         // [M][N]
             const int N,
             const int K,
             const int M) {
    const int U   = K >> 7;
    const int per = K >> 6;
    const int ng  = (N + WG_SIZE * 4 - 1) / (WG_SIZE * 4);
    const int g   = (int)get_group_id(0);
    const int gn  = g % ng;
    const int mt  = g / ng;
    const int t   = (int)get_local_id(0);
    const int r0  = gn * (WG_SIZE * 4) + t;
    const int m0  = mt * GEMM_MT;

    __local float xt[GEMM_MT][128];
    float acc[4][GEMM_MT];
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int i = 0; i < GEMM_MT; ++i) acc[r][i] = 0.0f;

    for (int u = 0; u < U; ++u) {
        for (int idx = t; idx < GEMM_MT * 128; idx += WG_SIZE) {
            const int mm = idx >> 7, kk = idx & 127;
            const int m = m0 + mm;
            xt[mm][kk] = (m < M) ? (float)LOAD1(x + (long)m * K, (u << 7) + kk) : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const int ub = u * N;
        const int n0 = r0, n1 = r0 + WG_SIZE, n2 = r0 + 2 * WG_SIZE, n3 = r0 + 3 * WG_SIZE;
        uint4 b0 = (uint4)(0), b1 = (uint4)(0), b2 = (uint4)(0), b3 = (uint4)(0);
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        if (n0 < N) { b0 = read_imageui(bits_img, ub + n0); s0 = vload_half(ub + n0, scales_t); }
        if (n1 < N) { b1 = read_imageui(bits_img, ub + n1); s1 = vload_half(ub + n1, scales_t); }
        if (n2 < N) { b2 = read_imageui(bits_img, ub + n2); s2 = vload_half(ub + n2, scales_t); }
        if (n3 < N) { b3 = read_imageui(bits_img, ub + n3); s3 = vload_half(ub + n3, scales_t); }

        #pragma unroll
        for (int mm = 0; mm < GEMM_MT; ++mm) {
            // NO early exit here. A data-dependent break makes this loop non-unrollable, which
            // turns acc[4][GEMM_MT] from registers into a dynamically-indexed PRIVATE array — and
            // private memory on Adreno is global memory. Every accumulator update then costs a
            // round-trip, 128 times per unit x 40 units. Measured: 344 B/work-item of private memory
            // at MT=1 climbing to 584 B at MT=16, and the batched path running ~100x below the
            // bandwidth its traffic implies. Out-of-range lanes are already harmless: the staging
            // loop zero-fills xt for m >= M and every store below is guarded, so the only thing the
            // break bought was the spill.
            float4 p0 = (float4)(0.0f), p1 = (float4)(0.0f),
                   p2 = (float4)(0.0f), p3 = (float4)(0.0f);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                uint w0 = ((const uint*)&b0)[q], w1 = ((const uint*)&b1)[q],
                     w2 = ((const uint*)&b2)[q], w3 = ((const uint*)&b3)[q];
                #pragma unroll
                for (int gg = 0; gg < 8; ++gg) {
                    // ONE activation load, FOUR masks — the reuse the first version threw away.
                    const int4 iv = as_int4(vload4(0, &xt[mm][q * 32 + gg * 4]));
                    p0 += as_float4(iv & (((int4)(w0) << (int4)(31,30,29,28)) >> (int4)(31)));
                    p1 += as_float4(iv & (((int4)(w1) << (int4)(31,30,29,28)) >> (int4)(31)));
                    p2 += as_float4(iv & (((int4)(w2) << (int4)(31,30,29,28)) >> (int4)(31)));
                    p3 += as_float4(iv & (((int4)(w3) << (int4)(31,30,29,28)) >> (int4)(31)));
                    w0 >>= 4; w1 >>= 4; w2 >>= 4; w3 >>= 4;
                }
            }
            // Clamped: out-of-range lanes now run, and must not read past the end of xsum. Their
            // results are discarded by the guarded stores, so any in-bounds row will do.
            const int msafe = min(m0 + mm, M - 1);
            const long xb = (long)msafe * per + (u << 1);
            const float xs = xsum[xb] + xsum[xb + 1];
            acc[0][mm] += s0 * (2.0f * (p0.x + p0.y + p0.z + p0.w) - xs);
            acc[1][mm] += s1 * (2.0f * (p1.x + p1.y + p1.z + p1.w) - xs);
            acc[2][mm] += s2 * (2.0f * (p2.x + p2.y + p2.z + p2.w) - xs);
            acc[3][mm] += s3 * (2.0f * (p3.x + p3.y + p3.z + p3.w) - xs);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // COMPILE-TIME trip count, runtime guard INSIDE. `mm < GEMM_MT && m0 + mm < M` reads as a bounds
    // check but it is a data-dependent loop bound, so the compiler cannot unroll it — and an
    // un-unrolled loop that indexes acc[r][mm] forces the whole accumulator block out of registers
    // into private memory, which on Adreno is global memory. Measured on an Adreno 840: 2944 B/wi of
    // private memory and 0.6 GB/s, against 8.7 GB/s for the GEMV doing the same arithmetic. Removing
    // the identical break from the compute loop alone was not enough — one un-unrollable loop
    // anywhere that touches acc is sufficient to spill it.
    #pragma unroll
    for (int mm = 0; mm < GEMM_MT; ++mm) {
        if (m0 + mm >= M) continue;
        const long ob = (long)(m0 + mm) * N;
        if (r0 < N)               STORE1(acc[0][mm], out, ob + r0);
        if (r0 + WG_SIZE < N)     STORE1(acc[1][mm], out, ob + r0 + WG_SIZE);
        if (r0 + 2 * WG_SIZE < N) STORE1(acc[2][mm], out, ob + r0 + 2 * WG_SIZE);
        if (r0 + 3 * WG_SIZE < N) STORE1(acc[3][mm], out, ob + r0 + 3 * WG_SIZE);
    }
}

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemm(__global const uint4* bits_t,    // [U][N]
             __global const half* scales_t,   // [U][N]
             __global const storage_t* x,     // [M][K]
             __global const float* xsum,      // [M][K/64]
             __global storage_t* out,         // [M][N]
             const int N,
             const int K,
             const int M) {
    const int U   = K >> 7;
    const int per = K >> 6;
    const int ng  = (N + WG_SIZE * 4 - 1) / (WG_SIZE * 4);
    const int g   = (int)get_group_id(0);
    const int gn  = g % ng;
    const int mt  = g / ng;
    const int t   = (int)get_local_id(0);
    const int r0  = gn * (WG_SIZE * 4) + t;
    const int m0  = mt * GEMM_MT;

    __local float xt[GEMM_MT][128];
    float acc[4][GEMM_MT];
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int i = 0; i < GEMM_MT; ++i) acc[r][i] = 0.0f;

    for (int u = 0; u < U; ++u) {
        for (int idx = t; idx < GEMM_MT * 128; idx += WG_SIZE) {
            const int mm = idx >> 7, kk = idx & 127;
            const int m = m0 + mm;
            xt[mm][kk] = (m < M) ? (float)LOAD1(x + (long)m * K, (u << 7) + kk) : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const long ub = (long)u * N;
        const int n0 = r0, n1 = r0 + WG_SIZE, n2 = r0 + 2 * WG_SIZE, n3 = r0 + 3 * WG_SIZE;
        uint4 b0 = (uint4)(0), b1 = (uint4)(0), b2 = (uint4)(0), b3 = (uint4)(0);
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        if (n0 < N) { b0 = bits_t[ub + n0]; s0 = vload_half(ub + n0, scales_t); }
        if (n1 < N) { b1 = bits_t[ub + n1]; s1 = vload_half(ub + n1, scales_t); }
        if (n2 < N) { b2 = bits_t[ub + n2]; s2 = vload_half(ub + n2, scales_t); }
        if (n3 < N) { b3 = bits_t[ub + n3]; s3 = vload_half(ub + n3, scales_t); }

        #pragma unroll
        for (int mm = 0; mm < GEMM_MT; ++mm) {
            // NO early exit here. A data-dependent break makes this loop non-unrollable, which
            // turns acc[4][GEMM_MT] from registers into a dynamically-indexed PRIVATE array — and
            // private memory on Adreno is global memory. Every accumulator update then costs a
            // round-trip, 128 times per unit x 40 units. Measured: 344 B/work-item of private memory
            // at MT=1 climbing to 584 B at MT=16, and the batched path running ~100x below the
            // bandwidth its traffic implies. Out-of-range lanes are already harmless: the staging
            // loop zero-fills xt for m >= M and every store below is guarded, so the only thing the
            // break bought was the spill.
            float4 p0 = (float4)(0.0f), p1 = (float4)(0.0f),
                   p2 = (float4)(0.0f), p3 = (float4)(0.0f);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                uint w0 = ((const uint*)&b0)[q], w1 = ((const uint*)&b1)[q],
                     w2 = ((const uint*)&b2)[q], w3 = ((const uint*)&b3)[q];
                #pragma unroll
                for (int gg = 0; gg < 8; ++gg) {
                    // ONE activation load, FOUR masks — the reuse the first version threw away.
                    const int4 iv = as_int4(vload4(0, &xt[mm][q * 32 + gg * 4]));
                    p0 += as_float4(iv & (((int4)(w0) << (int4)(31,30,29,28)) >> (int4)(31)));
                    p1 += as_float4(iv & (((int4)(w1) << (int4)(31,30,29,28)) >> (int4)(31)));
                    p2 += as_float4(iv & (((int4)(w2) << (int4)(31,30,29,28)) >> (int4)(31)));
                    p3 += as_float4(iv & (((int4)(w3) << (int4)(31,30,29,28)) >> (int4)(31)));
                    w0 >>= 4; w1 >>= 4; w2 >>= 4; w3 >>= 4;
                }
            }
            // Clamped: out-of-range lanes now run, and must not read past the end of xsum. Their
            // results are discarded by the guarded stores, so any in-bounds row will do.
            const int msafe = min(m0 + mm, M - 1);
            const long xb = (long)msafe * per + (u << 1);
            const float xs = xsum[xb] + xsum[xb + 1];
            acc[0][mm] += s0 * (2.0f * (p0.x + p0.y + p0.z + p0.w) - xs);
            acc[1][mm] += s1 * (2.0f * (p1.x + p1.y + p1.z + p1.w) - xs);
            acc[2][mm] += s2 * (2.0f * (p2.x + p2.y + p2.z + p2.w) - xs);
            acc[3][mm] += s3 * (2.0f * (p3.x + p3.y + p3.z + p3.w) - xs);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // COMPILE-TIME trip count, runtime guard INSIDE. `mm < GEMM_MT && m0 + mm < M` reads as a bounds
    // check but it is a data-dependent loop bound, so the compiler cannot unroll it — and an
    // un-unrolled loop that indexes acc[r][mm] forces the whole accumulator block out of registers
    // into private memory, which on Adreno is global memory. Measured on an Adreno 840: 2944 B/wi of
    // private memory and 0.6 GB/s, against 8.7 GB/s for the GEMV doing the same arithmetic. Removing
    // the identical break from the compute loop alone was not enough — one un-unrollable loop
    // anywhere that touches acc is sufficient to spill it.
    #pragma unroll
    for (int mm = 0; mm < GEMM_MT; ++mm) {
        if (m0 + mm >= M) continue;
        const long ob = (long)(m0 + mm) * N;
        if (r0 < N)               STORE1(acc[0][mm], out, ob + r0);
        if (r0 + WG_SIZE < N)     STORE1(acc[1][mm], out, ob + r0 + WG_SIZE);
        if (r0 + 2 * WG_SIZE < N) STORE1(acc[2][mm], out, ob + r0 + 2 * WG_SIZE);
        if (r0 + 3 * WG_SIZE < N) STORE1(acc[3][mm], out, ob + r0 + 3 * WG_SIZE);
    }
}

// Batched activation sums: one work-item per (token, 64-lane half-unit). The GEMV form assumes a
// single row; prefill needs xsum laid out [M][K/64] to match the row-major x above.
__kernel void q1_xsum_m(__global const storage_t* x,
                        __global float* xsum,
                        const int K,
                        const int M) {
    const int i = (int)get_global_id(0);
    const int per = K >> 6;
    if (i >= M * per) return;
    const int m = i / per, h = i % per;
    const __global storage_t* xs = x + (long)m * K + (h << 6);
    float acc = 0.0f;
    for (int j = 0; j < 16; ++j) {
        const float4 v = LOAD4(xs, j);
        acc += v.x + v.y + v.z + v.w;
    }
    xsum[i] = acc;
}

// ── v4sk: v4 split across K — the fix for SMALL-N GEMVs ──────────────────
// v4 assigns one workgroup per 4*WG output rows, so its parallelism is set purely by N. Measured on
// Adreno 840 (12 CUs), that starves every narrow matrix in the model:
//     lm_head  N=248320 -> 485 groups -> 40.3 GB/s   (full speed)
//     mlp down N=5120   ->  10 groups -> far below
//     attn k/v N=1024   ->   2 groups
//     ssm a/b  N=48     ->   1 group
// Splitting the K loop gives each row-block `splits` workgroups that each accumulate a slice of the
// units, so occupancy stops depending on N. Partial sums land in `partial[split][N]` and q1_sk_reduce
// adds them. The maths is unchanged: the per-unit contributions are a plain sum over u, so cutting
// that sum into contiguous ranges and adding the pieces is exact, not an approximation.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void q1_gemv4_sk(__global const uint4* bits_t,   // [U][N]
                 __global const half* scales_t,  // [U][N]
                 __global const storage_t* x,
                 __global const float* xsum,
                 __global float* partial,        // [splits][N], fp32 always
                 const int N,
                 const int K,
                 const int splits) {
    const int U  = K >> 7;
    const int ng = (N + WG_SIZE * 4 - 1) / (WG_SIZE * 4);   // row-blocks
    const int g  = (int)get_group_id(0);
    const int gn = g % ng;          // which row-block
    const int sp = g / ng;          // which K slice
    const int t  = (int)get_local_id(0);
    const int r0 = gn * (WG_SIZE * 4) + t;
    // Contiguous unit range for this slice. Uneven splits are fine — every unit is covered once.
    const int u0 = (int)(((long)U * sp) / splits);
    const int u1 = (int)(((long)U * (sp + 1)) / splits);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int u = u0; u < u1; ++u) {
        const long ub = (long)u * N;
        const float xs128 = xsum[u << 1] + xsum[(u << 1) + 1];
        const __global storage_t* xp = x + (u << 7);
        uint4 b0 = (uint4)(0), b1 = (uint4)(0), b2 = (uint4)(0), b3 = (uint4)(0);
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        const int n0 = r0, n1 = r0 + WG_SIZE, n2 = r0 + 2 * WG_SIZE, n3 = r0 + 3 * WG_SIZE;
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
    __global float* pout = partial + (long)sp * N;
    if (r0 < N)                 pout[r0] = acc0;
    if (r0 + WG_SIZE < N)       pout[r0 + WG_SIZE] = acc1;
    if (r0 + 2 * WG_SIZE < N)   pout[r0 + 2 * WG_SIZE] = acc2;
    if (r0 + 3 * WG_SIZE < N)   pout[r0 + 3 * WG_SIZE] = acc3;
}

// Sum the split partials into the real output. One work-item per output row; `splits` is small
// (<= 8), so this is a handful of coalesced reads and costs far less than the parallelism it buys.
__kernel void q1_sk_reduce(__global const float* partial,
                           __global storage_t* out,
                           const int N,
                           const int splits,
                           const int out_off) {
    const int n = (int)get_global_id(0);
    if (n >= N) return;
    float a = 0.0f;
    for (int s = 0; s < splits; ++s) a += partial[(long)s * N + n];
    STORE1(a, out, out_off + n);
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

