// LFM2.5-tuned cooperative GEMV for the M=1 decode hot path.
//
// Replaces the WG=128 single-output gemv_rT_fp32acc baseline. Specializations
// match the four K values that appear in this port:
//   K=1024  — q_proj / k_proj / v_proj / out_proj / conv.in_proj / mlp.w1 / mlp.w3 / lm_head
//   K=4608  — mlp.w2 (down-proj)
// Every N at K=1024 is divisible by 4 (smallest is 512 = k_proj). MLP w2 has
// N=1024 which is also divisible by 4. So the no4 (4-output-per-WG) variant is
// always eligible.
//
// Threading: WG_SIZE=64 (one Adreno A6xx wave). Each WG produces 4 consecutive
// output rows, sharing the x reads and the 4 independent dot-product chains
// give the compiler register-level parallelism.
//
// Compared with the WG=128 single-output baseline:
//   - 4× fewer WGs ⇒ less scheduler/issue overhead
//   - x is loaded once per WG into registers, not 4× across 4 separate WGs
//   - 4 independent fp32 accumulators pipeline without dependency
//   - 64-thread WG fits the wave size exactly (no inter-wave barrier cost)

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

// EXPERIMENT (2026-05-07, lfm2): tried sub_group_reduce_add with
// qcom_reqd_sub_group_size("full") to replace the local-mem tree reduce.
// With the attribute, tokens were ID-for-ID correct but the kernels
// regressed 7× (11.2 → 1.5 tok/s) — full-wave forcing on a 50-fp32-acc
// no8_img kernel either spilled registers or halved active lanes when
// wave_size > 64. Without the attribute, sub_group_reduce_add with the
// compiler's heuristic-picked subgroup size produced wrong tokens (only
// half-wave summed, leaving GEMV outputs at ~1/2 of true value). Reverted;
// the existing barrier-tree reduce stays — it's intra-wave on WG=64 so
// the barriers compile to ~no-ops.

#ifndef WG_SIZE
#define WG_SIZE 64
#endif

#define TREE_REDUCE(ls)                                                  \
  do {                                                                   \
    barrier(CLK_LOCAL_MEM_FENCE);                                        \
    for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {                         \
      if (tid < s) (ls)[tid] += (ls)[tid + s];                           \
      barrier(CLK_LOCAL_MEM_FENCE);                                      \
    }                                                                    \
  } while (0)

// ───── K = 1024, single output, WG=64, fully unrolled ─────
// Hardcoded K=1024 lets the compiler fully unroll the 4-iteration inner loop
// — runtime-K kernels (gemv_fp32acc above) lose ~1.05× because the bound is
// not a compile-time constant. WG=64 = exactly one Adreno A6xx wave so the
// tree-reduce barrier is a no-op (intra-wave).
//
// Per thread: 4 wave-coalesced iterations × (1 vec4 W + 1 vec4 x) = 4 vec4
// of each = 16 fp16. Wave covers 64 × 4 = 256 fp16 = 4 cache lines per iter.
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k1024(
    __global const storage_t* W,    // [N, 1024]
    __global const storage_t* x,    // [1024]
    __global storage_t* y,          // [N]
    const int N) {
  __local float ls[64];
  const int row = get_group_id(0);
  const int tid = get_local_id(0);
  if (row >= N) return;

  const int W_base = row * 1024;
  float acc = 0.0f;

#ifdef USE_FP16
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int off = j * (64 * 4) + tid * 4;
    float4 wv = vload_half4(0, W + W_base + off);
    float4 xv = vload_half4(0, x + off);
    acc += dot(wv, xv);
  }
#else
  #pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int k = j * 64 + tid;
    acc += LOAD(W, W_base + k) * LOAD(x, k);
  }
#endif

  ls[tid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 32; s > 0; s >>= 1) {
    if (tid < s) ls[tid] += ls[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) STORE(y, row, ls[0]);
}

// ───── K = 1024, 4 outputs per WG, WG=128 ─────
// IMPORTANT: WG=128 here, NOT 64. The baseline WG=128 single-output kernel
// for LFM2.5 already saturates the cooperative-reduction approach at high
// thread counts (4608 WGs × 128 threads = 589K threads on the K=1024 N=4608
// MLP w1/w3 site — this matches Adreno 620 occupancy needs to hide DRAM
// latency). Switching to WG=64 alone caused a 4.6× regression because total
// threads dropped to 73K. We keep WG=128 here so total thread count is 147K
// (1152 WGs × 128) — still 4× less than the single-output baseline, but each
// thread now does 4× the work (4 dot-product chains per WG, x amortized 4×).
//
// Each thread per WG handles 2 vec4 of x and 2 vec4 × 4 outputs of W per WG.
// 2 wave-coalesced iterations cover 1024 fp16 = 256 vec4 per output.
#define K1024_WG 128
__kernel __attribute__((reqd_work_group_size(K1024_WG, 1, 1)))
void gemv_m1_k1024_no4(
    __global const storage_t* x,    // [1024]
    __global const storage_t* W,    // [N, 1024]
    __global storage_t* out,        // [N]
    const int N) {
  __local float ls[K1024_WG * 4];
  const int wg  = get_group_id(0);
  const int tid = get_local_id(0);
  const int n0  = wg * 4;
  if (n0 >= N) return;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

#ifdef USE_FP16
  const int base0 = (n0 + 0) * 1024;
  const int base1 = (n0 + 1) * 1024;
  const int base2 = (n0 + 2) * 1024;
  const int base3 = (n0 + 3) * 1024;

  // 2 iterations × 128 threads × 4 fp16 = 1024 fp16 per output. Wave-coalesced
  // (each 64-thread wave reads 256 contiguous fp16 = 4 cache lines per iter).
  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int off = j * (K1024_WG * 4) + tid * 4;
    float4 xv = vload_half4(0, x + off);
    float4 w0 = vload_half4(0, W + base0 + off);
    float4 w1 = vload_half4(0, W + base1 + off);
    float4 w2 = vload_half4(0, W + base2 + off);
    float4 w3 = vload_half4(0, W + base3 + off);
    acc0 += dot(xv, w0);
    acc1 += dot(xv, w1);
    acc2 += dot(xv, w2);
    acc3 += dot(xv, w3);
  }
#else
  for (int j = 0; j < 8; ++j) {
    const int k = j * K1024_WG + tid;
    float xv = LOAD(x, k);
    acc0 += xv * LOAD(W, (n0+0)*1024 + k);
    acc1 += xv * LOAD(W, (n0+1)*1024 + k);
    acc2 += xv * LOAD(W, (n0+2)*1024 + k);
    acc3 += xv * LOAD(W, (n0+3)*1024 + k);
  }
#endif

  ls[tid + 0 * K1024_WG] = acc0;
  ls[tid + 1 * K1024_WG] = acc1;
  ls[tid + 2 * K1024_WG] = acc2;
  ls[tid + 3 * K1024_WG] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = K1024_WG >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      ls[tid + 0 * K1024_WG] += ls[tid + 0 * K1024_WG + s];
      ls[tid + 1 * K1024_WG] += ls[tid + 1 * K1024_WG + s];
      ls[tid + 2 * K1024_WG] += ls[tid + 2 * K1024_WG + s];
      ls[tid + 3 * K1024_WG] += ls[tid + 3 * K1024_WG + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid < 4) {
    STORE(out, n0 + tid, ls[tid * K1024_WG]);
  }
}

// ───── K = 4608, 4 outputs per WG ─────
// MLP w2 down-projection: 4608 fp16 = 1152 vec4 per row. Each thread iterates
// 18 wave-coalesced steps (18 × 64 × 4 = 4608 fp16).
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k4608_no4(
    __global const storage_t* x,    // [4608]
    __global const storage_t* W,    // [N, 4608]
    __global storage_t* out,        // [N]
    const int N) {
  __local float ls[WG_SIZE * 4];
  const int wg  = get_group_id(0);
  const int tid = get_local_id(0);
  const int n0  = wg * 4;
  if (n0 >= N) return;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

#ifdef USE_FP16
  const int base0 = (n0 + 0) * 4608;
  const int base1 = (n0 + 1) * 4608;
  const int base2 = (n0 + 2) * 4608;
  const int base3 = (n0 + 3) * 4608;

  #pragma unroll
  for (int j = 0; j < 18; ++j) {
    const int off = j * (WG_SIZE * 4) + tid * 4;
    float4 xv = vload_half4(0, x + off);
    float4 w0 = vload_half4(0, W + base0 + off);
    float4 w1 = vload_half4(0, W + base1 + off);
    float4 w2 = vload_half4(0, W + base2 + off);
    float4 w3 = vload_half4(0, W + base3 + off);
    acc0 += dot(xv, w0);
    acc1 += dot(xv, w1);
    acc2 += dot(xv, w2);
    acc3 += dot(xv, w3);
  }
#else
  for (int j = 0; j < 72; ++j) {
    const int k = j * WG_SIZE + tid;
    float xv = LOAD(x, k);
    acc0 += xv * LOAD(W, (n0+0)*4608 + k);
    acc1 += xv * LOAD(W, (n0+1)*4608 + k);
    acc2 += xv * LOAD(W, (n0+2)*4608 + k);
    acc3 += xv * LOAD(W, (n0+3)*4608 + k);
  }
#endif

  ls[tid + 0 * WG_SIZE] = acc0;
  ls[tid + 1 * WG_SIZE] = acc1;
  ls[tid + 2 * WG_SIZE] = acc2;
  ls[tid + 3 * WG_SIZE] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      ls[tid + 0 * WG_SIZE] += ls[tid + 0 * WG_SIZE + s];
      ls[tid + 1 * WG_SIZE] += ls[tid + 1 * WG_SIZE + s];
      ls[tid + 2 * WG_SIZE] += ls[tid + 2 * WG_SIZE + s];
      ls[tid + 3 * WG_SIZE] += ls[tid + 3 * WG_SIZE + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid < 4) {
    STORE(out, n0 + tid, ls[tid * WG_SIZE]);
  }
}

// ───── Generic fallback for K %% (WG_SIZE * 4) == 0 ─────
// Used when N % 4 != 0 (none of the LFM2.5 sites today, but kept for safety).
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_generic(
    __global const storage_t* x,    // [K]
    __global const storage_t* W,    // [N, K]
    __global storage_t* out,        // [N]
    const int K,
    const int N) {
  __local float ls[WG_SIZE];
  const int n   = get_group_id(0);
  const int tid = get_local_id(0);
  if (n >= N) return;

  const int base = n * K;
  const int K4 = K >> 2;
  float acc = 0.0f;

#ifdef USE_FP16
  for (int k4 = tid; k4 < K4; k4 += WG_SIZE) {
    float4 xv = vload_half4(k4, x);
    float4 wv = vload_half4(k4, W + base);
    acc += dot(xv, wv);
  }
#else
  for (int k = tid; k < K; k += WG_SIZE) {
    acc += LOAD(x, k) * LOAD(W, base + k);
  }
#endif

  ls[tid] = acc;
  TREE_REDUCE(ls);
  if (tid == 0) STORE(out, n, ls[0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// IMAGE-BACKED GEMV variants — read W through Adreno's texture cache (L1)
// instead of the buffer cache (L2). Measured ceiling on Razr 2020:
//   buffer streaming: 7.85 GB/s
//   image  streaming: 13.46 GB/s   ← 1.71× faster
// Layout: W[N,K] fp16 wrapped as image2d_t with CL_RGBA / CL_HALF_FLOAT
// (4 fp16 per pixel). image_width = K/4, image_height = N. Same backing
// memory as the buffer (cl_khr_image2d_from_buffer / cl_qcom_create_buffer_from_image).
// Sampler: int coords, no boundary handling, no filtering — we always read
// deterministically in-bounds.
// ─────────────────────────────────────────────────────────────────────────────

__constant sampler_t kImgSampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

#ifdef USE_FP16

// K=1024, image-backed W, 4 outputs per WG. WG=64.
// Inner loop: 4 wave-stride vec4 iterations × 64 threads = 256 vec4 = 1024 fp16.
// (No tail loop — K is exactly divisible by WG×4.)
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k1024_no4_img(
    __global const storage_t* x,
    __read_only image2d_t W_img,
    __global storage_t* out,
    const int N) {
  const int n_base = (int)get_group_id(0) * 4;
  const int tid    = (int)get_local_id(0);
  if (n_base >= N) return;

  __global const half* xh = (__global const half*)x;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int x_off = j * (64 * 4) + tid * 4;
    const int pix   = j * 64 + tid;  // pixel column index
    float4 xv = vload_half4(0, xh + x_off);
    float4 w0 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }
  __local float partial[4][64];
  partial[0][tid] = acc0; partial[1][tid] = acc1;
  partial[2][tid] = acc2; partial[3][tid] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 32; s > 0; s >>= 1) {
    if (tid < s) {
      partial[0][tid] += partial[0][tid + s]; partial[1][tid] += partial[1][tid + s];
      partial[2][tid] += partial[2][tid + s]; partial[3][tid] += partial[3][tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    __global half* oh = (__global half*)out;
    vstore_half(partial[0][0], 0, oh + n_base + 0);
    vstore_half(partial[1][0], 0, oh + n_base + 1);
    vstore_half(partial[2][0], 0, oh + n_base + 2);
    vstore_half(partial[3][0], 0, oh + n_base + 3);
  }
}

// K=1024, image-backed W, 2 outputs per WG. WG=64. Lower register pressure
// fallback if K=1024_no4_img regresses (buffer no4 did).
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k1024_no2_img(
    __global const storage_t* x,
    __read_only image2d_t W_img,
    __global storage_t* out,
    const int N) {
  const int n_base = (int)get_group_id(0) * 2;
  const int tid    = (int)get_local_id(0);
  if (n_base >= N) return;

  __global const half* xh = (__global const half*)x;
  float acc0 = 0.0f, acc1 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int x_off = j * (64 * 4) + tid * 4;
    const int pix   = j * 64 + tid;
    float4 xv = vload_half4(0, xh + x_off);
    float4 w0 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 1)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
  }
  __local float partial[2][64];
  partial[0][tid] = acc0; partial[1][tid] = acc1;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 32; s > 0; s >>= 1) {
    if (tid < s) {
      partial[0][tid] += partial[0][tid + s]; partial[1][tid] += partial[1][tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    __global half* oh = (__global half*)out;
    vstore_half(partial[0][0], 0, oh + n_base + 0);
    vstore_half(partial[1][0], 0, oh + n_base + 1);
  }
}

// K=1024, image-backed W, 8 outputs per WG. WG=64.
// Doubles the per-thread arithmetic density vs no4 (32 dot products vs 16),
// which gives the texture engine more in-flight reads to hide latency.
// Required because K=1024 no4_img stalls at ~60% of texture ceiling (vs
// K=4608 no4_img at 85%) — the difference is exactly per-thread work count.
//
// Per thread: 4 K-iterations × (1 x vec4 + 8 W vec4) = 36 vec4 reads = 288 B.
// Register: 8 fp32 acc + 9 fp16x4 in flight = ~50 fp32-equivalent regs. Fits
// the wave's register file in the image path (the texture engine has its
// own pipeline that doesn't compete for VGPRs the way buffer reads do).
// EXPERIMENT (lever #4 from Adreno OpenCL guide §6.4 / §7.1.3): promote
// the activation x to on-chip __constant memory. x is 1024 fp16 = 2048 B,
// well under the per-kernel constant cache budget. Each K-iteration reads
// the same x[off..off+3] vec4 across all 64 lanes — exactly the
// "uniform broadcast" pattern the constant cache is designed for ("can
// broadcast into ALUs in no time"). Without max_constant_size the compiler
// falls back to off-chip system memory.
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k1024_no8_img(
    __constant storage_t* x __attribute__((max_constant_size(2048))),
    __read_only image2d_t W_img,
    __global storage_t* out,
    const int N) {
  const int n_base = (int)get_group_id(0) * 8;
  const int tid    = (int)get_local_id(0);
  if (n_base >= N) return;

  __constant const half* xh = (__constant const half*)x;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int x_off = j * (64 * 4) + tid * 4;
    const int pix   = j * 64 + tid;
    float4 xv = vload_half4(0, xh + x_off);
    float4 w0 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 3)));
    float4 w4 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 4)));
    float4 w5 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 5)));
    float4 w6 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 6)));
    float4 w7 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 7)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
    acc4 += dot(xv, w4); acc5 += dot(xv, w5);
    acc6 += dot(xv, w6); acc7 += dot(xv, w7);
  }
  __local float partial[8][64];
  partial[0][tid] = acc0; partial[1][tid] = acc1;
  partial[2][tid] = acc2; partial[3][tid] = acc3;
  partial[4][tid] = acc4; partial[5][tid] = acc5;
  partial[6][tid] = acc6; partial[7][tid] = acc7;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 32; s > 0; s >>= 1) {
    if (tid < s) {
      partial[0][tid] += partial[0][tid + s]; partial[1][tid] += partial[1][tid + s];
      partial[2][tid] += partial[2][tid + s]; partial[3][tid] += partial[3][tid + s];
      partial[4][tid] += partial[4][tid + s]; partial[5][tid] += partial[5][tid + s];
      partial[6][tid] += partial[6][tid + s]; partial[7][tid] += partial[7][tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    __global half* oh = (__global half*)out;
    vstore_half(partial[0][0], 0, oh + n_base + 0);
    vstore_half(partial[1][0], 0, oh + n_base + 1);
    vstore_half(partial[2][0], 0, oh + n_base + 2);
    vstore_half(partial[3][0], 0, oh + n_base + 3);
    vstore_half(partial[4][0], 0, oh + n_base + 4);
    vstore_half(partial[5][0], 0, oh + n_base + 5);
    vstore_half(partial[6][0], 0, oh + n_base + 6);
    vstore_half(partial[7][0], 0, oh + n_base + 7);
  }
}

// K=4608, image-backed W, 4 outputs per WG. WG=64.
// Inner loop: 18 wave-stride vec4 iterations × 64 threads = 1152 vec4 = 4608 fp16.
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k4608_no4_img(
    __global const storage_t* x,
    __read_only image2d_t W_img,
    __global storage_t* out,
    const int N) {
  const int n_base = (int)get_group_id(0) * 4;
  const int tid    = (int)get_local_id(0);
  if (n_base >= N) return;

  __global const half* xh = (__global const half*)x;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 18; ++j) {
    const int x_off = j * (64 * 4) + tid * 4;
    const int pix   = j * 64 + tid;
    float4 xv = vload_half4(0, xh + x_off);
    float4 w0 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }
  __local float partial[4][64];
  partial[0][tid] = acc0; partial[1][tid] = acc1;
  partial[2][tid] = acc2; partial[3][tid] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 32; s > 0; s >>= 1) {
    if (tid < s) {
      partial[0][tid] += partial[0][tid + s]; partial[1][tid] += partial[1][tid + s];
      partial[2][tid] += partial[2][tid + s]; partial[3][tid] += partial[3][tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    __global half* oh = (__global half*)out;
    vstore_half(partial[0][0], 0, oh + n_base + 0);
    vstore_half(partial[1][0], 0, oh + n_base + 1);
    vstore_half(partial[2][0], 0, oh + n_base + 2);
    vstore_half(partial[3][0], 0, oh + n_base + 3);
  }
}

#endif  // USE_FP16

// ─────────────────────────────────────────────────────────────────────────────
// STREAM benchmark kernels — pure-bandwidth probe with no real compute.
// Used by NNOPT_BW_PROBE=1 to measure the practical streaming-read ceiling
// from BOTH the buffer cache (gemv_stream_buf) and the texture cache
// (gemv_stream_img). The DCE guard ("if acc.x > 1e30f write...") keeps the
// reads live to the compiler without ever firing at runtime.
//
// Methodology: run with N bytes ≫ L2 (Adreno 620 has ~64 KB L2) so every
// read goes to DRAM. Take min over multiple runs to filter frequency-scaling
// spikes. Achieved GB/s = bytes / elapsed.
// ─────────────────────────────────────────────────────────────────────────────

#ifdef USE_FP16

__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_stream_buf(
    __global const half* src,
    __global float* dce_sink,
    const int iters_per_thread) {
  // Coalesced wave-stride: thread tid in WG g, iteration j reads byte offset
  // ((g * WG_ * iters_per_thread) + j * WG_ + tid) * 8, so the 64 threads of
  // a wave-iteration cover 64 × 8 = 512 contiguous bytes (8 cache lines) —
  // same pattern as the live GEMV kernels.
  const int g    = (int)get_group_id(0);
  const int tid  = (int)get_local_id(0);
  const int WG_  = 64;
  const int wg_base = g * WG_ * iters_per_thread;
  __global const half* p = src;
  float4 acc = (float4)(0.0f);
  for (int j = 0; j < iters_per_thread; ++j) {
    int off = (wg_base + j * WG_ + tid) * 4;  // each elem = vec4 of fp16
    acc += vload_half4(0, p + off);
  }
  if (acc.x + acc.y + acc.z + acc.w > 1e30f) dce_sink[g * WG_ + tid] = acc.x;
}

__constant sampler_t kBwSampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_stream_img(
    __read_only image2d_t src_img,
    __global float* dce_sink,
    const int rows_per_wg,
    const int row_pixels) {
  const int g   = (int)get_group_id(0);
  const int tid = (int)get_local_id(0);
  const int WG_ = 64;
  const int row_base = g * rows_per_wg;
  const int iter = row_pixels / WG_;
  float4 acc = (float4)(0.0f);
  for (int y = 0; y < rows_per_wg; ++y) {
    int row = row_base + y;
    for (int j = 0; j < iter; ++j) {
      int pix = j * WG_ + tid;
      acc += convert_float4(read_imageh(src_img, kBwSampler, (int2)(pix, row)));
    }
  }
  if (acc.x + acc.y + acc.z + acc.w > 1e30f) dce_sink[g * WG_ + tid] = acc.x;
}

#endif  // USE_FP16

// Tiny kernel for the recordable-queues probe (cl_qcom_recordable_queues).
// Does almost nothing per dispatch so per-launch CPU overhead dominates,
// exposing any bookkeeping savings the recording API provides.
__kernel void probe_noop(__global int* counter, const int incr) {
  if (get_global_id(0) == 0) atomic_add(counter, incr);
}
