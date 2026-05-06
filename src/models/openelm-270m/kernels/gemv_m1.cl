// Custom GEMV for the M=1 decode path. Replaces CLBlast HGemm at decode in
// pytorch_linear() — CLBlast's M=1 path is naive (single-thread per output,
// no vec4, no cooperative reduction).
//
// Two specializations for the K values used in Mamba-130M:
//   gemv_m1_k768  — kp_vec4 = 3   (in_proj K=768, lm_head K=768)
//   gemv_m1_k1536 — kp_vec4 = 6   (out_proj K=1536, x_proj K=1536)
// Plus a generic fallback gemv_m1 for any K with K >= WG_SIZE && K % WG_SIZE == 0.
// Specialization matters: clang only fully unrolls the dot-chain when the
// loop bound is a compile-time constant, and the unrolled chain is what
// pipelines the float4 dot()s on Adreno.
//
// Threading: 1 workgroup per output column n. WG_SIZE threads cooperate over
// the K reduction via vec4 fp16 loads + __local-memory tree reduce.

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

#ifndef WG_SIZE
#define WG_SIZE 64
#endif

// Tree-reduce a per-thread acc into ls[0]. Caller writes ls[tid] = acc first.
#define TREE_REDUCE(ls)                                                  \
  do {                                                                   \
    barrier(CLK_LOCAL_MEM_FENCE);                                        \
    for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {                         \
      if (tid < s) (ls)[tid] += (ls)[tid + s];                           \
      barrier(CLK_LOCAL_MEM_FENCE);                                      \
    }                                                                    \
  } while (0)

// ───── Specialized: K = 48 (Mamba dt_proj) ─────
// dt_proj is the hottest small-K M=1 GEMV — at K=48 the predicate
// `K >= WG_SIZE` excluded it from the cooperative path, so the previous
// build sent it to CLBlast HGemm M=1 which clocked ~709 µs/call (47×
// slower than the 15 µs ceiling for 147 KB). 1 thread per output, vec4
// hard-unrolled inner. K=48 = 12 vec4 fp16.
__kernel void gemv_m1_k48(
    __global const storage_t* x,    // [48]
    __global const storage_t* W,    // [N, 48]
    __global storage_t* out,        // [N]
    const int N) {
  const int n = get_global_id(0);
  if (n >= N) return;

  const int base = n * 48;

  float acc = 0.0f;
#ifdef USE_FP16
  #pragma unroll
  for (int j = 0; j < 12; ++j) {
    const int off = j * 4;
    float4 xv = vload_half4(0, x + off);
    float4 wv = vload_half4(0, W + base + off);
    acc += dot(xv, wv);
  }
#else
  #pragma unroll
  for (int k = 0; k < 48; ++k) {
    acc += LOAD(x, k) * LOAD(W, base + k);
  }
#endif
  STORE(out, n, acc);
}

// ───── Coalesced access pattern for cooperative GEMV ─────
//
// CRITICAL: thread<→K-element mapping must put consecutive threads on
// consecutive bytes so the wave's 64-thread SIMD load coalesces into a
// single cache-line transaction. Earlier version had each thread read a
// CHUNK [tid*kp : (tid+1)*kp] which made the wave's first-iter access
// pattern stride-`kp` (24 bytes for K=768) → 24 cache-line transactions
// for 512 bytes of work. New pattern: thread tid reads 4 fp16 starting at
// (j * WG_SIZE * 4 + tid * 4) — wave's iter `j` covers exactly
// WG_SIZE * 4 fp16 = 512 contiguous bytes = 8 cache lines. 3× the BW.
//
// Output ordering of the partial sums changes (per-thread acc accumulates
// over a different subset of K) but final tree-reduction sum is the same
// math — fp32 accumulator absorbs the reordering within ULP.

// ───── Specialized: K = 768 (3 vec4 / thread, 3 wave iterations) ─────
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k768(
    __global const storage_t* x,    // [768]
    __global const storage_t* W,    // [N, 768]
    __global storage_t* out,        // [N]
    const int N) {
  __local float ls[WG_SIZE];
  const int n   = get_group_id(0);
  const int tid = get_local_id(0);
  if (n >= N) return;

  const int base = n * 768;

  float acc = 0.0f;
#ifdef USE_FP16
  // 3 iterations, each one wave-coalesced: 256 fp16 = 512 bytes per iter.
  #pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int off = j * (WG_SIZE * 4) + tid * 4;
    float4 xv = vload_half4(0, x + off);
    float4 wv = vload_half4(0, W + base + off);
    acc += dot(xv, wv);
  }
#else
  for (int j = 0; j < 12; ++j) {
    const int k = j * WG_SIZE + tid;
    acc += LOAD(x, k) * LOAD(W, base + k);
  }
#endif

  ls[tid] = acc;
  TREE_REDUCE(ls);

  if (tid == 0) STORE(out, n, ls[0]);
}

// ───── Multi-output specialized: K = 768, 4 outputs per WG ─────
// Each WG produces 4 consecutive output columns, sharing the x reads
// and giving the compiler register-level parallelism across 4 dot-product
// chains. In benchmarks this typically gives 1.2-1.4× over 1-output-per-WG
// on Adreno because: (a) 4× fewer WGs ⇒ less scheduler overhead,
// (b) x is loaded once per WG into registers/L1 instead of per-output,
// (c) the 4 independent dot-product accumulators have no dependency between
// them and pipeline freely on the SP.
//
// Dispatch: gws = (N/4) * WG_SIZE, lws = WG_SIZE. Caller guarantees N%4==0.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k768_no4(
    __global const storage_t* x,    // [768]
    __global const storage_t* W,    // [N, 768]
    __global storage_t* out,        // [N]
    const int N) {
  __local float ls[WG_SIZE * 4];   // 4 accumulators × WG_SIZE threads
  const int wg  = get_group_id(0);
  const int tid = get_local_id(0);
  const int n0  = wg * 4;
  if (n0 >= N) return;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

#ifdef USE_FP16
  const int base0 = (n0 + 0) * 768;
  const int base1 = (n0 + 1) * 768;
  const int base2 = (n0 + 2) * 768;
  const int base3 = (n0 + 3) * 768;

  #pragma unroll
  for (int j = 0; j < 3; ++j) {
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
  for (int j = 0; j < 12; ++j) {
    const int k = j * WG_SIZE + tid;
    float xv = LOAD(x, k);
    acc0 += xv * LOAD(W, (n0+0)*768 + k);
    acc1 += xv * LOAD(W, (n0+1)*768 + k);
    acc2 += xv * LOAD(W, (n0+2)*768 + k);
    acc3 += xv * LOAD(W, (n0+3)*768 + k);
  }
#endif

  ls[tid + 0 * WG_SIZE] = acc0;
  ls[tid + 1 * WG_SIZE] = acc1;
  ls[tid + 2 * WG_SIZE] = acc2;
  ls[tid + 3 * WG_SIZE] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Tree-reduce each accumulator independently. All 4 lanes done in lockstep.
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      ls[tid + 0 * WG_SIZE] += ls[tid + 0 * WG_SIZE + s];
      ls[tid + 1 * WG_SIZE] += ls[tid + 1 * WG_SIZE + s];
      ls[tid + 2 * WG_SIZE] += ls[tid + 2 * WG_SIZE + s];
      ls[tid + 3 * WG_SIZE] += ls[tid + 3 * WG_SIZE + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // First 4 threads write the 4 outputs.
  if (tid < 4) {
    STORE(out, n0 + tid, ls[tid * WG_SIZE]);
  }
}

// ───── Multi-output specialized: K = 768, 8 outputs per WG ─────
// Doubles the per-WG output count over no4. x is loaded once and reused
// across 8 dot-product chains. W bandwidth per output is the same as no4
// (each output still reads its full 768 fp16 row), but:
//   - 8× fewer WGs than 1-output baseline (e.g. lm_head: 50280 → 6285)
//   - x reads amortized 8× per global load
//   - 8 independent dot accumulators → register-level parallelism
//
// Register pressure: 8 float accumulators + 8 float4 W regs + 1 float4 x
// = ~30 32-bit registers per thread. Within Adreno 620's per-wave budget.
// __local: 64 * 8 = 512 floats = 2 KB.
//
// Dispatch: gws = (N/8) * WG_SIZE, lws = WG_SIZE. Caller guarantees N%8==0.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k768_no8(
    __global const storage_t* x,    // [768]
    __global const storage_t* W,    // [N, 768]
    __global storage_t* out,        // [N]
    const int N) {
  __local float ls[WG_SIZE * 8];
  const int wg  = get_group_id(0);
  const int tid = get_local_id(0);
  const int n0  = wg * 8;
  if (n0 >= N) return;

  float acc0=0, acc1=0, acc2=0, acc3=0, acc4=0, acc5=0, acc6=0, acc7=0;

#ifdef USE_FP16
  const int base0 = (n0 + 0) * 768;
  const int base1 = (n0 + 1) * 768;
  const int base2 = (n0 + 2) * 768;
  const int base3 = (n0 + 3) * 768;
  const int base4 = (n0 + 4) * 768;
  const int base5 = (n0 + 5) * 768;
  const int base6 = (n0 + 6) * 768;
  const int base7 = (n0 + 7) * 768;

  #pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int off = j * (WG_SIZE * 4) + tid * 4;
    float4 xv = vload_half4(0, x + off);
    float4 w0 = vload_half4(0, W + base0 + off);
    float4 w1 = vload_half4(0, W + base1 + off);
    float4 w2 = vload_half4(0, W + base2 + off);
    float4 w3 = vload_half4(0, W + base3 + off);
    float4 w4 = vload_half4(0, W + base4 + off);
    float4 w5 = vload_half4(0, W + base5 + off);
    float4 w6 = vload_half4(0, W + base6 + off);
    float4 w7 = vload_half4(0, W + base7 + off);
    acc0 += dot(xv, w0);
    acc1 += dot(xv, w1);
    acc2 += dot(xv, w2);
    acc3 += dot(xv, w3);
    acc4 += dot(xv, w4);
    acc5 += dot(xv, w5);
    acc6 += dot(xv, w6);
    acc7 += dot(xv, w7);
  }
#else
  for (int j = 0; j < 12; ++j) {
    const int k = j * WG_SIZE + tid;
    float xv = LOAD(x, k);
    acc0 += xv * LOAD(W, (n0+0)*768 + k);
    acc1 += xv * LOAD(W, (n0+1)*768 + k);
    acc2 += xv * LOAD(W, (n0+2)*768 + k);
    acc3 += xv * LOAD(W, (n0+3)*768 + k);
    acc4 += xv * LOAD(W, (n0+4)*768 + k);
    acc5 += xv * LOAD(W, (n0+5)*768 + k);
    acc6 += xv * LOAD(W, (n0+6)*768 + k);
    acc7 += xv * LOAD(W, (n0+7)*768 + k);
  }
#endif

  ls[tid + 0 * WG_SIZE] = acc0;
  ls[tid + 1 * WG_SIZE] = acc1;
  ls[tid + 2 * WG_SIZE] = acc2;
  ls[tid + 3 * WG_SIZE] = acc3;
  ls[tid + 4 * WG_SIZE] = acc4;
  ls[tid + 5 * WG_SIZE] = acc5;
  ls[tid + 6 * WG_SIZE] = acc6;
  ls[tid + 7 * WG_SIZE] = acc7;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      ls[tid + 0 * WG_SIZE] += ls[tid + 0 * WG_SIZE + s];
      ls[tid + 1 * WG_SIZE] += ls[tid + 1 * WG_SIZE + s];
      ls[tid + 2 * WG_SIZE] += ls[tid + 2 * WG_SIZE + s];
      ls[tid + 3 * WG_SIZE] += ls[tid + 3 * WG_SIZE + s];
      ls[tid + 4 * WG_SIZE] += ls[tid + 4 * WG_SIZE + s];
      ls[tid + 5 * WG_SIZE] += ls[tid + 5 * WG_SIZE + s];
      ls[tid + 6 * WG_SIZE] += ls[tid + 6 * WG_SIZE + s];
      ls[tid + 7 * WG_SIZE] += ls[tid + 7 * WG_SIZE + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid < 8) {
    STORE(out, n0 + tid, ls[tid * WG_SIZE]);
  }
}

// ───── Multi-output specialized: K = 1536, 4 outputs per WG ─────
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k1536_no4(
    __global const storage_t* x,    // [1536]
    __global const storage_t* W,    // [N, 1536]
    __global storage_t* out,        // [N]
    const int N) {
  __local float ls[WG_SIZE * 4];
  const int wg  = get_group_id(0);
  const int tid = get_local_id(0);
  const int n0  = wg * 4;
  if (n0 >= N) return;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

#ifdef USE_FP16
  const int base0 = (n0 + 0) * 1536;
  const int base1 = (n0 + 1) * 1536;
  const int base2 = (n0 + 2) * 1536;
  const int base3 = (n0 + 3) * 1536;

  #pragma unroll
  for (int j = 0; j < 6; ++j) {
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
  for (int j = 0; j < 24; ++j) {
    const int k = j * WG_SIZE + tid;
    float xv = LOAD(x, k);
    acc0 += xv * LOAD(W, (n0+0)*1536 + k);
    acc1 += xv * LOAD(W, (n0+1)*1536 + k);
    acc2 += xv * LOAD(W, (n0+2)*1536 + k);
    acc3 += xv * LOAD(W, (n0+3)*1536 + k);
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

// ───── Image-buffer variants: read W via Adreno's texture cache ─────
//
// Adreno's texture cache is a separate L1 (independent of the buffer cache)
// tuned for streaming reads of read-only data. Wrapping the fp16 weight
// buffer as an image1d_buffer_t with format CL_RGBA + CL_HALF_FLOAT means
// each "pixel" = 4 fp16 = 8 bytes, accessed via read_imageh(). Adreno's
// texture path typically reaches 1.4-1.7× the BW of buffer vload_half4
// for hot streaming reads of large weight matrices.
//
// Same backing memory as the buffer (clCreateImage with desc.buffer = W) —
// no copy, no extra DRAM. Just an alternate access path that hits a
// different L1.
//
// Pixel-index math: row stride is K/4 pixels (K fp16 / 4 fp16 per pixel).
// Off-in-pixels = off-in-fp16 / 4 = j*WG_SIZE + tid.

#if defined(USE_FP16) && defined(ENABLE_IMAGE_KERNELS)

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k768_no4_img(
    __global const storage_t* x,         // [768]
    __read_only image1d_buffer_t W_img,  // viewed as half4 pixels, N*192 px
    __global storage_t* out,             // [N]
    const int N) {
  __local float ls[WG_SIZE * 4];
  const int wg  = get_group_id(0);
  const int tid = get_local_id(0);
  const int n0  = wg * 4;
  if (n0 >= N) return;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  // Each row of W is 768 fp16 = 192 half4 pixels.
  const int base0_px = (n0 + 0) * 192;
  const int base1_px = (n0 + 1) * 192;
  const int base2_px = (n0 + 2) * 192;
  const int base3_px = (n0 + 3) * 192;

  #pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int off    = j * (WG_SIZE * 4) + tid * 4;
    const int off_px = j * WG_SIZE + tid;             // off / 4
    float4 xv = vload_half4(0, x + off);
    // read_imagef on a CL_HALF_FLOAT format image returns float4 directly,
    // with the half→float conversion done in the texture-cache HW path
    // (cheaper than read_imageh + convert_float4, which goes through the
    // half-precision register file).
    float4 w0 = read_imagef(W_img, base0_px + off_px);
    float4 w1 = read_imagef(W_img, base1_px + off_px);
    float4 w2 = read_imagef(W_img, base2_px + off_px);
    float4 w3 = read_imagef(W_img, base3_px + off_px);
    acc0 += dot(xv, w0);
    acc1 += dot(xv, w1);
    acc2 += dot(xv, w2);
    acc3 += dot(xv, w3);
  }

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

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k1536_no4_img(
    __global const storage_t* x,         // [1536]
    __read_only image1d_buffer_t W_img,  // viewed as half4 pixels, N*384 px
    __global storage_t* out,             // [N]
    const int N) {
  __local float ls[WG_SIZE * 4];
  const int wg  = get_group_id(0);
  const int tid = get_local_id(0);
  const int n0  = wg * 4;
  if (n0 >= N) return;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  // Each row of W is 1536 fp16 = 384 half4 pixels.
  const int base0_px = (n0 + 0) * 384;
  const int base1_px = (n0 + 1) * 384;
  const int base2_px = (n0 + 2) * 384;
  const int base3_px = (n0 + 3) * 384;

  #pragma unroll
  for (int j = 0; j < 6; ++j) {
    const int off    = j * (WG_SIZE * 4) + tid * 4;
    const int off_px = j * WG_SIZE + tid;
    float4 xv = vload_half4(0, x + off);
    // read_imagef on a CL_HALF_FLOAT format image returns float4 directly,
    // with the half→float conversion done in the texture-cache HW path
    // (cheaper than read_imageh + convert_float4, which goes through the
    // half-precision register file).
    float4 w0 = read_imagef(W_img, base0_px + off_px);
    float4 w1 = read_imagef(W_img, base1_px + off_px);
    float4 w2 = read_imagef(W_img, base2_px + off_px);
    float4 w3 = read_imagef(W_img, base3_px + off_px);
    acc0 += dot(xv, w0);
    acc1 += dot(xv, w1);
    acc2 += dot(xv, w2);
    acc3 += dot(xv, w3);
  }

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

#endif  // USE_FP16

// ───── Multi-output K=1536, 4 outputs/WG, with FUSED RESIDUAL ADD ─────
//
// Replaces out_proj GEMV + element_add_inplace into hidden. Reads the
// existing `hidden[n]` value, computes the dot-product, writes
// hidden[n] = sum + hidden[n]. Saves one launch + one R/W roundtrip
// of the full hidden buffer per layer per token.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k1536_no4_radd(
    __global const storage_t* x,      // [1536]
    __global const storage_t* W,      // [N, 1536]
    __global       storage_t* hidden, // [N] — read for residual, write sum+residual
    const int N) {
  __local float ls[WG_SIZE * 4];
  const int wg  = get_group_id(0);
  const int tid = get_local_id(0);
  const int n0  = wg * 4;
  if (n0 >= N) return;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

#ifdef USE_FP16
  const int base0 = (n0 + 0) * 1536;
  const int base1 = (n0 + 1) * 1536;
  const int base2 = (n0 + 2) * 1536;
  const int base3 = (n0 + 3) * 1536;

  #pragma unroll
  for (int j = 0; j < 6; ++j) {
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
  for (int j = 0; j < 24; ++j) {
    const int k = j * WG_SIZE + tid;
    float xv = LOAD(x, k);
    acc0 += xv * LOAD(W, (n0+0)*1536 + k);
    acc1 += xv * LOAD(W, (n0+1)*1536 + k);
    acc2 += xv * LOAD(W, (n0+2)*1536 + k);
    acc3 += xv * LOAD(W, (n0+3)*1536 + k);
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

  // First 4 threads write the 4 outputs with fused residual add.
  if (tid < 4) {
    const int n = n0 + tid;
    const float r = LOAD(hidden, n);
    STORE(hidden, n, ls[tid * WG_SIZE] + r);
  }
}

// ───── Specialized: K = 1536 (6 vec4 / thread, 6 wave iterations) ─────
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k1536(
    __global const storage_t* x,    // [1536]
    __global const storage_t* W,    // [N, 1536]
    __global storage_t* out,        // [N]
    const int N) {
  __local float ls[WG_SIZE];
  const int n   = get_group_id(0);
  const int tid = get_local_id(0);
  if (n >= N) return;

  const int base = n * 1536;

  float acc = 0.0f;
#ifdef USE_FP16
  // 6 iterations, each one wave-coalesced: 256 fp16 = 512 bytes per iter.
  #pragma unroll
  for (int j = 0; j < 6; ++j) {
    const int off = j * (WG_SIZE * 4) + tid * 4;
    float4 xv = vload_half4(0, x + off);
    float4 wv = vload_half4(0, W + base + off);
    acc += dot(xv, wv);
  }
#else
  for (int j = 0; j < 24; ++j) {
    const int k = j * WG_SIZE + tid;
    acc += LOAD(x, k) * LOAD(W, base + k);
  }
#endif

  ls[tid] = acc;
  TREE_REDUCE(ls);

  if (tid == 0) STORE(out, n, ls[0]);
}

// ───── Generic fallback for any K with K >= WG_SIZE && K % WG_SIZE == 0 ─────
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1(
    __global const storage_t* x,    // [K]
    __global const storage_t* W,    // [N, K]
    __global storage_t* out,        // [N]
    const int K,
    const int N) {
  __local float ls[WG_SIZE];

  const int n   = get_group_id(0);
  const int tid = get_local_id(0);
  if (n >= N) return;

  const int base    = n * K;
  const int kp      = K / WG_SIZE;
  const int k_start = tid * kp;

  float acc = 0.0f;
#ifdef USE_FP16
  int j = 0;
  for (; j + 3 < kp; j += 4) {
    const int off = k_start + j;
    float4 xv = vload_half4(0, x + off);
    float4 wv = vload_half4(0, W + base + off);
    acc += dot(xv, wv);
  }
  for (; j < kp; ++j) {
    const int k = k_start + j;
    acc += LOAD(x, k) * LOAD(W, base + k);
  }
#else
  for (int j = 0; j < kp; ++j) {
    const int k = k_start + j;
    acc += LOAD(x, k) * LOAD(W, base + k);
  }
#endif

  ls[tid] = acc;
  TREE_REDUCE(ls);

  if (tid == 0) STORE(out, n, ls[0]);
}

// ───── Generic 4-output multi-output for any K with K >= WG_SIZE && K % WG_SIZE == 0 ─────
// Same pattern as gemv_m1_k768_no4 but parameterized over K. OpenELM-270M has
// 16 layers × variable K per site (qkv_proj K=1280, ffn proj_1 K=1280,
// ffn proj_2 K∈{768..5120}, attn out_proj K∈{768,1024,1280}, lm_head K=1280)
// — a generic _no4 covers them all. 4× fewer WGs than the 1-output baseline,
// x reads coalesced across 4 outputs, 4 independent fp32 accumulators per
// thread (register-level parallelism). Caller guarantees N % 4 == 0.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_no4(
    __global const storage_t* x,    // [K]
    __global const storage_t* W,    // [N, K]
    __global storage_t* out,        // [N]
    const int K,
    const int N) {
  __local float ls[WG_SIZE * 4];

  const int wg  = get_group_id(0);
  const int tid = get_local_id(0);
  const int n0  = wg * 4;
  if (n0 >= N) return;

  const int kp      = K / WG_SIZE;
  const int k_start = tid * kp;
  const int base0   = (n0 + 0) * K;
  const int base1   = (n0 + 1) * K;
  const int base2   = (n0 + 2) * K;
  const int base3   = (n0 + 3) * K;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

#ifdef USE_FP16
  int j = 0;
  for (; j + 3 < kp; j += 4) {
    const int off = k_start + j;
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
  for (; j < kp; ++j) {
    const int k = k_start + j;
    float xv = LOAD(x, k);
    acc0 += xv * LOAD(W, base0 + k);
    acc1 += xv * LOAD(W, base1 + k);
    acc2 += xv * LOAD(W, base2 + k);
    acc3 += xv * LOAD(W, base3 + k);
  }
#else
  for (int j = 0; j < kp; ++j) {
    const int k = k_start + j;
    float xv = LOAD(x, k);
    acc0 += xv * LOAD(W, base0 + k);
    acc1 += xv * LOAD(W, base1 + k);
    acc2 += xv * LOAD(W, base2 + k);
    acc3 += xv * LOAD(W, base3 + k);
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
