// Block-32 symmetric Q4 image-path GEMV variants for the LFM2.5 decode hot loop.
//
// Layout (matches scripts/quantize_q4.py):
//   weight (N, K)  row-major, 4 bits per weight, 2 packed in each byte
//                  (low nibble = element 2i, high nibble = element 2i+1).
//                  Stored as a uint8 buffer; bound here as image2d_t
//                  CL_RGBA / CL_UNSIGNED_INT8. Pixel = 4 bytes = 8 weights.
//                  Image width = K/8 pixels (128 for K=1024, 576 for K=4608).
//   scale  (N, K/32) fp16 per-block absolute-max scale. Block size = 32, so
//                    every 4 pixels share one scale (4 px × 8 weights = 32).
//                    Dequant:  (q_stored - 8) * scale[n_row][block_idx].
//
// Inner loop per thread per K-iteration:
//   1. Read 8 x activations (= 2 vec4) from __global half* x
//   2. For each of 4 output rows: read one packed-weight pixel (4 bytes)
//      → unpack to 8 signed ints, dot with the 8 activations, scale by the
//      block's fp16 scale, accumulate.
// Reduces 4 partial accumulators across WG=64 threads via __local tree-reduce.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef WG_SIZE
#define WG_SIZE 64
#endif

__constant sampler_t kImgSamplerQ4 =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

// Helper: unpack 4 packed-bytes (each = 2 nibbles) into 8 signed ints in {-7..7}.
// All compilers fold this to bit-and / bit-shift / subtract sequences.
#define UNPACK8_FROM_PIXEL(pixel, w0, w1, w2, w3, w4, w5, w6, w7) do { \
    w0 = (int)((pixel).x & 0xF) - 8;                                   \
    w1 = (int)(((pixel).x >> 4) & 0xF) - 8;                            \
    w2 = (int)((pixel).y & 0xF) - 8;                                   \
    w3 = (int)(((pixel).y >> 4) & 0xF) - 8;                            \
    w4 = (int)((pixel).z & 0xF) - 8;                                   \
    w5 = (int)(((pixel).z >> 4) & 0xF) - 8;                            \
    w6 = (int)((pixel).w & 0xF) - 8;                                   \
    w7 = (int)(((pixel).w >> 4) & 0xF) - 8;                            \
} while (0)

// ─── K=1024, image Q4 W, 4 outputs per WG. WG=64. ───
// Per WG: 2 K-iters × 64 threads × 8 weights/pixel = 1024 weights covered.
// Block size = 32 → 32 blocks per row → 32 scales per row.
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k1024_q4_no4_img(
    __global const half* x,
    __read_only image2d_t W_img,
    __global const half* scales,   // [N, K/32] = [N, 32] flattened
    __global half* out,
    const int N) {
  const int n_base = (int)get_group_id(0) * 4;
  const int tid    = (int)get_local_id(0);
  if (n_base >= N) return;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int pix   = j * 64 + tid;          // pixel column [0..127]
    const int x_off = pix * 8;                // activation start (8 elems per pixel)
    const int blk   = pix >> 2;               // block index (block = 4 pixels = 32 weights)

    float4 xv0 = vload_half4(0, x + x_off);
    float4 xv1 = vload_half4(0, x + x_off + 4);

    // Row 0
    {
      uint4 p = read_imageui(W_img, kImgSamplerQ4, (int2)(pix, n_base + 0));
      int w0, w1, w2, w3, w4, w5, w6, w7;
      UNPACK8_FROM_PIXEL(p, w0, w1, w2, w3, w4, w5, w6, w7);
      float d = xv0.x*(float)w0 + xv0.y*(float)w1 + xv0.z*(float)w2 + xv0.w*(float)w3
              + xv1.x*(float)w4 + xv1.y*(float)w5 + xv1.z*(float)w6 + xv1.w*(float)w7;
      acc0 += d * vload_half((n_base + 0) * 32 + blk, scales);
    }
    // Row 1
    {
      uint4 p = read_imageui(W_img, kImgSamplerQ4, (int2)(pix, n_base + 1));
      int w0, w1, w2, w3, w4, w5, w6, w7;
      UNPACK8_FROM_PIXEL(p, w0, w1, w2, w3, w4, w5, w6, w7);
      float d = xv0.x*(float)w0 + xv0.y*(float)w1 + xv0.z*(float)w2 + xv0.w*(float)w3
              + xv1.x*(float)w4 + xv1.y*(float)w5 + xv1.z*(float)w6 + xv1.w*(float)w7;
      acc1 += d * vload_half((n_base + 1) * 32 + blk, scales);
    }
    // Row 2
    {
      uint4 p = read_imageui(W_img, kImgSamplerQ4, (int2)(pix, n_base + 2));
      int w0, w1, w2, w3, w4, w5, w6, w7;
      UNPACK8_FROM_PIXEL(p, w0, w1, w2, w3, w4, w5, w6, w7);
      float d = xv0.x*(float)w0 + xv0.y*(float)w1 + xv0.z*(float)w2 + xv0.w*(float)w3
              + xv1.x*(float)w4 + xv1.y*(float)w5 + xv1.z*(float)w6 + xv1.w*(float)w7;
      acc2 += d * vload_half((n_base + 2) * 32 + blk, scales);
    }
    // Row 3
    {
      uint4 p = read_imageui(W_img, kImgSamplerQ4, (int2)(pix, n_base + 3));
      int w0, w1, w2, w3, w4, w5, w6, w7;
      UNPACK8_FROM_PIXEL(p, w0, w1, w2, w3, w4, w5, w6, w7);
      float d = xv0.x*(float)w0 + xv0.y*(float)w1 + xv0.z*(float)w2 + xv0.w*(float)w3
              + xv1.x*(float)w4 + xv1.y*(float)w5 + xv1.z*(float)w6 + xv1.w*(float)w7;
      acc3 += d * vload_half((n_base + 3) * 32 + blk, scales);
    }
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
    vstore_half(partial[0][0], 0, out + n_base + 0);
    vstore_half(partial[1][0], 0, out + n_base + 1);
    vstore_half(partial[2][0], 0, out + n_base + 2);
    vstore_half(partial[3][0], 0, out + n_base + 3);
  }
}

// ─── K=4608 (w2 down-proj), image Q4 W, 4 outputs per WG. WG=64. ───
// Per WG: 9 K-iters × 64 threads × 8 weights/pixel = 4608 weights covered.
// K=4608 / 32 = 144 blocks per row → 144 scales per row.
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k4608_q4_no4_img(
    __global const half* x,
    __read_only image2d_t W_img,
    __global const half* scales,   // [N, 144] flattened
    __global half* out,
    const int N) {
  const int n_base = (int)get_group_id(0) * 4;
  const int tid    = (int)get_local_id(0);
  if (n_base >= N) return;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 9; ++j) {
    const int pix   = j * 64 + tid;
    const int x_off = pix * 8;
    const int blk   = pix >> 2;

    float4 xv0 = vload_half4(0, x + x_off);
    float4 xv1 = vload_half4(0, x + x_off + 4);

    {
      uint4 p = read_imageui(W_img, kImgSamplerQ4, (int2)(pix, n_base + 0));
      int w0, w1, w2, w3, w4, w5, w6, w7;
      UNPACK8_FROM_PIXEL(p, w0, w1, w2, w3, w4, w5, w6, w7);
      float d = xv0.x*(float)w0 + xv0.y*(float)w1 + xv0.z*(float)w2 + xv0.w*(float)w3
              + xv1.x*(float)w4 + xv1.y*(float)w5 + xv1.z*(float)w6 + xv1.w*(float)w7;
      acc0 += d * vload_half((n_base + 0) * 144 + blk, scales);
    }
    {
      uint4 p = read_imageui(W_img, kImgSamplerQ4, (int2)(pix, n_base + 1));
      int w0, w1, w2, w3, w4, w5, w6, w7;
      UNPACK8_FROM_PIXEL(p, w0, w1, w2, w3, w4, w5, w6, w7);
      float d = xv0.x*(float)w0 + xv0.y*(float)w1 + xv0.z*(float)w2 + xv0.w*(float)w3
              + xv1.x*(float)w4 + xv1.y*(float)w5 + xv1.z*(float)w6 + xv1.w*(float)w7;
      acc1 += d * vload_half((n_base + 1) * 144 + blk, scales);
    }
    {
      uint4 p = read_imageui(W_img, kImgSamplerQ4, (int2)(pix, n_base + 2));
      int w0, w1, w2, w3, w4, w5, w6, w7;
      UNPACK8_FROM_PIXEL(p, w0, w1, w2, w3, w4, w5, w6, w7);
      float d = xv0.x*(float)w0 + xv0.y*(float)w1 + xv0.z*(float)w2 + xv0.w*(float)w3
              + xv1.x*(float)w4 + xv1.y*(float)w5 + xv1.z*(float)w6 + xv1.w*(float)w7;
      acc2 += d * vload_half((n_base + 2) * 144 + blk, scales);
    }
    {
      uint4 p = read_imageui(W_img, kImgSamplerQ4, (int2)(pix, n_base + 3));
      int w0, w1, w2, w3, w4, w5, w6, w7;
      UNPACK8_FROM_PIXEL(p, w0, w1, w2, w3, w4, w5, w6, w7);
      float d = xv0.x*(float)w0 + xv0.y*(float)w1 + xv0.z*(float)w2 + xv0.w*(float)w3
              + xv1.x*(float)w4 + xv1.y*(float)w5 + xv1.z*(float)w6 + xv1.w*(float)w7;
      acc3 += d * vload_half((n_base + 3) * 144 + blk, scales);
    }
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
    vstore_half(partial[0][0], 0, out + n_base + 0);
    vstore_half(partial[1][0], 0, out + n_base + 1);
    vstore_half(partial[2][0], 0, out + n_base + 2);
    vstore_half(partial[3][0], 0, out + n_base + 3);
  }
}

// ─── K=1024, image Q4 W, 8 outputs per WG. WG=64. ───
// Same outer geometry as q4_no4 (2 K-iters × 64 threads × 8 weights = 1024)
// but with 8 output rows per WG. Activations are read once per K-iter and
// reused across all 8 rows → halves the activation read traffic. Per-thread
// register pressure: 8 fp32 accumulators + (per-row, sequenced) 8 unpacked
// weights + per-iter 2 vec4 activations. Sequenced unpack-and-dot lets the
// compiler reuse weight registers across the 8 rows.
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k1024_q4_no8_img(
    __global const half* x,
    __read_only image2d_t W_img,
    __global const half* scales,
    __global half* out,
    const int N) {
  const int n_base = (int)get_group_id(0) * 8;
  const int tid    = (int)get_local_id(0);
  if (n_base >= N) return;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int pix   = j * 64 + tid;
    const int x_off = pix * 8;
    const int blk   = pix >> 2;

    float4 xv0 = vload_half4(0, x + x_off);
    float4 xv1 = vload_half4(0, x + x_off + 4);

    #define ROW_ACCUM(R, ACC) do {                                              \
      uint4 p = read_imageui(W_img, kImgSamplerQ4, (int2)(pix, n_base + (R)));  \
      int w0, w1, w2, w3, w4, w5, w6, w7;                                       \
      UNPACK8_FROM_PIXEL(p, w0, w1, w2, w3, w4, w5, w6, w7);                    \
      float d = xv0.x*(float)w0 + xv0.y*(float)w1 + xv0.z*(float)w2 + xv0.w*(float)w3 \
              + xv1.x*(float)w4 + xv1.y*(float)w5 + xv1.z*(float)w6 + xv1.w*(float)w7; \
      (ACC) += d * vload_half((n_base + (R)) * 32 + blk, scales);               \
    } while (0)
    ROW_ACCUM(0, acc0);
    ROW_ACCUM(1, acc1);
    ROW_ACCUM(2, acc2);
    ROW_ACCUM(3, acc3);
    ROW_ACCUM(4, acc4);
    ROW_ACCUM(5, acc5);
    ROW_ACCUM(6, acc6);
    ROW_ACCUM(7, acc7);
    #undef ROW_ACCUM
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
    vstore_half(partial[0][0], 0, out + n_base + 0);
    vstore_half(partial[1][0], 0, out + n_base + 1);
    vstore_half(partial[2][0], 0, out + n_base + 2);
    vstore_half(partial[3][0], 0, out + n_base + 3);
    vstore_half(partial[4][0], 0, out + n_base + 4);
    vstore_half(partial[5][0], 0, out + n_base + 5);
    vstore_half(partial[6][0], 0, out + n_base + 6);
    vstore_half(partial[7][0], 0, out + n_base + 7);
  }
}
