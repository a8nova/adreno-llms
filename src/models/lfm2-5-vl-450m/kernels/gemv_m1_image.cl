// Image2d-backed GEMV kernels for the M=1 decode hot path.
// Ported from lfm2-5-350m/kernels/gemv_m1.cl (proven on Adreno 620).
//
// Reads W through the Adreno texture cache (L1) instead of buffer cache (L2).
// Measured ceiling on Razr 2020:
//   buffer streaming: 7.85 GB/s
//   image  streaming: 13.46 GB/s  — 1.71x faster
//
// Layout: W[N,K] fp16 wrapped as image2d_t with CL_RGBA / CL_HALF_FLOAT
// (4 fp16 per pixel). image_width = K/4, image_height = N. Same backing
// memory as the buffer (cl_khr_image2d_from_buffer).
//
// IMPORTANT: This file is compiled as a SEPARATE cl_program from lfm2_ops.cl
// to avoid register allocation interference (per ARTICLE.md finding —
// combining image kernels with non-image kernels in one program causes the
// Adreno compiler to spill registers in the image kernels, regressing ~10x).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t kImgSampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

// K=1024, image-backed W, 4 outputs per WG. WG=64.
// Inner loop: 4 wave-stride vec4 iterations x 64 threads = 256 vec4 = 1024 fp16.
// (No tail loop — K is exactly divisible by WG*4.)
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k1024_no4_img(
    __global const half* x,
    __read_only image2d_t W_img,
    __global half* out,
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

// K=1024, image-backed W, 8 outputs per WG. WG=64.
// Doubles the per-thread arithmetic density vs no4 (32 dot products vs 16),
// which gives the texture engine more in-flight reads to hide latency.
// Required because K=1024 no4_img stalls at ~60% of texture ceiling (vs
// K=4608 no4_img at 85%) — the difference is exactly per-thread work count.
//
// x is promoted to __constant memory (1024 fp16 = 2048 B, well under the
// per-kernel constant cache budget). Each K-iteration reads the same x vec4
// across all 64 lanes — exactly the "uniform broadcast" pattern the constant
// cache is designed for.
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k1024_no8_img(
    __constant half* x __attribute__((max_constant_size(2048))),
    __read_only image2d_t W_img,
    __global half* out,
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
// Inner loop: 18 wave-stride vec4 iterations x 64 threads = 1152 vec4 = 4608 fp16.
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k4608_no4_img(
    __global const half* x,
    __read_only image2d_t W_img,
    __global half* out,
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
