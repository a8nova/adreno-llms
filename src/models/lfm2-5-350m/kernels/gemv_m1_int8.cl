// Int8 image-path GEMV variants for the LFM2.5 decode hot loop.
//
// Layout (matches scripts/quantize_weights.py):
//   weight (N, K)  row-major int8, packed 4 elements per RGBA pixel.
//                  Image2D: width = K/4 px, height = N, CL_RGBA / CL_SIGNED_INT8.
//   scale  (N,)    fp16 per-row absolute-max scale; w_real ≈ q * scale[n].
//
// Inner loop reads int4 from image, sign-extended to int → float, dot with
// vload_half4 activations, accumulate as float. Per-row scale folded in at
// the tail (single broadcast load, single FMA per output).
//
// Mirrors the fp16 image kernels in gemv_m1.cl exactly — same dispatch
// geometry (gws = N/stride * 64, lws = 64), same WG_SIZE, same `pix` indexing.
// Only the image format + tail-scale multiply differ.
//
// Why no qcom_dot8_acc: Adreno 619 v2 (SM6375) does not expose
// cl_qcom_dot_product8 (PDF §9.5.1 — extension is SM8x+ premium tier).
// Plain fma. The win comes from BW halving + image-cache hit on int8 reads.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef WG_SIZE
#define WG_SIZE 64
#endif

__constant sampler_t kImgSamplerI8 =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

// ─── K=1024, image int8 W, 4 outputs per WG. WG=64. ───
// Per WG: 4 wave-stride vec4 iterations × 64 threads = 256 vec4 = 1024 fp16.
// (No tail loop — K is exactly divisible by WG×4.)
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k1024_no4_img_int8(
    __global const half* x,
    __read_only image2d_t W_img,
    __global const half* scales,
    __global half* out,
    const int N) {
  const int n_base = (int)get_group_id(0) * 4;
  const int tid    = (int)get_local_id(0);
  if (n_base >= N) return;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int x_off = j * (64 * 4) + tid * 4;
    const int pix   = j * 64 + tid;
    float4 xv = vload_half4(0, x + x_off);
    float4 w0 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 3)));
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
    const float s0 = vload_half(n_base + 0, scales);
    const float s1 = vload_half(n_base + 1, scales);
    const float s2 = vload_half(n_base + 2, scales);
    const float s3 = vload_half(n_base + 3, scales);
    vstore_half(partial[0][0] * s0, 0, out + n_base + 0);
    vstore_half(partial[1][0] * s1, 0, out + n_base + 1);
    vstore_half(partial[2][0] * s2, 0, out + n_base + 2);
    vstore_half(partial[3][0] * s3, 0, out + n_base + 3);
  }
}

// ─── K=1024, image int8 W, 8 outputs per WG. WG=64. ───
// Mirrors fp16 no8_img: doubles per-thread arithmetic density vs no4 to give
// the texture engine more in-flight reads to hide latency. Same x-as-constant
// trick as the fp16 path.
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k1024_no8_img_int8(
    __constant half* x __attribute__((max_constant_size(2048))),
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
  for (int j = 0; j < 4; ++j) {
    const int x_off = j * (64 * 4) + tid * 4;
    const int pix   = j * 64 + tid;
    float4 xv = vload_half4(0, x + x_off);
    float4 w0 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 3)));
    float4 w4 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 4)));
    float4 w5 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 5)));
    float4 w6 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 6)));
    float4 w7 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 7)));
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
    const float s0 = vload_half(n_base + 0, scales);
    const float s1 = vload_half(n_base + 1, scales);
    const float s2 = vload_half(n_base + 2, scales);
    const float s3 = vload_half(n_base + 3, scales);
    const float s4 = vload_half(n_base + 4, scales);
    const float s5 = vload_half(n_base + 5, scales);
    const float s6 = vload_half(n_base + 6, scales);
    const float s7 = vload_half(n_base + 7, scales);
    vstore_half(partial[0][0] * s0, 0, out + n_base + 0);
    vstore_half(partial[1][0] * s1, 0, out + n_base + 1);
    vstore_half(partial[2][0] * s2, 0, out + n_base + 2);
    vstore_half(partial[3][0] * s3, 0, out + n_base + 3);
    vstore_half(partial[4][0] * s4, 0, out + n_base + 4);
    vstore_half(partial[5][0] * s5, 0, out + n_base + 5);
    vstore_half(partial[6][0] * s6, 0, out + n_base + 6);
    vstore_half(partial[7][0] * s7, 0, out + n_base + 7);
  }
}

// ─── K=4608 (w2 down-proj), image int8 W, 4 outputs per WG. WG=64. ───
// Inner loop: 18 wave-stride vec4 iterations × 64 threads = 1152 vec4 = 4608 fp16.
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k4608_no4_img_int8(
    __global const half* x,
    __read_only image2d_t W_img,
    __global const half* scales,
    __global half* out,
    const int N) {
  const int n_base = (int)get_group_id(0) * 4;
  const int tid    = (int)get_local_id(0);
  if (n_base >= N) return;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 18; ++j) {
    const int x_off = j * (64 * 4) + tid * 4;
    const int pix   = j * 64 + tid;
    float4 xv = vload_half4(0, x + x_off);
    float4 w0 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 3)));
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
    const float s0 = vload_half(n_base + 0, scales);
    const float s1 = vload_half(n_base + 1, scales);
    const float s2 = vload_half(n_base + 2, scales);
    const float s3 = vload_half(n_base + 3, scales);
    vstore_half(partial[0][0] * s0, 0, out + n_base + 0);
    vstore_half(partial[1][0] * s1, 0, out + n_base + 1);
    vstore_half(partial[2][0] * s2, 0, out + n_base + 2);
    vstore_half(partial[3][0] * s3, 0, out + n_base + 3);
  }
}

// ─── K=2560 (LFM2.5-230M w2 down-proj), image int8 W, 4 outputs per WG. WG=64. ───
// The 230M's intermediate_size is 2560 (block_auto_adjust_ff_dim=false), so its
// down-projection reads K=2560 instead of the 350M's 4608. Same structure as
// gemv_m1_k4608_no4_img_int8, just 10 wave-stride vec4 iters (10*256 = 2560).
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k2560_no4_img_int8(
    __global const half* x,
    __read_only image2d_t W_img,
    __global const half* scales,
    __global half* out,
    const int N) {
  const int n_base = (int)get_group_id(0) * 4;
  const int tid    = (int)get_local_id(0);
  if (n_base >= N) return;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 10; ++j) {
    const int x_off = j * (64 * 4) + tid * 4;
    const int pix   = j * 64 + tid;
    float4 xv = vload_half4(0, x + x_off);
    float4 w0 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 3)));
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
    const float s0 = vload_half(n_base + 0, scales);
    const float s1 = vload_half(n_base + 1, scales);
    const float s2 = vload_half(n_base + 2, scales);
    const float s3 = vload_half(n_base + 3, scales);
    vstore_half(partial[0][0] * s0, 0, out + n_base + 0);
    vstore_half(partial[1][0] * s1, 0, out + n_base + 1);
    vstore_half(partial[2][0] * s2, 0, out + n_base + 2);
    vstore_half(partial[3][0] * s3, 0, out + n_base + 3);
  }
}

// ─── K=4608 (w2 down-proj), image int8 W, 8 outputs per WG. WG=64. ───
// Same outer geometry as no4 but with 8 output rows per WG, doubling per-thread
// arithmetic density. 8 fp32 accumulators + 8 int4 reads per K-iter. Eligibility:
// N % 8 == 0 (LFM2 w2 is [1024, 4608] → N=1024 ✓).
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k4608_no8_img_int8(
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
  for (int j = 0; j < 18; ++j) {
    const int x_off = j * (64 * 4) + tid * 4;
    const int pix   = j * 64 + tid;
    float4 xv = vload_half4(0, x + x_off);
    float4 w0 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 3)));
    float4 w4 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 4)));
    float4 w5 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 5)));
    float4 w6 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 6)));
    float4 w7 = convert_float4(read_imagei(W_img, kImgSamplerI8, (int2)(pix, n_base + 7)));
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
    const float s0 = vload_half(n_base + 0, scales);
    const float s1 = vload_half(n_base + 1, scales);
    const float s2 = vload_half(n_base + 2, scales);
    const float s3 = vload_half(n_base + 3, scales);
    const float s4 = vload_half(n_base + 4, scales);
    const float s5 = vload_half(n_base + 5, scales);
    const float s6 = vload_half(n_base + 6, scales);
    const float s7 = vload_half(n_base + 7, scales);
    vstore_half(partial[0][0] * s0, 0, out + n_base + 0);
    vstore_half(partial[1][0] * s1, 0, out + n_base + 1);
    vstore_half(partial[2][0] * s2, 0, out + n_base + 2);
    vstore_half(partial[3][0] * s3, 0, out + n_base + 3);
    vstore_half(partial[4][0] * s4, 0, out + n_base + 4);
    vstore_half(partial[5][0] * s5, 0, out + n_base + 5);
    vstore_half(partial[6][0] * s6, 0, out + n_base + 6);
    vstore_half(partial[7][0] * s7, 0, out + n_base + 7);
  }
}
