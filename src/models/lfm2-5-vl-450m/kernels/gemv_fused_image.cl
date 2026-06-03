// Fused multi-output GEMV kernels (image2d). Separate cl_program from
// gemv_m1_image.cl to avoid register allocation interference on Adreno
// (ARTICLE.md: adding fused kernels to the same program as no8 caused 13x
// regression from register spill).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t kFusedSampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

// Fused 3-output GEMV (image2d, K=1024, 4 outputs/WG).
// Q+K+V (N_a=1024, N_b=512, N_c=512) or B+C+X (all 1024).
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_fused3_k1024_no4_img(
    __constant half* x __attribute__((max_constant_size(2048))),
    __read_only image2d_t W_a,
    __read_only image2d_t W_b,
    __read_only image2d_t W_c,
    __global half* y_a,
    __global half* y_b,
    __global half* y_c,
    const int N_a,
    const int N_b,
    const int N_c) {
  const int gid = (int)get_group_id(0);
  const int tid = (int)get_local_id(0);
  const int wg_a = N_a / 4;
  const int wg_ab = (N_a + N_b) / 4;

  int n_base;
  __read_only image2d_t W_img;
  __global half* y;
  if (gid < wg_a) {
    n_base = gid * 4; W_img = W_a; y = y_a;
  } else if (gid < wg_ab) {
    n_base = (gid - wg_a) * 4; W_img = W_b; y = y_b;
  } else {
    n_base = (gid - wg_ab) * 4; W_img = W_c; y = y_c;
  }

  __constant const half* xh = x;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int pix = j * 64 + tid;
    float4 xv = vload_half4(0, xh + j * 256 + tid * 4);
    acc0 += dot(xv, convert_float4(read_imageh(W_img, kFusedSampler, (int2)(pix, n_base + 0))));
    acc1 += dot(xv, convert_float4(read_imageh(W_img, kFusedSampler, (int2)(pix, n_base + 1))));
    acc2 += dot(xv, convert_float4(read_imageh(W_img, kFusedSampler, (int2)(pix, n_base + 2))));
    acc3 += dot(xv, convert_float4(read_imageh(W_img, kFusedSampler, (int2)(pix, n_base + 3))));
  }
  __local float p[4][64];
  p[0][tid] = acc0; p[1][tid] = acc1; p[2][tid] = acc2; p[3][tid] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 32; s > 0; s >>= 1) {
    if (tid < s) { p[0][tid] += p[0][tid+s]; p[1][tid] += p[1][tid+s]; p[2][tid] += p[2][tid+s]; p[3][tid] += p[3][tid+s]; }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    vstore_half(p[0][0], 0, y + n_base); vstore_half(p[1][0], 0, y + n_base + 1);
    vstore_half(p[2][0], 0, y + n_base + 2); vstore_half(p[3][0], 0, y + n_base + 3);
  }
}

// Fused 2-output GEMV (image2d, K=1024, 4 outputs/WG).
// MLP w1+w3 (gate + up projections).
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_fused2_k1024_no4_img(
    __constant half* x __attribute__((max_constant_size(2048))),
    __read_only image2d_t W_a,
    __read_only image2d_t W_b,
    __global half* y_a,
    __global half* y_b,
    const int N_a,
    const int N_b) {
  const int gid = (int)get_group_id(0);
  const int tid = (int)get_local_id(0);
  const int wg_a = N_a / 4;

  int n_base;
  __read_only image2d_t W_img;
  __global half* y;
  if (gid < wg_a) {
    n_base = gid * 4; W_img = W_a; y = y_a;
  } else {
    n_base = (gid - wg_a) * 4; W_img = W_b; y = y_b;
  }

  __constant const half* xh = x;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int pix = j * 64 + tid;
    float4 xv = vload_half4(0, xh + j * 256 + tid * 4);
    acc0 += dot(xv, convert_float4(read_imageh(W_img, kFusedSampler, (int2)(pix, n_base + 0))));
    acc1 += dot(xv, convert_float4(read_imageh(W_img, kFusedSampler, (int2)(pix, n_base + 1))));
    acc2 += dot(xv, convert_float4(read_imageh(W_img, kFusedSampler, (int2)(pix, n_base + 2))));
    acc3 += dot(xv, convert_float4(read_imageh(W_img, kFusedSampler, (int2)(pix, n_base + 3))));
  }
  __local float p[4][64];
  p[0][tid] = acc0; p[1][tid] = acc1; p[2][tid] = acc2; p[3][tid] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 32; s > 0; s >>= 1) {
    if (tid < s) { p[0][tid] += p[0][tid+s]; p[1][tid] += p[1][tid+s]; p[2][tid] += p[2][tid+s]; p[3][tid] += p[3][tid+s]; }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    vstore_half(p[0][0], 0, y + n_base); vstore_half(p[1][0], 0, y + n_base + 1);
    vstore_half(p[2][0], 0, y + n_base + 2); vstore_half(p[3][0], 0, y + n_base + 3);
  }
}
