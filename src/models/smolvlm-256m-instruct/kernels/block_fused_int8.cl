// Int8 image-path GEMV variants for the SmolLM2 decode hot loop.
//
// Layout convention (matches scripts/quantize_weights.py):
//   weight (N, K)     stored row-major int8, packed 4 elements per RGBA pixel.
//                     Image2D: width = K/4 px, height = N. CL_RGBA / CL_SIGNED_INT8.
//   scale  (N,)       fp16 per-row absolute-max scale: w_real ≈ q * scale[n].
//
// Hot inner loop: read int4 from image (sign-extended from -128..127),
//   convert_float4 → float4, vload_half4 activations → float4, dot product.
//   Accumulate float; pull per-row scale[n] OUT once at the tail.
//
// Pattern mirrors gemv_m1_k576_no4_img/fused_*_no4_img exactly so the
// dispatch geometry is unchanged; only the image format + a scale-multiply
// at the end differ.
//
// Why no qcom_dot8_acc: Adreno 619 v2 (SM6375) does not expose
// cl_qcom_dot_product8 (PDF §9.5.1 — extension present on SM8x+ premium tiers,
// absent on SM6375). Plain fma is what we get; the win comes from the BW
// halving + image-cache hit on int8 reads.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef WG_SIZE
#define WG_SIZE 64
#endif

__constant sampler_t kImgSamplerI =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

// ─────────────────────────────────────────────────────────────────────────────
// K=576, image2d int8 W, 4 outputs per WG. Used for Q / K / V / O.
//
// N must be a multiple of 4.
// Dispatch: gws = (N / 4) * WG_SIZE, lws = WG_SIZE.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k576_no4_img_int8(
    __global const half* x,                       // [K]
    __read_only image2d_t W_img,                  // (K/4=144 px wide, N tall, int8 RGBA)
    __global const half* scales,                  // [N] fp16 per-row absolute-max scale
    __global half* out,                           // [N]
    const int N) {
  const int n_base = (int)get_group_id(0) * 4;
  const int tid    = (int)get_local_id(0);
  if (n_base >= N) return;

  __local float partial[4][WG_SIZE];
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  // 2 full waves of vec4 (covers 2*256 = 512 fp16 of K=576).
  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int x_off = j * (WG_SIZE * 4) + tid * 4;
    const int pix   = j *  WG_SIZE      + tid;        // pixel column [0..143]
    float4 xv = vload_half4(0, x + x_off);
    float4 w0 = convert_float4(read_imagei(W_img, kImgSamplerI, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imagei(W_img, kImgSamplerI, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imagei(W_img, kImgSamplerI, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imagei(W_img, kImgSamplerI, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }
  // Tail: 576-512 = 64 fp16 = 16 vec4 → first 16 threads.
  if (tid < 16) {
    const int x_off = 2 * (WG_SIZE * 4) + tid * 4;
    const int pix   = 2 *  WG_SIZE      + tid;
    float4 xv = vload_half4(0, x + x_off);
    float4 w0 = convert_float4(read_imagei(W_img, kImgSamplerI, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imagei(W_img, kImgSamplerI, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imagei(W_img, kImgSamplerI, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imagei(W_img, kImgSamplerI, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }

  partial[0][tid] = acc0; partial[1][tid] = acc1;
  partial[2][tid] = acc2; partial[3][tid] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      partial[0][tid] += partial[0][tid + s];
      partial[1][tid] += partial[1][tid + s];
      partial[2][tid] += partial[2][tid + s];
      partial[3][tid] += partial[3][tid + s];
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

// ─────────────────────────────────────────────────────────────────────────────
// K=576, image2d int8 Wo, 4 outputs per WG, fused residual add.
// Replaces fused_oproj_residual_m1_no4_img on the int8 path.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fused_oproj_residual_m1_no4_img_int8(
    __global const half* attn_out,                // [Q_DIM]
    __read_only image2d_t Wo_img,                 // (Q_DIM/4 px, H tall, int8 RGBA)
    __global const half* scales,                  // [H] fp16
    __global half* residual,                      // [H] INOUT
    const int H) {
  const int n_base = (int)get_group_id(0) * 4;
  const int tid    = (int)get_local_id(0);
  if (n_base >= H) return;

  __local float partial[4][WG_SIZE];
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int x_off = j * (WG_SIZE * 4) + tid * 4;
    const int pix   = j *  WG_SIZE      + tid;
    float4 xv = vload_half4(0, attn_out + x_off);
    float4 w0 = convert_float4(read_imagei(Wo_img, kImgSamplerI, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imagei(Wo_img, kImgSamplerI, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imagei(Wo_img, kImgSamplerI, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imagei(Wo_img, kImgSamplerI, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }
  if (tid < 16) {
    const int x_off = 2 * (WG_SIZE * 4) + tid * 4;
    const int pix   = 2 *  WG_SIZE      + tid;
    float4 xv = vload_half4(0, attn_out + x_off);
    float4 w0 = convert_float4(read_imagei(Wo_img, kImgSamplerI, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imagei(Wo_img, kImgSamplerI, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imagei(Wo_img, kImgSamplerI, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imagei(Wo_img, kImgSamplerI, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }

  partial[0][tid] = acc0; partial[1][tid] = acc1;
  partial[2][tid] = acc2; partial[3][tid] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      partial[0][tid] += partial[0][tid + s];
      partial[1][tid] += partial[1][tid + s];
      partial[2][tid] += partial[2][tid + s];
      partial[3][tid] += partial[3][tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    const float s0 = vload_half(n_base + 0, scales);
    const float s1 = vload_half(n_base + 1, scales);
    const float s2 = vload_half(n_base + 2, scales);
    const float s3 = vload_half(n_base + 3, scales);
    const float r0 = vload_half(n_base + 0, residual);
    const float r1 = vload_half(n_base + 1, residual);
    const float r2 = vload_half(n_base + 2, residual);
    const float r3 = vload_half(n_base + 3, residual);
    vstore_half(r0 + partial[0][0] * s0, n_base + 0, residual);
    vstore_half(r1 + partial[1][0] * s1, n_base + 1, residual);
    vstore_half(r2 + partial[2][0] * s2, n_base + 2, residual);
    vstore_half(r3 + partial[3][0] * s3, n_base + 3, residual);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// K=576, image2d int8 Wgate AND Wup, single output per WG, vec4 inner loop.
// Matches fp16 fused_gate_up_silu_m1_v4_img layout (single-output to keep
// register pressure safe — Qwen Step 7 measured 1.78× regression on _no4 here).
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fused_gate_up_silu_m1_v4_img_int8(
    __global const half* x,                       // [H]
    __read_only image2d_t Wg_img,                 // (H/4 px, INTER tall, int8 RGBA)
    __read_only image2d_t Wu_img,                 // (H/4 px, INTER tall, int8 RGBA)
    __global const half* sg,                      // [INTER] fp16 gate-row scales
    __global const half* su,                      // [INTER] fp16 up-row scales
    __global half* out,                           // [INTER]
    const int INTER) {
  const int c   = (int)get_group_id(0);
  const int tid = (int)get_local_id(0);
  if (c >= INTER) return;

  __local float ls_gate[WG_SIZE];
  __local float ls_up  [WG_SIZE];

  float gate_acc = 0.0f, up_acc = 0.0f;

  // K=576 = 2 vec4 waves + 16-thread tail.
  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int x_off = j * (WG_SIZE * 4) + tid * 4;
    const int pix   = j *  WG_SIZE      + tid;
    float4 xv  = vload_half4(0, x + x_off);
    float4 wgv = convert_float4(read_imagei(Wg_img, kImgSamplerI, (int2)(pix, c)));
    float4 wuv = convert_float4(read_imagei(Wu_img, kImgSamplerI, (int2)(pix, c)));
    gate_acc += dot(xv, wgv);
    up_acc   += dot(xv, wuv);
  }
  if (tid < 16) {
    const int x_off = 2 * (WG_SIZE * 4) + tid * 4;
    const int pix   = 2 *  WG_SIZE      + tid;
    float4 xv  = vload_half4(0, x + x_off);
    float4 wgv = convert_float4(read_imagei(Wg_img, kImgSamplerI, (int2)(pix, c)));
    float4 wuv = convert_float4(read_imagei(Wu_img, kImgSamplerI, (int2)(pix, c)));
    gate_acc += dot(xv, wgv);
    up_acc   += dot(xv, wuv);
  }

  ls_gate[tid] = gate_acc;
  ls_up  [tid] = up_acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      ls_gate[tid] += ls_gate[tid + s];
      ls_up  [tid] += ls_up  [tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    const float g_scale = vload_half(c, sg);
    const float u_scale = vload_half(c, su);
    const float g = ls_gate[0] * g_scale;
    const float u = ls_up  [0] * u_scale;
    vstore_half((g / (1.0f + native_exp(-g))) * u, c, out);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// K=1536 (INTERMEDIATE_SIZE), image2d int8 Wdown, 4 outputs per WG, fused residual.
// K_PIX = 1536/4 = 384 = 6*64 → 6 full waves, no tail.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fused_down_residual_m1_no4_img_int8(
    __global const half* mlp_in,                  // [INTER]
    __read_only image2d_t Wd_img,                 // (INTER/4=384 px, H tall, int8 RGBA)
    __global const half* scales,                  // [H] fp16
    __global half* residual,                      // [H] INOUT
    const int H) {
  const int n_base = (int)get_group_id(0) * 4;
  const int tid    = (int)get_local_id(0);
  if (n_base >= H) return;

  __local float partial[4][WG_SIZE];
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  // 6 full waves of vec4 (covers 6*256 = 1536 fp16 = full K), no tail.
  #pragma unroll
  for (int j = 0; j < 6; ++j) {
    const int x_off = j * (WG_SIZE * 4) + tid * 4;
    const int pix   = j *  WG_SIZE      + tid;
    float4 xv = vload_half4(0, mlp_in + x_off);
    float4 w0 = convert_float4(read_imagei(Wd_img, kImgSamplerI, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imagei(Wd_img, kImgSamplerI, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imagei(Wd_img, kImgSamplerI, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imagei(Wd_img, kImgSamplerI, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }

  partial[0][tid] = acc0; partial[1][tid] = acc1;
  partial[2][tid] = acc2; partial[3][tid] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      partial[0][tid] += partial[0][tid + s];
      partial[1][tid] += partial[1][tid + s];
      partial[2][tid] += partial[2][tid + s];
      partial[3][tid] += partial[3][tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    const float s0 = vload_half(n_base + 0, scales);
    const float s1 = vload_half(n_base + 1, scales);
    const float s2 = vload_half(n_base + 2, scales);
    const float s3 = vload_half(n_base + 3, scales);
    const float r0 = vload_half(n_base + 0, residual);
    const float r1 = vload_half(n_base + 1, residual);
    const float r2 = vload_half(n_base + 2, residual);
    const float r3 = vload_half(n_base + 3, residual);
    vstore_half(r0 + partial[0][0] * s0, n_base + 0, residual);
    vstore_half(r1 + partial[1][0] * s1, n_base + 1, residual);
    vstore_half(r2 + partial[2][0] * s2, n_base + 2, residual);
    vstore_half(r3 + partial[3][0] * s3, n_base + 3, residual);
  }
}
