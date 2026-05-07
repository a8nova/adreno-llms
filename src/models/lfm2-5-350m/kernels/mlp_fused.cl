// MLP w3 GEMV + silu_mul fused into one kernel. Reads gate_inout[n] (= w1·x
// from a prior pytorch_linear call), reads W3 via image2d, computes
// silu(gate_inout[n]) * (W3·x), writes the result back into gate_inout.
//
// Lives in its own .cl file (separate cl_program) on purpose: putting this
// kernel in gemv_m1.cl caused the Adreno compiler to spill registers in the
// other no8_img kernels (presumably global cross-kernel register pressure
// analysis), regressing the decode rate by ~10× on Adreno 620. Compiling it
// as a standalone program isolates the allocation.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t kImgSampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

// K=1024, image-backed W3, 8 outputs per WG, fused silu_mul on writeback.
__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_m1_k1024_no8_silufused_img(
    __global const half* x,
    __read_only image2d_t W3_img,
    __global half* gate_inout,
    const int N) {
  const int n_base = (int)get_group_id(0) * 8;
  const int tid    = (int)get_local_id(0);
  if (n_base >= N) return;

  __local float partial[8][64];
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int x_off = j * (64 * 4) + tid * 4;
    const int pix   = j * 64 + tid;
    float4 xv = vload_half4(0, x + x_off);
    float4 w0 = convert_float4(read_imageh(W3_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(W3_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(W3_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(W3_img, kImgSampler, (int2)(pix, n_base + 3)));
    float4 w4 = convert_float4(read_imageh(W3_img, kImgSampler, (int2)(pix, n_base + 4)));
    float4 w5 = convert_float4(read_imageh(W3_img, kImgSampler, (int2)(pix, n_base + 5)));
    float4 w6 = convert_float4(read_imageh(W3_img, kImgSampler, (int2)(pix, n_base + 6)));
    float4 w7 = convert_float4(read_imageh(W3_img, kImgSampler, (int2)(pix, n_base + 7)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
    acc4 += dot(xv, w4); acc5 += dot(xv, w5);
    acc6 += dot(xv, w6); acc7 += dot(xv, w7);
  }
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
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float w3x  = partial[i][0];
      const float gate = vload_half(0, gate_inout + n_base + i);
      const float sig  = 1.0f / (1.0f + native_exp(-gate));
      vstore_half(gate * sig * w3x, 0, gate_inout + n_base + i);
    }
  }
}
