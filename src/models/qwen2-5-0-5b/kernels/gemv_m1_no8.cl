// Reference: model_info/transformers_src/modeling_qwen2.py (Qwen2Attention/Qwen2MLP)
// 8-output-per-WG GEMV variants for M=1 decode.
// Compiled in a SEPARATE cl_program from gemv_m1.cl so these larger kernels
// (8 reads/iter × 19 iters = 152 texture fetches unrolled for k4864) do not
// disturb the Adreno compiler's register allocation for the hot no4 kernels.
//
// Dispatch: gws = (N/8)*WG_SIZE, lws = WG_SIZE. N%8==0 required.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
#else
  typedef float storage_t;
#endif

#define WG_SIZE 64

__constant sampler_t kImgSampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

#ifdef USE_FP16

// K=896, image2d-backed W, 8 outputs per WG.
// K_PIX=224=3*64+32 → 3 full waves + 32-pixel tail (same as no4).
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k896_no8_img(
    __global const storage_t* x,
    __read_only image2d_t W_img,
    __global storage_t* out,
    const int N) {
  const int n_base = (int)get_group_id(0) * 8;
  const int tid    = (int)get_local_id(0);
  if (n_base >= N) return;

  __local float partial[8][WG_SIZE];

  __global const half* xh = (__global const half*)x;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int x_off = j * (WG_SIZE * 4) + tid * 4;
    const int pix   = j *  WG_SIZE      + tid;
    float4 xv = vload_half4(0, xh + x_off);
    float4 w0 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 3)));
    float4 w4 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 4)));
    float4 w5 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 5)));
    float4 w6 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 6)));
    float4 w7 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 7)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1); acc2 += dot(xv, w2); acc3 += dot(xv, w3);
    acc4 += dot(xv, w4); acc5 += dot(xv, w5); acc6 += dot(xv, w6); acc7 += dot(xv, w7);
  }
  if (tid < 32) {
    const int x_off = 3 * (WG_SIZE * 4) + tid * 4;
    const int pix   = 3 *  WG_SIZE      + tid;
    float4 xv = vload_half4(0, xh + x_off);
    float4 w0 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 3)));
    float4 w4 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 4)));
    float4 w5 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 5)));
    float4 w6 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 6)));
    float4 w7 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 7)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1); acc2 += dot(xv, w2); acc3 += dot(xv, w3);
    acc4 += dot(xv, w4); acc5 += dot(xv, w5); acc6 += dot(xv, w6); acc7 += dot(xv, w7);
  }
  partial[0][tid] = acc0; partial[1][tid] = acc1; partial[2][tid] = acc2; partial[3][tid] = acc3;
  partial[4][tid] = acc4; partial[5][tid] = acc5; partial[6][tid] = acc6; partial[7][tid] = acc7;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
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

// K=4864, image2d-backed W, 8 outputs per WG. K_PIX=1216=19*64 (no tail).
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k4864_no8_img(
    __global const storage_t* x,
    __read_only image2d_t W_img,
    __global storage_t* out,
    const int N) {
  const int n_base = (int)get_group_id(0) * 8;
  const int tid    = (int)get_local_id(0);
  if (n_base >= N) return;

  __local float partial[8][WG_SIZE];

  __global const half* xh = (__global const half*)x;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 19; ++j) {
    const int x_off = j * (WG_SIZE * 4) + tid * 4;
    const int pix   = j *  WG_SIZE      + tid;
    float4 xv = vload_half4(0, xh + x_off);
    float4 w0 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 3)));
    float4 w4 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 4)));
    float4 w5 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 5)));
    float4 w6 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 6)));
    float4 w7 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 7)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1); acc2 += dot(xv, w2); acc3 += dot(xv, w3);
    acc4 += dot(xv, w4); acc5 += dot(xv, w5); acc6 += dot(xv, w6); acc7 += dot(xv, w7);
  }
  partial[0][tid] = acc0; partial[1][tid] = acc1; partial[2][tid] = acc2; partial[3][tid] = acc3;
  partial[4][tid] = acc4; partial[5][tid] = acc5; partial[6][tid] = acc6; partial[7][tid] = acc7;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
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

#endif  // USE_FP16
