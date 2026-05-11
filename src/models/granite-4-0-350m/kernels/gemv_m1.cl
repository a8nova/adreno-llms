// Custom GEMV for the M=1 decode path. Replaces CLBlast HGemm M=1 in
// pytorch_linear() — CLBlast's M=1 path on Adreno 620 is multiple-dispatch
// and the host enqueue overhead dominates. This kernel is a single
// cooperative-WG dispatch per call (4 outputs per WG, WG=64 lanes covering
// the K reduction with vec4 fp16 loads + tree-reduce in __local).
//
// Ported verbatim from openelm-270m/kernels/gemv_m1.cl (lines 993–1057,
// `gemv_m1_no4_coalesced`). Same coalesced access pattern:
// `off = j*(WG_SIZE*4) + tid*4` → wave of 64 threads reads 256 contiguous
// bytes per iteration (8 cache lines, zero wastage).
//
// Granite K values that qualify (K % 256 == 0, N % 4 == 0):
//   q_proj / o_proj : K=1024 N=1024  ✓
//   k_proj / v_proj : K=1024 N=256   ✓
//   mlp.input_linear: K=1024 N=4096  ✓  (gate_up split happens in shared_mlp.cl)
//   lm_head         : K=1024 N=100352 ✓
//
// Dispatch: gws = (N/4) * WG_SIZE, lws = WG_SIZE. One workgroup per 4-output
// tile. Kernel reads x[K] once per WG (warm in L1), then K/256 vec4 loads of
// each of the 4 output rows of W.

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

// Step #11: enable cl_khr_subgroups for sub_group_reduce_add (replaces the
// __local tree-reduce, kills 6 barriers per WG). Adreno A6xx wave size is 64
// = our WG_SIZE, so sub_group_reduce_add is across the whole WG.
#ifdef ENABLE_SUBGROUPS
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

// ───── Specialized: K = 1024, 4 outputs per WG ─────
// Granite's dominant K (q/o/k/v/mlp_input/lm_head all have K=1024).
// Hardcoded loop bound (1024/(WG*4) = 4) lets clang fully unroll the dot
// chain — no loop counter, no boundary check, scheduled as 4 independent
// dot4 chains per output, instruction-pipelined across 4 outputs/WG.
// OpenELM Step 5b saw +1.61× from the analogous K=1280 specialization.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k1024_no4(
    __global const storage_t* x,    // [1024]
    __global const storage_t* W,    // [N, 1024]
    __global storage_t* out,        // [N]
    const int N) {
  __local float ls[WG_SIZE * 4];
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

  #pragma unroll
  for (int j = 0; j < 4; ++j) {
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
  for (int j = 0; j < 16; ++j) {
    const int k = j * WG_SIZE + tid;
    float xv = LOAD(x, k);
    acc0 += xv * LOAD(W, (n0+0)*1024 + k);
    acc1 += xv * LOAD(W, (n0+1)*1024 + k);
    acc2 += xv * LOAD(W, (n0+2)*1024 + k);
    acc3 += xv * LOAD(W, (n0+3)*1024 + k);
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

#ifdef ENABLE_SUBGROUPS
// ───── Step #11: K=1024, 4 outputs/WG, subgroup-reduced ─────
// Same as gemv_m1_k1024_no4 but uses sub_group_reduce_add to merge the 64
// per-lane partials directly (no __local memory, no 6× barriers in the tree
// reduce). On Adreno A6xx with WG=64 = wave size, the subgroup reduce is a
// single hardware shuffle.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k1024_no4_sg(
    __global const storage_t* x,    // [1024]
    __global const storage_t* W,    // [N, 1024]
    __global storage_t* out,        // [N]
    const int N) {
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

  #pragma unroll
  for (int j = 0; j < 4; ++j) {
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
  for (int j = 0; j < 16; ++j) {
    const int k = j * WG_SIZE + tid;
    float xv = LOAD(x, k);
    acc0 += xv * LOAD(W, (n0+0)*1024 + k);
    acc1 += xv * LOAD(W, (n0+1)*1024 + k);
    acc2 += xv * LOAD(W, (n0+2)*1024 + k);
    acc3 += xv * LOAD(W, (n0+3)*1024 + k);
  }
#endif

  acc0 = sub_group_reduce_add(acc0);
  acc1 = sub_group_reduce_add(acc1);
  acc2 = sub_group_reduce_add(acc2);
  acc3 = sub_group_reduce_add(acc3);

  if (tid == 0) {
    STORE(out, n0 + 0, acc0);
    STORE(out, n0 + 1, acc1);
    STORE(out, n0 + 2, acc2);
    STORE(out, n0 + 3, acc3);
  }
}
#endif // ENABLE_SUBGROUPS

// ───── Step #9: image2d_t variant of gemv_m1_k1024_no4 ─────
// Reads W via texture cache (image2d_t with CL_RGBA + CL_HALF_FLOAT).
// Caller wraps a [N, K=1024] fp16 buffer as an image with width=K/4=256
// pixels and height=N. SmolLM2 measured ~1.4× per-call from this swap.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k1024_no4_img(
    __global const storage_t* x,        // [1024]
    __read_only image2d_t W_img,        // CL_RGBA+HALF, width=256, height=N
    __global storage_t* out,            // [N]
    const int N) {
  __local float ls[WG_SIZE * 4];
  const int wg  = get_group_id(0);
  const int tid = get_local_id(0);
  const int n0  = wg * 4;
  if (n0 >= N) return;

  const sampler_t s_smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

#ifdef USE_FP16
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int off  = j * (WG_SIZE * 4) + tid * 4;  // element offset into x (& column*4 in W)
    const int kpix = off / 4;                       // RGBA pixel column in W_img
    float4 xv = vload_half4(0, x + off);
    float4 w0 = read_imagef(W_img, s_smp, (int2)(kpix, n0 + 0));
    float4 w1 = read_imagef(W_img, s_smp, (int2)(kpix, n0 + 1));
    float4 w2 = read_imagef(W_img, s_smp, (int2)(kpix, n0 + 2));
    float4 w3 = read_imagef(W_img, s_smp, (int2)(kpix, n0 + 3));
    acc0 += dot(xv, w0);
    acc1 += dot(xv, w1);
    acc2 += dot(xv, w2);
    acc3 += dot(xv, w3);
  }
#else
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int off  = j * (WG_SIZE * 4) + tid * 4;
    const int kpix = off / 4;
    float4 xv = vload4(0, x + off);
    float4 w0 = read_imagef(W_img, s_smp, (int2)(kpix, n0 + 0));
    float4 w1 = read_imagef(W_img, s_smp, (int2)(kpix, n0 + 1));
    float4 w2 = read_imagef(W_img, s_smp, (int2)(kpix, n0 + 2));
    float4 w3 = read_imagef(W_img, s_smp, (int2)(kpix, n0 + 3));
    acc0 += dot(xv, w0);
    acc1 += dot(xv, w1);
    acc2 += dot(xv, w2);
    acc3 += dot(xv, w3);
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

// ───── Step #4: Fused QKV GEMV M=1, K=1024 ─────
// Single-dispatch replacement for 3 separate gemv_m1_k1024_no4 calls
// (q_proj, k_proj, v_proj). Workgroup id selects which output matrix:
//   wg in [0,         Nq/4)        → Q output (4 outputs per WG)
//   wg in [Nq/4,      Nq/4 + Nkv/4)→ K output
//   wg in [Nq/4+Nkv/4, Nq/4 + 2*Nkv/4) → V output
// Same K=1024 hard-unrolled inner loop as gemv_m1_k1024_no4. Reduces 84
// dispatches per token (28 layers × 3) to 28 dispatches.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fused_qkv_gemv_m1_k1024_no4(
    __global const storage_t* x,    // [1024]
    __global const storage_t* Wq,   // [Nq, 1024]
    __global const storage_t* Wk,   // [Nkv, 1024]
    __global const storage_t* Wv,   // [Nkv, 1024]
    __global storage_t* q_out,      // [Nq]
    __global storage_t* k_out,      // [Nkv]
    __global storage_t* v_out,      // [Nkv]
    const int Nq,
    const int Nkv) {
  __local float ls[WG_SIZE * 4];
  const int wg  = get_group_id(0);
  const int tid = get_local_id(0);
  const int wgs_q = Nq / 4;
  const int wgs_k = Nkv / 4;

  __global const storage_t* W;
  __global storage_t* out_buf;
  int n0;
  if (wg < wgs_q) {
    W = Wq; out_buf = q_out; n0 = wg * 4;
  } else if (wg < wgs_q + wgs_k) {
    W = Wk; out_buf = k_out; n0 = (wg - wgs_q) * 4;
  } else if (wg < wgs_q + 2 * wgs_k) {
    W = Wv; out_buf = v_out; n0 = (wg - wgs_q - wgs_k) * 4;
  } else {
    return;
  }

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

#ifdef USE_FP16
  const int base0 = (n0 + 0) * 1024;
  const int base1 = (n0 + 1) * 1024;
  const int base2 = (n0 + 2) * 1024;
  const int base3 = (n0 + 3) * 1024;

  #pragma unroll
  for (int j = 0; j < 4; ++j) {
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
  for (int j = 0; j < 16; ++j) {
    const int k = j * WG_SIZE + tid;
    float xv = LOAD(x, k);
    acc0 += xv * LOAD(W, (n0+0)*1024 + k);
    acc1 += xv * LOAD(W, (n0+1)*1024 + k);
    acc2 += xv * LOAD(W, (n0+2)*1024 + k);
    acc3 += xv * LOAD(W, (n0+3)*1024 + k);
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
    STORE(out_buf, n0 + tid, ls[tid * WG_SIZE]);
  }
}

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_no4_coalesced(
    __global const storage_t* x,    // [K]
    __global const storage_t* W,    // [N, K] row-major (PyTorch nn.Linear weight)
    __global storage_t* out,        // [N]
    const int K,
    const int N) {
  __local float ls[WG_SIZE * 4];
  const int wg  = get_group_id(0);
  const int tid = get_local_id(0);
  const int n0  = wg * 4;
  if (n0 >= N) return;

  const int kpv = K / (WG_SIZE * 4);  // K / 256 iterations
  const int base0 = (n0 + 0) * K;
  const int base1 = (n0 + 1) * K;
  const int base2 = (n0 + 2) * K;
  const int base3 = (n0 + 3) * K;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

#ifdef USE_FP16
  for (int j = 0; j < kpv; ++j) {
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
  for (int j = 0; j < kpv * 4; ++j) {
    const int k = j * WG_SIZE + tid;
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
