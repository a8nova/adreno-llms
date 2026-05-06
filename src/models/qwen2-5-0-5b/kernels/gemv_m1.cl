// Reference: model_info/transformers_src/modeling_qwen2.py (Qwen2Attention/Qwen2MLP use nn.Linear projections; this file implements the decode-path GEMV equivalent of those linear layers)
// Custom GEMV for M=1 decode path (single-row matvec).
// Replaces CLBlast HGemm M=1 at: q_proj, k_proj, v_proj, o_proj,
// gate_proj, up_proj, down_proj, lm_head.
//
// Cooperative WG=64, vec4 fp16 inner loop, fp32 tree-reduce, fp32 acc.
// Coalesced wave-stride access pattern — each thread iteration `j`
// reads bytes (j*WG*4 + tid*4) so all 64 threads in iteration `j`
// cover contiguous bytes 0..511 (Mamba Step 7c lesson, +1.39×).
//
// Compute: out[n] = sum_k x[k] * W[n, k]
//   W is PyTorch nn.Linear weight [N, K] (row-major, row n = W[n*K..(n+1)*K]).
//   x is [K] (one row).
//   K must be divisible by (WG_SIZE * 4) — caller predicate guards this.
//
// Dispatch:
//   gws = N * WG_SIZE, lws = WG_SIZE → 1 WG per output row.
//
// Naming: kernels are specialized by K so the compiler hard-unrolls the
// outer K-step count. K=896 → 896/(64*4) = 3.5 — covered by gemv_m1_k896.
// K=4864 → 4864/(64*4) = 19 — gemv_m1_k4864.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
#else
  typedef float storage_t;
#endif

#define WG_SIZE 64

// Generic K kernel: K must be divisible by (WG_SIZE * 4) = 256.
// Coalesced wave-stride: thread tid in iteration j reads bytes (j*256 + tid*4).
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_generic(
    __global const storage_t* x,
    __global const storage_t* W,
    __global storage_t* out,
    const int N,
    const int K) {
  const int n   = (int)get_group_id(0);
  const int tid = (int)get_local_id(0);
  if (n >= N) return;

  const int W_base = n * K;
  __local float partial[WG_SIZE];

#ifdef USE_FP16
  const int kp = K / (WG_SIZE * 4);  // vec4 chunks per thread
  float acc = 0.0f;
  __global const half* xh = (__global const half*)x;
  __global const half* Wh = (__global const half*)W + W_base;

  for (int j = 0; j < kp; ++j) {
    const int off = j * (WG_SIZE * 4) + tid * 4;
    float4 xv = vload_half4(0, xh + off);
    float4 wv = vload_half4(0, Wh + off);
    acc += dot(xv, wv);
  }
  partial[tid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) partial[tid] += partial[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    vstore_half(partial[0], 0, (__global half*)out + n);
  }
#else
  const int kp = K / WG_SIZE;
  float acc = 0.0f;
  for (int j = 0; j < kp; ++j) {
    const int off = j * WG_SIZE + tid;
    acc += x[off] * W[W_base + off];
  }
  partial[tid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) partial[tid] += partial[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    out[n] = partial[0];
  }
#endif
}

// ─────────────────────────────────────────────────────────────────────
// STREAM benchmark kernels — pure-bandwidth probe with no real compute.
// Used by --bw-probe to measure the practical streaming-read ceiling
// from BOTH the buffer cache (gemv_stream_buf) and the texture cache
// (gemv_stream_img). The DCE guard ("if acc.x > 1e30f write...") makes
// the read live to the compiler without ever firing at runtime.
//
// Methodology: run with N bytes much larger than L2 (64 KB on Adreno
// 620) to ensure every read goes to DRAM, take min over multiple runs
// to filter out frequency-scaling spikes, divide bytes-read by elapsed
// to get achieved GB/s. THIS IS the practical ceiling for any kernel
// using the same cache path.
// ─────────────────────────────────────────────────────────────────────

#ifdef USE_FP16

__kernel
__attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_stream_buf(
    __global const half* src,
    __global float* dce_sink,
    const int iters_per_thread) {
  // Coalesced wave-stride: thread tid in WG g, iteration j reads bytes
  // ((g * WG_SIZE * iters_per_thread) + j * WG_SIZE + tid) * 8, so the
  // 64 threads of a wave-iteration cover 64 * 8 = 512 contiguous bytes
  // (8 cache lines). This is the same pattern as the live GEMV kernels
  // and is what Adreno's L1/L2 caches were tuned around.
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
  // Same wave-stride access pattern as the live image GEMV kernels:
  // each WG reads `rows_per_wg` consecutive rows; the 64 threads of a
  // wave-iteration j cover pixels (j*64 .. j*64+63) of the same row.
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

#endif

// Tiny kernel for the recordable-queues probe — does almost nothing,
// so per-launch CPU overhead dominates kernel runtime, exposing any
// bookkeeping savings the recording API provides.
__kernel void probe_noop(__global int* counter, const int incr) {
  if (get_global_id(0) == 0) atomic_add(counter, incr);
}

// kv_write — copies a KV_DIM-wide row from src into cache[start_pos*KV_DIM..].
// Replaces clEnqueueCopyBuffer in attention.cpp so KV writes can be recorded
// (only clEnqueueNDRangeKernel is recordable per Snapdragon Programming Guide
// §9.1.3). Each thread writes one fp16 row element.
// counter[0]=start_pos, counter[1]=seq_k — persistent buffer updated per step
// via clEnqueueWriteBuffer on live_q BEFORE replay. Buffer address is stable
// (baked into recording), contents change between replays.
__kernel void kv_write(
    __global const half* src,
    __global       half* cache,
    __global const int*  counter,  // [0]=start_pos, [1]=seq_k
    const int kv_dim) {
  const int gid = (int)get_global_id(0);
  if (gid >= kv_dim) return;
  const int start_pos = counter[0];
  cache[start_pos * kv_dim + gid] = src[gid];
}

// ─────────────────────────────────────────────────────────────────────
// argmax_partial / argmax_finalize — 2-pass argmax over a [N] fp16 vector.
// 152K elements at 1-WG-only is bottlenecked by the 64 threads doing 2375
// sequential comparisons each. Spread to NUM_PARTIALS WGs first, then
// reduce those NUM_PARTIALS partials in a single final WG.
//
// Pass 1: gws = NUM_PARTIALS * WG_SIZE, lws = WG_SIZE.
//         Each WG computes argmax over its slice → writes (val, idx) to
//         partial_vals[wg], partial_idxs[wg].
// Pass 2: gws = WG_SIZE, lws = WG_SIZE. 1 WG reduces NUM_PARTIALS entries.
//
// Final output: out_idx[0] = global argmax index.
// ─────────────────────────────────────────────────────────────────────
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void argmax_partial(
    __global const storage_t* x,
    __global float* partial_vals,
    __global int*   partial_idxs,
    const int N,
    const int num_partials) {
  const int wg  = (int)get_group_id(0);
  const int tid = (int)get_local_id(0);
  __local float lvals[WG_SIZE];
  __local int   lidxs[WG_SIZE];

  // Each WG owns the slice [wg*chunk, (wg+1)*chunk) (last WG handles the tail).
  const int chunk = (N + num_partials - 1) / num_partials;
  const int start = wg * chunk;
  const int end   = (start + chunk < N) ? (start + chunk) : N;

  float best_v = -3.402823466e+38f;
  int   best_i = (start < N) ? start : 0;
#ifdef USE_FP16
  __global const half* xh = (__global const half*)x;
  for (int i = start + tid; i < end; i += WG_SIZE) {
    float v = vload_half(i, xh);
    if (v > best_v) { best_v = v; best_i = i; }
  }
#else
  for (int i = start + tid; i < end; i += WG_SIZE) {
    float v = x[i];
    if (v > best_v) { best_v = v; best_i = i; }
  }
#endif
  lvals[tid] = best_v;
  lidxs[tid] = best_i;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      if (lvals[tid + s] > lvals[tid]) {
        lvals[tid] = lvals[tid + s];
        lidxs[tid] = lidxs[tid + s];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    partial_vals[wg] = lvals[0];
    partial_idxs[wg] = lidxs[0];
  }
}

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void argmax_finalize(
    __global const float* partial_vals,
    __global const int*   partial_idxs,
    __global int* out_idx,
    const int num_partials) {
  const int tid = (int)get_local_id(0);
  __local float lvals[WG_SIZE];
  __local int   lidxs[WG_SIZE];

  float best_v = -3.402823466e+38f;
  int   best_i = 0;
  for (int i = tid; i < num_partials; i += WG_SIZE) {
    float v = partial_vals[i];
    if (v > best_v) { best_v = v; best_i = partial_idxs[i]; }
  }
  lvals[tid] = best_v;
  lidxs[tid] = best_i;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      if (lvals[tid + s] > lvals[tid]) {
        lvals[tid] = lvals[tid + s];
        lidxs[tid] = lidxs[tid + s];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    out_idx[0] = lidxs[0];
  }
}

// K=896 (Qwen hidden size) — K is NOT divisible by (WG*4)=256.
// 896 = 3*256 + 128 → 3 full vec4 iters across the wave + 32-element tail.
// In the tail iter only the first 32 threads (tid < 32) do useful work
// (reading bytes (3*256 + tid*4) covers bytes 768..895 = the last 128 fp16).
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k896(
    __global const storage_t* x,
    __global const storage_t* W,
    __global storage_t* out,
    const int N) {
  const int K = 896;
  const int n   = (int)get_group_id(0);
  const int tid = (int)get_local_id(0);
  if (n >= N) return;

  const int W_base = n * K;
  __local float partial[WG_SIZE];

#ifdef USE_FP16
  __global const half* xh = (__global const half*)x;
  __global const half* Wh = (__global const half*)W + W_base;
  float acc = 0.0f;

  // 3 full waves of vec4 (3 * 256 = 768 fp16 covered).
  #pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int off = j * (WG_SIZE * 4) + tid * 4;
    float4 xv = vload_half4(0, xh + off);
    float4 wv = vload_half4(0, Wh + off);
    acc += dot(xv, wv);
  }
  // Tail: 128 fp16 left. 128 = 32 vec4 → first 32 threads do 1 vec4 each.
  if (tid < 32) {
    const int off = 3 * (WG_SIZE * 4) + tid * 4;
    float4 xv = vload_half4(0, xh + off);
    float4 wv = vload_half4(0, Wh + off);
    acc += dot(xv, wv);
  }
  partial[tid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) partial[tid] += partial[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    vstore_half(partial[0], 0, (__global half*)out + n);
  }
#else
  float acc = 0.0f;
  for (int k = tid; k < K; k += WG_SIZE) {
    acc += x[k] * W[W_base + k];
  }
  partial[tid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) partial[tid] += partial[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    out[n] = partial[0];
  }
#endif
}

// K=4864 single-output stub. Hot path always uses gemv_m1_k4864_no4_img;
// this minimal body satisfies clCreateKernel without adding a large unrolled
// kernel that disturbs the Adreno compiler's register allocation for the hot
// image kernels. Never actually dispatched (fallback_buffer guards against it).
__kernel void gemv_m1_k4864(
    __global const storage_t* x,
    __global const storage_t* W,
    __global storage_t* out,
    const int N) { (void)x; (void)W; (void)out; (void)N; }

// ─────────────────────────────────────────────────────────────────────
// Multi-output-per-WG variants ("no4"): 4 outputs per WG.
// 4× fewer workgroups, x is loaded once and reused across 4 dot-product
// chains, 4 independent fp32 accumulators per thread → register-level
// parallelism. Mamba P2-4 lesson: +1.21× over single-output cooperative.
// Caller must guarantee N % 4 == 0.
// ─────────────────────────────────────────────────────────────────────

// K=896, 4 outputs per WG. Dispatch: gws = (N/4) * WG_SIZE, lws = WG_SIZE.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k896_no4(
    __global const storage_t* x,
    __global const storage_t* W,
    __global storage_t* out,
    const int N) {
  const int K = 896;
  const int n_base = (int)get_group_id(0) * 4;
  const int tid    = (int)get_local_id(0);
  if (n_base >= N) return;

  __local float partial[4][WG_SIZE];

#ifdef USE_FP16
  __global const half* xh = (__global const half*)x;
  __global const half* Wh = (__global const half*)W;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int off = j * (WG_SIZE * 4) + tid * 4;
    float4 xv = vload_half4(0, xh + off);
    float4 w0 = vload_half4(0, Wh + (n_base + 0) * K + off);
    float4 w1 = vload_half4(0, Wh + (n_base + 1) * K + off);
    float4 w2 = vload_half4(0, Wh + (n_base + 2) * K + off);
    float4 w3 = vload_half4(0, Wh + (n_base + 3) * K + off);
    acc0 += dot(xv, w0);
    acc1 += dot(xv, w1);
    acc2 += dot(xv, w2);
    acc3 += dot(xv, w3);
  }
  // Tail: 128 fp16 left → first 32 threads do 1 vec4 each.
  if (tid < 32) {
    const int off = 3 * (WG_SIZE * 4) + tid * 4;
    float4 xv = vload_half4(0, xh + off);
    float4 w0 = vload_half4(0, Wh + (n_base + 0) * K + off);
    float4 w1 = vload_half4(0, Wh + (n_base + 1) * K + off);
    float4 w2 = vload_half4(0, Wh + (n_base + 2) * K + off);
    float4 w3 = vload_half4(0, Wh + (n_base + 3) * K + off);
    acc0 += dot(xv, w0);
    acc1 += dot(xv, w1);
    acc2 += dot(xv, w2);
    acc3 += dot(xv, w3);
  }
  partial[0][tid] = acc0;
  partial[1][tid] = acc1;
  partial[2][tid] = acc2;
  partial[3][tid] = acc3;
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
    __global half* oh = (__global half*)out;
    vstore_half(partial[0][0], 0, oh + n_base + 0);
    vstore_half(partial[1][0], 0, oh + n_base + 1);
    vstore_half(partial[2][0], 0, oh + n_base + 2);
    vstore_half(partial[3][0], 0, oh + n_base + 3);
  }
#else
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int k = tid; k < K; k += WG_SIZE) {
    float xk = x[k];
    acc[0] += xk * W[(n_base + 0) * K + k];
    acc[1] += xk * W[(n_base + 1) * K + k];
    acc[2] += xk * W[(n_base + 2) * K + k];
    acc[3] += xk * W[(n_base + 3) * K + k];
  }
  partial[0][tid] = acc[0];
  partial[1][tid] = acc[1];
  partial[2][tid] = acc[2];
  partial[3][tid] = acc[3];
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
    out[n_base + 0] = partial[0][0];
    out[n_base + 1] = partial[1][0];
    out[n_base + 2] = partial[2][0];
    out[n_base + 3] = partial[3][0];
  }
#endif
}

// K=4864, 4 outputs per WG.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k4864_no4(
    __global const storage_t* x,
    __global const storage_t* W,
    __global storage_t* out,
    const int N) {
  const int K = 4864;
  const int n_base = (int)get_group_id(0) * 4;
  const int tid    = (int)get_local_id(0);
  if (n_base >= N) return;

  __local float partial[4][WG_SIZE];

#ifdef USE_FP16
  __global const half* xh = (__global const half*)x;
  __global const half* Wh = (__global const half*)W;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 19; ++j) {
    const int off = j * (WG_SIZE * 4) + tid * 4;
    float4 xv = vload_half4(0, xh + off);
    float4 w0 = vload_half4(0, Wh + (n_base + 0) * K + off);
    float4 w1 = vload_half4(0, Wh + (n_base + 1) * K + off);
    float4 w2 = vload_half4(0, Wh + (n_base + 2) * K + off);
    float4 w3 = vload_half4(0, Wh + (n_base + 3) * K + off);
    acc0 += dot(xv, w0);
    acc1 += dot(xv, w1);
    acc2 += dot(xv, w2);
    acc3 += dot(xv, w3);
  }
  partial[0][tid] = acc0;
  partial[1][tid] = acc1;
  partial[2][tid] = acc2;
  partial[3][tid] = acc3;
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
    __global half* oh = (__global half*)out;
    vstore_half(partial[0][0], 0, oh + n_base + 0);
    vstore_half(partial[1][0], 0, oh + n_base + 1);
    vstore_half(partial[2][0], 0, oh + n_base + 2);
    vstore_half(partial[3][0], 0, oh + n_base + 3);
  }
#else
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int k = tid; k < K; k += WG_SIZE) {
    float xk = x[k];
    acc[0] += xk * W[(n_base + 0) * K + k];
    acc[1] += xk * W[(n_base + 1) * K + k];
    acc[2] += xk * W[(n_base + 2) * K + k];
    acc[3] += xk * W[(n_base + 3) * K + k];
  }
  partial[0][tid] = acc[0];
  partial[1][tid] = acc[1];
  partial[2][tid] = acc[2];
  partial[3][tid] = acc[3];
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
    out[n_base + 0] = partial[0][0];
    out[n_base + 1] = partial[1][0];
    out[n_base + 2] = partial[2][0];
    out[n_base + 3] = partial[3][0];
  }
#endif
}

// ─────────────────────────────────────────────────────────────────────
// IMAGE-BACKED GEMV variants — read weights through Adreno's texture
// cache instead of the buffer cache. Texture cache on Adreno is a
// dedicated read-only path with ~1.3-1.5× the effective BW of the
// buffer cache for fp16 weights. The single biggest mobile-GPU lever
// to push past the buffer-cache ceiling.
//
// Layout: W [N, K] fp16 wrapped as image2d_t with CL_RGBA / CL_HALF_FLOAT
// (4 fp16 per pixel). image_width = K/4, image_height = N.
// Caller creates the image view lazily via cl_khr_image2d_from_buffer
// (no data copy — same backing memory, different access path).
// Read: read_imageh(W_img, sampler, (int2)(col4, row)) returns half4.
//
// Sampler: CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST
// — integer coords, no boundary handling, no filtering. We always read
// in-bounds because gws is sized to (N/4)*WG_SIZE with each thread
// reading deterministic (col4, row) pairs.
//
// Same coalesced wave-stride pattern as gemv_m1_k896_no4 but column
// index is now in pixels (vec4-unit), not bytes. Each iter j has all
// 64 threads reading consecutive pixels in the same row — the texture
// cache is extremely happy with this access pattern.
// ─────────────────────────────────────────────────────────────────────

__constant sampler_t kImgSampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

#ifdef USE_FP16

// K=896, image2d-backed W, 4 outputs per WG (no4). N % 4 == 0.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k896_no4_img(
    __global const storage_t* x,
    __read_only image2d_t W_img,
    __global storage_t* out,
    const int N) {
  const int K       = 896;
  const int K_PIX   = K / 4;          // 224 vec4-pixels per row of W
  const int n_base  = (int)get_group_id(0) * 4;
  const int tid     = (int)get_local_id(0);
  if (n_base >= N) return;

  __local float partial[4][WG_SIZE];

  __global const half* xh = (__global const half*)x;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  // 3 full waves of vec4 (3 * 64 = 192 pixels covered).
  #pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int x_off = j * (WG_SIZE * 4) + tid * 4;
    const int pix   = j *  WG_SIZE      + tid;       // pixel column index
    float4 xv = vload_half4(0, xh + x_off);
    float4 w0 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }
  // Tail: K_PIX = 224 → already covered? 3*64 = 192, leaving 32 pixels.
  if (tid < 32) {
    const int x_off = 3 * (WG_SIZE * 4) + tid * 4;
    const int pix   = 3 *  WG_SIZE      + tid;
    float4 xv = vload_half4(0, xh + x_off);
    float4 w0 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }
  partial[0][tid] = acc0; partial[1][tid] = acc1;
  partial[2][tid] = acc2; partial[3][tid] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
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

// K=4864, image2d-backed W, 4 outputs per WG.
// K_PIX = 4864/4 = 1216 = 19 * 64 exact.
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k4864_no4_img(
    __global const storage_t* x,
    __read_only image2d_t W_img,
    __global storage_t* out,
    const int N) {
  const int K       = 4864;
  const int n_base  = (int)get_group_id(0) * 4;
  const int tid     = (int)get_local_id(0);
  if (n_base >= N) return;

  __local float partial[4][WG_SIZE];

  __global const half* xh = (__global const half*)x;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 19; ++j) {
    const int x_off = j * (WG_SIZE * 4) + tid * 4;
    const int pix   = j *  WG_SIZE      + tid;
    float4 xv = vload_half4(0, xh + x_off);
    float4 w0 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }
  partial[0][tid] = acc0; partial[1][tid] = acc1;
  partial[2][tid] = acc2; partial[3][tid] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
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

// ─────────────────────────────────────────────────────────────────────
// gemv_m1_kK_no4_img_add — fused projection+residual_add for the M=1
// decode hot path. Used by o_proj (K=896 N=896) and down_proj (K=4864
// N=896). Reads `out[n]` (= residual coming in), adds the GEMV result,
// writes back. Eliminates the separate element_add_inplace launch
// (24 layers × 2 residuals = 48 launches/token saved).
//
// Safety: by the time o_proj fires, hidden has already been read for
// input_layernorm; by the time down_proj fires, post_attn_layernorm has
// already read hidden. So reading hidden in the projection's epilogue is
// race-free — preceding kernels finished under the in-order queue.
//
// Caller dispatches with gws = (N/4) * WG_SIZE, lws = WG_SIZE — same as
// the non-add variant. Pass the residual destination (= hidden) as `out`.
// ─────────────────────────────────────────────────────────────────────

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k896_no4_img_add(
    __global const storage_t* x,
    __read_only image2d_t W_img,
    __global storage_t* out,
    const int N) {
  const int n_base  = (int)get_group_id(0) * 4;
  const int tid     = (int)get_local_id(0);
  if (n_base >= N) return;

  __local float partial[4][WG_SIZE];

  __global const half* xh = (__global const half*)x;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int x_off = j * (WG_SIZE * 4) + tid * 4;
    const int pix   = j *  WG_SIZE      + tid;
    float4 xv = vload_half4(0, xh + x_off);
    float4 w0 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }
  if (tid < 32) {
    const int x_off = 3 * (WG_SIZE * 4) + tid * 4;
    const int pix   = 3 *  WG_SIZE      + tid;
    float4 xv = vload_half4(0, xh + x_off);
    float4 w0 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }
  partial[0][tid] = acc0; partial[1][tid] = acc1;
  partial[2][tid] = acc2; partial[3][tid] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      partial[0][tid] += partial[0][tid + s]; partial[1][tid] += partial[1][tid + s];
      partial[2][tid] += partial[2][tid + s]; partial[3][tid] += partial[3][tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    __global half* oh = (__global half*)out;
    float4 r = vload_half4(0, oh + n_base);
    float4 sum = (float4)(partial[0][0], partial[1][0], partial[2][0], partial[3][0]) + r;
    vstore_half4(sum, 0, oh + n_base);
  }
}

__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void gemv_m1_k4864_no4_img_add(
    __global const storage_t* x,
    __read_only image2d_t W_img,
    __global storage_t* out,
    const int N) {
  const int n_base  = (int)get_group_id(0) * 4;
  const int tid     = (int)get_local_id(0);
  if (n_base >= N) return;

  __local float partial[4][WG_SIZE];

  __global const half* xh = (__global const half*)x;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  #pragma unroll
  for (int j = 0; j < 19; ++j) {
    const int x_off = j * (WG_SIZE * 4) + tid * 4;
    const int pix   = j *  WG_SIZE      + tid;
    float4 xv = vload_half4(0, xh + x_off);
    float4 w0 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 0)));
    float4 w1 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 1)));
    float4 w2 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 2)));
    float4 w3 = convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + 3)));
    acc0 += dot(xv, w0); acc1 += dot(xv, w1);
    acc2 += dot(xv, w2); acc3 += dot(xv, w3);
  }
  partial[0][tid] = acc0; partial[1][tid] = acc1;
  partial[2][tid] = acc2; partial[3][tid] = acc3;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      partial[0][tid] += partial[0][tid + s]; partial[1][tid] += partial[1][tid + s];
      partial[2][tid] += partial[2][tid + s]; partial[3][tid] += partial[3][tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    __global half* oh = (__global half*)out;
    float4 r = vload_half4(0, oh + n_base);
    float4 sum = (float4)(partial[0][0], partial[1][0], partial[2][0], partial[3][0]) + r;
    vstore_half4(sum, 0, oh + n_base);
  }
}

// ─────────────────────────────────────────────────────────────────────
// EXPERIMENTS THAT DIDN'T LAND ON Adreno 620 + driver 2021-11-15:
//
// 1. cl_khr_subgroups (sub_group_reduce_add) — driver advertises the
//    extension but compiler returns "OpenCL 2.0 built-in is not supported".
// 2. cl_qcom_subgroup_shuffle (sub_group_shuffle_xor) — same advertised,
//    same "implicit declaration" rejection by compiler.
// 3. Wave-coherent volatile-local reduction (CUDA-style barrier-free
//    tree-reduce relying on SIMD-lockstep within a 64-lane wave) —
//    measured 7× regression. Adreno's compiler treats `volatile __local`
//    as atomic-grade ordering, generating slow per-access barriers
//    rather than skipping them. SIMT-lockstep is not exposed at the
//    OpenCL language level on this driver.
//
// Net: per-WG reduction stays as standard tree-reduce + barriers.
// Re-test on Adreno 7xx-class drivers (Snapdragon 8 Gen 1+).
// ─────────────────────────────────────────────────────────────────────

#endif  // USE_FP16  (closes the gemv_m1_kK_no4_img block)
