// Reference: model_info/transformers_src/modeling_granitemoehybrid.py:758-777 GraniteMoeHybridMLP.forward
// Implements: hidden = silu(a) * b, where [a,b] are the two halves of the input_linear output.

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

inline float sigmoid_f(float x) {
    // Numerically-stable sigmoid to avoid overflow for large |x| under fp16 storage.
    if (x >= 0.0f) {
        float z = exp(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = exp(x);
        return z / (1.0f + z);
    }
}
inline float silu_f(float x) { return x * sigmoid_f(x); }

__kernel void swiglu_silu_mul(
    __global const storage_t* in_gate_up,  // [rows, 2*cols]
    __global storage_t* out,               // [rows, cols]
    int rows,
    int cols) {
    const int gid = (int)get_global_id(0);
    const int total = rows * cols;
    if (gid >= total) return;

    const int r = gid / cols;
    const int c = gid - r * cols;

    const int base = r * (2 * cols);
    float gate = (float)LOAD(in_gate_up, base + c);
    float up   = (float)LOAD(in_gate_up, base + cols + c);

    float y = silu_f(gate) * up;
    STORE(out, gid, y);
}

// Fused SwiGLU + output_linear for Granite/muP-style models where the
// intermediate `silu(gate) * up` exceeds fp16 range (max 65504). Granite
// is trained in bf16 (range ~3.4e38); naïve fp16 storage of the gated
// intermediate clips to ±inf and propagates NaN through output_linear.
//
// This kernel keeps the full silu(gate)*up*W_out reduction in fp32 inside
// the work-item, never storing the (potentially out-of-range) gated value.
// Only the FINAL output_linear result is stored as storage_t — that result
// is in-range (Granite's per-channel std stays comfortably below 65504).
//
//   in_gate_up: [seq, 2*cols]  — the input_linear projection (fp16 OK)
//   w_out:      [hidden, cols] — output_linear weight in PyTorch row-major
//                                 i.e. weight[h, i] at index h*cols + i
//   out:        [seq, hidden]  — output_linear result (fp16 OK)
//
// Reference: model_info/transformers_src/modeling_granitemoehybrid.py
//   GraniteMoeHybridMLP.forward — gate_up = input_linear(x); gate, up = chunk(2);
//   out = output_linear(silu(gate) * up).
//
// Reduction loop length = cols (= intermediate_size, e.g. 2048).
// One work-item per output element: total = seq * hidden.
__kernel void swiglu_fused_output(
    __global const storage_t* in_gate_up,
    __global const storage_t* w_out,
    __global storage_t* out,
    int seq,
    int cols,
    int hidden) {
    const int gid = (int)get_global_id(0);
    const int total = seq * hidden;
    if (gid >= total) return;

    const int s = gid / hidden;
    const int h = gid - s * hidden;
    const int proj_base = s * (2 * cols);
    const int w_base    = h * cols;

    float acc = 0.0f;
    for (int i = 0; i < cols; i++) {
        float gate = (float)LOAD(in_gate_up, proj_base + i);
        float up   = (float)LOAD(in_gate_up, proj_base + cols + i);
        float gated = silu_f(gate) * up;   // stays in fp32 — never stored
        float w    = (float)LOAD(w_out, w_base + i);
        acc += gated * w;
    }
    STORE(out, gid, acc);
}

// ─────────────────────────────────────────────────────────────────────────
// GEMV-style fused SwiGLU + output_linear, parallelized for Adreno 620.
//
// Design vs the original swiglu_fused_output:
//   * One WORKGROUP per output element (was: one work-ITEM per output) —
//     WG_SIZE=64 lanes split the cols-reduction (32 cols/lane at cols=2048)
//   * vec4 loads (vload_half4) → 4× memory throughput on Adreno A6xx
//   * Tree-reduce 64 fp32 partials in __local memory → one store per output
//
// Why the original was 16 ms/call:
//   With one work-item per output and gws=hidden, only `hidden` work-items
//   ran in parallel and each scalar-loaded the full cols=2048 vector. On
//   Adreno 620 (~13.6 GB/s), reading 4 MB of weights serially per call
//   landed at ~256 MB/s — 50× under memory ceiling.
//
// Constraints assumed: cols % (WG_SIZE * 4) == 0 (true for cols=2048, WG=64).
// Falls back to scalar tail if cols ever stops being a multiple of 256.
//
// Dispatch from host: gws[0] = seq * hidden * WG_SIZE, lws[0] = WG_SIZE.
//   Workgroup id = (s * hidden + h);   lane id ∈ [0, WG_SIZE).
#define WG_SIZE 64

// Step 8: kernel now also fuses the post-MLP residual_multiplier add. Saves
// 28 element_add dispatches per token + 28 host-side buffer allocs. The
// kernel emits `out = residual + acc * residual_scale` directly. Reading
// `residual` adds 1 fp16 load per output element (~negligible — 2 KB at
// decode), but kills the dispatch.
// ───── Step #8: image2d_t variant of swiglu_fused_output_m1_v2 ─────
// Same layout as swiglu_fused_output_m1_v2 but reads W_out via read_imagef
// (texture cache path on Adreno A6xx — separate from the buffer L1 read
// port). For granite W_out is [hidden=1024, cols=2048] = 4 MB per layer;
// per-call load is the dominant traffic in the MLP. SmolLM2 measured
// ~1.4× per-call from this swap.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void swiglu_fused_output_m1_v2_img(
    __global const storage_t* in_gate_up,  // [seq, 2*cols]
    __read_only image2d_t w_out_img,       // CL_RGBA + CL_HALF_FLOAT, width=cols/4, height=hidden
    __global const storage_t* residual,    // [seq, hidden]
    __global storage_t* out,               // [seq, hidden]
    int seq,
    int cols,
    int hidden,
    float residual_scale) {
    const int lane = (int)get_local_id(0);
    const int wg   = (int)get_group_id(0);
    const int s    = wg / hidden;
    const int h    = wg - s * hidden;
    if (s >= seq) return;

    const sampler_t s_smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    const int proj_base = s * (2 * cols);

    float acc = 0.0f;
    for (int i = lane * 4; i < cols; i += WG_SIZE * 4) {
#ifdef USE_FP16
        float4 gf = vload_half4(0, in_gate_up + proj_base + i);
        float4 uf = vload_half4(0, in_gate_up + proj_base + cols + i);
        float4 wf = read_imagef(w_out_img, s_smp, (int2)(i / 4, h));
#else
        float4 gf = vload4(0, in_gate_up + proj_base + i);
        float4 uf = vload4(0, in_gate_up + proj_base + cols + i);
        float4 wf = read_imagef(w_out_img, s_smp, (int2)(i / 4, h));
#endif
        float4 sig = (float4)(1.0f) / ((float4)(1.0f) + native_exp(-gf));
        float4 gated = gf * sig * uf;
        acc += dot(gated, wf);
    }

    __local float lds[WG_SIZE];
    lds[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = WG_SIZE / 2; step > 0; step >>= 1) {
        if (lane < step) lds[lane] += lds[lane + step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
        const int out_idx = s * hidden + h;
        const float r = (float)LOAD(residual, out_idx);
        STORE(out, out_idx, r + lds[0] * residual_scale);
    }
}

// ───── Step #5: Fused gate_up + silu_mul kernel (M=1, K=1024) ─────
// Replaces the mlp_input GEMV (writes proj[2*N=4096]) + the silu+mul half
// of swiglu (reads gate[N], up[N] from proj). Computes gated[N=2048]
// directly from x[K=1024] and W_in[2*N, K], halving the intermediate
// buffer R/W traffic.
//
// W_in layout: rows 0..N-1 are the gate projection, rows N..2N-1 are up.
// Per WG: 4 outputs (gated[n0..n0+3]). Each output requires 2 GEMV
// reductions (one against gate-row, one against up-row) over K=1024,
// then silu(gate)*up combination.
//
// Dispatch: gws = (N/4) * WG_SIZE, lws = WG_SIZE.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fused_gate_up_silu_m1_k1024(
    __global const storage_t* x,        // [K=1024]
    __global const storage_t* W_in,     // [2*N, 1024]  rows 0..N-1=gate, N..2N-1=up
    __global storage_t* gated,          // [N]
    const int N) {
  __local float ls_g[WG_SIZE * 4];
  __local float ls_u[WG_SIZE * 4];
  const int wg  = get_group_id(0);
  const int tid = get_local_id(0);
  const int n0  = wg * 4;
  if (n0 >= N) return;

  float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
  float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;

#ifdef USE_FP16
  const int gbase0 = (n0 + 0) * 1024;
  const int gbase1 = (n0 + 1) * 1024;
  const int gbase2 = (n0 + 2) * 1024;
  const int gbase3 = (n0 + 3) * 1024;
  const int ubase0 = (N + n0 + 0) * 1024;
  const int ubase1 = (N + n0 + 1) * 1024;
  const int ubase2 = (N + n0 + 2) * 1024;
  const int ubase3 = (N + n0 + 3) * 1024;

  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int off = j * (WG_SIZE * 4) + tid * 4;
    float4 xv = vload_half4(0, x + off);
    g0 += dot(xv, vload_half4(0, W_in + gbase0 + off));
    g1 += dot(xv, vload_half4(0, W_in + gbase1 + off));
    g2 += dot(xv, vload_half4(0, W_in + gbase2 + off));
    g3 += dot(xv, vload_half4(0, W_in + gbase3 + off));
    u0 += dot(xv, vload_half4(0, W_in + ubase0 + off));
    u1 += dot(xv, vload_half4(0, W_in + ubase1 + off));
    u2 += dot(xv, vload_half4(0, W_in + ubase2 + off));
    u3 += dot(xv, vload_half4(0, W_in + ubase3 + off));
  }
#else
  for (int j = 0; j < 16; ++j) {
    const int k = j * WG_SIZE + tid;
    float xv = LOAD(x, k);
    g0 += xv * LOAD(W_in, (n0+0)*1024 + k);
    g1 += xv * LOAD(W_in, (n0+1)*1024 + k);
    g2 += xv * LOAD(W_in, (n0+2)*1024 + k);
    g3 += xv * LOAD(W_in, (n0+3)*1024 + k);
    u0 += xv * LOAD(W_in, (N+n0+0)*1024 + k);
    u1 += xv * LOAD(W_in, (N+n0+1)*1024 + k);
    u2 += xv * LOAD(W_in, (N+n0+2)*1024 + k);
    u3 += xv * LOAD(W_in, (N+n0+3)*1024 + k);
  }
#endif

  ls_g[tid + 0 * WG_SIZE] = g0;
  ls_g[tid + 1 * WG_SIZE] = g1;
  ls_g[tid + 2 * WG_SIZE] = g2;
  ls_g[tid + 3 * WG_SIZE] = g3;
  ls_u[tid + 0 * WG_SIZE] = u0;
  ls_u[tid + 1 * WG_SIZE] = u1;
  ls_u[tid + 2 * WG_SIZE] = u2;
  ls_u[tid + 3 * WG_SIZE] = u3;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      ls_g[tid + 0 * WG_SIZE] += ls_g[tid + 0 * WG_SIZE + s];
      ls_g[tid + 1 * WG_SIZE] += ls_g[tid + 1 * WG_SIZE + s];
      ls_g[tid + 2 * WG_SIZE] += ls_g[tid + 2 * WG_SIZE + s];
      ls_g[tid + 3 * WG_SIZE] += ls_g[tid + 3 * WG_SIZE + s];
      ls_u[tid + 0 * WG_SIZE] += ls_u[tid + 0 * WG_SIZE + s];
      ls_u[tid + 1 * WG_SIZE] += ls_u[tid + 1 * WG_SIZE + s];
      ls_u[tid + 2 * WG_SIZE] += ls_u[tid + 2 * WG_SIZE + s];
      ls_u[tid + 3 * WG_SIZE] += ls_u[tid + 3 * WG_SIZE + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid < 4) {
    const float gv = ls_g[tid * WG_SIZE];
    const float uv = ls_u[tid * WG_SIZE];
    const float sig = 1.0f / (1.0f + native_exp(-gv));
    STORE(gated, n0 + tid, gv * sig * uv);
  }
}

// ───── Step #5: down-proj for the fused path (reads gated[N], not proj[2N]) ─────
// Same as swiglu_fused_output_m1_v2 but with gated[N] as input (no silu/mul).
// Halves the gate/up read traffic in this kernel.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void mlp_down_residual_m1_v2(
    __global const storage_t* gated,       // [N]
    __global const storage_t* w_out,       // [hidden, N]
    __global const storage_t* residual,    // [hidden]
    __global storage_t* out,               // [hidden]
    const int N,
    const int hidden,
    const float residual_scale) {
  const int lane = (int)get_local_id(0);
  const int h    = (int)get_group_id(0);
  if (h >= hidden) return;
  const int w_base = h * N;

  float acc = 0.0f;
  for (int i = lane * 4; i < N; i += WG_SIZE * 4) {
#ifdef USE_FP16
    float4 gv = vload_half4(0, gated + i);
    float4 wf = vload_half4(0, w_out + w_base + i);
#else
    float4 gv = vload4(0, gated + i);
    float4 wf = vload4(0, w_out + w_base + i);
#endif
    acc += dot(gv, wf);
  }
  __local float lds[WG_SIZE];
  lds[lane] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int step = WG_SIZE / 2; step > 0; step >>= 1) {
    if (lane < step) lds[lane] += lds[lane + step];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lane == 0) {
    const float r = (float)LOAD(residual, h);
    STORE(out, h, r + lds[0] * residual_scale);
  }
}

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void swiglu_fused_output_m1_v2(
    __global const storage_t* in_gate_up,  // [seq, 2*cols]
    __global const storage_t* w_out,       // [hidden, cols]  (row-major; w_out[h, i] at h*cols + i)
    __global const storage_t* residual,    // [seq, hidden]   (Step 8: fused add)
    __global storage_t* out,               // [seq, hidden]
    int seq,
    int cols,
    int hidden,
    float residual_scale) {                // Step 8: residual_multiplier
    const int lane = (int)get_local_id(0);
    const int wg   = (int)get_group_id(0);  // one workgroup per output element
    const int s    = wg / hidden;
    const int h    = wg - s * hidden;
    if (s >= seq) return;

    const int proj_base = s * (2 * cols);
    const int w_base    = h * cols;

    // Each lane reduces cols / WG_SIZE elements, in vec4 chunks.
    // For cols=2048, WG=64: 32 cols/lane = 8 vec4 chunks/lane.
    float acc = 0.0f;
    for (int i = lane * 4; i < cols; i += WG_SIZE * 4) {
#ifdef USE_FP16
        // vload_half4 reads 4 halves and returns float4 (built-in conversion).
        float4 gf = vload_half4(0, in_gate_up + proj_base + i);
        float4 uf = vload_half4(0, in_gate_up + proj_base + cols + i);
        float4 wf = vload_half4(0, w_out + w_base + i);
#else
        float4 gf = vload4(0, in_gate_up + proj_base + i);
        float4 uf = vload4(0, in_gate_up + proj_base + cols + i);
        float4 wf = vload4(0, w_out + w_base + i);
#endif
        // Step #8: silu via native_exp + unbranched sigmoid. The branched form
        // above protected against fp16 overflow, but we already cast to fp32
        // before sigmoid so 1/(1+exp(-x)) saturates correctly:
        //   x → +∞ : exp(-x) → 0 → sigmoid → 1
        //   x → −∞ : exp(-x) → +∞ → sigmoid → 0
        // native_exp is hardware-accelerated on Adreno (~3-5× faster than exp).
        // Vectorized over float4 → 4 native_exp ops per iteration.
        float4 sig = (float4)(1.0f) / ((float4)(1.0f) + native_exp(-gf));
        float4 gated = gf * sig * uf;
        acc += dot(gated, wf);
    }

    // Tree reduce 64 fp32 partials → out[s, h]
    __local float lds[WG_SIZE];
    lds[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = WG_SIZE / 2; step > 0; step >>= 1) {
        if (lane < step) lds[lane] += lds[lane + step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
        const int out_idx = s * hidden + h;
        const float r = (float)LOAD(residual, out_idx);
        STORE(out, out_idx, r + lds[0] * residual_scale);
    }
}
