// Reference: model_info/transformers_src/modeling_granitemoehybrid.py:735-752 GraniteMoeHybridRMSNormGated.forward
// Implements RMSNorm with optional gating branch (not used in this dense Granite config).

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

__kernel void rmsnorm_forward(
    __global const storage_t* x,
    __global const storage_t* weight,
    __global storage_t* out,
    int rows,
    int cols,
    float eps) {
    int r = (int)get_global_id(0);
    if (r >= rows) return;

    // variance = mean(x^2)
    float mean_sq = 0.0f;
    int base = r * cols;
    for (int c = 0; c < cols; ++c) {
        float v = (float)LOAD(x, base + c);
        mean_sq += v * v;
    }
    mean_sq /= (float)cols;
    float inv_rms = rsqrt(mean_sq + eps);

    for (int c = 0; c < cols; ++c) {
        float v = (float)LOAD(x, base + c);
        float w = (float)LOAD(weight, c);
        float y = v * inv_rms;
        STORE(out, base + c, y * w);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Cooperative-reduction RMSNorm. One WORKGROUP per row, WG_SIZE=64 lanes
// split the cols-reduction (vec4 loads + tree-reduce in __local).
//
// Why the original was 805 µs/call:
//   With one work-item per row and rows=1 at decode, only ONE lane on the
//   whole GPU did the reduction. 99% of compute was idle while one lane
//   serially summed 1024 squares + serial normalize.
//
// Constraints assumed: cols % (WG_SIZE * 4) == 0 (true for cols=1024, WG=64).
//
// Dispatch from host: gws[0] = rows * WG_SIZE, lws[0] = WG_SIZE.
#define RMSNORM_WG 64

__kernel __attribute__((reqd_work_group_size(RMSNORM_WG, 1, 1)))
void rmsnorm_forward_v2(
    __global const storage_t* x,
    __global const storage_t* weight,
    __global storage_t* out,
    int rows,
    int cols,
    float eps) {
    const int lane = (int)get_local_id(0);
    const int r    = (int)get_group_id(0);
    if (r >= rows) return;
    const int base = r * cols;

    // Phase 1: each lane sums x^2 over its assigned cols (vec4).
    float partial = 0.0f;
    for (int c = lane * 4; c < cols; c += RMSNORM_WG * 4) {
#ifdef USE_FP16
        float4 v4 = vload_half4(0, x + base + c);
#else
        float4 v4 = vload4(0, x + base + c);
#endif
        partial += dot(v4, v4);
    }

    // Tree reduce 64 partials → broadcast inv_rms.
    __local float lds[RMSNORM_WG];
    lds[lane] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = RMSNORM_WG / 2; step > 0; step >>= 1) {
        if (lane < step) lds[lane] += lds[lane + step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float mean_sq = lds[0] / (float)cols;
    const float inv_rms = rsqrt(mean_sq + eps);

    // Phase 2: each lane normalizes+scales its assigned cols (vec4).
    for (int c = lane * 4; c < cols; c += RMSNORM_WG * 4) {
#ifdef USE_FP16
        float4 v4 = vload_half4(0, x + base + c);
        float4 w4 = vload_half4(0, weight + c);
        float4 y4 = v4 * inv_rms * w4;
        vstore_half4(y4, 0, out + base + c);
#else
        float4 v4 = vload4(0, x + base + c);
        float4 w4 = vload4(0, weight + c);
        float4 y4 = v4 * inv_rms * w4;
        vstore4(y4, 0, out + base + c);
#endif
    }
}

// Optional gated path: out = weight * (x * silu(gate)) * rsqrt(mean((x*silu(gate))^2)+eps)
__kernel void rmsnorm_gated_forward(
    __global const storage_t* x,
    __global const storage_t* gate,
    __global const storage_t* weight,
    __global storage_t* out,
    int rows,
    int cols,
    float eps) {
    int r = (int)get_global_id(0);
    if (r >= rows) return;

    int base = r * cols;

    // First pass: compute mean(square(x * silu(gate)))
    float mean_sq = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float xv = (float)LOAD(x, base + c);
        float gv = (float)LOAD(gate, base + c);
        // SiLU(g) = g / (1 + exp(-g))
        float sig = 1.0f / (1.0f + exp(-gv));
        float g_silu = gv * sig;
        float v = xv * g_silu;
        mean_sq += v * v;
    }
    mean_sq /= (float)cols;
    float inv_rms = rsqrt(mean_sq + eps);

    for (int c = 0; c < cols; ++c) {
        float xv = (float)LOAD(x, base + c);
        float gv = (float)LOAD(gate, base + c);
        float sig = 1.0f / (1.0f + exp(-gv));
        float g_silu = gv * sig;
        float v = xv * g_silu;
        float w = (float)LOAD(weight, c);
        STORE(out, base + c, v * inv_rms * w);
    }
}
