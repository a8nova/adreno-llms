// Depth-Anything-V2 (DINOv2 backbone + DPT neck/head) OpenCL kernels.
// One FILE = one cl_program on purpose: Adreno applies a program-wide
// worst-case register footprint, so a fat kernel in a shared program cuts
// wave occupancy for every kernel in it (measured 2026-07-07: frame
// 6.6s -> 10.5s from two never-dispatched kernels). Keep op families
// isolated. All data buffers are storage_t (fp16 or fp32); accumulators
// are float.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)   vload_half((i),(p))
  #define STORE(p,i,v) vstore_half((v),(i),(p))
  #define LOAD4(p,i)   vload_half4(0, (p) + (i))
  #define STORE4(p,i,v) vstore_half4((v), 0, (p) + (i))
#else
  typedef float storage_t;
  #define LOAD(p,i)   ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
  #define LOAD4(p,i)   vload4(0, (p) + (i))
  #define STORE4(p,i,v) vstore4((v), 0, (p) + (i))
#endif

// ── gelu (exact erf) in-place-style: out[i]=gelu(in[i]) ──

__kernel void gelu(__global const storage_t* input, __global storage_t* output, const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = (float)LOAD(input, i);
    float y = 0.5f * x * (1.0f + erf(x * 0.70710678118654752440f));
    STORE(output, i, y);
}

// ── relu ──

// ── layer_scale: out[r,d] = in[r,d] * lambda[d]  (row-major [rows,D]) ──

// ── add_bias_rows: out[r,d] = in[r,d] + bias[d] (for linear layers done via GEMM) ──

__kernel void add_bias_rows(
    __global storage_t* buf,
    __global const storage_t* bias,
    const int rows, const int D)
{
    int idx = get_global_id(0);
    int total = rows * D;
    if (idx >= total) return;
    int d = idx % D;
    float v = (float)LOAD(buf, idx) + (float)LOAD(bias, d);
    STORE(buf, idx, v);
}

// ── add: out[i] = a[i] + b[i] (residual connections) ──
__kernel void add(__global const storage_t* a, __global const storage_t* b,
                  __global storage_t* out, const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    STORE(out, i, (float)LOAD(a, i) + (float)LOAD(b, i));
}

// ── Qwen3-VL additions ────────────────────────────────────────────────────
// The vision tower needs BOTH GELU forms and they are not interchangeable:
// the block MLP uses gelu_pytorch_tanh, the patch mergers use nn.GELU() (erf,
// already above). Unifying them silently changes the output.
__kernel void gelu_tanh(__global const storage_t* input,
                        __global storage_t* output, const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    const float x = (float)LOAD(input, i);
    const float c = 0.7978845608028654f;               // sqrt(2/pi)
    STORE(output, i, 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x * x * x))));
}

// Vision RoPE — rotate_half over the FULL head_dim, applied to q and k in place.
// NOT the text tower's interleaved PARTIAL rope (64 of 256 dims): here
// emb = cat(rot, rot), so element i pairs with i + head_dim/2.
// One work-item per (token, head, half-dim).
__kernel void rope_vision(__global storage_t* qk,
                          __global const float* cos_t,
                          __global const float* sin_t,
                          const int tokens, const int heads,
                          const int head_dim, const int row_stride,
                          const int head_off) {
    const int gid = get_global_id(0);
    const int hd2 = head_dim / 2;   // NOT `half` — reserved type name in OpenCL C
    if (gid >= tokens * heads * hd2) return;
    const int i = gid % hd2;
    const int h = (gid / hd2) % heads;
    const int t = gid / (hd2 * heads);
    const size_t base = (size_t)t * row_stride + head_off + (size_t)h * head_dim;
    const float a = (float)LOAD(qk, base + i);
    const float b = (float)LOAD(qk, base + i + hd2);
    const size_t c = (size_t)t * head_dim;
    STORE(qk, base + i,        a * cos_t[c + i]        - b * sin_t[c + i]);
    STORE(qk, base + i + hd2, b * cos_t[c + i + hd2] + a * sin_t[c + i + hd2]);
}
