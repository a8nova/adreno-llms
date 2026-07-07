// Elementwise activations. Owned by src/ops/moonshine_common.cpp + MLP ops.
// Split from the former kernels/moonshine.cl monolith (whisper-parity layout);
// kernel bodies are byte-identical to the monolith at the time of the split.
// Dtype-template preamble — DO NOT EDIT. Driven by host-side -DNNOPT_USE_FP16.
// Dtype-template preamble — DO NOT EDIT. Driven by host-side -DNNOPT_USE_FP16.
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

// ── tanh activation (in place-ish, elementwise) ─────────────────────────
__kernel void tanh_act(__global const storage_t* in, __global storage_t* out, const int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float x = LOAD(in, gid);
    STORE(out, gid, tanh(x));
}

// ── GELU (exact, erf-based — matches torch.nn.functional.gelu default) ───
__kernel void gelu_act(__global const storage_t* in, __global storage_t* out, const int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float x = LOAD(in, gid);
    // exact gelu: 0.5 * x * (1 + erf(x / sqrt(2)))
    float y = 0.5f * x * (1.0f + erf(x * 0.7071067811865476f));
    STORE(out, gid, y);
}

// ── SiLU activation ─────────────────────────────────────────────────────
__kernel void silu_act(__global const storage_t* in, __global storage_t* out, const int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float x = LOAD(in, gid);
    STORE(out, gid, x / (1.0f + exp(-x)));
}

// ── GLU-SiLU: out = silu(gate) * value ──────────────────────────────────
// value = first[rows, half], gate = second[rows, half] (from chunk(2)).
__kernel void glu_silu(
    __global const storage_t* value,   // [rows, half]
    __global const storage_t* gate,    // [rows, half]
    __global storage_t* out,           // [rows, half]
    const int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float g = LOAD(gate, gid);
    float v = LOAD(value, gid);
    float sg = g / (1.0f + exp(-g));
    STORE(out, gid, sg * v);
}
