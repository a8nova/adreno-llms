// Whisper-style audio encoder frontend kernel pack.
// Reference: modeling_whisper.py WhisperEncoder.forward (conv1 + gelu + conv2 + gelu + permute + pos add).
//
// Three kernels:
//   conv1d_k3_p1     — Conv1d with kernel=3, padding=1, stride configurable; optional fused GELU
//   add_positional_embeddings — read [H,T] in row-major, write [T,H] adding learned pos embed
//
// Single-fp16-or-fp32 source via USE_FP16 toggle.

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


inline float gelu_erf_f(float x) {
    // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    const float inv_sqrt2 = 0.7071067811865475f;
    return 0.5f * x * (1.0f + erf(x * inv_sqrt2));
}

// Conv1d with kernel=3, padding=1.   (LDS-tiled — opt #14)
// Input  x: [C_in, T_in]            (row-major, channel-major)
// Weight w: [C_out, C_in, 3]
// Bias   b: [C_out] (may be nullptr — caller passes zero buffer if absent)
// Output y: [C_out, T_out]          (row-major)
//
// stride applies on the time axis; T_out = (T_in - 1) / stride + 1 for k=3, p=1.
// When apply_gelu != 0, the kernel applies GELU to its output (fused).
//
// Each WORKGROUP owns exactly one output channel `co` (NDRange dim 1, local
// size 1) and a tile of W time-steps (dim 0). The group cooperatively loads
// co's weight row [C_in*3] into __local ONCE, then every time-step in the tile
// reuses it from LDS — removing the naive kernel's T_out-fold redundant global
// re-reads of the same weight row (Adreno guideline G2). The naive version was
// ~10% (conv2) + ~4% (conv1) of all GPU time.
//
// REQUIRED dispatch: 2D NDRange, gws = { round_up(T_out, W), C_out }, lws = { W, 1 }.
// (A 1D dispatch — or any lws that puts >1 co per group — breaks the LDS sharing
// and the [t_out, co] index mapping. Keep host and kernel in lockstep.)
#define CONV_WROW_MAX 1152   // max C_in (=hidden 384) * 3 for the whisper frontend
__kernel void conv1d_k3_p1(
    __global const storage_t* x,
    __global const storage_t* w,
    __global const storage_t* b,
    __global storage_t* y,
    int C_in,
    int C_out,
    int T_in,
    int T_out,
    int stride,
    int apply_gelu)
{
    __local float w_lds[CONV_WROW_MAX];   // this group's output-channel weight row

    int co    = (int)get_global_id(1);
    int t_out = (int)get_global_id(0);
    int lid   = (int)get_local_id(0);
    int lsz   = (int)get_local_size(0);

    // Cooperative weight-row load. ALL work-items participate (including padded
    // t_out >= T_out lanes) so the barrier is reached uniformly by the group.
    int wcount = C_in * 3;
    int w_base_co = co * wcount;
    for (int i = lid; i < wcount; i += lsz) {
        w_lds[i] = (float)LOAD(w, w_base_co + i);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (t_out >= T_out) return;   // guard AFTER the barrier — never skip it divergently

    int t_in_center = t_out * stride;
    int t0 = t_in_center - 1, t1 = t_in_center, t2 = t_in_center + 1;
    float acc = (b ? (float)LOAD(b, co) : 0.0f);
    for (int ci = 0; ci < C_in; ++ci) {
        int x_base = ci * T_in;
        int wo = ci * 3;
        if ((unsigned)t0 < (unsigned)T_in) acc += (float)LOAD(x, x_base + t0) * w_lds[wo + 0];
        if ((unsigned)t1 < (unsigned)T_in) acc += (float)LOAD(x, x_base + t1) * w_lds[wo + 1];
        if ((unsigned)t2 < (unsigned)T_in) acc += (float)LOAD(x, x_base + t2) * w_lds[wo + 2];
    }

    if (apply_gelu) acc = gelu_erf_f(acc);
    STORE(y, co * T_out + t_out, acc);
}

// Permute [H, T] -> [T, H] and add learned positional embedding [T, H].
// 2D NDRange: gws = { T, H } so every output element gets a unique thread.
//   x_HT  read as [H, T] (channel-major from conv outputs)
//   pos_TH read as [T, H] (learned positional embedding rows)
//   y_TH  write as [T, H]
__kernel void add_positional_embeddings(
    __global const storage_t* x_HT,
    __global const storage_t* pos_TH,
    __global storage_t* y_TH,
    int T,
    int H)
{
    int t = (int)get_global_id(0);
    int h = (int)get_global_id(1);
    if (t >= T || h >= H) return;
    int idx_in  = h * T + t;
    int idx_out = t * H + h;
    float v = (float)LOAD(x_HT, idx_in) + (float)LOAD(pos_TH, idx_out);
    STORE(y_TH, idx_out, v);
}
