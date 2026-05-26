#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)   vload_half((i),(p))
  #define STORE(p,i,v) vstore_half((v),(i),(p))
#else
  typedef float storage_t;
  #define LOAD(p,i)   ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
#endif

// Reference: model_info/transformers_src/modeling_vits.py (see VitsResidualCouplingLayer.forward and
// VitsResidualCouplingBlock.forward reverse=True path used by VitsModel.forward)
//
// flow_affine_coupling — one affine coupling layer of the VITS normalizing
// flow, INVERSE direction (used at inference).
//
// Forward (training):
//   x = concat(x_a, x_b)    where x_a, x_b are equal halves on channel dim
//   t = WN(x_a)             (a small Wavenet-style stack — host-side)
//   y_b = x_b + t           (PURE-ADDITIVE coupling — VITS uses additive;
//                            the "affine" name comes from the general flow
//                            family. Some implementations add a log_scale s
//                            and do y_b = x_b * exp(s) + t. HF VitsModel
//                            uses additive only — see VitsResidualCouplingLayer.)
//   y = concat(x_a, y_b)
//
// Inverse (inference):
//   y = concat(y_a, y_b)
//   t = WN(y_a)             (identical stack — symmetric)
//   x_b = y_b - t
//   x = concat(y_a, x_b)
//
// This kernel ONLY does the final mix step (subtract t from the second
// half, leave the first half untouched). The Wavenet stack is a sequence
// of conv_1d / leaky_relu launches in the host wrapper.
//
// Layout: [C, L] — channels-first. half_C = C / 2.

__kernel void coupling_inverse_mix(
    __global const storage_t* y,        // [C, L] — input (y_a top half, y_b bottom half)
    __global const storage_t* t,        // [half_C, L] — Wavenet output to subtract
    __global       storage_t* x,        // [C, L] — output
    const int C,
    const int L) {

    const int gid = get_global_id(0);
    const int total = C * L;
    if (gid >= total) return;

    const int ch = gid / L;
    const int l  = gid % L;
    const int half_C = C / 2;

    if (ch < half_C) {
        // First half: passthrough.
        x[gid] = y[gid];
    } else {
        // Second half: subtract translation.
        const int t_idx = (ch - half_C) * L + l;
        x[gid] = (storage_t)((float)y[gid] - (float)t[t_idx]);
    }
}

// Channel-flip — VITS flow alternates orientation between coupling layers.
// After each coupling, flip channels along dim 0 so the next layer's "first
// half" is the previous "second half". Equivalent to a permutation matrix.
__kernel void channel_flip(
    __global const storage_t* in,       // [C, L]
    __global       storage_t* out,      // [C, L]
    const int C,
    const int L) {
    const int gid = get_global_id(0);
    const int total = C * L;
    if (gid >= total) return;
    const int ch = gid / L;
    const int l  = gid % L;
    out[(C - 1 - ch) * L + l] = in[gid];
}

// Clamp-in-place: limit values to [-cap, +cap]. Used in flow_inverse to
// stop fp16 conv overflow from propagating to NaN through tanh.
__kernel void clamp_inplace(
    __global storage_t* x,
    const int N,
    const float cap) {
    const int gid = get_global_id(0);
    if (gid >= N) return;
    float v = (float)LOAD(x, gid);
    if (v > cap)  v =  cap;
    if (v < -cap) v = -cap;
    STORE(x, gid, v);
}

// WaveNet gated activation: given x [2H, T], compute
//   acts[h*T + t] = tanh(x[h*T + t]) * sigmoid(x[(h+H)*T + t])
// where H = hidden / 2 (yes — the kernel calls the "hidden" param the
// full 2*H pre-split width).
__kernel void wavenet_gate(
    __global const storage_t* x,        // [2H, T]
    __global       storage_t* acts,     // [H,  T]
    const int H,
    const int T) {
    const int gid = get_global_id(0);
    if (gid >= H * T) return;
    const int h = gid / T;
    const int t = gid % T;
    const float ta = LOAD(x,  h      * T + t);
    const float sb = LOAD(x, (h + H) * T + t);
    const float s_safe = sb < -15.0f ? -15.0f : (sb > 15.0f ? 15.0f : sb);
    const float val = tanh(ta) * (1.0f / (1.0f + exp(-s_safe)));
    STORE(acts, gid, val);
}

// out[i] = a[i] + b[i]    (both same length N)
__kernel void elem_add(
    __global const storage_t* a,
    __global const storage_t* b,
    __global       storage_t* out,
    const int N) {
    const int gid = get_global_id(0);
    if (gid >= N) return;
    STORE(out, gid, (float)LOAD(a, gid) + (float)LOAD(b, gid));
}

// In-place: a[i] += b[i]. Saves an output buffer alloc when caller can mutate a.
__kernel void elem_add_inplace_a(
    __global       storage_t* a,
    __global const storage_t* b,
    const int N) {
    const int gid = get_global_id(0);
    if (gid >= N) return;
    STORE(a, gid, (float)LOAD(a, gid) + (float)LOAD(b, gid));
}

// In-place: a[i] -= b[i].
__kernel void elem_sub_inplace_a(
    __global       storage_t* a,
    __global const storage_t* b,
    const int N) {
    const int gid = get_global_id(0);
    if (gid >= N) return;
    STORE(a, gid, (float)LOAD(a, gid) - (float)LOAD(b, gid));
}

// Split lower/upper channel halves: y_a = x[0:half, :], y_b = x[half:2*half, :]
__kernel void split_half_c(
    __global const storage_t* x,       // [2*half, T]
    __global       storage_t* y_a,     // [half, T]
    __global       storage_t* y_b,     // [half, T]
    const int half_c,
    const int T) {
    const int gid = get_global_id(0);
    const int total = half_c * T;
    if (gid >= total) return;
    const int c = gid / T;
    const int t = gid - c * T;
    STORE(y_a, gid, (float)LOAD(x, c * T + t));
    STORE(y_b, gid, (float)LOAD(x, (c + half_c) * T + t));
}

// Concat lower/upper channel halves: x[0:half] = y_a, x[half:2half] = y_b.
__kernel void concat_half_c(
    __global       storage_t* x,       // [2*half, T]
    __global const storage_t* y_a,     // [half, T]
    __global const storage_t* y_b,     // [half, T]
    const int half_c,
    const int T) {
    const int gid = get_global_id(0);
    const int total = half_c * T;
    if (gid >= total) return;
    const int c = gid / T;
    const int t = gid - c * T;
    STORE(x, c * T + t, (float)LOAD(y_a, gid));
    STORE(x, (c + half_c) * T + t, (float)LOAD(y_b, gid));
}

// Fused: y_b' = y_b - t_mean, then concat(y_a, y_b') → x. One dispatch
// replaces (elem_sub_inplace_a + concat_half_c). Saves 1 dispatch per stage
// × 4 stages × ~54 ms gap = ~216 ms wall.
__kernel void sub_then_concat_half_c(
    __global       storage_t* x,        // [2*half, T] — output
    __global const storage_t* y_a,      // [half, T]
    __global const storage_t* y_b,      // [half, T] — original (not modified)
    __global const storage_t* t_mean,   // [half, T] — subtract from y_b
    const int half_c,
    const int T) {
    const int gid = get_global_id(0);
    const int total = half_c * T;
    if (gid >= total) return;
    const int c = gid / T;
    const int t = gid - c * T;
    STORE(x, c * T + t, (float)LOAD(y_a, gid));
    STORE(x, (c + half_c) * T + t,
          (float)LOAD(y_b, gid) - (float)LOAD(t_mean, gid));
}

// Split [2*C, T] into res [C, T] and skip [C, T] along the channel axis.
__kernel void split_res_skip(
    __global const storage_t* rs,      // [2*C, T]
    __global       storage_t* res,     // [C, T]
    __global       storage_t* skip,    // [C, T]
    const int C,
    const int T) {
    const int gid = get_global_id(0);
    const int total = C * T;
    if (gid >= total) return;
    const int c = gid / T;
    const int t = gid - c * T;
    STORE(res,  gid, (float)LOAD(rs, c * T + t));
    STORE(skip, gid, (float)LOAD(rs, (c + C) * T + t));
}

// Fused: split rs into (res, skip) AND in-place add res to h AND in-place add
// skip to skip_sum — all in one dispatch. Replaces the 3-call sequence
// (split_res_skip + elem_add_inplace_a × 2) used in the WaveNet inner loop.
// Saves 2 dispatches per layer × 12 layers per inference = 24 dispatches ×
// ~54 ms gap on Adreno 620 = ~1.3 s wall.
//
// One workitem per (c, t) cell. C×T total workitems.
__kernel void split_rs_fold_h_skip(
    __global const storage_t* rs,        // [2*C, T] — wavenet res_skip output
    __global       storage_t* h,         // [C, T]   — in-place h += rs[:C, :]
    __global       storage_t* skip_sum,  // [C, T]   — in-place skip_sum += rs[C:, :]
    const int C,
    const int T) {
    const int gid = get_global_id(0);
    const int total = C * T;
    if (gid >= total) return;
    const int c = gid / T;
    const int t = gid - c * T;
    const float res_v  = (float)LOAD(rs, c * T + t);
    const float skip_v = (float)LOAD(rs, (c + C) * T + t);
    STORE(h,        gid, (float)LOAD(h,        gid) + res_v);
    STORE(skip_sum, gid, (float)LOAD(skip_sum, gid) + skip_v);
}

// Copy a contiguous slice of `src` into `dst`. For the WaveNet res-skip split:
// src is [2H, T] flattened; dst is [H, T]. Copies either the first H rows
// (start_row=0) or the last H rows (start_row=H) of the 2H-row buffer.
__kernel void copy_rows(
    __global const storage_t* src,
    __global       storage_t* dst,
    const int start_row,
    const int rows,
    const int cols) {
    const int gid = get_global_id(0);
    if (gid >= rows * cols) return;
    const int r = gid / cols;
    const int c = gid % cols;
    const int src_idx = (start_row + r) * cols + c;
    STORE(dst, gid, (float)LOAD(src, src_idx));
}

// Per-channel scale: out[c, t] *= scale[c]. scale buffer is [C].
__kernel void per_channel_scale(
    __global       storage_t* x,
    __global const storage_t* scale,
    const int C,
    const int T) {
    const int gid = get_global_id(0);
    if (gid >= C * T) return;
    const int c = gid / T;
    const float s = LOAD(scale, c);
    STORE(x, gid, (float)LOAD(x, gid) * s);
}
