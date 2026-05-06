// Reference: model_info/transformers_src/modeling_lfm2.py:240-318 Lfm2ShortConv.slow_forward
// Implements the per-layer depthwise causal conv path.
// Layout conventions in this file:
// - Most intermediate tensors use [H, S] (hidden-major) to make depthwise conv contiguous per channel.
// - Model entry/exit tensors remain [S, H] (seq-major).

#include "utils.cl"

__kernel void conv_copy_transpose(
    __global const storage_t* src, // [S, 3H]
    __global storage_t* dst,       // [3H, S]
    const int S,
    const int H,
    const int total) {
  const int gid = get_global_id(0);
  if (gid >= total) return;
  const int cols = 3 * H;
  const int r = gid / cols;   // token
  const int c = gid - r * cols; // channel in 3H
  // dst[c, r] = src[r, c]
  STORE(dst, c * S + r, LOAD(src, r * cols + c));
}

__kernel void conv_copy_transpose_back(
    __global const storage_t* src, // [H, S]
    __global storage_t* dst,       // [S, H]
    const int S,
    const int H,
    const int total) {
  const int gid = get_global_id(0);
  if (gid >= total) return;
  const int r = gid / H;     // token
  const int c = gid - r * H; // channel
  STORE(dst, r * H + c, LOAD(src, c * S + r));
}

__kernel void conv_split_chunk3(
    __global const storage_t* src, // [3H, S]
    __global storage_t* B,         // [H, S]
    __global storage_t* C,         // [H, S]
    __global storage_t* X,         // [H, S]
    const int S,
    const int H,
    const int total) {
  const int gid = get_global_id(0);
  if (gid >= total) return;
  const int c = gid / S;   // channel in [0, H)
  const int t = gid - c * S;
  STORE(B, gid, LOAD(src, c * S + t));
  STORE(C, gid, LOAD(src, (H + c) * S + t));
  STORE(X, gid, LOAD(src, (2 * H + c) * S + t));
}

__kernel void conv_pointwise_mul(
    __global const storage_t* B,   // [H, S]
    __global const storage_t* X,   // [H, S]
    __global storage_t* Bx,        // [H, S]
    const int total) {
  const int gid = get_global_id(0);
  if (gid >= total) return;
  STORE(Bx, gid, LOAD(B, gid) * LOAD(X, gid));
}

// For each (c,t):
//   window = [prev[c,0..L-2], Bx[c,0..S-1]]
//   conv_out[c,t] = sum_{j=0..L-1} conv_w[c,j] * window[t + (L-1) - j]
// where conv_w is [H,1,L] in weights, flattened as [H*L].
__kernel void conv1d_causal_with_cache(
    __global const storage_t* Bx,      // [H, S]
    __global const storage_t* conv_w,  // [H, L]
    __global const storage_t* prev,    // [H, L-1] (nullable if L==1)
    __global storage_t* out,           // [H, S]
    const int S,
    const int H,
    const int L,
    const int total) {
  const int gid = get_global_id(0);
  if (gid >= total) return;
  const int c = gid / S;
  const int t = gid - c * S;

  float acc = 0.0f;
  const int wbase = c * L;

  for (int j = 0; j < L; ++j) {
    // index into concatenated [prev, Bx]
    // PyTorch nn.Conv1d is cross-correlation, NOT mathematical convolution —
    // weights are NOT time-reversed. For output position t:
    //   sum_j w[c, j] * input_padded[c, t + j]
    // where input_padded has L-1 zero/prev values at the start.
    const int idx = t + j;
    float x = 0.0f;
    if (idx < (L - 1)) {
      if (prev) x = LOAD(prev, c * (L - 1) + idx);
    } else {
      const int tt = idx - (L - 1);
      if (tt >= 0 && tt < S) {
        x = LOAD(Bx, c * S + tt);
      }
    }
    acc += LOAD(conv_w, wbase + j) * x;
  }

  STORE(out, gid, acc);
}

// conv_state: [hidden, L_cache] where L_cache=MODEL_CONFIG::CONV_L_CACHE
// After processing seq_len new tokens (Bx in [seq, hidden]), update the rolling
// L_cache-length history so conv1d_causal_with_cache sees the correct prev values.
//
// Correct sliding-window semantics:
//   if seq_len >= L_cache: full replacement — new_state[j] = Bx[seq_len - L_cache + j, c]
//   if seq_len < L_cache:  shift left by seq_len, append new Bx at the end.
//     new_state[j] = old_state[j + seq_len]      for j < L_cache - seq_len
//     new_state[j] = Bx[j - (L_cache-seq_len), c] for j >= L_cache - seq_len
//
// The old (buggy) code always used the "if t >= 0 else 0" branch which zero-padded
// the front during single-token decode, discarding the prior prefill context.
__kernel void conv_update_state(
    __global const storage_t* Bx,         // [seq, hidden]
    __global storage_t* conv_state,       // [hidden, L_cache]
    const int seq_len,
    const int hidden,
    const int L_cache) {
  const int c = get_global_id(0);
  if (c >= hidden) return;

  if (seq_len >= L_cache) {
    // Full replacement: all new state values come from Bx
    for (int j = 0; j < L_cache; ++j) {
      const int t = seq_len - L_cache + j;
      STORE(conv_state, c * L_cache + j, LOAD(Bx, t * hidden + c));
    }
  } else {
    // Shift left by seq_len, append new Bx at the end.
    // Read old tail first (in-place update: read before write to avoid aliasing).
    // L_cache is small (2 for kernel_size=3), so private array is fine on device.
    float old_tail[8];
    const int to_keep = L_cache - seq_len;
    for (int j = 0; j < to_keep; ++j) {
      old_tail[j] = LOAD(conv_state, c * L_cache + (j + seq_len));
    }
    for (int j = 0; j < to_keep; ++j) {
      STORE(conv_state, c * L_cache + j, old_tail[j]);
    }
    for (int j = 0; j < seq_len; ++j) {
      STORE(conv_state, c * L_cache + to_keep + j, LOAD(Bx, j * hidden + c));
    }
  }
}

// Compute conv_out[t,c] = sum_j conv_state[c,j] * weight[c,j] (+ bias[c])
// This matches the slow path:
//   conv_state = update_conv_state(Bx)
//   conv_out = sum(conv_state * conv.weight[:,0,:], dim=-1)
__kernel void conv_apply_state_dot(
    __global const storage_t* conv_state,  // [hidden, L_cache]
    __global const storage_t* weight,      // [hidden, L_cache]
    __global const storage_t* bias,        // [hidden] or nullptr
    __global storage_t* conv_out,          // [hidden]
    const int hidden,
    const int L_cache,
    const int has_bias) {
  const int c = get_global_id(0);
  if (c >= hidden) return;
  float acc = 0.0f;
  const int base = c * L_cache;
  for (int j = 0; j < L_cache; ++j) {
    acc += LOAD(conv_state, base + j) * LOAD(weight, base + j);
  }
  if (has_bias) acc += LOAD(bias, c);
  STORE(conv_out, c, acc);
}

// y[t,c] = C[t,c] * conv_out[t,c] (conv_out is broadcast from [hidden] at decode)
__kernel void conv_mul_C(
    __global const storage_t* C,      // [seq, hidden]
    __global const storage_t* conv,   // [seq, hidden]
    __global storage_t* y,            // [seq, hidden]
    const int total) {
  const int gid = get_global_id(0);
  if (gid >= total) return;
  STORE(y, gid, LOAD(C, gid) * LOAD(conv, gid));
}
