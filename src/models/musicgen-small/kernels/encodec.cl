// =============================================================================
// encodec.cl — GPU SEANet decoder kernels (EnCodec 32 kHz, musicgen-small).
//
// Phase-A GPU port of src/encodec_host.cpp's conv stack (the ~11 s CPU tail):
// reflect-pad, Conv1d, ConvTranspose1d (gather form), ELU, residual add.
// The LSTM (layers.1) stays on the CPU in this phase — the [1024,250]
// activation roundtrip is ~1 MB, trivial.
//
// fp32 throughout (matches the host path's activation dtype; weights are the
// host-computed weight_norm EFFECTIVE weights, uploaded per clip). The host
// accumulates in DOUBLE; these kernels accumulate fp32 — and the convT gather
// order differs from the host's scatter order — so output is NOT bit-identical
// to the CPU path. Gate: cosine(PCM_gpu, PCM_cpu) on identical greedy codes.
//
// Index math (reflect pad, conv trim, convT trim) REPLICATES encodec_host.cpp
// exactly — any change there must land here too.
// =============================================================================

// Mirror nn.functional.pad(mode="reflect") with EnCodec's extra-zero-pad quirk
// (host pad1d_reflect): out length is ALREADY the trimmed `end` — values for
// o < end are unaffected by the trim, so the kernel just never writes the tail.
// work buffer = original samples [0..length-1] then zeros (extra_pad region).
__kernel void enc_pad_reflect(__global const float* in,   // [ch, Tin]
                              __global float* out,         // [ch, Tout]
                              const int Tin, const int Tout,
                              const int pad_left, const int work_len,
                              const int length) {
  const int o = get_global_id(0);
  const int c = get_global_id(1);
  if (o >= Tout) return;
  int src = o - pad_left;
  if (work_len > 1) {
    while (src < 0 || src >= work_len) {
      if (src < 0) src = -src;
      if (src >= work_len) src = 2 * (work_len - 1) - src;
    }
  } else {
    src = 0;
  }
  out[(size_t)c * Tout + o] = (src < length) ? in[(size_t)c * Tin + src] : 0.0f;
}

// Conv1d over the pre-padded input. One thread per (t, o).
// W: effective weight_norm weight, [out_ch, in_ch, k] row-major fp32.
__kernel void enc_conv1d(__global const float* xpad,  // [in_ch, Tpad]
                         __global const float* W,
                         __global const float* bias,  // [out_ch]
                         __global float* out,          // [out_ch, Tout]
                         const int in_ch, const int Tpad, const int Tout,
                         const int k, const int stride, const int dil) {
  const int t = get_global_id(0);
  const int o = get_global_id(1);
  if (t >= Tout) return;
  float acc = bias[o];
  const int base = t * stride;
  __global const float* wp = W + (size_t)o * in_ch * k;
  // Scalar ic loop — the 4-lane ILP variant (like convT's) MEASURED SLOWER
  // here (2.5 → 3.6 s/clip): with k in {1,3,7} the strided 4-way gathers cost
  // more than the broken dependency chain saves. Do not re-try.
  for (int ic = 0; ic < in_ch; ++ic) {
    __global const float* xr = xpad + (size_t)ic * Tpad + base;
    __global const float* wq = wp + (size_t)ic * k;
    for (int kk = 0; kk < k; ++kk)
      acc += wq[kk] * xr[kk * dil];
  }
  out[(size_t)o * Tout + t] = acc;
}

// ConvTranspose1d in GATHER form with the non-causal trim built in:
// out[o, t] (t in [0, Tout)) sums x[ic, ti]·W over the taps kk where
// traw = t + pad_left = ti*stride + kk has a valid integer ti < Tin.
// Valid kk form an arithmetic set kk ≡ traw (mod stride) — k/stride taps
// (2 for every SEANet stage), stepped directly (no modulo scan).
// Wt: effective weight TRANSPOSED BY THE HOST to [out_ch, k, in_ch] so the
// inner ic loop reads CONTIGUOUS memory. (The PyTorch [in,out,k] layout put
// consecutive ic reads out_ch*k floats apart — 32 KB stride at stage 3 —
// every access a cache miss: 19.1 s for 4 dispatches. This layout: contiguous
// wq[ic] + warp-coalesced x reads across adjacent t.)
__kernel void enc_convt1d(__global const float* x,    // [in_ch, Tin]
                          __global const float* Wt,   // [out_ch, k, in_ch]
                          __global const float* bias, // [out_ch]
                          __global float* out,         // [out_ch, Tout]
                          const int in_ch, const int Tin, const int out_ch,
                          const int Tout, const int k, const int stride,
                          const int pad_left) {
  const int t = get_global_id(0);
  const int o = get_global_id(1);
  if (t >= Tout) return;
  float acc = bias[o];
  const int traw = t + pad_left;
  for (int kk = traw % stride; kk < k; kk += stride) {
    const int ti = (traw - kk) / stride;
    if (ti < 0) break;                   // ti only decreases with kk
    if (ti >= Tin) continue;
    __global const float* wq = Wt + ((size_t)o * k + kk) * in_ch;
    __global const float* xc = x + ti;
    // 4 independent accumulator lanes + vectorized contiguous weight load:
    // the scalar form was a 2048-deep DEPENDENT fma chain (ALU-latency bound).
    // in_ch is a multiple of 4 for every SEANet layer (1024/512/256/128/64/32).
    float4 a4 = (float4)(0.0f);
    for (int ic = 0; ic < in_ch; ic += 4) {
      const float4 wv = vload4(0, wq + ic);
      const float4 xv = (float4)(xc[(size_t)ic * Tin],
                                 xc[(size_t)(ic + 1) * Tin],
                                 xc[(size_t)(ic + 2) * Tin],
                                 xc[(size_t)(ic + 3) * Tin]);
      a4 += wv * xv;
    }
    acc += a4.x + a4.y + a4.z + a4.w;
  }
  out[(size_t)o * Tout + t] = acc;
}

// ELU(alpha=1). exp(v)-1 (OpenCL has no expm1; difference is far below the
// cosine gate's resolution). In-place and out-of-place variants — the
// out-of-place one feeds ResnetBlock (which needs its input preserved for
// the identity shortcut).
__kernel void enc_elu(__global float* x, const int n) {
  const int i = get_global_id(0);
  if (i >= n) return;
  const float v = x[i];
  if (v <= 0.0f) x[i] = exp(v) - 1.0f;
}

__kernel void enc_elu_oop(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i >= n) return;
  const float v = in[i];
  out[i] = (v <= 0.0f) ? (exp(v) - 1.0f) : v;
}

// Residual add: a[i] += b[i]  (ResnetBlock identity shortcut).
__kernel void enc_add(__global float* a, __global const float* b, const int n) {
  const int i = get_global_id(0);
  if (i < n) a[i] += b[i];
}

// ── Phase B: LSTM on GPU ─────────────────────────────────────────────────────
// EncodecLSTM = 2 stacked LSTM(1024) + residual. The input projection
// gin[t,r] = bih[r]+bhh[r]+Wih[r,:]·x[:,t] has no recurrence dependency →
// ONE batched kernel over all t. The recurrence runs as 2 dispatches per
// step (gates GEMV on h_{t-1} + cell update), all enqueued async — the
// in-order queue serializes the h dependency for free.

__kernel void enc_lstm_gin(__global const float* Wih,   // [4H, H]
                           __global const float* bih,   // [4H]
                           __global const float* bhh,   // [4H]
                           __global const float* x,     // [H, T]
                           __global float* gin,         // [T, 4H]
                           const int H, const int T) {
  const int t = get_global_id(0);   // dim0 = t: x coalesced, Wih broadcast
  const int r = get_global_id(1);
  if (t >= T) return;
  __global const float* wr = Wih + (size_t)r * H;
  float4 a4 = (float4)(0.0f);
  for (int j = 0; j < H; j += 4) {
    const float4 wv = vload4(0, wr + j);
    const float4 xv = (float4)(x[(size_t)j * T + t], x[(size_t)(j + 1) * T + t],
                               x[(size_t)(j + 2) * T + t], x[(size_t)(j + 3) * T + t]);
    a4 += wv * xv;
  }
  gin[(size_t)t * 4 * H + r] = bih[r] + bhh[r] + a4.x + a4.y + a4.z + a4.w;
}

// One recurrence step's gate GEMV: g[r] = gin[t,r] + Whh[r,:]·h.
// 16-lane row-groups, h staged in __local, float4 lanes.
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void enc_lstm_gates(__global const float* Whh,  // [4H, H]
                    __global const float* gin,  // [T, 4H]
                    __global const float* h,    // [H]
                    __global float* g,          // [4H]
                    const int H, const int t, const int rows_per_wg) {
  const int wg = get_group_id(0);
  const int lid = get_local_id(0);
  const int grp = lid / 16;
  const int lane = lid % 16;
  __local float hs[1024];
  __local float part[64];
  for (int c = lid; c < H; c += 64) hs[c] = h[c];
  barrier(CLK_LOCAL_MEM_FENCE);
  const int n0 = wg * rows_per_wg;
  for (int n = n0 + grp; n < n0 + rows_per_wg; n += 4) {
    __global const float* wr = Whh + (size_t)n * H;
    float4 a4 = (float4)(0.0f);
    for (int j = lane * 4; j < H; j += 64)
      a4 += vload4(0, wr + j) * vload4(0, hs + j);
    part[lid] = a4.x + a4.y + a4.z + a4.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = 8; o > 0; o >>= 1) {
      if (lane < o) part[lid] += part[lid + o];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) g[n] = gin[(size_t)t * 4 * H + n] + part[grp * 16];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// One step's cell update. PyTorch gate order [i, f, g, o] in rows of g.
__kernel void enc_lstm_cell(__global const float* g,  // [4H]
                            __global float* c,        // [H]
                            __global float* h,        // [H]
                            __global float* y,        // [H, T] layer output
                            const int H, const int T, const int t) {
  const int j = get_global_id(0);
  if (j >= H) return;
  const float ig = 1.0f / (1.0f + exp(-g[j]));
  const float fg = 1.0f / (1.0f + exp(-g[H + j]));
  const float gg = tanh(g[2 * H + j]);
  const float og = 1.0f / (1.0f + exp(-g[3 * H + j]));
  const float cn = fg * c[j] + ig * gg;
  c[j] = cn;
  const float hn = og * tanh(cn);
  h[j] = hn;
  y[(size_t)j * T + t] = hn;
}

// ── Transposed-activation variants (2026-06-05) ─────────────────────────────
// The untiled kernels gather x COLUMNS (x[ic*Tin + ti], stride-Tin per element
// — a cache miss per read at large Tin); the 4 convT dispatches alone cost
// 3.0 s (~3 GMAC/s). A 16x16 __local-tile variant was FALSIFIED: 24 KB local
// → 1 WG/SP → occupancy collapse (slower than the gathers it removed).
// This variant keeps the original 64-thread WG shape and instead TRANSPOSES
// the activation once per stage (elementwise, ~ms): x_t[ti, ic] makes every
// inner read a contiguous vload4.
//   enc_convt1d_x: loop structure and accumulation order IDENTICAL to
//     enc_convt1d (ic-ascending 4-lane a4) → output bit-identical.
//   enc_conv1d_x: needs kk-outer/ic-inner order (W transposed to [o,k,in] by
//     the host like convT's) → accumulation ORDER CHANGES vs enc_conv1d —
//     gated on PCM cosine vs the untiled path, not bytes.

__kernel void enc_xt(__global const float* x,   // [ch, T]
                     __global float* xt,         // [T, ch]
                     const int ch, const int T) {
  const int t = get_global_id(0);
  const int c = get_global_id(1);
  if (t < T) xt[(size_t)t * ch + c] = x[(size_t)c * T + t];
}

__kernel void enc_convt1d_x(__global const float* xt,   // [Tin, in_ch]
                            __global const float* Wt,    // [out_ch, k, in_ch]
                            __global const float* bias,  // [out_ch]
                            __global float* out,          // [out_ch, Tout]
                            const int in_ch, const int Tin, const int out_ch,
                            const int Tout, const int k, const int stride,
                            const int pad_left) {
  const int t = get_global_id(0);
  const int o = get_global_id(1);
  if (t >= Tout) return;
  float acc = bias[o];
  const int traw = t + pad_left;
  for (int kk = traw % stride; kk < k; kk += stride) {
    const int ti = (traw - kk) / stride;
    if (ti < 0) break;
    if (ti >= Tin) continue;
    __global const float* wq = Wt + ((size_t)o * k + kk) * in_ch;
    __global const float* xc = xt + (size_t)ti * in_ch;
    float4 a4 = (float4)(0.0f);
    for (int ic = 0; ic < in_ch; ic += 4) {
      const float4 wv = vload4(0, wq + ic);
      const float4 xv = vload4(0, xc + ic);   // contiguous — was a 4-line gather
      a4 += wv * xv;
    }
    acc += a4.x + a4.y + a4.z + a4.w;
  }
  out[(size_t)o * Tout + t] = acc;
}

__kernel void enc_conv1d_x(__global const float* xpad_t, // [Tpad, in_ch]
                           __global const float* Wt,      // [out_ch, k, in_ch]
                           __global const float* bias,    // [out_ch]
                           __global float* out,            // [out_ch, Tout]
                           const int in_ch, const int Tpad, const int Tout,
                           const int k, const int stride, const int dil) {
  const int t = get_global_id(0);
  const int o = get_global_id(1);
  if (t >= Tout) return;
  float acc = bias[o];
  const int base = t * stride;
  for (int kk = 0; kk < k; ++kk) {
    __global const float* wq = Wt + ((size_t)o * k + kk) * in_ch;
    __global const float* xr = xpad_t + (size_t)(base + kk * dil) * in_ch;
    float4 a4 = (float4)(0.0f);
    for (int ic = 0; ic < in_ch; ic += 4) {
      a4 += vload4(0, wq + ic) * vload4(0, xr + ic);
    }
    acc += a4.x + a4.y + a4.z + a4.w;
  }
  out[(size_t)o * Tout + t] = acc;
}

// ── 4-outputs-per-thread variants (x4): amortize thread launch + weight reads.
// Late SEANet stages dispatch ~10M one-output threads (Tout 158k × out_ch 64);
// per-thread work is ~256 MACs, so launch overhead + per-thread weight reloads
// dominate. Each x4 thread computes outputs t, t+s, t+2s, t+3s — these share
// traw%stride, hence the SAME kk tap set and the SAME wq vectors, which are
// loaded once and applied to 4 x rows. Per-output accumulation order is
// IDENTICAL to the _x variants (same-seed wav byte gate holds for convT).
__kernel void enc_convt1d_x4(__global const float* xt,   // [Tin, in_ch]
                             __global const float* Wt,    // [out_ch, k, in_ch]
                             __global const float* bias,  // [out_ch]
                             __global float* out,          // [out_ch, Tout]
                             const int in_ch, const int Tin, const int out_ch,
                             const int Tout, const int k, const int stride,
                             const int pad_left) {
  const int tb = get_global_id(0);              // thread block index over t
  const int o  = get_global_id(1);
  const int g = tb / stride, lane = tb - g * stride;
  const int t0 = g * 4 * stride + lane;          // outputs t0 + j*stride, j=0..3
  if (t0 >= Tout) return;
  float acc[4];
  const int traw0 = t0 + pad_left;
  acc[0] = acc[1] = acc[2] = acc[3] = bias[o];
  for (int kk = traw0 % stride; kk < k; kk += stride) {
    const int ti0 = (traw0 - kk) / stride;       // ti for j: ti0 + j
    if (ti0 + 3 < 0) break;                      // all four lanes out of range
    __global const float* wq = Wt + ((size_t)o * k + kk) * in_ch;
    float4 a4[4] = {(float4)(0.0f), (float4)(0.0f), (float4)(0.0f), (float4)(0.0f)};
    for (int ic = 0; ic < in_ch; ic += 4) {
      const float4 wv = vload4(0, wq + ic);
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int ti = ti0 + j;
        if (ti >= 0 && ti < Tin)
          a4[j] += wv * vload4(0, xt + (size_t)ti * in_ch + ic);
      }
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      if (ti0 + j >= 0 && ti0 + j < Tin)
        acc[j] += a4[j].x + a4[j].y + a4[j].z + a4[j].w;
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int t = t0 + j * stride;
    if (t < Tout) out[(size_t)o * Tout + t] = acc[j];
  }
}

__kernel void enc_conv1d_x4(__global const float* xpad_t, // [Tpad, in_ch]
                            __global const float* Wt,      // [out_ch, k, in_ch]
                            __global const float* bias,    // [out_ch]
                            __global float* out,            // [out_ch, Tout]
                            const int in_ch, const int Tpad, const int Tout,
                            const int k, const int stride, const int dil) {
  const int tb = get_global_id(0);
  const int o  = get_global_id(1);
  const int t0 = tb * 4;                         // outputs t0..t0+3 (stride==1 use)
  if (t0 >= Tout) return;
  float acc[4];
  acc[0] = acc[1] = acc[2] = acc[3] = bias[o];
  for (int kk = 0; kk < k; ++kk) {
    __global const float* wq = Wt + ((size_t)o * k + kk) * in_ch;
    float4 a4[4] = {(float4)(0.0f), (float4)(0.0f), (float4)(0.0f), (float4)(0.0f)};
    // Tail threads (t0+j >= Tout) must not read past Tpad — clamp the row;
    // their results are discarded at the write guard.
    int rows[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      int r = (t0 + j) * stride + kk * dil;
      rows[j] = (r < Tpad) ? r : (Tpad - 1);
    }
    for (int ic = 0; ic < in_ch; ic += 4) {
      const float4 wv = vload4(0, wq + ic);
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        a4[j] += wv * vload4(0, xpad_t + (size_t)rows[j] * in_ch + ic);
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      acc[j] += a4[j].x + a4[j].y + a4[j].z + a4[j].w;
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j)
    if (t0 + j < Tout) out[(size_t)o * Tout + (t0 + j)] = acc[j];
}

// gin over TRANSPOSED x (x_t[t*H+j]): the original gathers x[j*T+t] — four
// stride-T cache lines per a4 iteration. Same j-ascending 4-lane order →
// bit-identical output; gated by same-seed wav md5 like enc_convt1d_x4.
__kernel void enc_lstm_gin_x(__global const float* Wih,   // [4H, H]
                             __global const float* bih,    // [4H]
                             __global const float* bhh,    // [4H]
                             __global const float* xt,     // [T, H]
                             __global float* gin,          // [T, 4H]
                             const int H, const int T) {
  const int t = get_global_id(0);
  const int r = get_global_id(1);
  if (t >= T) return;
  __global const float* wr = Wih + (size_t)r * H;
  __global const float* xr = xt + (size_t)t * H;
  float4 a4 = (float4)(0.0f);
  for (int j = 0; j < H; j += 4)
    a4 += vload4(0, wr + j) * vload4(0, xr + j);
  gin[(size_t)t * 4 * H + r] = bih[r] + bhh[r] + a4.x + a4.y + a4.z + a4.w;
}
