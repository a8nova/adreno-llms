// GPU text encoder kernels for VITS. H=192, Nh=2, D=96, W=4, FFN=768.
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

// Embedding gather + scale: out[t, h] = emb[id[t], h] * scale
__kernel void te_embed_scale(__global const storage_t* emb,
                             __global const int* ids,
                             __global storage_t* out,
                             const int T, const int H,
                             const float scale) {
  int t = get_global_id(0);
  int h = get_global_id(1);
  if (t >= T || h >= H) return;
  int id = ids[t];
  STORE(out, t * H + h, LOAD(emb, id * H + h) * scale);
}

// LayerNorm fused with residual add: x[t,:] = LN(x[t,:] + residual[t,:])
// One workitem per timestep. H=192 is small enough for a sequential reduction.
__kernel void te_layer_norm_residual(
    __global storage_t* x,
    __global const storage_t* residual,
    __global const storage_t* gamma,
    __global const storage_t* beta,
    const int T, const int H, const float eps) {
  int t = get_global_id(0);
  if (t >= T) return;
  int base = t * H;
  float sum = 0.0f;
  for (int h = 0; h < H; ++h) {
    float v = LOAD(x, base + h) + LOAD(residual, base + h);
    STORE(x, base + h, v);
    sum += v;
  }
  float mean = sum / (float)H;
  float var = 0.0f;
  for (int h = 0; h < H; ++h) {
    float d = LOAD(x, base + h) - mean;
    var += d * d;
  }
  float inv = 1.0f / sqrt(var / (float)H + eps);
  for (int h = 0; h < H; ++h) {
    float v = (LOAD(x, base + h) - mean) * inv;
    STORE(x, base + h, v * LOAD(gamma, h) + LOAD(beta, h));
  }
}

// Optimized fused attention per Qualcomm Adreno guide:
//   §10.2.2: vload_half4 + dot() for 4x fewer memory transactions
//   §10.3.6: branchless rel_pos via select/multiply (no wave divergence)
//   §8.3:    native_exp/native_recip for hardware-accelerated softmax
//   §6.4:    __constant for rel_k/rel_v (zero-latency broadcast)
//   §8.12:   mul24 for index math (single-instruction 24-bit multiply)
__kernel void te_fused_attention(
    __global const storage_t* QKV,     // [T, 3H]
    __constant storage_t* rel_k        // [2W+1, D] — on-chip constant RAM
        __attribute__((max_constant_size(1728))),
    __constant storage_t* rel_v        // [2W+1, D]
        __attribute__((max_constant_size(1728))),
    __global storage_t* ctx,           // [T, H]
    const int T, const int H, const int D,
    const int Nh, const int W, const float scale) {
  const int q = get_global_id(0);
  const int hd = get_global_id(1);
  if (q >= T || hd >= H) return;
  const int h = hd / D;
  const int d = hd % D;
  const int H3 = mul24(3, H);
  const int q_base = mad24(q, H3, mul24(h, D));
  const int k_off  = H;

  float scores[512];
  float mx = -1e30f;
  for (int k = 0; k < T; ++k) {
    float s = 0.0f;
    const int k_base = mad24(k, H3, mul24(h, D)) + k_off;
#ifdef USE_FP16
    // Vectorized dot product: D=96 = 24 × vload_half4
    for (int dd = 0; dd < D; dd += 4) {
      float4 qv = vload_half4(0, (__global const half*)QKV + q_base + dd);
      float4 kv = vload_half4(0, (__global const half*)QKV + k_base + dd);
      s += dot(qv, kv);
    }
#else
    for (int dd = 0; dd < D; ++dd)
      s += LOAD(QKV, q_base + dd) * LOAD(QKV, k_base + dd);
#endif
    s *= scale;

    // Branchless relative position bias (§10.3.6: eliminate divergence)
    const int offset = k - q;
    const float in_window = (offset >= -W && offset <= W) ? 1.0f : 0.0f;
    if (in_window > 0.0f) {
      float rs = 0.0f;
      const int roff = (offset + W) * D;
#ifdef USE_FP16
      for (int dd = 0; dd < D; dd += 4) {
        float4 qv = vload_half4(0, (__global const half*)QKV + q_base + dd);
        float4 rv = vload_half4(0, (__constant const half*)rel_k + roff + dd);
        rs += dot(qv, rv);
      }
#else
      for (int dd = 0; dd < D; ++dd)
        rs += LOAD(QKV, q_base + dd) * LOAD(rel_k, roff + dd);
#endif
      s += rs * scale;
    }
    scores[k] = s;
    mx = fmax(mx, s);
  }

  // Softmax with native_exp (§8.3: hardware EFU)
  float sum = 0.0f;
  for (int k = 0; k < T; ++k) {
    scores[k] = native_exp(scores[k] - mx);
    sum += scores[k];
  }
  const float inv = native_recip(sum);

  float out = 0.0f;
  const int v_off = mul24(2, H);
  for (int k = 0; k < T; ++k) {
    float a = scores[k] * inv;
    float v = LOAD(QKV, mad24(k, H3, v_off + hd));
    const int offset = k - q;
    if (offset >= -W && offset <= W)
      v += LOAD(rel_v, mad24(offset + W, D, d));
    out += a * v;
  }
  STORE(ctx, mad24(q, H, hd), out);
}

// Transpose [rows, cols] → [cols, rows]
__kernel void te_transpose(__global const storage_t* in,
                           __global storage_t* out,
                           const int rows, const int cols) {
  int r = get_global_id(0);
  int c = get_global_id(1);
  if (r >= rows || c >= cols) return;
  STORE(out, c * rows + r, LOAD(in, r * cols + c));
}

// In-place ReLU
__kernel void te_relu(__global storage_t* x, const int N) {
  int i = get_global_id(0);
  if (i >= N) return;
  float v = LOAD(x, i);
  if (v < 0.0f) STORE(x, i, 0.0f);
}

// Copy: dst = src (used for saving residual before in-place ops)
__kernel void te_copy(__global const storage_t* src,
                      __global storage_t* dst, const int N) {
  int i = get_global_id(0);
  if (i >= N) return;
  STORE(dst, i, LOAD(src, i));
}

// Row-major bias broadcast: y[t*OC + oc] += b[oc]
// Dispatch as [T, OC].
__kernel void te_bias_rowmajor(__global storage_t* y,
                               __global const storage_t* b,
                               const int T, const int OC) {
  int t = get_global_id(0);
  int oc = get_global_id(1);
  if (t >= T || oc >= OC) return;
  int idx = t * OC + oc;
  STORE(y, idx, LOAD(y, idx) + LOAD(b, oc));
}
