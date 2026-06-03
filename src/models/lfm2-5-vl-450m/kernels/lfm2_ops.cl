// Reference: transformers/models/lfm2/modeling_lfm2.py Lfm2RMSNorm, Lfm2MLP, Lfm2ShortConv, Lfm2Attention
// Utility kernels for the LFM2 language-model blocks.
#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)     vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)     ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
#endif

// Track 4 attempted but reverted: although cl_khr_subgroups appears in the
// device extension string, the Adreno 619v2 OpenCL 2.0 kernel compiler refused
// the #pragma + sub_group_reduce_add ("OpenCL 2.0 built-in is not supported").
// Need cl_qcom_subgroup_shuffle path instead — deferred.

__kernel void gather_embedding(__global const int* ids,
                               __global const storage_t* table,
                               __global storage_t* out,
                               const int rows,
                               const int hidden) {
  int gid = get_global_id(0);
  int total = rows * hidden;
  if (gid >= total) return;
  int row = gid / hidden;
  int col = gid - row * hidden;
  int tok = ids[row];
  STORE(out, gid, LOAD(table, tok * hidden + col));
}

// int8 gather: same as gather_embedding but reads int8 table + per-row fp16 scale.
#ifdef USE_FP16
__kernel void gather_embedding_int8(__global const int* ids,
                                    __global const char* table_int8,
                                    __global const half* scales,
                                    __global half* out,
                                    const int rows,
                                    const int hidden) {
  int gid = get_global_id(0);
  int total = rows * hidden;
  if (gid >= total) return;
  int row = gid / hidden;
  int col = gid - row * hidden;
  int tok = ids[row];
  float scale = (float)vload_half(tok, scales);
  float v = (float)table_int8[tok * hidden + col] * scale;
  vstore_half(v, gid, out);
}
#endif

__kernel void bias_add(__global storage_t* x,
                       __global const storage_t* b,
                       const int rows,
                       const int cols) {
  int gid = get_global_id(0);
  int total = rows * cols;
  if (gid >= total) return;
  int col = gid - (gid / cols) * cols;
  STORE(x, gid, LOAD(x, gid) + LOAD(b, col));
}

__kernel void gelu_tanh(__global const storage_t* input,
                        __global storage_t* output,
                        const int n) {
  int gid = get_global_id(0);
  if (gid >= n) return;
  float x = LOAD(input, gid);
  float y = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
  STORE(output, gid, y);
}

__kernel void pixel_unshuffle2_add_pos(__global const storage_t* patch_emb,
                                       __global const storage_t* pos,
                                       __global storage_t* out,
                                       const int grid,
                                       const int hidden) {
  int gid = get_global_id(0);
  const int out_grid = grid / 2;
  const int n_out = out_grid * out_grid;
  const int out_cols = hidden * 4;
  const int total = n_out * out_cols;
  if (gid >= total) return;
  int col = gid % out_cols;
  int p = gid / out_cols;
  int out_y = p / out_grid;
  int out_x = p - out_y * out_grid;
  int sub = col / hidden;
  int h = col - sub * hidden;
  int dy = sub / 2;
  int dx = sub - dy * 2;
  int in_y = out_y * 2 + dy;
  int in_x = out_x * 2 + dx;
  int in_p = in_y * grid + in_x;
  STORE(out, gid, LOAD(patch_emb, in_p * hidden + h) + LOAD(pos, p * hidden + h));
}

// 32-WI workgroup reduction over each row. cols is always a multiple of 4
// (hidden_size=1024). Dispatch: global=(rows*32, 1, 1), local=(32, 1, 1).
#ifdef USE_FP16
// Track — fused residual_add + rms_norm. Replaces two separate dispatches
// (element_add → rms_norm) with one. Writes BOTH the combined a+b (consumed by
// the NEXT layer's residual add) AND the normalized version (consumed by the
// downstream GEMV). Saves one dispatch per layer-end pair × 16 layers × 2
// residuals per layer = ~32 dispatches per decode token.
__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void rms_norm_rows_add(__global const half* a,
                       __global const half* b,
                       __global const half* weight,
                       __global half* out_sum,
                       __global half* out_norm,
                       const int rows,
                       const int cols,
                       const float eps) {
  int row = get_group_id(0);
  if (row >= rows) return;
  int lid = get_local_id(0);
  int C4 = cols >> 2;
  __global const half4* a4 = (__global const half4*)(a + row * cols);
  __global const half4* b4 = (__global const half4*)(b + row * cols);
  __global       half4* s4 = (__global       half4*)(out_sum + row * cols);
  __global const half4* w4 = (__global const half4*)weight;
  __global       half4* o4 = (__global       half4*)(out_norm + row * cols);

  // Pass 1 — combined = a + b, store to out_sum, accumulate sum-of-squares.
  float ss = 0.0f;
  for (int c = lid; c < C4; c += 32) {
    float4 af = convert_float4(a4[c]);
    float4 bf = convert_float4(b4[c]);
    float4 cf = af + bf;
    s4[c] = convert_half4(cf);
    ss += cf.s0*cf.s0 + cf.s1*cf.s1 + cf.s2*cf.s2 + cf.s3*cf.s3;
  }
  __local float lsum[32];
  lsum[lid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 16; s > 0; s >>= 1) {
    if (lid < s) lsum[lid] += lsum[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float inv = rsqrt(lsum[0] / (float)cols + eps);

  // Pass 2 — re-load combined from out_sum (in L2 from pass 1) and normalize.
  for (int c = lid; c < C4; c += 32) {
    float4 sf = convert_float4(s4[c]);
    float4 wf = convert_float4(w4[c]);
    o4[c] = convert_half4(sf * inv * wf);
  }
}

__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void rms_norm_rows(__global const half* input,
                   __global const half* weight,
                   __global half* output,
                   const int rows,
                   const int cols,
                   const float eps) {
  int row = get_group_id(0);
  if (row >= rows) return;
  int lid = get_local_id(0);
  int C4 = cols >> 2;
  __global const half4* in4  = (__global const half4*)(input + row * cols);
  __global const half4* w4   = (__global const half4*)weight;
  __global       half4* out4 = (__global       half4*)(output + row * cols);

  float ss = 0.0f;
  for (int c = lid; c < C4; c += 32) {
    float4 vf = convert_float4(in4[c]);
    ss += vf.s0*vf.s0 + vf.s1*vf.s1 + vf.s2*vf.s2 + vf.s3*vf.s3;
  }
  __local float lsum[32];
  lsum[lid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 16; s > 0; s >>= 1) {
    if (lid < s) lsum[lid] += lsum[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float inv = rsqrt(lsum[0] / (float)cols + eps);

  for (int c = lid; c < C4; c += 32) {
    float4 vf = convert_float4(in4[c]);
    float4 wf = convert_float4(w4[c]);
    out4[c] = convert_half4(vf * inv * wf);
  }
}

// Per-head RMSNorm. head_dim=64 (16 half4s per head). Dispatch:
// global=(rows*heads*32, 1, 1), local=(32, 1, 1).
__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void rms_norm_heads(__global const half* input,
                    __global const half* weight,
                    __global half* output,
                    const int rows,
                    const int heads,
                    const int head_dim,
                    const float eps) {
  int rh = get_group_id(0);
  int total = rows * heads;
  if (rh >= total) return;
  int lid = get_local_id(0);
  int D4 = head_dim >> 2;
  __global const half4* in4  = (__global const half4*)(input + rh * head_dim);
  __global const half4* w4   = (__global const half4*)weight;
  __global       half4* out4 = (__global       half4*)(output + rh * head_dim);

  float ss = 0.0f;
  for (int d = lid; d < D4; d += 32) {
    float4 vf = convert_float4(in4[d]);
    ss += vf.s0*vf.s0 + vf.s1*vf.s1 + vf.s2*vf.s2 + vf.s3*vf.s3;
  }
  __local float lsum[32];
  lsum[lid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 16; s > 0; s >>= 1) {
    if (lid < s) lsum[lid] += lsum[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float inv = rsqrt(lsum[0] / (float)head_dim + eps);

  for (int d = lid; d < D4; d += 32) {
    float4 vf = convert_float4(in4[d]);
    float4 wf = convert_float4(w4[d]);
    out4[d] = convert_half4(vf * inv * wf);
  }
}
#else
__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void rms_norm_rows(__global const float* input,
                   __global const float* weight,
                   __global float* output,
                   const int rows,
                   const int cols,
                   const float eps) {
  int row = get_group_id(0);
  if (row >= rows) return;
  int lid = get_local_id(0);
  float ss = 0.0f;
  for (int c = lid; c < cols; c += 32) ss += input[row * cols + c] * input[row * cols + c];
  __local float lsum[32];
  lsum[lid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 16; s > 0; s >>= 1) { if (lid < s) lsum[lid] += lsum[lid + s]; barrier(CLK_LOCAL_MEM_FENCE); }
  float inv = rsqrt(lsum[0] / (float)cols + eps);
  for (int c = lid; c < cols; c += 32) output[row * cols + c] = input[row * cols + c] * inv * weight[c];
}

__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void rms_norm_heads(__global const float* input,
                    __global const float* weight,
                    __global float* output,
                    const int rows,
                    const int heads,
                    const int head_dim,
                    const float eps) {
  int rh = get_group_id(0);
  if (rh >= rows * heads) return;
  int lid = get_local_id(0);
  float ss = 0.0f;
  for (int d = lid; d < head_dim; d += 32) ss += input[rh * head_dim + d] * input[rh * head_dim + d];
  __local float lsum[32];
  lsum[lid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 16; s > 0; s >>= 1) { if (lid < s) lsum[lid] += lsum[lid + s]; barrier(CLK_LOCAL_MEM_FENCE); }
  float inv = rsqrt(lsum[0] / (float)head_dim + eps);
  for (int d = lid; d < head_dim; d += 32) output[rh * head_dim + d] = input[rh * head_dim + d] * inv * weight[d];
}
#endif

__kernel void swiglu(__global const storage_t* gate,
                     __global const storage_t* up,
                     __global storage_t* out,
                     const int n) {
  int gid = get_global_id(0);
  if (gid >= n) return;
  float g = LOAD(gate, gid);
  float u = LOAD(up, gid);
  STORE(out, gid, (g / (1.0f + native_exp(-g))) * u);
}

__kernel void mul_buffers(__global const storage_t* a,
                          __global const storage_t* b,
                          __global storage_t* out,
                          const int n) {
  int gid = get_global_id(0);
  if (gid >= n) return;
  STORE(out, gid, LOAD(a, gid) * LOAD(b, gid));
}

// Residual add used by Lfm2DecoderLayer.forward:
//   hidden_states = hidden_states + residual
//   hidden_states = hidden_states + self.feed_forward(...)
// Host element_add() copies the residual/input into arg0 first, then dispatches
// this in-place add kernel against arg1.
__kernel void element_add(__global storage_t* a,
                          __global const storage_t* b,
                          const int n) {
  int gid = get_global_id(0);
  if (gid >= n) return;
  STORE(a, gid, LOAD(a, gid) + LOAD(b, gid));
}

// Track: fused 3-buffer add — out[i] = a[i] + b[i] — eliminates the
// clCreateBuffer + clEnqueueCopyBuffer + clCreateKernel sequence inside the
// element_add host helper which dispatches ~32x per decode token. Saves the
// per-call copy step.
__kernel void element_add3(__global const storage_t* a,
                           __global const storage_t* b,
                           __global storage_t* out,
                           const int n) {
  int gid = get_global_id(0);
  if (gid >= n) return;
  STORE(out, gid, LOAD(a, gid) + LOAD(b, gid));
}

// in_proj output is [T, 3H] containing [B, C, x]. Compute B*x and C as contiguous [T,H].
__kernel void lfm2_split_bcx(__global const storage_t* bcx,
                             __global storage_t* bx,
                             __global storage_t* cbuf,
                             const int rows,
                             const int hidden) {
  int gid = get_global_id(0);
  int total = rows * hidden;
  if (gid >= total) return;
  int row = gid / hidden;
  int col = gid - row * hidden;
  int base = row * (3 * hidden) + col;
  float B = LOAD(bcx, base);
  float C = LOAD(bcx, base + hidden);
  float x = LOAD(bcx, base + 2 * hidden);
  STORE(bx, gid, B * x);
  STORE(cbuf, gid, C);
}

// Depthwise causal conv on B*x. PyTorch Conv1d has padding=L-1 then slices [:seqlen].
// Weight layout is [hidden, 1, K] contiguous as [hidden, K].
// Depthwise conv with optional left-pad cache. When start_pos == 0 (prefill or
// no-cache) the cache_bx buffer is ignored and left-pad is zero — matches the
// reference Conv1d(padding=ksize-1)[:, :, :seqlen] semantics. When start_pos > 0
// (incremental decode) cache_bx[pad_idx * hidden + c] supplies the (pad - pad_idx)
// most-recently-seen bx rows (i.e. cache_bx[0] is bx[start_pos - pad],
// cache_bx[pad-1] is bx[start_pos - 1]).
// Track 5 — start_pos comes from counter[0] (buffer-based, recordable).
__kernel void lfm2_depthwise_conv3(__global const storage_t* bx,
                                   __global const storage_t* cache_bx,
                                   __global const storage_t* w,
                                   __global storage_t* out,
                                   const int rows,
                                   const int hidden,
                                   const int ksize,
                                   __global const int* counter) {
  int start_pos = counter[0];
  int gid = get_global_id(0);
  int total = rows * hidden;
  if (gid >= total) return;
  int t = gid / hidden;
  int c = gid - t * hidden;
  float acc = 0.0f;
  int pad = ksize - 1;
  for (int k = 0; k < ksize; ++k) {
    int src_t = t - pad + k;          // logical position relative to bx[0]
    float bx_v;
    if (src_t >= 0 && src_t < rows) {
      bx_v = LOAD(bx, src_t * hidden + c);
    } else if (src_t < 0 && start_pos > 0) {
      // src_t in [-pad, -1] maps to cache_bx[pad + src_t] (most-recent at pad-1)
      int cache_idx = pad + src_t;
      bx_v = LOAD(cache_bx, cache_idx * hidden + c);
    } else {
      bx_v = 0.0f;
    }
    acc += bx_v * LOAD(w, c * ksize + k);
  }
  STORE(out, gid, acc);
}

// Reshape [T, QH*D] / [T, KVH*D] projections into head-major [heads, T, D].
__kernel void seq_to_heads(__global const storage_t* in,
                           __global storage_t* out,
                           const int rows,
                           const int heads,
                           const int head_dim) {
  int gid = get_global_id(0);
  int total = rows * heads * head_dim;
  if (gid >= total) return;
  int d = gid % head_dim;
  int tmp = gid / head_dim;
  int h = tmp % heads;
  int t = tmp / heads;
  int in_idx = t * (heads * head_dim) + h * head_dim + d;
  int out_idx = h * rows * head_dim + t * head_dim + d;
  STORE(out, out_idx, LOAD(in, in_idx));
}

__kernel void heads_to_seq(__global const storage_t* in,
                           __global storage_t* out,
                           const int rows,
                           const int heads,
                           const int head_dim) {
  int gid = get_global_id(0);
  int total = rows * heads * head_dim;
  if (gid >= total) return;
  int d = gid % head_dim;
  int tmp = gid / head_dim;
  int h = tmp % heads;
  int t = tmp / heads;
  int in_idx = h * rows * head_dim + t * head_dim + d;
  int out_idx = t * (heads * head_dim) + h * head_dim + d;
  STORE(out, out_idx, LOAD(in, in_idx));
}

// LFM2 rotate_half uses [-second_half, first_half], not even/odd interleave.
// Track 5 — start_pos comes from counter[0] (buffer-based, recordable).
__kernel void apply_rope_lfm2(__global storage_t* q,
                              __global storage_t* k,
                              const int rows,
                              const int q_heads,
                              const int kv_heads,
                              const int head_dim,
                              __global const int* counter,
                              const float theta) {
  int start_pos = counter[0];
  int gid = get_global_id(0);
  int half_dim = head_dim / 2;
  int q_total = rows * q_heads * half_dim;
  int k_total = rows * kv_heads * half_dim;
  if (gid < q_total) {
    int d = gid % half_dim;
    int tmp = gid / half_dim;
    int t = tmp % rows;
    int h = tmp / rows;
    int base = h * rows * head_dim + t * head_dim;
    float angle = (float)(start_pos + t) / pow(theta, (2.0f * (float)d) / (float)head_dim);
    float c = cos(angle);
    float s = sin(angle);
    float x1 = LOAD(q, base + d);
    float x2 = LOAD(q, base + d + half_dim);
    STORE(q, base + d, x1 * c - x2 * s);
    STORE(q, base + d + half_dim, x2 * c + x1 * s);
  }
  if (gid < k_total) {
    int d = gid % half_dim;
    int tmp = gid / half_dim;
    int t = tmp % rows;
    int h = tmp / rows;
    int base = h * rows * head_dim + t * head_dim;
    float angle = (float)(start_pos + t) / pow(theta, (2.0f * (float)d) / (float)head_dim);
    float c = cos(angle);
    float s = sin(angle);
    float x1 = LOAD(k, base + d);
    float x2 = LOAD(k, base + d + half_dim);
    STORE(k, base + d, x1 * c - x2 * s);
    STORE(k, base + d + half_dim, x2 * c + x1 * s);
  }
}

// One work-item computes one (query-head, query-token, key-token) score.
// Q is laid out [q_heads, q_rows, head_dim]; K is in a persistent KV cache
// [kv_heads, k_stride, head_dim] of which only the first k_rows positions
// (absolute time indices 0..k_rows-1) are valid this call. start_pos is the
// absolute time index of the first Q row. Scores are written [q_heads, q_rows, k_rows].
// head_dim=64 always for LFM2 (16 head_dim/4 vector iterations); vectorized half4.
#ifdef USE_FP16
__kernel void lfm2_attn_scores(__global const half* q,
                               __global const half* k,
                               __global half* scores,
                               const int q_rows,
                               const int k_rows,
                               const int k_stride,
                               const int q_heads,
                               const int kv_heads,
                               const int head_dim,
                               const int start_pos,
                               const float scale) {
  int gid = get_global_id(0);
  int total = q_heads * q_rows * k_rows;
  if (gid >= total) return;
  int tk = gid % k_rows;
  int tmp = gid / k_rows;
  int tq = tmp % q_rows;
  int qh = tmp / q_rows;
  int kvh = qh / (q_heads / kv_heads);
  int D4 = head_dim >> 2;
  __global const half4* qv = (__global const half4*)(q + qh * q_rows * head_dim + tq * head_dim);
  __global const half4* kv = (__global const half4*)(k + kvh * k_stride * head_dim + tk * head_dim);
  float4 acc4 = (float4)(0.0f);
  for (int d = 0; d < D4; ++d) {
    acc4 += convert_float4(qv[d]) * convert_float4(kv[d]);
  }
  float acc = acc4.s0 + acc4.s1 + acc4.s2 + acc4.s3;
  int abs_q = start_pos + tq;
  int abs_k = tk;
  float v = (abs_k > abs_q) ? -INFINITY : acc * scale;
  vstore_half(v, gid, scores);
}
#else
__kernel void lfm2_attn_scores(__global const float* q,
                               __global const float* k,
                               __global float* scores,
                               const int q_rows,
                               const int k_rows,
                               const int k_stride,
                               const int q_heads,
                               const int kv_heads,
                               const int head_dim,
                               const int start_pos,
                               const float scale) {
  int gid = get_global_id(0);
  int total = q_heads * q_rows * k_rows;
  if (gid >= total) return;
  int tk = gid % k_rows;
  int tmp = gid / k_rows;
  int tq = tmp % q_rows;
  int qh = tmp / q_rows;
  int kvh = qh / (q_heads / kv_heads);
  float acc = 0.0f;
  int qbase = qh * q_rows * head_dim + tq * head_dim;
  int kbase = kvh * k_stride * head_dim + tk * head_dim;
  for (int d = 0; d < head_dim; ++d) acc += q[qbase + d] * k[kbase + d];
  int abs_q = start_pos + tq;
  int abs_k = tk;
  float v = (abs_k > abs_q) ? -INFINITY : acc * scale;
  scores[gid] = v;
}
#endif

__kernel void lfm2_softmax_rows(__global storage_t* scores,
                                const int total_rows,
                                const int cols) {
  int row = get_global_id(0);
  if (row >= total_rows) return;
  int base = row * cols;
  float mx = -INFINITY;
  for (int i = 0; i < cols; ++i) {
    float v = LOAD(scores, base + i);
    mx = fmax(mx, v);
  }
  float sum = 0.0f;
  for (int i = 0; i < cols; ++i) {
    float e = native_exp(LOAD(scores, base + i) - mx);
    sum += e;
    STORE(scores, base + i, e);
  }
  float inv = 1.0f / fmax(sum, 1.0e-20f);
  for (int i = 0; i < cols; ++i) STORE(scores, base + i, LOAD(scores, base + i) * inv);
}

// probs is [q_heads, q_rows, k_rows]; V is [kv_heads, v_stride, head_dim] of
// which the first k_rows positions are valid. Output is [q_heads, q_rows, head_dim].
__kernel void lfm2_attn_apply(__global const storage_t* probs,
                              __global const storage_t* v,
                              __global storage_t* out_heads,
                              const int q_rows,
                              const int k_rows,
                              const int v_stride,
                              const int q_heads,
                              const int kv_heads,
                              const int head_dim) {
  int gid = get_global_id(0);
  int total = q_heads * q_rows * head_dim;
  if (gid >= total) return;
  int d = gid % head_dim;
  int tmp = gid / head_dim;
  int tq = tmp % q_rows;
  int qh = tmp / q_rows;
  int kvh = qh / (q_heads / kv_heads);
  float acc = 0.0f;
  int pbase = (qh * q_rows + tq) * k_rows;
  int vbase = kvh * v_stride * head_dim + d;
  for (int tk = 0; tk < k_rows; ++tk) {
    acc += LOAD(probs, pbase + tk) * LOAD(v, vbase + tk * head_dim);
  }
  STORE(out_heads, gid, acc);
}

// Matrix-vector product: y[N] = W[N, K] @ x[K], with W stored row-major
// (PyTorch nn.Linear weight layout). Used as the pytorch_linear M==1 fast path:
// skips CLBlast's per-call CPU overhead (~25ms per HGEMM dispatch on Android),
// which dominates the decode-step host cost when ~100 projections fire per token.
//
// Workgroup-level reduction: 32 work-items per output row cooperate on
// K/4 half4 chunks then reduce in local memory. Adreno 6xx prefers half4
// vector loads (better cache-line utilization than scalar vload_half) AND
// benefits from finer-grained parallelism than 1-WI-per-row. K is always a
// multiple of 4 in this model (head_dim=64, hidden=1024, intermediate=6656,
// vocab=65536). Dispatch: global=(N*32, 1, 1), local=(32, 1, 1).
// Fused 3-output GEMV: when 3 PyTorch nn.Linear modules share input x[K], we
// dispatch ONE kernel with global=(N_a+N_b+N_c)*64 work-items instead of three
// gemv_pytorch_linear calls. The shared x[K] gets read once from DRAM per cache
// warmup; subsequent reads hit L1. Saves 2 kernel launches per fusion point —
// applied to Q+K+V projection (attention), w1+w3 (MLP gate+up), and B+C+x
// (conv in_proj). The 3 weight matrices may have different N dimensions
// (e.g. QKV with GQA: N_q=1024, N_k=N_v=512). Outputs are written to 3
// separate buffers; the kernel uses group_id to pick which (W, y) it serves.
#ifdef USE_FP16
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_fused3(__global const half* x,    // [K]
                 __global const half* wa,   // [N_a, K]
                 __global const half* wb,   // [N_b, K]
                 __global const half* wc,   // [N_c, K]
                 __global half* ya,         // [N_a]
                 __global half* yb,         // [N_b]
                 __global half* yc,         // [N_c]
                 const int N_a,
                 const int N_b,
                 const int N_c,
                 const int K) {
  int n_global = get_group_id(0);
  int lid = get_local_id(0);
  // Pick which (W, y) this workgroup serves.
  int n_local;
  __global const half* w;
  __global half* y;
  if (n_global < N_a) {
    n_local = n_global;
    w = wa + (size_t)n_local * (size_t)K;
    y = ya;
  } else if (n_global < N_a + N_b) {
    n_local = n_global - N_a;
    w = wb + (size_t)n_local * (size_t)K;
    y = yb;
  } else {
    n_local = n_global - N_a - N_b;
    w = wc + (size_t)n_local * (size_t)K;
    y = yc;
  }
  int K4 = K >> 2;
  __global const half4* x4 = (__global const half4*)x;
  __global const half4* w4 = (__global const half4*)w;

  float4 acc4 = (float4)(0.0f);
  for (int k = lid; k < K4; k += 64) {
    acc4 += convert_float4(x4[k]) * convert_float4(w4[k]);
  }
  float acc = acc4.s0 + acc4.s1 + acc4.s2 + acc4.s3;

  __local float lsum[64];
  lsum[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 32; s > 0; s >>= 1) {
    if (lid < s) lsum[lid] += lsum[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) vstore_half(lsum[0], n_local, y);
}

// Fused 2-output GEMV (same input, 2 weight matrices). Used for MLP w1+w3.
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_fused2(__global const half* x,
                 __global const half* wa,
                 __global const half* wb,
                 __global half* ya,
                 __global half* yb,
                 const int N_a,
                 const int N_b,
                 const int K) {
  int n_global = get_group_id(0);
  int lid = get_local_id(0);
  int n_local;
  __global const half* w;
  __global half* y;
  if (n_global < N_a) {
    n_local = n_global;
    w = wa + (size_t)n_local * (size_t)K;
    y = ya;
  } else {
    n_local = n_global - N_a;
    w = wb + (size_t)n_local * (size_t)K;
    y = yb;
  }
  int K4 = K >> 2;
  __global const half4* x4 = (__global const half4*)x;
  __global const half4* w4 = (__global const half4*)w;
  float4 acc4 = (float4)(0.0f);
  for (int k = lid; k < K4; k += 64) acc4 += convert_float4(x4[k]) * convert_float4(w4[k]);
  float acc = acc4.s0 + acc4.s1 + acc4.s2 + acc4.s3;
  __local float lsum[64];
  lsum[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 32; s > 0; s >>= 1) { if (lid < s) lsum[lid] += lsum[lid + s]; barrier(CLK_LOCAL_MEM_FENCE); }
  if (lid == 0) vstore_half(lsum[0], n_local, y);
}
#endif

// int8 GEMV: y[N] = scale[n] * (W_int8[N, K] @ x[K]).  W is row-major int8;
// scales is per-output-row fp16. Used as the decode (M=1) fast path for
// quantized weights. Halves weight memory bandwidth vs fp16 GEMV.
#ifdef USE_FP16
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_int8_pytorch_linear(__global const half* x,
                              __global const char* w_int8,
                              __global const half* scales,
                              __global half* y,
                              const int N,
                              const int K) {
  int n = get_group_id(0);
  if (n >= N) return;
  int lid = get_local_id(0);
  int K4 = K >> 2;
  __global const char4* w4 = (__global const char4*)(w_int8 + n * K);
  __global const half4* x4 = (__global const half4*)x;
  float4 acc4 = (float4)(0.0f);
  for (int k = lid; k < K4; k += 64) {
    char4 wb = w4[k];
    float4 wv;
    wv.s0 = (float)wb.s0;
    wv.s1 = (float)wb.s1;
    wv.s2 = (float)wb.s2;
    wv.s3 = (float)wb.s3;
    acc4 += convert_float4(x4[k]) * wv;
  }
  float acc = acc4.s0 + acc4.s1 + acc4.s2 + acc4.s3;
  __local float lsum[64];
  lsum[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 32; s > 0; s >>= 1) {
    if (lid < s) lsum[lid] += lsum[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    float scale_n = (float)vload_half(n, scales);
    vstore_half(lsum[0] * scale_n, n, y);
  }
}

// Dequantize int8 weight matrix [N, K] back to fp16 using per-row scales.
// Used in the prefill path: when M > 1 we can't run CLBlast on int8 input, so
// we dequant the layer's weight into a scratch fp16 buffer and run the
// existing fp16 GEMM. One WI per output element. Reused per layer so the
// scratch buffer recycles.
__kernel void dequant_int8_to_fp16(__global const char* w_int8,
                                   __global const half* scales,
                                   __global half* w_fp16,
                                   const int N,
                                   const int K) {
  int gid = get_global_id(0);
  int total = N * K;
  if (gid >= total) return;
  int n = gid / K;
  float scale = (float)vload_half(n, scales);
  float v = (float)w_int8[gid] * scale;
  vstore_half(v, gid, w_fp16);
}
#endif

#ifdef USE_FP16
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_pytorch_linear(__global const half* x,     // [K]
                         __global const half* w,     // [N, K] row-major
                         __global half* y,           // [N]
                         const int N,
                         const int K) {
  int n = get_group_id(0);
  if (n >= N) return;
  int lid = get_local_id(0);
  // Track-7-lite — half8 (128-bit) coalesced loads per Qualcomm guide §7.2.2:
  // "vectorized load/store, 128-bit aligned" is the max coalesced size on
  // Adreno 6xx. Doubles bytes-per-load vs the previous half4 path. K must be
  // multiple of 8 for full-vector load (true for every LFM2 weight: hidden=1024,
  // intermediate=4608, kv_dim=512, head_dim=64, conv_inner=1536, vocab=65536).
  int K8 = K >> 3;
  __global const half8* x8 = (__global const half8*)x;
  __global const half8* w8 = (__global const half8*)(w + n * K);

  float8 acc8 = (float8)(0.0f);
  for (int k = lid; k < K8; k += 64) {
    half8 xv = x8[k];
    half8 wv = w8[k];
    acc8 += convert_float8(xv) * convert_float8(wv);
  }
  float acc = acc8.s0 + acc8.s1 + acc8.s2 + acc8.s3 +
              acc8.s4 + acc8.s5 + acc8.s6 + acc8.s7;

  // Tail when K not multiple of 8 (none of LFM2's K shapes hit this, but
  // safe for future weight shapes). K_remainder is < 8 halves; one WI handles.
  if (lid == 0 && (K & 7)) {
    int tail_start = (K8 << 3);
    for (int k = tail_start; k < K; ++k) {
      acc += (float)vload_half(k, x) * (float)vload_half(n * K + k, w);
    }
  }

  __local float lsum[64];
  lsum[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 32; s > 0; s >>= 1) {
    if (lid < s) lsum[lid] += lsum[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) vstore_half(lsum[0], n, y);
}
#else
__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void gemv_pytorch_linear(__global const float* x,
                         __global const float* w,
                         __global float* y,
                         const int N,
                         const int K) {
  int n = get_group_id(0);
  if (n >= N) return;
  int lid = get_local_id(0);
  float acc = 0.0f;
  for (int k = lid; k < K; k += 32) {
    acc += x[k] * w[n * K + k];
  }
  __local float lsum[32];
  lsum[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 16; s > 0; s >>= 1) {
    if (lid < s) lsum[lid] += lsum[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) y[n] = lsum[0];
}
#endif

// FLASH ATTENTION for the q_rows==1 (decode) case. Fuses scores → softmax → apply
// in a single kernel with online softmax (no scores buffer materialization). One
// workgroup per query head; LWS=64 = head_dim. Each WI is responsible for one
// output dim and cooperates on partial dot-products for scoring. ~3× fewer kernel
// dispatches per layer and significantly better cache behaviour than the
// scores+softmax+apply chain.
//
// Q: [q_heads, 1, head_dim=64]
// K, V: [kv_heads, k_stride, head_dim=64], valid rows [0, k_rows)
// out: [q_heads, 1, head_dim=64]
#ifdef USE_FP16
// Track 5 — k_rows comes from counter[1] (buffer-based, recordable).
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void lfm2_flash_attn_decode(
    __global const half* q,
    __global const half* k,
    __global const half* v,
    __global half* out_heads,
    __global const int* counter,
    const int k_stride,
    const int q_heads,
    const int kv_heads,
    const int head_dim,
    const float scale) {
  int k_rows = counter[1];
  int qh = get_group_id(0);
  int d = get_local_id(0);
  if (qh >= q_heads) return;
  int kvh = qh / (q_heads / kv_heads);

  float q_val = vload_half(qh * head_dim + d, q);

  __local float lreduce[64];
  float m = -INFINITY;
  float s = 0.0f;
  float o = 0.0f;

  // Track 3 — native_exp replaces exp for transcendental throughput.
  // Track 4 (subgroup reduction) deferred — driver rejected cl_khr_subgroups.
  for (int t = 0; t < k_rows; ++t) {
    int kv_off = kvh * k_stride * head_dim + t * head_dim + d;
    float k_val = vload_half(kv_off, k);
    lreduce[d] = q_val * k_val;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (d < 32) lreduce[d] += lreduce[d + 32];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (d < 16) lreduce[d] += lreduce[d + 16];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (d <  8) lreduce[d] += lreduce[d +  8];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (d <  4) lreduce[d] += lreduce[d +  4];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (d <  2) lreduce[d] += lreduce[d +  2];
    barrier(CLK_LOCAL_MEM_FENCE);
    float score = (lreduce[0] + lreduce[1]) * scale;

    float new_m = fmax(m, score);
    float a = native_exp(m - new_m);
    float b = native_exp(score - new_m);
    float v_val = vload_half(kv_off, v);
    o = o * a + b * v_val;
    s = s * a + b;
    m = new_m;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  vstore_half((half)(o / s), qh * head_dim + d, out_heads);
}
#else
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void lfm2_flash_attn_decode(
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global float* out_heads,
    __global const int* counter,
    const int k_stride,
    const int q_heads,
    const int kv_heads,
    const int head_dim,
    const float scale) {
  int k_rows = counter[1];
  int qh = get_group_id(0);
  int d = get_local_id(0);
  if (qh >= q_heads) return;
  int kvh = qh / (q_heads / kv_heads);
  float q_val = q[qh * head_dim + d];
  __local float lreduce[64];
  float m = -INFINITY, s = 0.0f, o = 0.0f;
  for (int t = 0; t < k_rows; ++t) {
    int kv_off = kvh * k_stride * head_dim + t * head_dim + d;
    lreduce[d] = q_val * k[kv_off];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (d < 32) lreduce[d] += lreduce[d + 32]; barrier(CLK_LOCAL_MEM_FENCE);
    if (d < 16) lreduce[d] += lreduce[d + 16]; barrier(CLK_LOCAL_MEM_FENCE);
    if (d <  8) lreduce[d] += lreduce[d +  8]; barrier(CLK_LOCAL_MEM_FENCE);
    if (d <  4) lreduce[d] += lreduce[d +  4]; barrier(CLK_LOCAL_MEM_FENCE);
    if (d <  2) lreduce[d] += lreduce[d +  2]; barrier(CLK_LOCAL_MEM_FENCE);
    float score = (lreduce[0] + lreduce[1]) * scale;
    float new_m = fmax(m, score);
    float a = exp(m - new_m), b = exp(score - new_m);
    o = o * a + b * v[kv_off];
    s = s * a + b;
    m = new_m;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  out_heads[qh * head_dim + d] = o / s;
}
#endif

// Append `new_rows` head-major K (or V) rows at absolute time position `dst_start`
// into a persistent KV cache laid out [heads, stride, head_dim].
// Source layout: [heads, new_rows, head_dim].
// Original scalar variant — used by op_lfm2_conv_block.cpp for conv-cache
// writeback (dst_start always 0). Kept scalar so non-decode call sites don't
// need the counter buffer.
__kernel void kv_cache_write(__global const storage_t* src,
                             __global storage_t* cache,
                             const int new_rows,
                             const int heads,
                             const int head_dim,
                             const int stride,
                             const int dst_start) {
  int gid = get_global_id(0);
  int total = heads * new_rows * head_dim;
  if (gid >= total) return;
  int d = gid % head_dim;
  int tmp = gid / head_dim;
  int t = tmp % new_rows;
  int h = tmp / new_rows;
  int src_idx = h * new_rows * head_dim + t * head_dim + d;
  int dst_idx = h * stride * head_dim + (dst_start + t) * head_dim + d;
  STORE(cache, dst_idx, LOAD(src, src_idx));
}

// Track 5 — buffer-based variant for KV cache writes in the attention block.
// dst_start comes from counter[0]. Recordable because the buffer's cl_mem
// handle is stable across replays (host rewrites contents per iteration).
__kernel void kv_cache_write_counter(__global const storage_t* src,
                                     __global storage_t* cache,
                                     const int new_rows,
                                     const int heads,
                                     const int head_dim,
                                     const int stride,
                                     __global const int* counter) {
  int dst_start = counter[0];
  int gid = get_global_id(0);
  int total = heads * new_rows * head_dim;
  if (gid >= total) return;
  int d = gid % head_dim;
  int tmp = gid / head_dim;
  int t = tmp % new_rows;
  int h = tmp / new_rows;
  int src_idx = h * new_rows * head_dim + t * head_dim + d;
  int dst_idx = h * stride * head_dim + (dst_start + t) * head_dim + d;
  STORE(cache, dst_idx, LOAD(src, src_idx));
}

// Tiled flash attention for prefill. BQ=4 query rows per WG, BK=64 key rows tiled.
// 16 WIs per Q row for reductions, all 64 WIs cooperate on P@V matmul.
// Local memory: ~11 KB per workgroup.
// Dispatch: global = q_heads * ceil(q_rows/4) * 64, local = 64.
#ifdef USE_FP16
#define FA_BQ 4
#define FA_BK 64
#define FA_WG 64
#define FA_HD 64
#define FA_WI_PER_ROW (FA_WG / FA_BQ)

__kernel __attribute__((reqd_work_group_size(FA_WG, 1, 1)))
void lfm2_flash_attn_prefill(
    __global const half* Q,
    __global const half* K_cache,
    __global const half* V_cache,
    __global half*       out_heads,
    const int q_rows,
    const int k_rows,
    const int k_stride,
    const int q_heads,
    const int kv_heads,
    const int head_dim,
    const int start_pos,
    const float scale)
{
    const int num_q_tiles = (q_rows + FA_BQ - 1) / FA_BQ;
    const int gid = get_group_id(0);
    const int qh = gid / num_q_tiles;
    const int qt = gid % num_q_tiles;
    const int lid = get_local_id(0);

    if (qh >= q_heads) return;

    const int kvh = qh / (q_heads / kv_heads);
    const int q_start = qt * FA_BQ;
    const int bq = min(FA_BQ, q_rows - q_start);

    const int my_row = lid / FA_WI_PER_ROW;
    const int lane   = lid % FA_WI_PER_ROW;

    __local half  Q_loc[FA_BQ * FA_HD];
    __local half  KV_loc[FA_BK * FA_HD];
    __local float S_loc[FA_BQ * FA_BK];
    __local float O_loc[FA_BQ * FA_HD];
    __local float m_loc[FA_BQ];
    __local float l_loc[FA_BQ];
    __local float red[FA_WG];

    for (int e = lid; e < FA_BQ * FA_HD; e += FA_WG)
        O_loc[e] = 0.0f;
    if (lid < FA_BQ) { m_loc[lid] = -INFINITY; l_loc[lid] = 0.0f; }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        __global const half* Qbase = Q + (size_t)qh * q_rows * FA_HD + (size_t)q_start * FA_HD;
        int qelems = bq * FA_HD;
        __global const half4* Qsrc = (__global const half4*)Qbase;
        __local  half4*       Qdst = (__local  half4*)Q_loc;
        int q4 = qelems >> 2;
        for (int i = lid; i < q4; i += FA_WG) Qdst[i] = Qsrc[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kk = 0; kk < k_rows; kk += FA_BK) {
        const int bk = min(FA_BK, k_rows - kk);

        {
            __global const half* Kbase = K_cache + (size_t)kvh * k_stride * FA_HD + (size_t)kk * FA_HD;
            int kelems = bk * FA_HD;
            __global const half4* Ksrc = (__global const half4*)Kbase;
            __local  half4*       Kdst = (__local  half4*)KV_loc;
            int k4 = kelems >> 2;
            for (int i = lid; i < k4; i += FA_WG) Kdst[i] = Ksrc[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // S = Q @ K^T (scaled, causal masked)
        int stotal = bq * bk;
        for (int s = lid; s < stotal; s += FA_WG) {
            int qi = s / bk;
            int kj = s - qi * bk;
            __local const half4* qv = (__local const half4*)(Q_loc + qi * FA_HD);
            __local const half4* kv = (__local const half4*)(KV_loc + kj * FA_HD);
            float4 a4 = (float4)(0.0f);
            for (int d = 0; d < (FA_HD >> 2); ++d)
                a4 += convert_float4(qv[d]) * convert_float4(kv[d]);
            float dot = a4.s0 + a4.s1 + a4.s2 + a4.s3;
            int abs_q = start_pos + q_start + qi;
            int abs_k = kk + kj;
            S_loc[qi * FA_BK + kj] = (abs_k > abs_q) ? -INFINITY : dot * scale;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Load V tile (reuse KV_loc)
        {
            __global const half* Vbase = V_cache + (size_t)kvh * k_stride * FA_HD + (size_t)kk * FA_HD;
            int velems = bk * FA_HD;
            __global const half4* Vsrc = (__global const half4*)Vbase;
            __local  half4*       Vdst = (__local  half4*)KV_loc;
            int v4 = velems >> 2;
            for (int i = lid; i < v4; i += FA_WG) Vdst[i] = Vsrc[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Online softmax (16 WIs per Q row, 4 rows in parallel)
        if (my_row < bq) {
            const int qi = my_row;
            const int rbase = qi * FA_WI_PER_ROW;

            float lmax = -INFINITY;
            for (int j = lane; j < bk; j += FA_WI_PER_ROW)
                lmax = fmax(lmax, S_loc[qi * FA_BK + j]);
            red[lid] = lmax;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 8) red[rbase + lane] = fmax(red[rbase + lane], red[rbase + lane + 8]);
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 4) red[rbase + lane] = fmax(red[rbase + lane], red[rbase + lane + 4]);
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 2) red[rbase + lane] = fmax(red[rbase + lane], red[rbase + lane + 2]);
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 1) red[rbase + lane] = fmax(red[rbase + lane], red[rbase + lane + 1]);
            barrier(CLK_LOCAL_MEM_FENCE);

            float new_m = red[rbase];
            float old_m = m_loc[qi];
            float comb_m = fmax(old_m, new_m);
            float alpha = native_exp(old_m - comb_m);

            for (int d = lane; d < FA_HD; d += FA_WI_PER_ROW)
                O_loc[qi * FA_HD + d] *= alpha;

            float pl = 0.0f;
            for (int j = lane; j < bk; j += FA_WI_PER_ROW) {
                float ev = native_exp(S_loc[qi * FA_BK + j] - comb_m);
                S_loc[qi * FA_BK + j] = ev;
                pl += ev;
            }
            red[lid] = pl;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 8) red[rbase + lane] += red[rbase + lane + 8];
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 4) red[rbase + lane] += red[rbase + lane + 4];
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 2) red[rbase + lane] += red[rbase + lane + 2];
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 1) red[rbase + lane] += red[rbase + lane + 1];
            barrier(CLK_LOCAL_MEM_FENCE);

            if (lane == 0) {
                m_loc[qi] = comb_m;
                l_loc[qi] = l_loc[qi] * alpha + red[rbase];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // O += P @ V.  P is in S_loc, V is in KV_loc.
        for (int e = lid; e < bq * FA_HD; e += FA_WG) {
            int qi = e / FA_HD;
            int d  = e - qi * FA_HD;
            float o_acc = 0.0f;
            for (int j = 0; j < bk; ++j)
                o_acc += S_loc[qi * FA_BK + j] * convert_float(KV_loc[j * FA_HD + d]);
            O_loc[e] += o_acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int e = lid; e < bq * FA_HD; e += FA_WG) {
        int qi = e / FA_HD;
        int d  = e - qi * FA_HD;
        float inv_l = 1.0f / fmax(l_loc[qi], 1.0e-20f);
        int gidx = qh * q_rows * FA_HD + (q_start + qi) * FA_HD + d;
        vstore_half(O_loc[e] * inv_l, gidx, out_heads);
    }
}
#endif

// ──────────────────────────────────────────────────────────────────────
// SigLIP vision encoder kernels (LFM2-VL vision tower).
// ──────────────────────────────────────────────────────────────────────

// LayerNorm with subtractive mean + scale + bias. Two reduction passes:
//   pass 1: mean(x)
//   pass 2: var(x - mean)
//   then  : y = ((x - mean) * rsqrt(var + eps)) * w + b
// WG=32; cols must be multiple of 4 (cols=768 → 192 half4s).
// Dispatch: global=(rows*32, 1, 1), local=(32, 1, 1).
#ifdef USE_FP16
__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void layer_norm_rows_bias(__global const half* input,
                          __global const half* weight,
                          __global const half* bias,
                          __global half* output,
                          const int rows,
                          const int cols,
                          const float eps) {
  int row = get_group_id(0);
  if (row >= rows) return;
  int lid = get_local_id(0);
  int C4 = cols >> 2;
  __global const half4* in4 = (__global const half4*)(input + row * cols);
  __global const half4* w4  = (__global const half4*)weight;
  __global const half4* b4  = (__global const half4*)bias;
  __global       half4* o4  = (__global       half4*)(output + row * cols);
  __local float lscratch[32];

  // Pass 1 — mean.
  float sum = 0.0f;
  for (int c = lid; c < C4; c += 32) {
    float4 vf = convert_float4(in4[c]);
    sum += vf.s0 + vf.s1 + vf.s2 + vf.s3;
  }
  lscratch[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 16; s > 0; s >>= 1) {
    if (lid < s) lscratch[lid] += lscratch[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float mean = lscratch[0] / (float)cols;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Pass 2 — variance about mean.
  float ss = 0.0f;
  for (int c = lid; c < C4; c += 32) {
    float4 vf = convert_float4(in4[c]) - (float4)(mean);
    ss += vf.s0*vf.s0 + vf.s1*vf.s1 + vf.s2*vf.s2 + vf.s3*vf.s3;
  }
  lscratch[lid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 16; s > 0; s >>= 1) {
    if (lid < s) lscratch[lid] += lscratch[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float inv = rsqrt(lscratch[0] / (float)cols + eps);

  // Pass 3 — normalize, scale, shift.
  for (int c = lid; c < C4; c += 32) {
    float4 vf = (convert_float4(in4[c]) - (float4)(mean)) * inv;
    float4 wf = convert_float4(w4[c]);
    float4 bf = convert_float4(b4[c]);
    o4[c] = convert_half4(vf * wf + bf);
  }
}
#else
__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void layer_norm_rows_bias(__global const float* input,
                          __global const float* weight,
                          __global const float* bias,
                          __global float* output,
                          const int rows,
                          const int cols,
                          const float eps) {
  int row = get_group_id(0);
  if (row >= rows) return;
  int lid = get_local_id(0);
  __local float lscratch[32];

  float sum = 0.0f;
  for (int c = lid; c < cols; c += 32) sum += input[row * cols + c];
  lscratch[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 16; s > 0; s >>= 1) { if (lid < s) lscratch[lid] += lscratch[lid + s]; barrier(CLK_LOCAL_MEM_FENCE); }
  float mean = lscratch[0] / (float)cols;
  barrier(CLK_LOCAL_MEM_FENCE);

  float ss = 0.0f;
  for (int c = lid; c < cols; c += 32) {
    float d = input[row * cols + c] - mean;
    ss += d * d;
  }
  lscratch[lid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 16; s > 0; s >>= 1) { if (lid < s) lscratch[lid] += lscratch[lid + s]; barrier(CLK_LOCAL_MEM_FENCE); }
  float inv = rsqrt(lscratch[0] / (float)cols + eps);
  for (int c = lid; c < cols; c += 32) {
    output[row * cols + c] = (input[row * cols + c] - mean) * inv * weight[c] + bias[c];
  }
}
#endif

// SigLIP flash attention (bidirectional, with per-key-position pad mask).
// Mirrors lfm2_flash_attn_prefill but:
//   - NO causal mask (bidirectional)
//   - pad_mask[abs_k] == 0 ⇒ score = -INFINITY (padded key)
//   - GQA disabled (q_heads == kv_heads, both arms accept the same value)
// BQ=4 q rows per WG, BK=64 k rows per tile, WG=64.
#ifdef USE_FP16
#define SFA_BQ 4
#define SFA_BK 64
#define SFA_WG 64
#define SFA_HD 64
#define SFA_WI_PER_ROW (SFA_WG / SFA_BQ)

__kernel __attribute__((reqd_work_group_size(SFA_WG, 1, 1)))
void siglip_flash_attn_prefill(
    __global const half* Q,
    __global const half* K_in,
    __global const half* V_in,
    __global const int*  pad_mask,
    __global half*       out_heads,
    const int q_rows,
    const int k_rows,
    const int q_heads,
    const int kv_heads,
    const int head_dim,
    const float scale) {
    const int num_q_tiles = (q_rows + SFA_BQ - 1) / SFA_BQ;
    const int gid = get_group_id(0);
    const int qh = gid / num_q_tiles;
    const int qt = gid % num_q_tiles;
    const int lid = get_local_id(0);
    if (qh >= q_heads) return;

    const int kvh = qh / (q_heads / kv_heads);
    const int q_start = qt * SFA_BQ;
    const int bq = min(SFA_BQ, q_rows - q_start);

    const int my_row = lid / SFA_WI_PER_ROW;
    const int lane   = lid % SFA_WI_PER_ROW;

    __local half  Q_loc[SFA_BQ * SFA_HD];
    __local half  KV_loc[SFA_BK * SFA_HD];
    __local float S_loc[SFA_BQ * SFA_BK];
    __local float O_loc[SFA_BQ * SFA_HD];
    __local float m_loc[SFA_BQ];
    __local float l_loc[SFA_BQ];
    __local float red[SFA_WG];

    for (int e = lid; e < SFA_BQ * SFA_HD; e += SFA_WG) O_loc[e] = 0.0f;
    if (lid < SFA_BQ) { m_loc[lid] = -INFINITY; l_loc[lid] = 0.0f; }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        __global const half* Qbase = Q + (size_t)qh * q_rows * SFA_HD + (size_t)q_start * SFA_HD;
        int qelems = bq * SFA_HD;
        __global const half4* Qsrc = (__global const half4*)Qbase;
        __local  half4*       Qdst = (__local  half4*)Q_loc;
        int q4 = qelems >> 2;
        for (int i = lid; i < q4; i += SFA_WG) Qdst[i] = Qsrc[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kk = 0; kk < k_rows; kk += SFA_BK) {
        const int bk = min(SFA_BK, k_rows - kk);

        {
            __global const half* Kbase = K_in + (size_t)kvh * k_rows * SFA_HD + (size_t)kk * SFA_HD;
            int kelems = bk * SFA_HD;
            __global const half4* Ksrc = (__global const half4*)Kbase;
            __local  half4*       Kdst = (__local  half4*)KV_loc;
            int k4 = kelems >> 2;
            for (int i = lid; i < k4; i += SFA_WG) Kdst[i] = Ksrc[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int stotal = bq * bk;
        for (int s = lid; s < stotal; s += SFA_WG) {
            int qi = s / bk;
            int kj = s - qi * bk;
            __local const half4* qv = (__local const half4*)(Q_loc + qi * SFA_HD);
            __local const half4* kvv = (__local const half4*)(KV_loc + kj * SFA_HD);
            float4 a4 = (float4)(0.0f);
            for (int d = 0; d < (SFA_HD >> 2); ++d)
                a4 += convert_float4(qv[d]) * convert_float4(kvv[d]);
            float dot = a4.s0 + a4.s1 + a4.s2 + a4.s3;
            int abs_k = kk + kj;
            int valid = pad_mask[abs_k];
            S_loc[qi * SFA_BK + kj] = (valid == 0) ? -INFINITY : dot * scale;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        {
            __global const half* Vbase = V_in + (size_t)kvh * k_rows * SFA_HD + (size_t)kk * SFA_HD;
            int velems = bk * SFA_HD;
            __global const half4* Vsrc = (__global const half4*)Vbase;
            __local  half4*       Vdst = (__local  half4*)KV_loc;
            int v4 = velems >> 2;
            for (int i = lid; i < v4; i += SFA_WG) Vdst[i] = Vsrc[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (my_row < bq) {
            const int qi = my_row;
            const int rbase = qi * SFA_WI_PER_ROW;
            float lmax = -INFINITY;
            for (int j = lane; j < bk; j += SFA_WI_PER_ROW)
                lmax = fmax(lmax, S_loc[qi * SFA_BK + j]);
            red[lid] = lmax;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 8) red[rbase + lane] = fmax(red[rbase + lane], red[rbase + lane + 8]);
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 4) red[rbase + lane] = fmax(red[rbase + lane], red[rbase + lane + 4]);
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 2) red[rbase + lane] = fmax(red[rbase + lane], red[rbase + lane + 2]);
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 1) red[rbase + lane] = fmax(red[rbase + lane], red[rbase + lane + 1]);
            barrier(CLK_LOCAL_MEM_FENCE);

            float new_m = red[rbase];
            float old_m = m_loc[qi];
            float comb_m = fmax(old_m, new_m);
            // Guard for all-masked tile (new_m == -INFINITY): alpha = 1, no
            // updates this iteration.
            float alpha = (comb_m == -INFINITY) ? 1.0f : native_exp(old_m - comb_m);

            for (int d = lane; d < SFA_HD; d += SFA_WI_PER_ROW)
                O_loc[qi * SFA_HD + d] *= alpha;

            float pl = 0.0f;
            for (int j = lane; j < bk; j += SFA_WI_PER_ROW) {
                float sval = S_loc[qi * SFA_BK + j];
                float ev = (sval == -INFINITY || comb_m == -INFINITY) ? 0.0f : native_exp(sval - comb_m);
                S_loc[qi * SFA_BK + j] = ev;
                pl += ev;
            }
            red[lid] = pl;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 8) red[rbase + lane] += red[rbase + lane + 8];
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 4) red[rbase + lane] += red[rbase + lane + 4];
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 2) red[rbase + lane] += red[rbase + lane + 2];
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lane < 1) red[rbase + lane] += red[rbase + lane + 1];
            barrier(CLK_LOCAL_MEM_FENCE);

            if (lane == 0) {
                m_loc[qi] = comb_m;
                l_loc[qi] = l_loc[qi] * alpha + red[rbase];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int e = lid; e < bq * SFA_HD; e += SFA_WG) {
            int qi = e / SFA_HD;
            int d  = e - qi * SFA_HD;
            float o_acc = 0.0f;
            for (int j = 0; j < bk; ++j)
                o_acc += S_loc[qi * SFA_BK + j] * convert_float(KV_loc[j * SFA_HD + d]);
            O_loc[e] += o_acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int e = lid; e < bq * SFA_HD; e += SFA_WG) {
        int qi = e / SFA_HD;
        int d  = e - qi * SFA_HD;
        float inv_l = 1.0f / fmax(l_loc[qi], 1.0e-20f);
        int gidx = qh * q_rows * SFA_HD + (q_start + qi) * SFA_HD + d;
        vstore_half(O_loc[e] * inv_l, gidx, out_heads);
    }
}
#endif

// Pixel-unshuffle 2x2 on (H, W, C) → (H/2, W/2, 4C). Channel order per output
// pixel is [TL, TR, BL, BR] of the matching 2x2 input block, concatenated along
// the channel axis (matches projector reference shape).
__kernel void pixel_unshuffle_2d(__global const storage_t* in,
                                 __global storage_t* out,
                                 const int H,
                                 const int W,
                                 const int C) {
  int gid = get_global_id(0);
  const int Ho = H >> 1;
  const int Wo = W >> 1;
  const int Co = C * 4;
  const int total = Ho * Wo * Co;
  if (gid >= total) return;
  int k = gid % Co;
  int tmp = gid / Co;
  int j = tmp % Wo;
  int i = tmp / Wo;
  int sub = k / C;
  int h = k - sub * C;
  int dy = sub >> 1;
  int dx = sub & 1;
  int in_y = (i << 1) + dy;
  int in_x = (j << 1) + dx;
  STORE(out, gid, LOAD(in, (in_y * W + in_x) * C + h));
}

// Bilinear resize of a 16x16 position table to (H_out, W_out) along spatial
// dims, hidden dim preserved. PyTorch F.interpolate(mode='bilinear',
// align_corners=False, antialias=True): for upsampling antialias is a no-op
// and the formula reduces to standard 2x2-tap bilinear with
//   src = (out + 0.5) * (in / out) - 0.5
// clamped to [0, in-1]. C=hidden treated as multi-feature dim; H_in=W_in=16.
__kernel void bilinear_position_resize(__global const storage_t* in,
                                       __global storage_t* out,
                                       const int H_out,
                                       const int W_out,
                                       const int C) {
  int gid = get_global_id(0);
  const int total = H_out * W_out * C;
  if (gid >= total) return;
  int c = gid % C;
  int tmp = gid / C;
  int j = tmp % W_out;
  int i = tmp / W_out;
  const int H_in = 16;
  const int W_in = 16;
  float fy = ((float)i + 0.5f) * ((float)H_in / (float)H_out) - 0.5f;
  float fx = ((float)j + 0.5f) * ((float)W_in / (float)W_out) - 0.5f;
  int y0 = (int)floor(fy);
  int x0 = (int)floor(fx);
  int y1 = y0 + 1;
  int x1 = x0 + 1;
  float wy = fy - (float)y0;
  float wx = fx - (float)x0;
  y0 = max(0, min(H_in - 1, y0));
  y1 = max(0, min(H_in - 1, y1));
  x0 = max(0, min(W_in - 1, x0));
  x1 = max(0, min(W_in - 1, x1));
  float p00 = LOAD(in, (y0 * W_in + x0) * C + c);
  float p01 = LOAD(in, (y0 * W_in + x1) * C + c);
  float p10 = LOAD(in, (y1 * W_in + x0) * C + c);
  float p11 = LOAD(in, (y1 * W_in + x1) * C + c);
  float v = (1.0f - wy) * ((1.0f - wx) * p00 + wx * p01)
          + wy        * ((1.0f - wx) * p10 + wx * p11);
  STORE(out, gid, v);
}

// SigLIP padding fill: rows [num_valid..N_seq) get position-0's resized
// embedding (matches PyTorch reference: padded positions reuse pos_resized[0]).
__kernel void siglip_fill_padding(__global storage_t* x,
                                  __global const storage_t* pos_resized,
                                  const int num_valid,
                                  const int N_seq,
                                  const int hidden) {
  int gid = get_global_id(0);
  int npad = (N_seq - num_valid) * hidden;
  if (gid >= npad) return;
  int row = num_valid + gid / hidden;
  int col = gid - (gid / hidden) * hidden;
  STORE(x, row * hidden + col, LOAD(pos_resized, col));
}

// Plain GELU (erf-based) — used by the LFM2-VL multimodal projector.
// gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
__kernel void gelu_erf(__global const storage_t* input,
                       __global storage_t* output,
                       const int n) {
  int gid = get_global_id(0);
  if (gid >= n) return;
  float x = LOAD(input, gid);
  float y = 0.5f * x * (1.0f + erf(x * 0.7071067811865475f));
  STORE(output, gid, y);
}

// Element-wise add: out[i] = a[i] + b[i] (broadcast-free; both rows×cols).
// Used by the SigLIP encoder when combining patch_emb + resized_position.
// Same op as element_add3 — kept aliased for the encoder pipeline naming.
__kernel void siglip_add_pos(__global const storage_t* patch_emb,
                             __global const storage_t* pos_resized,
                             __global storage_t* out,
                             const int n) {
  int gid = get_global_id(0);
  if (gid >= n) return;
  STORE(out, gid, LOAD(patch_emb, gid) + LOAD(pos_resized, gid));
}

// Final readback helper: copy last row of logits storage buffer into float buffer is handled on host.
