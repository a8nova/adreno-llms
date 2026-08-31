// SinkAttention core (DeferredLocalDotProductSelfAttention), single query frame.
// Reference: magenta_rt/mlx/attention.py LocalDotProductSelfAttention.__call__
//
// kv positions for the first/standalone frame = [sink, current_self] (kv_len=2).
// Per head h (H heads, D dims/head):
//   scale_vec[d] = inv_sqrt_d * softplus(per_dim_scale[d])
//   s_sink = sum_d  q[h,d] * sink_k[h,d]              (sink_k pre-divided by scale ⇒ unscaled q)
//   s_cur  = sum_d (q[h,d]*scale_vec[d]) * k[h,d]
//   softmax([s_sink, s_cur]) → (w0, w1)
//   out[h,d] = w0*sink_v[h,d] + w1*v[h,d]
//
// qkv layout: [q(H*D) | k(H*D) | v(H*D)] fp32 (output of qkv_proj linear).
// sink_k/sink_v/per_dim_scale fp16 (vload_half). out [H*D] fp32. One work-item per head.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void sink_attention_f32(
    __global const float* qkv,          // [3*H*D]
    __global const half*  sink_k,       // [H*D]
    __global const half*  sink_v,       // [H*D]
    __global const half*  per_dim_scale,// [D]
    __global float*       out,          // [H*D]
    const int H,
    const int D,
    const float inv_sqrt_d) {
  const int h = get_global_id(0);
  if (h >= H) return;
  const int HD = H * D;
  const size_t qb = (size_t)h * D;          // q[h,*]
  const size_t kb = (size_t)HD + qb;        // k[h,*]
  const size_t vb = (size_t)2 * HD + qb;    // v[h,*]

  // scale_vec = r_softplus_0 * query_scale * softplus(per_dim_scale);
  // r_softplus_0 = 1/ln2 normalizes so per_dim_scale=0 → scale=query_scale (attention.py:38).
  const float r_softplus_0 = 1.442695041f;
  float s_sink = 0.0f, s_cur = 0.0f;
  for (int d = 0; d < D; d++) {
    const float qd = qkv[qb + d];
    const float sv = r_softplus_0 * inv_sqrt_d * log1p(exp(vload_half(d, per_dim_scale)));
    s_sink += qd * vload_half(qb + d, sink_k);
    s_cur  += (qd * sv) * qkv[kb + d];
  }
  const float mx = s_sink > s_cur ? s_sink : s_cur;
  const float e0 = exp(s_sink - mx), e1 = exp(s_cur - mx);
  const float z = e0 + e1;
  const float w0 = e0 / z, w1 = e1 / z;
  for (int d = 0; d < D; d++)
    out[qb + d] = w0 * vload_half(qb + d, sink_v) + w1 * qkv[vb + d];
}
