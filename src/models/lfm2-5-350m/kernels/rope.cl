// Reference: model_info/transformers_src/modeling_lfm2.py:156-224 apply_rotary_pos_emb/rotate_half
// Rotary Positional Embedding (RoPE) — applies q, k rotation in-place.
// Ported from: model_info/transformers_src/modeling_lfm2.py:156-224 apply_rotary_pos_emb/rotate_half
// NOTE: This kernel is called on flattened [seq, heads*head_dim] buffers (C++ stores
// q as [seq_q, q_heads*head_dim] and k as [seq_q, kv_heads*head_dim]).
// We interpret those buffers as logical [seq, heads, head_dim] with head_dim contiguous
// within each head, which matches the packing of [seq, heads*head_dim].
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


// q: [seq_q, q_heads, head_dim] flattened as [seq_q * q_heads * head_dim]
// k: [seq_q, kv_heads, head_dim] flattened as [seq_q * kv_heads * head_dim]
// cos, sin: [max_seq_len, head_dim] (created from inv_freq and position_ids)
// start_pos: absolute position offset
__kernel void rope_apply_qk(
    __global storage_t* q,
    __global storage_t* k,
    __global const storage_t* cos,
    __global const storage_t* sin,
    const int seq_q,
    const int q_heads,
    const int kv_heads,
    const int head_dim,
    const int start_pos) {
  int gid = get_global_id(0);
  const int half_dim = head_dim / 2;

  const int q_pairs = seq_q * q_heads * half_dim;
  const int k_pairs = seq_q * kv_heads * half_dim;
  const int total_pairs = q_pairs + k_pairs;
  if (gid >= total_pairs) return;

  const int is_q = (gid < q_pairs) ? 1 : 0;
  int local_idx = is_q ? gid : (gid - q_pairs);

  const int heads = is_q ? q_heads : kv_heads;
  const int t = local_idx / (heads * half_dim);
  const int rem = local_idx - t * heads * half_dim;
  const int h = rem / half_dim;
  const int hd = rem - h * half_dim;

  __global storage_t* buf = is_q ? q : k;
  // Base index for [t, h, 0] in a flattened [seq, heads, head_dim] layout.
  const int base = (t * heads + h) * head_dim;

    // rotate_half: [-x2, x1]
  // LOAD returns float for fp16 (via vload_half) and storage_t for fp32.
  // Cast to float explicitly so arithmetic is always fp32.
  float x1 = (float)LOAD(buf, base + hd);
  float x2 = (float)LOAD(buf, base + hd + half_dim);

  const int abs_t = start_pos + t;

  // cos/sin were built as [seq_k, head_dim] for the CURRENT call, so abs_t is in [0, seq_k).
  // Do NOT index with a fixed MAX_POSITION_EMBEDDINGS stride here.
  // (Previous versions did, which walked off the end of the allocated cos/sin buffers
  // and produced NaNs immediately after RoPE.)
  float c = (float)LOAD(cos, abs_t * head_dim + hd);
  float s = (float)LOAD(sin, abs_t * head_dim + hd);

  // (x * cos) + (rotate_half(x) * sin)
  // out1 = x1*c + (-x2)*s
  // out2 = x2*c + ( x1)*s
  float out1 = x1 * c - x2 * s;
  float out2 = x2 * c + x1 * s;

  // IMPORTANT: STORE expects a float value, but storage_t is half under fp16.
  // Explicit cast prevents any compiler-specific ambiguity on some OpenCL drivers.
  STORE(buf, base + hd, (storage_t)out1);
  STORE(buf, base + hd + half_dim, (storage_t)out2);
}
