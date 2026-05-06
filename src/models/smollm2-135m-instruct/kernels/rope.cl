// Reference: model_info/transformers_src/modeling_llama.py:apply_rotary_pos_emb,rotate_half,LlamaRotaryEmbedding.forward
// Rotary Positional Embedding (RoPE) — applies q, k rotation in-place.
// Auto-generated transformer scaffold (Rule ROPE-01). Edit only if your
// model uses a non-standard RoPE variant (linear scaling, NTK-aware,
// dynamic, etc.) — read modeling_*.py for the formula and adapt.
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


// q: [seq_q, q_heads * head_dim]   (NEW query rows for prefill or decode)
// k: [seq_q, kv_heads * head_dim]  (NEW key rows; will be appended to KV cache)
// cos, sin: [max_seq_len, head_dim]  (precomputed at init, NOT per-call)
// start_pos: absolute position of the first row in q/k. 0 for prefill;
//            current cache fill for decode. cos/sin are indexed at
//            (start_pos + t) so rotation matches the absolute position.
//
// Global work: total_pairs = seq_q * (q_heads + kv_heads) * (head_dim/2).
// One work-item updates one (token, head, pair) and writes both halves.
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

  const int q_cols = q_heads * head_dim;
  const int k_cols = kv_heads * head_dim;

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
  const int row_stride = is_q ? q_cols : k_cols;
  const int base = t * row_stride + h * head_dim;

  float x1 = LOAD(buf, base + hd);
  float x2 = LOAD(buf, base + hd + half_dim);

  const int abs_t = start_pos + t;
  float c = LOAD((__global storage_t*)cos, abs_t * head_dim + hd);
  float s = LOAD((__global storage_t*)sin, abs_t * head_dim + hd);

  float out1 = x1 * c - x2 * s;
  float out2 = x2 * c + x1 * s;

  STORE(buf, base + hd, out1);
  STORE(buf, base + hd + half_dim, out2);
}
