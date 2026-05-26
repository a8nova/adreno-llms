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

// Token embedding gather.
// Global work size: (num_tokens, hidden_size)
// out[t, h] = wte[token_ids[t], h]
__kernel void embedding_forward(
    __global const int* token_ids,          // [num_tokens]
    __global const storage_t* wte,          // [vocab_size, hidden_size]
    __global storage_t* out,                // [num_tokens, hidden_size]
    const int num_tokens,
    const int hidden_size,
    const int vocab_size) {
  const int t = (int)get_global_id(0);
  const int h = (int)get_global_id(1);
  if (t >= num_tokens || h >= hidden_size) return;

  const int tok = token_ids[t];

  // Defensive guard: if tok is invalid (e.g., tokenizer drift / OOB read), write 0.
  // This prevents wte OOB reads from crashing or cascading garbage through the graph.
  if (tok < 0 || tok >= vocab_size) {
    const long out_idx = (long)t * (long)hidden_size + (long)h;
    STORE(out, out_idx, (storage_t)0.0f);
    return;
  }

  const long wte_idx = (long)tok * (long)hidden_size + (long)h;
  const long out_idx = (long)t * (long)hidden_size + (long)h;

  const float v = (float)LOAD(wte, wte_idx);
  STORE(out, out_idx, (storage_t)v);
}
