// Reference: model_info/transformers_src/modeling_musicgen.py (nn.Embedding semantics)
//
// Embedding lookup: ids [num_tokens] -> out [num_tokens, hidden_size].
// IMPORTANT: ids is int32, stored row-major.
//
// NOTE: This kernel must support both:
//   1) text encoder token embeddings where ids is [batch, seq_len]
//   2) audio codebook embeddings where ids may be [batch, 1] or [batch*num_codebooks, 1]
//
// The correct semantics are always:
//   id = ids[tok]
// not ids[0]. Any "always ids[0]" shortcut breaks prefill embeddings.

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

__kernel void embedding_lookup(__global const int* ids,
                              __global const storage_t* wte,
                              __global storage_t* out,
                              const int num_tokens,
                              const int hidden_size,
                              const int vocab_size,
                              const int start_pos,
                              const int layer_idx)
{
  int gid = get_global_id(0);
  int total = num_tokens * hidden_size;
  if (gid >= total) return;

  int tok = gid / hidden_size;
  int col = gid - tok * hidden_size;

  // ids is int32 on device.
  int id = ids[tok];
  if (id < 0) id = 0;
  if (id >= vocab_size) id = vocab_size - 1;

  int w_off = id * hidden_size + col;
  float v = (float)LOAD(wte, w_off);
  STORE(out, gid, v);

  // start_pos/layer_idx are threaded for ABS-POS-EMBED models; embedding itself
  // doesn't use them (no positional add in this kernel).
  (void)start_pos;
  (void)layer_idx;
}
