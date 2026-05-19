// Reference: model_info/transformers_src/modeling_vits.py (VitsTextEncoder embeddings)
// Minimal kernels for deterministic scaffold: embedding gather + stats concat.

// Port note: OpenCL program compilation does not set an include search path on Android.
// Inline the utils.cl dtype preamble here instead of relying on #include.
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

__kernel void gather_token_embed(__global const storage_t* emb_table,  // [V, H]
                                 __global const int* input_ids,        // [T]
                                 __global storage_t* out_hidden,       // [T, H]
                                 const int T,
                                 const int H) {
  int t = (int)get_global_id(0);
  if (t >= T) return;
  int id = input_ids[t];
  int src_off = id * H;
  int dst_off = t * H;
  for (int h = 0; h < H; ++h) {
    STORE(out_hidden, dst_off + h, LOAD(emb_table, src_off + h));
  }
}

__kernel void concat_stats(__global const storage_t* hidden,  // [T, H]
                           __global storage_t* stats,        // [T, 2H]
                           const int T,
                           const int H) {
  int t = (int)get_global_id(0);
  if (t >= T) return;
  int h0 = t * H;
  int s0 = t * (2 * H);
  for (int h = 0; h < H; ++h) {
    const float v = LOAD(hidden, h0 + h);
    STORE(stats, s0 + h, v);
    STORE(stats, s0 + H + h, 0.0f);
  }
}
