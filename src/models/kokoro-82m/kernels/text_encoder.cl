// Reference: /Users/alazarshenkute/.nnopt/repos/kokoro/kokoro/modules.py: TextEncoder.forward (embedding + cnn stack)
//
// Kernels for Kokoro TextEncoder. Layout is NCL (batch=1 implicit):
//   embedding_out: [C, T]
//
// NOTE: This file currently provides only embedding gather. Conv stack + LayerNorm
// are ported later.

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

__kernel void embedding_gather_ncl(
    __global const int* token_ids,          // [T]
    __global const storage_t* emb_table,    // [V, C] row-major
    __global storage_t* out,                // [C, T]
    const int T,
    const int C) {

    int gid = get_global_id(0);
    int total = T * C;
    if (gid >= total) return;

    int c = gid / T;
    int t = gid - c * T;
    int tok = token_ids[t];
    if (tok < 0) tok = 0;
    // No explicit bounds check against vocab here; upstream should guarantee.

    float v = LOAD(emb_table, tok * C + c);
    STORE(out, c * T + t, v);
}
