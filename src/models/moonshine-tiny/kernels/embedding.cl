// Token-embedding gather. Owned by src/ops/Embedding.cpp.
// Split from the former kernels/moonshine.cl monolith (whisper-parity layout);
// kernel bodies are byte-identical to the monolith at the time of the split.
// Dtype-template preamble — DO NOT EDIT. Driven by host-side -DNNOPT_USE_FP16.
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

// ── embedding gather: output[t, :] = table[ids[t], :] ───────────────────
__kernel void embed_gather(
    __global const int* ids,           // [T]
    __global const storage_t* table,   // [vocab, dim]
    __global storage_t* output,        // [T, dim]
    const int T,
    const int dim) {
    int gid = get_global_id(0);
    const int total = T * dim;
    if (gid >= total) return;
    int t = gid / dim;
    int d = gid - t * dim;
    int tok = ids[t];
    STORE(output, gid, LOAD(table, tok * dim + d));
}
