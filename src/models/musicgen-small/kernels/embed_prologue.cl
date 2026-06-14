// EMBED PROLOGUE KERNEL (Fix-12 Stage 4+5) — fuses the per-step decoder input
// prologue into ONE dispatch: read the 4 codebook ids (from the GPU-resident
// delay grid column `step`, or a host-provided id buffer), sum their token
// embedding rows, add the sinusoidal positional row at abs pos `step`, and write
// the [hidden] result. Replaces 4 Embedding kernels + 4 tiny id uploads + 3
// element_add + 1 pos element_add (~8 dispatches + 4 uploads/step → 1 dispatch,
// 0 uploads). One workgroup, EMBED_WG threads cooperating over hidden.
//
// emb_tables: num_codebooks embedding tables, each [vocab, hidden], passed as one
//   concatenated buffer [num_codebooks*vocab, hidden] (cb-major). id for cb is
//   read from grid[cb*steps1 + step] (Stage 4) or host_ids[cb] (fallback).
// pos_w: sinusoidal table [max_pos, hidden]; the row at `step` is added.
// out: [hidden] summed embedding (fp16 store).
// AGENT_DIRECTIVE_FP32_ACCUM: the running sum is fp32 in local, stored fp16.

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

#define EMBED_WG 256

__kernel __attribute__((reqd_work_group_size(EMBED_WG, 1, 1)))
void embed_prologue(
    __global const storage_t* emb_tables,  // [num_codebooks*vocab, hidden]
    __global const storage_t* pos_w,       // [max_pos, hidden]
    __global storage_t* out,               // [out_rows, hidden]
    __global const int* grid,              // [num_codebooks, steps1] (Stage 4) or null
    __global const int* host_ids,          // [num_codebooks] (fallback) or null
    const int use_grid,
    const int steps1,
    __global const int* sp,  // step-params: step = sp[0]. Was a literal int —
                             // buffered so recordable-queue replays need no
                             // per-step kernel-arg override (driver rejects).
    const int num_codebooks,
    const int embed_rows,   // rows per codebook table (vocab+1: includes BOS/pad)
    const int hidden,
    const int out_rows) {   // duplicate the result into rows 0..out_rows-1 of out
                            // (CFG decode wants row0=row1=emb; writing both here
                            // removed the 2 emb→x CopyBuffers, which the
                            // recordable-queue replay path cannot record)

  const int lid = get_local_id(0);
  const int step = sp[0];

  // Resolve the 4 ids once (small; do redundantly per thread — cheaper than a barrier).
  // For each codebook cb, the embedding row base = (cb*embed_rows + id)*hidden.
  // ids may equal BOS (= embed_rows-1), so the table MUST have embed_rows rows.
  for (int c = lid; c < hidden; c += EMBED_WG) {
    float acc = 0.0f;
    for (int cb = 0; cb < num_codebooks; ++cb) {
      int id = use_grid ? grid[cb * steps1 + step] : host_ids[cb];
      acc += LOAD(emb_tables, ((long)cb * embed_rows + id) * hidden + c);
    }
    acc += LOAD(pos_w, (long)step * hidden + c);
    for (int r = 0; r < out_rows; ++r)
      STORE(out, (long)r * hidden + c, acc);
  }
}
