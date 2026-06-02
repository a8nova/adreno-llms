// Embedding + learned positional embeddings kernel.
// Reference: model_info/transformers_src/modeling_whisper.py WhisperDecoder.forward (inputs_embeds + positions)

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)    vload_half((i),(p))
  #define STORE(p,i,v) vstore_half((v),(i),(p))
#else
  typedef float storage_t;
  #define LOAD(p,i)    ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
#endif

__kernel void embedding_with_pos(
    __global const int* ids,
    __global const storage_t* wte,     // [vocab, hidden]
    __global const storage_t* wpe,     // [max_pos, hidden]
    __global storage_t* out,           // [seq, hidden]
    const int seq_len,
    const int hidden,
    const int start_pos)
{
    int gid = (int)get_global_id(0);
    int total = seq_len * hidden;
    if (gid >= total) return;

    int t = gid / hidden;
    int h = gid - t * hidden;
    int token = ids[t];
    int abs_pos = start_pos + t;

    float a = (float)LOAD(wte, token * hidden + h);
    float b = (float)LOAD(wpe, abs_pos * hidden + h);
    STORE(out, gid, a + b);
}
