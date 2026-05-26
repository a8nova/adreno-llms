// Reference: model_info/transformers_src/modeling_llama.py:84-168 LlamaRotaryEmbedding.forward

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

// Rotary embedding kernels for LlamaRotaryEmbedding.
//
// Reference (Transformers): transformers/models/llama/modeling_llama.py
// forward():
//   freqs = (inv_freq[None,:,None] @ position_ids[:,None,:]).transpose(1,2)
//   emb = cat((freqs,freqs), dim=-1)
//   cos = cos(emb) * attention_scaling
//   sin = sin(emb) * attention_scaling
//   return cos.to(x.dtype), sin.to(x.dtype)
//
// This file implements the non-matmul portion (cat + cos/sin + scaling).
// The matmul producing `freqs` is dispatched via CLBlast by the host.

__kernel void rotary_embedding_cos_sin_scale(
    __global const storage_t* emb,          // [batch, seq_len, rotary_dim] where rotary_dim = 2 * (dim/2)
    __global storage_t* cos_out,            // same shape as emb
    __global storage_t* sin_out,            // same shape as emb
    const int batch,
    const int seq_len,
    const int rotary_dim,
    const float attention_scaling)
{
    const int gid = (int)get_global_id(0);
    const int total = batch * seq_len * rotary_dim;
    if (gid >= total) return;

    const float e = (float)LOAD(emb, gid);
    const float c = cos(e) * attention_scaling;
    const float s = sin(e) * attention_scaling;

    STORE(cos_out, gid, (storage_t)c);
    STORE(sin_out, gid, (storage_t)s);
}

// Optional fused kernel: compute emb/cos/sin directly from inv_freq and position_ids.
// This matches the reference math:
//   freqs[b,s,i] = inv_freq[i] * position_ids[b,s]
//   emb = cat((freqs,freqs), dim=-1)
//   cos/sin on emb then scale.
//
// Use when host prefers to avoid an explicit GEMM for this small matmul.
__kernel void rotary_embedding_from_inv_freq_and_positions(
    __global const storage_t* inv_freq,     // [half_dim]
    __global const int* position_ids,       // [batch, seq_len]
    __global storage_t* cos_out,            // [batch, seq_len, 2*half_dim]
    __global storage_t* sin_out,            // [batch, seq_len, 2*half_dim]
    const int batch,
    const int seq_len,
    const int half_dim,
    const float attention_scaling)
{
    const int gid = (int)get_global_id(0);
    const int total = batch * seq_len * (2 * half_dim);
    if (gid >= total) return;

    const int d2 = 2 * half_dim;
    const int bs = gid / d2;
    const int j = gid - bs * d2;
    const int b = bs / seq_len;
    const int s = bs - b * seq_len;

    const int i = (j < half_dim) ? j : (j - half_dim);

    const float inv = (float)LOAD(inv_freq, i);
    const float pos = (float)position_ids[b * seq_len + s];
    const float e = inv * pos;

    const float c = cos(e) * attention_scaling;
    const float sn = sin(e) * attention_scaling;

    STORE(cos_out, gid, (storage_t)c);
    STORE(sin_out, gid, (storage_t)sn);
}
