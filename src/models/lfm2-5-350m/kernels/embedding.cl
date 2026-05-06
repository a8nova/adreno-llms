// Reference: model_info/transformers_src/modeling_lfm2.py:320-377 Lfm2Model.forward (embed_tokens) and typical embedding lookup
// Note: This port uses a simple embedding gather. LFM2 uses RoPE for attention; there is NO learned position embedding.

#include "utils.cl"

// input_ids: int32[seq]
// embed: storage_t[vocab, hidden]
// out: storage_t[seq, hidden]
__kernel void embedding_forward(
    __global const int* input_ids,
    __global const storage_t* embed,
    __global storage_t* out,
    const int seq_len,
    const int hidden_size) {
  const int t = get_global_id(0);
  if (t >= seq_len) return;
  const int token_id = input_ids[t];
  if (token_id < 0) return;

  const int src_base = token_id * hidden_size;
  const int dst_base = t * hidden_size;
  for (int d = 0; d < hidden_size; ++d) {
    const float v = LOAD(embed, src_base + d);
    STORE(out, dst_base + d, v);
  }
}
