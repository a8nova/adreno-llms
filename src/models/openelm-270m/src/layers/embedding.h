#pragma once
// Reference: model_info/modeling_openelm.py (OpenELMModel.forward) and kernels/embedding.cl::embedding_forward

#include "opencl_context.h"
#include "weights.h"

#include <CL/cl.h>
#include <string>

class Embedding {
public:
  Embedding(OpenCLContext& cl_ctx, Weights& weights);
  ~Embedding();

  bool initialize();

  // token_ids: host pointer. seq_len rows output.
  // start_pos: absolute position offset (0 for prefill, prompt_len+step for decode).
  cl_mem forward(cl_command_queue queue, const int* token_ids, int seq_len, int start_pos);

  // Step 9: chained-decode variant. Embedding reads its single token id from
  // an existing GPU-side int32 buffer (`token_ids_dev`) at byte offset
  // `dev_offset_bytes`. Used to plumb the previous step's argmax_result_
  // directly into this step's embedding without a host roundtrip.
  // Always seq_len==1 at decode (caller guarantees).
  cl_mem forward_from_device_token(cl_command_queue queue,
                                   cl_mem token_ids_dev,
                                   size_t dev_offset_bytes,
                                   int start_pos);

private:
  OpenCLContext& cl_ctx_;
  Weights& weights_;

  cl_program program_ = nullptr;
  cl_kernel kernel_ = nullptr;

  cl_mem token_emb_ = nullptr; // transformer.token_embeddings.weight
  cl_mem pos_emb_ = nullptr;   // optional; OpenELM is RoPE-only, so nullptr

  // Persistent token-IDs upload buffer (Step 4a). Avoids the per-decode-token
  // clCreateBuffer + clReleaseMemObject round-trip; we write into a sticky
  // buffer instead. Capacity grows but never shrinks.
  cl_mem token_ids_buf_ = nullptr;
  size_t token_ids_buf_cap_bytes_ = 0;

  // Persistent zero-padded wpe buffer (Step 4 nearby fix). OpenELM has no
  // learned positional embeddings, so the embedding kernel gets zeros here.
  // Sized once to MAX_CONTEXT_LENGTH × HIDDEN_SIZE; reused every call.
  cl_mem zero_wpe_ = nullptr;
  size_t zero_wpe_cap_bytes_ = 0;

  // Step Z: persistent output buffer. forward() returns this BORROWED, so the
  // caller does not allocate or release. Saves 1 alloc + 1 release per token.
  cl_mem out_buf_ = nullptr;
  size_t out_buf_cap_bytes_ = 0;
};
