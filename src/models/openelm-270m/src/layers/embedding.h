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

private:
  OpenCLContext& cl_ctx_;
  Weights& weights_;

  cl_program program_ = nullptr;
  cl_kernel kernel_ = nullptr;

  cl_mem token_emb_ = nullptr; // transformer.token_embeddings.weight
  cl_mem pos_emb_ = nullptr;   // optional; OpenELM is RoPE-only, so nullptr
};
