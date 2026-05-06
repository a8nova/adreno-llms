// Reference: model_info/transformers_src/modeling_lfm2.py:320-377 Lfm2Model.__init__/forward (embed_tokens)
#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include "opencl_context.h"
#include "weights.h"

class Embedding {
public:
  Embedding(OpenCLContext& cl_ctx, Weights& weights);
  ~Embedding();

  bool initialize();

  // Returns [seq_len, hidden]
  cl_mem forward(cl_command_queue queue, const int* input_ids, int seq_len);

private:
  OpenCLContext& cl_ctx_;
  Weights& weights_;

  cl_program program_ = nullptr;
  cl_kernel kernel_ = nullptr;

  cl_mem embed_w_ = nullptr; // model.embed_tokens.weight

  // Persistent buffers — borrowed handle on return.
  cl_mem buf_out_ = nullptr;            // [seq, H]
  cl_mem buf_input_ids_ = nullptr;       // [seq] cl_int — re-uploaded per call
  int    buf_seq_capacity_ = 0;
};
