// Reference: model_info/transformers_src/modeling_lfm2.py:123-145 Lfm2MLP
#pragma once

#include "opencl_context.h"
#include "weights.h"
#include <CL/cl.h>
#include <string>

class Mlp {
public:
  // prefix example: "model.layers.0.feed_forward" (weights: w1/w2/w3)
  Mlp(OpenCLContext& cl_ctx, Weights& weights, const std::string& prefix, int layer_idx);
  ~Mlp();

  bool initialize();

  // input: [seq, hidden]
  // output: [seq, hidden]
  cl_mem forward(cl_command_queue queue, cl_mem input, int seq_len);

private:
  OpenCLContext& cl_ctx_;
  Weights& weights_;
  std::string prefix_;
  int layer_idx_ = -1;

  cl_program program_ = nullptr;
  cl_kernel silu_mul_kernel_ = nullptr;

  // Weights
  cl_mem w1_ = nullptr;
  cl_mem w2_ = nullptr;
  cl_mem w3_ = nullptr;

  // Persistent activation buffers — borrowed handle on return.
  cl_mem buf_gate_ = nullptr;  // [seq, I]
  cl_mem buf_up_   = nullptr;  // [seq, I]
  cl_mem buf_out_  = nullptr;  // [seq, H]
  int    buf_seq_capacity_ = 0;
  int    intermediate_size_ = 0;
  bool   ensure_buffers_(int seq_len);
};
