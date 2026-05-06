// Reference: model_info/transformers_src/modeling_lfm2.py:425-469 Lfm2DecoderLayer (operator_norm)
#pragma once

#include "opencl_context.h"
#include "weights.h"

#include <CL/cl.h>
#include <string>

class OperatorNorm {
public:
  OperatorNorm(OpenCLContext& cl_ctx, Weights& weights, const std::string& prefix, int layer_idx);
  ~OperatorNorm();

  bool initialize();

  cl_mem forward(cl_command_queue queue, cl_mem input, int seq_len);

private:
  OpenCLContext& cl_ctx_;
  Weights& weights_;
  std::string prefix_;
  int layer_idx_ = -1;

  cl_program program_ = nullptr;
  cl_kernel kernel_ = nullptr;

  // Persistent output — borrowed handle on return.
  cl_mem buf_out_ = nullptr;
  int    buf_seq_capacity_ = 0;
};
