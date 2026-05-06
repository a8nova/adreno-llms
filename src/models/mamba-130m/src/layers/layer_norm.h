// Reference: model_info/transformers_src/modeling_mamba.py (HuggingFace Transformers v4.39.0.dev0)
// Implements Mamba's RMSNorm-style normalization when config.rms_norm is true.
#pragma once

#include "opencl_context.h"
#include "weights.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <string>

class LayerNorm {
public:
  LayerNorm(OpenCLContext& cl_ctx, Weights& weights, const std::string& weight_key,
            int hidden_size, float eps);
  ~LayerNorm();

  bool initialize();

  // input: [seq_len, hidden_size]
  // output: [seq_len, hidden_size]
  cl_mem forward(cl_command_queue queue, cl_mem input, int seq_len);

private:
  OpenCLContext& cl_ctx_;
  Weights& weights_;
  std::string weight_key_;
  int hidden_size_;
  float eps_;

  cl_program program_ = nullptr;
  cl_kernel kernel_ = nullptr;

  cl_mem gamma_ = nullptr;

  // Persistent output buffer (sized for ≥ M); reused across forward() calls
  // to skip the ~50 µs clCreateBuffer cost per call (25 calls/token here).
  cl_mem output_buf_ = nullptr;
  int    output_capacity_M_ = 0;

  // Caller takes a reference to the returned buffer via clRetainMemObject
  // inside forward(); their clReleaseMemObject decrements back to our owned ref.
};
