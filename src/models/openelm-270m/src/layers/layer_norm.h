#pragma once
// Reference: model_info/modeling_openelm.py:15-53 (OpenELMRMSNorm)
// OpenELM uses RMSNorm in every position labelled "layer_norm" in C++. This
// header exposes both a class wrapper (used by Model) and a free function
// (used by Attention's optional q_norm/k_norm and Mlp's ffn_norm).

#include "opencl_context.h"
#include "weights.h"

#include <CL/cl.h>
#include <string>

class LayerNorm {
 public:
  LayerNorm(OpenCLContext& cl_ctx, Weights& weights, int layer_idx, const std::string& weight_key);
  ~LayerNorm();

  bool initialize();

  // input: [seq_len, hidden] storage_t
  // returns: NEW [seq_len, hidden] cl_mem (caller releases). nullptr on failure.
  cl_mem forward(cl_command_queue queue, cl_mem input, int seq_len);

 private:
  OpenCLContext& cl_ctx_;
  Weights& weights_;
  int layer_idx_;
  std::string weight_key_;
  cl_mem weight_ = nullptr;
};

// Free helper used by Attention (optional q_norm/k_norm) and Mlp (ffn_norm).
// Builds a NEW [rows, cols] cl_mem and returns it; caller releases.
// Returns nullptr on failure (logged via NNOPT_ERROR_FMT).
cl_mem rmsnorm_forward(OpenCLContext& cl_ctx,
                       cl_command_queue queue,
                       cl_mem input,
                       cl_mem weight,
                       int rows,
                       int cols,
                       float eps);
