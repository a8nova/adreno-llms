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
  // returns: BORROWED [seq_len, hidden] cl_mem owned by *this. Valid until
  // the next call. Caller MUST NOT release. Step Z: replaces the per-call
  // alloc that was here before — saves ~30 µs of driver round-trip per
  // norm dispatch (33 norms/token at decode = ~1 ms/token).
  cl_mem forward(cl_command_queue queue, cl_mem input, int seq_len);

 private:
  OpenCLContext& cl_ctx_;
  Weights& weights_;
  int layer_idx_;
  std::string weight_key_;
  cl_mem weight_ = nullptr;

  // Persistent output buffer. Grown on demand by forward(); released in dtor.
  cl_mem out_buf_ = nullptr;
  size_t out_buf_cap_bytes_ = 0;
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

// Variant that writes into a caller-provided output buffer. No allocation,
// no return-buffer ownership transfer. Used at attn.q_norm and attn.k_norm
// where the output is immediately copied into a persistent buf_qn_/buf_kn_;
// here we eliminate both the alloc and the copy by writing straight into
// the destination.
bool rmsnorm_forward_into(OpenCLContext& cl_ctx,
                          cl_command_queue queue,
                          cl_mem input,
                          cl_mem weight,
                          cl_mem output,
                          int rows,
                          int cols,
                          float eps);
