#pragma once
// Reference: model_info/modeling_openelm.py (OpenELMFeedForward / OpenELMDecoderLayer)

#include "opencl_context.h"
#include "weights.h"

#include <CL/cl.h>
#include <string>

class Mlp {
 public:
  Mlp(OpenCLContext& cl_ctx, Weights& weights, int layer_idx);
  ~Mlp();

  bool initialize();

  // input: [seq_len, hidden]
  // output: [seq_len, hidden] — BORROWED handle, valid until next forward().
  // Caller MUST NOT release. Owned by *this.
  cl_mem forward(cl_command_queue queue, cl_mem input, int seq_len);

 private:
  bool ensure_activation_buffers_(int seq_len);
  void release_activation_buffers_();

  OpenCLContext& cl_ctx_;
  Weights& weights_;
  int layer_idx_ = 0;

  // Prefix strings for weight key construction. Computed once in the
  // constructor so per-call forward() doesn't reformat strings.
  std::string prefix_layer_;
  std::string prefix_ffn_;

  // Weights (OpenELM is fused-FFN: proj_1 -> [2I,H] gates+up, proj_2 -> [H,I] down)
  cl_mem ffn_norm_w_ = nullptr;  // transformer.layers.{i}.ffn_norm.weight
  cl_mem proj1_w_ = nullptr;     // transformer.layers.{i}.ffn.proj_1.weight  [2*I, H]
  cl_mem proj2_w_ = nullptr;     // transformer.layers.{i}.ffn.proj_2.weight  [H, I]

  // Kernels (kernels/mlp.cl::swiglu)
  cl_program program_ = nullptr;
  cl_kernel kernel_swiglu_ = nullptr;

  // Persistent activation buffers. Sticky capacity: grow on first call (with
  // the largest seq_len = prefill P), reuse at decode (seq_len=1) without realloc.
  cl_mem buf_proj1_out_ = nullptr;   // [seq_len, 2*I]
  cl_mem buf_gate_      = nullptr;   // [seq_len, I]
  cl_mem buf_proj2_out_ = nullptr;   // [seq_len, H] — returned BORROWED
  size_t buf_proj1_out_cap_ = 0;
  size_t buf_gate_cap_      = 0;
  size_t buf_proj2_out_cap_ = 0;
};
