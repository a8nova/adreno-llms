// Reference: model_info/transformers_src/modeling_mamba.py:~620-760 (MambaModel.forward, MambaForCausalLM.forward)
#pragma once

#include "opencl_context.h"
#include "weights.h"

#include <CL/cl.h>
#include <string>

class Backbone {
public:
  Backbone(OpenCLContext& cl_ctx, Weights& weights);
  ~Backbone();

  bool initialize();

  // Token embedding only (no position embedding for Mamba).
  // input_ids are int32 on host.
  cl_mem embed(cl_command_queue queue, const int* input_ids, int seq_len);

  // Final RMSNorm (implemented via layer_norm.cl::rms_norm).
  cl_mem final_norm(cl_command_queue queue, cl_mem hidden, int seq_len);

  // LM head: logits = hidden @ embeddings.T
  cl_mem lm_head(cl_command_queue queue, cl_mem hidden, int seq_len);

  int vocab_size() const { return vocab_size_; }
  int hidden_size() const { return hidden_size_; }

private:
  OpenCLContext& cl_ctx_;
  Weights& weights_;

  int vocab_size_ = 0;
  int hidden_size_ = 0;

  cl_mem embed_w_ = nullptr;      // backbone.embeddings.weight [vocab, hidden]
  cl_mem norm_f_w_ = nullptr;     // backbone.norm_f.weight [hidden]

  // kernels/embedding.cl::embedding_forward
  cl_program embed_program_ = nullptr;
  cl_kernel embed_kernel_ = nullptr;

  // kernels/layer_norm.cl::rms_norm
  cl_program rms_program_ = nullptr;
  cl_kernel rms_kernel_ = nullptr;
};
