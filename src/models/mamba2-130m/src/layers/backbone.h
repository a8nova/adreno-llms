// Reference: model_info/transformers_src/modeling_mamba2.py:Mamba2Model.forward
#pragma once

#include "opencl_context.h"
#include "weights.h"

#include <CL/cl.h>

class Backbone {
public:
  Backbone(OpenCLContext& cl_ctx, Weights& weights);
  ~Backbone();

  bool initialize();

  // Embedding lookup (token embedding only for this architecture)
  // input_ids: [seq_len]
  // output: [seq_len, d_model]
  cl_mem embed(cl_command_queue queue, const int32_t* input_ids, int seq_len);

  // Lever D: GPU-side ids variant for async decode pipeline. Reads token ids
  // from a GPU buffer (typically the previous iteration's argmax output).
  // The kernel reads token_ids[0..seq_len-1] — caller controls indexing by
  // passing the appropriate cl_mem (a sub-view if needed).
  cl_mem embed_gpu(cl_command_queue queue, cl_mem ids_buf, int seq_len);

  cl_mem norm_f(cl_command_queue queue, cl_mem input, int seq_len);

private:
  OpenCLContext& cl_ctx_;
  Weights& weights_;

  cl_program embed_program_ = nullptr;
  cl_kernel embed_kernel_ = nullptr;

  cl_program rms_program_ = nullptr;
  cl_kernel rms_kernel_ = nullptr;

  cl_mem embed_w_ = nullptr;   // backbone.embedding.weight
  cl_mem norm_w_ = nullptr;    // backbone.norm_f.weight

  // Lever 4: persistent activation buffers — sized once for max seq_len so
  // decode (seq_len=1) reuses without realloc. Same Step-5 pattern that
  // landed inside Ssm. OWNERSHIP: returned cl_mem from embed()/norm_f() is a
  // BORROWED handle; the caller MUST NOT release it. The buffer is
  // overwritten on the next call.
  int   buf_capacity_rows_ = 0;
  cl_mem buf_ids_ = nullptr;       // [rows] int32 — uploaded each forward()
  cl_mem buf_embed_out_ = nullptr; // [rows, d_model]
  cl_mem buf_norm_f_out_ = nullptr; // [rows, d_model]

  bool ensure_buffers_(int rows);
  void release_buffers_();
};
