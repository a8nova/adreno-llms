// Reference: model_info/transformers_src/modeling_lfm2.py:213-320 Lfm2ShortConv
#pragma once

#include "opencl_context.h"
#include "weights.h"
#include <CL/cl.h>
#include <string>

class Convolution {
public:
  Convolution(OpenCLContext& cl_ctx, Weights& weights, const std::string& prefix, int layer_idx);
  ~Convolution();

  bool initialize();

  // hidden_states: [seq, hidden]
  // start_pos: absolute position of first token in this call (0 for prefill, >0 for decode)
  // Returns: [seq, hidden]
  cl_mem forward(cl_command_queue queue, cl_mem hidden_states, int seq_len, int start_pos);

private:
  OpenCLContext& cl_ctx_;
  Weights& weights_;
  std::string prefix_;
  int layer_idx_ = -1;

  cl_program program_ = nullptr;
  cl_kernel copy_transpose_kernel_ = nullptr;     // [seq, hidden] -> [hidden, seq]
  cl_kernel split_chunk3_kernel_ = nullptr;       // [hidden*3, seq] -> B,C,X [hidden, seq]
  cl_kernel elem_mul_kernel_ = nullptr;           // Bx = B*X
  cl_kernel conv1d_cache_kernel_ = nullptr;       // causal conv with cache
  cl_kernel update_cache_kernel_ = nullptr;       // update ring/shift cache
  cl_kernel elem_mul2_kernel_ = nullptr;          // Y = C * conv_out
  cl_kernel copy_transpose_back_kernel_ = nullptr; // [hidden, seq] -> [seq, hidden]
  cl_kernel block_decode_kernel_ = nullptr;        // fused per-channel kernel for seq_len==1

  // persistent conv state cache: [hidden, L_cache-1]
  cl_mem conv_state_ = nullptr;

  // cached weights
  cl_mem in_proj_w_ = nullptr;   // [3*hidden, hidden]
  cl_mem out_proj_w_ = nullptr;  // [hidden, hidden]
  cl_mem conv_w_ = nullptr;      // [hidden, 1, L_cache] (depthwise)

  // dims
  int hidden_size_ = 0;
  int l_cache_ = 0;

  // Persistent activation buffers — lazy-allocated on first forward() call,
  // grown if a longer seq_len is requested, released by destructor. Replaces
  // 10 clCreateBuffer + 10 clReleaseMemObject per call.
  // The forward() output `buf_out_` is BORROWED by the caller (must NOT release).
  cl_mem buf_in_proj_     = nullptr;  // [seq, 3H]
  cl_mem buf_in_proj_T_   = nullptr;  // [3H, seq]
  cl_mem buf_B_           = nullptr;  // [H, seq]
  cl_mem buf_C_           = nullptr;  // [H, seq]
  cl_mem buf_X_           = nullptr;  // [H, seq]
  cl_mem buf_Bx_          = nullptr;  // [H, seq]
  cl_mem buf_conv_out_T_  = nullptr;  // [H, seq]
  cl_mem buf_Bx_seq_      = nullptr;  // [seq, H]
  cl_mem buf_y_T_         = nullptr;  // [H, seq]
  cl_mem buf_y_seq_       = nullptr;  // [seq, H]
  cl_mem buf_out_         = nullptr;  // [seq, H]  ← borrowed return
  int    buf_seq_capacity_ = 0;
  bool   ensure_buffers_(int seq_len);
};
