// Reference: model_info/transformers_src/modeling_lfm2.py:233-319 Lfm2ShortConv.forward/slow_forward
#include "layers/convolution.h"

#include "debug_utils.h"
#include "kernel_profiler.h"
#include "model_config.h"
#include "opencl_context.h"
#include "utils.h"
#include "weights.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

Convolution::Convolution(OpenCLContext& cl_ctx, Weights& weights, const std::string& prefix, int layer_idx)
    : cl_ctx_(cl_ctx), weights_(weights), prefix_(prefix), layer_idx_(layer_idx) {}

Convolution::~Convolution() {
  if (copy_transpose_kernel_) clReleaseKernel(copy_transpose_kernel_);
  if (split_chunk3_kernel_) clReleaseKernel(split_chunk3_kernel_);
  if (elem_mul_kernel_) clReleaseKernel(elem_mul_kernel_);
  if (conv1d_cache_kernel_) clReleaseKernel(conv1d_cache_kernel_);
  if (update_cache_kernel_) clReleaseKernel(update_cache_kernel_);
  if (elem_mul2_kernel_) clReleaseKernel(elem_mul2_kernel_);
  if (copy_transpose_back_kernel_) clReleaseKernel(copy_transpose_back_kernel_);
  if (block_decode_kernel_) clReleaseKernel(block_decode_kernel_);

  if (conv_state_) clReleaseMemObject(conv_state_);

  if (buf_in_proj_)    clReleaseMemObject(buf_in_proj_);
  if (buf_in_proj_T_)  clReleaseMemObject(buf_in_proj_T_);
  if (buf_B_)          clReleaseMemObject(buf_B_);
  if (buf_C_)          clReleaseMemObject(buf_C_);
  if (buf_X_)          clReleaseMemObject(buf_X_);
  if (buf_Bx_)         clReleaseMemObject(buf_Bx_);
  if (buf_conv_out_T_) clReleaseMemObject(buf_conv_out_T_);
  if (buf_Bx_seq_)     clReleaseMemObject(buf_Bx_seq_);
  if (buf_y_T_)        clReleaseMemObject(buf_y_T_);
  if (buf_y_seq_)      clReleaseMemObject(buf_y_seq_);
  if (buf_out_)        clReleaseMemObject(buf_out_);

  if (program_) clReleaseProgram(program_);
}

bool Convolution::ensure_buffers_(int seq_len) {
  if (seq_len <= buf_seq_capacity_ && buf_out_) return true;
  // (Re-)allocate at the requested capacity. Caller passes seq_len already
  // capped to the largest expected (prefill). After that, every decode call
  // (seq_len=1) reuses the same backing storage.
  if (buf_in_proj_)    { clReleaseMemObject(buf_in_proj_);    buf_in_proj_ = nullptr; }
  if (buf_in_proj_T_)  { clReleaseMemObject(buf_in_proj_T_);  buf_in_proj_T_ = nullptr; }
  if (buf_B_)          { clReleaseMemObject(buf_B_);          buf_B_ = nullptr; }
  if (buf_C_)          { clReleaseMemObject(buf_C_);          buf_C_ = nullptr; }
  if (buf_X_)          { clReleaseMemObject(buf_X_);          buf_X_ = nullptr; }
  if (buf_Bx_)         { clReleaseMemObject(buf_Bx_);         buf_Bx_ = nullptr; }
  if (buf_conv_out_T_) { clReleaseMemObject(buf_conv_out_T_); buf_conv_out_T_ = nullptr; }
  if (buf_Bx_seq_)     { clReleaseMemObject(buf_Bx_seq_);     buf_Bx_seq_ = nullptr; }
  if (buf_y_T_)        { clReleaseMemObject(buf_y_T_);        buf_y_T_ = nullptr; }
  if (buf_y_seq_)      { clReleaseMemObject(buf_y_seq_);      buf_y_seq_ = nullptr; }
  if (buf_out_)        { clReleaseMemObject(buf_out_);        buf_out_ = nullptr; }

  const int H = hidden_size_;
  cl_context ctx = cl_ctx_.context();
  cl_int err = CL_SUCCESS;
  auto alloc = [&](cl_mem* dst, size_t bytes, const char* tag) {
    *dst = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS) {
      NNOPT_ERROR_FMT("Convolution[%d]: ensure_buffers_(%s) alloc failed: %d", layer_idx_, tag, (int)err);
      return false;
    }
    return true;
  };
  const size_t s = (size_t)seq_len;
  const size_t Hs = (size_t)H * s * sizeof(nnopt_storage_t);
  if (!alloc(&buf_in_proj_,    s * (size_t)(3*H) * sizeof(nnopt_storage_t), "in_proj"))    return false;
  if (!alloc(&buf_in_proj_T_,  (size_t)(3*H) * s * sizeof(nnopt_storage_t), "in_proj_T"))  return false;
  if (!alloc(&buf_B_,           Hs, "B"))            return false;
  if (!alloc(&buf_C_,           Hs, "C"))            return false;
  if (!alloc(&buf_X_,           Hs, "X"))            return false;
  if (!alloc(&buf_Bx_,          Hs, "Bx"))           return false;
  if (!alloc(&buf_conv_out_T_,  Hs, "conv_out_T"))   return false;
  if (!alloc(&buf_Bx_seq_,      Hs, "Bx_seq"))       return false;
  if (!alloc(&buf_y_T_,         Hs, "y_T"))          return false;
  if (!alloc(&buf_y_seq_,       Hs, "y_seq"))        return false;
  if (!alloc(&buf_out_,         Hs, "out"))          return false;
  buf_seq_capacity_ = seq_len;
  return true;
}

bool Convolution::initialize() {
  // Required weights (Convolution.json)
  conv_w_ = weights_.get_buffer(prefix_ + ".conv.weight");
  in_proj_w_ = weights_.get_buffer(prefix_ + ".in_proj.weight");
  out_proj_w_ = weights_.get_buffer(prefix_ + ".out_proj.weight");
  if (!conv_w_ || !in_proj_w_ || !out_proj_w_) {
    NNOPT_ERROR_FMT("Convolution[%d]: missing required weight(s) under %s", layer_idx_, prefix_.c_str());
    return false;
  }

  // Cache dims
  {
    auto s = weights_.get_shape(prefix_ + ".conv.weight"); // [H,1,L]
    if (s.size() != 3) {
      NNOPT_ERROR_FMT("Convolution[%d]: unexpected conv.weight rank=%zu", layer_idx_, s.size());
      return false;
    }
    hidden_size_ = s[0];
    l_cache_ = s[2];
  }

  program_ = cl_ctx_.build_program_from_file("kernels/convolution.cl");
  if (!program_) return false;

  cl_int err = CL_SUCCESS;
  copy_transpose_kernel_ = clCreateKernel(program_, "conv_copy_transpose", &err);
  if (err != CL_SUCCESS || !copy_transpose_kernel_) {
    NNOPT_ERROR_FMT("Convolution[%d]: clCreateKernel(conv_copy_transpose) failed: %d", layer_idx_, (int)err);
    return false;
  }
  split_chunk3_kernel_ = clCreateKernel(program_, "conv_split_chunk3", &err);
  if (err != CL_SUCCESS || !split_chunk3_kernel_) {
    NNOPT_ERROR_FMT("Convolution[%d]: clCreateKernel(conv_split_chunk3) failed: %d", layer_idx_, (int)err);
    return false;
  }
  elem_mul_kernel_ = clCreateKernel(program_, "conv_pointwise_mul", &err);
  if (err != CL_SUCCESS || !elem_mul_kernel_) {
    NNOPT_ERROR_FMT("Convolution[%d]: clCreateKernel(conv_pointwise_mul) failed: %d", layer_idx_, (int)err);
    return false;
  }
  conv1d_cache_kernel_ = clCreateKernel(program_, "conv1d_causal_with_cache", &err);
  if (err != CL_SUCCESS || !conv1d_cache_kernel_) {
    NNOPT_ERROR_FMT("Convolution[%d]: clCreateKernel(conv1d_causal_with_cache) failed: %d", layer_idx_, (int)err);
    return false;
  }
  update_cache_kernel_ = clCreateKernel(program_, "conv_update_state", &err);
  if (err != CL_SUCCESS || !update_cache_kernel_) {
    NNOPT_ERROR_FMT("Convolution[%d]: clCreateKernel(conv_update_state) failed: %d", layer_idx_, (int)err);
    return false;
  }
  elem_mul2_kernel_ = clCreateKernel(program_, "conv_mul_C", &err);
  if (err != CL_SUCCESS || !elem_mul2_kernel_) {
    NNOPT_ERROR_FMT("Convolution[%d]: clCreateKernel(conv_mul_C) failed: %d", layer_idx_, (int)err);
    return false;
  }
  copy_transpose_back_kernel_ = clCreateKernel(program_, "conv_copy_transpose_back", &err);
  if (err != CL_SUCCESS || !copy_transpose_back_kernel_) {
    NNOPT_ERROR_FMT("Convolution[%d]: clCreateKernel(conv_copy_transpose_back) failed: %d", layer_idx_, (int)err);
    return false;
  }
  block_decode_kernel_ = clCreateKernel(program_, "conv_block_decode", &err);
  if (err != CL_SUCCESS || !block_decode_kernel_) {
    NNOPT_ERROR_FMT("Convolution[%d]: clCreateKernel(conv_block_decode) failed: %d", layer_idx_, (int)err);
    return false;
  }

  // conv_state_ stores last (L-1) Bx values for each channel.
  // Layout: [hidden, L-1]
  {
    const int L = l_cache_;
    const int state_len = std::max(0, L - 1);
    if (state_len > 0) {
      const size_t bytes = (size_t)hidden_size_ * (size_t)state_len * sizeof(nnopt_storage_t);
      conv_state_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
      if (err != CL_SUCCESS || !conv_state_) {
        NNOPT_ERROR_FMT("Convolution[%d]: clCreateBuffer(conv_state) failed: %d", layer_idx_, (int)err);
        return false;
      }
      // Initialize to zeros.
      std::vector<nnopt_storage_t> zeros((size_t)hidden_size_ * (size_t)state_len);
      err = clEnqueueWriteBuffer(cl_ctx_.queue(), conv_state_, CL_TRUE, 0, bytes, zeros.data(), 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Convolution[%d]: init conv_state write failed: %d", layer_idx_, (int)err);
        return false;
      }
    }
  }

  NNOPT_LAYER_INIT_FMT("block%d_sub_conv", layer_idx_);
  return true;
}

cl_mem Convolution::forward(cl_command_queue queue, cl_mem hidden_states, int seq_len, int start_pos) {
  if (!hidden_states) return nullptr;
  if (seq_len <= 0) return nullptr;

  const int H = hidden_size_ > 0 ? hidden_size_ : MODEL_CONFIG::HIDDEN_SIZE;
  const int L = l_cache_ > 0 ? l_cache_ : MODEL_CONFIG::CONV_L_CACHE;
  const int state_len = std::max(0, L - 1);

  char name[64];
  snprintf(name, sizeof(name), "conv_%d", layer_idx_);
  NNOPT_LAYER_CHECK_INPUT(name, queue, hidden_states, (size_t)seq_len * (size_t)H);

  if (!ensure_buffers_(seq_len)) return nullptr;
  cl_int err = CL_SUCCESS;

  // 1) in_proj: [seq,H] -> [seq,3H]  (writes buf_in_proj_)
  pytorch_linear(queue, seq_len, 3 * H, H, hidden_states, in_proj_w_, buf_in_proj_);

  // ── Decode fast path (seq_len == 1): one fused kernel collapses the next
  // 7 launches (transposes are no-ops at S=1; split / Bx-mul / conv1d /
  // state-update / mul-by-C all reduce to per-channel work). Keeps prefill
  // and any seq_q>1 call on the canonical multi-launch path below for
  // correctness.
  if (seq_len == 1 && conv_state_ && (l_cache_ - 1) > 0) {
    cl_int e = CL_SUCCESS;
    int H_arg = H;
    int L_arg = l_cache_;
    e  = clSetKernelArg(block_decode_kernel_, 0, sizeof(cl_mem), &buf_in_proj_);
    e |= clSetKernelArg(block_decode_kernel_, 1, sizeof(cl_mem), &conv_w_);
    e |= clSetKernelArg(block_decode_kernel_, 2, sizeof(cl_mem), &conv_state_);
    e |= clSetKernelArg(block_decode_kernel_, 3, sizeof(cl_mem), &buf_y_seq_);
    e |= clSetKernelArg(block_decode_kernel_, 4, sizeof(int),    &H_arg);
    e |= clSetKernelArg(block_decode_kernel_, 5, sizeof(int),    &L_arg);
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("Convolution[%d]: setArgs(block_decode): %d", layer_idx_, e); return nullptr; }
    const size_t WG = 64;
    size_t gws = (size_t)((H + WG - 1) / WG) * WG;  // round-up to lws multiple
    size_t lws = WG;
    e = clEnqueueNDRangeKernel(queue, block_decode_kernel_, 1, nullptr, &gws, &lws, 0, nullptr,
                               KernelProfiler::event_for("conv_block_decode"));
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("Convolution[%d]: dispatch(block_decode): %d", layer_idx_, e); return nullptr; }

    // out_proj: buf_y_seq_ -> buf_out_
    if (!pytorch_linear(queue, seq_len, H, H, buf_y_seq_, out_proj_w_, buf_out_)) return nullptr;
    NNOPT_LAYER_CHECK_FMT("conv_%d", layer_idx_, queue, buf_out_, (size_t)seq_len * (size_t)H);
    (void)start_pos;
    return buf_out_;
  }

  // 2) transpose buf_in_proj_ -> buf_in_proj_T_: [3H, seq]
  {
    const int total = 3 * H * seq_len;
    err  = clSetKernelArg(copy_transpose_kernel_, 0, sizeof(cl_mem), &buf_in_proj_);
    err |= clSetKernelArg(copy_transpose_kernel_, 1, sizeof(cl_mem), &buf_in_proj_T_);
    err |= clSetKernelArg(copy_transpose_kernel_, 2, sizeof(int),    &seq_len);
    err |= clSetKernelArg(copy_transpose_kernel_, 3, sizeof(int),    &H);
    err |= clSetKernelArg(copy_transpose_kernel_, 4, sizeof(int),    &total);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Convolution[%d]: setArgs(copy_transpose): %d", layer_idx_, err); return nullptr; }
    size_t gws = (size_t)total;
    err = clEnqueueNDRangeKernel(queue, copy_transpose_kernel_, 1, nullptr, &gws, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("conv_copy_transpose"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Convolution[%d]: dispatch(copy_transpose): %d", layer_idx_, err); return nullptr; }
  }

  // 3) split buf_in_proj_T_ -> buf_B_, buf_C_, buf_X_
  {
    const int total = H * seq_len;
    err  = clSetKernelArg(split_chunk3_kernel_, 0, sizeof(cl_mem), &buf_in_proj_T_);
    err |= clSetKernelArg(split_chunk3_kernel_, 1, sizeof(cl_mem), &buf_B_);
    err |= clSetKernelArg(split_chunk3_kernel_, 2, sizeof(cl_mem), &buf_C_);
    err |= clSetKernelArg(split_chunk3_kernel_, 3, sizeof(cl_mem), &buf_X_);
    err |= clSetKernelArg(split_chunk3_kernel_, 4, sizeof(int),    &seq_len);
    err |= clSetKernelArg(split_chunk3_kernel_, 5, sizeof(int),    &H);
    err |= clSetKernelArg(split_chunk3_kernel_, 6, sizeof(int),    &total);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Convolution[%d]: setArgs(split_chunk3): %d", layer_idx_, err); return nullptr; }
    size_t gws = (size_t)total;
    err = clEnqueueNDRangeKernel(queue, split_chunk3_kernel_, 1, nullptr, &gws, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("conv_split_chunk3"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Convolution[%d]: dispatch(split_chunk3): %d", layer_idx_, err); return nullptr; }
  }

  // 4) buf_Bx_ = buf_B_ * buf_X_
  {
    const int total = H * seq_len;
    err  = clSetKernelArg(elem_mul_kernel_, 0, sizeof(cl_mem), &buf_B_);
    err |= clSetKernelArg(elem_mul_kernel_, 1, sizeof(cl_mem), &buf_X_);
    err |= clSetKernelArg(elem_mul_kernel_, 2, sizeof(cl_mem), &buf_Bx_);
    err |= clSetKernelArg(elem_mul_kernel_, 3, sizeof(int),    &total);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Convolution[%d]: setArgs(elem_mul): %d", layer_idx_, err); return nullptr; }
    size_t gws = (size_t)total;
    err = clEnqueueNDRangeKernel(queue, elem_mul_kernel_, 1, nullptr, &gws, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("conv_pointwise_mul"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Convolution[%d]: dispatch(elem_mul): %d", layer_idx_, err); return nullptr; }
  }

  NNOPT_LAYER_CHECK_FMT("conv_%d_Bx", layer_idx_, queue, buf_Bx_, (size_t)H * (size_t)seq_len);
  NNOPT_LAYER_CHECK_FMT("conv_%d_B",  layer_idx_, queue, buf_B_,  (size_t)H * (size_t)seq_len);
  NNOPT_LAYER_CHECK_FMT("conv_%d_C",  layer_idx_, queue, buf_C_,  (size_t)H * (size_t)seq_len);
  NNOPT_LAYER_CHECK_FMT("conv_%d_Xc", layer_idx_, queue, buf_X_,  (size_t)H * (size_t)seq_len);

  // 5) conv1d with cache → buf_conv_out_T_
  {
    cl_mem prev = (state_len > 0) ? conv_state_ : nullptr;
    const int total = H * seq_len;
    err  = clSetKernelArg(conv1d_cache_kernel_, 0, sizeof(cl_mem), &buf_Bx_);
    err |= clSetKernelArg(conv1d_cache_kernel_, 1, sizeof(cl_mem), &conv_w_);
    err |= clSetKernelArg(conv1d_cache_kernel_, 2, sizeof(cl_mem), &prev);
    err |= clSetKernelArg(conv1d_cache_kernel_, 3, sizeof(cl_mem), &buf_conv_out_T_);
    err |= clSetKernelArg(conv1d_cache_kernel_, 4, sizeof(int),    &seq_len);
    err |= clSetKernelArg(conv1d_cache_kernel_, 5, sizeof(int),    &H);
    err |= clSetKernelArg(conv1d_cache_kernel_, 6, sizeof(int),    &L);
    err |= clSetKernelArg(conv1d_cache_kernel_, 7, sizeof(int),    &total);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Convolution[%d]: setArgs(conv1d): %d", layer_idx_, err); return nullptr; }
    size_t gws = (size_t)total;
    err = clEnqueueNDRangeKernel(queue, conv1d_cache_kernel_, 1, nullptr, &gws, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("conv1d_causal_with_cache"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Convolution[%d]: dispatch(conv1d): %d", layer_idx_, err); return nullptr; }
  }

  // 6) update conv_state_ with last (L-1) Bx values
  if (conv_state_ && state_len > 0) {
    const int total = H * seq_len;
    err  = clSetKernelArg(copy_transpose_back_kernel_, 0, sizeof(cl_mem), &buf_Bx_);
    err |= clSetKernelArg(copy_transpose_back_kernel_, 1, sizeof(cl_mem), &buf_Bx_seq_);
    err |= clSetKernelArg(copy_transpose_back_kernel_, 2, sizeof(int),    &seq_len);
    err |= clSetKernelArg(copy_transpose_back_kernel_, 3, sizeof(int),    &H);
    err |= clSetKernelArg(copy_transpose_back_kernel_, 4, sizeof(int),    &total);
    if (err == CL_SUCCESS) {
      size_t gws = (size_t)total;
      err = clEnqueueNDRangeKernel(queue, copy_transpose_back_kernel_, 1, nullptr, &gws, nullptr, 0, nullptr,
                                   KernelProfiler::event_for("conv_copy_transpose_back_state"));
    }
    if (err == CL_SUCCESS) {
      err  = clSetKernelArg(update_cache_kernel_, 0, sizeof(cl_mem), &buf_Bx_seq_);
      err |= clSetKernelArg(update_cache_kernel_, 1, sizeof(cl_mem), &conv_state_);
      err |= clSetKernelArg(update_cache_kernel_, 2, sizeof(int),    &seq_len);
      err |= clSetKernelArg(update_cache_kernel_, 3, sizeof(int),    &H);
      err |= clSetKernelArg(update_cache_kernel_, 4, sizeof(int),    &state_len);
      if (err == CL_SUCCESS) {
        size_t gws = (size_t)H;
        err = clEnqueueNDRangeKernel(queue, update_cache_kernel_, 1, nullptr, &gws, nullptr, 0, nullptr,
                                     KernelProfiler::event_for("conv_update_state"));
      }
    }
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Convolution[%d]: state update failed: %d", layer_idx_, err); return nullptr; }
  }

  // 7) buf_y_T_ = buf_C_ * buf_conv_out_T_
  {
    const int total = H * seq_len;
    err  = clSetKernelArg(elem_mul2_kernel_, 0, sizeof(cl_mem), &buf_C_);
    err |= clSetKernelArg(elem_mul2_kernel_, 1, sizeof(cl_mem), &buf_conv_out_T_);
    err |= clSetKernelArg(elem_mul2_kernel_, 2, sizeof(cl_mem), &buf_y_T_);
    err |= clSetKernelArg(elem_mul2_kernel_, 3, sizeof(int),    &total);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Convolution[%d]: setArgs(elem_mul2): %d", layer_idx_, err); return nullptr; }
    size_t gws = (size_t)total;
    err = clEnqueueNDRangeKernel(queue, elem_mul2_kernel_, 1, nullptr, &gws, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("conv_mul_C"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Convolution[%d]: dispatch(elem_mul2): %d", layer_idx_, err); return nullptr; }
  }

  // 8) transpose buf_y_T_ -> buf_y_seq_ [seq, H]
  {
    const int total = H * seq_len;
    err  = clSetKernelArg(copy_transpose_back_kernel_, 0, sizeof(cl_mem), &buf_y_T_);
    err |= clSetKernelArg(copy_transpose_back_kernel_, 1, sizeof(cl_mem), &buf_y_seq_);
    err |= clSetKernelArg(copy_transpose_back_kernel_, 2, sizeof(int),    &seq_len);
    err |= clSetKernelArg(copy_transpose_back_kernel_, 3, sizeof(int),    &H);
    err |= clSetKernelArg(copy_transpose_back_kernel_, 4, sizeof(int),    &total);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Convolution[%d]: setArgs(transpose_back): %d", layer_idx_, err); return nullptr; }
    size_t gws = (size_t)total;
    err = clEnqueueNDRangeKernel(queue, copy_transpose_back_kernel_, 1, nullptr, &gws, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("conv_copy_transpose_back"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Convolution[%d]: dispatch(transpose_back): %d", layer_idx_, err); return nullptr; }
  }

  // 9) out_proj: buf_y_seq_ -> buf_out_
  if (!pytorch_linear(queue, seq_len, H, H, buf_y_seq_, out_proj_w_, buf_out_)) return nullptr;

  NNOPT_LAYER_CHECK_FMT("conv_%d", layer_idx_, queue, buf_out_, (size_t)seq_len * (size_t)H);

  (void)start_pos;  // cache is single-stream
  return buf_out_;  // BORROWED — caller must NOT release
}
