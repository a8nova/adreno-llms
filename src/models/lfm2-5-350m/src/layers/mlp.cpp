// Reference: model_info/transformers_src/modeling_lfm2.py:135-154 Lfm2MLP.forward
// Implements: w2(silu(w1(x)) * w3(x))

#include "layers/mlp.h"

#include "debug_utils.h"
#include "kernel_profiler.h"
#include "model_config.h"
#include "opencl_context.h"
#include "utils.h"
#include "weights.h"

#include <algorithm>
#include <string>
#include <vector>

namespace {
static bool set_arg_checked(cl_kernel k, cl_uint idx, size_t sz, const void* ptr, const char* name) {
  const cl_int err = clSetKernelArg(k, idx, sz, ptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("clSetKernelArg(%s) idx=%u failed: %d", name, (unsigned)idx, (int)err);
    return false;
  }
  return true;
}
} // namespace

Mlp::Mlp(OpenCLContext& cl_ctx, Weights& weights, const std::string& prefix, int layer_idx)
    : cl_ctx_(cl_ctx), weights_(weights), prefix_(prefix), layer_idx_(layer_idx) {}

Mlp::~Mlp() {
  if (silu_mul_kernel_) clReleaseKernel(silu_mul_kernel_);
  if (program_) clReleaseProgram(program_);
  if (buf_gate_) clReleaseMemObject(buf_gate_);
  if (buf_up_)   clReleaseMemObject(buf_up_);
  if (buf_out_)  clReleaseMemObject(buf_out_);
}

bool Mlp::ensure_buffers_(int seq_len) {
  if (seq_len <= buf_seq_capacity_ && buf_out_) return true;
  if (intermediate_size_ <= 0) {
    const auto w1_shape = weights_.get_shape(prefix_ + ".w1.weight");
    if (w1_shape.size() != 2) return false;
    intermediate_size_ = w1_shape[0];
  }
  if (buf_gate_) { clReleaseMemObject(buf_gate_); buf_gate_ = nullptr; }
  if (buf_up_)   { clReleaseMemObject(buf_up_);   buf_up_   = nullptr; }
  if (buf_out_)  { clReleaseMemObject(buf_out_);  buf_out_  = nullptr; }
  cl_context ctx = cl_ctx_.context();
  cl_int err = CL_SUCCESS;
  const size_t s = (size_t)seq_len;
  const size_t I_bytes = s * (size_t)intermediate_size_ * sizeof(nnopt_storage_t);
  const size_t H_bytes = s * (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t);
  buf_gate_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, I_bytes, nullptr, &err);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Mlp[%d]: alloc gate: %d", layer_idx_, err); return false; }
  buf_up_   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, I_bytes, nullptr, &err);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Mlp[%d]: alloc up: %d",   layer_idx_, err); return false; }
  buf_out_  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, H_bytes, nullptr, &err);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Mlp[%d]: alloc out: %d",  layer_idx_, err); return false; }
  buf_seq_capacity_ = seq_len;
  return true;
}

bool Mlp::initialize() {
  // Contract keys (Mlp.json):
  //   feed_forward_w1_weight: model.layers.{i}.feed_forward.w1.weight
  //   feed_forward_w3_weight: model.layers.{i}.feed_forward.w3.weight
  //   feed_forward_w2_weight: model.layers.{i}.feed_forward.w2.weight
  const std::string w1_key = prefix_ + ".w1.weight";
  const std::string w2_key = prefix_ + ".w2.weight";
  const std::string w3_key = prefix_ + ".w3.weight";
  w1_ = weights_.get_buffer(w1_key);
  w2_ = weights_.get_buffer(w2_key);
  w3_ = weights_.get_buffer(w3_key);
  if (!w1_ || !w2_ || !w3_) {
    NNOPT_ERROR_FMT("Mlp[%d]: missing weights under %s", layer_idx_, prefix_.c_str());
    return false;
  }

  program_ = cl_ctx_.build_program_from_file("kernels/mlp.cl");
  if (!program_) return false;

  cl_int err = CL_SUCCESS;
  silu_mul_kernel_ = clCreateKernel(program_, "silu_mul", &err);
  if (err != CL_SUCCESS || !silu_mul_kernel_) {
    NNOPT_ERROR_FMT("Mlp[%d]: clCreateKernel(silu_mul) failed: %d", layer_idx_, (int)err);
    return false;
  }

  NNOPT_LAYER_INIT_FMT("mlp_%d", layer_idx_);
  return true;
}

cl_mem Mlp::forward(cl_command_queue queue, cl_mem x, int seq_len) {
  if (!x || seq_len <= 0) return nullptr;
  if (!ensure_buffers_(seq_len)) return nullptr;
  const int H = MODEL_CONFIG::HIDDEN_SIZE;
  const int I = intermediate_size_;

  // gate = w1(x)  → buf_gate_
  if (!pytorch_linear(queue, seq_len, I, H, x, w1_, buf_gate_)) return nullptr;
  // up = w3(x)    → buf_up_
  if (!pytorch_linear(queue, seq_len, I, H, x, w3_, buf_up_))   return nullptr;
  // buf_gate_ = silu(buf_gate_) * buf_up_
  const int total = seq_len * I;
  if (!set_arg_checked(silu_mul_kernel_, 0, sizeof(cl_mem), &buf_gate_, "a") ||
      !set_arg_checked(silu_mul_kernel_, 1, sizeof(cl_mem), &buf_up_,   "b") ||
      !set_arg_checked(silu_mul_kernel_, 2, sizeof(cl_mem), &buf_gate_, "out") ||
      !set_arg_checked(silu_mul_kernel_, 3, sizeof(int),    &total,     "n")) return nullptr;
  size_t gws[1] = {(size_t)total};
  cl_int err = clEnqueueNDRangeKernel(queue, silu_mul_kernel_, 1, nullptr, gws, nullptr, 0, nullptr,
                                      KernelProfiler::event_for("silu_mul"));
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Mlp[%d]: silu_mul: %d", layer_idx_, err); return nullptr; }
  NNOPT_DEBUG_SYNC(queue);
  // out = w2(gate)  → buf_out_
  if (!pytorch_linear(queue, seq_len, H, I, buf_gate_, w2_, buf_out_)) return nullptr;
  return buf_out_;  // BORROWED
}
