// Reference: model_info/modeling_openelm.py (OpenELMFeedForward.forward / OpenELMDecoderLayer.forward)

#include "layers/mlp.h"

#include "debug_utils.h"
#include "kernel_profiler.h"
#include "layers/layer_norm.h"
#include "model_config.h"
#include "utils.h"
#include "weights.h"

#include <CL/cl.h>
#include <algorithm>
#include <clblast.h>
#include <cmath>
#include <cstring>
#include <string>

namespace {
inline bool set_arg_checked(cl_kernel k, cl_uint idx, size_t sz, const void* val, const char* name) {
  const cl_int err = clSetKernelArg(k, idx, sz, val);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("clSetKernelArg(%u) %s failed (%d)", idx, name, (int)err);
    return false;
  }
  return true;
}
} // namespace

Mlp::Mlp(OpenCLContext& cl_ctx, Weights& weights, int layer_idx)
    : cl_ctx_(cl_ctx), weights_(weights), layer_idx_(layer_idx) {
  prefix_layer_ = std::string("transformer.layers.") + std::to_string(layer_idx_);
  prefix_ffn_ = prefix_layer_ + ".ffn";
}

Mlp::~Mlp() {
  release_activation_buffers_();
  if (kernel_swiglu_) clReleaseKernel(kernel_swiglu_);
  if (program_) clReleaseProgram(program_);
}

void Mlp::release_activation_buffers_() {
  if (buf_proj1_out_) clReleaseMemObject(buf_proj1_out_); buf_proj1_out_ = nullptr; buf_proj1_out_cap_ = 0;
  if (buf_gate_)      clReleaseMemObject(buf_gate_);      buf_gate_      = nullptr; buf_gate_cap_      = 0;
  if (buf_proj2_out_) clReleaseMemObject(buf_proj2_out_); buf_proj2_out_ = nullptr; buf_proj2_out_cap_ = 0;
}

namespace {
inline bool grow_buffer(cl_context ctx, cl_mem* buf, size_t* cap, size_t required_bytes, const char* label, int layer_idx) {
  if (*cap >= required_bytes && *buf != nullptr) return true;
  if (*buf) clReleaseMemObject(*buf);
  cl_int err = CL_SUCCESS;
  *buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, required_bytes, nullptr, &err);
  if (err != CL_SUCCESS || !*buf) {
    NNOPT_ERROR_FMT("Mlp[%d]: grow %s to %zu B failed (%d)", layer_idx, label, required_bytes, (int)err);
    *buf = nullptr;
    *cap = 0;
    return false;
  }
  *cap = required_bytes;
  return true;
}
}  // namespace

bool Mlp::ensure_activation_buffers_(int seq_len) {
  const int H = MODEL_CONFIG::MODEL_DIM;
  const int I = MODEL_CONFIG::FFN_INTERMEDIATE_SIZE[layer_idx_];
  const size_t fp = sizeof(nnopt_storage_t);

  cl_context ctx = cl_ctx_.context();

  if (!grow_buffer(ctx, &buf_proj1_out_, &buf_proj1_out_cap_,
                   (size_t)seq_len * (size_t)(2 * I) * fp, "proj1_out", layer_idx_)) return false;
  if (!grow_buffer(ctx, &buf_gate_,      &buf_gate_cap_,
                   (size_t)seq_len * (size_t)I       * fp, "gate",      layer_idx_)) return false;
  if (!grow_buffer(ctx, &buf_proj2_out_, &buf_proj2_out_cap_,
                   (size_t)seq_len * (size_t)H       * fp, "proj2_out", layer_idx_)) return false;
  return true;
}

bool Mlp::initialize() {
  // Weights
  ffn_norm_w_ = weights_.get_buffer(prefix_layer_ + ".ffn_norm.weight");
  proj1_w_ = weights_.get_buffer(prefix_ffn_ + ".proj_1.weight");
  proj2_w_ = weights_.get_buffer(prefix_ffn_ + ".proj_2.weight");

  if (!ffn_norm_w_ || !proj1_w_ || !proj2_w_) {
    NNOPT_ERROR_FMT("Mlp[%d]: missing weight(s) ffn_norm=%p proj1=%p proj2=%p", layer_idx_,
                    (void*)ffn_norm_w_, (void*)proj1_w_, (void*)proj2_w_);
    return false;
  }

  // Kernels
  program_ = cl_ctx_.build_program_from_file("kernels/mlp.cl");
  if (!program_) return false;
  cl_int err = CL_SUCCESS;
  kernel_swiglu_ = clCreateKernel(program_, "swiglu_forward", &err);
  if (err != CL_SUCCESS || !kernel_swiglu_) {
    NNOPT_ERROR_FMT("Mlp[%d]: clCreateKernel(swiglu_forward) failed (%d)", layer_idx_, (int)err);
    return false;
  }

  return true;
}

cl_mem Mlp::forward(cl_command_queue queue, cl_mem input, int seq_len) {
  NNOPT_CHECKPOINT("Mlp::forward");

  const int H = MODEL_CONFIG::MODEL_DIM;
  const int I = MODEL_CONFIG::FFN_INTERMEDIATE_SIZE[layer_idx_];

  if (!ensure_activation_buffers_(seq_len)) return nullptr;

  cl_int err = CL_SUCCESS;

  // NOTE: ffn_norm is applied by Model::forward via post_attn_norm_[i] BEFORE
  // calling Mlp::forward(input, ...). OpenELMFeedForwardNetwork.forward does
  // NOT renormalize — applying rmsnorm here would double-normalize the input.
  if (layer_idx_ == 0) {
    NNOPT_LAYER_CHECK("block0_sub_ffn_norm_out", queue, input, (size_t)seq_len * (size_t)H);
  }

  // proj_1: [M,H] x [2I,H]^T -> [M,2I]
  const int N1 = 2 * I;
  if (!pytorch_linear(queue, seq_len, N1, H, input, proj1_w_, buf_proj1_out_)) {
    NNOPT_ERROR_FMT("Mlp[%d]: pytorch_linear proj_1 failed", layer_idx_);
    return nullptr;
  }
  if (layer_idx_ == 0) {
    NNOPT_LAYER_CHECK("block0_sub_ffn_proj_1_out", queue, buf_proj1_out_, (size_t)seq_len * (size_t)N1);
  }

  // SwiGLU: out[i, c] = silu(proj1_out[i, c]) * proj1_out[i, I + c]
  if (!set_arg_checked(kernel_swiglu_, 0, sizeof(cl_mem), &buf_proj1_out_, "gate_up") ||
      !set_arg_checked(kernel_swiglu_, 1, sizeof(cl_mem), &buf_gate_, "out") ||
      !set_arg_checked(kernel_swiglu_, 2, sizeof(int), &I, "intermediate")) {
    return nullptr;
  }
  {
    const size_t gws = (size_t)seq_len * (size_t)I;
    cl_event* evt = KernelProfiler::event_for("swiglu");
    err = clEnqueueNDRangeKernel(queue, kernel_swiglu_, 1, nullptr, &gws, nullptr, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
      NNOPT_ERROR_FMT("Mlp[%d]: swiglu_forward launch failed (%d)", layer_idx_, (int)err);
      return nullptr;
    }
    NNOPT_DEBUG_SYNC(queue);
  }

  if (layer_idx_ == 0) {
    NNOPT_LAYER_CHECK("block0_sub_ffn_proj_2_in", queue, buf_gate_, (size_t)seq_len * (size_t)I);
  }

  // proj_2: [M,I] x [H,I]^T -> [M,H]. When pending_residual_inout_ is set
  // (Step 5), try the fused-radd path: residual += proj_2(gate). Falls back
  // to un-fused write if shape ineligible.
  if (pending_residual_inout_ != nullptr) {
    if (pytorch_linear_radd(queue, seq_len, H, I, buf_gate_, proj2_w_, pending_residual_inout_)) {
      if (layer_idx_ == 0) {
        NNOPT_LAYER_CHECK("block0_sub_ffn_proj_2_out", queue, pending_residual_inout_, (size_t)seq_len * (size_t)H);
      }
      return pending_residual_inout_;
    }
    // Fallthrough — radd ineligible.
  }
  if (!pytorch_linear(queue, seq_len, H, I, buf_gate_, proj2_w_, buf_proj2_out_)) {
    NNOPT_ERROR_FMT("Mlp[%d]: pytorch_linear proj_2 failed", layer_idx_);
    return nullptr;
  }

  if (layer_idx_ == 0) {
    NNOPT_LAYER_CHECK("block0_sub_ffn_proj_2_out", queue, buf_proj2_out_, (size_t)seq_len * (size_t)H);
  }

  // BORROWED handle — caller must NOT release. Owned by *this until next call.
  return buf_proj2_out_;
}

cl_mem Mlp::forward_with_residual(cl_command_queue queue,
                                  cl_mem input,
                                  int seq_len,
                                  cl_mem residual_inout) {
  pending_residual_inout_ = residual_inout;
  cl_mem result = forward(queue, input, seq_len);
  pending_residual_inout_ = nullptr;
  return result;
}
