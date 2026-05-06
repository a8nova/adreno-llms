// Reference: model_info/transformers_src/modeling_mamba.py (Transformers 4.39 dev) MambaMixer.slow_forward
// Implements the Mamba-1 mixer (in_proj -> depthwise causal conv1d -> SiLU -> x_proj -> dt_proj+softplus
// -> selective_scan -> gate(y*silu(z)) -> out_proj).

#include "layers/ssm.h"

#include "debug_utils.h"
#include "kernel_profiler.h"
#include "model_config.h"
#include "opencl_context.h"
#include "utils.h"
#include "weights.h"

#include <clblast.h>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

Ssm::Ssm(OpenCLContext& cl_ctx, Weights& weights, const std::string& prefix, int layer_idx)
    : cl_ctx_(cl_ctx), weights_(weights), prefix_(prefix), layer_idx_(layer_idx) {}

Ssm::~Ssm() {
  // NOTE: weight buffers are owned by Weights (do NOT clRelease)

  if (dt_bias_expanded_) clReleaseMemObject(dt_bias_expanded_);
  if (A_expanded_) clReleaseMemObject(A_expanded_);
  if (D_expanded_) clReleaseMemObject(D_expanded_);

  if (conv_state_) clReleaseMemObject(conv_state_);
  if (ssm_state_) clReleaseMemObject(ssm_state_);

  if (buf_xz_)     clReleaseMemObject(buf_xz_);
  if (buf_x_)      clReleaseMemObject(buf_x_);
  if (buf_z_)      clReleaseMemObject(buf_z_);
  if (buf_x_conv_) clReleaseMemObject(buf_x_conv_);
  if (buf_xproj_)  clReleaseMemObject(buf_xproj_);
  if (buf_dt_raw_) clReleaseMemObject(buf_dt_raw_);
  if (buf_B_)      clReleaseMemObject(buf_B_);
  if (buf_C_)      clReleaseMemObject(buf_C_);
  if (buf_dt_)     clReleaseMemObject(buf_dt_);
  if (buf_y_)      clReleaseMemObject(buf_y_);
  if (buf_out_)    clReleaseMemObject(buf_out_);

  if (scan_kernel_) clReleaseKernel(scan_kernel_);
  if (softplus_kernel_) clReleaseKernel(softplus_kernel_);
  if (silu_mul_kernel_) clReleaseKernel(silu_mul_kernel_);
  if (scan_fused_kernel_) clReleaseKernel(scan_fused_kernel_);
  if (program_scan_) clReleaseProgram(program_scan_);

  if (conv_kernel_) clReleaseKernel(conv_kernel_);
  if (update_conv_state_kernel_) clReleaseKernel(update_conv_state_kernel_);
  if (silu_inplace_kernel_) clReleaseKernel(silu_inplace_kernel_);
  if (program_conv_) clReleaseProgram(program_conv_);

  if (split_kernel_) clReleaseKernel(split_kernel_);
  if (split_xproj_kernel_) clReleaseKernel(split_xproj_kernel_);
  if (bias_add_kernel_) clReleaseKernel(bias_add_kernel_);
  if (program_ssm_) clReleaseProgram(program_ssm_);
}

// Allocate (or reallocate) the 11 activation buffers used inside Ssm::forward
// so they're sized for at least `M` rows. Returns false on alloc failure.
// Called on the first forward() call and any time seq_len exceeds the current
// capacity (rare — only happens if a prefill comes in larger than any prior
// prefill).
bool Ssm::ensure_activation_buffers_(int M) {
  if (M <= buf_capacity_M_ && buf_xz_) return true;

  // Free the old set if growing.
  if (buf_xz_)     { clReleaseMemObject(buf_xz_);     buf_xz_     = nullptr; }
  if (buf_x_)      { clReleaseMemObject(buf_x_);      buf_x_      = nullptr; }
  if (buf_z_)      { clReleaseMemObject(buf_z_);      buf_z_      = nullptr; }
  if (buf_x_conv_) { clReleaseMemObject(buf_x_conv_); buf_x_conv_ = nullptr; }
  if (buf_xproj_)  { clReleaseMemObject(buf_xproj_);  buf_xproj_  = nullptr; }
  if (buf_dt_raw_) { clReleaseMemObject(buf_dt_raw_); buf_dt_raw_ = nullptr; }
  if (buf_B_)      { clReleaseMemObject(buf_B_);      buf_B_      = nullptr; }
  if (buf_C_)      { clReleaseMemObject(buf_C_);      buf_C_      = nullptr; }
  if (buf_dt_)     { clReleaseMemObject(buf_dt_);     buf_dt_     = nullptr; }
  if (buf_y_)      { clReleaseMemObject(buf_y_);      buf_y_      = nullptr; }
  if (buf_out_)    { clReleaseMemObject(buf_out_);    buf_out_    = nullptr; }

  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx_.context();
  const int N_inproj = 2 * d_inner_;
  const int N_xproj  = dt_rank_ + 2 * d_state_;

  buf_xz_     = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * (size_t)N_inproj * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS) return false;
  buf_x_      = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * (size_t)d_inner_ * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS) return false;
  buf_z_      = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * (size_t)d_inner_ * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS) return false;
  buf_x_conv_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * (size_t)d_inner_ * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS) return false;
  buf_xproj_  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * (size_t)N_xproj  * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS) return false;
  buf_dt_raw_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * (size_t)dt_rank_ * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS) return false;
  buf_B_      = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * (size_t)d_state_ * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS) return false;
  buf_C_      = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * (size_t)d_state_ * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS) return false;
  buf_dt_     = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * (size_t)d_inner_ * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS) return false;
  buf_y_      = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * (size_t)d_inner_ * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS) return false;
  buf_out_    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * (size_t)d_model_ * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS) return false;

  buf_capacity_M_ = M;
  return true;
}

bool Ssm::initialize() {
  // Weight keys per scaffold contract:
  // backbone.layers.{i}.mixer.{in_proj,x_proj,dt_proj,out_proj,conv1d,A_log,D}.*
  in_proj_ = weights_.get_buffer(prefix_ + ".in_proj.weight");
  x_proj_ = weights_.get_buffer(prefix_ + ".x_proj.weight");
  dt_proj_ = weights_.get_buffer(prefix_ + ".dt_proj.weight");
  dt_bias_ = weights_.get_buffer(prefix_ + ".dt_proj.bias");
  out_proj_ = weights_.get_buffer(prefix_ + ".out_proj.weight");
  conv_w_ = weights_.get_buffer(prefix_ + ".conv1d.weight");
  conv_b_ = weights_.get_buffer(prefix_ + ".conv1d.bias", /*optional=*/true);
  A_log_ = weights_.get_buffer(prefix_ + ".A_log");
  D_ = weights_.get_buffer(prefix_ + ".D");

  if (!in_proj_ || !x_proj_ || !dt_proj_ || !dt_bias_ || !out_proj_ || !conv_w_ || !A_log_ || !D_) {
    NNOPT_ERROR_FMT("Ssm::initialize missing required weights under prefix=%s", prefix_.c_str());
    return false;
  }

  d_model_ = MODEL_CONFIG::HIDDEN_SIZE;
  d_inner_ = MODEL_CONFIG::INTERMEDIATE_SIZE;
  d_state_ = MODEL_CONFIG::STATE_SIZE;
  d_conv_ = MODEL_CONFIG::CONV_KERNEL;
  dt_rank_ = MODEL_CONFIG::TIME_STEP_RANK;

  // Persistent states: NOTE scaffold kernels expect conv_state and h (ssm_state) are fp32.
  {
    cl_int err = CL_SUCCESS;
    conv_state_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                 (size_t)d_inner_ * (size_t)(d_conv_ - 1) * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
      NNOPT_ERROR_FMT("Ssm: conv_state alloc failed: %d", (int)err);
      return false;
    }
    ssm_state_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                (size_t)d_inner_ * (size_t)d_state_ * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
      NNOPT_ERROR_FMT("Ssm: ssm_state alloc failed: %d", (int)err);
      return false;
    }

    std::vector<float> zeros_conv((size_t)d_inner_ * (size_t)(d_conv_ - 1), 0.0f);
    std::vector<float> zeros_ssm((size_t)d_inner_ * (size_t)d_state_, 0.0f);
    clEnqueueWriteBuffer(cl_ctx_.queue(), conv_state_, CL_TRUE, 0, zeros_conv.size() * sizeof(float),
                         zeros_conv.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(cl_ctx_.queue(), ssm_state_, CL_TRUE, 0, zeros_ssm.size() * sizeof(float),
                         zeros_ssm.data(), 0, nullptr, nullptr);
  }

  if (!prepare_expanded_params_()) {
    NNOPT_ERROR_FMT("Ssm::initialize prepare_expanded_params_ failed for prefix=%s", prefix_.c_str());
    return false;
  }

  // Build kernels
  program_scan_ = cl_ctx_.build_program_from_file("kernels/selective_scan.cl");
  if (!program_scan_) return false;
  cl_int err = CL_SUCCESS;
  scan_kernel_ = clCreateKernel(program_scan_, "selective_scan", &err);
  if (err != CL_SUCCESS) return false;
  softplus_kernel_ = clCreateKernel(program_scan_, "softplus", &err);
  if (err != CL_SUCCESS) return false;
  silu_mul_kernel_ = clCreateKernel(program_scan_, "silu_mul", &err);
  if (err != CL_SUCCESS) return false;
  // Fused selective_scan kernel — non-fatal if it fails to build (decode
  // falls back to unfused forward()). Saves 3 launches per layer per token.
  cl_int fused_err = CL_SUCCESS;
  scan_fused_kernel_ = clCreateKernel(program_scan_, "selective_scan_fused", &fused_err);
  if (fused_err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Ssm[%d]: selective_scan_fused build failed (decode will use unfused path): %d",
                    layer_idx_, (int)fused_err);
    scan_fused_kernel_ = nullptr;
  }

  program_conv_ = cl_ctx_.build_program_from_file("kernels/causal_conv1d.cl");
  if (!program_conv_) return false;
  conv_kernel_ = clCreateKernel(program_conv_, "causal_conv1d", &err);
  if (err != CL_SUCCESS) return false;
  update_conv_state_kernel_ = clCreateKernel(program_conv_, "update_conv_state", &err);
  if (err != CL_SUCCESS) return false;
  silu_inplace_kernel_ = clCreateKernel(program_conv_, "silu_inplace", &err);
  if (err != CL_SUCCESS) return false;

  program_ssm_ = cl_ctx_.build_program_from_file("kernels/ssm.cl");
  if (!program_ssm_) return false;
  split_kernel_ = clCreateKernel(program_ssm_, "split_xz", &err);
  if (err != CL_SUCCESS) return false;
  split_xproj_kernel_ = clCreateKernel(program_ssm_, "split_xproj", &err);
  if (err != CL_SUCCESS) return false;
  bias_add_kernel_ = clCreateKernel(program_ssm_, "bias_add_rows", &err);
  if (err != CL_SUCCESS) return false;

  NNOPT_LAYER_INIT_FMT("ssm_%d", layer_idx_);
  return true;
}

bool Ssm::prepare_expanded_params_() {
  // dt_bias_expanded_ [d_inner] in storage dtype (nnopt_storage_t)
  if (dt_bias_expanded_) {
    clReleaseMemObject(dt_bias_expanded_);
    dt_bias_expanded_ = nullptr;
  }
  cl_int err = CL_SUCCESS;
  dt_bias_expanded_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                     (size_t)d_inner_ * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS) return false;

  // Read dt_bias from device into host float (handle fp16/fp32 storage via nnopt_storage_t)
  const size_t dt_bias_numel = weights_.get_num_elements(prefix_ + ".dt_proj.bias");
  std::vector<nnopt_storage_t> dtb(dt_bias_numel);
  clEnqueueReadBuffer(cl_ctx_.queue(), dt_bias_, CL_TRUE, 0, dt_bias_numel * sizeof(nnopt_storage_t), dtb.data(), 0,
                      nullptr, nullptr);
  std::vector<nnopt_storage_t> dtb_exp((size_t)d_inner_);
  if (dt_bias_numel == 1) {
    for (int i = 0; i < d_inner_; ++i) dtb_exp[i] = dtb[0];
  } else if ((int)dt_bias_numel == d_inner_) {
    std::memcpy(dtb_exp.data(), dtb.data(), (size_t)d_inner_ * sizeof(nnopt_storage_t));
  } else {
    for (int i = 0; i < d_inner_; ++i) dtb_exp[i] = dtb[i % dt_bias_numel];
  }
  clEnqueueWriteBuffer(cl_ctx_.queue(), dt_bias_expanded_, CL_TRUE, 0,
                       dtb_exp.size() * sizeof(nnopt_storage_t), dtb_exp.data(), 0, nullptr, nullptr);

  // Expand A: A_expanded_ is ALWAYS fp32.
  // Reference: modeling_mamba.py (MambaMixer.slow_forward)
  //   A = -torch.exp(self.A_log.float())
  // Under fp16 storage, A_log can represent values that overflow half.
  // Keeping A in fp32 avoids -inf in A which then poisons selective_scan.
  const size_t A_numel = weights_.get_num_elements(prefix_ + ".A_log");
  std::vector<nnopt_storage_t> a_in(A_numel);
  clEnqueueReadBuffer(cl_ctx_.queue(), A_log_, CL_TRUE, 0, A_numel * sizeof(nnopt_storage_t), a_in.data(), 0, nullptr,
                      nullptr);

  std::vector<float> a_exp_f32((size_t)d_inner_ * (size_t)d_state_);

  auto to_f32 = [](nnopt_storage_t v) -> float {
#ifdef NNOPT_USE_FP16
    return nnopt_f16_to_f32((uint16_t)v);
#else
    return (float)v;
#endif
  };

  if (A_numel == (size_t)d_inner_ * (size_t)d_state_) {
    for (size_t i = 0; i < a_exp_f32.size(); ++i) {
      float al = to_f32(a_in[i]);
      a_exp_f32[i] = -expf(al);
    }
  } else if (A_numel == (size_t)d_inner_) {
    for (int c = 0; c < d_inner_; ++c) {
      float al = to_f32(a_in[(size_t)c]);
      float v = -expf(al);
      for (int s = 0; s < d_state_; ++s) a_exp_f32[(size_t)c * (size_t)d_state_ + (size_t)s] = v;
    }
  } else {
    NNOPT_ERROR_FMT("Ssm: unexpected A_log numel=%zu", A_numel);
    return false;
  }

  if (A_expanded_) {
    clReleaseMemObject(A_expanded_);
    A_expanded_ = nullptr;
  }
  A_expanded_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, a_exp_f32.size() * sizeof(float), nullptr, &err);
  if (err != CL_SUCCESS) return false;
  clEnqueueWriteBuffer(cl_ctx_.queue(), A_expanded_, CL_TRUE, 0, a_exp_f32.size() * sizeof(float), a_exp_f32.data(), 0,
                       nullptr, nullptr);

  // Expand D to [d_inner]
  const size_t D_numel = weights_.get_num_elements(prefix_ + ".D");
  std::vector<nnopt_storage_t> d_in(D_numel);
  clEnqueueReadBuffer(cl_ctx_.queue(), D_, CL_TRUE, 0, D_numel * sizeof(nnopt_storage_t), d_in.data(), 0, nullptr,
                      nullptr);
  std::vector<nnopt_storage_t> d_exp((size_t)d_inner_);
  if (D_numel == 1) {
    for (int i = 0; i < d_inner_; ++i) d_exp[i] = d_in[0];
  } else if ((int)D_numel == d_inner_) {
    std::memcpy(d_exp.data(), d_in.data(), (size_t)d_inner_ * sizeof(nnopt_storage_t));
  } else {
    for (int i = 0; i < d_inner_; ++i) d_exp[i] = d_in[i % D_numel];
  }

  if (D_expanded_) {
    clReleaseMemObject(D_expanded_);
    D_expanded_ = nullptr;
  }
  D_expanded_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, d_exp.size() * sizeof(nnopt_storage_t), nullptr,
                               &err);
  if (err != CL_SUCCESS) return false;
  clEnqueueWriteBuffer(cl_ctx_.queue(), D_expanded_, CL_TRUE, 0, d_exp.size() * sizeof(nnopt_storage_t), d_exp.data(),
                       0, nullptr, nullptr);

  return true;
}

void Ssm::reset_state(cl_command_queue queue) {
  if (!conv_state_ || !ssm_state_) return;
  const size_t conv_bytes = (size_t)d_inner_ * (size_t)(d_conv_ - 1) * sizeof(float);
  const size_t ssm_bytes = (size_t)d_inner_ * (size_t)d_state_ * sizeof(float);
  std::vector<float> zeros_conv((size_t)d_inner_ * (size_t)(d_conv_ - 1), 0.0f);
  std::vector<float> zeros_ssm((size_t)d_inner_ * (size_t)d_state_, 0.0f);
  // SYNC-OK: blocking writes reset persistent recurrent state.
  clEnqueueWriteBuffer(queue, conv_state_, CL_TRUE, 0, conv_bytes, zeros_conv.data(), 0, nullptr, nullptr);
  clEnqueueWriteBuffer(queue, ssm_state_, CL_TRUE, 0, ssm_bytes, zeros_ssm.data(), 0, nullptr, nullptr);
}

cl_mem Ssm::forward(cl_command_queue queue, cl_mem input, int seq_len) {
  NNOPT_LAYER_FWD("ssm");

  cl_int err = CL_SUCCESS;
  (void)err;

  const int M = seq_len;

  // Lazily allocate persistent activation buffers (sized for ≥ M). Reused
  // across decode steps — eliminates ~11 alloc/free per layer per token.
  if (!ensure_activation_buffers_(M)) {
    NNOPT_ERROR_FMT("Ssm[%d]: activation buffer alloc failed (M=%d)", layer_idx_, M);
    return nullptr;
  }

  // NOTE: The SxS sub-op chain dumps are defined in (model metadata).
  // IMPORTANT: "block0_sub_ssm_hidden_states" is an internal-local in PyTorch (MambaMixer.slow_forward)
  // that corresponds to the *post in_proj* hidden_states (NOT the mixer input).
  // Reference: model_info/transformers_src/modeling_mamba.py MambaMixer.slow_forward

  // 1) in_proj: [M, d_model] -> [M, 2*d_inner]   — writes into persistent buf_xz_
  const int N_inproj = 2 * d_inner_;
  cl_mem xz = buf_xz_;
  if (!pytorch_linear(queue, M, N_inproj, d_model_, input, in_proj_, xz)) {
    return nullptr;
  }

  // PyTorch name mapping:
  // - projected_states: output of in_proj (before chunk)
  // - hidden_states: first chunk (x) before conv
  // - gate: second chunk (z) before silu
  if (layer_idx_ == 0) {
    NNOPT_LAYER_CHECK("block0_sub_ssm_projected_states", queue, xz, (size_t)M * (size_t)N_inproj);
  }

  // Split xz -> x,z   (persistent buf_x_, buf_z_)
  cl_mem x = buf_x_;
  cl_mem z = buf_z_;
  {
    // Cooperative dispatch — 1 WG per row, 64 threads cooperate over d_inner
    // via vec4 fp16. d_inner=1536 ⇒ 24 fp16 / thread = 6 vec4.
    int total_rows = M;
    clSetKernelArg(split_kernel_, 0, sizeof(cl_mem), &xz);
    clSetKernelArg(split_kernel_, 1, sizeof(cl_mem), &x);
    clSetKernelArg(split_kernel_, 2, sizeof(cl_mem), &z);
    clSetKernelArg(split_kernel_, 3, sizeof(int), &d_inner_);
    clSetKernelArg(split_kernel_, 4, sizeof(int), &total_rows);
    const size_t WG = 64;
    size_t gws[1] = {(size_t)total_rows * WG};
    size_t lws[1] = {WG};
    cl_event* evt = KernelProfiler::event_for("ssm_split_xz");
    clEnqueueNDRangeKernel(queue, split_kernel_, 1, nullptr, gws, lws, 0, nullptr, evt);
  }

  // 2) causal conv1d + update state + SiLU
  if (layer_idx_ == 0) {
    NNOPT_LAYER_CHECK("block0_sub_ssm_input_states", queue, x, (size_t)M * (size_t)d_inner_);
  }
  cl_mem x_conv = buf_x_conv_;
  if (layer_idx_ == 0) {
    NNOPT_LAYER_CHECK("block0_sub_ssm_gate", queue, z, (size_t)M * (size_t)d_inner_);
  }
  {
    // causal_conv1d(x, w, b, state(fp32), out, seq_len, channels, k)
    clSetKernelArg(conv_kernel_, 0, sizeof(cl_mem), &x);
    clSetKernelArg(conv_kernel_, 1, sizeof(cl_mem), &conv_w_);
    clSetKernelArg(conv_kernel_, 2, sizeof(cl_mem), &conv_b_);
    clSetKernelArg(conv_kernel_, 3, sizeof(cl_mem), &conv_state_);
    clSetKernelArg(conv_kernel_, 4, sizeof(cl_mem), &x_conv);
    clSetKernelArg(conv_kernel_, 5, sizeof(int), &M);
    clSetKernelArg(conv_kernel_, 6, sizeof(int), &d_inner_);
    clSetKernelArg(conv_kernel_, 7, sizeof(int), &d_conv_);
    size_t gws0[1] = {(size_t)M * (size_t)d_inner_};
    cl_event* evt0 = KernelProfiler::event_for("ssm_conv1d");
    clEnqueueNDRangeKernel(queue, conv_kernel_, 1, nullptr, gws0, nullptr, 0, nullptr, evt0);

    clSetKernelArg(update_conv_state_kernel_, 0, sizeof(cl_mem), &x);
    clSetKernelArg(update_conv_state_kernel_, 1, sizeof(cl_mem), &conv_state_);
    clSetKernelArg(update_conv_state_kernel_, 2, sizeof(int), &M);
    clSetKernelArg(update_conv_state_kernel_, 3, sizeof(int), &d_inner_);
    clSetKernelArg(update_conv_state_kernel_, 4, sizeof(int), &d_conv_);
    size_t gws1[1] = {(size_t)d_inner_};
    cl_event* evt1 = KernelProfiler::event_for("ssm_conv_state_update");
    clEnqueueNDRangeKernel(queue, update_conv_state_kernel_, 1, nullptr, gws1, nullptr, 0, nullptr, evt1);

    // SiLU is now FUSED into causal_conv1d's STORE. Saves a separate
    // silu_inplace launch per layer per token (24 launches/token at 1 layer
    // = ~5 µs each on GPU + 30 µs host overhead). Verify by running with
    // NNOPT_KERNEL_PROFILE=1 — ssm_silu_inplace should not appear.
  }
  if (layer_idx_ == 0) {
    // conv_state_ / ssm_state_ are fp32 recurrent states; element count is fp32 floats.
    NNOPT_LAYER_CHECK("block0_sub_ssm_conv_state", queue, conv_state_, (size_t)d_inner_ * (size_t)(d_conv_ - 1));
    NNOPT_LAYER_CHECK("block0_sub_ssm_ssm_state", queue, ssm_state_, (size_t)d_inner_ * (size_t)d_state_);
  }

  // 3) x_proj: [M, d_inner] -> [M, dt_rank + 2*d_state]   (persistent buf_xproj_)
  const int N_xproj = dt_rank_ + 2 * d_state_;
  cl_mem xproj = buf_xproj_;
  // (no dump here) "hidden_states" refers to mixer input, not x_conv.

  if (!pytorch_linear(queue, M, N_xproj, d_inner_, x_conv, x_proj_, xproj)) {
    return nullptr;
  }
  if (layer_idx_ == 0) {
    NNOPT_LAYER_CHECK("block0_sub_ssm_x_proj_out", queue, xproj, (size_t)M * (size_t)N_xproj);
  }

  // Split xproj -> dt_raw, B, C   (persistent buf_dt_raw_, buf_B_, buf_C_)
  if (layer_idx_ == 0) {
    NNOPT_LAYER_CHECK("block0_sub_ssm_ssm_parameters", queue, xproj, (size_t)M * (size_t)N_xproj);
  }
  cl_mem dt_raw = buf_dt_raw_;
  cl_mem B      = buf_B_;
  cl_mem C      = buf_C_;
  {
    // Cooperative dispatch — 1 WG=16 per row. Smaller WG matches tiny rows
    // (80 elems for Mamba-130M).
    int total_rows = M;
    clSetKernelArg(split_xproj_kernel_, 0, sizeof(cl_mem), &xproj);
    clSetKernelArg(split_xproj_kernel_, 1, sizeof(cl_mem), &dt_raw);
    clSetKernelArg(split_xproj_kernel_, 2, sizeof(cl_mem), &B);
    clSetKernelArg(split_xproj_kernel_, 3, sizeof(cl_mem), &C);
    clSetKernelArg(split_xproj_kernel_, 4, sizeof(int), &dt_rank_);
    clSetKernelArg(split_xproj_kernel_, 5, sizeof(int), &d_state_);
    clSetKernelArg(split_xproj_kernel_, 6, sizeof(int), &total_rows);
    const size_t WG_XPROJ = 16;
    size_t gws[1] = {(size_t)total_rows * WG_XPROJ};
    size_t lws[1] = {WG_XPROJ};
    cl_event* evt = KernelProfiler::event_for("ssm_split_xproj");
    clEnqueueNDRangeKernel(queue, split_xproj_kernel_, 1, nullptr, gws, lws, 0, nullptr, evt);
  }
  if (layer_idx_ == 0) {
    NNOPT_LAYER_CHECK("block0_sub_ssm_time_step", queue, dt_raw, (size_t)M * (size_t)dt_rank_);
    NNOPT_LAYER_CHECK("block0_sub_ssm_B", queue, B, (size_t)M * (size_t)d_state_);
    NNOPT_LAYER_CHECK("block0_sub_ssm_C", queue, C, (size_t)M * (size_t)d_state_);
  }

  // 4) dt_proj: [M, dt_rank] -> [M, d_inner] (raw, pre-bias, pre-softplus)
  cl_mem dt_raw_post = buf_dt_;
  if (!pytorch_linear(queue, M, d_inner_, dt_rank_, dt_raw, dt_proj_, dt_raw_post)) {
    return nullptr;
  }

  // 5) Fused selective_scan: absorbs bias_add(dt) + softplus(dt) on read AND
  // silu_mul(z, y) on write. Replaces FOUR launches (bias_add, softplus,
  // selective_scan, silu_mul) with ONE. Saves ~3 ms host overhead per token
  // at 24 layers.
  //
  // Falls back to the unfused chain if scan_fused_kernel_ failed to build
  // (older driver) — preserves correctness.
  cl_mem y = buf_y_;
  if (scan_fused_kernel_) {
    clSetKernelArg(scan_fused_kernel_, 0, sizeof(cl_mem), &x_conv);            // x (post-conv1d, post-silu fused)
    clSetKernelArg(scan_fused_kernel_, 1, sizeof(cl_mem), &dt_raw_post);       // dt_raw (post-dt_proj, pre-bias)
    clSetKernelArg(scan_fused_kernel_, 2, sizeof(cl_mem), &dt_bias_expanded_); // dt_bias
    clSetKernelArg(scan_fused_kernel_, 3, sizeof(cl_mem), &z);                 // z (pre-silu)
    clSetKernelArg(scan_fused_kernel_, 4, sizeof(cl_mem), &A_expanded_);
    clSetKernelArg(scan_fused_kernel_, 5, sizeof(cl_mem), &B);
    clSetKernelArg(scan_fused_kernel_, 6, sizeof(cl_mem), &C);
    clSetKernelArg(scan_fused_kernel_, 7, sizeof(cl_mem), &D_expanded_);
    clSetKernelArg(scan_fused_kernel_, 8, sizeof(cl_mem), &ssm_state_);
    clSetKernelArg(scan_fused_kernel_, 9, sizeof(cl_mem), &y);
    clSetKernelArg(scan_fused_kernel_, 10, sizeof(int), &M);
    clSetKernelArg(scan_fused_kernel_, 11, sizeof(int), &d_inner_);
    clSetKernelArg(scan_fused_kernel_, 12, sizeof(int), &d_state_);
    size_t gws[1] = {(size_t)d_inner_};
    cl_event* evt = KernelProfiler::event_for("ssm_selective_scan_fused");
    clEnqueueNDRangeKernel(queue, scan_fused_kernel_, 1, nullptr, gws, nullptr, 0, nullptr, evt);
  } else {
    // Fallback: original 4-launch chain. Same math, just slower host side.
    cl_mem dt = dt_raw_post;
    int rows = M, cols = d_inner_;
    clSetKernelArg(bias_add_kernel_, 0, sizeof(cl_mem), &dt);
    clSetKernelArg(bias_add_kernel_, 1, sizeof(cl_mem), &dt_bias_expanded_);
    clSetKernelArg(bias_add_kernel_, 2, sizeof(int), &rows);
    clSetKernelArg(bias_add_kernel_, 3, sizeof(int), &cols);
    size_t gws_ba[1] = {(size_t)rows * (size_t)cols};
    clEnqueueNDRangeKernel(queue, bias_add_kernel_, 1, nullptr, gws_ba, nullptr, 0, nullptr, nullptr);

    int total = M * d_inner_;
    clSetKernelArg(softplus_kernel_, 0, sizeof(cl_mem), &dt);
    clSetKernelArg(softplus_kernel_, 1, sizeof(int), &total);
    size_t gws_sp[1] = {(size_t)total};
    clEnqueueNDRangeKernel(queue, softplus_kernel_, 1, nullptr, gws_sp, nullptr, 0, nullptr, nullptr);

    clSetKernelArg(scan_kernel_, 0, sizeof(cl_mem), &x_conv);
    clSetKernelArg(scan_kernel_, 1, sizeof(cl_mem), &dt);
    clSetKernelArg(scan_kernel_, 2, sizeof(cl_mem), &A_expanded_);
    clSetKernelArg(scan_kernel_, 3, sizeof(cl_mem), &B);
    clSetKernelArg(scan_kernel_, 4, sizeof(cl_mem), &C);
    clSetKernelArg(scan_kernel_, 5, sizeof(cl_mem), &D_expanded_);
    clSetKernelArg(scan_kernel_, 6, sizeof(cl_mem), &ssm_state_);
    clSetKernelArg(scan_kernel_, 7, sizeof(cl_mem), &y);
    clSetKernelArg(scan_kernel_, 8, sizeof(int), &M);
    clSetKernelArg(scan_kernel_, 9, sizeof(int), &d_inner_);
    clSetKernelArg(scan_kernel_, 10, sizeof(int), &d_state_);
    size_t gws_s[1] = {(size_t)d_inner_};
    clEnqueueNDRangeKernel(queue, scan_kernel_, 1, nullptr, gws_s, nullptr, 0, nullptr, nullptr);

    int total2 = M * d_inner_;
    clSetKernelArg(silu_mul_kernel_, 0, sizeof(cl_mem), &z);
    clSetKernelArg(silu_mul_kernel_, 1, sizeof(cl_mem), &y);
    clSetKernelArg(silu_mul_kernel_, 2, sizeof(int), &total2);
    size_t gws_sm[1] = {(size_t)total2};
    clEnqueueNDRangeKernel(queue, silu_mul_kernel_, 1, nullptr, gws_sm, nullptr, 0, nullptr, nullptr);
  }

  if (layer_idx_ == 0) {
    NNOPT_LAYER_CHECK("block0_sub_ssm_scan_output", queue, y, (size_t)M * (size_t)d_inner_);
    NNOPT_LAYER_CHECK("block0_sub_ssm_contextualized_states", queue, y, (size_t)M * (size_t)d_inner_);
    NNOPT_LAYER_CHECK("block0_sub_ssm_result", queue, y, (size_t)M * (size_t)d_inner_);
  }

  // 7) out_proj: [M, d_inner] -> [M, d_model]   (persistent buf_out_)
  cl_mem out = buf_out_;
  if (!pytorch_linear(queue, M, d_model_, d_inner_, y, out_proj_, out)) {
    return nullptr;
  }

  // Caller (model.cpp) currently does clReleaseMemObject(out) after the
  // residual add. With persistent ownership we must NOT honor that release.
  // We retain the buffer so subsequent forward() calls find it intact.
  // Bump the refcount once per forward so the caller's release decrements
  // back to our owned reference.
  clRetainMemObject(out);

  NNOPT_LAYER_FWD_DONE("ssm");
  return out;
}

// Decode-only fused path: writes residual[h] += out_proj(...) directly via
// gemv_m1_k1536_no4_radd. Replaces:
//    Ssm::forward(...) returning mixer_out
//    element_add_inplace(hidden, mixer_out)
//    clReleaseMemObject(mixer_out)
// with a single forward call that mutates `residual` (= hidden) in place.
// Saves one element_add launch + alloc per layer per token. Only valid for
// seq_len==1 (decode); for prefill, fall back to forward() + element_add.
bool Ssm::forward_fused_decode(cl_command_queue queue, cl_mem input, cl_mem residual, int seq_len) {
  if (seq_len != 1 || !scan_fused_kernel_) return false;
  NNOPT_LAYER_FWD("ssm_fused_decode");

  cl_int err = CL_SUCCESS;
  (void)err;
  const int M = 1;

  if (!ensure_activation_buffers_(M)) {
    NNOPT_ERROR_FMT("Ssm[%d]: activation buffer alloc failed (M=%d)", layer_idx_, M);
    return false;
  }

  // 1) in_proj
  const int N_inproj = 2 * d_inner_;
  cl_mem xz = buf_xz_;
  if (!pytorch_linear(queue, M, N_inproj, d_model_, input, in_proj_, xz)) return false;

  // 2) split_xz
  cl_mem x = buf_x_, z = buf_z_;
  {
    int total_rows = M;
    clSetKernelArg(split_kernel_, 0, sizeof(cl_mem), &xz);
    clSetKernelArg(split_kernel_, 1, sizeof(cl_mem), &x);
    clSetKernelArg(split_kernel_, 2, sizeof(cl_mem), &z);
    clSetKernelArg(split_kernel_, 3, sizeof(int), &d_inner_);
    clSetKernelArg(split_kernel_, 4, sizeof(int), &total_rows);
    const size_t WG = 64;
    size_t gws[1] = {(size_t)total_rows * WG};
    size_t lws[1] = {WG};
    cl_event* evt = KernelProfiler::event_for("ssm_split_xz");
    clEnqueueNDRangeKernel(queue, split_kernel_, 1, nullptr, gws, lws, 0, nullptr, evt);
  }

  // 3) conv1d (with fused silu) + update_conv_state
  cl_mem x_conv = buf_x_conv_;
  {
    clSetKernelArg(conv_kernel_, 0, sizeof(cl_mem), &x);
    clSetKernelArg(conv_kernel_, 1, sizeof(cl_mem), &conv_w_);
    clSetKernelArg(conv_kernel_, 2, sizeof(cl_mem), &conv_b_);
    clSetKernelArg(conv_kernel_, 3, sizeof(cl_mem), &conv_state_);
    clSetKernelArg(conv_kernel_, 4, sizeof(cl_mem), &x_conv);
    clSetKernelArg(conv_kernel_, 5, sizeof(int), &M);
    clSetKernelArg(conv_kernel_, 6, sizeof(int), &d_inner_);
    clSetKernelArg(conv_kernel_, 7, sizeof(int), &d_conv_);
    size_t gws0[1] = {(size_t)M * (size_t)d_inner_};
    cl_event* evt0 = KernelProfiler::event_for("ssm_conv1d");
    clEnqueueNDRangeKernel(queue, conv_kernel_, 1, nullptr, gws0, nullptr, 0, nullptr, evt0);

    clSetKernelArg(update_conv_state_kernel_, 0, sizeof(cl_mem), &x);
    clSetKernelArg(update_conv_state_kernel_, 1, sizeof(cl_mem), &conv_state_);
    clSetKernelArg(update_conv_state_kernel_, 2, sizeof(int), &M);
    clSetKernelArg(update_conv_state_kernel_, 3, sizeof(int), &d_inner_);
    clSetKernelArg(update_conv_state_kernel_, 4, sizeof(int), &d_conv_);
    size_t gws1[1] = {(size_t)d_inner_};
    cl_event* evt1 = KernelProfiler::event_for("ssm_conv_state_update");
    clEnqueueNDRangeKernel(queue, update_conv_state_kernel_, 1, nullptr, gws1, nullptr, 0, nullptr, evt1);
  }

  // 4) x_proj + split_xproj
  const int N_xproj = dt_rank_ + 2 * d_state_;
  cl_mem xproj = buf_xproj_;
  if (!pytorch_linear(queue, M, N_xproj, d_inner_, x_conv, x_proj_, xproj)) return false;

  cl_mem dt_raw = buf_dt_raw_, B = buf_B_, C = buf_C_;
  {
    int total_rows = M;
    clSetKernelArg(split_xproj_kernel_, 0, sizeof(cl_mem), &xproj);
    clSetKernelArg(split_xproj_kernel_, 1, sizeof(cl_mem), &dt_raw);
    clSetKernelArg(split_xproj_kernel_, 2, sizeof(cl_mem), &B);
    clSetKernelArg(split_xproj_kernel_, 3, sizeof(cl_mem), &C);
    clSetKernelArg(split_xproj_kernel_, 4, sizeof(int), &dt_rank_);
    clSetKernelArg(split_xproj_kernel_, 5, sizeof(int), &d_state_);
    clSetKernelArg(split_xproj_kernel_, 6, sizeof(int), &total_rows);
    const size_t WG_XPROJ = 16;
    size_t gws[1] = {(size_t)total_rows * WG_XPROJ};
    size_t lws[1] = {WG_XPROJ};
    cl_event* evt = KernelProfiler::event_for("ssm_split_xproj");
    clEnqueueNDRangeKernel(queue, split_xproj_kernel_, 1, nullptr, gws, lws, 0, nullptr, evt);
  }

  // 5) dt_proj
  cl_mem dt_raw_post = buf_dt_;
  if (!pytorch_linear(queue, M, d_inner_, dt_rank_, dt_raw, dt_proj_, dt_raw_post)) return false;

  // 6) Fused selective_scan (softplus + scan + silu_mul)
  cl_mem y = buf_y_;
  {
    clSetKernelArg(scan_fused_kernel_, 0, sizeof(cl_mem), &x_conv);
    clSetKernelArg(scan_fused_kernel_, 1, sizeof(cl_mem), &dt_raw_post);
    clSetKernelArg(scan_fused_kernel_, 2, sizeof(cl_mem), &dt_bias_expanded_);
    clSetKernelArg(scan_fused_kernel_, 3, sizeof(cl_mem), &z);
    clSetKernelArg(scan_fused_kernel_, 4, sizeof(cl_mem), &A_expanded_);
    clSetKernelArg(scan_fused_kernel_, 5, sizeof(cl_mem), &B);
    clSetKernelArg(scan_fused_kernel_, 6, sizeof(cl_mem), &C);
    clSetKernelArg(scan_fused_kernel_, 7, sizeof(cl_mem), &D_expanded_);
    clSetKernelArg(scan_fused_kernel_, 8, sizeof(cl_mem), &ssm_state_);
    clSetKernelArg(scan_fused_kernel_, 9, sizeof(cl_mem), &y);
    clSetKernelArg(scan_fused_kernel_, 10, sizeof(int), &M);
    clSetKernelArg(scan_fused_kernel_, 11, sizeof(int), &d_inner_);
    clSetKernelArg(scan_fused_kernel_, 12, sizeof(int), &d_state_);
    size_t gws[1] = {(size_t)d_inner_};
    cl_event* evt = KernelProfiler::event_for("ssm_selective_scan_fused");
    clEnqueueNDRangeKernel(queue, scan_fused_kernel_, 1, nullptr, gws, nullptr, 0, nullptr, evt);
  }

  // 7) Fused out_proj + residual_add: hidden[h] = sum + hidden[h]
  if (!pytorch_linear_radd(queue, M, d_model_, d_inner_, y, out_proj_, residual)) {
    // Shape didn't qualify or kernel failed. Fall back: do the unfused
    // out_proj into buf_out_, then element_add_inplace at the caller. We
    // signal this by writing into buf_out_ AND returning true with no
    // residual mutation? — no, that would silently break correctness.
    // Better: return false so caller falls back to the full unfused path.
    NNOPT_ERROR_FMT("Ssm[%d]: pytorch_linear_radd failed (M=%d N=%d K=%d) — caller must fall back",
                    layer_idx_, M, d_model_, d_inner_);
    return false;
  }

  NNOPT_LAYER_FWD_DONE("ssm_fused_decode");
  return true;
}
