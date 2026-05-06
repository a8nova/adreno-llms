// Reference: model_info/transformers_src/modeling_mamba.py:~590-910 MambaMixer (slow_forward)
#pragma once

#include "opencl_context.h"
#include "weights.h"
#include <CL/cl.h>
#include <string>

// One Mamba (Mamba-1) mixer block: in_proj -> conv1d -> SiLU -> x_proj -> dt_proj(+softplus)
// -> selective_scan -> gate(y * silu(z)) -> out_proj.
class Ssm {
public:
  Ssm(OpenCLContext &cl_ctx, Weights &weights, const std::string &prefix, int layer_idx);
  ~Ssm();

  bool initialize();

  // input: [seq, d_model]
  // output: [seq, d_model]
  cl_mem forward(cl_command_queue queue, cl_mem input, int seq_len);

  // Fused-decode forward. `residual` is the pre-rmsnorm hidden buffer; this
  // function writes residual[h] += out_proj(...) directly via the
  // gemv_m1_k1536_no4_radd path AND skips bias_add/softplus/silu_mul/silu_inplace
  // (their math is folded into causal_conv1d, selective_scan_fused, and the
  // residual GEMV). Returns true on success. The caller MUST NOT call
  // element_add_inplace afterward — the residual_add has already been applied.
  // Falls back internally to the unfused forward() + caller-side residual_add
  // when seq_len != 1 or any fused-kernel build failed; in that case returns
  // false and the caller must do the unfused path.
  bool forward_fused_decode(cl_command_queue queue, cl_mem input, cl_mem residual, int seq_len);

  // Reset persistent recurrent state (conv + SSM).
  void reset_state(cl_command_queue queue);

private:
  OpenCLContext &cl_ctx_;
  Weights &weights_;
  std::string prefix_;
  int layer_idx_;

  // Weights (device buffers owned by Weights; do NOT clRelease)
  cl_mem in_proj_ = nullptr;
  cl_mem x_proj_ = nullptr;
  cl_mem dt_proj_ = nullptr;
  cl_mem out_proj_ = nullptr;
  cl_mem conv_w_ = nullptr;
  cl_mem conv_b_ = nullptr;  // optional
  cl_mem A_log_ = nullptr;
  cl_mem D_ = nullptr;
  cl_mem dt_bias_ = nullptr;

  // Expanded params (owned by this layer; MUST clRelease)
  cl_mem dt_bias_expanded_ = nullptr; // [d_inner]
  cl_mem A_expanded_ = nullptr;       // [d_inner, d_state]
  cl_mem D_expanded_ = nullptr;       // [d_inner]

  // Programs/kernels
  cl_program program_scan_ = nullptr;
  cl_kernel scan_kernel_ = nullptr;
  cl_kernel softplus_kernel_ = nullptr;
  cl_kernel silu_mul_kernel_ = nullptr;
  // Fused selective_scan: absorbs softplus(dt_raw + dt_bias) on read AND
  // silu(z)*y on write. Saves 3 launches per layer per token.
  cl_kernel scan_fused_kernel_ = nullptr;

  cl_program program_conv_ = nullptr;
  cl_kernel conv_kernel_ = nullptr;
  cl_kernel update_conv_state_kernel_ = nullptr;
  cl_kernel silu_inplace_kernel_ = nullptr;

  cl_program program_ssm_ = nullptr;
  cl_kernel split_kernel_ = nullptr;
  cl_kernel split_xproj_kernel_ = nullptr;
  cl_kernel bias_add_kernel_ = nullptr;

  // Persistent per-layer state on device
  cl_mem conv_state_ = nullptr; // [d_inner, d_conv-1]
  cl_mem ssm_state_ = nullptr;  // [d_inner, d_state]

  // Persistent activation buffers (decode M=1 hot path).
  // Allocated on first forward() with the given M; reused on subsequent calls.
  // Sized for M=1 by default; grown on demand if a longer prefill is seen.
  // Eliminates ~11 buffer allocs/frees per layer per token (~13 ms saved at
  // ~50 us per alloc, which is dominated by driver bookkeeping not memory
  // touch).
  cl_mem buf_xz_     = nullptr;
  cl_mem buf_x_      = nullptr;
  cl_mem buf_z_      = nullptr;
  cl_mem buf_x_conv_ = nullptr;
  cl_mem buf_xproj_  = nullptr;
  cl_mem buf_dt_raw_ = nullptr;
  cl_mem buf_B_      = nullptr;
  cl_mem buf_C_      = nullptr;
  cl_mem buf_dt_     = nullptr;
  cl_mem buf_y_      = nullptr;
  cl_mem buf_out_    = nullptr;
  int    buf_capacity_M_ = 0;  // current allocated M; reallocate if seq_len > this

  bool ensure_activation_buffers_(int M);

  // Cached dims
  int d_model_ = 0;
  int d_inner_ = 0;
  int d_state_ = 0;
  int d_conv_ = 0;
  int dt_rank_ = 0;

  bool prepare_expanded_params_();
};
