// Reference: model_info/transformers_src/modeling_mamba2.py:Mamba2Mixer.forward / Mamba2Block.forward
#pragma once

#include "opencl_context.h"
#include "weights.h"

#include <CL/cl.h>
#include <string>
#include <vector>

class Ssm {
public:
  Ssm(OpenCLContext& cl_ctx, Weights& weights, const std::string& prefix, int layer_idx);
  ~Ssm();

  bool initialize();

  // Inject a prebuilt utils.cl program from the Model (so all layers share one build).
  // Must be called before initialize().
  void set_utils_program(cl_program utils_program) { utils_program_ = utils_program; }

  // Forward for a block mixer.
  // input:  [seq_len, d_model]
  // output: [seq_len, d_model]
  // start_pos is needed to maintain conv/ssm state across decode steps.
  //
  // OWNERSHIP: returns a BORROWED handle to a persistent buffer owned by this
  // Ssm instance. The caller MUST NOT release it. The buffer is overwritten on
  // the next forward() call against the same Ssm; per-layer instances each own
  // their own buffer so successive layers don't alias.
  //
  // FUSED-RESIDUAL FAST PATH (Lever 2):
  // When `hidden_residual` is non-null AND seq_len == 1, the out_proj GEMV is
  // dispatched via gemv_m1_k1536_no4_radd which writes
  //   hidden_residual[n] += dot(W[n,:], normed)
  // directly. No separate residual add kernel is needed; caller skips its
  // own element_add_inplace. The returned handle is `hidden_residual` itself
  // (same buffer the caller passed in). Falls back to the existing
  // buf_out_-returning path if the radd kernel isn't available.
  // For prefill (seq_len > 1) or hidden_residual==nullptr, behaves as before.
  cl_mem forward(cl_command_queue queue, cl_mem input, int seq_len, int start_pos,
                 cl_mem hidden_residual = nullptr);

  void reset_state(cl_command_queue queue);

private:
  OpenCLContext& cl_ctx_;
  Weights& weights_;
  const std::string prefix_;
  const int layer_idx_;

  // Derived dims from weight shapes.
  int d_model_ = 0;
  int d_inner_ = 0;
  int conv_ch_ = 0;
  int d_conv_ = 0;
  int n_v_heads_ = 0;
  int d_state_ = 0;
  int n_qk_heads_ = 0;
  int n_groups_ = 0;   // Mamba2: B/C projection groups (1 for state-spaces/mamba2-130m).
  int head_dim_ = 0;   // d_inner / n_v_heads. 64 for mamba2-130m.

  // The causal_conv1d kernel is depthwise and expects its "d_inner" argument to equal the
  // channel count of x. In this port we run it over xBC with conv_ch_ channels (not d_inner_).
  // Keep this explicit to avoid accidentally passing d_inner_ and reading past conv_state_.
  int conv_d_inner_ = 0;

  // Programs/kernels (scaffold kernels reused).
  cl_program conv_program_ = nullptr;
  cl_kernel conv_kernel_ = nullptr;
  cl_kernel conv_silu_kernel_ = nullptr;       // Lever C: fused silu+conv
  cl_kernel update_conv_state_kernel_ = nullptr;
  cl_kernel silu_inplace_kernel_ = nullptr;

  cl_program scan_program_ = nullptr;
  cl_kernel scan_kernel_ = nullptr;
  cl_kernel softplus_kernel_ = nullptr;
  cl_kernel silu_mul_kernel_ = nullptr;
  cl_kernel ssd_scan_kernel_ = nullptr;       // mamba2_ssd_scan (scalar+vec4 inner loop)
  cl_kernel ssd_scan_coop_kernel_ = nullptr;  // mamba2_ssd_scan_coop (WG=128, 1 state/thread, tree-reduce)
  cl_kernel ssd_scan_t_kernel_ = nullptr;     // Lever E: transposed h_state layout
  bool ssd_scan_use_coop_ = false;            // chosen at init: true iff state_size == 128
  bool ssd_scan_use_t_ = false;               // Lever E: opt-in via NNOPT_SSD_SCAN_T or default-on
  bool use_fp16_hstate_ = false;              // Lever I: fp16 ssm_state storage

  cl_program utils_program_ = nullptr;
  cl_kernel split_kernel_ = nullptr;

  // Mixer norm (RMSNorm-like): y = x * rsqrt(mean(x^2)+eps) * weight
  cl_program norm_program_ = nullptr;
  cl_kernel norm_kernel_ = nullptr;

  // Persistent recurrent state on device (fp32), per layer.
  cl_mem conv_state_ = nullptr; // float: [conv_ch, d_conv-1]
  cl_mem ssm_state_ = nullptr;  // float: [d_inner, d_state]

  // Persistent activation buffers, lazy-allocated by ensure_activation_buffers_.
  // Sized for the largest seq_len ever seen on this layer (typically the
  // prefill length); decode (seq_len=1) reuses without realloc. Skipping the
  // ~50 µs/buffer driver cost of clCreateBuffer/clReleaseMemObject across
  // 13 buffers × 24 layers × N tokens is the point of Step 5.
  int buf_capacity_rows_ = 0;
  bool use_subbuffer_splits_ = false;       // Lever F: skip split kernels at decode
  cl_mem buf_projected_states_ = nullptr;   // [rows, projection_size]
  cl_mem buf_gate_ = nullptr;               // [rows, d_inner]
  cl_mem buf_rest_after_gate_ = nullptr;    // [rows, projection_size - d_inner]
  cl_mem buf_hidden_states_B_C_ = nullptr;  // [rows, conv_ch]
  cl_mem buf_dt_raw_ = nullptr;             // [rows, n_v_heads]
  cl_mem buf_hbc_conv_ = nullptr;           // [rows, conv_ch]
  cl_mem buf_x_inner_ = nullptr;            // [rows, d_inner]
  cl_mem buf_bc_ = nullptr;                 // [rows, 2 * n_groups * state_size]
  cl_mem buf_B_ = nullptr;                  // [rows, n_groups * state_size]
  cl_mem buf_C_ = nullptr;                  // [rows, n_groups * state_size]
  cl_mem buf_y_scan_ = nullptr;             // [rows, d_inner]
  cl_mem buf_normed_ = nullptr;             // [rows, d_inner]
  cl_mem buf_out_ = nullptr;                // [rows, d_model]   ← returned (borrowed)

  // Lever F: decode-only subbuffer aliases (single-row views into the parent
  // projection / hbc buffers). Used when seq_len==1 so we can skip the split
  // kernels entirely. Allocated once per parent-grow.
  cl_mem buf_gate_dec_ = nullptr;
  cl_mem buf_hbc_dec_ = nullptr;
  cl_mem buf_dt_dec_ = nullptr;
  cl_mem buf_x_inner_dec_ = nullptr;
  cl_mem buf_bc_dec_ = nullptr;
  cl_mem buf_B_dec_ = nullptr;
  cl_mem buf_C_dec_ = nullptr;
  bool   buf_decode_subbufs_ready_ = false;

  // Weight buffers.
  cl_mem in_proj_w_ = nullptr;
  cl_mem conv_w_ = nullptr;
  cl_mem conv_b_ = nullptr;
  cl_mem dt_bias_ = nullptr;
  cl_mem a_log_ = nullptr;
  cl_mem d_skip_ = nullptr;
  cl_mem norm_w_ = nullptr;
  cl_mem out_proj_w_ = nullptr;

  bool derive_dims_from_weights();
  bool build_kernels();

  // Allocate (or grow) the 13 persistent activation buffers to fit `rows` work
  // rows. Returns false if any clCreateBuffer fails. Must be called once at the
  // top of forward() before any kernel dispatch.
  bool ensure_activation_buffers_(int rows);

  // Release all 13 persistent activation buffers. Used by ensure_ on grow and
  // by the destructor.
  void release_activation_buffers_();
};
