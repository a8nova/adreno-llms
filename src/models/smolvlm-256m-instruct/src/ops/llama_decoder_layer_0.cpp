// Reference: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/modeling_llama.py LlamaDecoderLayer.forward
//
// PyTorch (excerpt):
//   residual = hidden_states
//   hidden_states = self.input_layernorm(hidden_states)
//   hidden_states, _ = self.self_attn(...)
//   hidden_states = residual + hidden_states
//   residual = hidden_states
//   hidden_states = self.post_attention_layernorm(hidden_states)
//   hidden_states = self.mlp(hidden_states)
//   hidden_states = residual + hidden_states

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "forward_dispatch.h"  // op_LlamaRMSNorm / op_LlamaSdpaAttention / op_LlamaMLP declarations
#include "profile.h"

#include <CL/cl.h>
#include <string>

namespace {
struct DecoderLayerState {
  bool initialized = false;
  cl_program program = nullptr;
  cl_kernel k_add_residual = nullptr;
};

DecoderLayerState& state() {
  static DecoderLayerState s;
  return s;
}

bool ensure_initialized(OpenCLContext& cl_ctx) {
  auto& s = state();
  if (s.initialized) return true;

  s.program = cl_ctx.build_program_from_file("kernels/decoder_layer.cl");  // PROGRAM-INIT-OK
  if (!s.program) {
    NNOPT_ERROR("op_LlamaDecoderLayer: failed to build kernels/decoder_layer.cl");
    return false;
  }
  cl_int err = CL_SUCCESS;
  s.k_add_residual = clCreateKernel(s.program, "add_residual_inplace", &err);
  if (err != CL_SUCCESS || !s.k_add_residual) {
    NNOPT_ERROR_FMT("op_LlamaDecoderLayer: clCreateKernel(add_residual_inplace) failed (%d)", (int)err);
    return false;
  }
  s.initialized = true;
  return true;
}

bool add_residual_inplace(cl_command_queue queue, cl_kernel k, cl_mem x_inout, cl_mem res, int n) {
  if (!set_arg_checked(k, 0, sizeof(cl_mem), &x_inout, "x")) return false;
  if (!set_arg_checked(k, 1, sizeof(cl_mem), &res, "res")) return false;
  if (!set_arg_checked(k, 2, sizeof(int), &n, "n")) return false;
  const size_t gws[1] = {(size_t)n};
  const cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("op_LlamaDecoderLayer: add_residual_inplace dispatch failed (%d)", (int)err);
    return false;
  }
  return true;
}

}  // namespace

extern "C" cl_mem op_LlamaDecoderLayer(OpenCLContext& cl_ctx,
                                       Weights& weights,
                                       cl_command_queue queue,
                                       cl_mem hidden_states,  // [rows, hidden]
                                       int rows,
                                       int hidden_size,
                                       int intermediate_size,
                                       int num_q_heads,
                                       int num_kv_heads,
                                       int head_dim,
                                       int start_pos,
                                       // norms
                                       const char* in_norm_w,
                                       const char* post_norm_w,
                                       float rms_eps,
                                       // attn
                                       const char* q_w,
                                       const char* k_w,
                                       const char* v_w,
                                       const char* o_w,
                                       // mlp
                                       const char* gate_w,
                                       const char* up_w,
                                       const char* down_w) {
  if (!ensure_initialized(cl_ctx)) return nullptr;
  if (!queue || !hidden_states) {
    NNOPT_ERROR("op_LlamaDecoderLayer: null queue/hidden_states");
    return nullptr;
  }

  const int n = rows * hidden_size;

  // residual1 = hidden_states (no copy — `hidden_states` is preserved through
  // RMSNorm + Attention; both allocate fresh output buffers and do not
  // mutate their inputs).

  // input_layernorm
  NNOPT_PROFILE_BEGIN(queue, "22_rmsnorm_in");
  cl_mem norm1 = op_LlamaRMSNorm(cl_ctx, weights, queue, hidden_states, rows, hidden_size, rms_eps, in_norm_w);
  NNOPT_PROFILE_END(queue, "22_rmsnorm_in");
  if (!norm1) return nullptr;

  // self_attn
  NNOPT_PROFILE_BEGIN(queue, "23_self_attn");
  cl_mem attn_out = op_LlamaSdpaAttention(cl_ctx,
                                         weights,
                                         queue,
                                         norm1,
                                         rows,
                                         hidden_size,
                                         num_q_heads,
                                         num_kv_heads,
                                         head_dim,
                                         start_pos,
                                         q_w,
                                         k_w,
                                         v_w,
                                         o_w);
  NNOPT_PROFILE_END(queue, "23_self_attn");
  clReleaseMemObject(norm1);
  if (!attn_out) return nullptr;

  // Fused: attn_out += hidden_states  AND  norm2 = rmsnorm(attn_out + hidden_states).
  // attn_out now holds (residual1 + attn_out), which we reuse as residual2.
  NNOPT_PROFILE_BEGIN(queue, "25_rmsnorm_post");
  cl_mem norm2 = op_LlamaRMSNormWithResidual(cl_ctx, weights, queue,
                                             attn_out, hidden_states,
                                             rows, hidden_size, rms_eps, post_norm_w);
  NNOPT_PROFILE_END(queue, "25_rmsnorm_post");
  if (!norm2) {
    clReleaseMemObject(attn_out);
    return nullptr;
  }

  // mlp — on decode (rows==1, fp16), fuse the post-MLP residual add into the
  // down_proj GEMV write so we skip the separate add_residual dispatch.
  // residual2 (= attn_out, which already holds residual1 + attn) is passed
  // directly into the fused kernel.
  NNOPT_PROFILE_BEGIN(queue, "26_mlp");
#ifdef NNOPT_USE_FP16
  const bool use_fused_residual = (rows == 1);
#else
  const bool use_fused_residual = false;
#endif
  cl_mem mlp_out = nullptr;
  if (use_fused_residual) {
    mlp_out = op_LlamaMLP_with_residual(cl_ctx, weights, queue, norm2, attn_out,
                                        rows, hidden_size, intermediate_size,
                                        gate_w, up_w, down_w);
  } else {
    mlp_out = op_LlamaMLP(cl_ctx, weights, queue, norm2, rows, hidden_size,
                          intermediate_size, gate_w, up_w, down_w);
  }
  NNOPT_PROFILE_END(queue, "26_mlp");
  clReleaseMemObject(norm2);
  if (!mlp_out) {
    clReleaseMemObject(attn_out);
    return nullptr;
  }

  if (!use_fused_residual) {
    // hidden_states = residual2 + mlp_out (inplace on mlp_out, using attn_out as residual2)
    NNOPT_PROFILE_BEGIN(queue, "24_residual_add");
    if (!add_residual_inplace(queue, state().k_add_residual, mlp_out, attn_out, n)) {
      NNOPT_PROFILE_END(queue, "24_residual_add");
      clReleaseMemObject(mlp_out);
      clReleaseMemObject(attn_out);
      return nullptr;
    }
    NNOPT_PROFILE_END(queue, "24_residual_add");
  }
  clReleaseMemObject(attn_out);

  return mlp_out;
}
