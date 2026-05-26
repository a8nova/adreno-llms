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

  // R6.4: on decode (rows==1, fp16) skip the standalone rmsnorm dispatch and
  // let the QKV image GEMV compute inv_rms inline. ~575 µs/layer saved.
#ifdef NNOPT_USE_FP16
  const bool use_fused_in_norm = (rows == 1);
#else
  const bool use_fused_in_norm = false;
#endif

  cl_mem norm1 = nullptr;
  if (!use_fused_in_norm) {
    NNOPT_PROFILE_BEGIN(queue, "22_rmsnorm_in");
    norm1 = op_LlamaRMSNorm(cl_ctx, weights, queue, hidden_states, rows, hidden_size, rms_eps, in_norm_w);
    NNOPT_PROFILE_END(queue, "22_rmsnorm_in");
    if (!norm1) return nullptr;
  }

  // self_attn — on decode we pass the RAW hidden_states + in_norm_w so the
  // QKV kernel applies rmsnorm inline; on prefill we pass the pre-computed
  // norm1 (fused path is gated to seq_q==1 anyway).
  NNOPT_PROFILE_BEGIN(queue, "23_self_attn");
  cl_mem attn_input = use_fused_in_norm ? hidden_states : norm1;
  const char* fused_in_norm_arg = use_fused_in_norm ? in_norm_w : nullptr;
  cl_mem attn_out = op_LlamaSdpaAttention(cl_ctx,
                                         weights,
                                         queue,
                                         attn_input,
                                         rows,
                                         hidden_size,
                                         num_q_heads,
                                         num_kv_heads,
                                         head_dim,
                                         start_pos,
                                         q_w,
                                         k_w,
                                         v_w,
                                         o_w,
                                         fused_in_norm_arg,
                                         rms_eps);
  NNOPT_PROFILE_END(queue, "23_self_attn");
  if (norm1) clReleaseMemObject(norm1);
  if (!attn_out) return nullptr;

  // R6.5 (reverted): the rmsnorm_post + residual_add + gate_up fusion saves
  // 1 dispatch per layer (~655 µs of rmsnorm_post) but regressed gate_up from
  // 1145 → 2651 µs because 1536 WGs each re-read (attn_out + hidden_states +
  // gamma) for the inline sum + scale computation — net +25 ms/token wall.
  // The fused kernel (`gemv_m1_rmsnorm_residual_image_swiglu_fp16`) and the
  // `op_LlamaMLP_with_residual_and_rmsnorm` entry point are kept for future
  // attempts (e.g. broadcasting scale via global mem instead of redundant
  // compute). For now, force false.
  const bool use_fused_post_norm = false;

  cl_mem norm2 = nullptr;
  if (!use_fused_post_norm) {
    NNOPT_PROFILE_BEGIN(queue, "25_rmsnorm_post");
    norm2 = op_LlamaRMSNormWithResidual(cl_ctx, weights, queue,
                                        attn_out, hidden_states,
                                        rows, hidden_size, rms_eps, post_norm_w);
    NNOPT_PROFILE_END(queue, "25_rmsnorm_post");
    if (!norm2) {
      clReleaseMemObject(attn_out);
      return nullptr;
    }
  }

  // mlp — on decode (rows==1, fp16), fuse the post-MLP residual add into the
  // down_proj GEMV write so we skip the separate add_residual dispatch.
  // R6.5 path: rmsnorm_post is also folded inside, attn_out is kept raw and
  // hidden_states is passed as the residual source.
  NNOPT_PROFILE_BEGIN(queue, "26_mlp");
#ifdef NNOPT_USE_FP16
  const bool use_fused_residual = (rows == 1);
#else
  const bool use_fused_residual = false;
#endif
  cl_mem mlp_out = nullptr;
  if (use_fused_post_norm) {
    mlp_out = op_LlamaMLP_with_residual_and_rmsnorm(cl_ctx, weights, queue,
                                                    attn_out, hidden_states,
                                                    rows, hidden_size, intermediate_size,
                                                    post_norm_w, rms_eps,
                                                    gate_w, up_w, down_w);
  } else if (use_fused_residual) {
    mlp_out = op_LlamaMLP_with_residual(cl_ctx, weights, queue, norm2, attn_out,
                                        rows, hidden_size, intermediate_size,
                                        gate_w, up_w, down_w);
  } else {
    mlp_out = op_LlamaMLP(cl_ctx, weights, queue, norm2, rows, hidden_size,
                          intermediate_size, gate_w, up_w, down_w);
  }
  NNOPT_PROFILE_END(queue, "26_mlp");
  if (norm2) clReleaseMemObject(norm2);
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
