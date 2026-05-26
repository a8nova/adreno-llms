// Reference: /Users/alazarshenkute/.nnopt/ref_venvs/env_stable/lib/python3.12/site-packages/transformers/models/idefics3/modeling_idefics3.py:401-447 Idefics3EncoderLayer.forward
// Implements:
//   residual = x
//   x = layer_norm1(x)
//   x = self_attn(x, attention_mask)
//   x = residual + x
//   residual = x
//   x = layer_norm2(x)
//   x = mlp(x)
//   x = residual + x
// Returns hidden_states (attn_weights ignored for inference).

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "forward_dispatch.h"
#include "profile.h"

#include <CL/cl.h>
#include <cstddef>
#include <string>
#include <algorithm>

extern "C" cl_mem op_Idefics3EncoderLayer(OpenCLContext& cl_ctx,
                                          Weights& weights,
                                          cl_command_queue queue,
                                          cl_mem hidden_states,
                                          cl_mem attention_mask_opt,  // [B,1,T,T] or null
                                          int B,
                                          int T,
                                          int C,
                                          int num_heads,
                                          float ln_eps,
                                          // layer_norm1
                                          const char* ln1_w,
                                          const char* ln1_b,
                                          // attention projections
                                          const char* q_w,
                                          const char* q_b,
                                          const char* k_w,
                                          const char* k_b,
                                          const char* v_w,
                                          const char* v_b,
                                          const char* o_w,
                                          const char* o_b,
                                          // layer_norm2
                                          const char* ln2_w,
                                          const char* ln2_b,
                                          // mlp
                                          int intermediate_size,
                                          const char* fc1_w,
                                          const char* fc1_b,
                                          const char* fc2_w,
                                          const char* fc2_b) {
  if (!queue || !hidden_states) {
    NNOPT_ERROR("op_Idefics3EncoderLayer: null arg");
    return nullptr;
  }

  const int rows = B * T;

  NNOPT_PROFILE_BEGIN(queue, "vis_norm1");
  cl_mem x_norm1 = op_LayerNorm(cl_ctx, weights, queue, hidden_states, rows, C, ln_eps, ln1_w, ln1_b);
  NNOPT_PROFILE_END(queue, "vis_norm1");
  if (!x_norm1) {
    NNOPT_ERROR("op_Idefics3EncoderLayer: layer_norm1 failed");
    return nullptr;
  }

  NNOPT_PROFILE_BEGIN(queue, "vis_attn");
  cl_mem attn_out = op_Idefics3VisionAttention(cl_ctx, weights, queue,
                                               x_norm1,
                                               attention_mask_opt,
                                               B, T, C, num_heads,
                                               q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b);
  NNOPT_PROFILE_END(queue, "vis_attn");
  clReleaseMemObject(x_norm1);
  if (!attn_out) {
    NNOPT_ERROR("op_Idefics3EncoderLayer: self_attn failed");
    return nullptr;
  }

  // residual add
  // NOTE: hidden_states is [rows, C] where rows=B*T. The residual add must use
  // element count = rows*C (not bytes).
  const size_t elems = (size_t)rows * (size_t)C;

  // element_add allocates new buffer.
  // Use utils_program from utils.cl (build once here).
  static cl_program utils_prog = nullptr;
  static bool utils_init = false;
  if (!utils_init) {
    utils_prog = cl_ctx.build_program_from_file("kernels/utils.cl");  // PROGRAM-INIT-OK
    if (!utils_prog) {
      NNOPT_ERROR("op_Idefics3EncoderLayer: failed to build kernels/utils.cl");
      clReleaseMemObject(attn_out);
      return nullptr;
    }
    utils_init = true;
  }

  // element_add expects n == number of elements, not number of bytes.
  // The vision encoder operates at hidden size = 1152 (vision_config.hidden_size),
  // while the text model is hidden size = 576. Passing C from the caller can be wrong
  // in the vision path; use the actual buffer sizes to derive n.
  size_t hs_bytes = 0;
  size_t attn_bytes = 0;
  (void)clGetMemObjectInfo(hidden_states, CL_MEM_SIZE, sizeof(hs_bytes), &hs_bytes, nullptr);
  (void)clGetMemObjectInfo(attn_out, CL_MEM_SIZE, sizeof(attn_bytes), &attn_bytes, nullptr);
  const size_t elem_bytes = sizeof(nnopt_storage_t);
  const size_t n_from_hs = (elem_bytes > 0) ? (hs_bytes / elem_bytes) : 0;
  const size_t n_from_attn = (elem_bytes > 0) ? (attn_bytes / elem_bytes) : 0;
  const size_t n_add = std::min(n_from_hs, n_from_attn);
  if (n_add == 0) {
    NNOPT_ERROR_FMT("op_Idefics3EncoderLayer: residual add1 has zero n (hs_bytes=%zu attn_bytes=%zu elem=%zu)",
                    hs_bytes, attn_bytes, elem_bytes);
    clReleaseMemObject(attn_out);
    return nullptr;
  }

  NNOPT_PROFILE_BEGIN(queue, "vis_res1");
  cl_mem x_res1 = element_add(queue, utils_prog, hidden_states, attn_out, n_add);
  NNOPT_PROFILE_END(queue, "vis_res1");
  clReleaseMemObject(attn_out);
  if (!x_res1) {
    NNOPT_ERROR("op_Idefics3EncoderLayer: residual add1 failed");
    return nullptr;
  }

  NNOPT_PROFILE_BEGIN(queue, "vis_norm2");
  cl_mem x_norm2 = op_LayerNorm(cl_ctx, weights, queue, x_res1, rows, C, ln_eps, ln2_w, ln2_b);
  NNOPT_PROFILE_END(queue, "vis_norm2");
  if (!x_norm2) {
    NNOPT_ERROR("op_Idefics3EncoderLayer: layer_norm2 failed");
    clReleaseMemObject(x_res1);
    return nullptr;
  }

  NNOPT_PROFILE_BEGIN(queue, "vis_mlp");
  cl_mem mlp_out = op_Idefics3VisionMLP(cl_ctx, weights, queue,
                                        x_norm2,
                                        rows,
                                        C,
                                        intermediate_size,
                                        fc1_w, fc1_b, fc2_w, fc2_b);
  NNOPT_PROFILE_END(queue, "vis_mlp");
  clReleaseMemObject(x_norm2);
  if (!mlp_out) {
    NNOPT_ERROR("op_Idefics3EncoderLayer: mlp failed");
    clReleaseMemObject(x_res1);
    return nullptr;
  }

  // residual add2
  // Same bug class as add1: the caller-provided C can be wrong in the vision path.
  // Derive n from actual buffer sizes to avoid shape mismatch.
  size_t res1_bytes = 0;
  size_t mlp_bytes = 0;
  (void)clGetMemObjectInfo(x_res1, CL_MEM_SIZE, sizeof(res1_bytes), &res1_bytes, nullptr);
  (void)clGetMemObjectInfo(mlp_out, CL_MEM_SIZE, sizeof(mlp_bytes), &mlp_bytes, nullptr);
  const size_t n_from_res1 = (elem_bytes > 0) ? (res1_bytes / elem_bytes) : 0;
  const size_t n_from_mlp = (elem_bytes > 0) ? (mlp_bytes / elem_bytes) : 0;
  const size_t n_add2 = std::min(n_from_res1, n_from_mlp);
  if (n_add2 == 0) {
    NNOPT_ERROR_FMT("op_Idefics3EncoderLayer: residual add2 has zero n (res1_bytes=%zu mlp_bytes=%zu elem=%zu)",
                    res1_bytes, mlp_bytes, elem_bytes);
    clReleaseMemObject(mlp_out);
    clReleaseMemObject(x_res1);
    return nullptr;
  }

  cl_mem out = element_add(queue, utils_prog, x_res1, mlp_out, n_add2);
  clReleaseMemObject(mlp_out);
  clReleaseMemObject(x_res1);
  if (!out) {
    NNOPT_ERROR("op_Idefics3EncoderLayer: residual add2 failed");
    return nullptr;
  }

  return out;
}
