// Reference: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/modeling_llama.py LlamaMLP.forward
//
// PyTorch:
//   down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
//   return down_proj
//
// This op implements SwiGLU MLP for SmolVLM text decoder (Llama-family).

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "profile.h"

#include <CL/cl.h>
#include <cstddef>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
// Per-layer persistent gate buffer for decode (seq_q==1). Sized to
// intermediate_size · 2B per layer. Indexed by layer index parsed from gate_w.
struct MlpDecodeScratch {
  cl_mem gate = nullptr;
  int intermediate = 0;
};
static std::vector<MlpDecodeScratch>& mlp_decode_cache() {
  static std::vector<MlpDecodeScratch> v;
  return v;
}
static int extract_mlp_layer_idx(const char* key) {
  if (!key) return -1;
  const std::string s(key);
  const std::string anchor = "layers.";
  size_t pos = s.find(anchor);
  if (pos == std::string::npos) return -1;
  size_t start = pos + anchor.size();
  size_t end = start;
  while (end < s.size() && std::isdigit((unsigned char)s[end])) end++;
  if (end == start) return -1;
  return std::atoi(s.substr(start, end - start).c_str());
}
}  // namespace

namespace {
struct MlpState {
  bool initialized = false;
  cl_program program = nullptr;
  cl_kernel k_swiglu = nullptr;
};

MlpState& state() {
  static MlpState s;
  return s;
}

bool ensure_initialized(OpenCLContext& cl_ctx) {
  auto& s = state();
  if (s.initialized) return true;

  s.program = cl_ctx.build_program_from_file("kernels/llama_mlp.cl");  // PROGRAM-INIT-OK
  if (!s.program) {
    NNOPT_ERROR("LlamaMLP: failed to build kernels/llama_mlp.cl");
    return false;
  }

  cl_int err = CL_SUCCESS;
  s.k_swiglu = clCreateKernel(s.program, "swiglu", &err);
  if (err != CL_SUCCESS || !s.k_swiglu) {
    NNOPT_ERROR_FMT("LlamaMLP: clCreateKernel(swiglu) failed (%d)", (int)err);
    return false;
  }

  s.initialized = true;
  return true;
}
} // namespace

extern "C" cl_mem op_LlamaMLP(OpenCLContext& cl_ctx,
                               Weights& weights,
                               cl_command_queue queue,
                               cl_mem input, // [rows, hidden]
                               int rows,
                               int hidden_size,
                               int intermediate_size,
                               const char* gate_w,
                               const char* up_w,
                               const char* down_w) {
  if (!ensure_initialized(cl_ctx)) return nullptr;
  if (!queue || !input) {
    NNOPT_ERROR("op_LlamaMLP: null queue/input");
    return nullptr;
  }

  cl_mem W_gate = weights.get_buffer(std::string(gate_w));
  cl_mem W_up = weights.get_buffer(std::string(up_w));
  cl_mem W_down = weights.get_buffer(std::string(down_w));
  if (!W_gate || !W_up || !W_down) {
    NNOPT_ERROR("op_LlamaMLP: missing weight");
    return nullptr;
  }

  cl_int err = CL_SUCCESS;
  const size_t gate_elems = (size_t)rows * (size_t)intermediate_size;
  // Note: per-layer gate persistence tried and regressed slightly. Reverted.
  bool gate_is_persistent = false;
  cl_mem gate = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                               gate_elems * sizeof(nnopt_storage_t), nullptr, &err);
  if (!gate) { NNOPT_ERROR("op_LlamaMLP: alloc gate failed"); return nullptr; }
  cl_mem up = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                             gate_elems * sizeof(nnopt_storage_t), nullptr, &err);
  if (!up) {
    if (!gate_is_persistent) clReleaseMemObject(gate);
    NNOPT_ERROR("op_LlamaMLP: alloc up failed");
    return nullptr;
  }

#ifdef NNOPT_USE_FP16
  // M=1 decode fast path: one fused kernel for gate+up GEMV + SwiGLU.
  // Eliminates: 1 GEMV dispatch + 1 swiglu dispatch + the entire `up` buffer.
  if (rows == 1) {
    clReleaseMemObject(up);
    NNOPT_PROFILE_BEGIN(queue, "26a_mlp_gate_up");
    const bool ok = gemv_m1_swiglu_fp16_dispatch(queue, intermediate_size, hidden_size,
                                                 input, W_gate, W_up, gate);
    NNOPT_PROFILE_END(queue, "26a_mlp_gate_up");
    if (!ok) {
      NNOPT_ERROR("op_LlamaMLP: fused gate_up_swiglu failed");
      if (!gate_is_persistent) clReleaseMemObject(gate);
      return nullptr;
    }
  } else
#endif
  {
  NNOPT_PROFILE_BEGIN(queue, "26a_mlp_gate_up");
  if (!pytorch_linear(queue, rows, intermediate_size, hidden_size, input, W_gate, gate)) {
    NNOPT_PROFILE_END(queue, "26a_mlp_gate_up");
    NNOPT_ERROR("op_LlamaMLP: gate_proj GEMM failed");
    if (!gate_is_persistent) clReleaseMemObject(gate);
    clReleaseMemObject(up);
    return nullptr;
  }
  if (!pytorch_linear(queue, rows, intermediate_size, hidden_size, input, W_up, up)) {
    NNOPT_PROFILE_END(queue, "26a_mlp_gate_up");
    NNOPT_ERROR("op_LlamaMLP: up_proj GEMM failed");
    if (!gate_is_persistent) clReleaseMemObject(gate);
    clReleaseMemObject(up);
    return nullptr;
  }
  NNOPT_PROFILE_END(queue, "26a_mlp_gate_up");

  // swiglu: gate = silu(gate) * up (in-place into gate)
  NNOPT_PROFILE_BEGIN(queue, "26b_swiglu");
  {
    cl_kernel k = state().k_swiglu;
    const int total = (int)gate_elems;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &gate, "gate")) return nullptr;
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &up, "up")) return nullptr;
    if (!set_arg_checked(k, 2, sizeof(int), &total, "total")) return nullptr;
    const size_t gws[1] = {(size_t)total};
    cl_int e2 = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, nullptr, 0, nullptr, nullptr);
    if (e2 != CL_SUCCESS) {
      NNOPT_PROFILE_END(queue, "26b_swiglu");
      NNOPT_ERROR_FMT("op_LlamaMLP: swiglu dispatch failed (%d)", (int)e2);
      if (!gate_is_persistent) clReleaseMemObject(gate);
      clReleaseMemObject(up);
      return nullptr;
    }
  }
  NNOPT_PROFILE_END(queue, "26b_swiglu");

  clReleaseMemObject(up);
  }

  cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                              (size_t)rows * (size_t)hidden_size * sizeof(nnopt_storage_t), nullptr, &err);
  if (!out) {
    NNOPT_ERROR("op_LlamaMLP: alloc out failed");
    if (!gate_is_persistent) clReleaseMemObject(gate);
    return nullptr;
  }

  NNOPT_PROFILE_BEGIN(queue, "26c_mlp_down");
  if (!pytorch_linear(queue, rows, hidden_size, intermediate_size, gate, W_down, out)) {
    NNOPT_PROFILE_END(queue, "26c_mlp_down");
    NNOPT_ERROR("op_LlamaMLP: down_proj GEMM failed");
    clReleaseMemObject(out);
    if (!gate_is_persistent) clReleaseMemObject(gate);
    return nullptr;
  }
  NNOPT_PROFILE_END(queue, "26c_mlp_down");

  if (!gate_is_persistent) clReleaseMemObject(gate);
  return out;
}

extern "C" cl_mem op_LlamaMLP_with_residual(OpenCLContext& cl_ctx,
                                            Weights& weights,
                                            cl_command_queue queue,
                                            cl_mem input,
                                            cl_mem residual,
                                            int rows,
                                            int hidden_size,
                                            int intermediate_size,
                                            const char* gate_w,
                                            const char* up_w,
                                            const char* down_w) {
  if (!ensure_initialized(cl_ctx)) return nullptr;
  if (!queue || !input || !residual) {
    NNOPT_ERROR("op_LlamaMLP_with_residual: null queue/input/residual");
    return nullptr;
  }

#ifdef NNOPT_USE_FP16
  // Decode-only fast path: fuse the post-MLP add_residual into the down_proj
  // GEMV write — saves one dispatch per layer per token. Falls back to the
  // standard MLP + caller-side residual_add when rows>1 (prefill) since
  // CLBlast handles those at high enough M that the fusion gain is in the
  // noise.
  if (rows != 1) {
    cl_mem mlp_out = op_LlamaMLP(cl_ctx, weights, queue, input, rows, hidden_size,
                                 intermediate_size, gate_w, up_w, down_w);
    if (!mlp_out) return nullptr;
    // No fused residual on prefill — caller must add residual itself if needed.
    // Decoder layer doesn't actually call this on prefill (rows>1) — only on
    // decode (rows==1). Returning mlp_out and letting caller add residual keeps
    // the contract honest.
    (void)residual;
    return mlp_out;
  }

  cl_mem W_gate = weights.get_buffer(std::string(gate_w));
  cl_mem W_up = weights.get_buffer(std::string(up_w));
  cl_mem W_down = weights.get_buffer(std::string(down_w));
  if (!W_gate || !W_up || !W_down) {
    NNOPT_ERROR("op_LlamaMLP_with_residual: missing weight");
    return nullptr;
  }

  cl_int err = CL_SUCCESS;
  cl_mem gate = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                               (size_t)intermediate_size * sizeof(nnopt_storage_t),
                               nullptr, &err);
  if (!gate) {
    NNOPT_ERROR("op_LlamaMLP_with_residual: alloc gate failed");
    return nullptr;
  }

  // Opt #34: try image-backed weights for both gate+up (swiglu) and down_proj.
  // L1 texture-cache path. Falls back to buffer GEMV on image-creation failure.
  cl_context ctx_h = cl_ctx.context();
  cl_mem W_gate_img = get_or_create_weight_image(ctx_h, queue, W_gate,
                                                 intermediate_size, hidden_size);
  cl_mem W_up_img   = get_or_create_weight_image(ctx_h, queue, W_up,
                                                 intermediate_size, hidden_size);

  NNOPT_PROFILE_BEGIN(queue, "26a_mlp_gate_up");
  bool gate_up_ok = false;
  if (W_gate_img && W_up_img) {
    gate_up_ok = gemv_m1_image_swiglu_fp16_dispatch(queue, intermediate_size, hidden_size,
                                                    input, W_gate_img, W_up_img, gate);
  }
  if (!gate_up_ok) {
    gate_up_ok = gemv_m1_swiglu_fp16_dispatch(queue, intermediate_size, hidden_size,
                                              input, W_gate, W_up, gate);
  }
  NNOPT_PROFILE_END(queue, "26a_mlp_gate_up");
  if (!gate_up_ok) {
    NNOPT_ERROR("op_LlamaMLP_with_residual: fused gate_up_swiglu failed");
    clReleaseMemObject(gate);
    return nullptr;
  }

  cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                              (size_t)hidden_size * sizeof(nnopt_storage_t),
                              nullptr, &err);
  if (!out) {
    NNOPT_ERROR("op_LlamaMLP_with_residual: alloc out failed");
    clReleaseMemObject(gate);
    return nullptr;
  }

  cl_mem W_down_img = get_or_create_weight_image(ctx_h, queue, W_down,
                                                 hidden_size, intermediate_size);
  NNOPT_PROFILE_BEGIN(queue, "26c_mlp_down");
  bool ok = false;
  if (W_down_img) {
    ok = gemv_m1_image_residual_fp16_dispatch(queue, hidden_size, intermediate_size,
                                              gate, W_down_img, residual, out);
  }
  if (!ok) {
    ok = gemv_m1_residual_fp16_dispatch(queue, hidden_size, intermediate_size,
                                        gate, W_down, residual, out);
  }
  NNOPT_PROFILE_END(queue, "26c_mlp_down");
  clReleaseMemObject(gate);
  if (!ok) {
    NNOPT_ERROR("op_LlamaMLP_with_residual: fused down_proj+residual failed");
    clReleaseMemObject(out);
    return nullptr;
  }
  return out;
#else
  // Non-fp16: just do MLP + return; caller must add residual itself.
  (void)residual;
  return op_LlamaMLP(cl_ctx, weights, queue, input, rows, hidden_size,
                     intermediate_size, gate_w, up_w, down_w);
#endif
}

// R6.5: fused rmsnorm_post + residual_add + MLP. Decode-only (rows==1, fp16).
// Falls back to the conventional path (caller runs rmsnorm separately) when
// not eligible. The fused kernel folds the rmsnorm_post dispatch into gate_up.
extern "C" cl_mem op_LlamaMLP_with_residual_and_rmsnorm(OpenCLContext& cl_ctx,
                                                        Weights& weights,
                                                        cl_command_queue queue,
                                                        cl_mem attn_out_raw,
                                                        cl_mem hidden_states_residual,
                                                        int rows,
                                                        int hidden_size,
                                                        int intermediate_size,
                                                        const char* post_norm_w,
                                                        float rms_eps,
                                                        const char* gate_w,
                                                        const char* up_w,
                                                        const char* down_w) {
  if (!ensure_initialized(cl_ctx)) return nullptr;
  if (!queue || !attn_out_raw || !hidden_states_residual) {
    NNOPT_ERROR("op_LlamaMLP_with_residual_and_rmsnorm: null args");
    return nullptr;
  }

#ifdef NNOPT_USE_FP16
  if (rows != 1) {
    // Prefill: not in the fused path. Caller should run rmsnorm + MLP separately.
    NNOPT_ERROR("op_LlamaMLP_with_residual_and_rmsnorm called with rows>1; not supported");
    return nullptr;
  }

  cl_mem gamma_post = weights.get_buffer(std::string(post_norm_w));
  cl_mem W_gate = weights.get_buffer(std::string(gate_w));
  cl_mem W_up   = weights.get_buffer(std::string(up_w));
  cl_mem W_down = weights.get_buffer(std::string(down_w));
  if (!gamma_post || !W_gate || !W_up || !W_down) {
    NNOPT_ERROR("op_LlamaMLP_with_residual_and_rmsnorm: missing weight");
    return nullptr;
  }

  cl_context ctx_h = cl_ctx.context();
  cl_int err = CL_SUCCESS;

  // Side buffer for the residual sum (attn_out + hidden_states) that the
  // downstream fused down+residual consumes. Allocated per call (cheap on
  // Adreno; persistence was tried & flat in #18a).
  cl_mem sum_buf = clCreateBuffer(ctx_h, CL_MEM_READ_WRITE,
                                  (size_t)hidden_size * sizeof(nnopt_storage_t),
                                  nullptr, &err);
  if (!sum_buf) {
    NNOPT_ERROR("op_LlamaMLP_with_residual_and_rmsnorm: alloc sum_buf failed");
    return nullptr;
  }

  cl_mem gate = clCreateBuffer(ctx_h, CL_MEM_READ_WRITE,
                               (size_t)intermediate_size * sizeof(nnopt_storage_t),
                               nullptr, &err);
  if (!gate) {
    NNOPT_ERROR("op_LlamaMLP_with_residual_and_rmsnorm: alloc gate failed");
    clReleaseMemObject(sum_buf);
    return nullptr;
  }

  cl_mem W_gate_img = get_or_create_weight_image(ctx_h, queue, W_gate,
                                                 intermediate_size, hidden_size);
  cl_mem W_up_img   = get_or_create_weight_image(ctx_h, queue, W_up,
                                                 intermediate_size, hidden_size);

  NNOPT_PROFILE_BEGIN(queue, "26a_mlp_gate_up");
  bool gate_up_ok = false;
  if (W_gate_img && W_up_img) {
    gate_up_ok = gemv_m1_rmsnorm_residual_image_swiglu_fp16_dispatch(
        queue, intermediate_size, hidden_size, rms_eps,
        attn_out_raw, hidden_states_residual, gamma_post,
        W_gate_img, W_up_img, gate, sum_buf);
  }
  NNOPT_PROFILE_END(queue, "26a_mlp_gate_up");
  if (!gate_up_ok) {
    NNOPT_ERROR("op_LlamaMLP_with_residual_and_rmsnorm: fused rmsnorm+swiglu failed");
    clReleaseMemObject(gate);
    clReleaseMemObject(sum_buf);
    return nullptr;
  }

  cl_mem out = clCreateBuffer(ctx_h, CL_MEM_READ_WRITE,
                              (size_t)hidden_size * sizeof(nnopt_storage_t),
                              nullptr, &err);
  if (!out) {
    NNOPT_ERROR("op_LlamaMLP_with_residual_and_rmsnorm: alloc out failed");
    clReleaseMemObject(gate);
    clReleaseMemObject(sum_buf);
    return nullptr;
  }

  cl_mem W_down_img = get_or_create_weight_image(ctx_h, queue, W_down,
                                                 hidden_size, intermediate_size);
  NNOPT_PROFILE_BEGIN(queue, "26c_mlp_down");
  bool ok = false;
  if (W_down_img) {
    ok = gemv_m1_image_residual_fp16_dispatch(queue, hidden_size, intermediate_size,
                                              gate, W_down_img, sum_buf, out);
  }
  if (!ok) {
    ok = gemv_m1_residual_fp16_dispatch(queue, hidden_size, intermediate_size,
                                        gate, W_down, sum_buf, out);
  }
  NNOPT_PROFILE_END(queue, "26c_mlp_down");
  clReleaseMemObject(gate);
  clReleaseMemObject(sum_buf);
  if (!ok) {
    NNOPT_ERROR("op_LlamaMLP_with_residual_and_rmsnorm: down+residual failed");
    clReleaseMemObject(out);
    return nullptr;
  }
  return out;
#else
  (void)attn_out_raw; (void)hidden_states_residual; (void)post_norm_w; (void)rms_eps;
  (void)rows; (void)hidden_size; (void)intermediate_size;
  (void)gate_w; (void)up_w; (void)down_w;
  NNOPT_ERROR("op_LlamaMLP_with_residual_and_rmsnorm: fp32 build — not implemented");
  return nullptr;
#endif
}
