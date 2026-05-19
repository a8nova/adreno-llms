// Reference: /Users/alazarshenkute/.nnopt/ref_venvs/env_stable/lib/python3.12/site-packages/torch/nn/modules/normalization.py:180-182 LayerNorm.forward
// Implements torch.nn.functional.layer_norm using kernels/layer_norm.cl::layernorm_forward.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"

#include <CL/cl.h>
#include <cstddef>
#include <string>

namespace {

struct LayerNormState {
  bool initialized = false;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
};

LayerNormState& state() {
  static LayerNormState s;
  return s;
}

bool ensure_initialized(OpenCLContext& cl_ctx) {
  auto& s = state();
  if (s.initialized) return true;

  s.program = cl_ctx.build_program_from_file("kernels/layer_norm.cl");  // PROGRAM-INIT-OK
  if (!s.program) {
    NNOPT_ERROR("op_LayerNorm: failed to build kernels/layer_norm.cl");
    return false;
  }

  cl_int err = CL_SUCCESS;
  s.kernel = clCreateKernel(s.program, "layernorm_forward", &err);
  if (err != CL_SUCCESS || !s.kernel) {
    NNOPT_ERROR_FMT("op_LayerNorm: clCreateKernel(layernorm_forward) failed (%d)", (int)err);
    return false;
  }

  s.initialized = true;
  return true;
}

}  // namespace

extern "C" cl_mem op_LayerNorm(OpenCLContext& cl_ctx,
                               Weights& weights,
                               cl_command_queue queue,
                               cl_mem input,
                               int rows,
                               int cols,
                               float eps,
                               const char* weight_key_weight,
                               const char* weight_key_bias) {
  if (!ensure_initialized(cl_ctx)) return nullptr;

  if (!queue || !input || !weight_key_weight || !weight_key_bias) {
    NNOPT_ERROR("op_LayerNorm: null arg");
    return nullptr;
  }

  cl_mem w = weights.get_buffer(std::string(weight_key_weight));
  cl_mem b = weights.get_buffer(std::string(weight_key_bias));
  if (!w || !b) {
    NNOPT_ERROR("op_LayerNorm: missing weight/bias buffer");
    return nullptr;
  }

  cl_int err = CL_SUCCESS;
  const size_t elems = (size_t)rows * (size_t)cols;
  cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                              elems * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !out) {
    NNOPT_ERROR_FMT("op_LayerNorm: clCreateBuffer(out) failed (%d)", (int)err);
    return nullptr;
  }

  auto cleanup = [&]() -> cl_mem {
    if (out) {
      clReleaseMemObject(out);
      out = nullptr;
    }
    return nullptr;
  };

  cl_kernel k = state().kernel;
  if (!set_arg_checked(k, 0, sizeof(cl_mem), &input, "x")) return cleanup();
  if (!set_arg_checked(k, 1, sizeof(cl_mem), &w, "weight")) return cleanup();
  if (!set_arg_checked(k, 2, sizeof(cl_mem), &b, "bias")) return cleanup();
  if (!set_arg_checked(k, 3, sizeof(cl_mem), &out, "out")) return cleanup();
  if (!set_arg_checked(k, 4, sizeof(int), &rows, "rows")) return cleanup();
  if (!set_arg_checked(k, 5, sizeof(int), &cols, "cols")) return cleanup();
  if (!set_arg_checked(k, 6, sizeof(float), &eps, "eps")) return cleanup();

  // One workgroup per row. WG_SIZE is baked into the kernel (reqd_work_group_size).
  const size_t lws[1] = {64};
  const size_t gws[1] = {(size_t)rows * lws[0]};
  err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, lws, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("op_LayerNorm: dispatch failed (%d)", (int)err);
    return cleanup();
  }

  return out;
}
