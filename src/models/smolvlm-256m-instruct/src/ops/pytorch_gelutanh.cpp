// Reference: /Users/alazarshenkute/.nnopt/ref_venvs/env_stable/lib/python3.12/site-packages/transformers/activations.py:28-46 PytorchGELUTanh.forward
// Implements nn.functional.gelu(input, approximate="tanh") via kernels/pytorch_gelutanh.cl.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"

#include <CL/cl.h>
#include <cstddef>

namespace {

struct GeluState {
  bool initialized = false;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
};

GeluState& state() {
  static GeluState s;
  return s;
}

bool ensure_initialized(OpenCLContext& cl_ctx) {
  auto& s = state();
  if (s.initialized) return true;

  s.program = cl_ctx.build_program_from_file("kernels/pytorch_gelutanh.cl");  // PROGRAM-INIT-OK
  if (!s.program) {
    NNOPT_ERROR("op_PytorchGELUTanh: failed to build kernels/pytorch_gelutanh.cl");
    return false;
  }

  cl_int err = CL_SUCCESS;
  s.kernel = clCreateKernel(s.program, "gelu_tanh_forward", &err);
  if (err != CL_SUCCESS || !s.kernel) {
    NNOPT_ERROR_FMT("op_PytorchGELUTanh: clCreateKernel(gelu_tanh_forward) failed (%d)", (int)err);
    return false;
  }

  s.initialized = true;
  return true;
}

}  // namespace

extern "C" cl_mem op_PytorchGELUTanh(OpenCLContext& cl_ctx,
                                     Weights& weights,
                                     cl_command_queue queue,
                                     cl_mem input,
                                     int n_elements) {
  (void)weights;
  if (!ensure_initialized(cl_ctx)) return nullptr;

  if (!queue || !input) {
    NNOPT_ERROR("op_PytorchGELUTanh: null arg");
    return nullptr;
  }

  cl_int err = CL_SUCCESS;
  cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                              (size_t)n_elements * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !out) {
    NNOPT_ERROR_FMT("op_PytorchGELUTanh: clCreateBuffer(out) failed (%d)", (int)err);
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
  if (!set_arg_checked(k, 1, sizeof(cl_mem), &out, "out")) return cleanup();
  if (!set_arg_checked(k, 2, sizeof(int), &n_elements, "n_elements")) return cleanup();

  const size_t gws[1] = {(size_t)n_elements};
  err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("op_PytorchGELUTanh: dispatch failed (%d)", (int)err);
    return cleanup();
  }

  return out;
}
