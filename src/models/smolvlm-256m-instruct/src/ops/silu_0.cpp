// Reference: /Users/alazarshenkute/.nnopt/ref_venvs/env_stable/lib/python3.12/site-packages/torch/nn/modules/activation.py:358-397 SiLU.forward
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"

#include <CL/cl.h>
#include <cstddef>
#include <string>

namespace {
struct SiluState {
  bool initialized = false;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
};

SiluState& state() {
  static SiluState s;
  return s;
}

bool ensure_initialized(OpenCLContext& cl_ctx) {
  auto& s = state();
  if (s.initialized) return true;

  s.program = cl_ctx.build_program_from_file("kernels/silu.cl");  // PROGRAM-INIT-OK
  if (!s.program) {
    NNOPT_ERROR("SiLU: build_program_from_file(kernels/silu.cl) failed");
    return false;
  }
  cl_int err = CL_SUCCESS;
  s.kernel = clCreateKernel(s.program, "silu_forward", &err);
  if (err != CL_SUCCESS || !s.kernel) {
    NNOPT_ERROR_FMT("SiLU: clCreateKernel(silu_forward) failed (%d)", (int)err);
    return false;
  }

  s.initialized = true;
  return true;
}
}  // namespace

extern "C" cl_mem op_SiLU(OpenCLContext& cl_ctx,
                           Weights& /*weights*/,  // no weights
                           cl_command_queue queue,
                           cl_mem input,
                           int n_elements) {
  if (!ensure_initialized(cl_ctx)) return nullptr;
  if (!queue || !input) {
    NNOPT_ERROR("op_SiLU: null queue/input");
    return nullptr;
  }
  if (n_elements <= 0) {
    NNOPT_ERROR("op_SiLU: n_elements <= 0");
    return nullptr;
  }

  cl_int err = CL_SUCCESS;
  cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                              (size_t)n_elements * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !out) {
    NNOPT_ERROR_FMT("op_SiLU: clCreateBuffer failed (%d)", (int)err);
    return nullptr;
  }

  cl_kernel k = state().kernel;
  if (!set_arg_checked(k, 0, sizeof(cl_mem), &input, "x")) {
    clReleaseMemObject(out);
    return nullptr;
  }
  if (!set_arg_checked(k, 1, sizeof(cl_mem), &out, "y")) {
    clReleaseMemObject(out);
    return nullptr;
  }
  if (!set_arg_checked(k, 2, sizeof(int), &n_elements, "n_elements")) {
    clReleaseMemObject(out);
    return nullptr;
  }

  const size_t gws[1] = {(size_t)n_elements};
  err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("op_SiLU: clEnqueueNDRangeKernel failed (%d)", (int)err);
    clReleaseMemObject(out);
    return nullptr;
  }

  return out;
}
