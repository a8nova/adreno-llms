// Reference: /Users/alazarshenkute/.nnopt/ref_venvs/env_stable/lib/python3.12/site-packages/torch/nn/modules/linear.py:89-90 Linear.forward
// Implements torch.nn.functional.linear using CLBlast GEMM via utils.h::pytorch_linear.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"

#include <CL/cl.h>
#include <cstddef>
#include <string>

extern "C" cl_mem op_Linear(OpenCLContext& cl_ctx,
                            Weights& weights,
                            cl_command_queue queue,
                            cl_mem input,
                            int rows,
                            int in_features,
                            int out_features,
                            const char* weight_key_w,
                            const char* weight_key_b_optional) {
  if (!queue || !input || !weight_key_w) {
    NNOPT_ERROR("op_Linear: null arg");
    return nullptr;
  }

  cl_mem W = weights.get_buffer(std::string(weight_key_w));
  if (!W) {
    NNOPT_ERROR_FMT("op_Linear: missing weight %s", weight_key_w);
    return nullptr;
  }

  cl_int err = CL_SUCCESS;
  const size_t out_elems = (size_t)rows * (size_t)out_features;
  cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                              out_elems * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !out) {
    NNOPT_ERROR_FMT("op_Linear: clCreateBuffer(out) failed (%d)", (int)err);
    return nullptr;
  }

  auto cleanup = [&]() -> cl_mem {
    if (out) {
      clReleaseMemObject(out);
      out = nullptr;
    }
    return nullptr;
  };

  // out[rows, out_features] = input[rows, in_features] @ W[out_features, in_features]^T
  if (!pytorch_linear(queue, /*M=*/rows, /*N=*/out_features, /*K=*/in_features, input, W, out)) {
    NNOPT_ERROR("op_Linear: pytorch_linear failed");
    return cleanup();
  }

  // Optional bias add.
  if (weight_key_b_optional && weight_key_b_optional[0] != '\0') {
    cl_mem B = weights.get_buffer(std::string(weight_key_b_optional), /*optional=*/true);
    if (B) {
      // Use the shared bias_add kernel (rows x cols).
      static cl_program bias_prog = nullptr;
      static cl_kernel bias_k = nullptr;
      static bool bias_init = false;
      if (!bias_init) {
        bias_prog = cl_ctx.build_program_from_file("kernels/bias_add.cl");  // PROGRAM-INIT-OK
        if (!bias_prog) {
          NNOPT_ERROR("op_Linear: failed to build kernels/bias_add.cl");
          return cleanup();
        }
        cl_int e2 = CL_SUCCESS;
        bias_k = clCreateKernel(bias_prog, "bias_add_2d", &e2);
        if (e2 != CL_SUCCESS || !bias_k) {
          NNOPT_ERROR_FMT("op_Linear: clCreateKernel(bias_add_2d) failed (%d)", (int)e2);
          return cleanup();
        }
        bias_init = true;
      }

      if (!set_arg_checked(bias_k, 0, sizeof(cl_mem), &out, "x")) return cleanup();
      if (!set_arg_checked(bias_k, 1, sizeof(cl_mem), &B, "bias")) return cleanup();
      if (!set_arg_checked(bias_k, 2, sizeof(int), &rows, "rows")) return cleanup();
      if (!set_arg_checked(bias_k, 3, sizeof(int), &out_features, "cols")) return cleanup();

      const size_t gws[2] = {(size_t)rows, (size_t)out_features};
      cl_int e3 = clEnqueueNDRangeKernel(queue, bias_k, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
      if (e3 != CL_SUCCESS) {
        NNOPT_ERROR_FMT("op_Linear: bias_add dispatch failed (%d)", (int)e3);
        return cleanup();
      }
    }
  }

  return out;
}
