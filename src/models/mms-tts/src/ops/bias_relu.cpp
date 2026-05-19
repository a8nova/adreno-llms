// Reference: model_info/transformers_src/modeling_vits.py (ReLU activation usage)
// Minimal helper: apply optional bias per-channel then ReLU.

#include "opencl_context.h"
#include "debug_utils.h"
#include "utils.h"
#include "profiler.h"

#include <CL/cl.h>

namespace {
static cl_program g_prog = nullptr;
static cl_kernel g_kernel = nullptr;

static bool ensure_kernel(OpenCLContext& cl_ctx) {
  if (g_prog && g_kernel) return true;
  // NOTE: kernels/utils.cl is scaffold-owned and may not include bias_relu.
  // This op has its own kernel source in kernels/bias_relu.cl.
  g_prog = cl_ctx.build_program_from_file("kernels/bias_relu.cl");
  if (!g_prog) return false;
  cl_int err = CL_SUCCESS;
  g_kernel = clCreateKernel(g_prog, "bias_relu", &err);
  if (err != CL_SUCCESS || !g_kernel) {
    NNOPT_ERROR_FMT("op_bias_relu: clCreateKernel(bias_relu) failed (%d)", (int)err);
    return false;
  }
  return true;
}
}

extern "C" cl_mem op_bias_relu(OpenCLContext& cl_ctx,
                                cl_command_queue queue,
                                cl_mem x,
                                size_t n,
                                cl_mem bias_optional) {
  if (!queue || !x || n == 0) {
    NNOPT_ERROR("op_bias_relu: bad args");
    return nullptr;
  }
  if (!ensure_kernel(cl_ctx)) return nullptr;

  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx.context();
  cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !out) {
    NNOPT_ERROR_FMT("op_bias_relu: clCreateBuffer(out) failed (%d)", (int)err);
    return nullptr;
  }

  int has_bias = (bias_optional != nullptr) ? 1 : 0;
  if (!set_arg_checked(g_kernel, 0, sizeof(cl_mem), &x, "x")) { clReleaseMemObject(out); return nullptr; }
  if (!set_arg_checked(g_kernel, 1, sizeof(cl_mem), &bias_optional, "bias")) { clReleaseMemObject(out); return nullptr; }
  if (!set_arg_checked(g_kernel, 2, sizeof(cl_mem), &out, "out")) { clReleaseMemObject(out); return nullptr; }
  if (!set_arg_checked(g_kernel, 3, sizeof(int), &has_bias, "has_bias")) { clReleaseMemObject(out); return nullptr; }

  const int N = (int)n;
  const int C = 0;  // broadcast channels unknown in this generic op; caller can extend later.
  if (!set_arg_checked(g_kernel, 4, sizeof(int), &N, "N")) { clReleaseMemObject(out); return nullptr; }
  if (!set_arg_checked(g_kernel, 5, sizeof(int), &C, "C")) { clReleaseMemObject(out); return nullptr; }

  size_t gws = (size_t)N;
  err = clEnqueueNDRangeKernel(queue, g_kernel, 1, nullptr, &gws, nullptr, 0, nullptr,
                              KernelProfiler::event_for("op_bias_relu"));
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("op_bias_relu: enqueue failed (%d)", (int)err);
    clReleaseMemObject(out);
    return nullptr;
  }
  return out;
}
