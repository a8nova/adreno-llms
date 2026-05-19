// Reference: model_info/transformers_src/modeling_idefics3.py Idefics3Connector.pixel_shuffle
// Host wrapper for kernels/pixel_shuffle.cl — single-tile SmolVLM path.

#include "opencl_context.h"
#include "debug_utils.h"
#include "utils.h"
#include "forward_dispatch.h"

#include <CL/cl.h>

namespace {
struct State {
  bool initialized = false;
  cl_program prog = nullptr;
  cl_kernel k = nullptr;
};

State& state() {
  static State s;
  return s;
}

bool ensure_initialized(OpenCLContext& cl_ctx) {
  auto& s = state();
  if (s.initialized) return true;
  s.prog = cl_ctx.build_program_from_file("kernels/pixel_shuffle.cl");  // PROGRAM-INIT-OK
  if (!s.prog) {
    NNOPT_ERROR("op_PixelShuffle: failed to build kernels/pixel_shuffle.cl");
    return false;
  }
  cl_int err = CL_SUCCESS;
  s.k = clCreateKernel(s.prog, "pixel_shuffle", &err);
  if (err != CL_SUCCESS || !s.k) {
    NNOPT_ERROR_FMT("op_PixelShuffle: clCreateKernel failed (%d)", (int)err);
    return false;
  }
  s.initialized = true;
  return true;
}
}  // namespace

// Allocates and returns a fresh [OUT_GRID*OUT_GRID, IN_C*SCALE*SCALE] storage_t buffer.
// IN_H must equal IN_W and be divisible by SCALE.
extern "C" cl_mem op_PixelShuffle(OpenCLContext& cl_ctx,
                                  cl_command_queue queue,
                                  cl_mem in_buf,
                                  int IN_H,
                                  int IN_W,
                                  int IN_C,
                                  int SCALE) {
  if (!queue || !in_buf) {
    NNOPT_ERROR("op_PixelShuffle: null arg");
    return nullptr;
  }
  if (IN_H <= 0 || IN_W <= 0 || IN_C <= 0 || SCALE <= 0) {
    NNOPT_ERROR_FMT("op_PixelShuffle: invalid dims H=%d W=%d C=%d S=%d", IN_H, IN_W, IN_C, SCALE);
    return nullptr;
  }
  if ((IN_H % SCALE) != 0 || (IN_W % SCALE) != 0) {
    NNOPT_ERROR_FMT("op_PixelShuffle: H/W not divisible by SCALE: %d/%d/%d", IN_H, IN_W, SCALE);
    return nullptr;
  }
  if (!ensure_initialized(cl_ctx)) return nullptr;

  const int OUT_GRID = IN_H / SCALE;
  const int OUT_ROWS = OUT_GRID * OUT_GRID;
  const int OUT_COLS = IN_C * SCALE * SCALE;

  cl_int err = CL_SUCCESS;
  const size_t bytes = (size_t)OUT_ROWS * (size_t)OUT_COLS * sizeof(nnopt_storage_t);
  cl_mem out_buf = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
  if (err != CL_SUCCESS || !out_buf) {
    NNOPT_ERROR_FMT("op_PixelShuffle: clCreateBuffer(out) failed (%d)", (int)err);
    return nullptr;
  }

  auto& s = state();
  if (!set_arg_checked(s.k, 0, sizeof(cl_mem), &in_buf,  "in_buf"))   { clReleaseMemObject(out_buf); return nullptr; }
  if (!set_arg_checked(s.k, 1, sizeof(cl_mem), &out_buf, "out_buf"))  { clReleaseMemObject(out_buf); return nullptr; }
  if (!set_arg_checked(s.k, 2, sizeof(int),    &IN_H,    "IN_H"))     { clReleaseMemObject(out_buf); return nullptr; }
  if (!set_arg_checked(s.k, 3, sizeof(int),    &IN_W,    "IN_W"))     { clReleaseMemObject(out_buf); return nullptr; }
  if (!set_arg_checked(s.k, 4, sizeof(int),    &IN_C,    "IN_C"))     { clReleaseMemObject(out_buf); return nullptr; }
  if (!set_arg_checked(s.k, 5, sizeof(int),    &SCALE,   "SCALE"))    { clReleaseMemObject(out_buf); return nullptr; }
  if (!set_arg_checked(s.k, 6, sizeof(int),    &OUT_GRID,"OUT_GRID")) { clReleaseMemObject(out_buf); return nullptr; }
  if (!set_arg_checked(s.k, 7, sizeof(int),    &OUT_COLS,"OUT_COLS")) { clReleaseMemObject(out_buf); return nullptr; }

  const size_t gws[2] = {(size_t)OUT_ROWS, (size_t)OUT_COLS};
  err = clEnqueueNDRangeKernel(queue, s.k, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("op_PixelShuffle: dispatch failed (%d)", (int)err);
    clReleaseMemObject(out_buf);
    return nullptr;
  }
  return out_buf;
}
