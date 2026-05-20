// Reference: /Users/alazarshenkute/.nnopt/ref_venvs/env_stable/lib/python3.12/site-packages/torch/nn/modules/conv.py:439-460 Conv2d.forward
// Implements a minimal conv2d for inference: y = conv2d(x, weight, bias, stride, padding, dilation, groups).
// This port supports the specific usage in SmolVLM's vision patch embedding.
//
// NOTE: current implementation is generic for groups=1, dilation=1, padding=0,
// stride=(S,S), kernel=(K,K). It uses an im2col + GEMM path via CLBlast.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"

#include <CL/cl.h>
#include <clblast.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

cl_mem alloc_buf(OpenCLContext& cl_ctx, size_t elems, const char* name) {
  cl_int err = CL_SUCCESS;
  cl_mem b = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                            elems * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !b) {
    NNOPT_ERROR_FMT("conv2d: clCreateBuffer(%s) failed (%d)", name, (int)err);
    return nullptr;
  }
  return b;
}

}  // namespace

extern "C" cl_mem op_Conv2d(OpenCLContext& cl_ctx,
                             Weights& weights,
                             cl_command_queue queue,
                             cl_mem input,  // [N, Cin, Hin, Win]
                             int N,
                             int Cin,
                             int Hin,
                             int Win,
                             int Cout,
                             int Kh,
                             int Kw,
                             int stride_h,
                             int stride_w,
                             int pad_h,
                             int pad_w,
                             const char* weight_key_w,
                             const char* weight_key_b_optional) {
  if (!queue || !input || !weight_key_w) {
    NNOPT_ERROR("op_Conv2d: null queue/input/weight_key");
    return nullptr;
  }

  // This op is used for patch embedding (groups=1).
  const int groups = 1;
  const int dilation_h = 1;
  const int dilation_w = 1;

  if (groups != 1 || dilation_h != 1 || dilation_w != 1) {
    NNOPT_ERROR("op_Conv2d: only groups=1,dilation=1 supported");
    return nullptr;
  }

  const int Hout = (Hin + 2 * pad_h - dilation_h * (Kh - 1) - 1) / stride_h + 1;
  const int Wout = (Win + 2 * pad_w - dilation_w * (Kw - 1) - 1) / stride_w + 1;
  if (Hout <= 0 || Wout <= 0) {
    NNOPT_ERROR_FMT("op_Conv2d: invalid output spatial (%d,%d)", Hout, Wout);
    return nullptr;
  }

  cl_mem W = weights.get_buffer(std::string(weight_key_w));
  if (!W) {
    NNOPT_ERROR_FMT("op_Conv2d: missing weight %s", weight_key_w);
    return nullptr;
  }

  cl_mem B = nullptr;
  if (weight_key_b_optional && weight_key_b_optional[0] != '\0') {
    B = weights.get_buffer(std::string(weight_key_b_optional), /*optional=*/true);
  }

  // im2col: [N*Hout*Wout, Cin*Kh*Kw]
  const int M = N * Hout * Wout;
  const int K = Cin * Kh * Kw;
  const int Ncol = Cout;

  cl_mem col = alloc_buf(cl_ctx, (size_t)M * (size_t)K, "im2col");
  cl_mem out = alloc_buf(cl_ctx, (size_t)M * (size_t)Cout, "out");
  if (!col || !out) {
    if (col) clReleaseMemObject(col);
    if (out) clReleaseMemObject(out);
    return nullptr;
  }

  auto cleanup = [&]() -> cl_mem {
    if (col) {
      clReleaseMemObject(col);
      col = nullptr;
    }
    if (out) {
      clReleaseMemObject(out);
      out = nullptr;
    }
    return nullptr;
  };

  // Build and run an im2col kernel (simple scalar version).
  // PROGRAM-INIT-OK: cached static program.
  static cl_program prog = nullptr;
  static cl_kernel im2col_k = nullptr;
  static bool init = false;
  if (!init) {
    prog = cl_ctx.build_program_from_file("kernels/im2col.cl");
    if (!prog) {
      NNOPT_ERROR("op_Conv2d: failed to build kernels/im2col.cl");
      return cleanup();
    }
    cl_int err = CL_SUCCESS;
    im2col_k = clCreateKernel(prog, "im2col_nchw", &err);
    if (err != CL_SUCCESS || !im2col_k) {
      NNOPT_ERROR_FMT("op_Conv2d: clCreateKernel(im2col_nchw) failed (%d)", (int)err);
      return cleanup();
    }
    init = true;
  }

  if (!set_arg_checked(im2col_k, 0, sizeof(cl_mem), &input, "input")) return cleanup();
  if (!set_arg_checked(im2col_k, 1, sizeof(cl_mem), &col, "col")) return cleanup();
  if (!set_arg_checked(im2col_k, 2, sizeof(int), &N, "N")) return cleanup();
  if (!set_arg_checked(im2col_k, 3, sizeof(int), &Cin, "Cin")) return cleanup();
  if (!set_arg_checked(im2col_k, 4, sizeof(int), &Hin, "Hin")) return cleanup();
  if (!set_arg_checked(im2col_k, 5, sizeof(int), &Win, "Win")) return cleanup();
  if (!set_arg_checked(im2col_k, 6, sizeof(int), &Hout, "Hout")) return cleanup();
  if (!set_arg_checked(im2col_k, 7, sizeof(int), &Wout, "Wout")) return cleanup();
  if (!set_arg_checked(im2col_k, 8, sizeof(int), &Kh, "Kh")) return cleanup();
  if (!set_arg_checked(im2col_k, 9, sizeof(int), &Kw, "Kw")) return cleanup();
  if (!set_arg_checked(im2col_k, 10, sizeof(int), &stride_h, "stride_h")) return cleanup();
  if (!set_arg_checked(im2col_k, 11, sizeof(int), &stride_w, "stride_w")) return cleanup();
  if (!set_arg_checked(im2col_k, 12, sizeof(int), &pad_h, "pad_h")) return cleanup();
  if (!set_arg_checked(im2col_k, 13, sizeof(int), &pad_w, "pad_w")) return cleanup();

  const size_t gws[2] = {(size_t)M, (size_t)K};
  cl_int err = clEnqueueNDRangeKernel(queue, im2col_k, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("op_Conv2d: im2col dispatch failed (%d)", (int)err);
    return cleanup();
  }

  // GEMM: out[M, Cout] = col[M, K] @ W_col[Cout, K]^T
  // W is in PyTorch Conv2d layout: [Cout, Cin, Kh, Kw] contiguous.
  // We treat it as row-major [Cout, K].
  if (!pytorch_linear(queue, /*M=*/M, /*N=*/Cout, /*K=*/K, col, W, out)) {
    NNOPT_ERROR("op_Conv2d: pytorch_linear GEMM failed");
    return cleanup();
  }

  // Optional bias add: out[m,c] += B[c]
  if (B) {
    static cl_program bias_prog = nullptr;
    static cl_kernel bias_k = nullptr;
    static bool bias_init = false;
    if (!bias_init) {
      bias_prog = cl_ctx.build_program_from_file("kernels/bias_add.cl");
      if (!bias_prog) {
        NNOPT_ERROR("op_Conv2d: failed to build kernels/bias_add.cl");
        return cleanup();
      }
      cl_int e2 = CL_SUCCESS;
      bias_k = clCreateKernel(bias_prog, "bias_add_2d", &e2);
      if (e2 != CL_SUCCESS || !bias_k) {
        NNOPT_ERROR_FMT("op_Conv2d: clCreateKernel(bias_add_2d) failed (%d)", (int)e2);
        return cleanup();
      }
      bias_init = true;
    }

    if (!set_arg_checked(bias_k, 0, sizeof(cl_mem), &out, "x")) return cleanup();
    if (!set_arg_checked(bias_k, 1, sizeof(cl_mem), &B, "bias")) return cleanup();
    if (!set_arg_checked(bias_k, 2, sizeof(int), &M, "rows")) return cleanup();
    if (!set_arg_checked(bias_k, 3, sizeof(int), &Cout, "cols")) return cleanup();

    const size_t gws2[2] = {(size_t)M, (size_t)Cout};
    cl_int e3 = clEnqueueNDRangeKernel(queue, bias_k, 2, nullptr, gws2, nullptr, 0, nullptr, nullptr);
    if (e3 != CL_SUCCESS) {
      NNOPT_ERROR_FMT("op_Conv2d: bias_add dispatch failed (%d)", (int)e3);
      return cleanup();
    }
  }

  // Reshape is a view in PyTorch. We return a flat buffer of size M*Cout;
  // caller is responsible for interpreting as [N, Cout, Hout, Wout].
  clReleaseMemObject(col);
  return out;
}
