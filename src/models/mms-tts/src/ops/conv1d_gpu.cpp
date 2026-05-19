// Shared GPU conv1d implementation: im2col + CLBlast HGemm + bias broadcast.
//
// Step C parity with vocoder.cpp's conv1d_gemm_impl:
//   • 3D NDRange im2col_1d / leaky_im2col_1d (kills 4 div/mod per workitem — Adreno guide §8.12)
//   • Cached im2col workspace (Adreno guide §5.7.1: "Avoid creating or releasing
//     memory objects between the kernel calls.")
//   • Cached cl_kernel objects (no clCreateKernel per call)
//   • When has_bias && C_in >= C_out: fused im2col_1d_fused_bias kernel pre-fills
//     the HGemm output buffer with bias, HGemm runs with beta=1, and the trailing
//     add_bias_broadcast dispatch is SKIPPED entirely (Adreno guide §8.1: kernel
//     fusion when data traffic reduces).
//
// Why these matter for flow_inverse: ~12 conv calls per inverse-coupling stage
// × 4 stages = 48 conv calls. Each was 3 dispatches; now 2 (when C_in >= C_out)
// or still 3 but with cached im2col workspace (when C_in < C_out — conv_pre, in_layer).

#include "conv1d_gpu.h"
#include "debug_utils.h"
#include "utils.h"
#include "profiler.h"

#include <CL/cl.h>
#include <clblast.h>
#include <cstdio>
#include <cstdlib>

static bool set_arg_(cl_kernel k, cl_uint idx, size_t sz, const void* v) {
  cl_int e = clSetKernelArg(k, idx, sz, v);
  return e == CL_SUCCESS;
}

cl_mem conv1d_gpu(OpenCLContext& cl_ctx,
                  cl_command_queue queue,
                  cl_mem in,
                  cl_mem w,
                  cl_mem bias,
                  int C_in, int C_out, int L_in,
                  int K, int stride, int padding, int dilation,
                  bool has_bias,
                  const char* label) {
  const int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
  if (L_out <= 0) {
    NNOPT_ERROR_FMT("conv1d_gpu(%s): L_out=%d", label, L_out);
    return nullptr;
  }
  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx.context();

  // Cached im2col workspace — same pattern as vocoder.cpp's conv1d_gemm_impl
  // (Adreno guide §5.7.1 — avoid per-call clCreateBuffer).
  const size_t im2col_n = (size_t)C_in * (size_t)K * (size_t)L_out;
  static cl_mem g_im2col_ws = nullptr;
  static size_t g_im2col_ws_n = 0;
  if (im2col_n > g_im2col_ws_n) {
    if (g_im2col_ws) { clReleaseMemObject(g_im2col_ws); g_im2col_ws = nullptr; }
    g_im2col_ws = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 im2col_n * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !g_im2col_ws) {
      NNOPT_ERROR_FMT("conv1d_gpu(%s): im2col workspace (%d)", label, (int)err);
      g_im2col_ws_n = 0;
      return nullptr;
    }
    g_im2col_ws_n = im2col_n;
  }
  cl_mem im2col = g_im2col_ws;

  cl_program prog = cl_ctx.get_program("kernels/im2col.cl");
  if (!prog) prog = cl_ctx.build_program_from_file("kernels/im2col.cl");
  if (!prog) return nullptr;

  // Step C — fused im2col + bias-init path when conditions allow.
  // Requires has_bias AND C_in >= C_out so the bias-init pass (workitems with
  // k==0, ic<C_out) covers the full [C_out, L_out] grid exactly once.
  static int s_fused_bias = -1;
  if (s_fused_bias < 0) {
    const char* env = std::getenv("NNOPT_VOC_FUSED_BIAS");
    s_fused_bias = (env && env[0] == '0') ? 0 : 1;
  }
  const bool use_fused = s_fused_bias && has_bias && bias && (C_in >= C_out);

  const size_t n_out = (size_t)C_out * (size_t)L_out;
  cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                              n_out * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !out) {
    NNOPT_ERROR_FMT("conv1d_gpu(%s): out buffer (%d)", label, (int)err);
    return nullptr;
  }

  if (use_fused) {
    // ===== Fused path: im2col_1d_fused_bias + HGemm beta=1 (2 dispatches) =====
    static cl_kernel kim_fused = nullptr;
    if (!kim_fused) {
      kim_fused = clCreateKernel(prog, "im2col_1d_fused_bias", &err);
      if (err != CL_SUCCESS || !kim_fused) {
        NNOPT_ERROR_FMT("conv1d_gpu(%s): kernel im2col_1d_fused_bias (%d)", label, (int)err);
        clReleaseMemObject(out);
        return nullptr;
      }
    }
    // Dummy resid buffer (kernel skips read when has_resid==0, but driver wants non-NULL arg).
    static cl_mem g_dummy_resid = nullptr;
    if (!g_dummy_resid) {
      g_dummy_resid = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 16, nullptr, &err);
      if (err != CL_SUCCESS || !g_dummy_resid) {
        clReleaseMemObject(out);
        return nullptr;
      }
    }
    const int leaky_flag = 0;             // conv1d_gpu doesn't fuse leaky (flow/text encoder convs don't need it)
    const int has_bias_flag = 1;
    const int has_resid_flag = 0;
    bool ok = true;
    ok = ok && set_arg_(kim_fused,  0, sizeof(cl_mem), &in);
    ok = ok && set_arg_(kim_fused,  1, sizeof(cl_mem), &im2col);
    ok = ok && set_arg_(kim_fused,  2, sizeof(cl_mem), &out);
    ok = ok && set_arg_(kim_fused,  3, sizeof(cl_mem), &bias);
    ok = ok && set_arg_(kim_fused,  4, sizeof(cl_mem), &g_dummy_resid);
    ok = ok && set_arg_(kim_fused,  5, sizeof(int),    &C_in);
    ok = ok && set_arg_(kim_fused,  6, sizeof(int),    &C_out);
    ok = ok && set_arg_(kim_fused,  7, sizeof(int),    &L_in);
    ok = ok && set_arg_(kim_fused,  8, sizeof(int),    &L_out);
    ok = ok && set_arg_(kim_fused,  9, sizeof(int),    &K);
    ok = ok && set_arg_(kim_fused, 10, sizeof(int),    &stride);
    ok = ok && set_arg_(kim_fused, 11, sizeof(int),    &padding);
    ok = ok && set_arg_(kim_fused, 12, sizeof(int),    &dilation);
    ok = ok && set_arg_(kim_fused, 13, sizeof(int),    &leaky_flag);
    ok = ok && set_arg_(kim_fused, 14, sizeof(int),    &has_bias_flag);
    ok = ok && set_arg_(kim_fused, 15, sizeof(int),    &has_resid_flag);
    if (!ok) { clReleaseMemObject(out); return nullptr; }
    const size_t gws_im[3] = {(size_t)L_out, (size_t)K, (size_t)C_in};
    err = clEnqueueNDRangeKernel(queue, kim_fused, 3, nullptr, gws_im, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("conv1d_gpu.im2col_fused"));
    if (err != CL_SUCCESS) {
      NNOPT_ERROR_FMT("conv1d_gpu(%s): fused im2col dispatch (%d)", label, (int)err);
      clReleaseMemObject(out);
      return nullptr;
    }
    const int gemm_M = C_out, gemm_K = C_in * K, gemm_N = L_out;
#ifdef NNOPT_USE_FP16
    cl_half h_one = static_cast<cl_half>(nnopt_f32_to_f16(1.0f));
    auto status = clblast::Gemm<cl_half>(
        clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
        gemm_M, gemm_N, gemm_K,
        h_one, w, 0, gemm_K, im2col, 0, gemm_N,
        h_one,                                  // beta=1: preserve fused bias-init
        out, 0, gemm_N,
        &queue, KernelProfiler::event_for("conv1d_gpu.hgemm"));
#else
    auto status = clblast::Gemm<float>(
        clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
        gemm_M, gemm_N, gemm_K,
        1.0f, w, 0, gemm_K, im2col, 0, gemm_N,
        1.0f, out, 0, gemm_N,
        &queue, KernelProfiler::event_for("conv1d_gpu.sgemm"));
#endif
    if (status != clblast::StatusCode::kSuccess) {
      NNOPT_ERROR_FMT("conv1d_gpu(%s): fused Gemm status=%d M=%d N=%d K=%d",
                      label, (int)status, gemm_M, gemm_N, gemm_K);
      clReleaseMemObject(out);
      return nullptr;
    }
    return out;
  }

  // ===== Slow path: im2col (3D) + HGemm beta=0 + bias_add (3 dispatches) =====
  static cl_kernel kim_plain = nullptr;
  if (!kim_plain) {
    kim_plain = clCreateKernel(prog, "im2col_1d", &err);
    if (err != CL_SUCCESS || !kim_plain) {
      NNOPT_ERROR_FMT("conv1d_gpu(%s): kernel im2col_1d (%d)", label, (int)err);
      clReleaseMemObject(out);
      return nullptr;
    }
  }
  bool ok = true;
  ok = ok && set_arg_(kim_plain, 0, sizeof(cl_mem), &in);
  ok = ok && set_arg_(kim_plain, 1, sizeof(cl_mem), &im2col);
  ok = ok && set_arg_(kim_plain, 2, sizeof(int),    &C_in);
  ok = ok && set_arg_(kim_plain, 3, sizeof(int),    &L_in);
  ok = ok && set_arg_(kim_plain, 4, sizeof(int),    &L_out);
  ok = ok && set_arg_(kim_plain, 5, sizeof(int),    &K);
  ok = ok && set_arg_(kim_plain, 6, sizeof(int),    &stride);
  ok = ok && set_arg_(kim_plain, 7, sizeof(int),    &padding);
  ok = ok && set_arg_(kim_plain, 8, sizeof(int),    &dilation);
  if (!ok) { clReleaseMemObject(out); return nullptr; }
  const size_t gws_im[3] = {(size_t)L_out, (size_t)K, (size_t)C_in};
  err = clEnqueueNDRangeKernel(queue, kim_plain, 3, nullptr, gws_im, nullptr, 0, nullptr,
                               KernelProfiler::event_for("conv1d_gpu.im2col"));
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("conv1d_gpu(%s): im2col dispatch (%d)", label, (int)err);
    clReleaseMemObject(out);
    return nullptr;
  }

  const int gemm_M = C_out, gemm_K = C_in * K, gemm_N = L_out;
#ifdef NNOPT_USE_FP16
  cl_half h_one  = static_cast<cl_half>(nnopt_f32_to_f16(1.0f));
  cl_half h_zero = static_cast<cl_half>(nnopt_f32_to_f16(0.0f));
  auto status = clblast::Gemm<cl_half>(
      clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
      gemm_M, gemm_N, gemm_K,
      h_one, w, 0, gemm_K, im2col, 0, gemm_N,
      h_zero, out, 0, gemm_N,
      &queue, KernelProfiler::event_for("conv1d_gpu.hgemm"));
#else
  auto status = clblast::Gemm<float>(
      clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
      gemm_M, gemm_N, gemm_K,
      1.0f, w, 0, gemm_K, im2col, 0, gemm_N,
      0.0f, out, 0, gemm_N,
      &queue, KernelProfiler::event_for("conv1d_gpu.sgemm"));
#endif
  if (status != clblast::StatusCode::kSuccess) {
    NNOPT_ERROR_FMT("conv1d_gpu(%s): Gemm status=%d M=%d N=%d K=%d",
                    label, (int)status, gemm_M, gemm_N, gemm_K);
    clReleaseMemObject(out);
    return nullptr;
  }

  if (has_bias && bias) {
    static cl_kernel kb = nullptr;
    if (!kb) {
      kb = clCreateKernel(prog, "add_bias_broadcast", &err);
      if (err != CL_SUCCESS || !kb) { clReleaseMemObject(out); return nullptr; }
    }
    ok = true;
    ok = ok && set_arg_(kb, 0, sizeof(cl_mem), &out);
    ok = ok && set_arg_(kb, 1, sizeof(cl_mem), &bias);
    ok = ok && set_arg_(kb, 2, sizeof(int),    &C_out);
    ok = ok && set_arg_(kb, 3, sizeof(int),    &L_out);
    if (!ok) { clReleaseMemObject(out); return nullptr; }
    const size_t gws_b[1] = {n_out};
    err = clEnqueueNDRangeKernel(queue, kb, 1, nullptr, gws_b, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("conv1d_gpu.bias"));
    if (err != CL_SUCCESS) { clReleaseMemObject(out); return nullptr; }
  }
  return out;
}

cl_mem gated_activation_gpu(OpenCLContext& cl_ctx,
                            cl_command_queue queue,
                            cl_mem x_2c, int C, int T,
                            const char* label) {
  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx.context();
  cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                              (size_t)C * (size_t)T * sizeof(nnopt_storage_t),
                              nullptr, &err);
  if (err != CL_SUCCESS || !out) return nullptr;
  cl_program prog = cl_ctx.get_program("kernels/flow_affine_coupling.cl");
  if (!prog) prog = cl_ctx.build_program_from_file("kernels/flow_affine_coupling.cl");
  if (!prog) { clReleaseMemObject(out); return nullptr; }
  static cl_kernel kg = nullptr;
  if (!kg) {
    kg = clCreateKernel(prog, "wavenet_gate", &err);
    if (err != CL_SUCCESS || !kg) { clReleaseMemObject(out); return nullptr; }
  }
  bool ok = true;
  ok = ok && set_arg_(kg, 0, sizeof(cl_mem), &x_2c);
  ok = ok && set_arg_(kg, 1, sizeof(cl_mem), &out);
  ok = ok && set_arg_(kg, 2, sizeof(int), &C);
  ok = ok && set_arg_(kg, 3, sizeof(int), &T);
  if (!ok) { clReleaseMemObject(out); return nullptr; }
  const size_t gws[1] = {(size_t)C * (size_t)T};
  err = clEnqueueNDRangeKernel(queue, kg, 1, nullptr, gws, nullptr, 0, nullptr,
                               KernelProfiler::event_for(label));
  if (err != CL_SUCCESS) { clReleaseMemObject(out); return nullptr; }
  return out;
}

bool elem_add_inplace_gpu(OpenCLContext& cl_ctx,
                          cl_command_queue queue,
                          cl_mem a, cl_mem b, int N,
                          const char* label) {
  cl_int err = CL_SUCCESS;
  cl_program prog = cl_ctx.get_program("kernels/flow_affine_coupling.cl");
  if (!prog) prog = cl_ctx.build_program_from_file("kernels/flow_affine_coupling.cl");
  if (!prog) return false;
  static cl_kernel kadd = nullptr;
  if (!kadd) {
    kadd = clCreateKernel(prog, "elem_add_inplace_a", &err);
    if (err != CL_SUCCESS || !kadd) return false;
  }
  bool ok = true;
  ok = ok && set_arg_(kadd, 0, sizeof(cl_mem), &a);
  ok = ok && set_arg_(kadd, 1, sizeof(cl_mem), &b);
  ok = ok && set_arg_(kadd, 2, sizeof(int), &N);
  if (!ok) return false;
  const size_t gws[1] = {(size_t)N};
  err = clEnqueueNDRangeKernel(queue, kadd, 1, nullptr, gws, nullptr, 0, nullptr,
                               KernelProfiler::event_for(label));
  return err == CL_SUCCESS;
}
