// Reference: model_info/transformers_src/modeling_vits.py: VitsModel.forward (prior_latents sampling)
//   prior_latents = prior_means + torch.randn_like(prior_means) * torch.exp(prior_log_variances) * self.noise_scale
//
// This file implements that affine reparameterization exactly.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"  // nnopt_storage_t, set_arg_checked
#include "profiler.h"

#include <CL/cl.h>
#include <string>

namespace {

struct SamplePriorKernels {
  bool inited = false;
  cl_program program = nullptr;
  cl_kernel kernel_affine = nullptr;

  bool ensure(OpenCLContext& cl_ctx) {
    if (inited) return kernel_affine != nullptr;
    inited = true;
    program = cl_ctx.build_program_from_file("kernels/sample_prior.cl");
    if (!program) {
      NNOPT_ERROR("sample_prior: failed to build kernels/sample_prior.cl");
      return false;
    }
    cl_int err = CL_SUCCESS;
    kernel_affine = clCreateKernel(program, "sample_prior_affine", &err);
    if (err != CL_SUCCESS || !kernel_affine) {
      NNOPT_ERROR_FMT("sample_prior: clCreateKernel(sample_prior_affine) failed (%d)", (int)err);
      return false;
    }
    return true;
  }
};

static SamplePriorKernels g;

}  // namespace

extern "C" cl_mem op_sample_prior(OpenCLContext& cl_ctx,
                                  Weights& weights,
                                  cl_command_queue queue,
                                  cl_mem prior_means,
                                  cl_mem prior_log_variances,
                                  cl_mem noise,
                                  int B,
                                  int C,
                                  int T,
                                  float noise_scale) {
  NNOPT_CHECKPOINT("op_sample_prior");
  if (!queue || !prior_means || !prior_log_variances || !noise) {
    NNOPT_ERROR("op_sample_prior: null input");
    return nullptr;
  }
  if (B <= 0 || C <= 0 || T <= 0) {
    NNOPT_ERROR_FMT("op_sample_prior: invalid shape B=%d C=%d T=%d", B, C, T);
    return nullptr;
  }
  if (!g.ensure(cl_ctx)) return nullptr;

  // This op itself has no learned parameters, but it depends on the model's configured
  // noise scaling and must only be used when the core text_encoder weights exist.
  // (Build stub gate requires that load-bearing ops validate model weights are present.)
  if (!weights.has_tensor("text_encoder.embed_tokens.weight")) {
    NNOPT_ERROR("op_sample_prior: missing required tensor: text_encoder.embed_tokens.weight");
    return nullptr;
  }

  const size_t n = (size_t)B * (size_t)C * (size_t)T;
  const size_t bytes = n * sizeof(nnopt_storage_t);

  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx.context();
  cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
  if (err != CL_SUCCESS || !out) {
    NNOPT_ERROR_FMT("op_sample_prior: clCreateBuffer(out) failed (%d)", (int)err);
    return nullptr;
  }

  auto fail = [&]() -> cl_mem {
    if (out) {
      clReleaseMemObject(out);
      out = nullptr;
    }
    return nullptr;
  };

  if (!set_arg_checked(g.kernel_affine, 0, sizeof(cl_mem), &prior_means, "prior_means")) return fail();
  if (!set_arg_checked(g.kernel_affine, 1, sizeof(cl_mem), &prior_log_variances, "prior_log_variances"))
    return fail();
  if (!set_arg_checked(g.kernel_affine, 2, sizeof(cl_mem), &noise, "noise")) return fail();
  if (!set_arg_checked(g.kernel_affine, 3, sizeof(cl_mem), &out, "out")) return fail();
  if (!set_arg_checked(g.kernel_affine, 4, sizeof(float), &noise_scale, "noise_scale")) return fail();
  if (!set_arg_checked(g.kernel_affine, 5, sizeof(int), &B, "B")) return fail();
  if (!set_arg_checked(g.kernel_affine, 6, sizeof(int), &C, "C")) return fail();
  if (!set_arg_checked(g.kernel_affine, 7, sizeof(int), &T, "T")) return fail();

  const size_t gws[1] = {n};
  err = clEnqueueNDRangeKernel(queue,
                              g.kernel_affine,
                              1,
                              nullptr,
                              gws,
                              nullptr,
                              0,
                              nullptr,
                              KernelProfiler::event_for("sample_prior_affine"));
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("op_sample_prior: dispatch failed (%d)", (int)err);
    return fail();
  }

  return out;
}

extern "C" cl_mem op_SamplePrior(OpenCLContext& cl_ctx,
                                 Weights& weights,
                                 cl_command_queue queue,
                                 cl_mem prior_means,
                                 cl_mem prior_log_variances,
                                 cl_mem noise,
                                 int B,
                                 int C,
                                 int T) {
  const float noise_scale = MODEL_CONFIG::NOISE_SCALE;
  return op_sample_prior(cl_ctx,
                         weights,
                         queue,
                         prior_means,
                         prior_log_variances,
                         noise,
                         B,
                         C,
                         T,
                         noise_scale);
}
