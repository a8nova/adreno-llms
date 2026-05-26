// Reference: kernels/splice_image_tokens.cl splice_image_tokens
// Host wrapper for the in-place multi-position splice.
//
// Semantic: text_embeds[positions[i], :] = image_features[i, :] for i in [0, N).
// Mutates text_embeds in place; T_in == T_out.

#include "opencl_context.h"
#include "debug_utils.h"
#include "utils.h"

#include <CL/cl.h>
#include <vector>

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
  s.prog = cl_ctx.build_program_from_file("kernels/splice_image_tokens.cl");  // PROGRAM-INIT-OK
  if (!s.prog) {
    NNOPT_ERROR("splice_image_tokens: failed to build kernels/splice_image_tokens.cl");
    return false;
  }
  cl_int err = CL_SUCCESS;
  s.k = clCreateKernel(s.prog, "splice_image_tokens", &err);
  if (err != CL_SUCCESS || !s.k) {
    NNOPT_ERROR_FMT("splice_image_tokens: clCreateKernel failed (%d)", (int)err);
    return false;
  }
  s.initialized = true;
  return true;
}
}  // namespace

// In-place multi-position splice.
//   text_embeds   : [T_in, D] storage_t — mutated in place
//   image_features: [N, D]   storage_t — projector output
//   positions     : [N]      int32     — row indices into text_embeds
//   N, D          : dims
//
// Returns true on success.
bool splice_image_tokens(
    OpenCLContext& cl_ctx,
    cl_mem text_embeds,
    cl_mem image_features,
    cl_mem positions,
    int N,
    int D) {
  if (!ensure_initialized(cl_ctx)) return false;
  if (!text_embeds || !image_features || !positions) {
    NNOPT_ERROR("splice_image_tokens: null buffer arg");
    return false;
  }
  if (N <= 0 || D <= 0) {
    NNOPT_ERROR_FMT("splice_image_tokens: invalid dims N=%d D=%d", N, D);
    return false;
  }

  auto& s = state();
  if (!set_arg_checked(s.k, 0, sizeof(cl_mem), &text_embeds, "text_embeds")) return false;
  if (!set_arg_checked(s.k, 1, sizeof(cl_mem), &image_features, "image_features")) return false;
  if (!set_arg_checked(s.k, 2, sizeof(cl_mem), &positions, "positions")) return false;
  if (!set_arg_checked(s.k, 3, sizeof(int), &N, "N")) return false;
  if (!set_arg_checked(s.k, 4, sizeof(int), &D, "D")) return false;

  const size_t gws[2] = {(size_t)N, (size_t)D};
  cl_int err = clEnqueueNDRangeKernel(cl_ctx.queue(), s.k, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("splice_image_tokens: dispatch failed (%d)", (int)err);
    return false;
  }
  return true;
}
