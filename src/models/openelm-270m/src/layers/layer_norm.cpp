// Reference: model_info/modeling_openelm.py:15-53 (OpenELMRMSNorm)

#include "layers/layer_norm.h"

#include "debug_utils.h"
#include "kernel_profiler.h"
#include "model_config.h"
#include "utils.h"

#include <CL/cl.h>

LayerNorm::LayerNorm(OpenCLContext& cl_ctx, Weights& weights, int layer_idx, const std::string& weight_key)
    : cl_ctx_(cl_ctx), weights_(weights), layer_idx_(layer_idx), weight_key_(weight_key) {}

LayerNorm::~LayerNorm() {
  if (out_buf_) clReleaseMemObject(out_buf_);
}

bool LayerNorm::initialize() {
  weight_ = weights_.get_buffer(weight_key_);
  if (!weight_) {
    NNOPT_ERROR_FMT("LayerNorm[%d]: missing weight \"%s\"", layer_idx_, weight_key_.c_str());
    return false;
  }
  return true;
}

cl_mem LayerNorm::forward(cl_command_queue queue, cl_mem input, int seq_len) {
  if (!weight_) {
    NNOPT_ERROR_FMT("LayerNorm[%d]: forward before initialize", layer_idx_);
    return nullptr;
  }
  const size_t needed = (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t);
  if (out_buf_cap_bytes_ < needed) {
    if (out_buf_) clReleaseMemObject(out_buf_);
    cl_int err = CL_SUCCESS;
    out_buf_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, needed, nullptr, &err);
    if (err != CL_SUCCESS || !out_buf_) {
      NNOPT_ERROR_FMT("LayerNorm[%d]: persistent out_buf alloc failed (%d)", layer_idx_, (int)err);
      out_buf_ = nullptr;
      out_buf_cap_bytes_ = 0;
      return nullptr;
    }
    out_buf_cap_bytes_ = needed;
  }
  if (!rmsnorm_forward_into(cl_ctx_, queue, input, weight_, out_buf_,
                            seq_len, MODEL_CONFIG::HIDDEN_SIZE, MODEL_CONFIG::NORM_EPS)) {
    return nullptr;
  }
  return out_buf_;
}

// rmsnorm.cl declares __attribute__((reqd_work_group_size(64,1,1))) — caller
// MUST pass lws=64 or the runtime fails CL_INVALID_WORK_GROUP_SIZE silently
// and produces orthogonal output (cos≈0 in SxS).
static constexpr size_t RMSNORM_WG = 64;

// Internal: dispatch the rmsnorm kernel into a given output buffer.
static bool rmsnorm_dispatch_(OpenCLContext& cl_ctx,
                              cl_command_queue queue,
                              cl_mem input,
                              cl_mem weight,
                              cl_mem output,
                              int rows,
                              int cols,
                              float eps);

bool rmsnorm_forward_into(OpenCLContext& cl_ctx,
                          cl_command_queue queue,
                          cl_mem input,
                          cl_mem weight,
                          cl_mem output,
                          int rows,
                          int cols,
                          float eps) {
  return rmsnorm_dispatch_(cl_ctx, queue, input, weight, output, rows, cols, eps);
}

cl_mem rmsnorm_forward(OpenCLContext& cl_ctx,
                       cl_command_queue queue,
                       cl_mem input,
                       cl_mem weight,
                       int rows,
                       int cols,
                       float eps) {
  cl_int err = CL_SUCCESS;
  cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                              (size_t)rows * (size_t)cols * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !out) {
    NNOPT_ERROR_FMT("rmsnorm_forward: clCreateBuffer out failed (%d)", (int)err);
    return nullptr;
  }
  if (!rmsnorm_dispatch_(cl_ctx, queue, input, weight, out, rows, cols, eps)) {
    clReleaseMemObject(out);
    return nullptr;
  }
  return out;
}

static bool rmsnorm_dispatch_(OpenCLContext& cl_ctx,
                              cl_command_queue queue,
                              cl_mem input,
                              cl_mem weight,
                              cl_mem output,
                              int rows,
                              int cols,
                              float eps) {
  static cl_program s_prog = nullptr;
  static cl_kernel s_kernel = nullptr;
  if (!s_kernel) {
    s_prog = cl_ctx.build_program_from_file("kernels/rmsnorm.cl");
    if (!s_prog) {
      NNOPT_ERROR("rmsnorm_forward: failed to build kernels/rmsnorm.cl");
      return false;
    }
    cl_int kerr = CL_SUCCESS;
    s_kernel = clCreateKernel(s_prog, "rmsnorm_forward", &kerr);
    if (kerr != CL_SUCCESS || !s_kernel) {
      NNOPT_ERROR_FMT("rmsnorm_forward: clCreateKernel failed (%d)", (int)kerr);
      return false;
    }
  }

  cl_int err = CL_SUCCESS;
  err = clSetKernelArg(s_kernel, 0, sizeof(cl_mem), &input);
  if (err == CL_SUCCESS) err = clSetKernelArg(s_kernel, 1, sizeof(cl_mem), &weight);
  if (err == CL_SUCCESS) err = clSetKernelArg(s_kernel, 2, sizeof(cl_mem), &output);
  if (err == CL_SUCCESS) err = clSetKernelArg(s_kernel, 3, sizeof(int), &rows);
  if (err == CL_SUCCESS) err = clSetKernelArg(s_kernel, 4, sizeof(int), &cols);
  if (err == CL_SUCCESS) err = clSetKernelArg(s_kernel, 5, sizeof(float), &eps);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("rmsnorm_forward: clSetKernelArg failed (%d)", (int)err);
    return false;
  }

  const size_t gws = (size_t)rows * RMSNORM_WG;
  const size_t lws = RMSNORM_WG;
  cl_event* evt = KernelProfiler::event_for("rmsnorm");
  err = clEnqueueNDRangeKernel(queue, s_kernel, 1, nullptr, &gws, &lws, 0, nullptr, evt);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("rmsnorm_forward: clEnqueueNDRangeKernel failed (%d)", (int)err);
    return false;
  }
  NNOPT_DEBUG_SYNC(queue);
  return true;
}
