// Reference: model_info/transformers_src/modeling_lfm2.py:320-377 Lfm2Model.forward (embed_tokens)

#include "layers/embedding.h"

#include "debug_utils.h"
#include "kernel_profiler.h"
#include "model_config.h"
#include "opencl_context.h"
#include "utils.h"
#include "weights.h"

#include <CL/cl.h>
#include <cstddef>
#include <cstdint>
#include <vector>

Embedding::Embedding(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {}

Embedding::~Embedding() {
  if (kernel_) clReleaseKernel(kernel_);
  if (program_) clReleaseProgram(program_);
  if (buf_out_) clReleaseMemObject(buf_out_);
  if (buf_input_ids_) clReleaseMemObject(buf_input_ids_);
}

bool Embedding::initialize() {
  embed_w_ = weights_.get_buffer("model.embed_tokens.weight");
  if (!embed_w_) {
    NNOPT_ERROR("Embedding: missing weight model.embed_tokens.weight");
    return false;
  }

  program_ = cl_ctx_.build_program_from_file("kernels/embedding.cl");
  if (!program_) return false;

  cl_int err = CL_SUCCESS;
  kernel_ = clCreateKernel(program_, "embedding_forward", &err);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Embedding: clCreateKernel(embedding_forward) failed: %d", (int)err);
    return false;
  }

  NNOPT_LAYER_INIT("embedding");
  return true;
}

cl_mem Embedding::forward(cl_command_queue queue, const int* input_ids, int seq_len) {
  if (!kernel_ || !embed_w_) return nullptr;
  if (seq_len <= 0) return nullptr;

  cl_context ctx = cl_ctx_.context();
  cl_int err = CL_SUCCESS;

  // Lazy-grow persistent buffers: input_ids (small) and output [seq, H].
  if (seq_len > buf_seq_capacity_ || !buf_out_) {
    if (buf_out_) { clReleaseMemObject(buf_out_); buf_out_ = nullptr; }
    if (buf_input_ids_) { clReleaseMemObject(buf_input_ids_); buf_input_ids_ = nullptr; }
    const size_t out_elems = (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE;
    buf_out_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_elems * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: alloc buf_out: %d", err); return nullptr; }
    buf_input_ids_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (size_t)seq_len * sizeof(int), nullptr, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: alloc buf_input_ids: %d", err); return nullptr; }
    buf_seq_capacity_ = seq_len;
  }

  // Upload current input_ids into persistent buffer (no realloc).
  err = clEnqueueWriteBuffer(queue, buf_input_ids_, CL_FALSE, 0,
                             (size_t)seq_len * sizeof(int), input_ids, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: ids write: %d", err); return nullptr; }

  int hidden = MODEL_CONFIG::HIDDEN_SIZE;
  err  = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buf_input_ids_);
  err |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &embed_w_);
  err |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &buf_out_);
  err |= clSetKernelArg(kernel_, 3, sizeof(int),    &seq_len);
  err |= clSetKernelArg(kernel_, 4, sizeof(int),    &hidden);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: setArgs: %d", err); return nullptr; }

  size_t gws[1] = {(size_t)seq_len};
  err = clEnqueueNDRangeKernel(queue, kernel_, 1, nullptr, gws, nullptr, 0, nullptr,
                               KernelProfiler::event_for("embedding"));
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: enqueue: %d", err); return nullptr; }
  NNOPT_DEBUG_SYNC(queue);
  return buf_out_;  // BORROWED
}

cl_mem Embedding::forward_from_device_token(cl_command_queue queue, cl_mem token_ids_dev,
                                            size_t dev_offset_bytes) {
  if (!kernel_ || !embed_w_) return nullptr;
  if (!token_ids_dev) return nullptr;

  cl_context ctx = cl_ctx_.context();
  cl_int err = CL_SUCCESS;

  // Reuse the same persistent buffers as forward(). Ensure capacity for 1.
  if (buf_seq_capacity_ < 1 || !buf_out_) {
    if (buf_out_) { clReleaseMemObject(buf_out_); buf_out_ = nullptr; }
    if (buf_input_ids_) { clReleaseMemObject(buf_input_ids_); buf_input_ids_ = nullptr; }
    buf_out_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                              (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding(chained): alloc buf_out: %d", err); return nullptr; }
    buf_input_ids_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(int), nullptr, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding(chained): alloc buf_input_ids: %d", err); return nullptr; }
    buf_seq_capacity_ = 1;
  }

  // Pull the single int32 token id from the device-side buffer.
  err = clEnqueueCopyBuffer(queue, token_ids_dev, buf_input_ids_,
                            dev_offset_bytes, 0, sizeof(int), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding(chained): copy token id: %d", err); return nullptr; }

  int hidden = MODEL_CONFIG::HIDDEN_SIZE;
  int seq_len = 1;
  err  = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buf_input_ids_);
  err |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &embed_w_);
  err |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &buf_out_);
  err |= clSetKernelArg(kernel_, 3, sizeof(int),    &seq_len);
  err |= clSetKernelArg(kernel_, 4, sizeof(int),    &hidden);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding(chained): setArgs: %d", err); return nullptr; }

  size_t gws[1] = {1};
  err = clEnqueueNDRangeKernel(queue, kernel_, 1, nullptr, gws, nullptr, 0, nullptr,
                               KernelProfiler::event_for("embedding_chained"));
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding(chained): enqueue: %d", err); return nullptr; }
  return buf_out_;  // BORROWED
}
