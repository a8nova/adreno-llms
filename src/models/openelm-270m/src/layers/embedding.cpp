// Reference: model_info/modeling_openelm.py (OpenELMModel.forward) and kernels/embedding.cl::embedding_forward

#include "layers/embedding.h"

#include "debug_utils.h"
#include "kernel_profiler.h"
#include "model_config.h"
#include "utils.h"

#include <CL/cl.h>
#include <vector>

Embedding::Embedding(OpenCLContext& cl_ctx, Weights& weights) : cl_ctx_(cl_ctx), weights_(weights) {}

Embedding::~Embedding() {
  if (kernel_) {
    clReleaseKernel(kernel_);
    kernel_ = nullptr;
  }
  if (program_) {
    clReleaseProgram(program_);
    program_ = nullptr;
  }
  if (token_ids_buf_) {
    clReleaseMemObject(token_ids_buf_);
    token_ids_buf_ = nullptr;
  }
  if (zero_wpe_) {
    clReleaseMemObject(zero_wpe_);
    zero_wpe_ = nullptr;
  }
  if (out_buf_) {
    clReleaseMemObject(out_buf_);
    out_buf_ = nullptr;
  }
}

// Step Z: ensure the persistent output buffer can hold seq_len rows.
static inline bool ensure_emb_out_buf_(cl_context ctx,
                                       cl_mem* buf,
                                       size_t* cap_bytes,
                                       int seq_len) {
  const size_t needed = (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t);
  if (*cap_bytes >= needed && *buf) return true;
  if (*buf) clReleaseMemObject(*buf);
  cl_int err = CL_SUCCESS;
  *buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, needed, nullptr, &err);
  if (err != CL_SUCCESS || !*buf) {
    NNOPT_ERROR_FMT("Embedding: persistent out_buf alloc failed (%d)", (int)err);
    *buf = nullptr;
    *cap_bytes = 0;
    return false;
  }
  *cap_bytes = needed;
  return true;
}

bool Embedding::initialize() {
  // OpenELM uses RoPE; it does NOT have a learned position embedding table.
  token_emb_ = weights_.get_buffer("transformer.token_embeddings.weight");
  if (!token_emb_) {
    NNOPT_ERROR("Embedding: missing transformer.token_embeddings.weight");
    return false;
  }

  // Optional positional embedding table. If absent, we treat it as zeros by
  // using a lazily-created zero buffer of the same shape (host-side op).
  // For OpenELM it should be absent.
  pos_emb_ = weights_.get_buffer("transformer.position_embeddings.weight", /*optional=*/true);

  program_ = cl_ctx_.build_program_from_file("kernels/embedding.cl");
  if (!program_) {
    NNOPT_ERROR("Embedding: failed to build kernels/embedding.cl");
    return false;
  }

  cl_int err = CL_SUCCESS;
  kernel_ = clCreateKernel(program_, "embedding_forward", &err);
  if (err != CL_SUCCESS || !kernel_) {
    NNOPT_ERROR_FMT("Embedding: clCreateKernel(embedding_forward) failed (%d)", (int)err);
    return false;
  }

  NNOPT_LAYER_INIT("embedding");
  return true;
}

cl_mem Embedding::forward_from_device_token(cl_command_queue queue,
                                            cl_mem token_ids_dev,
                                            size_t dev_offset_bytes,
                                            int start_pos) {
  // Mirrors forward() but skips the host→device upload by copying a single
  // int32 from `token_ids_dev` into the persistent token_ids_buf_, then
  // dispatching the standard embedding kernel.
  if (!kernel_ || !token_emb_) {
    NNOPT_ERROR("Embedding::forward_from_device_token before initialize");
    return nullptr;
  }
  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx_.context();
  const size_t needed_ids_bytes = sizeof(int);
  if (token_ids_buf_cap_bytes_ < needed_ids_bytes) {
    if (token_ids_buf_) clReleaseMemObject(token_ids_buf_);
    token_ids_buf_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, needed_ids_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !token_ids_buf_) {
      NNOPT_ERROR_FMT("Embedding: alloc chained token_ids buffer failed (%d)", (int)err);
      token_ids_buf_ = nullptr;
      token_ids_buf_cap_bytes_ = 0;
      return nullptr;
    }
    token_ids_buf_cap_bytes_ = needed_ids_bytes;
  }
  err = clEnqueueCopyBuffer(queue, token_ids_dev, token_ids_buf_,
                            dev_offset_bytes, 0, sizeof(int), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Embedding: chained copy from argmax_result failed (%d)", (int)err);
    return nullptr;
  }

  // Step Z: persistent output buffer — chained forward returns BORROWED.
  if (!ensure_emb_out_buf_(ctx, &out_buf_, &out_buf_cap_bytes_, /*seq_len=*/1)) return nullptr;
  cl_mem out = out_buf_;

  cl_mem wpe = pos_emb_;
  if (!wpe) {
    const size_t needed_wpe_bytes =
        (size_t)MODEL_CONFIG::MAX_CONTEXT_LENGTH * (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t);
    if (zero_wpe_cap_bytes_ < needed_wpe_bytes) {
      if (zero_wpe_) clReleaseMemObject(zero_wpe_);
      zero_wpe_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, needed_wpe_bytes, nullptr, &err);
      if (err != CL_SUCCESS || !zero_wpe_) {
        NNOPT_ERROR_FMT("Embedding: chained zero_wpe alloc failed (%d)", (int)err);
        zero_wpe_ = nullptr;
        zero_wpe_cap_bytes_ = 0;
        return nullptr;
      }
      const nnopt_storage_t zero = nnopt_storage_t{};
      err = clEnqueueFillBuffer(queue, zero_wpe_, &zero, sizeof(nnopt_storage_t), 0, needed_wpe_bytes,
                                0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding: chained zero_wpe fill failed (%d)", (int)err);
        return nullptr;
      }
      zero_wpe_cap_bytes_ = needed_wpe_bytes;
    }
    wpe = zero_wpe_;
  }

  const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
  if (clSetKernelArg(kernel_, 0, sizeof(cl_mem), &token_ids_buf_) != CL_SUCCESS ||
      clSetKernelArg(kernel_, 1, sizeof(cl_mem), &token_emb_) != CL_SUCCESS ||
      clSetKernelArg(kernel_, 2, sizeof(cl_mem), &wpe) != CL_SUCCESS ||
      clSetKernelArg(kernel_, 3, sizeof(cl_mem), &out) != CL_SUCCESS ||
      clSetKernelArg(kernel_, 4, sizeof(int), &hidden) != CL_SUCCESS ||
      clSetKernelArg(kernel_, 5, sizeof(int), &start_pos) != CL_SUCCESS) {
    NNOPT_ERROR("Embedding: chained setKernelArg failed");
    return nullptr;
  }
  const size_t gws[1] = {(size_t)1};
  cl_event* evt = KernelProfiler::event_for("embedding_chained");
  err = clEnqueueNDRangeKernel(queue, kernel_, 1, nullptr, gws, nullptr, 0, nullptr, evt);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Embedding: chained enqueue failed (%d)", (int)err);
    return nullptr;
  }
  return out;  // BORROWED.
}

cl_mem Embedding::forward(cl_command_queue queue, const int* token_ids, int seq_len, int start_pos) {
  if (!kernel_ || !token_emb_) {
    NNOPT_ERROR("Embedding::forward called before initialize");
    return nullptr;
  }

  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx_.context();

  // Step 4a: persistent token-IDs upload buffer. Grow on demand; reuse the
  // capacity sticky-style. Saves ~50 µs of clCreateBuffer/clReleaseMemObject
  // overhead per decode token.
  const size_t needed_ids_bytes = (size_t)seq_len * sizeof(int);
  if (token_ids_buf_cap_bytes_ < needed_ids_bytes) {
    if (token_ids_buf_) clReleaseMemObject(token_ids_buf_);
    token_ids_buf_ = clCreateBuffer(ctx, CL_MEM_READ_ONLY, needed_ids_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !token_ids_buf_) {
      NNOPT_ERROR_FMT("Embedding: failed to grow token_ids buffer (%d)", (int)err);
      token_ids_buf_ = nullptr;
      token_ids_buf_cap_bytes_ = 0;
      return nullptr;
    }
    token_ids_buf_cap_bytes_ = needed_ids_bytes;
  }
  err = clEnqueueWriteBuffer(queue, token_ids_buf_, CL_FALSE, 0, needed_ids_bytes,
                             (const void*)token_ids, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Embedding: clEnqueueWriteBuffer token_ids failed (%d)", (int)err);
    return nullptr;
  }

  // Step Z: persistent output buffer. forward() returns this BORROWED.
  if (!ensure_emb_out_buf_(ctx, &out_buf_, &out_buf_cap_bytes_, seq_len)) return nullptr;
  cl_mem out = out_buf_;

  // For RoPE-only models, we want tok_emb + 0. The kernel currently expects a
  // wpe pointer. If pos_emb_ is null, we use a persistent zero buffer sized to
  // MAX_CONTEXT_LENGTH × HIDDEN_SIZE (allocated once, zero-filled lazily).
  cl_mem wpe = pos_emb_;
  if (!wpe) {
    const size_t needed_wpe_bytes =
        (size_t)MODEL_CONFIG::MAX_CONTEXT_LENGTH * (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t);
    if (zero_wpe_cap_bytes_ < needed_wpe_bytes) {
      if (zero_wpe_) clReleaseMemObject(zero_wpe_);
      zero_wpe_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, needed_wpe_bytes, nullptr, &err);
      if (err != CL_SUCCESS || !zero_wpe_) {
        NNOPT_ERROR_FMT("Embedding: failed to allocate zero_wpe buffer (%d)", (int)err);
        zero_wpe_ = nullptr;
        zero_wpe_cap_bytes_ = 0;
        clReleaseMemObject(out);
        return nullptr;
      }
      const nnopt_storage_t zero = nnopt_storage_t{};
      err = clEnqueueFillBuffer(queue, zero_wpe_, &zero, sizeof(nnopt_storage_t), 0, needed_wpe_bytes,
                                0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding: clEnqueueFillBuffer zero_wpe failed (%d)", (int)err);
        return nullptr;
      }
      zero_wpe_cap_bytes_ = needed_wpe_bytes;
    }
    wpe = zero_wpe_;
    // No-op: start_pos still indexes into a fully-zero table.
  }

  const int hidden = MODEL_CONFIG::HIDDEN_SIZE;

  err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &token_ids_buf_);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Embedding: setArg token_ids failed (%d)", (int)err);
    goto cleanup;
  }
  err = clSetKernelArg(kernel_, 1, sizeof(cl_mem), &token_emb_);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Embedding: setArg wte failed (%d)", (int)err);
    goto cleanup;
  }
  err = clSetKernelArg(kernel_, 2, sizeof(cl_mem), &wpe);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Embedding: setArg wpe failed (%d)", (int)err);
    goto cleanup;
  }
  err = clSetKernelArg(kernel_, 3, sizeof(cl_mem), &out);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Embedding: setArg output failed (%d)", (int)err);
    goto cleanup;
  }
  err = clSetKernelArg(kernel_, 4, sizeof(int), &hidden);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Embedding: setArg hidden failed (%d)", (int)err);
    goto cleanup;
  }
  err = clSetKernelArg(kernel_, 5, sizeof(int), &start_pos);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Embedding: setArg start_pos failed (%d)", (int)err);
    goto cleanup;
  }

  {
    const size_t gws[1] = {(size_t)seq_len};
    cl_event* evt = KernelProfiler::event_for("embedding");
    err = clEnqueueNDRangeKernel(queue, kernel_, 1, nullptr, gws, nullptr, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
      NNOPT_ERROR_FMT("Embedding: enqueue failed (%d)", (int)err);
      goto cleanup;
    }
  }

  NNOPT_DEBUG_SYNC(queue);

cleanup:
  if (err != CL_SUCCESS) {
    return nullptr;
  }
  return out;  // BORROWED — owned by *this until next call.
}
