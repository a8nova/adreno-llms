// Reference: model_info/modeling_openelm.py (OpenELMModel.forward) and kernels/embedding.cl::embedding_forward

#include "layers/embedding.h"

#include "debug_utils.h"
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

cl_mem Embedding::forward(cl_command_queue queue, const int* token_ids, int seq_len, int start_pos) {
  if (!kernel_ || !token_emb_) {
    NNOPT_ERROR("Embedding::forward called before initialize");
    return nullptr;
  }

  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx_.context();

  // Upload token IDs
  cl_mem token_ids_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       (size_t)seq_len * sizeof(int), (void*)token_ids, &err);
  if (err != CL_SUCCESS || !token_ids_buf) {
    NNOPT_ERROR_FMT("Embedding: failed to create token_ids buffer (%d)", (int)err);
    return nullptr;
  }

  // Output: [seq_len, hidden]
  const size_t out_elems = (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE;
  const size_t out_bytes = out_elems * sizeof(nnopt_storage_t);
  cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_bytes, nullptr, &err);
  if (err != CL_SUCCESS || !out) {
    NNOPT_ERROR_FMT("Embedding: failed to create output buffer (%d)", (int)err);
    clReleaseMemObject(token_ids_buf);
    return nullptr;
  }

  // For RoPE-only models, we want tok_emb + 0. The kernel currently expects a
  // wpe pointer. If pos_emb_ is null, we allocate a temporary zero buffer.
  cl_mem wpe = pos_emb_;
  cl_mem tmp_zero = nullptr;
  if (!wpe) {
    // One row of zeros is enough because the kernel indexes wpe by abs_pos.
    // But to keep semantics safe, allocate MAX_SEQ_LEN rows (still small-ish)
    // is too big. Instead, allocate seq_len rows and clamp start_pos to 0.
    // Since OpenELM has no wpe in PyTorch, the additive term must be zero for
    // all positions; using a seq_len table and forcing start_pos=0 preserves that.
    const size_t wpe_elems = out_elems;
    const size_t wpe_bytes = wpe_elems * sizeof(nnopt_storage_t);
    tmp_zero = clCreateBuffer(ctx, CL_MEM_READ_WRITE, wpe_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !tmp_zero) {
      NNOPT_ERROR_FMT("Embedding: failed to create tmp_zero wpe buffer (%d)", (int)err);
      clReleaseMemObject(token_ids_buf);
      clReleaseMemObject(out);
      return nullptr;
    }

    // Fill with 0
    const nnopt_storage_t zero = nnopt_storage_t{};
    err = clEnqueueFillBuffer(queue, tmp_zero, &zero, sizeof(nnopt_storage_t), 0, wpe_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      NNOPT_ERROR_FMT("Embedding: clEnqueueFillBuffer failed (%d)", (int)err);
      clReleaseMemObject(tmp_zero);
      clReleaseMemObject(token_ids_buf);
      clReleaseMemObject(out);
      return nullptr;
    }
    wpe = tmp_zero;
    start_pos = 0;
  }

  const int hidden = MODEL_CONFIG::HIDDEN_SIZE;

  err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &token_ids_buf);
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
    err = clEnqueueNDRangeKernel(queue, kernel_, 1, nullptr, gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      NNOPT_ERROR_FMT("Embedding: enqueue failed (%d)", (int)err);
      goto cleanup;
    }
  }

  // Keep sync conservative for now (debug-first).
  clFinish(queue);

cleanup:
  if (tmp_zero) clReleaseMemObject(tmp_zero);
  clReleaseMemObject(token_ids_buf);

  if (err != CL_SUCCESS) {
    clReleaseMemObject(out);
    return nullptr;
  }
  return out;
}
