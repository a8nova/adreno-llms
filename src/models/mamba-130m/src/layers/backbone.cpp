// Reference: model_info/transformers_src/modeling_mamba.py:590-750 (approx) MambaModel.forward + MambaForCausalLM.forward
// Implements embedding, final RMSNorm, and lm_head projection.

#include "layers/backbone.h"

#include "debug_utils.h"
#include "kernel_profiler.h"
#include "model_config.h"
#include "opencl_context.h"
#include "utils.h"
#include "weights.h"

#include <clblast.h>
#include <string>

Backbone::Backbone(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {}

Backbone::~Backbone() {
  if (embed_program_) clReleaseProgram(embed_program_);
  if (embed_kernel_) clReleaseKernel(embed_kernel_);
  if (rms_program_) clReleaseProgram(rms_program_);
  if (rms_kernel_) clReleaseKernel(rms_kernel_);
}

bool Backbone::initialize() {
  embed_program_ = cl_ctx_.build_program_from_file("kernels/embedding.cl");
  if (!embed_program_) {
    NNOPT_ERROR("Backbone: failed to build kernels/embedding.cl");
    return false;
  }
  cl_int err = CL_SUCCESS;
  embed_kernel_ = clCreateKernel(embed_program_, "embedding_forward", &err);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone: clCreateKernel(embedding_forward) failed: %d", (int)err);
    return false;
  }

  rms_program_ = cl_ctx_.build_program_from_file("kernels/layer_norm.cl");
  if (!rms_program_) {
    NNOPT_ERROR("Backbone: failed to build kernels/layer_norm.cl");
    return false;
  }
  rms_kernel_ = clCreateKernel(rms_program_, "rms_norm", &err);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone: clCreateKernel(rms_norm) failed: %d", (int)err);
    return false;
  }

  // Verify required weights exist (they are lazy-uploaded by Weights)
  if (!weights_.has_tensor("backbone.embeddings.weight")) {
    NNOPT_ERROR("Backbone: missing backbone.embeddings.weight");
    return false;
  }
  if (!weights_.has_tensor("backbone.norm_f.weight")) {
    NNOPT_ERROR("Backbone: missing backbone.norm_f.weight");
    return false;
  }
  // lm_head.weight is tied to backbone.embeddings.weight for this model (see modeling_mamba.py: MambaForCausalLM._tied_weights_keys)

  return true;
}

cl_mem Backbone::embed(cl_command_queue queue, const int* token_ids, int seq_len) {
  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx_.context();

  cl_mem tok_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 (size_t)seq_len * sizeof(int), (void*)token_ids, &err);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::embed: clCreateBuffer(token_ids) failed: %d", (int)err);
    return nullptr;
  }

  cl_mem embed_w = weights_.get_buffer("backbone.embeddings.weight");
  if (!embed_w) {
    NNOPT_ERROR("Backbone::embed: get_buffer(backbone.embeddings.weight) failed");
    clReleaseMemObject(tok_buf);
    return nullptr;
  }

  const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
  cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                             (size_t)seq_len * hidden * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::embed: clCreateBuffer(out) failed: %d", (int)err);
    clReleaseMemObject(tok_buf);
    return nullptr;
  }

  // embedding_forward(token_ids, wte, wpe, output, hidden_size, start_pos)
  // Mamba has NO absolute position embedding; pass wpe as a valid dummy buffer and keep start_pos=0.
  // The scaffold's embedding kernel still expects wpe; passing nullptr here can crash Adreno's driver
  // during clSetKernelArg validation.
  const int start_pos = 0;

  cl_mem wpe_dummy = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)hidden * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !wpe_dummy) {
    NNOPT_ERROR_FMT("Backbone::embed: clCreateBuffer(wpe_dummy) failed: %d", (int)err);
    clReleaseMemObject(tok_buf);
    clReleaseMemObject(out);
    return nullptr;
  }
  // Zero it so it adds nothing.
  err = clEnqueueFillBuffer(queue, wpe_dummy, "\0", 1, 0, (size_t)hidden * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::embed: fill wpe_dummy failed: %d", (int)err);
    clReleaseMemObject(wpe_dummy);
    clReleaseMemObject(tok_buf);
    clReleaseMemObject(out);
    return nullptr;
  }

  int arg = 0;
  err = clSetKernelArg(embed_kernel_, arg++, sizeof(cl_mem), &tok_buf);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::embed: setArg %d failed: %d", arg - 1, (int)err);
    clReleaseMemObject(wpe_dummy);
    clReleaseMemObject(tok_buf);
    clReleaseMemObject(out);
    return nullptr;
  }
  err = clSetKernelArg(embed_kernel_, arg++, sizeof(cl_mem), &embed_w);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::embed: setArg %d failed: %d", arg - 1, (int)err);
    clReleaseMemObject(wpe_dummy);
    clReleaseMemObject(tok_buf);
    clReleaseMemObject(out);
    return nullptr;
  }
  err = clSetKernelArg(embed_kernel_, arg++, sizeof(cl_mem), &wpe_dummy);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::embed: setArg %d failed: %d", arg - 1, (int)err);
    clReleaseMemObject(wpe_dummy);
    clReleaseMemObject(tok_buf);
    clReleaseMemObject(out);
    return nullptr;
  }
  err = clSetKernelArg(embed_kernel_, arg++, sizeof(cl_mem), &out);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::embed: setArg %d failed: %d", arg - 1, (int)err);
    clReleaseMemObject(wpe_dummy);
    clReleaseMemObject(tok_buf);
    clReleaseMemObject(out);
    return nullptr;
  }
  err = clSetKernelArg(embed_kernel_, arg++, sizeof(int), &hidden);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::embed: setArg %d failed: %d", arg - 1, (int)err);
    clReleaseMemObject(wpe_dummy);
    clReleaseMemObject(tok_buf);
    clReleaseMemObject(out);
    return nullptr;
  }
  err = clSetKernelArg(embed_kernel_, arg++, sizeof(int), &start_pos);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::embed: setArg %d failed: %d", arg - 1, (int)err);
    clReleaseMemObject(wpe_dummy);
    clReleaseMemObject(tok_buf);
    clReleaseMemObject(out);
    return nullptr;
  }

  // Cooperative dispatch — 1 WG per row, 64 threads cooperate over hidden.
  // hidden_size=768 ⇒ 12 fp16 / thread = 3 vec4.
  const size_t WG_E = 64;
  size_t gws[1] = {(size_t)seq_len * WG_E};
  size_t lws[1] = {WG_E};
  cl_event* prof_evt = KernelProfiler::event_for("embedding");
  err = clEnqueueNDRangeKernel(queue, embed_kernel_, 1, nullptr, gws, lws, 0, nullptr, prof_evt);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::embed: enqueue failed: %d", (int)err);
    clReleaseMemObject(wpe_dummy);
    clReleaseMemObject(tok_buf);
    clReleaseMemObject(out);
    return nullptr;
  }

  clReleaseMemObject(wpe_dummy);
  clReleaseMemObject(tok_buf);
  return out;
}

cl_mem Backbone::final_norm(cl_command_queue queue, cl_mem hidden, int seq_len) {
  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx_.context();

  const int dim = MODEL_CONFIG::HIDDEN_SIZE;
  cl_mem w = weights_.get_buffer("backbone.norm_f.weight");
  if (!w) {
    NNOPT_ERROR("Backbone::final_norm: get_buffer(backbone.norm_f.weight) failed");
    return nullptr;
  }

  cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                             (size_t)seq_len * dim * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::final_norm: clCreateBuffer(out) failed: %d", (int)err);
    return nullptr;
  }

  float eps = MODEL_CONFIG::LAYER_NORM_EPS;
  int arg = 0;
  err = clSetKernelArg(rms_kernel_, arg++, sizeof(cl_mem), &hidden);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::final_norm: setArg %d failed: %d", arg - 1, (int)err);
    clReleaseMemObject(out);
    return nullptr;
  }
  err = clSetKernelArg(rms_kernel_, arg++, sizeof(cl_mem), &out);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::final_norm: setArg %d failed: %d", arg - 1, (int)err);
    clReleaseMemObject(out);
    return nullptr;
  }
  err = clSetKernelArg(rms_kernel_, arg++, sizeof(cl_mem), &w);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::final_norm: setArg %d failed: %d", arg - 1, (int)err);
    clReleaseMemObject(out);
    return nullptr;
  }
  err = clSetKernelArg(rms_kernel_, arg++, sizeof(int), &dim);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::final_norm: setArg %d failed: %d", arg - 1, (int)err);
    clReleaseMemObject(out);
    return nullptr;
  }
  err = clSetKernelArg(rms_kernel_, arg++, sizeof(float), &eps);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::final_norm: setArg %d failed: %d", arg - 1, (int)err);
    clReleaseMemObject(out);
    return nullptr;
  }

  size_t gws[1] = {(size_t)seq_len};
  err = clEnqueueNDRangeKernel(queue, rms_kernel_, 1, nullptr, gws, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::final_norm: enqueue failed: %d", (int)err);
    clReleaseMemObject(out);
    return nullptr;
  }
  return out;
}

cl_mem Backbone::lm_head(cl_command_queue queue, cl_mem hidden_normed, int seq_len) {
  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx_.context();

  // Tied weights: lm_head.weight aliases backbone.embeddings.weight
  cl_mem w = weights_.get_buffer("backbone.embeddings.weight");
  if (!w) {
    NNOPT_ERROR("Backbone::lm_head: get_buffer(lm_head.weight) failed");
    return nullptr;
  }

  const int M = seq_len;
  const int K = MODEL_CONFIG::HIDDEN_SIZE;
  const int N = MODEL_CONFIG::VOCAB_SIZE;
  cl_mem logits = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                (size_t)M * N * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("Backbone::lm_head: clCreateBuffer(logits) failed: %d", (int)err);
    return nullptr;
  }

  // Tied lm_head uses embedding weight, which is an Embedding/Conv1D-style matrix (stored [in, out]).
  // We need y = x @ W where W is [hidden, vocab]. The stored weight is [vocab, hidden] in HF state dict
  // for embeddings, so use a raw GEMM with Transpose::kYes to interpret it as [hidden, vocab].
  // NOTE: we intentionally do NOT use pytorch_linear/pytorch_conv1d here due to the tied-weight special case.
  auto status = clblast::Gemm<nnopt_storage_t>(
      clblast::Layout::kRowMajor,
      clblast::Transpose::kNo,
      clblast::Transpose::kYes,
      M, N, K,
      (nnopt_storage_t)1, hidden_normed, 0, K,
      w, 0, K,
      (nnopt_storage_t)0, logits, 0, N,
      &queue, nullptr);
  if (status != clblast::StatusCode::kSuccess) {
    printf("GEMM_ERROR={\"status\":%d,\"layer\":\"backbone_lm_head\"}\n", (int)status);
    fflush(stdout);
    NNOPT_ERROR("Backbone::lm_head: clblast::Gemm failed");
    clReleaseMemObject(logits);
    return nullptr;
  }

  (void)ctx;
  return logits;
}
