// Reference: /Users/alazarshenkute/.nnopt/ref_venvs/env_stable/lib/python3.12/site-packages/torch/nn/modules/sparse.py:109-114 Embedding.forward
// Implements torch.nn.functional.embedding gather for inference (no max_norm, no scale_grad_by_freq).

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "model_config.h"

#include <CL/cl.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace {

struct EmbeddingOpState {
  bool initialized = false;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
};

EmbeddingOpState& state() {
  static EmbeddingOpState s;
  return s;
}

bool ensure_initialized(OpenCLContext& cl_ctx) {
  auto& s = state();
  if (s.initialized) return true;

  s.program = cl_ctx.build_program_from_file("kernels/embedding.cl");  // PROGRAM-INIT-OK
  if (!s.program) {
    NNOPT_ERROR("op_embedding: failed to build kernels/embedding.cl");
    return false;
  }

  cl_int err = CL_SUCCESS;
  s.kernel = clCreateKernel(s.program, "embedding_forward", &err);
  if (err != CL_SUCCESS || !s.kernel) {
    NNOPT_ERROR_FMT("op_embedding: clCreateKernel(embedding_forward) failed (%d)", (int)err);
    return false;
  }

  s.initialized = true;
  return true;
}

}  // namespace

extern "C" cl_mem op_Embedding(OpenCLContext& cl_ctx,
                               Weights& weights,
                               cl_command_queue queue,
                               cl_mem input_ids_i32,
                               int num_tokens,
                               int hidden_size,
                               const char* weight_key_wte) {
  if (!ensure_initialized(cl_ctx)) return nullptr;

  if (!queue || !input_ids_i32 || !weight_key_wte) {
    NNOPT_ERROR("op_Embedding: null queue/input/weight_key");
    return nullptr;
  }

  cl_mem wte = weights.get_buffer(std::string(weight_key_wte));
  if (!wte) {
    NNOPT_ERROR_FMT("op_Embedding: missing weight %s", weight_key_wte);
    return nullptr;
  }

  cl_int err = CL_SUCCESS;
  const size_t out_elems = (size_t)num_tokens * (size_t)hidden_size;
  cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                              out_elems * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !out) {
    NNOPT_ERROR_FMT("op_Embedding: clCreateBuffer(out) failed (%d)", (int)err);
    return nullptr;
  }

  auto cleanup = [&]() -> cl_mem {
    if (out) {
      clReleaseMemObject(out);
      out = nullptr;
    }
    return nullptr;
  };

  cl_kernel k = state().kernel;
  if (!set_arg_checked(k, 0, sizeof(cl_mem), &input_ids_i32, "token_ids")) return cleanup();
  if (!set_arg_checked(k, 1, sizeof(cl_mem), &wte, "wte")) return cleanup();
  if (!set_arg_checked(k, 2, sizeof(cl_mem), &out, "out")) return cleanup();
  if (!set_arg_checked(k, 3, sizeof(int), &num_tokens, "num_tokens")) return cleanup();
  if (!set_arg_checked(k, 4, sizeof(int), &hidden_size, "hidden_size")) return cleanup();

  // IMPORTANT: embedding_forward reads token_ids[t] and uses it to index wte.
  // If token_ids contains an out-of-range value (e.g. tokenizer drift or OOB),
  // the kernel will read wte out-of-bounds and can crash the process on some
  // drivers. Thread vocab_size through so the kernel can guard.
  const int vocab_size = MODEL_CONFIG::VOCAB_SIZE;
  if (!set_arg_checked(k, 5, sizeof(int), &vocab_size, "vocab_size")) return cleanup();

  // 2D dispatch: (tokens, hidden)
  const size_t gws[2] = {(size_t)num_tokens, (size_t)hidden_size};
  err = clEnqueueNDRangeKernel(queue, k, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("op_Embedding: clEnqueueNDRangeKernel failed (%d)", (int)err);
    return cleanup();
  }

  return out;
}
