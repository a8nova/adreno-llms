// Reference: model_info/transformers_src/modeling_lfm2.py:320-518 (Lfm2Model.forward, Lfm2DecoderLayer.forward, Lfm2ForCausalLM.forward)

#include "model.h"

#include "benchmark.h"
#include "debug_utils.h"
#include "kernel_profiler.h"
#include "model_config.h"
#include "utils.h"

#include <clblast.h>

#include <cmath>
#include <cstdio>
#include <iostream>

Model::Model(OpenCLContext& cl_ctx, Weights& weights) : cl_ctx_(cl_ctx), weights_(weights) {
  NNOPT_CHECKPOINT("Model constructor");
}

Model::~Model() {
  if (argmax_block_kernel_) clReleaseKernel(argmax_block_kernel_);
  if (argmax_final_kernel_) clReleaseKernel(argmax_final_kernel_);
  if (argmax_program_)      clReleaseProgram(argmax_program_);
  if (argmax_partials_val_) clReleaseMemObject(argmax_partials_val_);
  if (argmax_partials_idx_) clReleaseMemObject(argmax_partials_idx_);
  if (argmax_out_idx_)      clReleaseMemObject(argmax_out_idx_);
  if (buf_logits_)          clReleaseMemObject(buf_logits_);
  if (utils_program_) clReleaseProgram(utils_program_);
}

static cl_mem ensure_logits_buf(cl_context ctx, cl_mem& buf, int& cap, int seq_len) {
  if (seq_len <= cap && buf) return buf;
  if (buf) { clReleaseMemObject(buf); buf = nullptr; }
  cl_int err = CL_SUCCESS;
  buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                       (size_t)seq_len * MODEL_CONFIG::VOCAB_SIZE * sizeof(nnopt_storage_t),
                       nullptr, &err);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("logits alloc: %d", err); return nullptr; }
  cap = seq_len;
  return buf;
}

bool Model::ensure_argmax_resources_() {
  if (argmax_initialized_) return true;

  argmax_program_ = cl_ctx_.build_program_from_file("kernels/argmax.cl");
  if (!argmax_program_) {
    NNOPT_ERROR("Failed to build kernels/argmax.cl");
    return false;
  }
  cl_int err = CL_SUCCESS;
  argmax_block_kernel_ = clCreateKernel(argmax_program_, "argmax_block", &err);
  if (err != CL_SUCCESS || !argmax_block_kernel_) { NNOPT_ERROR_FMT("argmax_block create: %d", err); return false; }
  argmax_final_kernel_ = clCreateKernel(argmax_program_, "argmax_final", &err);
  if (err != CL_SUCCESS || !argmax_final_kernel_) { NNOPT_ERROR_FMT("argmax_final create: %d", err); return false; }

  // Vocab is 65536 → 64 chunks × 1024 fp16 each = 64 partials.
  const int num_wg = 64;
  argmax_partials_val_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, sizeof(float) * num_wg, nullptr, &err);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("argmax partials_val alloc: %d", err); return false; }
  argmax_partials_idx_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, sizeof(int) * num_wg, nullptr, &err);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("argmax partials_idx alloc: %d", err); return false; }
  argmax_out_idx_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, sizeof(int), nullptr, &err);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("argmax out_idx alloc: %d", err); return false; }

  argmax_initialized_ = true;
  return true;
}

bool Model::initialize() {
  NNOPT_CHECKPOINT("Model::initialize() started");

  utils_program_ = cl_ctx_.build_program_from_file("kernels/utils.cl");
  if (!utils_program_) {
    NNOPT_ERROR("Failed to build kernels/utils.cl");
    return false;
  }

  embedding_.reset(new Embedding(cl_ctx_, weights_));
  if (!embedding_->initialize()) {
    NNOPT_ERROR("embedding.initialize() FAILED");
    return false;
  }
  NNOPT_LAYER_INIT("embedding");

  embedding_norm_.reset(new OperatorNorm(cl_ctx_, weights_, /*prefix=*/"model.embedding_norm", /*layer_idx=*/-1));
  if (!embedding_norm_->initialize()) {
    NNOPT_ERROR("embedding_norm.initialize() FAILED");
    return false;
  }
  NNOPT_LAYER_INIT("embedding_norm");

  for (int i = 0; i < MODEL_CONFIG::NUM_LAYERS; ++i) {
    // operator_norm
    {
      operator_norm_[i].reset(new OperatorNorm(
          cl_ctx_, weights_, /*prefix=*/"model.layers." + std::to_string(i) + ".operator_norm", /*layer_idx=*/i));
      if (!operator_norm_[i]->initialize()) {
        NNOPT_ERROR_FMT("operator_norm_[%d].initialize() FAILED", i);
        return false;
      }
      NNOPT_LAYER_INIT_FMT("operator_norm_%d", i);
    }

    // ffn_norm (same RMSNorm op; different weight)
    {
      ffn_norm_[i].reset(new OperatorNorm(
          cl_ctx_, weights_, /*prefix=*/"model.layers." + std::to_string(i) + ".ffn_norm", /*layer_idx=*/i));
      if (!ffn_norm_[i]->initialize()) {
        NNOPT_ERROR_FMT("ffn_norm_[%d].initialize() FAILED", i);
        return false;
      }
      NNOPT_LAYER_INIT_FMT("ffn_norm_%d", i);
    }

    // attention or conv
    if (layer_has_attention(i)) {
      const std::string pfx = "model.layers." + std::to_string(i) + ".self_attn";
      attn_[i].reset(new Attention(cl_ctx_, weights_, i, pfx));
      if (!attn_[i]->initialize()) {
        NNOPT_ERROR_FMT("attn_[%d].initialize() FAILED", i);
        return false;
      }
      NNOPT_LAYER_INIT_FMT("attn_%d", i);
    }
    if (layer_has_convolution(i)) {
      const std::string pfx = "model.layers." + std::to_string(i) + ".conv";
      conv_[i].reset(new Convolution(cl_ctx_, weights_, /*prefix=*/pfx, /*layer_idx=*/i));
      if (!conv_[i]->initialize()) {
        NNOPT_ERROR_FMT("conv_[%d].initialize() FAILED", i);
        return false;
      }
      NNOPT_LAYER_INIT_FMT("conv_%d", i);
    }

    // mlp
    {
      const std::string pfx = "model.layers." + std::to_string(i) + ".feed_forward";
      mlp_[i].reset(new Mlp(cl_ctx_, weights_, /*prefix=*/pfx, /*layer_idx=*/i));
      if (!mlp_[i]->initialize()) {
        NNOPT_ERROR_FMT("mlp_[%d].initialize() FAILED", i);
        return false;
      }
      NNOPT_LAYER_INIT_FMT("mlp_%d", i);
    }
  }

  NNOPT_CHECKPOINT("Model::initialize() complete");
  return true;
}

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids) { return forward(input_ids, /*start_pos=*/0); }

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids, int start_pos) {
  NNOPT_CHECKPOINT("forward() started");

  const int seq_len = (int)input_ids.size();
  if (seq_len <= 0) return {};

  cl_command_queue queue = cl_ctx_.queue();

  // Embed
  cl_mem hidden = embedding_->forward(queue, (const int*)input_ids.data(), seq_len);
  if (!hidden) {
    NNOPT_ERROR("embedding forward failed");
    return {};
  }
  NNOPT_LAYER_CHECK("embedding", queue, hidden, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

  // NOTE: In LFM2/LFM2.5, `embedding_norm` is misleadingly named — per
  // Lfm2Model.forward (modeling_lfm2.py:521-532), the decoder loop runs on the
  // RAW embed_tokens output, and `embedding_norm` is applied AFTER the loop as
  // the final pre-lm_head layernorm. Do NOT apply it here.

  for (int i = 0; i < MODEL_CONFIG::NUM_LAYERS; ++i) {
    // operator_norm — borrowed
    cl_mem op_norm = operator_norm_[i]->forward(queue, hidden, seq_len);
    NNOPT_LAYER_CHECK_FMT(
        "operator_norm_%d", i, queue, op_norm, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

    cl_mem op_out = nullptr;
    if (layer_has_attention(i)) {
      op_out = attn_[i]->forward(queue, op_norm, seq_len, start_pos);
      NNOPT_LAYER_CHECK_FMT("attn_%d", i, queue, op_out, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
    } else if (layer_has_convolution(i)) {
      op_out = conv_[i]->forward(queue, op_norm, seq_len, start_pos);
      NNOPT_LAYER_CHECK_FMT("conv_%d", i, queue, op_out, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
    } else {
      NNOPT_ERROR_FMT("layer %d has neither attention nor convolution", i);
      return {};
    }
    // op_norm and op_out are BORROWED — do not release.

    // hidden += op_out (in place; op_out persistent)
    element_add_inplace(queue, utils_program_, hidden, op_out, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

    cl_mem ffn_norm = ffn_norm_[i]->forward(queue, hidden, seq_len);
    NNOPT_LAYER_CHECK_FMT("ffn_norm_%d", i, queue, ffn_norm, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

    cl_mem mlp_out = mlp_[i]->forward(queue, ffn_norm, seq_len);
    NNOPT_LAYER_CHECK_FMT("mlp_%d", i, queue, mlp_out, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

    element_add_inplace(queue, utils_program_, hidden, mlp_out, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
  }

  // Final pre-lm_head norm — borrowed.
  hidden = embedding_norm_->forward(queue, hidden, seq_len);
  NNOPT_LAYER_CHECK("embedding_norm", queue, hidden, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
  NNOPT_LAYER_CHECK("final_norm",     queue, hidden, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

  cl_int err = CL_SUCCESS;
  cl_mem W = weights_.get_buffer("model.embed_tokens.weight");
  if (!W) { NNOPT_ERROR("lm_head weight not found"); return {}; }
  cl_mem logits_buf = ensure_logits_buf(cl_ctx_.context(), buf_logits_, buf_logits_seq_capacity_, seq_len);
  if (!logits_buf) return {};

  if (!pytorch_linear(queue, seq_len, MODEL_CONFIG::VOCAB_SIZE, MODEL_CONFIG::HIDDEN_SIZE,
                      hidden, W, logits_buf)) {
    NNOPT_ERROR("pytorch_linear(lm_head) failed");
    return {};
  }
  NNOPT_LAYER_CHECK("lm_head", queue, logits_buf, (size_t)seq_len * MODEL_CONFIG::VOCAB_SIZE);

  std::vector<float> logits(MODEL_CONFIG::VOCAB_SIZE);
  const size_t row_bytes    = (size_t)MODEL_CONFIG::VOCAB_SIZE * sizeof(nnopt_storage_t);
  const size_t offset_bytes = (size_t)(seq_len - 1) * row_bytes;
  std::vector<nnopt_storage_t> row_storage(MODEL_CONFIG::VOCAB_SIZE);
  err = clEnqueueReadBuffer(queue, logits_buf, CL_TRUE, offset_bytes, row_bytes, row_storage.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("clEnqueueReadBuffer(logits) failed: %d", err);
    logits.clear();
  } else {
#ifdef NNOPT_USE_FP16
    for (int i = 0; i < MODEL_CONFIG::VOCAB_SIZE; ++i) logits[i] = nnopt_f16_to_f32((uint16_t)row_storage[i]);
#else
    for (int i = 0; i < MODEL_CONFIG::VOCAB_SIZE; ++i) logits[i] = (float)row_storage[i];
#endif
  }
  NNOPT_CHECKPOINT("forward() complete");
  return logits;
}

std::vector<float> Model::forward_decode(int32_t token_id, int start_pos) {
  NNOPT_CHECKPOINT("forward_decode() started");
  NNOPT_TODO("Model::forward_decode (delegating to prefill path until wired)");
  return forward(std::vector<int32_t>{token_id}, start_pos);
}

// Greedy fast path. Mirrors forward() up to and including lm_head, then
// dispatches the 2-pass GPU argmax kernel and reads back a single int32.
// Saves ~100 KB readback + ~65k host fp16->fp32 + host argmax per call.
int32_t Model::forward_greedy(const std::vector<int32_t>& input_ids, int start_pos) {
  if (!ensure_argmax_resources_()) return -1;

  const int seq_len = (int)input_ids.size();
  if (seq_len <= 0) return -1;

  cl_command_queue queue = cl_ctx_.queue();

  cl_mem hidden = embedding_->forward(queue, (const int*)input_ids.data(), seq_len);
  if (!hidden) return -1;

  for (int i = 0; i < MODEL_CONFIG::NUM_LAYERS; ++i) {
    cl_mem op_norm = operator_norm_[i]->forward(queue, hidden, seq_len);
    cl_mem op_out = nullptr;
    if (layer_has_attention(i)) {
      op_out = attn_[i]->forward(queue, op_norm, seq_len, start_pos);
    } else if (layer_has_convolution(i)) {
      op_out = conv_[i]->forward(queue, op_norm, seq_len, start_pos);
    } else {
      return -1;
    }
    // op_norm and op_out are BORROWED — do not release.

    element_add_inplace(queue, utils_program_, hidden, op_out, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

    cl_mem ffn_norm = ffn_norm_[i]->forward(queue, hidden, seq_len);
    cl_mem mlp_out = mlp_[i]->forward(queue, ffn_norm, seq_len);
    // ffn_norm and mlp_out are BORROWED — do not release.

    element_add_inplace(queue, utils_program_, hidden, mlp_out, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
  }

  hidden = embedding_norm_->forward(queue, hidden, seq_len);

  cl_int err = CL_SUCCESS;
  cl_mem W = weights_.get_buffer("model.embed_tokens.weight");
  if (!W) return -1;
  cl_mem logits_buf = ensure_logits_buf(cl_ctx_.context(), buf_logits_, buf_logits_seq_capacity_, seq_len);
  if (!logits_buf) return -1;

  if (!pytorch_linear(queue, seq_len, MODEL_CONFIG::VOCAB_SIZE, MODEL_CONFIG::HIDDEN_SIZE,
                      hidden, W, logits_buf)) {
    return -1;
  }

  // Argmax over the LAST row only. logits_buf is [seq_len, vocab]; we use
  // a sub-buffer view at offset (seq_len-1)*vocab. clCreateSubBuffer requires
  // alignment to CL_DEVICE_MEM_BASE_ADDR_ALIGN (1024 bits = 128 bytes on
  // most Adreno). offset = (seq_len-1) * 65536 * 2 bytes. seq_len-1 is 0
  // for decode (single-token), so offset is 0 — natural alignment. For
  // prefill (seq_len>1) we still pass logits_buf at offset 0 but only
  // operate on the last row.
  //
  // Rather than building a sub-buffer, pass the logits_buf and an offset
  // index. The kernel takes V (vocab) and a chunk_start; we set chunk_start
  // by re-binding the pointer with an offset via clEnqueueCopyBuffer would
  // be wasteful. Simplest: pass the logits_buf with cl_buffer_region for
  // the sub-buffer.
  cl_buffer_region region = {
      (size_t)(seq_len - 1) * MODEL_CONFIG::VOCAB_SIZE * sizeof(nnopt_storage_t),
      (size_t)MODEL_CONFIG::VOCAB_SIZE * sizeof(nnopt_storage_t)
  };
  cl_mem last_row = clCreateSubBuffer(logits_buf, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("argmax: sub-buffer create failed: %d (off=%zu)", err, region.origin);
    return -1;
  }

  const int V = MODEL_CONFIG::VOCAB_SIZE;
  const int CHUNK = 1024;
  const int num_wg = (V + CHUNK - 1) / CHUNK;

  clSetKernelArg(argmax_block_kernel_, 0, sizeof(cl_mem), &last_row);
  clSetKernelArg(argmax_block_kernel_, 1, sizeof(cl_mem), &argmax_partials_val_);
  clSetKernelArg(argmax_block_kernel_, 2, sizeof(cl_mem), &argmax_partials_idx_);
  clSetKernelArg(argmax_block_kernel_, 3, sizeof(int),    &V);
  size_t gws_b = (size_t)num_wg * 64;
  size_t lws_b = 64;
  err = clEnqueueNDRangeKernel(queue, argmax_block_kernel_, 1, nullptr, &gws_b, &lws_b, 0, nullptr,
                               KernelProfiler::event_for("argmax_block"));
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("argmax_block dispatch: %d", err);
    clReleaseMemObject(last_row);
    return -1;
  }

  clSetKernelArg(argmax_final_kernel_, 0, sizeof(cl_mem), &argmax_partials_val_);
  clSetKernelArg(argmax_final_kernel_, 1, sizeof(cl_mem), &argmax_partials_idx_);
  clSetKernelArg(argmax_final_kernel_, 2, sizeof(cl_mem), &argmax_out_idx_);
  clSetKernelArg(argmax_final_kernel_, 3, sizeof(int),    &num_wg);
  size_t gws_f = 64;
  size_t lws_f = 64;
  err = clEnqueueNDRangeKernel(queue, argmax_final_kernel_, 1, nullptr, &gws_f, &lws_f, 0, nullptr,
                               KernelProfiler::event_for("argmax_final"));
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("argmax_final dispatch: %d", err);
    clReleaseMemObject(last_row);
    return -1;
  }

  int32_t result = -1;
  err = clEnqueueReadBuffer(queue, argmax_out_idx_, CL_TRUE, 0, sizeof(int32_t), &result, 0, nullptr, nullptr);

  clReleaseMemObject(last_row);  // sub-buffer is per-call (offset varies for prefill)
  if (err != CL_SUCCESS) return -1;
  return result;
}

std::vector<int32_t> Model::generate(const std::vector<int32_t>& prompt_ids, int max_new_tokens, SamplerConfig sampler_config, TokenCallback on_token) {
  NNOPT_CHECKPOINT("generate() started");
  Sampler sampler(sampler_config);
  auto ids = prompt_ids;

  if (prompt_ids.empty()) {
    NNOPT_ERROR("generate(): empty prompt");
    return ids;
  }

  // GPU argmax fast path: pure greedy means temperature<=0 and rep_penalty==1.
  // Top-k / top-p are inapplicable at temperature=0 anyway. Anything else
  // falls back to the host-side sampler (which needs the full logits row).
  const bool greedy_fast = (sampler_config.temperature <= 0.0f)
                        && (sampler_config.repetition_penalty == 1.0f);

  std::vector<int32_t> generated;
  int next_token;
  if (greedy_fast) {
    next_token = forward_greedy(prompt_ids, /*start_pos=*/0);
    if (next_token < 0) { NNOPT_ERROR("prefill forward_greedy failed"); return ids; }
  } else {
    auto logits = forward(prompt_ids, /*start_pos=*/0);
    if (logits.empty()) { NNOPT_ERROR("prefill forward() returned empty logits"); return ids; }
    next_token = sampler.sample(logits, generated);
  }
  ids.push_back(next_token);
  generated.push_back(next_token);
  NNOPT_BENCH_FIRST_TOKEN();
  if (on_token) on_token(next_token, ids);
  if (sampler_config.eos_token_id >= 0 && next_token == sampler_config.eos_token_id) {
    NNOPT_CHECKPOINT("generate() complete (eos at first token)");
    return ids;
  }

  for (int i = 1; i < max_new_tokens; i++) {
    const int pos = (int)prompt_ids.size() + (i - 1);
    if (greedy_fast) {
      next_token = forward_greedy(std::vector<int32_t>{next_token}, pos);
      if (next_token < 0) { NNOPT_ERROR("decode forward_greedy failed"); break; }
    } else {
      auto logits = forward(std::vector<int32_t>{next_token}, pos);
      if (logits.empty()) { NNOPT_ERROR("decode forward() returned empty logits"); break; }
      next_token = sampler.sample(logits, generated);
    }
    ids.push_back(next_token);
    generated.push_back(next_token);
    if (on_token) on_token(next_token, ids);

    if (sampler_config.eos_token_id >= 0 && next_token == sampler_config.eos_token_id) break;
  }

  NNOPT_CHECKPOINT("generate() complete");
  return ids;
}
