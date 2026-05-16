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

  // ── cl_qcom_recordable_queues setup (opt-in via NNOPT_RECORD=1). The
  // first forward_greedy decode step will both run the layers live AND
  // capture them into a recording on record_queue_. Subsequent decode
  // steps replay the recording with start_pos / seq_k arg overrides.
  if (const char* r = std::getenv("NNOPT_RECORD"); r && r[0] == '1') {
    if (cl_ctx_.has_recordable_queues()) {
      record_queue_ = cl_ctx_.create_recordable_queue();
      record_enabled_ = (record_queue_ != nullptr);
      if (record_enabled_) {
        collect_record_args_();
        std::cerr << "RecordQ: enabled (" << rec_start_pos_args_.size()
                  << " start_pos args, " << rec_seq_k_args_.size()
                  << " seq_k args)" << std::endl;
      }
    } else {
      std::cerr << "RecordQ: NNOPT_RECORD=1 set but extension unavailable" << std::endl;
    }
  }

  NNOPT_CHECKPOINT("Model::initialize() complete");
  return true;
}

void Model::collect_record_args_() {
  // For each attention layer at decode (seq_q=1):
  //   rope_kernel    arg 8 = start_pos
  //   kv_write_kernel arg 2 = start_pos
  //   scores_kernel   arg 4 = seq_k
  //   softmax_kernel  arg 2 = seq_k
  //   out_kernel      arg 4 = seq_k
  rec_start_pos_args_.clear();
  rec_seq_k_args_.clear();
  for (int i : ATTENTION_LAYER_INDICES) {
    if (!attn_[i]) continue;
    if (cl_kernel k = attn_[i]->rope_kernel())     rec_start_pos_args_.push_back({k, 8});
    if (cl_kernel k = attn_[i]->kv_write_kernel()) rec_start_pos_args_.push_back({k, 2});
    if (cl_kernel k = attn_[i]->scores_kernel())   rec_seq_k_args_.push_back({k, 4});
    if (cl_kernel k = attn_[i]->softmax_kernel())  rec_seq_k_args_.push_back({k, 2});
    if (cl_kernel k = attn_[i]->out_kernel())      rec_seq_k_args_.push_back({k, 4});
  }
}

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids) { return forward(input_ids, /*start_pos=*/0); }

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids, int start_pos) {
  NNOPT_CHECKPOINT("forward() started");

  const int seq_len = (int)input_ids.size();
  if (seq_len <= 0) return {};

  // Under NNOPT_QUANT=int8 the per-row symmetric weights are int8, but
  // CLBlast Hgemm at M>1 can't ingest int8 (returns err=-1010). The int8
  // image-path GEMV only fires at M=1 (see utils.cpp::run_gemv_m1_image_int8).
  // So we run prefill as a sequence of single-token forwards — each one
  // takes the M=1 fast path. Slow but correct. Matches the SmolLM2 port.
  if (seq_len > 1) {
    if (const char* q = std::getenv("NNOPT_QUANT"); q && (std::string(q) == "int8" || std::string(q) == "q4")) {
      std::vector<float> last_logits;
      for (int i = 0; i < seq_len; ++i) {
        last_logits = forward(std::vector<int32_t>{input_ids[i]}, start_pos + i);
      }
      return last_logits;
    }
  }

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
  // Prefer the int8 lm_head alias when present (NNOPT_QUANT=int8 + quantize
  // script run with --emit-lm-head-int8). Falls back to the tied
  // embed_tokens.weight (fp16 path) under fp16 builds or if the alias is
  // missing.
  cl_mem W = weights_.has_tensor("lm_head.weight")
               ? weights_.get_buffer("lm_head.weight")
               : weights_.get_buffer("model.embed_tokens.weight");
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

  // Same int8 prefill-via-decode-loop as the dense forward() above. The
  // CLBlast Hgemm path used at M>1 cannot consume int8 weights. Run prefill
  // tokens one-by-one through the M=1 int8 fast path, then argmax once.
  if (seq_len > 1) {
    if (const char* q = std::getenv("NNOPT_QUANT"); q && (std::string(q) == "int8" || std::string(q) == "q4")) {
      int32_t last_token = -1;
      for (int i = 0; i < seq_len; ++i) {
        last_token = forward_greedy(std::vector<int32_t>{input_ids[i]}, start_pos + i);
        if (last_token < 0) return -1;
      }
      return last_token;
    }
  }

  cl_command_queue queue = cl_ctx_.queue();

  cl_mem hidden = embedding_->forward(queue, (const int*)input_ids.data(), seq_len);
  if (!hidden) return -1;

  // Record/replay path (NNOPT_RECORD=1, seq_len==1 only — recording is only
  // valid for the steady-state decode geometry where every kernel-arg layout
  // is identical across calls and only start_pos / seq_k mutate).
  bool replayed = false;
  if (record_enabled_ && recording_built_ && seq_len == 1) {
    cur_start_pos_ = start_pos;
    cur_seq_k_     = start_pos + 1;
    std::vector<cl_array_arg_qcom> args;
    args.reserve(rec_start_pos_args_.size() + rec_seq_k_args_.size());
    for (auto& p : rec_start_pos_args_) args.push_back({p.kernel, p.arg_indx, sizeof(int), &cur_start_pos_});
    for (auto& p : rec_seq_k_args_)     args.push_back({p.kernel, p.arg_indx, sizeof(int), &cur_seq_k_});
    cl_int rerr = cl_ctx_.enqueue_recording(queue, recording_, args.size(), args.data());
    if (rerr != CL_SUCCESS) {
      NNOPT_ERROR_FMT("RecordQ: enqueue_recording err=%d — disabling, falling back to live dispatch", rerr);
      record_enabled_ = false;
    } else {
      replayed = true;
    }
  }

  if (!replayed) {
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
  }

  // First-call recording build: AFTER the live dispatch produced this step's
  // logits, re-issue the layer block to the record_queue_ so the recording
  // captures the dispatch sequence. End the recording; subsequent decode
  // steps replay via enqueue_recording. The record_queue_ dispatches don't
  // execute (recording-only mode) — they only build the recording.
  if (record_enabled_ && !recording_built_ && seq_len == 1) {
    recording_ = cl_ctx_.new_recording(record_queue_);
    if (!recording_) {
      std::cerr << "RecordQ: new_recording failed; disabling" << std::endl;
      record_enabled_ = false;
    } else {
      bool rec_ok = true;
      for (int i = 0; i < MODEL_CONFIG::NUM_LAYERS && rec_ok; ++i) {
        cl_mem op_norm = operator_norm_[i]->forward(record_queue_, hidden, 1);
        if (!op_norm) { rec_ok = false; break; }
        cl_mem op_out = nullptr;
        if (layer_has_attention(i)) {
          op_out = attn_[i]->forward(record_queue_, op_norm, 1, start_pos);
        } else if (layer_has_convolution(i)) {
          op_out = conv_[i]->forward(record_queue_, op_norm, 1, start_pos);
        } else { rec_ok = false; break; }
        if (!op_out) { rec_ok = false; break; }
        element_add_inplace(record_queue_, utils_program_, hidden, op_out, (size_t)MODEL_CONFIG::HIDDEN_SIZE);
        cl_mem ffn_norm = ffn_norm_[i]->forward(record_queue_, hidden, 1);
        cl_mem mlp_out  = mlp_[i]->forward(record_queue_, ffn_norm, 1);
        if (!ffn_norm || !mlp_out) { rec_ok = false; break; }
        element_add_inplace(record_queue_, utils_program_, hidden, mlp_out, (size_t)MODEL_CONFIG::HIDDEN_SIZE);
      }
      cl_int eerr = cl_ctx_.end_recording(recording_);
      if (!rec_ok || eerr != CL_SUCCESS) {
        std::cerr << "RecordQ: recording build failed (ok=" << rec_ok << " end_err=" << eerr << "); disabling" << std::endl;
        cl_ctx_.release_recording(recording_);
        recording_ = nullptr;
        record_enabled_ = false;
      } else {
        recording_built_ = true;
        std::cerr << "RecordQ: recording built (16 layers, start_pos@record=" << start_pos << ")" << std::endl;
      }
    }
  }

  hidden = embedding_norm_->forward(queue, hidden, seq_len);

  cl_int err = CL_SUCCESS;
  // Prefer the int8 lm_head alias when present (NNOPT_QUANT=int8 + quantize
  // script run with --emit-lm-head-int8). Falls back to the tied
  // embed_tokens.weight (fp16 path) under fp16 builds or if the alias is
  // missing.
  cl_mem W = weights_.has_tensor("lm_head.weight")
               ? weights_.get_buffer("lm_head.weight")
               : weights_.get_buffer("model.embed_tokens.weight");
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

// Step 10: chained-decode forward. Mirrors forward_greedy at seq_len=1, but
// (a) reads the input token from argmax_out_idx_ via embedding_->forward_from_device_token,
// (b) skips the per-call sub-buffer (offset=0 at seq_len=1, so logits_buf_
//     itself is the row), and (c) does NOT block on a host readback.
// Caller pipelines an async readback of argmax_out_buffer().
bool Model::forward_greedy_chained_enqueue(int start_pos) {
  if (!ensure_argmax_resources_()) return false;

  const int seq_len = 1;
  cl_command_queue queue = cl_ctx_.queue();

  cl_mem hidden = embedding_->forward_from_device_token(queue, argmax_out_idx_, /*offset=*/0);
  if (!hidden) { NNOPT_ERROR("chained: embedding forward failed"); return false; }

  for (int i = 0; i < MODEL_CONFIG::NUM_LAYERS; ++i) {
    cl_mem op_norm = operator_norm_[i]->forward(queue, hidden, seq_len);
    cl_mem op_out = nullptr;
    if (layer_has_attention(i)) {
      op_out = attn_[i]->forward(queue, op_norm, seq_len, start_pos);
    } else if (layer_has_convolution(i)) {
      op_out = conv_[i]->forward(queue, op_norm, seq_len, start_pos);
    } else {
      NNOPT_ERROR_FMT("chained: layer %d has no op", i);
      return false;
    }
    if (!op_out) { NNOPT_ERROR_FMT("chained: op forward null at layer %d", i); return false; }

    element_add_inplace(queue, utils_program_, hidden, op_out, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

    cl_mem ffn_norm = ffn_norm_[i]->forward(queue, hidden, seq_len);
    cl_mem mlp_out = mlp_[i]->forward(queue, ffn_norm, seq_len);
    if (!mlp_out) { NNOPT_ERROR_FMT("chained: mlp null at layer %d", i); return false; }

    element_add_inplace(queue, utils_program_, hidden, mlp_out, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
  }

  hidden = embedding_norm_->forward(queue, hidden, seq_len);
  if (!hidden) { NNOPT_ERROR("chained: embedding_norm failed"); return false; }

  // Prefer the int8 lm_head alias when present (NNOPT_QUANT=int8 + quantize
  // script run with --emit-lm-head-int8). Falls back to the tied
  // embed_tokens.weight (fp16 path) under fp16 builds or if the alias is
  // missing.
  cl_mem W = weights_.has_tensor("lm_head.weight")
               ? weights_.get_buffer("lm_head.weight")
               : weights_.get_buffer("model.embed_tokens.weight");
  if (!W) return false;
  cl_mem logits_buf = ensure_logits_buf(cl_ctx_.context(), buf_logits_, buf_logits_seq_capacity_, seq_len);
  if (!logits_buf) return false;

  if (!pytorch_linear(queue, seq_len, MODEL_CONFIG::VOCAB_SIZE, MODEL_CONFIG::HIDDEN_SIZE,
                      hidden, W, logits_buf)) {
    return false;
  }

  // At seq_len=1, the (only) row begins at offset 0, so logits_buf is itself
  // the "last row" — no sub-buffer needed.
  const int V = MODEL_CONFIG::VOCAB_SIZE;
  const int CHUNK = 1024;
  const int num_wg = (V + CHUNK - 1) / CHUNK;

  cl_int err = clSetKernelArg(argmax_block_kernel_, 0, sizeof(cl_mem), &logits_buf);
  err |= clSetKernelArg(argmax_block_kernel_, 1, sizeof(cl_mem), &argmax_partials_val_);
  err |= clSetKernelArg(argmax_block_kernel_, 2, sizeof(cl_mem), &argmax_partials_idx_);
  err |= clSetKernelArg(argmax_block_kernel_, 3, sizeof(int),    &V);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("chained: argmax_block setArgs: %d", err); return false; }
  size_t gws_b = (size_t)num_wg * 64;
  size_t lws_b = 64;
  err = clEnqueueNDRangeKernel(queue, argmax_block_kernel_, 1, nullptr, &gws_b, &lws_b, 0, nullptr,
                               KernelProfiler::event_for("argmax_block_chained"));
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("chained: argmax_block dispatch: %d", err); return false; }

  err = clSetKernelArg(argmax_final_kernel_, 0, sizeof(cl_mem), &argmax_partials_val_);
  err |= clSetKernelArg(argmax_final_kernel_, 1, sizeof(cl_mem), &argmax_partials_idx_);
  err |= clSetKernelArg(argmax_final_kernel_, 2, sizeof(cl_mem), &argmax_out_idx_);
  err |= clSetKernelArg(argmax_final_kernel_, 3, sizeof(int),    &num_wg);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("chained: argmax_final setArgs: %d", err); return false; }
  size_t gws_f = 64;
  size_t lws_f = 64;
  err = clEnqueueNDRangeKernel(queue, argmax_final_kernel_, 1, nullptr, &gws_f, &lws_f, 0, nullptr,
                               KernelProfiler::event_for("argmax_final_chained"));
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("chained: argmax_final dispatch: %d", err); return false; }

  return true;
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

  // Step 10 fast path: pipelined chained decode. Embedding reads its single
  // input token directly from argmax_out_idx_ (written by the previous
  // forward); the host readback runs asynchronously and we wait on it one
  // iteration LATER, so the readback latency is hidden behind the next
  // forward's enqueue. After prefill, argmax_out_idx_ already holds the
  // first generated token, so iteration i=1 generates the second token.
  //
  // Skipped under NNOPT_RECORD=1: chained path uses a different forward
  // (forward_greedy_chained_enqueue) that doesn't have the recording build
  // wired up. Recording lives only in forward_greedy() for now; the two are
  // mutually exclusive for the current experiment.
  if (greedy_fast && max_new_tokens > 1 && !record_enabled_) {
    cl_command_queue queue = cl_ctx_.queue();
    int32_t host_int[2] = {0, 0};
    cl_event readback_event[2] = {nullptr, nullptr};
    bool slot_pending[2] = {false, false};
    int prev_slot = -1;
    bool eos_seen = false;

    for (int i = 1; i < max_new_tokens && !eos_seen; i++) {
      const int pos = (int)prompt_ids.size() + (i - 1);
      const int cur_slot = i & 1;

      if (!forward_greedy_chained_enqueue(pos)) {
        NNOPT_ERROR("chained decode forward failed");
        break;
      }

      cl_int rerr = clEnqueueReadBuffer(queue, argmax_out_idx_, CL_FALSE, 0,
                                        sizeof(int32_t), &host_int[cur_slot],
                                        0, nullptr, &readback_event[cur_slot]);
      if (rerr != CL_SUCCESS) {
        NNOPT_ERROR_FMT("chained async readback enqueue failed (%d)", (int)rerr);
        break;
      }
      slot_pending[cur_slot] = true;
      clFlush(queue);

      if (prev_slot >= 0 && slot_pending[prev_slot]) {
        clWaitForEvents(1, &readback_event[prev_slot]);
        clReleaseEvent(readback_event[prev_slot]);
        readback_event[prev_slot] = nullptr;
        slot_pending[prev_slot] = false;
        int32_t t = host_int[prev_slot];
        ids.push_back(t);
        generated.push_back(t);
        if (on_token) on_token(t, ids);
        if (sampler_config.eos_token_id >= 0 && t == sampler_config.eos_token_id) {
          eos_seen = true;
        }
      }
      prev_slot = cur_slot;
    }

    if (prev_slot >= 0 && slot_pending[prev_slot]) {
      clWaitForEvents(1, &readback_event[prev_slot]);
      clReleaseEvent(readback_event[prev_slot]);
      readback_event[prev_slot] = nullptr;
      slot_pending[prev_slot] = false;
      if (!eos_seen) {
        int32_t t = host_int[prev_slot];
        ids.push_back(t);
        generated.push_back(t);
        if (on_token) on_token(t, ids);
      }
    }

    NNOPT_CHECKPOINT("generate() complete (chained)");
    return ids;
  }

  // Non-greedy or 1-token decode: original blocking loop.
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
