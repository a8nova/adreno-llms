// Reference: model_info/transformers_src/modeling_mamba.py MambaForCausalLM.forward
// Implements: embeddings -> (for each layer: RMSNorm -> MambaMixer -> residual add) -> final RMSNorm -> lm_head.

#include "model.h"

#include "benchmark.h"
#include "debug_utils.h"
#include "kernel_profiler.h"
#include "model_config.h"
#include "utils.h"

#include <cstring>
#include <memory>
#include <vector>

Model::Model(OpenCLContext& cl_ctx, Weights& weights) : cl_ctx_(cl_ctx), weights_(weights) {
  NNOPT_CHECKPOINT("Model constructor");

  backbone_ = std::make_unique<Backbone>(cl_ctx_, weights_);
  if (!backbone_->initialize()) {
    NNOPT_ERROR("backbone.initialize FAILED");
    return;
  }
  NNOPT_LAYER_INIT("backbone");

  norms_.reserve(MODEL_CONFIG::NUM_LAYERS);
  mixers_.reserve(MODEL_CONFIG::NUM_LAYERS);
  for (int i = 0; i < MODEL_CONFIG::NUM_LAYERS; i++) {
    {
      auto ln = std::make_unique<LayerNorm>(
          cl_ctx_, weights_, "backbone.layers." + std::to_string(i) + ".norm.weight",
          MODEL_CONFIG::HIDDEN_SIZE, MODEL_CONFIG::RMS_NORM_EPS);
      if (!ln->initialize()) {
        NNOPT_ERROR_FMT("norm[%d].initialize FAILED", i);
        return;
      }
      norms_.push_back(std::move(ln));
      NNOPT_LAYER_INIT_FMT("norm_%d", i);
    }

    {
      auto ssm = std::make_unique<Ssm>(cl_ctx_, weights_, "backbone.layers." + std::to_string(i) + ".mixer", i);
      if (!ssm->initialize()) {
        NNOPT_ERROR_FMT("ssm[%d].initialize FAILED", i);
        return;
      }
      mixers_.push_back(std::move(ssm));
      NNOPT_LAYER_INIT_FMT("mixer_%d", i);
    }
  }

  // HF key is backbone.norm_f.weight (see weights/model.meta.json)
  final_norm_ = std::make_unique<LayerNorm>(cl_ctx_, weights_, "backbone.norm_f.weight",
                                           MODEL_CONFIG::HIDDEN_SIZE, MODEL_CONFIG::RMS_NORM_EPS);
  if (!final_norm_->initialize()) {
    NNOPT_ERROR("final_norm.initialize FAILED");
    return;
  }
  NNOPT_LAYER_INIT("final_norm");

  // lm_head is tied to embeddings.weight in this model; computed in forward() via GEMM.
  init_ok_ = true;
}

Model::~Model() {
  if (argmax_kernel_)  clReleaseKernel(argmax_kernel_);
  if (argmax_program_) clReleaseProgram(argmax_program_);
  if (argmax_out_buf_) clReleaseMemObject(argmax_out_buf_);
  if (logits_buf_)     clReleaseMemObject(logits_buf_);
}

// Lazily allocate / grow the persistent logits buffer to fit seq_len.
static cl_mem ensure_logits_buf(cl_context ctx, cl_mem& buf, int& cap, int seq_len) {
  if (seq_len <= cap && buf) return buf;
  if (buf) { clReleaseMemObject(buf); buf = nullptr; }
  cl_int err = CL_SUCCESS;
  buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                       (size_t)seq_len * (size_t)MODEL_CONFIG::VOCAB_SIZE * sizeof(nnopt_storage_t),
                       nullptr, &err);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("ensure_logits_buf: alloc failed: %d", (int)err);
    buf = nullptr;
    return nullptr;
  }
  cap = seq_len;
  return buf;
}

void Model::reset_state() {
  // SSM states live in each Ssm instance; reset all.
  for (int i = 0; i < (int)mixers_.size(); i++) mixers_[i]->reset_state(cl_ctx_.queue());
}

bool Model::ensure_argmax_program_() {
  if (argmax_kernel_) return true;
  argmax_program_ = cl_ctx_.build_program_from_file("kernels/argmax.cl");
  if (!argmax_program_) {
    NNOPT_ERROR("ensure_argmax_program_: build_program_from_file(kernels/argmax.cl) failed");
    return false;
  }
  cl_int err = CL_SUCCESS;
  argmax_kernel_ = clCreateKernel(argmax_program_, "argmax_logits", &err);
  if (err != CL_SUCCESS || !argmax_kernel_) {
    NNOPT_ERROR_FMT("ensure_argmax_program_: clCreateKernel failed: %d", (int)err);
    clReleaseProgram(argmax_program_);
    argmax_program_ = nullptr;
    return false;
  }
  argmax_out_buf_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, sizeof(int32_t), nullptr, &err);
  if (err != CL_SUCCESS || !argmax_out_buf_) {
    NNOPT_ERROR_FMT("ensure_argmax_program_: argmax_out_buf alloc failed: %d", (int)err);
    return false;
  }
  return true;
}

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids) {
  NNOPT_CHECKPOINT("forward() started");
  cl_command_queue queue = cl_ctx_.queue();
  const int seq_len = (int)input_ids.size();

  cl_program utils_program = cl_ctx_.utils_program();
  if (!utils_program) {
    NNOPT_ERROR("utils_program missing");
    return {};
  }
  // IMPORTANT: OpenCLContext owns utils_program; do NOT clReleaseProgram(utils_program).

  // backbone_->embed takes a host pointer and uploads internally — pass the
  // input vector directly. Eliminates the upload-then-read-back round-trip
  // (was 2× CL_TRUE syncs per forward; per 32-token decode that's 66 host↔
  // device round trips for no semantic gain — input_ids is the host vector).
  cl_int err = CL_SUCCESS;
  (void)err;  // referenced below when LAYER_CHECK is enabled in debug builds
  cl_mem hidden = backbone_->embed(queue, (const int*)input_ids.data(), seq_len);
  if (!hidden) return {};
  NNOPT_LAYER_CHECK("backbone", queue, hidden, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

  for (int i = 0; i < MODEL_CONFIG::NUM_LAYERS; i++) {
    char name_norm[64];
    snprintf(name_norm, sizeof(name_norm), "norm_%d", i);
    NNOPT_LAYER_CHECK_INPUT(name_norm, queue, hidden, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

    cl_mem normed = norms_[i]->forward(queue, hidden, seq_len);
    if (!normed) {
      clReleaseMemObject(hidden);
      return {};
    }
    NNOPT_LAYER_CHECK(name_norm, queue, normed, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

    // Fused decode fast path: forward_fused_decode mutates `hidden` directly
    // via gemv_m1_k1536_no4_radd, skipping a separate element_add_inplace.
    // Only valid for seq_len==1 (decode); prefill falls back to the
    // unfused forward + residual chain.
    bool fused_ok = false;
    // Empirical (Adreno 620): fused-radd path is a regression vs plain
    // out_proj + residual_add (radd kernel is 1.34× slower per call).
    // Disabled; opt-in via env NNOPT_SSM_FUSED_DECODE=1 for re-evaluation.
    static const bool s_use_fused_decode = []() {
      const char* e = std::getenv("NNOPT_SSM_FUSED_DECODE");
      return e && e[0] != '0';
    }();
    if (s_use_fused_decode && seq_len == 1) {
      fused_ok = mixers_[i]->forward_fused_decode(queue, normed, hidden, seq_len);
    }

    if (fused_ok) {
      clReleaseMemObject(normed);
      // hidden has been mutated in-place; nothing else to do.
    } else {
      cl_mem mixer_out = mixers_[i]->forward(queue, normed, seq_len);
      clReleaseMemObject(normed);
      if (!mixer_out) {
        clReleaseMemObject(hidden);
        return {};
      }

      char name_ssm[64];
      snprintf(name_ssm, sizeof(name_ssm), "mixer_%d", i);
      NNOPT_LAYER_CHECK(name_ssm, queue, mixer_out, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

      // hidden += mixer_out  (residual). element_add_inplace caches its kernel
      // object across calls — no per-call clCreateKernel, no extra alloc/copy.
      if (!element_add_inplace(queue, utils_program, hidden, mixer_out,
                               (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE)) {
        clReleaseMemObject(mixer_out);
        clReleaseMemObject(hidden);
        return {};
      }
      clReleaseMemObject(mixer_out);
    }
  }

  NNOPT_LAYER_CHECK_INPUT("final_norm", queue, hidden, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
  cl_mem final_h = final_norm_->forward(queue, hidden, seq_len);
  clReleaseMemObject(hidden);
  if (!final_h) return {};
  NNOPT_LAYER_CHECK("final_norm", queue, final_h, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

  // lm_head: tied to embedding weights (backbone.embeddings.weight)
  cl_mem wte = weights_.get_buffer("backbone.embeddings.weight");
  if (!wte) {
    NNOPT_ERROR("missing backbone.embeddings.weight for tied lm_head");
    clReleaseMemObject(final_h);
    return {};
  }

  cl_mem logits_buf = ensure_logits_buf(cl_ctx_.context(), logits_buf_, logits_capacity_M_, seq_len);
  if (!logits_buf) {
    clReleaseMemObject(final_h);
    return {};
  }

  // out[M,V] = hidden[M,H] @ wte[V,H]^T
  if (!pytorch_linear(queue, seq_len, MODEL_CONFIG::VOCAB_SIZE, MODEL_CONFIG::HIDDEN_SIZE, final_h, wte, logits_buf)) {
    clReleaseMemObject(final_h);
    return {};
  }
  clReleaseMemObject(final_h);
  NNOPT_LAYER_CHECK("lm_head", queue, logits_buf, (size_t)seq_len * MODEL_CONFIG::VOCAB_SIZE);

  // Read back ONLY last token logits for sampler
  std::vector<float> logits(MODEL_CONFIG::VOCAB_SIZE);
  const size_t offset_bytes = (size_t)(seq_len - 1) * (size_t)MODEL_CONFIG::VOCAB_SIZE * sizeof(nnopt_storage_t);

#ifdef NNOPT_USE_FP16
  std::vector<nnopt_storage_t> tmp(MODEL_CONFIG::VOCAB_SIZE);
  err = clEnqueueReadBuffer(queue, logits_buf, CL_TRUE, offset_bytes, (size_t)MODEL_CONFIG::VOCAB_SIZE * sizeof(nnopt_storage_t),
                            tmp.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("read logits(fp16) failed: %d", (int)err);
    return {};
  }
  for (int i = 0; i < MODEL_CONFIG::VOCAB_SIZE; i++) logits[i] = nnopt_f16_to_f32(tmp[i]);
#else
  err = clEnqueueReadBuffer(queue, logits_buf, CL_TRUE, offset_bytes, (size_t)MODEL_CONFIG::VOCAB_SIZE * sizeof(float),
                            logits.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("read logits(fp32) failed: %d", (int)err);
    return {};
  }
#endif

  NNOPT_CHECKPOINT("forward() complete");
  return logits;
}

// Greedy fast path: same forward as above through lm_head, then a single-WG
// argmax kernel over the last-token slice of the logits buffer. Reads back
// just one int32 instead of 50,280 fp16. Eliminates the host-side
// std::max_element + fp16→fp32 conversion + the bulk of the readback wait.
int32_t Model::forward_argmax(const std::vector<int32_t>& input_ids) {
  NNOPT_CHECKPOINT("forward_argmax() started");
  cl_command_queue queue = cl_ctx_.queue();
  const int seq_len = (int)input_ids.size();

  if (!ensure_argmax_program_()) return -1;

  cl_program utils_program = cl_ctx_.utils_program();
  if (!utils_program) {
    NNOPT_ERROR("utils_program missing");
    return -1;
  }

  cl_mem hidden = backbone_->embed(queue, (const int*)input_ids.data(), seq_len);
  if (!hidden) return -1;

  for (int i = 0; i < MODEL_CONFIG::NUM_LAYERS; i++) {
    cl_mem normed = norms_[i]->forward(queue, hidden, seq_len);
    if (!normed) {
      clReleaseMemObject(hidden);
      return -1;
    }
    bool fused_ok = false;
    // Empirical (Adreno 620): fused-radd path is a regression vs plain
    // out_proj + residual_add (radd kernel is 1.34× slower per call).
    // Disabled; opt-in via env NNOPT_SSM_FUSED_DECODE=1 for re-evaluation.
    static const bool s_use_fused_decode = []() {
      const char* e = std::getenv("NNOPT_SSM_FUSED_DECODE");
      return e && e[0] != '0';
    }();
    if (s_use_fused_decode && seq_len == 1) {
      fused_ok = mixers_[i]->forward_fused_decode(queue, normed, hidden, seq_len);
    }
    if (fused_ok) {
      clReleaseMemObject(normed);
    } else {
      cl_mem mixer_out = mixers_[i]->forward(queue, normed, seq_len);
      clReleaseMemObject(normed);
      if (!mixer_out) {
        clReleaseMemObject(hidden);
        return -1;
      }
      if (!element_add_inplace(queue, utils_program, hidden, mixer_out,
                               (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE)) {
        clReleaseMemObject(mixer_out);
        clReleaseMemObject(hidden);
        return -1;
      }
      clReleaseMemObject(mixer_out);
    }
  }

  cl_mem final_h = final_norm_->forward(queue, hidden, seq_len);
  clReleaseMemObject(hidden);
  if (!final_h) return -1;

  cl_mem wte = weights_.get_buffer("backbone.embeddings.weight");
  if (!wte) {
    NNOPT_ERROR("missing backbone.embeddings.weight for tied lm_head");
    clReleaseMemObject(final_h);
    return -1;
  }

  cl_mem logits_buf = ensure_logits_buf(cl_ctx_.context(), logits_buf_, logits_capacity_M_, seq_len);
  if (!logits_buf) {
    clReleaseMemObject(final_h);
    return -1;
  }

  if (!pytorch_linear(queue, seq_len, MODEL_CONFIG::VOCAB_SIZE, MODEL_CONFIG::HIDDEN_SIZE,
                     final_h, wte, logits_buf)) {
    clReleaseMemObject(final_h);
    return -1;
  }
  clReleaseMemObject(final_h);

  // GPU argmax over the last-token slice. We pass the offset as a kernel
  // arg (in storage_t units) — avoids clCreateSubBuffer which has 128-byte
  // origin alignment that doesn't always hold for arbitrary seq_len.
  int vocab  = MODEL_CONFIG::VOCAB_SIZE;
  int offset = (seq_len - 1) * vocab;  // last-token row inside logits[seq_len, vocab]
  cl_int err = clSetKernelArg(argmax_kernel_, 0, sizeof(cl_mem), &logits_buf);
  if (err == CL_SUCCESS) err = clSetKernelArg(argmax_kernel_, 1, sizeof(cl_mem), &argmax_out_buf_);
  if (err == CL_SUCCESS) err = clSetKernelArg(argmax_kernel_, 2, sizeof(int),    &vocab);
  if (err == CL_SUCCESS) err = clSetKernelArg(argmax_kernel_, 3, sizeof(int),    &offset);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("forward_argmax: setKernelArg failed: %d", (int)err);
    return -1;
  }

  size_t WG_AM = 256;
  size_t gws = WG_AM;
  size_t lws = WG_AM;
  cl_event* prof_evt = KernelProfiler::event_for("argmax");
  err = clEnqueueNDRangeKernel(queue, argmax_kernel_, 1, nullptr, &gws, &lws, 0, nullptr, prof_evt);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("forward_argmax: argmax enqueue failed: %d", (int)err);
    return -1;
  }

  int32_t token_id = -1;
  err = clEnqueueReadBuffer(queue, argmax_out_buf_, CL_TRUE, 0, sizeof(int32_t),
                            &token_id, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("forward_argmax: read argmax_out failed: %d", (int)err);
    token_id = -1;
  }

  NNOPT_CHECKPOINT("forward_argmax() complete");
  return token_id;
}

std::vector<int32_t> Model::generate(const std::vector<int32_t>& prompt_ids, int max_new_tokens, SamplerConfig sampler_config, TokenCallback on_token) {
  NNOPT_CHECKPOINT("generate() started");
  reset_state();

  auto ids = prompt_ids;

  // Greedy fast path: temperature <= 0 with no repetition penalty means each
  // sample step is just argmax(logits). We do that on GPU and read back one
  // int32 instead of 50,280 fp16 — saves ~5 ms of host work per token.
  // Repetition penalty / top-k / top-p still need full logits, so stay on
  // the host path for those.
  const bool greedy_fast =
      (sampler_config.temperature <= 0.0f) &&
      (sampler_config.repetition_penalty == 1.0f);

  if (greedy_fast) {
    // Prefill: produce next-token argmax via fast path.
    int next_token = forward_argmax(prompt_ids);
    if (next_token < 0) return ids;
    ids.push_back(next_token);
    NNOPT_BENCH_FIRST_TOKEN();
    if (on_token) on_token(next_token);
    if (sampler_config.eos_token_id >= 0 && next_token == sampler_config.eos_token_id) {
      // EOS at first decode — done.
    } else {
      for (int i = 1; i < max_new_tokens; i++) {
        int tok = forward_argmax({next_token});
        if (tok < 0) break;
        next_token = tok;
        ids.push_back(next_token);
        if (on_token) on_token(next_token);
        if (sampler_config.eos_token_id >= 0 && next_token == sampler_config.eos_token_id) break;
      }
    }
    NNOPT_CHECKPOINT("generate() complete");
    if (KernelProfiler::enabled()) {
      clFinish(cl_ctx_.queue());
      KernelProfiler::dump_summary();
    }
    return ids;
  }

  // Legacy path: full logits readback + host sampling (top-k / top-p / etc.)
  Sampler sampler(sampler_config);
  auto logits = forward(prompt_ids);

  for (int i = 0; i < max_new_tokens; i++) {
    std::vector<int32_t> generated(ids.begin() + prompt_ids.size(), ids.end());
    int next_token = sampler.sample(logits, generated);
    ids.push_back(next_token);
    NNOPT_BENCH_FIRST_TOKEN();
    if (on_token) on_token(next_token);

    if (sampler_config.eos_token_id >= 0 && next_token == sampler_config.eos_token_id) break;

    logits = forward({next_token});
  }

  NNOPT_CHECKPOINT("generate() complete");

  // If profiling was enabled, drain pending events and print the breakdown.
  // Per-event clGetEventProfilingInfo blocks until the event completes — at
  // this point the inference loop is done so the queue has drained anyway.
  if (KernelProfiler::enabled()) {
    clFinish(cl_ctx_.queue());  // ensure all events ready to read
    KernelProfiler::dump_summary();
  }

  return ids;
}
