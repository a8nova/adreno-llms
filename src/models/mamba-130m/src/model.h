#pragma once
// Reference: model_info/transformers_src/modeling_mamba.py MambaForCausalLM.forward
// Model driver for Mamba (SSM) architecture.

#include "opencl_context.h"
#include "sampler.h"
#include "weights.h"

#include "layers/backbone.h"
#include "layers/layer_norm.h"
#include "layers/ssm.h"

#include "model_config.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

class Model {
public:
  Model(OpenCLContext &cl_ctx, Weights &weights);
  ~Model();

  // True only if every layer initialized successfully. Adreno's OpenCL
  // compiler intermittently returns CL_OUT_OF_HOST_MEMORY (-6) on the
  // first program build of a session even when the kernel source is fine;
  // when that happens, layer init fails and any subsequent forward()
  // would dereference an empty mixers_ vector. main.cpp must check ok()
  // before calling generate() — typical recovery is a redeploy + retry.
  bool ok() const { return init_ok_; }

  // Stateful forward: updates internal recurrent state inside each Ssm layer.
  // Returns the last-token logits as fp32 (legacy API used by non-greedy
  // sampling paths). Has a per-call cost of ~100 KB readback + fp16→fp32
  // conversion of the full logits vector — avoid in tight decode loops.
  std::vector<float> forward(const std::vector<int32_t> &input_ids);

  // Greedy fast path: forward + GPU-side argmax. Returns just the token id.
  // Skips the full-logits readback (only 1 int32 crosses the host↔device
  // boundary). Use this whenever sampler_config.temperature <= 0 in
  // generate() — eliminates ~5 ms/token of host work.
  int32_t forward_argmax(const std::vector<int32_t> &input_ids);

  // Streaming callback: called once per newly-generated token id (NOT for
  // prompt tokens). Use this to print/decode tokens live as they're produced.
  // The callback runs synchronously on the generate() thread between tokens —
  // keep it light (a tokenizer.decode + fputs is fine; heavy work will tank
  // tok/s). Pass nullptr (or omit) to run without streaming.
  using TokenCallback = std::function<void(int32_t)>;

  std::vector<int32_t> generate(const std::vector<int32_t> &prompt_ids,
                               int max_new_tokens = 64,
                               SamplerConfig sampler_config = SamplerConfig{},
                               TokenCallback on_token = nullptr);

  void reset_state();

private:
  OpenCLContext &cl_ctx_;
  Weights &weights_;
  bool init_ok_ = false;

  // Layers
  std::unique_ptr<Backbone> backbone_;
  std::vector<std::unique_ptr<LayerNorm>> norms_;
  std::vector<std::unique_ptr<Ssm>> mixers_;
  std::unique_ptr<LayerNorm> final_norm_;

  // GPU argmax kernel — lazily built on first greedy forward.
  cl_program argmax_program_ = nullptr;
  cl_kernel  argmax_kernel_  = nullptr;
  cl_mem     argmax_out_buf_ = nullptr;  // single int32 output, persistent

  // Persistent lm_head logits buffer (sized to MAX seq_len × VOCAB_SIZE).
  // 100 KB × seq_len = 400 KB peak for prefill of 4. Reused across calls.
  cl_mem logits_buf_     = nullptr;
  int    logits_capacity_M_ = 0;

  // Internal: shared forward body that produces the lm_head logits buffer.
  // Returns the cl_mem and writes to *seq_len_out and *vocab_size_out.
  // Caller is responsible for clReleaseMemObject on the returned buffer.
  cl_mem forward_to_logits(const std::vector<int32_t> &input_ids, int *seq_len_out);

  bool ensure_argmax_program_();

  // NOTE: This model uses tied weights: lm_head = backbone.embeddings.weight.
  // We compute logits via pytorch_linear(hidden, wte) inside Model::forward.
};
