#pragma once
// Auto-generated model interface for state-spaces/mamba2-130m (SSM/Mamba architecture)
// Reference: model_info/transformers_src/modeling_mamba2.py:763-831  (Mamba2Model.forward)
// Reference: model_info/transformers_src/modeling_mamba2.py:874-932  (Mamba2ForCausalLM.forward)

#include "opencl_context.h"
#include "weights.h"
#include "sampler.h"
#include "model_config.h"

#include "layers/backbone.h"
#include "layers/layer_norm.h"
#include "layers/lm_head.h"
#include "layers/ssm.h"

#include <CL/cl.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

class Model {
public:
    Model(OpenCLContext& cl_ctx, Weights& weights);
    ~Model();

    bool initialize();

    // Run a forward pass and return last-token logits (host float32).
    // NOTE: For SSM models this is STATEFUL; call reset_state() before a new sequence.
    std::vector<float> forward(const std::vector<int32_t>& input_ids);

    // Start-position aware forward (used for decode streaming so the mixer can update recurrent state).
    std::vector<float> forward(const std::vector<int32_t>& input_ids, int start_pos);

    using TokenCallback = std::function<void(int32_t)>;

    std::vector<int32_t> generate(const std::vector<int32_t>& prompt_ids,
                                 int max_new_tokens = 64,
                                 SamplerConfig sampler_config = SamplerConfig{},
                                 TokenCallback on_token = nullptr);

    void reset_state();

    // Greedy fast path (Lever 1): runs forward() but reduces logits on GPU
    // and reads back a single int32 instead of 100 KB of fp16. Returns -1
    // on error. Only valid when the caller has already determined that
    // sampling reduces to argmax (temperature <= 0 AND repetition_penalty
    // == 1.0); otherwise call forward() and Sampler::sample.
    int32_t forward_argmax_greedy(const std::vector<int32_t>& input_ids, int start_pos);

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;

    std::unique_ptr<Backbone> backbone_;
    std::unique_ptr<LmHead> lm_head_;
    std::unique_ptr<LayerNorm> norm_[MODEL_CONFIG::NUM_LAYERS];
    std::unique_ptr<Ssm> mixer_[MODEL_CONFIG::NUM_LAYERS];

    cl_program utils_program_ = nullptr;

    // GPU argmax kernel + persistent 4-byte index buffer (Lever 1).
    // Lever H: 2-pass argmax (partial + final) for parallelism.
    cl_program argmax_program_ = nullptr;
    cl_kernel  argmax_kernel_ = nullptr;
    cl_kernel  argmax_partial_kernel_ = nullptr;
    cl_kernel  argmax_final_kernel_ = nullptr;
    cl_mem     argmax_idx_buf_ = nullptr;
    cl_mem     argmax_scratch_v_ = nullptr;  // float × NUM_WG
    cl_mem     argmax_scratch_i_ = nullptr;  // int × NUM_WG
    static const int kArgmaxNumWG = 32;

    // Lever D: history buffer (int32 × max_new_tokens) for async decode,
    // plus a tiny 4-byte "current input token" buffer that the embedding
    // kernel reads from. We copy history[i-1] → cur_token_buf_ before each
    // iteration so the embedding always reads token_ids[0].
    cl_mem  ids_history_buf_ = nullptr;
    cl_mem  cur_token_buf_ = nullptr;
    int     ids_history_capacity_ = 0;

    // Common compute path: runs the full block stack + lm_head and returns
    // the logits buffer (caller releases). Returns nullptr on error.
    cl_mem compute_logits_(const std::vector<int32_t>& input_ids, int start_pos);

    // Lever D: GPU-side input variant. Reads input ids from a GPU buffer,
    // bypasses the host write to buf_ids_. seq_len must be 1 (decode only).
    cl_mem compute_logits_gpu_input_(cl_mem ids_buf, int start_pos);

    bool ensure_ids_history_(int N);
};
