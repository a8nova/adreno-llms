#pragma once
// Reference: model_info/transformers_src/modeling_lfm2.py:378-518 (Lfm2Model/Lfm2ForCausalLM wiring)
// Auto-generated model interface for LiquidAI/LFM2.5-350M-Base (agent-filled)

#include "opencl_context.h"
#include "weights.h"
#include "sampler.h"

#include "layers/embedding.h"
#include "layers/operator_norm.h"
#include "layers/attention.h"
#include "layers/convolution.h"
#include "layers/mlp.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

// Streaming callback: invoked once per sampled token (BOTH first and
// subsequent decode steps). `new_token` is the token just produced;
// `all_ids` is the full ids buffer (prompt_ids + every token so far,
// including new_token). Caller can incrementally decode + print.
// Pass nullptr for non-streaming behaviour.
using TokenCallback = std::function<void(int32_t /*new_token*/,
                                         const std::vector<int32_t>& /*all_ids*/)>;

class Model {
public:
    Model(OpenCLContext& cl_ctx, Weights& weights);
    ~Model();

    bool initialize();

    std::vector<float> forward(const std::vector<int32_t>& input_ids);
    std::vector<float> forward(const std::vector<int32_t>& input_ids, int start_pos);

    std::vector<float> forward_decode(int32_t token_id, int start_pos);

    // Greedy fast path: runs forward through lm_head, then a 2-pass GPU
    // argmax kernel and reads back a single int32 (4 bytes) instead of the
    // full 128 KB logits row + host fp16->fp32 conversion + host argmax.
    // Returns -1 on failure. Only used when sampler is pure greedy
    // (temperature<=0 AND repetition_penalty==1).
    int32_t forward_greedy(const std::vector<int32_t>& input_ids, int start_pos);

    // Step 10: chained-decode forward (single token). Embedding reads its
    // input token from `argmax_out_idx_` (written by the previous forward),
    // and the final argmax writes back into `argmax_out_idx_`. NO host
    // readback inside this function — caller is responsible for pipelining
    // a CL_FALSE clEnqueueReadBuffer of argmax_out_buffer() and waiting on
    // the event when needed (e.g. for streaming/EOS check).
    bool forward_greedy_chained_enqueue(int start_pos);
    cl_mem argmax_out_buffer() const { return argmax_out_idx_; }

    std::vector<int32_t> generate(
        const std::vector<int32_t>& prompt_ids,
        int max_new_tokens = 64,
        SamplerConfig sampler_config = SamplerConfig{},
        TokenCallback on_token = nullptr
    );

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;

    static constexpr int VOCAB_SIZE = 65536;
    static constexpr int HIDDEN_SIZE = 1024;
    static constexpr int NUM_LAYERS = 16;
    static constexpr int NUM_HEADS = 16;
    static constexpr int NUM_KV_HEADS = 8;
    static constexpr int HEAD_DIM = 64;
    static constexpr int MAX_SEQ_LEN = 128000;

    // Hybrid-architecture per-layer-class dispatch helpers
    static constexpr int ATTENTION_LAYER_INDICES[6] = {2, 5, 8, 10, 12, 14};
    static constexpr int NUM_ATTENTION_LAYERS = 6;
    inline static bool layer_has_attention(int i) {
        for (int j = 0; j < NUM_ATTENTION_LAYERS; ++j) if (ATTENTION_LAYER_INDICES[j] == i) return true;
        return false;
    }

    static constexpr int CONVOLUTION_LAYER_INDICES[10] = {0, 1, 3, 4, 6, 7, 9, 11, 13, 15};
    static constexpr int NUM_CONVOLUTION_LAYERS = 10;
    inline static bool layer_has_convolution(int i) {
        for (int j = 0; j < NUM_CONVOLUTION_LAYERS; ++j) if (CONVOLUTION_LAYER_INDICES[j] == i) return true;
        return false;
    }

    // Layers
    std::unique_ptr<Embedding> embedding_;
    std::unique_ptr<OperatorNorm> embedding_norm_;

    std::unique_ptr<OperatorNorm> operator_norm_[NUM_LAYERS];
    std::unique_ptr<Attention> attn_[NUM_LAYERS];
    std::unique_ptr<Convolution> conv_[NUM_LAYERS];
    std::unique_ptr<Mlp> mlp_[NUM_LAYERS];

    std::unique_ptr<OperatorNorm> ffn_norm_[NUM_LAYERS];

    // Utils kernels
    cl_program utils_program_ = nullptr;

    // Persistent logits buffer (lazy-grow): max_seq_len × VOCAB_SIZE.
    cl_mem buf_logits_ = nullptr;
    int    buf_logits_seq_capacity_ = 0;

    // GPU argmax (greedy decode fast path)
    cl_program argmax_program_      = nullptr;
    cl_kernel  argmax_block_kernel_ = nullptr;
    cl_kernel  argmax_final_kernel_ = nullptr;
    cl_mem     argmax_partials_val_ = nullptr;  // float [64]
    cl_mem     argmax_partials_idx_ = nullptr;  // int   [64]
    cl_mem     argmax_out_idx_      = nullptr;  // int   [1]
    bool       argmax_initialized_  = false;
    bool ensure_argmax_resources_();
};
