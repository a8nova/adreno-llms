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
        TokenCallback on_token = nullptr,
        // Prefix-cache: the first n_past tokens of prompt_ids are ALREADY resident in the KV cache
        // from a previous generate() call (same persistent Model), so prefill only skips them and
        // processes prompt_ids[n_past:] at start_pos=n_past. 0 = full prefill (default). The caller
        // (serve loop) guarantees prompt_ids[0..n_past-1] are byte-identical to what populated those
        // KV slots. Powers warm multi-turn chat: a follow-up only prefills its new tokens.
        int n_past = 0
    );

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;

    static constexpr int VOCAB_SIZE = 65536;
    static constexpr int HIDDEN_SIZE = 1024;
    static constexpr int NUM_HEADS = 16;
    static constexpr int NUM_KV_HEADS = 8;
    static constexpr int HEAD_DIM = 64;
    static constexpr int MAX_SEQ_LEN = 2048;  // capped from 128000 — see model_config.h

    // ── Hybrid-architecture layout, discovered at runtime from the loaded
    // weights (detect_architecture_). The LFM2.5 family shares one codebase
    // across sizes that differ ONLY in layer count and the conv/attention
    // interleave — e.g. 350M: 16 layers, attn @ {2,5,8,10,12,14}; 230M:
    // 14 layers, attn @ {2,4,6,8,10,12}. Nothing here is baked at compile
    // time, so the same binary runs every size whose weights we load.
    int num_layers_ = 0;
    std::vector<char> is_attention_layer_;   // char, not bool, to allow vector indexing without vector<bool>
    std::vector<char> is_conv_layer_;
    std::vector<int>  attention_layer_indices_;
    void detect_architecture_();
    bool layer_has_attention(int i) const {
        return i >= 0 && i < (int)is_attention_layer_.size() && is_attention_layer_[i];
    }
    bool layer_has_convolution(int i) const {
        return i >= 0 && i < (int)is_conv_layer_.size() && is_conv_layer_[i];
    }

    // Layers (sized to num_layers_ in initialize()).
    std::unique_ptr<Embedding> embedding_;
    std::unique_ptr<OperatorNorm> embedding_norm_;

    std::vector<std::unique_ptr<OperatorNorm>> operator_norm_;
    std::vector<std::unique_ptr<Attention>> attn_;
    std::vector<std::unique_ptr<Convolution>> conv_;
    std::vector<std::unique_ptr<Mlp>> mlp_;

    std::vector<std::unique_ptr<OperatorNorm>> ffn_norm_;

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

    // ── cl_qcom_recordable_queues integration (NNOPT_RECORD=1).
    // Records the 16-layer-block decode-step dispatch sequence once at the
    // first decode call, replays it on subsequent steps with per-step arg
    // overrides for start_pos / seq_k. lm_head + argmax stay outside the
    // recording (the tiled int8 lm_head needs clCreateSubBuffer per call,
    // and only clEnqueueNDRangeKernel is recordable per PDF §9.1.3).
    bool                record_enabled_   = false;
    cl_command_queue    record_queue_     = nullptr;
    cl_recording_qcom   recording_        = nullptr;
    bool                recording_built_  = false;
    struct PerStepArg { cl_kernel kernel; cl_uint arg_indx; };
    std::vector<PerStepArg> rec_start_pos_args_;
    std::vector<PerStepArg> rec_seq_k_args_;
    int32_t             cur_start_pos_ = 0;
    int32_t             cur_seq_k_     = 0;
    void collect_record_args_();
};
