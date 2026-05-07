#pragma once
// Auto-generated model interface for apple/OpenELM-270M
// Reference: model_info/modeling_openelm.py:620-747  (OpenELMModel.forward)
//   Backbone dispatch class. The C++ Model below mirrors this PyTorch class —
//   add layer member arrays here matching (model metadata),
//   then port OpenELMModel.forward() into src/model.cpp.
// Reference: model_info/modeling_openelm.py:836-910  (OpenELMForCausalLM.forward)
//   Head wrapper (lm_head + final norm placement). See model.cpp for the
//   backbone↔head split.
#include "opencl_context.h"
#include "weights.h"
#include "sampler.h"
#include "model_config.h"
#include "layers/embedding.h"
#include "layers/layer_norm.h"
#include "layers/attention.h"
#include "layers/mlp.h"
#include <functional>
#include <memory>
#include <vector>
#include <cstdint>

class Model {
public:
    Model(OpenCLContext& cl_ctx, Weights& weights);
    ~Model();

    // Build per-layer kernels and per-layer instance state. MUST be called
    // exactly once after construction and BEFORE forward()/generate().
    // Returns false on the first per-layer init failure (logged via
    // NNOPT_ERROR_FMT). Constructor is intentionally void: a void ctor cannot
    // signal init failure to main(), so init lives here.
    bool initialize();

    // Run forward pass on input token IDs, return logits.
    // Legacy single-arg form is a thin wrapper over the two-arg form.
    std::vector<float> forward(const std::vector<int32_t>& input_ids);

    // Two-arg form with explicit cache offset (Rules KV-01 / GEN-01).
    // start_pos == 0  -> prefill (writes KV cache rows [0, seq_len))
    // start_pos > 0   -> decode (reads cache rows [0, start_pos), writes
    //                    rows [start_pos, start_pos + seq_len))
    std::vector<float> forward(const std::vector<int32_t>& input_ids, int start_pos);

    // Decode fast path (single-token forward at decode). Dispatched from
    // forward(seq_len==1, start_pos>0) to call scaffold-provided fused
    // kernels in kernels/block_fused.cl. Agent fills in the body.
    std::vector<float> forward_decode(int32_t token_id, int start_pos);

    // Greedy fast-path: full forward + GPU argmax over the last token's
    // logits, returning a single int32 token id (4 B device→host) instead of
    // the full [vocab] fp16 readback (~100 KB). Used by generate() when
    // sampler.temperature<=0 AND repetition_penalty==1. Returns -1 on error.
    int32_t forward_argmax_greedy(const std::vector<int32_t>& input_ids, int start_pos);

    // Step 9: enqueue-only chained variant. Reads its single input token from
    // argmax_result_ (written by the previous forward) instead of host memory,
    // and DOES NOT block on a host readback. Caller is responsible for
    // pipelining a CL_FALSE clEnqueueReadBuffer of argmax_result_ if it needs
    // the int32 on the host (e.g. for streaming/EOS). Returns true on success.
    bool forward_argmax_greedy_chained_enqueue(int start_pos);
    cl_mem argmax_result_buffer() const { return argmax_result_; }

    // Streaming callback: invoked once per newly-generated token (NOT for
    // prompt tokens). Use this to print/decode tokens live as they are
    // produced. Runs synchronously on the generate() thread between tokens —
    // keep it light (a tokenizer.decode + fputs is fine; heavy work tanks
    // tok/s). Pass nullptr (or omit) to run without streaming.
    using TokenCallback = std::function<void(int32_t)>;

    // Auto-regressive generation with configurable sampling
    std::vector<int32_t> generate(
        const std::vector<int32_t>& prompt_ids,
        int max_new_tokens = 64,
        SamplerConfig sampler_config = SamplerConfig{},
        TokenCallback on_token = nullptr
    );

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;

    std::unique_ptr<Embedding> embedding_;
    std::unique_ptr<LayerNorm> final_norm_;
    std::unique_ptr<LayerNorm> pre_attn_norm_[MODEL_CONFIG::NUM_TRANSFORMER_LAYERS];
    std::unique_ptr<Attention> attn_[MODEL_CONFIG::NUM_TRANSFORMER_LAYERS];
    std::unique_ptr<LayerNorm> post_attn_norm_[MODEL_CONFIG::NUM_TRANSFORMER_LAYERS];
    std::unique_ptr<Mlp> mlp_[MODEL_CONFIG::NUM_TRANSFORMER_LAYERS];

    // GPU argmax (Step 3). Lazily built on first forward_argmax_greedy() call.
    cl_program argmax_program_ = nullptr;
    cl_kernel  argmax_kernel_  = nullptr;
    cl_mem     argmax_result_  = nullptr;   // 1 × int32 device buffer

    // Step Z: persistent logits buffer for the lm_head. Sticky capacity.
    cl_mem  logits_buf_ = nullptr;
    size_t  logits_buf_cap_bytes_ = 0;
    bool ensure_logits_buf_(int seq_len);

    bool ensure_argmax_resources_();

    // Model dimensions (prefer MODEL_CONFIG::* at call sites; kept for convenience)
    static constexpr int VOCAB_SIZE = MODEL_CONFIG::VOCAB_SIZE;
};
