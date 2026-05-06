#pragma once
// Auto-generated model interface for HuggingFaceTB/SmolLM2-135M-Instruct
// Reference: model_info/transformers_src/modeling_llama.py:375-428  (LlamaModel.forward)
//   Backbone dispatch class. The C++ Model below mirrors this PyTorch class —
//   add layer member arrays here matching (model metadata),
//   then port LlamaModel.forward() into src/model.cpp.
// Reference: model_info/transformers_src/modeling_llama.py:445-501  (LlamaForCausalLM.forward)
//   Head wrapper (lm_head + final norm placement). See model.cpp for the
//   backbone↔head split.
#include "opencl_context.h"
#include "weights.h"
#include "sampler.h"
#include "model_config.h"
#include "layers/attention.h"
#include "layers/embedding.h"
#include "layers/layer_norm.h"
#include "layers/mlp.h"
#include "layers/lm_head.h"
#include <functional>
#include <vector>
#include <cstdint>
#include <memory>

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

    // Greedy decode fast path: skips the V=49152 fp16 logits readback
    // (98 KB GPU→CPU per token, blocking) by running argmax on-GPU and
    // reading back a single int. Only valid when temperature <= 0 AND
    // repetition_penalty == 1.0. Returns -1 on failure; caller falls back
    // to forward_decode + Sampler.
    int32_t forward_decode_greedy(int32_t token_id, int start_pos);

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

    // ── Submodules (mirrors LlamaModel)
    std::unique_ptr<Embedding> embed_tokens_;
    std::unique_ptr<LayerNorm> final_norm_;
    std::unique_ptr<LmHead> lm_head_;

    // GPU argmax fast path (greedy decode only). Lazy-initialized on first call.
    cl_program argmax_prog_      = nullptr;
    cl_kernel  argmax_partial_   = nullptr;
    cl_kernel  argmax_final_     = nullptr;
    cl_mem     argmax_scratch_v_ = nullptr;  // float[NUM_WG]
    cl_mem     argmax_scratch_i_ = nullptr;  // int[NUM_WG]
    cl_mem     argmax_out_idx_   = nullptr;  // int[1]
    cl_mem     argmax_cur_token_ = nullptr;  // int[1]
    bool       argmax_ready_     = false;
    bool       argmax_tried_     = false;
    bool ensure_argmax_program(cl_command_queue queue);

    std::unique_ptr<LayerNorm> input_layernorm_[MODEL_CONFIG::NUM_HIDDEN_LAYERS];
    std::unique_ptr<Attention> self_attn_[MODEL_CONFIG::NUM_HIDDEN_LAYERS];
    std::unique_ptr<LayerNorm> post_attention_layernorm_[MODEL_CONFIG::NUM_HIDDEN_LAYERS];
    std::unique_ptr<Mlp> mlp_[MODEL_CONFIG::NUM_HIDDEN_LAYERS];
};
