#pragma once
// Auto-generated model interface for ibm-granite/granite-4.0-350m
// Reference: GraniteMoeHybridModel.forward (modeling_granitemoehybrid.py:1151-1210)
//   Backbone dispatch class. The C++ Model below mirrors this PyTorch class —
//   add layer member arrays here matching each layer contract, then port
//   GraniteMoeHybridModel.forward() into src/model.cpp.
// Reference: GraniteMoeHybridForCausalLM.forward (modeling_granitemoehybrid.py:1328-1413)
//   Head wrapper (lm_head + final norm placement). See model.cpp for the
//   backbone↔head split.
#include "opencl_context.h"
#include "weights.h"
#include "sampler.h"
#include "model_config.h"

#include "layers/embedding.h"
#include "layers/layer_norm.h"
#include "layers/attention.h"
#include "layers/shared_mlp.h"

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

    // Step #1: greedy GPU-argmax fast path. Mirrors forward() through the
    // layers + lm_head into the persistent logits buffer, then runs a
    // single-WG argmax kernel and reads back ONE int32 token id (vs the 200 KB
    // host-side fp16 readback in forward()). Used by generate() when
    // sampler.temperature == 0.
    int32_t forward_argmax_greedy(const std::vector<int32_t>& input_ids, int start_pos);

    // Step #3: chained-decode variant. Reads token id from argmax_result_
    // (written by the prior call) — NO host upload — runs forward + lm_head
    // + argmax, leaves the new token id in argmax_result_ on device. Issues
    // an async (CL_FALSE) read into *out_event so the caller can ping-pong
    // host slots and overlap GPU work with host enqueue of the next iter.
    // Returns false on failure; on success the caller must wait on *out_event
    // and read host_slot before using the token id.
    bool forward_argmax_greedy_chained(int start_pos,
                                       int32_t* host_slot,
                                       cl_event* out_event);

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

    // Layers
    std::unique_ptr<Embedding> embedding_;
    std::unique_ptr<LayerNorm> final_norm_;

    std::unique_ptr<LayerNorm> input_layernorm_[MODEL_CONFIG::NUM_LAYERS];
    std::unique_ptr<Attention> attn_[MODEL_CONFIG::NUM_LAYERS];
    std::unique_ptr<LayerNorm> post_attention_layernorm_[MODEL_CONFIG::NUM_LAYERS];
    std::unique_ptr<SharedMlp> shared_mlp_[MODEL_CONFIG::NUM_LAYERS];

    // Persistent logits scratch buffer (Step 6) — was clCreateBuffer'd per
    // forward call. Geometric growth, sized in seq_len*vocab. Released in
    // destructor.
    cl_mem logits_buf_ = nullptr;
    size_t logits_buf_cap_bytes_ = 0;

    // Step #1: GPU argmax resources. Lazy-built on first use of the greedy
    // fast path (forward_argmax_greedy). Released in destructor.
    cl_program argmax_program_ = nullptr;
    cl_kernel  argmax_kernel_  = nullptr;
    cl_mem     argmax_result_  = nullptr;  // 1 × int32 device buffer
    bool ensure_argmax_resources_();
};
