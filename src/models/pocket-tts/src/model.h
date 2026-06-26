#pragma once
// Auto-generated model interface for kyutai/pocket-tts
// Reference: (top-level model class not auto-detected — see model_info/transformers_src/)
#include "opencl_context.h"
#include "weights.h"
#include "sampler.h"
#include <functional>
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

    // Model dimensions (constexpr avoids ODR linker issues on Android)
    static constexpr int VOCAB_SIZE = 4001;
    static constexpr int HIDDEN_SIZE = 1024;
    static constexpr int NUM_LAYERS = 6;
    static constexpr int MAX_SEQ_LEN = 10000;

    // ── Hybrid-architecture per-layer-class dispatch helpers ─────────
    // Use these in Model::initialize() and Model::forward() to gate
    // per-layer instantiation/dispatch on the actual indices that own
    // each sub-block. Derived mechanically from safetensors key presence.
    // Per-class layer index map — 'mlp' is hybrid: only 2/6 layers have it.
    static constexpr int MLP_LAYER_INDICES[2] = {0, 2};
    static constexpr int NUM_MLP_LAYERS = 2;
    inline static bool layer_has_mlp(int i) {
        for (int j = 0; j < NUM_MLP_LAYERS; ++j)
            if (MLP_LAYER_INDICES[j] == i) return true;
        return false;
    }

    // Per-class layer index map — 'convolution' is hybrid: only 9/6 layers have it.
    static constexpr int CONVOLUTION_LAYER_INDICES[9] = {0, 1, 2, 3, 5, 6, 8, 9, 11};
    static constexpr int NUM_CONVOLUTION_LAYERS = 9;
    inline static bool layer_has_convolution(int i) {
        for (int j = 0; j < NUM_CONVOLUTION_LAYERS; ++j)
            if (CONVOLUTION_LAYER_INDICES[j] == i) return true;
        return false;
    }

    // Per-class layer index map — 'freqs' is hybrid: only 2/6 layers have it.
    static constexpr int FREQS_LAYER_INDICES[2] = {0, 1};
    static constexpr int NUM_FREQS_LAYERS = 2;
    inline static bool layer_has_freqs(int i) {
        for (int j = 0; j < NUM_FREQS_LAYERS; ++j)
            if (FREQS_LAYER_INDICES[j] == i) return true;
        return false;
    }

    // Per-class layer index map — 'layer_scale_1' is hybrid: only 2/6 layers have it.
    static constexpr int LAYER_SCALE_1_LAYER_INDICES[2] = {0, 1};
    static constexpr int NUM_LAYER_SCALE_1_LAYERS = 2;
    inline static bool layer_has_layer_scale_1(int i) {
        for (int j = 0; j < NUM_LAYER_SCALE_1_LAYERS; ++j)
            if (LAYER_SCALE_1_LAYER_INDICES[j] == i) return true;
        return false;
    }

    // Per-class layer index map — 'layer_scale_2' is hybrid: only 2/6 layers have it.
    static constexpr int LAYER_SCALE_2_LAYER_INDICES[2] = {0, 1};
    static constexpr int NUM_LAYER_SCALE_2_LAYERS = 2;
    inline static bool layer_has_layer_scale_2(int i) {
        for (int j = 0; j < NUM_LAYER_SCALE_2_LAYERS; ++j)
            if (LAYER_SCALE_2_LAYER_INDICES[j] == i) return true;
        return false;
    }
};
