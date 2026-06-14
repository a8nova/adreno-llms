#pragma once
// Auto-generated graph-mode model interface for hexgrad/Kokoro-82M.
//
// In graph mode the Model class is intentionally thin. The per-op code that
// implements the forward() data flow lives in src/ops/*.cpp.

#include "opencl_context.h"
#include "weights.h"
#include "sampler.h"
#include "tokenizer.h"
#include <vector>
#include <cstdint>

class Model {
public:
    Model(OpenCLContext& cl_ctx, Weights& weights);
    ~Model();

    bool initialize();

    // Not used for TTS (errors out if called).
    std::vector<float> forward(const std::vector<int32_t>& input_ids, int start_pos);

    // Graph-mode TTS: single-shot forward.
    //   input_ids: tokenized phoneme ids including BOS/EOS (length T).
    //   ref_s:     256 floats — the per-length style vector indexed from a
    //              Kokoro voice pack at index (T-1). Predictor uses
    //              ref_s[128:], decoder uses ref_s[:128].
    int forward_graph(const std::vector<int32_t>& input_ids,
                      const std::vector<float>& ref_s,
                      std::vector<int16_t>& out_pcm_int16);

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
};
