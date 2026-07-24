// Auto-generated graph-mode model implementation for KittenML/kitten-tts-nano-0.1.
// Backbone: unknown | total nodes captured: 0
//
// FRAMEWORK FILE — DO NOT EDIT (the agent restructures the encode/decode
// methods inside this file when wiring main.cpp for enc-dec models).
// Model::forward() delegates to model_forward_graph(...) which is provided
// by the agent in src/backbone.cpp.

#include "model.h"
#include "model_config.h"
#include "debug_utils.h"
#include "forward_dispatch.h"

#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

std::vector<float> model_forward_graph(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    const std::vector<float>& style,
    const std::vector<float>& rng_uniform,
    const std::vector<float>& rng_normal,
    int start_pos);

Model::Model(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {}

Model::~Model() {
}

bool Model::initialize() {
    NNOPT_CHECKPOINT("Model::initialize() — graph mode");
    return true;
}

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids, int start_pos) {
    NNOPT_CHECKPOINT("Model::forward() — graph mode (delegating to model_forward_graph)");
    // Legacy text-LM wrapper path: no aux fixtures. TTS callers use
    // forward_graph() below which threads style/rng into the graph.
    static const std::vector<float> kNoFixture;
    return model_forward_graph(cl_ctx_, weights_, input_ids,
                               kNoFixture, kNoFixture, kNoFixture, start_pos);
}

int Model::forward_graph(
    const std::vector<int32_t>& input_ids,
    const std::vector<float>& style,
    const std::vector<float>& rng_uniform,
    const std::vector<float>& rng_normal,
    std::vector<int16_t>& pcm)
{
    NNOPT_CHECKPOINT("Model::forward_graph() — TTS single-shot");
    std::vector<float> wav = model_forward_graph(
        cl_ctx_, weights_, input_ids, style, rng_uniform, rng_normal, /*start_pos=*/0);
    if (wav.empty()) {
        NNOPT_ERROR("Model::forward_graph: graph produced empty waveform");
        return 1;
    }
    // Convert float waveform in [-1,1] to 16-bit signed PCM with clamping.
    pcm.resize(wav.size());
    for (size_t i = 0; i < wav.size(); i++) {
        float s = wav[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        pcm[i] = static_cast<int16_t>(std::lround(s * 32767.0f));
    }
    return 0;
}
