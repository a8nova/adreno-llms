// Graph-mode model implementation for hexgrad/Kokoro-82M.
//
// FRAMEWORK FILE — DO NOT EDIT for per-op code; the ops live in src/ops/.

#include "model.h"
#include "model_config.h"
#include "debug_utils.h"
#include "forward_dispatch.h"

#include <vector>
#include <cstdint>

// Provided by src/ops/backbone.cpp.
int model_forward_graph_tts(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    const std::vector<float>& ref_s,
    std::vector<int16_t>& out_pcm_int16);

Model::Model(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {}

Model::~Model() = default;

bool Model::initialize() {
    NNOPT_CHECKPOINT("Model::initialize() — graph mode");
    return true;
}

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids, int start_pos) {
    (void)input_ids; (void)start_pos;
    NNOPT_ERROR("Model::forward() called in TTS port — use forward_graph instead");
    return std::vector<float>(MODEL_CONFIG::N_TOKEN, 0.0f);
}

int Model::forward_graph(const std::vector<int32_t>& input_ids,
                         const std::vector<float>& ref_s,
                         std::vector<int16_t>& out_pcm_int16) {
    NNOPT_CHECKPOINT("Model::forward_graph() — TTS graph mode");
    return model_forward_graph_tts(cl_ctx_, weights_, input_ids, ref_s, out_pcm_int16);
}
