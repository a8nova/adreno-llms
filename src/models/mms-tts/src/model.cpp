// Auto-generated model implementation for facebook/mms-tts-eng.
// Modality: TTS (VITS).
//
// Model::forward_graph() delegates to tts_forward_graph(...) implemented by
// the agent in src/ops/backbone.cpp (per docs/MODALITY_TTS.md).

#include "model.h"
#include "model_config.h"
#include "debug_utils.h"
#include "forward_dispatch.h"

#include <vector>
#include <cstdint>

extern "C" int tts_forward_graph(OpenCLContext& cl_ctx,
                                  Weights& weights,
                                  const std::vector<int32_t>& input_ids,
                                  const std::vector<float>& duration_noise,
                                  const std::vector<float>& prior_noise,
                                  std::vector<int16_t>& out_pcm_int16);

Model::Model(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {}

Model::~Model() = default;

bool Model::initialize() {
    NNOPT_CHECKPOINT("Model::initialize() — TTS graph");
    return true;
}

int Model::forward_graph(const std::vector<int32_t>& input_ids,
                         const std::vector<float>& duration_noise,
                         const std::vector<float>& prior_noise,
                         std::vector<int16_t>& out_pcm_int16) {
    NNOPT_CHECKPOINT("Model::forward_graph() — delegating to tts_forward_graph");
    return tts_forward_graph(cl_ctx_, weights_, input_ids, duration_noise, prior_noise, out_pcm_int16);
}
