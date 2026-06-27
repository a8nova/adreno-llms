// Auto-generated graph-mode model implementation for functiongemma-270m-it.
// Backbone: Gemma3ForCausalLM | total nodes captured: 313
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

std::vector<float> model_forward_graph(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
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
    return model_forward_graph(cl_ctx_, weights_, input_ids, start_pos);
}
