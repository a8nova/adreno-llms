// Graph-mode model implementation for depth-anything/Depth-Anything-V2-Small-hf.
// Model::forward delegates to model_forward_graph() in src/backbone.cpp.

#include "model.h"
#include "model_config.h"
#include "debug_utils.h"

#include <vector>
#include <cstdint>

// Provided by src/backbone.cpp — the full DINOv2->neck->head depth pipeline.
std::vector<float> depth_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<float>& pixel_values,
    int C, int H, int W);

Model::Model(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {}

Model::~Model() {}

bool Model::initialize() {
    NNOPT_CHECKPOINT("Model::initialize() — depth pipeline (lazy-init in backbone)");
    return true;
}

std::vector<float> Model::forward_depth(const std::vector<float>& pixel_values,
                                        int C, int H, int W) {
    NNOPT_CHECKPOINT("Model::forward_depth — delegating to depth_forward");
    return depth_forward(cl_ctx_, weights_, pixel_values, C, H, W);
}
