#pragma once
// Vision depth-estimation model interface for depth-anything/Depth-Anything-V2-Small-hf.
// The full pipeline (DINOv2 backbone -> DPT neck -> depth head) lives in
// src/backbone.cpp::depth_forward.

#include "opencl_context.h"
#include "weights.h"
#include <vector>
#include <cstdint>

class Model {
public:
    Model(OpenCLContext& cl_ctx, Weights& weights);
    ~Model();

    bool initialize();

    // Vision entry point. pixel_values is a row-major [C, H, W] fp32 tensor
    // (batch 1). Returns the predicted depth map flattened row-major [H, W].
    std::vector<float> forward_depth(const std::vector<float>& pixel_values,
                                     int C, int H, int W);

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
};
