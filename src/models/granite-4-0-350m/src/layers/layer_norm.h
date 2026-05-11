#pragma once
// Reference: model_info/transformers_src/modeling_granitemoehybrid.py:735-752 GraniteMoeHybridRMSNormGated.forward

#include "opencl_context.h"
#include "weights.h"
#include "model_config.h"

#include <CL/cl.h>
#include <string>

class LayerNorm {
public:
    enum class Kind {
        InputLayerNorm,
        PostAttentionLayerNorm,
        FinalNorm,
    };

    LayerNorm(OpenCLContext& cl_ctx, Weights& weights, int layer_idx, Kind kind);
    ~LayerNorm();

    bool initialize();

    // hidden_states: [rows, hidden]
    cl_mem forward(cl_command_queue queue, cl_mem hidden_states, int rows);

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
    int layer_idx_;
    Kind kind_;

    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;
    cl_kernel kernel_v2_ = nullptr;  // WG=64 cooperative reduction

    cl_mem weight_ = nullptr;  // owned by Weights

    bool set_weights();
    std::string weight_key() const;
};
