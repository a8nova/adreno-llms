// Reference: model_info/transformers_src/modeling_mamba2.py (RMSNorm/LayerNorm usage in Mamba2Model)
#pragma once

#include "opencl_context.h"
#include "weights.h"

#include <CL/cl.h>
#include <string>

class LayerNorm {
public:
    LayerNorm(OpenCLContext& cl_ctx,
              Weights& weights,
              const std::string& weight_key,
              int hidden_size,
              float eps,
              int layer_idx);
    ~LayerNorm();

    bool initialize();

    // input/output are [seq_len, hidden_size]
    cl_mem forward(cl_command_queue queue, cl_mem input, int seq_len);

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
    std::string weight_key_;
    int hidden_size_ = 0;
    float eps_ = 1e-5f;
    int layer_idx_ = -1;

    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;

    cl_mem gamma_ = nullptr;

    // Lever 4: persistent output buffer. Returned cl_mem is BORROWED — caller
    // must NOT release. Sized for max(seq_len) ever seen on this instance.
    int   buf_capacity_rows_ = 0;
    cl_mem buf_out_ = nullptr;

    bool ensure_out_(int rows);
};
