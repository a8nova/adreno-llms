// Reference: model_info/transformers_src/modeling_qwen2.py:177-198 Qwen2RMSNorm.forward
#pragma once

#include "opencl_context.h"
#include "weights.h"
#include "model_config.h"

#include <CL/cl.h>
#include <string>

// NOTE: Despite the file name, this implements RMSNorm for Qwen2.
class LayerNorm {
public:
    LayerNorm(OpenCLContext& cl_ctx, Weights& weights, const std::string& weight_key,
              int hidden_size, float eps);
    ~LayerNorm();

    bool initialize();
    cl_mem forward(cl_command_queue queue, cl_mem input, int seq_len);

    // Accessor for the persistent output buffer — used by the
    // recordable_queues replay path to feed final_norm's output into
    // the post-recording lm_head dispatch on the live queue (without
    // re-running forward).
    cl_mem buf_out_value() const { return buf_out_; }

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
    std::string weight_key_;
    int hidden_size_ = 0;
    float eps_ = 1e-6f;

    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;

    cl_mem weight_buf_ = nullptr;

    // Persistent output buffer — eliminates one clCreateBuffer per forward.
    // Each LayerNorm instance has its own; lazy-allocated, grows on demand.
    cl_mem buf_out_ = nullptr;
    int    buf_capacity_rows_ = 0;
};
