// Reference: model_info/transformers_src/modeling_qwen2.py:15-33 Qwen2MLP
#pragma once

#include "opencl_context.h"
#include "weights.h"
#include <CL/cl.h>
#include <string>

class Mlp {
public:
    Mlp(OpenCLContext& cl_ctx, Weights& weights, const std::string& prefix, int layer_idx);
    ~Mlp();

    bool initialize();

    // Prefill path: returns a newly-allocated output buffer [seq_len, hidden].
    // Decode fast path: when seq_len==1 AND residual_dest != nullptr, the
    // down_proj is dispatched as a fused gemv+residual_add into
    // residual_dest, and residual_dest is returned (retained). Caller
    // detects the fused case via pointer equality (returned == residual_dest)
    // and skips the explicit element_add.
    cl_mem forward(cl_command_queue queue, cl_mem input, int seq_len,
                   cl_mem residual_dest = nullptr);

    // Decode fast path (optional): modifies residual in-place (residual += mlp(normed)).
    // Returns true on success.
    bool forward_decode_into_residual(cl_command_queue queue, cl_mem normed_in, cl_mem residual);

private:
    cl_mem create_buffer_(int rows, int cols);

    OpenCLContext& cl_ctx_;
    Weights& weights_;
    std::string prefix_;
    int layer_idx_ = -1;

    // Dimensions
    int hidden_size_ = 0;
    int intermediate_size_ = 0;

    // Weights
    cl_mem w_gate_ = nullptr;  // [intermediate, hidden]
    cl_mem w_up_ = nullptr;    // [intermediate, hidden]
    cl_mem w_down_ = nullptr;  // [hidden, intermediate]

    // Kernels
    cl_program program_ = nullptr;
    cl_kernel swiglu_kernel_ = nullptr;

    // Persistent activation buffers — lazy-allocated on first forward call,
    // grown on-demand if a later prefill needs more rows. Eliminates the
    // 3× clCreateBuffer + 3× clReleaseMemObject per layer per token at
    // decode (Mamba Step 6 lesson, +1.08×).
    cl_mem buf_gate_ = nullptr;
    cl_mem buf_up_   = nullptr;
    cl_mem buf_out_  = nullptr;
    int    buf_capacity_rows_ = 0;
    bool ensure_activation_buffers_(int max_rows);
};
