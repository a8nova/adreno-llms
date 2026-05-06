#pragma once
// Reference: model_info/transformers_src/modeling_llama.py:171-186 LlamaMLP

#include <CL/cl.h>

class OpenCLContext;
class Weights;

class Mlp {
public:
    Mlp(OpenCLContext& cl_ctx, Weights& weights, int layer_idx);
    ~Mlp();

    bool initialize();
    bool set_weights();

    // Forward: x[seq, hidden] -> out[seq, hidden]. Caller owns returned buffer.
    cl_mem forward(cl_command_queue queue, cl_mem input, int seq_len);

    // Decode fast path (M=1). Updates residual in-place.
    bool forward_decode_into_residual(cl_command_queue queue, cl_mem x, cl_mem residual);

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
    int layer_idx_;

    cl_program mlp_program_ = nullptr;
    cl_kernel silu_mul_kernel_ = nullptr;

    // Decode fast-path kernels (block_fused.cl)
    cl_program block_fused_prog_ = nullptr;
    cl_kernel fused_gate_up_silu_m1_ = nullptr;
    cl_kernel fused_gate_up_silu_m1_v4_ = nullptr;  // vec4 inner loop (buffer)
    cl_kernel fused_gate_up_silu_m1_v4_img_ = nullptr;  // vec4 + image2d_t (preferred)
    cl_kernel fused_down_res_m1_ = nullptr;

    // Persistent decode buffer — allocated once in initialize().
    cl_mem decode_act_buf_ = nullptr;  // [INTER]

    cl_mem w_gate_ = nullptr;
    cl_mem w_up_ = nullptr;
    cl_mem w_down_ = nullptr;

    // Image2d_t-backed view of w_down_ (Adreno texture cache).
    // K=INTERMEDIATE_SIZE=1536; output dim H=576 fits one image easily.
    cl_mem w_down_img_ = nullptr;
    cl_mem w_gate_img_ = nullptr;
    cl_mem w_up_img_   = nullptr;
    cl_kernel fused_down_no4_img_ = nullptr;  // fused_down_residual_m1_no4_img
    bool      down_img_ready_     = false;
    bool      gate_up_img_ready_  = false;
};
