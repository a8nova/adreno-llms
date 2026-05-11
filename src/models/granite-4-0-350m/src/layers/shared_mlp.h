#pragma once
// Reference: model_info/transformers_src/modeling_granitemoehybrid.py:760-789 GraniteMoeHybridMLP.forward

#include "opencl_context.h"
#include "weights.h"

#include <CL/cl.h>
#include <string>

class SharedMlp {
public:
    SharedMlp(OpenCLContext& cl_ctx, Weights& weights, int layer_idx);
    ~SharedMlp();

    bool initialize();

    // input: [M, H]
    // output: [M, H]   = residual + (silu(gate)*up · w_out) * residual_scale
    // Step 8: residual + scale folded into swiglu_fused_output_v2 — caller no
    // longer needs to invoke element_add after this.
    cl_mem forward(cl_command_queue queue, cl_mem input, cl_mem residual, float residual_scale, int M);

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
    int layer_idx_;

    cl_program program_ = nullptr;
    cl_kernel silu_mul_kernel_ = nullptr;
    cl_kernel fused_output_kernel_ = nullptr;
    cl_kernel fused_output_kernel_v2_ = nullptr;  // GEMV-style WG=64 cooperative reduction
    // Step #5: split MLP into (1) fused gate+up+silu_mul → gated[N], (2) down-proj
    cl_kernel fused_gate_up_silu_kernel_ = nullptr;
    cl_kernel mlp_down_residual_kernel_v2_ = nullptr;
    // Persistent gated[intermediate] buffer for #5 fused path.
    cl_mem gated_ = nullptr;
    size_t gated_cap_bytes_ = 0;
    // Step #8: image2d_t variant of swiglu_fused_output_v2 — w_out_ wrapped
    // as a CL_RGBA + CL_HALF_FLOAT image to use Adreno's texture cache path.
    cl_kernel fused_output_kernel_v2_img_ = nullptr;
    cl_mem    w_out_img_ = nullptr;
    // Step #9: image2d_t for MLP gate_up (input_linear) — replaces the
    // pytorch_linear→gemv_m1_k1024_no4 buffer path with the image-backed
    // gemv_m1_k1024_no4_img.
    cl_mem    w_in_img_ = nullptr;

    // Weights
    cl_mem w_in_ = nullptr;   // [2*I, H] == [4096, 1024]
    cl_mem w_out_ = nullptr;  // [H, I]   == [1024, 2048]

    // Persistent scratch — input_linear projection [seq, 2*intermediate]. Pre-Step 4
    // this was clCreateBuffer'd per forward call. Sized geometrically.
    cl_mem proj_ = nullptr;
    size_t proj_cap_bytes_ = 0;
};
