#pragma once
// Reference: model_info/transformers_src/modeling_llama.py:53-72 LlamaRMSNorm

#include <CL/cl.h>
#include "model_config.h"
#include <string>

class OpenCLContext;
class Weights;

class LayerNorm {
public:
    // is_post_attn selects model.layers.{i}.post_attention_layernorm.weight vs input_layernorm.weight
    // is_final_norm selects model.norm.weight (RMS norm after all layers)
    LayerNorm(OpenCLContext& cl_ctx, Weights& weights, int layer_idx, bool is_post_attn);
    LayerNorm(OpenCLContext& cl_ctx, Weights& weights, int layer_idx, bool is_post_attn, bool is_final_norm);
    ~LayerNorm();

    bool initialize();
    bool set_weights();

    // input:  [M, HIDDEN_SIZE]
    // output: [M, HIDDEN_SIZE]  (caller owns returned cl_mem)
    cl_mem forward(cl_command_queue queue, cl_mem input, int M);

    // Decode fast path (M=1). Returns the persistent decode_out_buf_.
    // IMPORTANT: caller must NOT release the returned cl_mem — it is owned
    // by this LayerNorm instance and reused across decode steps.
    cl_mem forward_decode(cl_command_queue queue, cl_mem input);

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
    int layer_idx_;
    bool is_post_attn_;
    bool is_final_norm_;

    std::string weight_key_;

    cl_mem weight_ = nullptr;        // owned by Weights
    cl_mem decode_out_buf_ = nullptr; // [HIDDEN_SIZE] — persistent decode output (non-owned by callers)

    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;     // rmsnorm_forward (kernels/rmsnorm.cl)
};
