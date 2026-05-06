#pragma once
// Reference: model_info/transformers_src/modeling_llama.py:445-501 LlamaForCausalLM.forward
// lm_head is a simple linear projection from hidden_size -> vocab.

#include <CL/cl.h>

class OpenCLContext;
class Weights;

class LmHead {
public:
    LmHead(OpenCLContext& cl_ctx, Weights& weights);
    ~LmHead();

    bool initialize();

    // hidden: [M, hidden]
    // returns: [M, vocab] (caller owns)
    cl_mem forward(cl_command_queue queue, cl_mem hidden, int M);

private:
    bool set_weights();

    OpenCLContext& cl_ctx_;
    Weights& weights_;

    cl_mem w_ = nullptr; // aliased from embed_tokens.weight if tied

    // Decode fast-path GEMV (M=1) — dispatched automatically from forward() when M==1.
    cl_program block_fused_prog_ = nullptr;
    cl_kernel fused_lm_head_m1_ = nullptr;
};
