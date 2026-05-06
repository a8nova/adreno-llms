// Reference: model_info/transformers_src/modeling_mamba2.py (Mamba2ForCausalLM.forward uses lm_head linear)
#pragma once

#include "opencl_context.h"
#include "weights.h"
#include <string>

class LmHead {
public:
    LmHead(OpenCLContext& cl_ctx, Weights& weights,
           const std::string& weight_key,
           int hidden_size,
           int vocab_size);
    ~LmHead();

    bool initialize();

    // Lever 4: returns a BORROWED handle to the persistent logits buffer
    // owned by this LmHead. Caller MUST NOT release it. The buffer is
    // overwritten on the next forward() call.
    cl_mem forward(cl_command_queue queue, cl_mem hidden, int seq_len);

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
    std::string weight_key_;
    int hidden_size_ = 0;
    int vocab_size_ = 0;

    cl_mem w_ = nullptr;

    int   buf_capacity_rows_ = 0;
    cl_mem buf_logits_ = nullptr;

    bool ensure_logits_(int rows, int padded_vocab);
};
