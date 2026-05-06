#pragma once
// Reference: model_info/transformers_src/modeling_llama.py:375-428 (LlamaModel.forward)
// Reference: model_info/transformers_src/modeling_llama.py:73-137 (LlamaRotaryEmbedding)
// Note: despite the HF class name, the actual token embedding lookup is
// `self.embed_tokens(input_ids)`; RoPE tables are handled in Attention.

#include <CL/cl.h>
#include "model_config.h"

class OpenCLContext;
class Weights;

class Embedding {
public:
    Embedding(OpenCLContext& cl_ctx, Weights& weights);
    ~Embedding();

    bool initialize();
    bool set_weights();

    // input_ids: device buffer int32 [seq_len]
    // returns: device buffer storage_t [seq_len, hidden]
    cl_mem forward(cl_command_queue queue, cl_mem input_ids, int seq_len);

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;

    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;

    cl_mem wte_ = nullptr; // model.embed_tokens.weight
};
