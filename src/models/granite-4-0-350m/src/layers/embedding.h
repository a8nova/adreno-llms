#pragma once

// Reference: model_info/transformers_src/modeling_granitemoehybrid.py:1151-1210 GraniteMoeHybridModel.forward (embed_tokens usage)

#include "opencl_context.h"
#include "weights.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstdint>
#include <vector>

class Embedding {
public:
    Embedding(OpenCLContext& cl_ctx, Weights& weights);
    ~Embedding();

    bool initialize();

    // token_ids: device-side int32 token IDs buffer
    // seq_len: number of tokens in token_ids
    // start_pos: absolute position of token_ids[0] in the full sequence
    // Returns: [seq_len, hidden] storage buffer (nnopt_storage_t); caller owns.
    cl_mem forward(cl_command_queue queue, cl_mem token_ids, int seq_len, int start_pos);

    // RoPE tables for Attention::rope_apply_qk
    // layout: [MODEL_CONFIG::MAX_SEQ_LEN, MODEL_CONFIG::HEAD_DIM] in storage_t
    cl_mem rope_cos() const { return rope_cos_; }
    cl_mem rope_sin() const { return rope_sin_; }

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;

    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;

    cl_mem embed_tokens_weight_ = nullptr;

    // Granite uses RoPE (no absolute wpe table). embedding.cl still expects a
    // wpe buffer arg; we pass a dummy buffer (same as wte).
    cl_mem position_weight_ = nullptr;

    cl_mem rope_cos_ = nullptr;
    cl_mem rope_sin_ = nullptr;
};
