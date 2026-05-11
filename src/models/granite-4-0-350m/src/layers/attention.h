#pragma once
// Reference: model_info/transformers_src/modeling_granitemoehybrid.py:121-191 GraniteMoeHybridAttention

#include "opencl_context.h"
#include "weights.h"
#include <CL/cl.h>
#include <string>

class Attention {
public:
    Attention(OpenCLContext& cl_ctx, Weights& weights, int layer_idx);
    ~Attention();

    bool initialize();

    // input: [seq_len, hidden]
    // returns: [seq_len, hidden]
    // start_pos: absolute position offset for RoPE + KV cache write
    // cos/sin: RoPE tables from Embedding (shape [MAX_SEQ_LEN, HEAD_DIM])
    cl_mem forward(cl_command_queue queue, cl_mem input, int seq_len, int start_pos, cl_mem cos, cl_mem sin);

private:
    bool ensure_kv_cache(cl_command_queue queue);

    OpenCLContext& cl_ctx_;
    Weights& weights_;
    int layer_idx_;

    cl_program program_ = nullptr;
    cl_kernel rope_kernel_ = nullptr;
    cl_kernel gqa_scores_kernel_ = nullptr;
    cl_kernel gqa_softmax_kernel_ = nullptr;
    cl_kernel gqa_out_kernel_ = nullptr;

    cl_mem q_w_ = nullptr;
    cl_mem k_w_ = nullptr;
    cl_mem v_w_ = nullptr;
    cl_mem o_w_ = nullptr;

    cl_mem k_cache_ = nullptr;
    cl_mem v_cache_ = nullptr;

    // Persistent scratch buffers — sized geometrically by ensure_cap_ helper
    // in the .cpp. Reused across forward() calls; freed in destructor. Pre-Step 4
    // these were clCreateBuffer'd every call (2 allocs × 28 layers × N tokens =
    // ~1800 allocs per 32-token decode, ~2.3 ms each on Adreno).
    cl_mem attn_scores_ = nullptr;
    size_t attn_scores_cap_bytes_ = 0;
    cl_mem attn_ctx_ = nullptr;
    size_t attn_ctx_cap_bytes_ = 0;
};
