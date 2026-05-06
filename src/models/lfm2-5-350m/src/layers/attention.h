#pragma once
// Reference: model_info/transformers_src/modeling_lfm2.py:226-283 Lfm2Attention.forward
// Auto-generated transformer Attention scaffold.
//
// Rules embedded in this file:
//   KV-01 (k_cache_/v_cache_ persistent class members)
//   ROPE-01 (cos_/sin_ rope tables, computed once in initialize())
//   PROG-01 (cl_program members, built once in initialize())
//   SYNC-01 (no clFinish in forward — see attention.cpp)

#include <CL/cl.h>
#include "model_config.h"

#include <string>

class OpenCLContext;
class Weights;

class Attention {
public:
    Attention(OpenCLContext& cl_ctx, Weights& weights, int layer_idx);
    Attention(OpenCLContext& cl_ctx, Weights& weights, int layer_idx, const std::string& layer_prefix);
    ~Attention();

    bool initialize();

    // Forward — handles both prefill and decode via the start_pos param.
    // input: [seq_q, hidden]
    // returns: [seq_q, hidden]
    cl_mem forward(cl_command_queue queue,
                   cl_mem input,
                   int seq_q,
                   int start_pos);

private:
    bool ensure_rope_tables(cl_command_queue queue, int seq_len);
    bool ensure_kv_cache();

    OpenCLContext& cl_ctx_;
    Weights& weights_;
    int layer_idx_;
    std::string layer_prefix_;

    cl_mem cos_ = nullptr;
    cl_mem sin_ = nullptr;
    int rope_seq_len_ = 0;

    cl_mem k_cache_ = nullptr;
    cl_mem v_cache_ = nullptr;

    cl_program rope_program_ = nullptr;
    cl_kernel rope_kernel_ = nullptr;

    cl_program attn_program_ = nullptr;
    cl_kernel scores_kernel_ = nullptr;
    cl_kernel softmax_kernel_ = nullptr;
    cl_kernel out_kernel_ = nullptr;

    // Per-head RMSNorm for query/key (Lfm2 q_layernorm/k_layernorm,
    // applied BEFORE RoPE on the head_dim axis).
    cl_program rmsnorm_program_ = nullptr;
    cl_kernel rmsnorm_kernel_ = nullptr;

    // weights
    cl_mem wq_ = nullptr;
    cl_mem wk_ = nullptr;
    cl_mem wv_ = nullptr;
    cl_mem wo_ = nullptr;
    cl_mem q_ln_w_ = nullptr;
    cl_mem k_ln_w_ = nullptr;

    // Persistent activation buffers — borrowed handle on return.
    cl_mem buf_q_         = nullptr;  // [seq_q, Q_DIM]
    cl_mem buf_k_         = nullptr;  // [seq_q, KV_DIM]
    cl_mem buf_v_         = nullptr;  // [seq_q, KV_DIM]
    cl_mem buf_q_norm_tmp_ = nullptr; // [seq_q, Q_DIM]   for q_layernorm
    cl_mem buf_k_norm_tmp_ = nullptr; // [seq_q, KV_DIM]  for k_layernorm
    cl_mem buf_scores_    = nullptr;  // [QH, seq_q, seq_k_max]
    cl_mem buf_attn_out_  = nullptr;  // [seq_q, Q_DIM]
    cl_mem buf_proj_      = nullptr;  // [seq_q, H]
    int    buf_seq_q_capacity_ = 0;
    int    buf_seq_k_capacity_ = 0;
    bool   ensure_buffers_(int seq_q, int seq_k);
};
