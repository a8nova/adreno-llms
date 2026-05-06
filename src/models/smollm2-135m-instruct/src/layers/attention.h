#pragma once
// Reference: model_info/transformers_src/modeling_llama.py:LlamaAttention.forward
// Auto-generated transformer Attention scaffold.
//
// Rules embedded in this file:
//   KV-01 (k_cache_/v_cache_ persistent class members)
//   ROPE-01 (cos_/sin_ rope tables, computed once in initialize())
//   PROG-01 (cl_program members, built once in initialize())
//   SYNC-01 (no clFinish in forward — see attention.cpp)
//
// The agent fills in: per-model weight keys (q_proj, k_proj, v_proj, o_proj
// for Llama-family; c_attn for GPT-2 family), bias handling if present,
// and any architecture-specific RoPE variants. The KV cache, signature,
// and batched-kernel call sites are scaffold-emitted starting points.

#include <CL/cl.h>
#include "model_config.h"

class OpenCLContext;
class Weights;

class Attention {
public:
    Attention(OpenCLContext& cl_ctx, Weights& weights, int layer_idx);
    ~Attention();

    bool initialize();
    bool set_weights();

    // Forward — handles both prefill and decode via the start_pos param.
    //   input:     [seq_q, hidden]  (post-norm hidden states for the new tokens only)
    //   cos, sin:  optional precomputed RoPE tables; pass nullptr to use
    //              the layer's own tables (created lazily in ensure_rope_tables).
    //   seq_q:     number of NEW tokens being processed (P at prefill, 1 at decode).
    //   start_pos: absolute position of the first new token. 0 at prefill.
    //              Determines RoPE rotation, KV cache write offset, causal mask.
    // Returns: cl_mem of shape [seq_q, hidden]; CALLER owns and releases.
    cl_mem forward(cl_command_queue queue,
                   cl_mem input,
                   cl_mem cos,
                   cl_mem sin,
                   int seq_q,
                   int start_pos);

    // Decode fast path (M=1). Replaces CLBlast QKV + o_proj GEMMs with
    // GEMV kernels from block_fused.cl. Updates residual in-place.
    bool forward_decode_into_residual(cl_command_queue queue,
                                      cl_mem x, int start_pos,
                                      cl_mem residual);

private:
    bool ensure_rope_tables(cl_command_queue queue, int seq_len);
    bool ensure_kv_cache();   // Allocates k_cache_/v_cache_ once, on first forward.

    OpenCLContext& cl_ctx_;
    Weights& weights_;
    int layer_idx_;

    // ── Persistent RoPE tables (ROPE-01).
    // Generated once for [seq_len, head_dim] in fp16 storage. Re-allocated
    // only when seq_len grows (rope_seq_len_ tracks current capacity).
    cl_mem cos_ = nullptr;
    cl_mem sin_ = nullptr;
    int rope_seq_len_ = 0;

    // ── KV cache (KV-01).
    // Layout: [MAX_POSITION_EMBEDDINGS, num_kv_heads * head_dim], token-major.
    // Allocated once in initialize(); written at offset (start_pos * KV_DIM)
    // per forward call; read up to (start_pos + seq_q) for attention.
    cl_mem k_cache_ = nullptr;
    cl_mem v_cache_ = nullptr;

    // ── Persistent cl_program members (PROG-01) — built once in initialize().
    cl_program rope_program_ = nullptr;
    cl_kernel rope_kernel_ = nullptr;

    cl_program attn_program_ = nullptr;
    cl_kernel scores_kernel_ = nullptr;          // gqa_attn_scores
    cl_kernel softmax_kernel_ = nullptr;         // gqa_softmax
    cl_kernel out_kernel_ = nullptr;             // gqa_attn_out

    // ── Decode fast-path kernels (block_fused.cl) — built in initialize().
    cl_program block_fused_prog_ = nullptr;
    cl_kernel fused_qkv_m1_ = nullptr;              // fused_qkv_gemv_m1
    cl_kernel fused_rope_kvwrite_m1_ = nullptr;     // fused_rope_kvwrite_m1
    cl_kernel fused_decode_attn_m1_ = nullptr;      // scores+softmax+attn_out fused (NEW)
    cl_kernel fused_oproj_res_m1_ = nullptr;        // fused_oproj_residual_m1

    // ── Persistent decode buffers — allocated once in initialize(), reused each step.
    cl_mem decode_q_buf_        = nullptr;  // [Q_DIM]
    cl_mem decode_k_buf_        = nullptr;  // [KV_DIM]
    cl_mem decode_v_buf_        = nullptr;  // [KV_DIM]
    cl_mem decode_scores_buf_   = nullptr;  // [QH * MAX_POSITION_EMBEDDINGS] (unused — scores in local mem)
    cl_mem decode_attn_out_buf_ = nullptr;  // [Q_DIM]

    // ── Weights — agent fills in per-model. Common patterns:
    //   Llama family: wq_, wk_, wv_, wo_  (separate q_proj/k_proj/v_proj)
    //   GPT-2 family: w_qkv_ (combined c_attn weight), w_proj_
    //   Phi family:   w_qkv_ (Wqkv), w_proj_
    // Update this list to match the model's actual weight keys (see the
    // (model metadata) contract for exact strings).
    cl_mem wq_ = nullptr;
    cl_mem wk_ = nullptr;
    cl_mem wv_ = nullptr;
    cl_mem wo_ = nullptr;
};
