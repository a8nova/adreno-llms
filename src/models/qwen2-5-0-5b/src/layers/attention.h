#pragma once
// Reference: model_info/transformers_src/modeling_qwen2.py:64-176 Qwen2Attention.forward
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
    //   counter:   persistent cl_mem with [0]=start_pos, [1]=seq_k. Written by
    //              host via clEnqueueWriteBuffer before each decode step; kernels
    //              (rope, kv_write, attn_scores, softmax, attn_out) read from it.
    // Returns: cl_mem of shape [seq_q, hidden]; CALLER owns and releases.
    cl_mem forward(cl_command_queue queue,
                   cl_mem input,
                   cl_mem cos,
                   cl_mem sin,
                   int seq_q,
                   int start_pos,
                   cl_mem counter,
                   cl_mem residual_dest = nullptr);

    // Pre-allocate buf_scores_ at MAX_POSITION_EMBEDDINGS capacity so the
    // buffer address is stable across recording replay iterations. Call this
    // once before building the cl_qcom_recordable_queues recording.
    void preallocate_decode_buffers_max();

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

    // Bias-add for q/k/v projections (Qwen2Attention has biases).
    cl_program bias_program_ = nullptr;
    cl_kernel bias_kernel_ = nullptr;

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

    // Biases (Qwen2Attention: q/k/v have bias; o_proj has no bias)
    cl_mem bq_ = nullptr;
    cl_mem bk_ = nullptr;
    cl_mem bv_ = nullptr;

    // Persistent activation buffers — lazy-allocated on first forward,
    // grown on-demand for prefill. Eliminates 6 alloc/free per layer
    // per token at decode (Mamba Step 6 lesson).
    //   buf_q_       [seq, Q_DIM]
    //   buf_k_       [seq, KV_DIM]   (rotated, then copied into k_cache_)
    //   buf_v_       [seq, KV_DIM]   (copied into v_cache_)
    //   buf_scores_  [QH, seq, seq_k]
    //   buf_attn_o_  [seq, Q_DIM]
    //   buf_proj_    [seq, H]
    cl_mem buf_q_       = nullptr;
    cl_mem buf_k_       = nullptr;
    cl_mem buf_v_       = nullptr;
    cl_mem buf_scores_  = nullptr;
    cl_mem buf_attn_o_  = nullptr;
    cl_mem buf_proj_    = nullptr;
    int    buf_capacity_seq_q_ = 0;
    int    buf_capacity_seq_k_ = 0;
    bool ensure_activation_buffers_(int seq_q, int seq_k);
};
