#pragma once
// Reference: model_info/modeling_openelm.py:263-404 OpenELMMultiHeadCausalAttention.forward
//
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
    // Returns: cl_mem of shape [seq_q, hidden] — BORROWED handle, valid until
    //          the next call to this layer's forward(). Caller MUST NOT release.
    cl_mem forward(cl_command_queue queue,
                   cl_mem input,
                   cl_mem cos,
                   cl_mem sin,
                   int seq_q,
                   int start_pos);

    // Convenience overload that uses the layer's own lazy RoPE tables
    // (cos_/sin_). Equivalent to forward(queue, input, nullptr, nullptr, ...).
    cl_mem forward(cl_command_queue queue,
                   cl_mem input,
                   int seq_q,
                   int start_pos);

private:
    bool ensure_rope_tables(int seq_len);
    bool ensure_kv_cache();   // Allocates k_cache_/v_cache_ once, on first forward.

    // Lazy/grow allocator for the per-call activation buffers below. Sized for
    // (seq_q, seq_k) where seq_k = start_pos + seq_q. Per-buffer capacities are
    // tracked in *_cap_bytes_ so a larger prefill grows them and decode reuses
    // without reallocating. Eliminates ~9 clCreateBuffer + clReleaseMemObject
    // round-trips per token (~50 µs each on Adreno) — Step 2 lever.
    bool ensure_activation_buffers_(int seq_q, int seq_k);
    void release_activation_buffers_();

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

    // ── Persistent activation buffers (Step 2). Grown on demand by
    // ensure_activation_buffers_; never resized down. Returned forward()
    // hands out buf_proj_ as a BORROWED handle — caller must NOT release.
    cl_mem buf_qkv_      = nullptr;   // [seq_q, QKV_DIM]
    cl_mem buf_q_        = nullptr;   // [seq_q, Q_DIM]
    cl_mem buf_k_        = nullptr;   // [seq_q, KV_DIM]
    cl_mem buf_v_        = nullptr;   // [seq_q, KV_DIM]
    cl_mem buf_qn_       = nullptr;   // [seq_q, Q_DIM]   (q post q_norm)
    cl_mem buf_kn_       = nullptr;   // [seq_q, KV_DIM]  (k post k_norm)
    cl_mem buf_scores_   = nullptr;   // [QH, seq_q, seq_k]
    cl_mem buf_ctx_out_  = nullptr;   // [seq_q, Q_DIM]
    cl_mem buf_proj_     = nullptr;   // [seq_q, HIDDEN]   (returned to caller)
    size_t buf_qkv_cap_      = 0;
    size_t buf_q_cap_        = 0;
    size_t buf_k_cap_        = 0;
    size_t buf_v_cap_        = 0;
    size_t buf_qn_cap_       = 0;
    size_t buf_kn_cap_       = 0;
    size_t buf_scores_cap_   = 0;
    size_t buf_ctx_out_cap_  = 0;
    size_t buf_proj_cap_     = 0;
};
