// Auto-generated graph dispatch header.
// Each ported node provides a function declaration here. The agent appends
// to this file via PortNode/MarkNodePorted as nodes are ported.
//
// Model: myshell-ai/OpenVoiceV2 (unknown)
// Total nodes: 0
#pragma once

#include "opencl_context.h"
#include "weights.h"
#include <CL/cl.h>
#include <vector>

// ── Root entry point — implemented in src/ops/backbone.cpp ──
// The framework's Model::forward() in src/model.cpp delegates to this
// function. The agent implements it when porting the root graph node
// (the model class — typically OpenELMModel / LlamaModel / etc.). It
// composes the per-class ops in execution order:
//
//   embedding(input_ids) → loop decoder layers → final norm → lm_head → logits
//
// Until backbone.cpp defines this symbol, the linker fails with an
// unresolved-symbol error — that is the SIGNAL to port the root node.
std::vector<float> model_forward_graph(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    int start_pos);

// Graph-node forward declarations are appended below by the agent as
// each node is ported. Each declaration follows the pattern:
//
//   cl_mem op_<sanitized_node_id>(OpenCLContext& cl_ctx,
//                                  Weights& weights,
//                                  cl_command_queue queue,
//                                  cl_mem input,
//                                  /* per-node shape args */);

// === BEGIN AGENT-MAINTAINED DECLARATIONS ===
// PortNode appends one extern "C" declaration per ported class BEFORE
// the SENTINEL line below. Use the Edit tool with:
//   old_string = "// __NNOPT_DISPATCH_APPEND_SENTINEL__\n"
//   new_string = <your new decl>\n// __NNOPT_DISPATCH_APPEND_SENTINEL__\n
// This sentinel never shifts as declarations accumulate, so the Edit
// always matches. Do NOT delete the sentinel — it's load-bearing.

// ── Universal plumbing-op signatures (pre-emitted) ───────────────
// These ops appear in nearly every Llama-family + VLM port. Pre-
// declaring them eliminates the per-cycle "undeclared identifier"
// cascade that used to cost the agent 4–6 build iterations. If a
// port needs a different signature for one of these, EDIT the decl
// below — do NOT add a duplicate. Vision-family ops vary by model
// (Idefics3 / SigLIP / Phi3-V / Qwen2-VL …) so they remain agent-
// appended below the sentinel.

extern "C" cl_mem op_Embedding(OpenCLContext& cl_ctx,
                                Weights& weights,
                                cl_command_queue queue,
                                cl_mem input_ids_i32,
                                int num_tokens,
                                int hidden_size,
                                const char* weight_key_wte);

extern "C" cl_mem op_Linear(OpenCLContext& cl_ctx,
                             Weights& weights,
                             cl_command_queue queue,
                             cl_mem input,
                             int rows,
                             int in_features,
                             int out_features,
                             const char* weight_key_w,
                             const char* weight_key_b_optional);

extern "C" cl_mem op_LayerNorm(OpenCLContext& cl_ctx,
                                Weights& weights,
                                cl_command_queue queue,
                                cl_mem input,
                                int rows,
                                int cols,
                                float eps,
                                const char* weight_key_weight,
                                const char* weight_key_bias);

extern "C" cl_mem op_LlamaRMSNorm(OpenCLContext& cl_ctx,
                                   Weights& weights,
                                   cl_command_queue queue,
                                   cl_mem input,
                                   int rows,
                                   int cols,
                                   float eps,
                                   const char* weight_key_w);

extern "C" cl_mem op_Conv2d(OpenCLContext& cl_ctx,
                             Weights& weights,
                             cl_command_queue queue,
                             cl_mem input,
                             int N,
                             int Cin,
                             int Hin,
                             int Win,
                             int Cout,
                             int Kh,
                             int Kw,
                             int stride_h,
                             int stride_w,
                             int pad_h,
                             int pad_w,
                             const char* weight_key_w,
                             const char* weight_key_b_optional);

extern "C" cl_mem op_PytorchGELUTanh(OpenCLContext& cl_ctx,
                                      Weights& weights,
                                      cl_command_queue queue,
                                      cl_mem input,
                                      int n_elements);

extern "C" cl_mem op_LlamaSdpaAttention(OpenCLContext& cl_ctx,
                                         Weights& weights,
                                         cl_command_queue queue,
                                         cl_mem hidden_states,
                                         int rows,
                                         int hidden_size,
                                         int num_q_heads,
                                         int num_kv_heads,
                                         int head_dim,
                                         int start_pos,
                                         const char* q_w,
                                         const char* k_w,
                                         const char* v_w,
                                         const char* o_w);

extern "C" cl_mem op_LlamaMLP(OpenCLContext& cl_ctx,
                               Weights& weights,
                               cl_command_queue queue,
                               cl_mem input,
                               int rows,
                               int hidden_size,
                               int intermediate_size,
                               const char* gate_w,
                               const char* up_w,
                               const char* down_w);

extern "C" cl_mem op_LlamaDecoderLayer(OpenCLContext& cl_ctx,
                                        Weights& weights,
                                        cl_command_queue queue,
                                        cl_mem hidden_states,
                                        int rows,
                                        int hidden_size,
                                        int intermediate_size,
                                        int num_q_heads,
                                        int num_kv_heads,
                                        int head_dim,
                                        int start_pos,
                                        const char* in_norm_w,
                                        const char* post_norm_w,
                                        float rms_eps,
                                        const char* q_w,
                                        const char* k_w,
                                        const char* v_w,
                                        const char* o_w,
                                        const char* gate_w,
                                        const char* up_w,
                                        const char* down_w);

// ── Per-layer block ops (auto-derived from contract.submodels[lm]) ───────
// Model type: unknown  weight_prefix: ""
// Layer kinds present: full_attention  (over 1 layers)
// Implement each block-fn body in src/ops/<fnname>.cpp; the dispatch loop
// in src/ops/backbone.cpp calls them by kind based on contract layer_kinds[i].

extern "C" cl_mem op_unknown_full_attention_block(OpenCLContext& cl_ctx,
                                                  Weights& weights,
                                                  cl_command_queue queue,
                                                  cl_mem hidden_in,
                                                  int seq_len,
                                                  int hidden_size,
                                                  int layer_idx,
                                                  int start_pos);

// __NNOPT_DISPATCH_APPEND_SENTINEL__
// === END AGENT-MAINTAINED DECLARATIONS ===
