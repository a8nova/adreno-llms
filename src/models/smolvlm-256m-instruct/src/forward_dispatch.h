// Auto-generated graph dispatch header.
// Each ported node provides a function declaration here. The agent appends
// to this file via PortNode/MarkNodePorted as nodes are ported.
//
// Model: HuggingFaceTB/SmolVLM-256M-Instruct (Idefics3ForConditionalGeneration)
// Total nodes: 550
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
extern "C" cl_mem op_Embedding(OpenCLContext& cl_ctx,
                               Weights& weights,
                               cl_command_queue queue,
                               cl_mem input_ids_i32,
                               int num_tokens,
                               int hidden_size,
                               const char* weight_key_wte);

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

extern "C" cl_mem op_LlamaRMSNorm(OpenCLContext& cl_ctx,
                                 Weights& weights,
                                 cl_command_queue queue,
                                 cl_mem input,
                                 int rows,
                                 int cols,
                                 float eps,
                                 const char* weight_key_w);

// Fused: x_inout += residual; returns rmsnorm(x_inout). After the call,
// x_inout holds the post-add value (the running residual stream).
extern "C" cl_mem op_LlamaRMSNormWithResidual(OpenCLContext& cl_ctx,
                                              Weights& weights,
                                              cl_command_queue queue,
                                              cl_mem x_inout,
                                              cl_mem residual,
                                              int rows,
                                              int cols,
                                              float eps,
                                              const char* weight_key_w);

extern "C" cl_mem op_LayerNorm(OpenCLContext& cl_ctx,
                               Weights& weights,
                               cl_command_queue queue,
                               cl_mem input,
                               int rows,
                               int cols,
                               float eps,
                               const char* weight_key_weight,
                               const char* weight_key_bias);

extern "C" cl_mem op_Idefics3VisionEmbeddings(OpenCLContext& cl_ctx,
                                              Weights& weights,
                                              cl_command_queue queue,
                                              cl_mem patch_embedding_out,   // [B, C, H, W]
                                              cl_mem position_embed_out,    // [B, H*W, C]
                                              int batch,
                                              int channels,
                                              int height,
                                              int width);

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

extern "C" cl_mem op_Idefics3VisionAttention(OpenCLContext& cl_ctx,
                                             Weights& weights,
                                             cl_command_queue queue,
                                             cl_mem hidden_states,
                                             cl_mem attention_mask_opt,
                                             int B,
                                             int T,
                                             int C,
                                             int num_heads,
                                             const char* q_w,
                                             const char* q_b,
                                             const char* k_w,
                                             const char* k_b,
                                             const char* v_w,
                                             const char* v_b,
                                             const char* o_w,
                                             const char* o_b);

extern "C" cl_mem op_Idefics3VisionMLP(OpenCLContext& cl_ctx,
                                       Weights& weights,
                                       cl_command_queue queue,
                                       cl_mem hidden_states,
                                       int rows,
                                       int C,
                                       int intermediate_size,
                                       const char* fc1_w,
                                       const char* fc1_b,
                                       const char* fc2_w,
                                       const char* fc2_b);

extern "C" cl_mem op_Idefics3EncoderLayer(OpenCLContext& cl_ctx,
                                          Weights& weights,
                                          cl_command_queue queue,
                                          cl_mem hidden_states,
                                          cl_mem attention_mask_opt,
                                          int B,
                                          int T,
                                          int C,
                                          int num_heads,
                                          float ln_eps,
                                          const char* ln1_w,
                                          const char* ln1_b,
                                          const char* q_w,
                                          const char* q_b,
                                          const char* k_w,
                                          const char* k_b,
                                          const char* v_w,
                                          const char* v_b,
                                          const char* o_w,
                                          const char* o_b,
                                          const char* ln2_w,
                                          const char* ln2_b,
                                          int intermediate_size,
                                          const char* fc1_w,
                                          const char* fc1_b,
                                          const char* fc2_w,
                                          const char* fc2_b);

extern "C" cl_mem op_Linear(OpenCLContext& cl_ctx,
                            Weights& weights,
                            cl_command_queue queue,
                            cl_mem input,
                            int rows,
                            int in_features,
                            int out_features,
                            const char* weight_key_w,
                            const char* weight_key_b_optional);

extern "C" cl_mem op_PytorchGELUTanh(OpenCLContext& cl_ctx,
                                     Weights& weights,
                                     cl_command_queue queue,
                                     cl_mem input,
                                     int n_elements);

// R6.4: optional `fused_in_norm_w` lets the attention op skip the separate
// rmsnorm dispatch upstream and fold the rmsnorm into the QKV image GEMV. When
// non-null AND rows==1 AND fp16 build, `hidden_states` is treated as the RAW
// (pre-norm) input; the kernel computes inv_rms inside each WG and applies
// `* inv_rms * gamma` before the matmul. nullptr → caller already normalized.
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
                                        const char* o_w,
                                        const char* fused_in_norm_w,
                                        float rms_eps);

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

// Same as op_LlamaMLP but fuses the post-MLP residual add into the down_proj
// GEMV write on the M=1 fp16 decode path. Returns a fresh buffer holding
// (down_proj(swiglu(gate, up)) + residual). For prefill (rows>1) or non-fp16
// builds it falls back to op_LlamaMLP + caller-side add_residual semantics by
// computing MLP into a fresh buffer and adding the residual elementwise.
extern "C" cl_mem op_LlamaMLP_with_residual(OpenCLContext& cl_ctx,
                                            Weights& weights,
                                            cl_command_queue queue,
                                            cl_mem input,
                                            cl_mem residual,
                                            int rows,
                                            int hidden_size,
                                            int intermediate_size,
                                            const char* gate_w,
                                            const char* up_w,
                                            const char* down_w);

// R6.5: MLP entry point that folds rmsnorm_post (residual_add + norm) into the
// gate_up swiglu GEMV. Caller passes (attn_out_raw, hidden_states_residual,
// post_norm_w, rms_eps); the op internally allocates a sum_buf, runs the fused
// rmsnorm+residual+swiglu, then the existing fused down+residual using sum_buf
// as residual. Decode-only (rows==1, fp16) — prefill/fp32 falls back to the
// caller's pre-norm + op_LlamaMLP_with_residual path.
extern "C" cl_mem op_LlamaMLP_with_residual_and_rmsnorm(OpenCLContext& cl_ctx,
                                                        Weights& weights,
                                                        cl_command_queue queue,
                                                        cl_mem attn_out_raw,
                                                        cl_mem hidden_states_residual,
                                                        int rows,
                                                        int hidden_size,
                                                        int intermediate_size,
                                                        const char* post_norm_w,
                                                        float rms_eps,
                                                        const char* gate_w,
                                                        const char* up_w,
                                                        const char* down_w);

extern "C" cl_mem op_Idefics3VisionTransformer(OpenCLContext& cl_ctx,
                                               Weights& weights,
                                               cl_command_queue queue,
                                               cl_mem pixel_values_nchw,
                                               cl_mem patch_attention_mask_opt,
                                               int B,
                                               int C,
                                               int H,
                                               int W,
                                               int patch_size,
                                               const char* patch_emb_w,
                                               const char* patch_emb_b,
                                               const char* pos_emb_w);

extern "C" cl_mem op_Idefics3Connector(OpenCLContext& cl_ctx,
                                       Weights& weights,
                                       cl_command_queue queue,
                                       cl_mem vision_hidden_states,
                                       int rows,
                                       int in_features,
                                       int out_features,
                                       const char* proj_w);

extern "C" cl_mem op_PixelShuffle(OpenCLContext& cl_ctx,
                                  cl_command_queue queue,
                                  cl_mem in_buf,
                                  int IN_H,
                                  int IN_W,
                                  int IN_C,
                                  int SCALE);

// __NNOPT_DISPATCH_APPEND_SENTINEL__
// === END AGENT-MAINTAINED DECLARATIONS ===
