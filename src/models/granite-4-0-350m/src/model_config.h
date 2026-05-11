// Auto-generated from model_info/config.json at scaffold time.
//
// You CAN edit this file directly when a value is wrong — it is NOT authority.
// Preferred debugging path when a constant looks suspicious:
//   1) Verify against PyTorch's runtime in reference/config_used.json
//      (captured by GenerateReference). That file shows what the model
//      actually consumed, including nested keys (rope_parameters.rope_theta etc).
//   2) If wrong, EITHER edit this file (workspace-local fix) OR fix
//      scaffoldTs.ts::generateModelConfigHeader (tool-side fix that re-emits
//      correctly on next Scaffold for this AND every future port).
//   3) Re-run Build and proceed.
//
// Every numeric dimension in layer code MUST come from here. Use:
//   MODEL_CONFIG::HIDDEN_SIZE                  — scalar by name
//   MODEL_CONFIG::NUM_QUERY_HEADS[layer_idx_]  — per-layer array by index
// Build refuses bare integer literals in dimension contexts in src/layers/*.
#pragma once

#include <limits>  // for std::numeric_limits — used by sentinel-NaN floats

namespace MODEL_CONFIG {

// ── Scalar dimensions ──
constexpr bool  ATTENTION_BIAS           = false;
constexpr int   ATTENTION_DROPOUT        = 0;
constexpr float ATTENTION_MULTIPLIER     = 0.015625f;
constexpr int   BOS_TOKEN_ID             = 100257;
constexpr int   EMBEDDING_MULTIPLIER     = 12;
constexpr int   EOS_TOKEN_ID             = 100257;
constexpr int   HIDDEN_SIZE              = 1024;
constexpr float INITIALIZER_RANGE        = 0.1f;
constexpr int   INTERMEDIATE_SIZE        = 2048;
constexpr int   LOGITS_SCALING           = 4;
constexpr int   MAMBA_CHUNK_SIZE         = 256;
constexpr bool  MAMBA_CONV_BIAS          = true;
constexpr int   MAMBA_D_CONV             = 4;
constexpr int   MAMBA_D_HEAD             = 16;
constexpr int   MAMBA_D_STATE            = 256;
constexpr int   MAMBA_EXPAND             = 2;
constexpr int   MAMBA_N_GROUPS           = 1;
constexpr int   MAMBA_N_HEADS            = 128;
constexpr bool  MAMBA_PROJ_BIAS          = false;
constexpr int   MAX_POSITION_EMBEDDINGS  = 32768;
constexpr int   NUM_ATTENTION_HEADS      = 16;
constexpr int   NUM_EXPERTS_PER_TOK      = 0;
constexpr int   NUM_HIDDEN_LAYERS        = 28;
constexpr int   NUM_KEY_VALUE_HEADS      = 4;
constexpr int   NUM_LOCAL_EXPERTS        = 0;
constexpr bool  OUTPUT_ROUTER_LOGITS     = false;
constexpr int   PAD_TOKEN_ID             = 100256;
constexpr float RESIDUAL_MULTIPLIER      = 0.263f;
constexpr float RMS_NORM_EPS             = 1.000000e-5f;
constexpr int   ROPE_THETA               = 10000000;
constexpr float ROUTER_AUX_LOSS_COEF     = 0.01f;
constexpr int   SHARED_INTERMEDIATE_SIZE = 2048;
constexpr bool  TIE_WORD_EMBEDDINGS      = true;
constexpr bool  USE_CACHE                = true;
constexpr int   VOCAB_SIZE               = 100352;
constexpr int   HEAD_DIM                 = 64;  // derived: HIDDEN_SIZE / NUM_ATTENTION_HEADS
constexpr bool  USES_ROPE                = true;  // derived from rope_theta in config.json
constexpr bool  USES_GQA                 = true;  // derived: num_kv_heads != num_attention_heads

// ── Cross-family aliases ──
// Same semantic dim, different HF naming convention. Both names compile.
constexpr int   N_EMBD              = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int   D_MODEL             = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int   MODEL_DIM           = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int   N_HEAD              = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS
constexpr int   NUM_HEADS           = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS
constexpr int   N_LAYER             = NUM_HIDDEN_LAYERS;  // alias of NUM_HIDDEN_LAYERS
constexpr int   NUM_LAYERS          = NUM_HIDDEN_LAYERS;  // alias of NUM_HIDDEN_LAYERS
constexpr int   N_CTX               = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int   N_POSITIONS         = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int   MAX_SEQ_LEN         = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int   MAX_SEQUENCE_LENGTH = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int   N_VOCAB             = VOCAB_SIZE;  // alias of VOCAB_SIZE
constexpr int   N_INNER             = INTERMEDIATE_SIZE;  // alias of INTERMEDIATE_SIZE
constexpr int   FFN_DIM             = INTERMEDIATE_SIZE;  // alias of INTERMEDIATE_SIZE
constexpr float LAYER_NORM_EPSILON  = RMS_NORM_EPS;  // alias of RMS_NORM_EPS
constexpr float LAYER_NORM_EPS      = RMS_NORM_EPS;  // alias of RMS_NORM_EPS
constexpr float NORM_EPS            = RMS_NORM_EPS;  // alias of RMS_NORM_EPS
constexpr int   NUM_KV_HEADS        = NUM_KEY_VALUE_HEADS;  // alias of NUM_KEY_VALUE_HEADS
constexpr int   N_KV_HEAD           = NUM_KEY_VALUE_HEADS;  // alias of NUM_KEY_VALUE_HEADS
constexpr int   ROPE_BASE           = ROPE_THETA;  // alias of ROPE_THETA
constexpr int   ROPE_FREQ_CONSTANT  = ROPE_THETA;  // alias of ROPE_THETA
constexpr int   ROPE_FREQ_BASE      = ROPE_THETA;  // alias of ROPE_THETA

// ── Skipped (non-numeric or unsupported) ──
// architectures: array element types not all numeric (len=1)
// hidden_act: type=string
// init_method: type=string
// layer_types: array element types not all numeric (len=28)
// model_type: type=string
// normalization_function: type=string
// position_embedding_type: type=string
// rope_scaling: type=null
// torch_dtype: type=string
// transformers_version: type=string

}  // namespace MODEL_CONFIG
