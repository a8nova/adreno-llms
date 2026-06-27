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
constexpr int   _SLIDING_WINDOW_PATTERN     = 6;
constexpr bool  ATTENTION_BIAS              = false;
constexpr int   ATTENTION_DROPOUT           = 0;
constexpr int   BOS_TOKEN_ID                = 2;
constexpr int   HEAD_DIM                    = 256;
constexpr int   HIDDEN_SIZE                 = 640;
constexpr float INITIALIZER_RANGE           = 0.02f;
constexpr int   INTERMEDIATE_SIZE           = 2048;
constexpr int   MAX_POSITION_EMBEDDINGS     = 32768;
constexpr int   NUM_ATTENTION_HEADS         = 4;
constexpr int   NUM_HIDDEN_LAYERS           = 18;
constexpr int   NUM_KEY_VALUE_HEADS         = 1;
constexpr int   PAD_TOKEN_ID                = 0;
constexpr int   QUERY_PRE_ATTN_SCALAR       = 256;
constexpr float RMS_NORM_EPS                = 1.000000e-6f;
constexpr int   ROPE_LOCAL_BASE_FREQ        = 10000;
constexpr int   ROPE_THETA                  = 1000000;
constexpr int   SLIDING_WINDOW              = 512;
constexpr bool  USE_BIDIRECTIONAL_ATTENTION = false;
constexpr bool  USE_CACHE                   = true;
constexpr int   VOCAB_SIZE                  = 262144;
constexpr bool  USES_ROPE                   = true;  // derived from rope_theta in config.json
constexpr bool  USES_GQA                    = true;  // derived: num_kv_heads != num_attention_heads

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

// ── Array dimensions ──
// fixed length 2 (not num_hidden_layers)
constexpr int EOS_TOKEN_ID[2] = { 1, 50 };

// ── Skipped (non-numeric or unsupported) ──
// architectures: array element types not all numeric (len=1)
// attn_logit_softcapping: type=null
// dtype: type=string
// final_logit_softcapping: type=null
// hidden_activation: type=string
// layer_types: array element types not all numeric (len=18)
// model_type: type=string
// transformers_version: type=string

}  // namespace MODEL_CONFIG
