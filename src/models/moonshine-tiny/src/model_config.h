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
constexpr bool  ATTENTION_BIAS              = false;
constexpr int   ATTENTION_DROPOUT           = 0;
constexpr int   BOS_TOKEN_ID                = 1;
constexpr int   PAD_TOKEN_ID                = 2;
constexpr int   DECODER_NUM_ATTENTION_HEADS = 8;
constexpr int   DECODER_NUM_HIDDEN_LAYERS   = 6;
constexpr int   DECODER_NUM_KEY_VALUE_HEADS = 8;
constexpr int   DECODER_START_TOKEN_ID      = 1;
constexpr int   ENCODER_NUM_ATTENTION_HEADS = 8;
constexpr int   ENCODER_NUM_HIDDEN_LAYERS   = 6;
constexpr int   ENCODER_NUM_KEY_VALUE_HEADS = 8;
constexpr int   EOS_TOKEN_ID                = 2;

// HIDDEN_SIZE — source: weights (13 evidence keys → 288). Agreement: unanimous_weights_config_agrees.
//   model.decoder.embed_tokens.weight [32768,288] → 288
//   model.decoder.layers.0.self_attn.o_proj.weight [288,288] → 288
//   model.decoder.layers.1.self_attn.o_proj.weight [288,288] → 288
//   …(+10 more evidence keys)
constexpr int   HIDDEN_SIZE                 = 288;
constexpr float INITIALIZER_RANGE           = 0.02f;

// INTERMEDIATE_SIZE — source: weights (24 evidence keys → 1152). Agreement: unanimous_weights_config_agrees.
//   model.decoder.layers.0.mlp.fc1.weight [2304,288] → 2304
//   model.decoder.layers.1.mlp.fc1.weight [2304,288] → 2304
//   model.decoder.layers.2.mlp.fc1.weight [2304,288] → 2304
//   …(+21 more evidence keys)
constexpr int   INTERMEDIATE_SIZE           = 1152;
constexpr bool  IS_ENCODER_DECODER          = true;
constexpr int   MAX_POSITION_EMBEDDINGS     = 194;
constexpr float PARTIAL_ROTARY_FACTOR       = 0.9f;
constexpr int   ROPE_THETA                  = 10000;
constexpr bool  USE_CACHE                   = true;

// VOCAB_SIZE — source: weights (1 evidence key → 32768). Agreement: unanimous_weights_config_agrees.
//   model.decoder.embed_tokens.weight [32768,288] → 32768
constexpr int   VOCAB_SIZE                  = 32768;
constexpr int   PAD_HEAD_DIM_TO_MULTIPLE_OF = 8;
constexpr bool  USES_ROPE                   = true;  // derived from rope_theta in config.json
constexpr bool  USES_GQA                    = false;  // derived: num_kv_heads != num_attention_heads

// ── Cross-family aliases ──
// Same semantic dim, different HF naming convention. Both names compile.
constexpr int N_EMBD              = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int D_MODEL             = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int MODEL_DIM           = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int N_CTX               = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int N_POSITIONS         = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int MAX_SEQ_LEN         = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int MAX_SEQUENCE_LENGTH = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int N_VOCAB             = VOCAB_SIZE;  // alias of VOCAB_SIZE
constexpr int N_INNER             = INTERMEDIATE_SIZE;  // alias of INTERMEDIATE_SIZE
constexpr int FFN_DIM             = INTERMEDIATE_SIZE;  // alias of INTERMEDIATE_SIZE
constexpr int ROPE_BASE           = ROPE_THETA;  // alias of ROPE_THETA
constexpr int ROPE_FREQ_CONSTANT  = ROPE_THETA;  // alias of ROPE_THETA
constexpr int ROPE_FREQ_BASE      = ROPE_THETA;  // alias of ROPE_THETA

// ── Skipped (non-numeric or unsupported) ──
// architectures: array element types not all numeric (len=1)
// decoder_hidden_act: type=string
// encoder_hidden_act: type=string
// model_type: type=string
// rope_scaling: type=null
// torch_dtype: type=string
// transformers_version: type=string

}  // namespace MODEL_CONFIG
