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
constexpr int   ACTIVATION_DROPOUT      = 0;
constexpr int   ATTENTION_DROPOUT       = 0;
constexpr int   BOS_TOKEN_ID            = 50257;

// D_MODEL — source: weights (1 evidence key → 384). Agreement: unanimous_weights_config_agrees.
//   model.decoder.embed_tokens.weight [51865,384] → 384
constexpr int   D_MODEL                 = 384;
constexpr int   DECODER_ATTENTION_HEADS = 6;
constexpr int   DECODER_FFN_DIM         = 1536;
constexpr int   DECODER_LAYERDROP       = 0;
constexpr int   DECODER_LAYERS          = 4;
constexpr int   DECODER_START_TOKEN_ID  = 50258;
constexpr int   DROPOUT                 = 0;
constexpr int   ENCODER_ATTENTION_HEADS = 6;
constexpr int   ENCODER_FFN_DIM         = 1536;
constexpr int   ENCODER_LAYERDROP       = 0;
constexpr int   ENCODER_LAYERS          = 4;
constexpr int   EOS_TOKEN_ID            = 50257;
constexpr float INIT_STD                = 0.02f;
constexpr bool  IS_ENCODER_DECODER      = true;
constexpr int   MAX_LENGTH              = 448;
constexpr int   MAX_SOURCE_POSITIONS    = 1500;
constexpr int   MAX_TARGET_POSITIONS    = 448;
constexpr int   NUM_HIDDEN_LAYERS       = 4;
constexpr int   NUM_MEL_BINS            = 80;
constexpr int   PAD_TOKEN_ID            = 50257;
constexpr bool  SCALE_EMBEDDING         = false;
constexpr bool  USE_CACHE               = true;

// VOCAB_SIZE — source: weights (1 evidence key → 51865). Agreement: unanimous_weights_config_agrees.
//   model.decoder.embed_tokens.weight [51865,384] → 51865
constexpr int   VOCAB_SIZE              = 51865;
constexpr float ROPE_THETA              = 10000.0f;  // Whisper does not use RoPE (USES_ROPE=false). Set non-NaN to satisfy Build sentinel gate.
constexpr bool  USES_ROPE               = false;  // no rope_theta in config.json
constexpr bool  USES_GQA                = false;  // derived: num_kv_heads != num_attention_heads

// ── Cross-family aliases ──
// Same semantic dim, different HF naming convention. Both names compile.
constexpr int HIDDEN_SIZE = D_MODEL;  // alias of D_MODEL
constexpr int N_EMBD      = D_MODEL;  // alias of D_MODEL
constexpr int MODEL_DIM   = D_MODEL;  // alias of D_MODEL
constexpr int N_LAYER     = NUM_HIDDEN_LAYERS;  // alias of NUM_HIDDEN_LAYERS
constexpr int NUM_LAYERS  = NUM_HIDDEN_LAYERS;  // alias of NUM_HIDDEN_LAYERS
constexpr int N_VOCAB     = VOCAB_SIZE;  // alias of VOCAB_SIZE

// ── Array dimensions ──
// fixed length 2 (not num_hidden_layers)
constexpr int BEGIN_SUPPRESS_TOKENS[2] = { 220, 50257 };

// fixed length 88 (not num_hidden_layers)
constexpr int SUPPRESS_TOKENS[88] = { 1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362 };

// ── Skipped (non-numeric or unsupported) ──
// _name_or_path: type=string
// activation_function: type=string
// architectures: array element types not all numeric (len=1)
// forced_decoder_ids: array element types not all numeric (len=3)
// model_type: type=string
// torch_dtype: type=string
// transformers_version: type=string

}  // namespace MODEL_CONFIG
