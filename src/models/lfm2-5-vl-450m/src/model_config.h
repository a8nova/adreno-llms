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
constexpr bool  DO_IMAGE_SPLITTING                = true;
constexpr int   DOWNSAMPLE_FACTOR                 = 2;
constexpr int   ENCODER_PATCH_SIZE                = 16;
constexpr int   IMAGE_TOKEN_ID                    = 396;
constexpr int   MAX_IMAGE_TOKENS                  = 256;
constexpr int   MAX_PIXELS_TOLERANCE              = 2;
constexpr int   MAX_TILES                         = 10;
constexpr int   MIN_IMAGE_TOKENS                  = 64;
constexpr int   MIN_TILES                         = 2;
constexpr bool  PROJECTOR_BIAS                    = true;
constexpr int   PROJECTOR_HIDDEN_SIZE             = 2048;
constexpr bool  PROJECTOR_USE_LAYERNORM           = false;
constexpr bool  BLOCK_AUTO_ADJUST_FF_DIM          = true;  // flattened from text_config.block_auto_adjust_ff_dim
constexpr int   BLOCK_DIM                         = 1024;  // flattened from text_config.block_dim
constexpr int   BLOCK_FF_DIM                      = 6656;  // flattened from text_config.block_ff_dim
constexpr int   BLOCK_FFN_DIM_MULTIPLIER          = 1;  // flattened from text_config.block_ffn_dim_multiplier
constexpr int   BLOCK_MLP_INIT_SCALE              = 1;  // flattened from text_config.block_mlp_init_scale
constexpr int   BLOCK_MULTIPLE_OF                 = 256;  // flattened from text_config.block_multiple_of
constexpr float BLOCK_NORM_EPS                    = 1.000000e-5f;  // flattened from text_config.block_norm_eps
constexpr int   BLOCK_OUT_INIT_SCALE              = 1;  // flattened from text_config.block_out_init_scale
constexpr bool  BLOCK_USE_SWIGLU                  = true;  // flattened from text_config.block_use_swiglu
constexpr bool  BLOCK_USE_XAVIER_INIT             = true;  // flattened from text_config.block_use_xavier_init
constexpr int   CONV_L_CACHE                      = 3;  // flattened from text_config.conv_L_cache
constexpr bool  CONV_BIAS                         = false;  // flattened from text_config.conv_bias
constexpr int   CONV_DIM                          = 1024;  // flattened from text_config.conv_dim
constexpr int   CONV_DIM_OUT                      = 1024;  // flattened from text_config.conv_dim_out
constexpr bool  CONV_USE_XAVIER_INIT              = true;  // flattened from text_config.conv_use_xavier_init
constexpr int   EOS_TOKEN_ID                      = 7;  // flattened from text_config.eos_token_id

// HIDDEN_SIZE — source: weights (17 evidence keys → 1024). Agreement: unanimous_weights_config_disagrees.
//   model.language_model.embed_tokens.weight [65536,1024] → 1024
//   model.language_model.layers.0.feed_forward.w1.weight [4608,1024] → 1024
//   model.language_model.layers.1.feed_forward.w1.weight [4608,1024] → 1024
//   …(+14 more evidence keys)
// flattened from text_config.hidden_size
constexpr int   HIDDEN_SIZE                       = 1024;
constexpr float INITIALIZER_RANGE                 = 0.02f;  // flattened from text_config.initializer_range

// INTERMEDIATE_SIZE — source: weights (72 evidence keys → 4608). Agreement: unanimous_weights_config_disagrees.
//   model.language_model.layers.0.feed_forward.w1.weight [4608,1024] → 4608
//   model.language_model.layers.1.feed_forward.w1.weight [4608,1024] → 4608
//   model.language_model.layers.10.feed_forward.w1.weight [4608,1024] → 4608
//   …(+69 more evidence keys)
// flattened from text_config.intermediate_size
constexpr int   INTERMEDIATE_SIZE                 = 4608;
constexpr int   MAX_POSITION_EMBEDDINGS           = 128000;  // flattened from text_config.max_position_embeddings
constexpr float NORM_EPS                          = 1.000000e-5f;  // flattened from text_config.norm_eps
constexpr int   NUM_ATTENTION_HEADS               = 16;  // flattened from text_config.num_attention_heads
constexpr int   NUM_HEADS                         = 16;  // flattened from text_config.num_heads
constexpr int   NUM_HIDDEN_LAYERS                 = 16;  // flattened from text_config.num_hidden_layers
constexpr int   NUM_KEY_VALUE_HEADS               = 8;  // flattened from text_config.num_key_value_heads
constexpr bool  USE_CACHE                         = true;  // flattened from text_config.use_cache
constexpr bool  USE_POS_ENC                       = true;  // flattened from text_config.use_pos_enc

// VOCAB_SIZE — source: weights (1 evidence key → 65536). Agreement: unanimous_weights_config_disagrees.
//   model.language_model.embed_tokens.weight [65536,1024] → 65536
// flattened from text_config.vocab_size
constexpr int   VOCAB_SIZE                        = 65536;
constexpr int   TILE_SIZE                         = 512;
constexpr bool  USE_IMAGE_SPECIAL_TOKENS          = true;
constexpr bool  USE_THUMBNAIL                     = true;
constexpr int   ATTENTION_DROPOUT                 = 0;  // flattened from vision_config.attention_dropout
constexpr int   VISION_CONFIG_HIDDEN_SIZE         = 768;  // flattened from vision_config.hidden_size
constexpr int   VISION_CONFIG_INTERMEDIATE_SIZE   = 3072;  // flattened from vision_config.intermediate_size
constexpr float LAYER_NORM_EPS                    = 1.000000e-6f;  // flattened from vision_config.layer_norm_eps
constexpr int   VISION_CONFIG_NUM_ATTENTION_HEADS = 12;  // flattened from vision_config.num_attention_heads
constexpr int   NUM_CHANNELS                      = 3;  // flattened from vision_config.num_channels
constexpr int   VISION_CONFIG_NUM_HIDDEN_LAYERS   = 12;  // flattened from vision_config.num_hidden_layers
constexpr int   NUM_PATCHES                       = 256;  // flattened from vision_config.num_patches
constexpr int   PATCH_SIZE                        = 16;  // flattened from vision_config.patch_size
constexpr bool  VISION_USE_HEAD                   = false;  // flattened from vision_config.vision_use_head
constexpr int   HEAD_DIM                          = 64;  // derived: HIDDEN_SIZE / NUM_ATTENTION_HEADS
constexpr float ROPE_THETA                        = 1000000.0f;  // flattened from text_config.rope_parameters.rope_theta
constexpr bool  USES_ROPE                         = true;  // text_config.use_pos_enc + rope_parameters present
constexpr bool  USES_GQA                          = true;  // derived: num_kv_heads != num_attention_heads

// ── Cross-family aliases ──
// Same semantic dim, different HF naming convention. Both names compile.
constexpr int   N_EMBD              = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int   D_MODEL             = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int   MODEL_DIM           = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int   N_HEAD              = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS
constexpr int   N_LAYER             = NUM_HIDDEN_LAYERS;  // alias of NUM_HIDDEN_LAYERS
constexpr int   NUM_LAYERS          = NUM_HIDDEN_LAYERS;  // alias of NUM_HIDDEN_LAYERS
constexpr int   N_CTX               = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int   N_POSITIONS         = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int   MAX_SEQ_LEN         = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int   MAX_SEQUENCE_LENGTH = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int   N_VOCAB             = VOCAB_SIZE;  // alias of VOCAB_SIZE
constexpr int   N_INNER             = INTERMEDIATE_SIZE;  // alias of INTERMEDIATE_SIZE
constexpr int   FFN_DIM             = INTERMEDIATE_SIZE;  // alias of INTERMEDIATE_SIZE
constexpr float LAYER_NORM_EPSILON  = LAYER_NORM_EPS;  // alias of LAYER_NORM_EPS
constexpr float RMS_NORM_EPS        = LAYER_NORM_EPS;  // alias of LAYER_NORM_EPS
constexpr int   NUM_KV_HEADS        = NUM_KEY_VALUE_HEADS;  // alias of NUM_KEY_VALUE_HEADS
constexpr int   N_KV_HEAD           = NUM_KEY_VALUE_HEADS;  // alias of NUM_KEY_VALUE_HEADS

// ── Skipped (non-numeric or unsupported) ──
// architectures: array element types not all numeric (len=1)
// dtype: type=string
// model_type: type=string
// projector_hidden_act: type=string
// text_config._name_or_path: nested type=string
// text_config.architectures: nested array non-numeric (len=1)
// text_config.dtype: nested type=string
// text_config.layer_types: nested array non-numeric (len=16)
// text_config.model_type: nested type=string
// text_config.rope_parameters: nested type=object
// transformers_version: type=string
// vision_config.dtype: nested type=string
// vision_config.hidden_act: nested type=string
// vision_config.model_type: nested type=string

}  // namespace MODEL_CONFIG
