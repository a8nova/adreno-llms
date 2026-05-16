// AUTO-GENERATED from model_info/config.json at scaffold time.
// DO NOT EDIT MANUALLY. Regenerate with: rm src/model_config.h && Build
//
// Every numeric dimension in layer code MUST come from here. Use:
//   MODEL_CONFIG::HIDDEN_SIZE                  — scalar by name
//   MODEL_CONFIG::NUM_QUERY_HEADS[layer_idx_]  — per-layer array by index
// Build refuses bare integer literals in dimension contexts in src/layers/*.
#pragma once

namespace MODEL_CONFIG {

// ── Scalar dimensions ──
constexpr bool  BLOCK_AUTO_ADJUST_FF_DIM = true;
constexpr int   BLOCK_DIM                = 1024;
constexpr int   BLOCK_FF_DIM             = 6656;
constexpr int   BLOCK_FFN_DIM_MULTIPLIER = 1;
constexpr int   BLOCK_MLP_INIT_SCALE     = 1;
constexpr int   BLOCK_MULTIPLE_OF        = 256;
constexpr float BLOCK_NORM_EPS           = 1.000000e-5f;
constexpr int   BLOCK_OUT_INIT_SCALE     = 1;
constexpr bool  BLOCK_USE_SWIGLU         = true;
constexpr bool  BLOCK_USE_XAVIER_INIT    = true;
constexpr int   BOS_TOKEN_ID             = 1;
constexpr int   CONV_L_CACHE             = 3;
constexpr bool  CONV_BIAS                = false;
constexpr int   CONV_DIM                 = 1024;
constexpr bool  CONV_USE_XAVIER_INIT     = true;
constexpr int   EOS_TOKEN_ID             = 7;
constexpr int   HIDDEN_SIZE              = 1024;
constexpr float INITIALIZER_RANGE        = 0.02f;
constexpr int   INTERMEDIATE_SIZE        = 6656;
// Architectural max is 128000 (LFM2.5 long-context). At that size KV cache for
// 6 attention layers needs 6 * 2 * 128000 * 8 * 64 * sizeof(fp16) = 1572 MB —
// blows the 1.8 GB GPU budget on Adreno 619 v2 alongside 676 MB weights.
// Capped to 2048: KV cache ~25 MB total, fits Tab A9+. Bump if longer
// contexts are needed and a bigger device is available.
constexpr int   MAX_POSITION_EMBEDDINGS  = 2048;
constexpr float NORM_EPS                 = 1.000000e-5f;
constexpr int   NUM_ATTENTION_HEADS      = 16;
constexpr int   NUM_HEADS                = 16;
constexpr int   NUM_HIDDEN_LAYERS        = 16;
constexpr int   NUM_KEY_VALUE_HEADS      = 8;
constexpr int   PAD_TOKEN_ID             = 0;
constexpr bool  TIE_EMBEDDING            = true;
constexpr bool  USE_CACHE                = true;
constexpr bool  USE_POS_ENC              = true;
constexpr int   VOCAB_SIZE               = 65536;
constexpr int   HEAD_DIM                 = 64;  // derived: HIDDEN_SIZE / NUM_ATTENTION_HEADS
constexpr float ROPE_THETA               = 1000000.0f;  // from config_used.json::rope_theta_used
constexpr bool  USES_ROPE                = true;  // rope_theta present in config
constexpr bool  USES_GQA                 = true;  // derived: num_kv_heads != num_attention_heads

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
constexpr float LAYER_NORM_EPSILON  = NORM_EPS;  // alias of NORM_EPS
constexpr float LAYER_NORM_EPS      = NORM_EPS;  // alias of NORM_EPS
constexpr float RMS_NORM_EPS        = NORM_EPS;  // alias of NORM_EPS
constexpr int   NUM_KV_HEADS        = NUM_KEY_VALUE_HEADS;  // alias of NUM_KEY_VALUE_HEADS
constexpr int   N_KV_HEAD           = NUM_KEY_VALUE_HEADS;  // alias of NUM_KEY_VALUE_HEADS

// ── Skipped (non-numeric or unsupported) ──
// architectures: array element types not all numeric (len=1)
// dtype: type=string
// layer_types: array element types not all numeric (len=16)
// model_type: type=string
// rope_parameters: type=object
// transformers_version: type=string

}  // namespace MODEL_CONFIG
