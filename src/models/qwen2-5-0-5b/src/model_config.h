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
constexpr int   ATTENTION_DROPOUT       = 0;
constexpr int   BOS_TOKEN_ID            = 151643;
constexpr int   EOS_TOKEN_ID            = 151643;
constexpr int   HIDDEN_SIZE             = 896;
constexpr float INITIALIZER_RANGE       = 0.02f;
constexpr int   INTERMEDIATE_SIZE       = 4864;
constexpr int   MAX_POSITION_EMBEDDINGS = 32768;
constexpr int   MAX_WINDOW_LAYERS       = 24;
constexpr int   NUM_ATTENTION_HEADS     = 14;
constexpr int   NUM_HIDDEN_LAYERS       = 24;
constexpr int   NUM_KEY_VALUE_HEADS     = 2;
constexpr float RMS_NORM_EPS            = 1.000000e-6f;
constexpr int   ROPE_THETA              = 1000000;
constexpr int   SLIDING_WINDOW          = 32768;
constexpr bool  TIE_WORD_EMBEDDINGS     = true;
constexpr bool  USE_CACHE               = true;
constexpr bool  USE_MROPE               = false;
constexpr bool  USE_SLIDING_WINDOW      = false;
constexpr int   VOCAB_SIZE              = 151936;
constexpr int   HEAD_DIM                = 64;  // derived: HIDDEN_SIZE / NUM_ATTENTION_HEADS
constexpr bool  USES_ROPE               = true;  // derived from rope_theta in config.json
constexpr bool  USES_GQA                = true;  // derived: num_kv_heads != num_attention_heads

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

// ── Skipped (non-numeric or unsupported) ──
// architectures: array element types not all numeric (len=1)
// hidden_act: type=string
// model_type: type=string
// torch_dtype: type=string
// transformers_version: type=string

}  // namespace MODEL_CONFIG
