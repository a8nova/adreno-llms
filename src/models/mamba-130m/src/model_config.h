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
constexpr int   BOS_TOKEN_ID             = 0;
constexpr int   CONV_KERNEL              = 4;
constexpr int   D_INNER                  = 1536;
constexpr int   D_MODEL                  = 768;
constexpr int   EOS_TOKEN_ID             = 0;
constexpr int   EXPAND                   = 2;
constexpr bool  FUSED_ADD_NORM           = true;
constexpr int   HIDDEN_SIZE              = 768;
constexpr float INITIALIZER_RANGE        = 0.1f;
constexpr int   INTERMEDIATE_SIZE        = 1536;
constexpr float LAYER_NORM_EPSILON       = 1.000000e-5f;
constexpr int   N_LAYER                  = 24;
constexpr int   NUM_HIDDEN_LAYERS        = 24;
constexpr int   PAD_TOKEN_ID             = 0;
constexpr int   PAD_VOCAB_SIZE_MULTIPLE  = 8;
constexpr bool  RESCALE_PRENORM_RESIDUAL = false;
constexpr bool  RESIDUAL_IN_FP32         = true;
constexpr bool  RMS_NORM                 = true;
constexpr int   STATE_SIZE               = 16;
constexpr float TIME_STEP_FLOOR          = 0.0001f;
constexpr float TIME_STEP_MAX            = 0.1f;
constexpr float TIME_STEP_MIN            = 0.001f;
constexpr int   TIME_STEP_RANK           = 48;
constexpr int   TIME_STEP_SCALE          = 1;
constexpr bool  USE_BIAS                 = false;
constexpr bool  USE_CACHE                = true;
constexpr bool  USE_CONV_BIAS            = true;
constexpr int   VOCAB_SIZE               = 50280;
constexpr float ROPE_THETA               = 0.0f;  // sentinel: model has no RoPE config (absolute pos embed). Gate runtime use on USES_ROPE.
constexpr bool  USES_ROPE                = false;  // no rope_theta in config.json
constexpr bool  USES_GQA                 = false;  // derived: num_kv_heads != num_attention_heads

// ── Cross-family aliases ──
// Same semantic dim, different HF naming convention. Both names compile.
constexpr int   N_EMBD         = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int   MODEL_DIM      = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int   NUM_LAYERS     = NUM_HIDDEN_LAYERS;  // alias of NUM_HIDDEN_LAYERS
constexpr int   N_VOCAB        = VOCAB_SIZE;  // alias of VOCAB_SIZE
constexpr int   N_INNER        = INTERMEDIATE_SIZE;  // alias of INTERMEDIATE_SIZE
constexpr int   FFN_DIM        = INTERMEDIATE_SIZE;  // alias of INTERMEDIATE_SIZE
constexpr float LAYER_NORM_EPS = LAYER_NORM_EPSILON;  // alias of LAYER_NORM_EPSILON
constexpr float RMS_NORM_EPS   = LAYER_NORM_EPSILON;  // alias of LAYER_NORM_EPSILON
constexpr float NORM_EPS       = LAYER_NORM_EPSILON;  // alias of LAYER_NORM_EPSILON

// ── Skipped (non-numeric or unsupported) ──
// architectures: array element types not all numeric (len=1)
// hidden_act: type=string
// model_type: type=string
// ssm_cfg: type=object
// time_step_init_scheme: type=string
// torch_dtype: type=string
// transformers_version: type=string

}  // namespace MODEL_CONFIG
