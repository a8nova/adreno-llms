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
constexpr int   HIDDEN_SIZE                = 384;  // flattened from backbone_config.hidden_size
constexpr int   IMAGE_SIZE                 = 518;  // flattened from backbone_config.image_size
constexpr int   NUM_ATTENTION_HEADS        = 6;  // flattened from backbone_config.num_attention_heads
constexpr int   BACKBONE_CONFIG_PATCH_SIZE = 14;  // flattened from backbone_config.patch_size
constexpr bool  RESHAPE_HIDDEN_STATES      = false;  // flattened from backbone_config.reshape_hidden_states
constexpr int   FUSION_HIDDEN_SIZE         = 64;
constexpr int   HEAD_HIDDEN_SIZE           = 32;
constexpr int   HEAD_IN_INDEX              = -1;
constexpr float INITIALIZER_RANGE          = 0.02f;
constexpr int   PATCH_SIZE                 = 14;
constexpr int   REASSEMBLE_HIDDEN_SIZE     = 384;
constexpr bool  USE_PRETRAINED_BACKBONE    = false;
constexpr int   HEAD_DIM                   = 64;  // derived: HIDDEN_SIZE / NUM_ATTENTION_HEADS

// ── DINOv2 / DepthAnything constants (transformers defaults; not in config.json) ──
constexpr int   NUM_HIDDEN_LAYERS          = 12;    // Dinov2Config default
constexpr float LAYER_NORM_EPS             = 1e-6f; // Dinov2Config default
constexpr int   MLP_RATIO                  = 4;     // Dinov2Config default -> intermediate = 384*4 = 1536
constexpr int   INTERMEDIATE_SIZE         = 1536;  // HIDDEN_SIZE * MLP_RATIO
constexpr float LAYERSCALE_VALUE          = 1.0f;  // Dinov2Config default (weights override; buffer used)
constexpr int   NUM_CHANNELS              = 3;
constexpr float MAX_DEPTH                 = 1.0f;   // relative depth estimation
constexpr bool  APPLY_LAYERNORM           = true;  // Dinov2 backbone applies final LN to each out feature
// out_indices [3,6,9,12] index into hidden_states where hidden_states[0]=embeddings,
// hidden_states[k]=output of encoder layer k-1. So feature maps come after layers 3,6,9,12.
constexpr int   FEATURE_LAYER_INDICES[4]  = { 3, 6, 9, 12 };
constexpr float ROPE_THETA                 = std::numeric_limits<float>::quiet_NaN();  // SENTINEL (NaN): no rope_theta found in config.json (top-level OR nested). Reading this WILL produce NaN — gate on USES_ROPE before use, OR fix scaffoldTs::generateModelConfigHeader to extract from your config's actual nested location.
constexpr bool  USES_ROPE                  = false;  // no rope_theta in config.json
constexpr bool  USES_GQA                   = false;  // derived: num_kv_heads != num_attention_heads

// ── Cross-family aliases ──
// Same semantic dim, different HF naming convention. Both names compile.
constexpr int N_EMBD              = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int D_MODEL             = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int MODEL_DIM           = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int N_HEAD              = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS
constexpr int NUM_HEADS           = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS
constexpr int NUM_KEY_VALUE_HEADS = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS (no GQA — kv heads == query heads)
constexpr int NUM_KV_HEADS        = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS (no GQA — kv heads == query heads)
constexpr int N_KV_HEAD           = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS (no GQA — kv heads == query heads)

// ── Array dimensions ──
// fixed length 4 (flattened from backbone_config.out_indices)
constexpr int OUT_INDICES[4] = { 3, 6, 9, 12 };

// fixed length 4 (not num_hidden_layers)
constexpr int NECK_HIDDEN_SIZES[4] = { 48, 96, 192, 384 };

// fixed length 4 (not num_hidden_layers)
constexpr float REASSEMBLE_FACTORS[4] = { 4.0f, 2.0f, 1.0f, 0.5f };

// ── Skipped (non-numeric or unsupported) ──
// _commit_hash: type=null
// architectures: array element types not all numeric (len=1)
// backbone: type=null
// backbone_config.architectures: nested array non-numeric (len=1)
// backbone_config.model_type: nested type=string
// backbone_config.out_features: nested array non-numeric (len=4)
// backbone_config.torch_dtype: nested type=string
// model_type: type=string
// torch_dtype: type=string
// transformers_version: type=null

}  // namespace MODEL_CONFIG
