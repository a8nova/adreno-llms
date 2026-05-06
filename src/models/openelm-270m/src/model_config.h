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
constexpr int   BOS_TOKEN_ID              = 1;
constexpr int   EOS_TOKEN_ID              = 2;
constexpr int   FFN_DIM_DIVISOR           = 256;
constexpr bool  FFN_WITH_GLU              = true;
constexpr int   HEAD_DIM                  = 64;
constexpr float INITIALIZER_RANGE         = 0.02f;
constexpr int   MAX_CONTEXT_LENGTH        = 2048;
constexpr int   MODEL_DIM                 = 1280;
constexpr bool  NORMALIZE_QK_PROJECTIONS  = true;
constexpr int   NUM_GQA_GROUPS            = 4;
constexpr int   NUM_TRANSFORMER_LAYERS    = 16;
constexpr int   ROPE_FREQ_CONSTANT        = 10000;
constexpr int   ROPE_MAX_LENGTH           = 4096;
constexpr bool  SHARE_INPUT_OUTPUT_LAYERS = true;
constexpr bool  USE_CACHE                 = true;
constexpr int   VOCAB_SIZE                = 32000;
constexpr float NORM_EPS                  = 1e-6f;
constexpr float ROPE_THETA                = std::numeric_limits<float>::quiet_NaN();  // SENTINEL (NaN): no rope_theta found in config.json (top-level OR nested). Reading this WILL produce NaN — gate on USES_ROPE before use, OR fix scaffoldTs::generateModelConfigHeader to extract from your config's actual nested location.
constexpr bool  USES_ROPE                 = false;  // no rope_theta in config.json
constexpr bool  USES_GQA                  = false;  // derived: num_kv_heads != num_attention_heads

// ── Cross-family aliases ──
// Same semantic dim, different HF naming convention. Both names compile.
constexpr int HIDDEN_SIZE = MODEL_DIM;  // alias of MODEL_DIM
constexpr int N_EMBD      = MODEL_DIM;  // alias of MODEL_DIM
constexpr int D_MODEL     = MODEL_DIM;  // alias of MODEL_DIM
constexpr int N_VOCAB     = VOCAB_SIZE;  // alias of VOCAB_SIZE

// ── Array dimensions ──
// per-layer (length == num_hidden_layers)
constexpr float FFN_MULTIPLIERS[16] = { 0.5f, 0.73f, 0.97f, 1.2f, 1.43f, 1.67f, 1.9f, 2.13f, 2.37f, 2.6f, 2.83f, 3.07f, 3.3f, 3.53f, 3.77f, 4.0f };

// per-layer (length == num_hidden_layers)
constexpr int NUM_KV_HEADS[16] = { 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5 };

// per-layer (length == num_hidden_layers)
constexpr int NUM_QUERY_HEADS[16] = { 12, 12, 12, 12, 12, 16, 16, 16, 16, 16, 16, 16, 20, 20, 20, 20 };

// per-layer FFN intermediate size: derived from FFN_MULTIPLIERS via OpenELM's
// _make_divisible(int(mult*MODEL_DIM), FFN_DIM_DIVISOR) (round-to-nearest).
// Verified against weights/model.fp16.meta.json proj_2 shape[1].
constexpr int FFN_INTERMEDIATE_SIZE[16] = { 768, 1024, 1280, 1536, 1792, 2048, 2560, 2816, 3072, 3328, 3584, 3840, 4352, 4608, 4864, 5120 };

// fixed length 2 (not num_hidden_layers)
constexpr float QKV_MULTIPLIERS[2] = { 0.5f, 1.0f };

// ── Skipped (non-numeric or unsupported) ──
// activation_fn_name: type=string
// architectures: array element types not all numeric (len=1)
// auto_map.AutoConfig: nested type=string
// auto_map.AutoModelForCausalLM: nested type=string
// auto_map: object with no numeric/boolean leaves
// model_type: type=string
// normalization_layer_name: type=string
// torch_dtype: type=string
// transformers_version: type=string

}  // namespace MODEL_CONFIG
