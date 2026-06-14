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
// Build refuses bare integer literals in dimension contexts in src/ops/*.
#pragma once

#include <limits>  // for std::numeric_limits — used by sentinel-NaN floats

namespace MODEL_CONFIG {

// ── Scalar dimensions ──
constexpr float ROPE_THETA = std::numeric_limits<float>::quiet_NaN();  // SENTINEL (NaN): no rope_theta found in config.json (top-level OR nested). Reading this WILL produce NaN — gate on USES_ROPE before use, OR fix scaffoldTs::generateModelConfigHeader to extract from your config's actual nested location.
constexpr bool  USES_ROPE  = false;  // no rope_theta in config.json
constexpr bool  USES_GQA   = false;  // derived: num_kv_heads != num_attention_heads

}  // namespace MODEL_CONFIG
