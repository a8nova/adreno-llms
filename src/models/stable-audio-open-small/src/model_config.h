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
constexpr int   SAMPLE_SIZE      = 524288;
constexpr int   SAMPLE_RATE      = 44100;
constexpr int   AUDIO_CHANNELS   = 2;
constexpr int   IO_CHANNELS      = 64;  // flattened from model.io_channels
constexpr bool  USE_EMA          = true;  // flattened from training.use_ema
constexpr bool  PRE_ENCODED      = false;  // flattened from training.pre_encoded
constexpr bool  LOG_LOSS_INFO    = false;  // flattened from training.log_loss_info
constexpr int   CLIP_GRAD_NORM   = 1;  // flattened from training.clip_grad_norm
constexpr int   CFG_DROPOUT_PROB = 0;  // flattened from training.cfg_dropout_prob
constexpr float ROPE_THETA       = std::numeric_limits<float>::quiet_NaN();  // SENTINEL (NaN): no rope_theta found in config.json (top-level OR nested). Reading this WILL produce NaN — gate on USES_ROPE before use, OR fix scaffoldTs::generateModelConfigHeader to extract from your config's actual nested location.
constexpr bool  USES_ROPE        = false;  // no rope_theta in config.json
constexpr bool  USES_GQA         = false;  // derived: num_kv_heads != num_attention_heads

// ── DiT (DiffusionTransformer) dimensions — hand-added from dit.py + config ──
// embed_dim=1024, depth=16, num_heads=8, dim_heads = embed_dim/num_heads = 128.
constexpr int   DIT_EMBED_DIM        = 1024;
constexpr int   DIT_DEPTH            = 16;
constexpr int   DIT_NUM_HEADS        = 8;
constexpr int   DIT_DIM_HEADS        = 128;   // 1024 / 8
constexpr int   DIT_FF_INNER         = 4096;  // dim * mult(4); GLU proj -> 8192 -> chunk 4096
constexpr int   DIT_LATENT_CHANNELS  = 64;    // io_channels (DiT dim_in)
constexpr int   DIT_COND_TOKEN_DIM   = 768;   // T5 cross-attn cond dim
constexpr int   DIT_GLOBAL_COND_DIM  = 768;   // seconds_total global cond dim
constexpr int   DIT_TIMESTEP_FEAT    = 256;   // FourierFeatures out (128 pairs -> 256)
constexpr int   DIT_ROPE_DIM         = 64;    // max(dim_heads//2, 32) = 64; partial rotary over first 64 head dims
constexpr float DIT_NORM_EPS         = 1e-5f; // LayerNorm eps (pre/ff/cross_attend norms)
constexpr float DIT_QK_NORM_EPS      = 1e-6f; // q_norm/k_norm LayerNorm eps
constexpr float DIT_ROPE_BASE        = 10000.0f;
constexpr int   DIT_PREPEND_LEN      = 1;     // global_cond prepended as 1 token

// ── Skipped (non-numeric or unsupported) ──
// model_type: type=string
// model.pretransform: nested type=object
// model.conditioning: nested type=object
// model.diffusion: nested type=object
// training.timestep_sampler: nested type=string
// training.arc: nested type=object
// training.optimizer_configs: nested type=object
// training.demo: nested type=object

}  // namespace MODEL_CONFIG
