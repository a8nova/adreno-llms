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
constexpr float ACTIVATION_DROPOUT                   = 0.1f;
constexpr float ATTENTION_DROPOUT                    = 0.1f;
constexpr int   DEPTH_SEPARABLE_CHANNELS             = 2;
constexpr int   DEPTH_SEPARABLE_NUM_LAYERS           = 3;
constexpr float DURATION_PREDICTOR_DROPOUT           = 0.5f;
constexpr int   DURATION_PREDICTOR_FILTER_CHANNELS   = 256;
constexpr int   DURATION_PREDICTOR_FLOW_BINS         = 10;
constexpr int   DURATION_PREDICTOR_KERNEL_SIZE       = 3;
constexpr int   DURATION_PREDICTOR_NUM_FLOWS         = 4;
constexpr int   DURATION_PREDICTOR_TAIL_BOUND        = 5;
constexpr int   FFN_DIM                              = 768;
constexpr int   FFN_KERNEL_SIZE                      = 3;
constexpr int   FLOW_SIZE                            = 192;
constexpr float HIDDEN_DROPOUT                       = 0.1f;
constexpr int   HIDDEN_SIZE                          = 192;
constexpr float INITIALIZER_RANGE                    = 0.02f;
constexpr float LAYER_NORM_EPS                       = 1.000000e-5f;
constexpr float LAYERDROP                            = 0.1f;
constexpr float LEAKY_RELU_SLOPE                     = 0.1f;
constexpr float NOISE_SCALE                          = 0.667f;
constexpr float NOISE_SCALE_DURATION                 = 0.8f;
constexpr int   NUM_ATTENTION_HEADS                  = 2;
constexpr int   NUM_HIDDEN_LAYERS                    = 6;
constexpr int   NUM_SPEAKERS                         = 1;
constexpr int   POSTERIOR_ENCODER_NUM_WAVENET_LAYERS = 16;
constexpr int   PRIOR_ENCODER_NUM_FLOWS              = 4;
constexpr int   PRIOR_ENCODER_NUM_WAVENET_LAYERS     = 4;
constexpr int   SAMPLING_RATE                        = 16000;
constexpr int   SPEAKER_EMBEDDING_SIZE               = 0;
constexpr int   SPEAKING_RATE                        = 1;
constexpr int   SPECTROGRAM_BINS                     = 513;
constexpr int   UPSAMPLE_INITIAL_CHANNEL             = 512;
constexpr bool  USE_BIAS                             = true;
constexpr bool  USE_STOCHASTIC_DURATION_PREDICTION   = true;
constexpr int   VOCAB_SIZE                           = 38;
constexpr int   WAVENET_DILATION_RATE                = 1;
constexpr int   WAVENET_DROPOUT                      = 0;
constexpr int   WAVENET_KERNEL_SIZE                  = 5;
constexpr int   WINDOW_SIZE                          = 4;
constexpr int   HEAD_DIM                             = 96;  // derived: HIDDEN_SIZE / NUM_ATTENTION_HEADS
constexpr float ROPE_THETA                           = 10000.0f;  // TTS/VITS does not use RoPE; set to a safe default to satisfy Build's sentinel-NaN gate.
constexpr bool  USES_ROPE                            = false;   // no rope_theta in config.json
constexpr bool  USES_GQA                             = false;  // derived: num_kv_heads != num_attention_heads

// ── Cross-family aliases ──
// Same semantic dim, different HF naming convention. Both names compile.
constexpr int   N_EMBD              = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int   D_MODEL             = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int   MODEL_DIM           = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int   N_HEAD              = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS
constexpr int   NUM_HEADS           = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS
constexpr int   N_LAYER             = NUM_HIDDEN_LAYERS;  // alias of NUM_HIDDEN_LAYERS
constexpr int   NUM_LAYERS          = NUM_HIDDEN_LAYERS;  // alias of NUM_HIDDEN_LAYERS
constexpr int   N_VOCAB             = VOCAB_SIZE;  // alias of VOCAB_SIZE
constexpr int   INTERMEDIATE_SIZE   = FFN_DIM;  // alias of FFN_DIM
constexpr int   N_INNER             = FFN_DIM;  // alias of FFN_DIM
constexpr float LAYER_NORM_EPSILON  = LAYER_NORM_EPS;  // alias of LAYER_NORM_EPS
constexpr float RMS_NORM_EPS        = LAYER_NORM_EPS;  // alias of LAYER_NORM_EPS
constexpr float NORM_EPS            = LAYER_NORM_EPS;  // alias of LAYER_NORM_EPS
constexpr int   NUM_KEY_VALUE_HEADS = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS (no GQA — kv heads == query heads)
constexpr int   NUM_KV_HEADS        = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS (no GQA — kv heads == query heads)
constexpr int   N_KV_HEAD           = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS (no GQA — kv heads == query heads)

// ── Array dimensions ──
// fixed length 3 (not num_hidden_layers)
constexpr int RESBLOCK_KERNEL_SIZES[3] = { 3, 7, 11 };

// fixed length 4 (not num_hidden_layers)
constexpr int UPSAMPLE_KERNEL_SIZES[4] = { 16, 16, 4, 4 };

// fixed length 4 (not num_hidden_layers)
constexpr int UPSAMPLE_RATES[4] = { 8, 8, 2, 2 };

// ── Skipped (non-numeric or unsupported) ──
// architectures: array element types not all numeric (len=1)
// hidden_act: type=string
// model_type: type=string
// resblock_dilation_sizes: array element types not all numeric (len=3)
// torch_dtype: type=string
// transformers_version: type=string

}  // namespace MODEL_CONFIG
