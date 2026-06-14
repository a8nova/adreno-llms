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
constexpr int   GEN_ISTFT_HOP_SIZE       = 5;  // flattened from istftnet.gen_istft_hop_size
constexpr int   GEN_ISTFT_N_FFT          = 20;  // flattened from istftnet.gen_istft_n_fft
constexpr int   UPSAMPLE_INITIAL_CHANNEL = 512;  // flattened from istftnet.upsample_initial_channel
constexpr int   DIM_IN                   = 64;
constexpr float DROPOUT                  = 0.2f;
constexpr int   HIDDEN_DIM               = 512;
constexpr int   MAX_CONV_DIM             = 512;
constexpr int   MAX_DUR                  = 50;
constexpr bool  MULTISPEAKER             = true;
constexpr int   N_LAYER                  = 3;
constexpr int   N_MELS                   = 80;
constexpr int   N_TOKEN                  = 178;
constexpr int   STYLE_DIM                = 128;
constexpr int   TEXT_ENCODER_KERNEL_SIZE = 5;
constexpr int   HIDDEN_SIZE              = 768;  // flattened from plbert.hidden_size
constexpr int   NUM_ATTENTION_HEADS      = 12;  // flattened from plbert.num_attention_heads
constexpr int   INTERMEDIATE_SIZE        = 2048;  // flattened from plbert.intermediate_size
constexpr int   MAX_POSITION_EMBEDDINGS  = 512;  // flattened from plbert.max_position_embeddings
constexpr int   NUM_HIDDEN_LAYERS        = 12;  // flattened from plbert.num_hidden_layers
constexpr float PLBERT_DROPOUT           = 0.1f;  // flattened from plbert.dropout
constexpr int   _                        = 1;  // flattened from vocab.;
constexpr int   VOCAB__                  = 2;  // flattened from vocab.:
constexpr int   A                        = 24;  // flattened from vocab.A
constexpr int   I                        = 25;  // flattened from vocab.I
constexpr int   O                        = 31;  // flattened from vocab.O
constexpr int   Q                        = 33;  // flattened from vocab.Q
constexpr int   S                        = 35;  // flattened from vocab.S
constexpr int   T                        = 36;  // flattened from vocab.T
constexpr int   W                        = 39;  // flattened from vocab.W
constexpr int   Y                        = 41;  // flattened from vocab.Y
constexpr int   VOCAB_A                  = 43;  // flattened from vocab.a
constexpr int   B                        = 44;  // flattened from vocab.b
constexpr int   C                        = 45;  // flattened from vocab.c
constexpr int   D                        = 46;  // flattened from vocab.d
constexpr int   E                        = 47;  // flattened from vocab.e
constexpr int   F                        = 48;  // flattened from vocab.f
constexpr int   H                        = 50;  // flattened from vocab.h
constexpr int   VOCAB_I                  = 51;  // flattened from vocab.i
constexpr int   J                        = 52;  // flattened from vocab.j
constexpr int   K                        = 53;  // flattened from vocab.k
constexpr int   L                        = 54;  // flattened from vocab.l
constexpr int   M                        = 55;  // flattened from vocab.m
constexpr int   N                        = 56;  // flattened from vocab.n
constexpr int   VOCAB_O                  = 57;  // flattened from vocab.o
constexpr int   P                        = 58;  // flattened from vocab.p
constexpr int   VOCAB_Q                  = 59;  // flattened from vocab.q
constexpr int   R                        = 60;  // flattened from vocab.r
constexpr int   VOCAB_S                  = 61;  // flattened from vocab.s
constexpr int   VOCAB_T                  = 62;  // flattened from vocab.t
constexpr int   U                        = 63;  // flattened from vocab.u
constexpr int   V                        = 64;  // flattened from vocab.v
constexpr int   VOCAB_W                  = 65;  // flattened from vocab.w
constexpr int   X                        = 66;  // flattened from vocab.x
constexpr int   VOCAB_Y                  = 67;  // flattened from vocab.y
constexpr int   Z                        = 68;  // flattened from vocab.z
constexpr int   HEAD_DIM                 = 64;  // derived: HIDDEN_SIZE / NUM_ATTENTION_HEADS
// Kokoro uses ALBERT absolute position embeddings (no RoPE).
// Set a harmless non-NaN default to satisfy the scaffold sentinel gate.
constexpr float ROPE_THETA               = 10000.0f;
constexpr bool  USES_ROPE                = false;  // no rope_theta in config.json
constexpr bool  USES_GQA                 = false;  // derived: num_kv_heads != num_attention_heads

// ── Cross-family aliases ──
// Same semantic dim, different HF naming convention. Both names compile.
constexpr int N_EMBD              = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int D_MODEL             = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int MODEL_DIM           = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int N_HEAD              = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS
constexpr int NUM_HEADS           = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS
constexpr int NUM_LAYERS          = NUM_HIDDEN_LAYERS;  // alias of NUM_HIDDEN_LAYERS
constexpr int N_CTX               = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int N_POSITIONS         = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int MAX_SEQ_LEN         = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int MAX_SEQUENCE_LENGTH = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int N_INNER             = INTERMEDIATE_SIZE;  // alias of INTERMEDIATE_SIZE
constexpr int FFN_DIM             = INTERMEDIATE_SIZE;  // alias of INTERMEDIATE_SIZE
constexpr int NUM_KEY_VALUE_HEADS = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS (no GQA — kv heads == query heads)
constexpr int NUM_KV_HEADS        = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS (no GQA — kv heads == query heads)
constexpr int N_KV_HEAD           = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS (no GQA — kv heads == query heads)

// ── Array dimensions ──
// fixed length 2 (flattened from istftnet.upsample_kernel_sizes)
constexpr int UPSAMPLE_KERNEL_SIZES[2] = { 20, 12 };

// fixed length 2 (flattened from istftnet.upsample_rates)
constexpr int UPSAMPLE_RATES[2] = { 10, 6 };

// per-layer (flattened from istftnet.resblock_kernel_sizes)
constexpr int RESBLOCK_KERNEL_SIZES[3] = { 3, 7, 11 };

// ── Skipped (non-numeric or unsupported) ──
// istftnet.resblock_dilation_sizes: nested array non-numeric (len=3)

}  // namespace MODEL_CONFIG
