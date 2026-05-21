// Generated from the upstream HuggingFace config.json.
//
// You CAN edit this file directly when a value is wrong — it is NOT authority.
// Preferred debugging path when a constant looks suspicious:
//   1) Verify against PyTorch's runtime in reference/config_used.json
//      (the reference capture). That file shows what the model actually
//      consumed, including nested keys (rope_parameters.rope_theta etc).
//   2) If wrong, edit this file directly.
//   3) Re-run the build and proceed.
//
// Every numeric dimension in layer code MUST come from here. Use:
//   MODEL_CONFIG::HIDDEN_SIZE                  — scalar by name
//   MODEL_CONFIG::NUM_QUERY_HEADS[layer_idx_]  — per-layer array by index
// Build refuses bare integer literals in dimension contexts in src/layers/*.
#pragma once

#include <limits>  // for std::numeric_limits — used by sentinel-NaN floats

namespace MODEL_CONFIG {

// ── Scalar dimensions ──
constexpr int   IMAGE_TOKEN_ID                             = 49190;
constexpr int   SCALE_FACTOR                               = 4;
constexpr bool  _ATTN_IMPLEMENTATION_AUTOSET               = false;  // flattened from text_config._attn_implementation_autoset
constexpr bool  _FLASH_ATTN_2_ENABLED                      = true;  // flattened from text_config._flash_attn_2_enabled
constexpr bool  ADD_CROSS_ATTENTION                        = false;  // flattened from text_config.add_cross_attention
constexpr bool  ATTENTION_BIAS                             = false;  // flattened from text_config.attention_bias
constexpr int   ATTENTION_DROPOUT                          = 0;  // flattened from text_config.attention_dropout
constexpr int   BOS_TOKEN_ID                               = 1;  // flattened from text_config.bos_token_id
constexpr int   CHUNK_SIZE_FEED_FORWARD                    = 0;  // flattened from text_config.chunk_size_feed_forward
constexpr int   DIVERSITY_PENALTY                          = 0;  // flattened from text_config.diversity_penalty
constexpr bool  DO_SAMPLE                                  = false;  // flattened from text_config.do_sample
constexpr bool  EARLY_STOPPING                             = false;  // flattened from text_config.early_stopping
constexpr int   ENCODER_NO_REPEAT_NGRAM_SIZE               = 0;  // flattened from text_config.encoder_no_repeat_ngram_size
constexpr int   EOS_TOKEN_ID                               = 2;  // flattened from text_config.eos_token_id
constexpr int   HEAD_DIM                                   = 64;  // flattened from text_config.head_dim
constexpr int   HIDDEN_SIZE                                = 576;  // flattened from text_config.hidden_size
constexpr float INITIALIZER_RANGE                          = 0.041666666666666664f;  // flattened from text_config.initializer_range
constexpr int   INTERMEDIATE_SIZE                          = 1536;  // flattened from text_config.intermediate_size
constexpr bool  IS_DECODER                                 = false;  // flattened from text_config.is_decoder
constexpr bool  IS_ENCODER_DECODER                         = false;  // flattened from text_config.is_encoder_decoder
constexpr bool  IS_LLAMA_CONFIG                            = true;  // flattened from text_config.is_llama_config
constexpr int   LENGTH_PENALTY                             = 1;  // flattened from text_config.length_penalty
constexpr int   MAX_LENGTH                                 = 20;  // flattened from text_config.max_length
constexpr int   MAX_POSITION_EMBEDDINGS                    = 8192;  // flattened from text_config.max_position_embeddings
constexpr int   MIN_LENGTH                                 = 0;  // flattened from text_config.min_length
constexpr bool  MLP_BIAS                                   = false;  // flattened from text_config.mlp_bias
constexpr int   NEFTUNE_NOISE_ALPHA                        = 0;  // flattened from text_config.neftune_noise_alpha
constexpr int   NO_REPEAT_NGRAM_SIZE                       = 0;  // flattened from text_config.no_repeat_ngram_size
constexpr int   NUM_ATTENTION_HEADS                        = 9;  // flattened from text_config.num_attention_heads
constexpr int   NUM_BEAM_GROUPS                            = 1;  // flattened from text_config.num_beam_groups
constexpr int   NUM_BEAMS                                  = 1;  // flattened from text_config.num_beams
constexpr int   NUM_HIDDEN_LAYERS                          = 30;  // flattened from text_config.num_hidden_layers
constexpr int   NUM_KEY_VALUE_HEADS                        = 3;  // flattened from text_config.num_key_value_heads
constexpr int   NUM_RETURN_SEQUENCES                       = 1;  // flattened from text_config.num_return_sequences
constexpr bool  OUTPUT_ATTENTIONS                          = false;  // flattened from text_config.output_attentions
constexpr bool  OUTPUT_HIDDEN_STATES                       = false;  // flattened from text_config.output_hidden_states
constexpr bool  OUTPUT_SCORES                              = false;  // flattened from text_config.output_scores
constexpr int   PAD_TOKEN_ID                               = 2;  // flattened from text_config.pad_token_id
constexpr int   PIXEL_SHUFFLE_FACTOR                       = 4;  // flattened from text_config.pixel_shuffle_factor
constexpr int   PRETRAINING_TP                             = 1;  // flattened from text_config.pretraining_tp
constexpr bool  QK_LAYER_NORMS                             = false;  // flattened from text_config.qk_layer_norms
constexpr bool  REMOVE_INVALID_VALUES                      = false;  // flattened from text_config.remove_invalid_values
constexpr int   REPETITION_PENALTY                         = 1;  // flattened from text_config.repetition_penalty
constexpr bool  RETURN_DICT                                = true;  // flattened from text_config.return_dict
constexpr bool  RETURN_DICT_IN_GENERATE                    = false;  // flattened from text_config.return_dict_in_generate
constexpr float RMS_NORM_EPS                               = 1.000000e-6f;  // NOTE: Idefics3RMSNorm default eps=1e-6 in modeling_idefics3.py
constexpr bool  ROPE_INTERLEAVED                           = false;  // flattened from text_config.rope_interleaved
constexpr int   ROPE_THETA                                 = 100000;  // flattened from text_config.rope_theta
constexpr int   TEMPERATURE                                = 1;  // flattened from text_config.temperature
constexpr bool  TF_LEGACY_LOSS                             = false;  // flattened from text_config.tf_legacy_loss
constexpr bool  TIE_ENCODER_DECODER                        = false;  // flattened from text_config.tie_encoder_decoder
constexpr bool  TEXT_CONFIG_TIE_WORD_EMBEDDINGS            = false;  // flattened from text_config.tie_word_embeddings
constexpr int   TOP_K                                      = 50;  // flattened from text_config.top_k
constexpr int   TOP_P                                      = 1;  // flattened from text_config.top_p
constexpr bool  TORCHSCRIPT                                = false;  // flattened from text_config.torchscript
constexpr int   TYPICAL_P                                  = 1;  // flattened from text_config.typical_p
constexpr bool  USE_BFLOAT16                               = false;  // flattened from text_config.use_bfloat16
constexpr bool  TEXT_CONFIG_USE_CACHE                      = true;  // flattened from text_config.use_cache
constexpr bool  USE_RESAMPLER                              = false;  // flattened from text_config.use_resampler
constexpr int   TEXT_CONFIG_VOCAB_SIZE                     = 49280;  // flattened from text_config.vocab_size
constexpr bool  TIE_WORD_EMBEDDINGS                        = false;
constexpr bool  USE_CACHE                                  = true;
constexpr bool  USE_BASE_SIGLIP                            = true;  // flattened from vision_config.use_base_siglip
constexpr bool  VISION_CONFIG__ATTN_IMPLEMENTATION_AUTOSET = false;  // flattened from vision_config._attn_implementation_autoset
constexpr bool  VISION_CONFIG_ADD_CROSS_ATTENTION          = false;  // flattened from vision_config.add_cross_attention
constexpr int   VISION_CONFIG_ATTENTION_DROPOUT            = 0;  // flattened from vision_config.attention_dropout
constexpr int   VISION_CONFIG_CHUNK_SIZE_FEED_FORWARD      = 0;  // flattened from vision_config.chunk_size_feed_forward
constexpr int   VISION_CONFIG_DIVERSITY_PENALTY            = 0;  // flattened from vision_config.diversity_penalty
constexpr bool  VISION_CONFIG_DO_SAMPLE                    = false;  // flattened from vision_config.do_sample
constexpr bool  VISION_CONFIG_EARLY_STOPPING               = false;  // flattened from vision_config.early_stopping
constexpr int   VISION_CONFIG_ENCODER_NO_REPEAT_NGRAM_SIZE = 0;  // flattened from vision_config.encoder_no_repeat_ngram_size
constexpr int   VISION_CONFIG_HIDDEN_SIZE                  = 768;  // flattened from vision_config.hidden_size
// IMAGE_SIZE was 512 (32x32 patch grid → 1024 patches → 64 image tokens after
// pixel-shuffle scale 4). Dropped to 384 (24x24 → 576 patches → 36 image
// tokens) to cut vision-tower compute by ~44%. Requires the position
// embedding to be bilinearly re-baked from the trained 32x32 grid to 24x24 —
// done offline by scripts/rebake_pos_embed_384.py. The connector, splice,
// and image_features path are all shape-driven dynamically; only this
// constant + NUM_IMAGE_PLACEHOLDERS in tokenizer.cpp need updating in code.
constexpr int   IMAGE_SIZE                                 = 384;  // flattened from vision_config.image_size
constexpr float VISION_CONFIG_INITIALIZER_RANGE            = 0.02f;  // flattened from vision_config.initializer_range
constexpr int   VISION_CONFIG_INTERMEDIATE_SIZE            = 3072;  // flattened from vision_config.intermediate_size
constexpr bool  VISION_CONFIG_IS_DECODER                   = false;  // flattened from vision_config.is_decoder
constexpr bool  VISION_CONFIG_IS_ENCODER_DECODER           = false;  // flattened from vision_config.is_encoder_decoder
constexpr float LAYER_NORM_EPS                             = 1.000000e-6f;  // flattened from vision_config.layer_norm_eps
constexpr int   VISION_CONFIG_LENGTH_PENALTY               = 1;  // flattened from vision_config.length_penalty
constexpr int   VISION_CONFIG_MAX_LENGTH                   = 20;  // flattened from vision_config.max_length
constexpr int   VISION_CONFIG_MIN_LENGTH                   = 0;  // flattened from vision_config.min_length
constexpr int   VISION_CONFIG_NO_REPEAT_NGRAM_SIZE         = 0;  // flattened from vision_config.no_repeat_ngram_size
constexpr int   VISION_CONFIG_NUM_ATTENTION_HEADS          = 12;  // flattened from vision_config.num_attention_heads
constexpr int   VISION_CONFIG_NUM_BEAM_GROUPS              = 1;  // flattened from vision_config.num_beam_groups
constexpr int   VISION_CONFIG_NUM_BEAMS                    = 1;  // flattened from vision_config.num_beams
constexpr int   NUM_CHANNELS                               = 3;  // flattened from vision_config.num_channels
constexpr int   VISION_CONFIG_NUM_HIDDEN_LAYERS            = 12;  // flattened from vision_config.num_hidden_layers
constexpr int   VISION_CONFIG_NUM_RETURN_SEQUENCES         = 1;  // flattened from vision_config.num_return_sequences
constexpr bool  VISION_CONFIG_OUTPUT_ATTENTIONS            = false;  // flattened from vision_config.output_attentions
constexpr bool  VISION_CONFIG_OUTPUT_HIDDEN_STATES         = false;  // flattened from vision_config.output_hidden_states
constexpr bool  VISION_CONFIG_OUTPUT_SCORES                = false;  // flattened from vision_config.output_scores
constexpr int   PATCH_SIZE                                 = 16;  // flattened from vision_config.patch_size
constexpr bool  VISION_CONFIG_REMOVE_INVALID_VALUES        = false;  // flattened from vision_config.remove_invalid_values
constexpr int   VISION_CONFIG_REPETITION_PENALTY           = 1;  // flattened from vision_config.repetition_penalty
constexpr bool  VISION_CONFIG_RETURN_DICT                  = true;  // flattened from vision_config.return_dict
constexpr bool  VISION_CONFIG_RETURN_DICT_IN_GENERATE      = false;  // flattened from vision_config.return_dict_in_generate
constexpr int   VISION_CONFIG_TEMPERATURE                  = 1;  // flattened from vision_config.temperature
constexpr bool  VISION_CONFIG_TF_LEGACY_LOSS               = false;  // flattened from vision_config.tf_legacy_loss
constexpr bool  VISION_CONFIG_TIE_ENCODER_DECODER          = false;  // flattened from vision_config.tie_encoder_decoder
constexpr bool  VISION_CONFIG_TIE_WORD_EMBEDDINGS          = false;  // flattened from vision_config.tie_word_embeddings
constexpr int   VISION_CONFIG_TOP_K                        = 50;  // flattened from vision_config.top_k
constexpr int   VISION_CONFIG_TOP_P                        = 1;  // flattened from vision_config.top_p
constexpr bool  VISION_CONFIG_TORCHSCRIPT                  = false;  // flattened from vision_config.torchscript
constexpr int   VISION_CONFIG_TYPICAL_P                    = 1;  // flattened from vision_config.typical_p
constexpr bool  VISION_CONFIG_USE_BFLOAT16                 = false;  // flattened from vision_config.use_bfloat16
constexpr int   VOCAB_SIZE                                 = 49280;
constexpr bool  USES_ROPE                                  = true;  // derived from rope_theta in config.json
constexpr bool  USES_GQA                                   = true;  // derived: num_kv_heads != num_attention_heads

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
constexpr float LAYER_NORM_EPSILON  = LAYER_NORM_EPS;  // alias of LAYER_NORM_EPS
constexpr float NORM_EPS            = LAYER_NORM_EPS;  // alias of LAYER_NORM_EPS
constexpr int   NUM_KV_HEADS        = NUM_KEY_VALUE_HEADS;  // alias of NUM_KEY_VALUE_HEADS
constexpr int   N_KV_HEAD           = NUM_KEY_VALUE_HEADS;  // alias of NUM_KEY_VALUE_HEADS
constexpr int   ROPE_BASE           = ROPE_THETA;  // alias of ROPE_THETA
constexpr int   ROPE_FREQ_CONSTANT  = ROPE_THETA;  // alias of ROPE_THETA
constexpr int   ROPE_FREQ_BASE      = ROPE_THETA;  // alias of ROPE_THETA

// ── Skipped (non-numeric or unsupported) ──
// architectures: array element types not all numeric (len=1)
// model_type: type=string
// text_config._name_or_path: nested type=string
// text_config.architectures: nested array non-numeric (len=1)
// text_config.bad_words_ids: nested type=null
// text_config.begin_suppress_tokens: nested type=null
// text_config.cross_attention_hidden_size: nested type=null
// text_config.decoder_start_token_id: nested type=null
// text_config.exponential_decay_length_penalty: nested type=null
// text_config.finetuning_task: nested type=null
// text_config.forced_bos_token_id: nested type=null
// text_config.forced_eos_token_id: nested type=null
// text_config.hidden_act: nested type=string
// text_config.id2label: nested type=object
// text_config.label2id: nested type=object
// text_config.model_type: nested type=string
// text_config.perceiver_config: nested type=object
// text_config.prefix: nested type=null
// text_config.problem_type: nested type=null
// text_config.pruned_heads: nested type=object
// text_config.rope_scaling: nested type=null
// text_config.sep_token_id: nested type=null
// text_config.suppress_tokens: nested type=null
// text_config.task_specific_params: nested type=null
// text_config.tokenizer_class: nested type=null
// text_config.torch_dtype: nested type=string
// text_config.transformers.js_config: nested type=object
// torch_dtype: type=string
// transformers_version: type=string
// transformers.js_config.kv_cache_dtype: nested type=object
// transformers.js_config: object with no numeric/boolean leaves
// vision_config._name_or_path: nested type=string
// vision_config.architectures: nested type=null
// vision_config.bad_words_ids: nested type=null
// vision_config.begin_suppress_tokens: nested type=null
// vision_config.bos_token_id: nested type=null
// vision_config.cross_attention_hidden_size: nested type=null
// vision_config.decoder_start_token_id: nested type=null
// vision_config.eos_token_id: nested type=null
// vision_config.exponential_decay_length_penalty: nested type=null
// vision_config.finetuning_task: nested type=null
// vision_config.forced_bos_token_id: nested type=null
// vision_config.forced_eos_token_id: nested type=null
// vision_config.hidden_act: nested type=string
// vision_config.id2label: nested type=object
// vision_config.label2id: nested type=object
// vision_config.max_image_size: nested type=object
// vision_config.model_type: nested type=string
// vision_config.pad_token_id: nested type=null
// vision_config.prefix: nested type=null
// vision_config.problem_type: nested type=null
// vision_config.pruned_heads: nested type=object
// vision_config.sep_token_id: nested type=null
// vision_config.size: nested type=object
// vision_config.suppress_tokens: nested type=null
// vision_config.task_specific_params: nested type=null
// vision_config.tokenizer_class: nested type=null
// vision_config.torch_dtype: nested type=null

}  // namespace MODEL_CONFIG
