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
constexpr bool  ADD_CROSS_ATTENTION                       = false;  // flattened from audio_encoder.add_cross_attention
constexpr int   AUDIO_CHANNELS                            = 1;  // flattened from audio_encoder.audio_channels
constexpr int   CHUNK_SIZE_FEED_FORWARD                   = 0;  // flattened from audio_encoder.chunk_size_feed_forward
constexpr int   CODEBOOK_DIM                              = 128;  // flattened from audio_encoder.codebook_dim
constexpr int   CODEBOOK_SIZE                             = 2048;  // flattened from audio_encoder.codebook_size
constexpr int   COMPRESS                                  = 2;  // flattened from audio_encoder.compress
constexpr int   DILATION_GROWTH_RATE                      = 2;  // flattened from audio_encoder.dilation_growth_rate
constexpr int   DIVERSITY_PENALTY                         = 0;  // flattened from audio_encoder.diversity_penalty
constexpr bool  DO_SAMPLE                                 = false;  // flattened from audio_encoder.do_sample
constexpr bool  EARLY_STOPPING                            = false;  // flattened from audio_encoder.early_stopping
constexpr int   ENCODER_NO_REPEAT_NGRAM_SIZE              = 0;  // flattened from audio_encoder.encoder_no_repeat_ngram_size
// NOTE: For MusicGen we treat "HIDDEN_SIZE" as the DECODER hidden size (d_model=1024).
// The audio_encoder (EnCodec) has its own internal channel sizes and is handled by
// its own traced modules; it must not override the global transformer hidden size.
constexpr int   HIDDEN_SIZE                               = 1024;  // derived from reference/config_used.json::hidden_size
constexpr bool  IS_DECODER                                = false;  // flattened from audio_encoder.is_decoder
constexpr bool  AUDIO_ENCODER_IS_ENCODER_DECODER          = false;  // flattened from audio_encoder.is_encoder_decoder
constexpr int   KERNEL_SIZE                               = 7;  // flattened from audio_encoder.kernel_size
constexpr int   LAST_KERNEL_SIZE                          = 7;  // flattened from audio_encoder.last_kernel_size
constexpr int   LENGTH_PENALTY                            = 1;  // flattened from audio_encoder.length_penalty
constexpr int   MAX_LENGTH                                = 20;  // flattened from audio_encoder.max_length
constexpr int   MIN_LENGTH                                = 0;  // flattened from audio_encoder.min_length
constexpr int   NO_REPEAT_NGRAM_SIZE                      = 0;  // flattened from audio_encoder.no_repeat_ngram_size
constexpr bool  NORMALIZE                                 = false;  // flattened from audio_encoder.normalize
constexpr int   NUM_BEAM_GROUPS                           = 1;  // flattened from audio_encoder.num_beam_groups
constexpr int   NUM_BEAMS                                 = 1;  // flattened from audio_encoder.num_beams
constexpr int   NUM_FILTERS                               = 64;  // flattened from audio_encoder.num_filters
constexpr int   NUM_LSTM_LAYERS                           = 2;  // flattened from audio_encoder.num_lstm_layers
constexpr int   NUM_RESIDUAL_LAYERS                       = 1;  // flattened from audio_encoder.num_residual_layers
constexpr int   NUM_RETURN_SEQUENCES                      = 1;  // flattened from audio_encoder.num_return_sequences
constexpr bool  OUTPUT_ATTENTIONS                         = false;  // flattened from audio_encoder.output_attentions
constexpr bool  OUTPUT_HIDDEN_STATES                      = false;  // flattened from audio_encoder.output_hidden_states
constexpr bool  OUTPUT_SCORES                             = false;  // flattened from audio_encoder.output_scores
constexpr bool  REMOVE_INVALID_VALUES                     = false;  // flattened from audio_encoder.remove_invalid_values
constexpr int   REPETITION_PENALTY                        = 1;  // flattened from audio_encoder.repetition_penalty
constexpr int   RESIDUAL_KERNEL_SIZE                      = 3;  // flattened from audio_encoder.residual_kernel_size
constexpr bool  RETURN_DICT                               = true;  // flattened from audio_encoder.return_dict
constexpr bool  RETURN_DICT_IN_GENERATE                   = false;  // flattened from audio_encoder.return_dict_in_generate
constexpr int   SAMPLING_RATE                             = 32000;  // flattened from audio_encoder.sampling_rate
constexpr int   TEMPERATURE                               = 1;  // flattened from audio_encoder.temperature
constexpr bool  TF_LEGACY_LOSS                            = false;  // flattened from audio_encoder.tf_legacy_loss
constexpr bool  TIE_ENCODER_DECODER                       = false;  // flattened from audio_encoder.tie_encoder_decoder
constexpr bool  TIE_WORD_EMBEDDINGS                       = true;  // flattened from audio_encoder.tie_word_embeddings
constexpr int   TOP_K                                     = 50;  // flattened from audio_encoder.top_k
constexpr int   TOP_P                                     = 1;  // flattened from audio_encoder.top_p
constexpr bool  TORCHSCRIPT                               = false;  // flattened from audio_encoder.torchscript
constexpr int   TRIM_RIGHT_RATIO                          = 1;  // flattened from audio_encoder.trim_right_ratio
constexpr int   TYPICAL_P                                 = 1;  // flattened from audio_encoder.typical_p
constexpr bool  USE_BFLOAT16                              = false;  // flattened from audio_encoder.use_bfloat16
constexpr bool  USE_CAUSAL_CONV                           = false;  // flattened from audio_encoder.use_causal_conv
constexpr bool  USE_CONV_SHORTCUT                         = false;  // flattened from audio_encoder.use_conv_shortcut
constexpr int   ACTIVATION_DROPOUT                        = 0;  // flattened from decoder.activation_dropout
constexpr bool  DECODER_ADD_CROSS_ATTENTION               = false;  // flattened from decoder.add_cross_attention
constexpr int   ATTENTION_DROPOUT                         = 0;  // flattened from decoder.attention_dropout
constexpr int   BOS_TOKEN_ID                              = 2048;  // flattened from decoder.bos_token_id
constexpr int   DECODER_CHUNK_SIZE_FEED_FORWARD           = 0;  // flattened from decoder.chunk_size_feed_forward
constexpr int   CLASSIFIER_DROPOUT                        = 0;  // flattened from decoder.classifier_dropout
constexpr int   DECODER_DIVERSITY_PENALTY                 = 0;  // flattened from decoder.diversity_penalty
constexpr bool  DECODER_DO_SAMPLE                         = false;  // flattened from decoder.do_sample
constexpr float DROPOUT                                   = 0.1f;  // flattened from decoder.dropout
constexpr bool  DECODER_EARLY_STOPPING                    = false;  // flattened from decoder.early_stopping
constexpr int   DECODER_ENCODER_NO_REPEAT_NGRAM_SIZE      = 0;  // flattened from decoder.encoder_no_repeat_ngram_size
constexpr int   FFN_DIM                                   = 4096;  // flattened from decoder.ffn_dim
constexpr int   DECODER_HIDDEN_SIZE                       = 1024;  // flattened from decoder.hidden_size
constexpr float INITIALIZER_FACTOR                        = 0.02f;  // flattened from decoder.initializer_factor
constexpr bool  DECODER_IS_DECODER                        = false;  // flattened from decoder.is_decoder
constexpr bool  DECODER_IS_ENCODER_DECODER                = false;  // flattened from decoder.is_encoder_decoder
constexpr int   LAYERDROP                                 = 0;  // flattened from decoder.layerdrop
constexpr int   DECODER_LENGTH_PENALTY                    = 1;  // flattened from decoder.length_penalty
constexpr int   DECODER_MAX_LENGTH                        = 20;  // flattened from decoder.max_length
constexpr int   MAX_POSITION_EMBEDDINGS                   = 2048;  // flattened from decoder.max_position_embeddings
constexpr int   DECODER_MIN_LENGTH                        = 0;  // flattened from decoder.min_length
constexpr int   DECODER_NO_REPEAT_NGRAM_SIZE              = 0;  // flattened from decoder.no_repeat_ngram_size
constexpr int   NUM_ATTENTION_HEADS                       = 16;  // flattened from decoder.num_attention_heads
constexpr int   DECODER_NUM_BEAM_GROUPS                   = 1;  // flattened from decoder.num_beam_groups
constexpr int   DECODER_NUM_BEAMS                         = 1;  // flattened from decoder.num_beams
constexpr int   NUM_CODEBOOKS                             = 4;  // flattened from decoder.num_codebooks
constexpr int   NUM_HIDDEN_LAYERS                         = 24;  // flattened from decoder.num_hidden_layers
constexpr int   DECODER_NUM_RETURN_SEQUENCES              = 1;  // flattened from decoder.num_return_sequences
constexpr bool  DECODER_OUTPUT_ATTENTIONS                 = false;  // flattened from decoder.output_attentions
constexpr bool  DECODER_OUTPUT_HIDDEN_STATES              = false;  // flattened from decoder.output_hidden_states
constexpr bool  DECODER_OUTPUT_SCORES                     = false;  // flattened from decoder.output_scores
constexpr int   PAD_TOKEN_ID                              = 2048;  // flattened from decoder.pad_token_id
constexpr bool  DECODER_REMOVE_INVALID_VALUES             = false;  // flattened from decoder.remove_invalid_values
constexpr int   DECODER_REPETITION_PENALTY                = 1;  // flattened from decoder.repetition_penalty
constexpr bool  DECODER_RETURN_DICT                       = true;  // flattened from decoder.return_dict
constexpr bool  DECODER_RETURN_DICT_IN_GENERATE           = false;  // flattened from decoder.return_dict_in_generate
constexpr bool  SCALE_EMBEDDING                           = false;  // flattened from decoder.scale_embedding
constexpr int   DECODER_TEMPERATURE                       = 1;  // flattened from decoder.temperature
constexpr bool  DECODER_TF_LEGACY_LOSS                    = false;  // flattened from decoder.tf_legacy_loss
constexpr bool  DECODER_TIE_ENCODER_DECODER               = false;  // flattened from decoder.tie_encoder_decoder
constexpr bool  DECODER_TIE_WORD_EMBEDDINGS               = false;  // flattened from decoder.tie_word_embeddings
constexpr int   DECODER_TOP_K                             = 50;  // flattened from decoder.top_k
constexpr int   DECODER_TOP_P                             = 1;  // flattened from decoder.top_p
constexpr bool  DECODER_TORCHSCRIPT                       = false;  // flattened from decoder.torchscript
constexpr int   DECODER_TYPICAL_P                         = 1;  // flattened from decoder.typical_p
constexpr bool  DECODER_USE_BFLOAT16                      = false;  // flattened from decoder.use_bfloat16
constexpr bool  USE_CACHE                                 = true;  // flattened from decoder.use_cache
constexpr int   VOCAB_SIZE                                = 2048;  // flattened from decoder.vocab_size
constexpr bool  IS_ENCODER_DECODER                        = true;
constexpr bool  TEXT_ENCODER_ADD_CROSS_ATTENTION          = false;  // flattened from text_encoder.add_cross_attention
constexpr int   TEXT_ENCODER_CHUNK_SIZE_FEED_FORWARD      = 0;  // flattened from text_encoder.chunk_size_feed_forward
constexpr int   D_FF                                      = 3072;  // flattened from text_encoder.d_ff
constexpr int   D_KV                                      = 64;  // flattened from text_encoder.d_kv
constexpr int   D_MODEL                                   = 768;  // flattened from text_encoder.d_model
constexpr int   DECODER_START_TOKEN_ID                    = 2048;  // decoder (music) BOS/pad — text_encoder.decoder_start_token_id leak fixed by scaffold flattener 2026-06-04
constexpr int   TEXT_ENCODER_DIVERSITY_PENALTY            = 0;  // flattened from text_encoder.diversity_penalty
constexpr bool  TEXT_ENCODER_DO_SAMPLE                    = false;  // flattened from text_encoder.do_sample
constexpr float DROPOUT_RATE                              = 0.1f;  // flattened from text_encoder.dropout_rate
constexpr bool  TEXT_ENCODER_EARLY_STOPPING               = false;  // flattened from text_encoder.early_stopping
constexpr int   TEXT_ENCODER_ENCODER_NO_REPEAT_NGRAM_SIZE = 0;  // flattened from text_encoder.encoder_no_repeat_ngram_size
constexpr int   EOS_TOKEN_ID                              = 1;  // flattened from text_encoder.eos_token_id
constexpr int   TEXT_ENCODER_INITIALIZER_FACTOR           = 1;  // flattened from text_encoder.initializer_factor
constexpr bool  TEXT_ENCODER_IS_DECODER                   = false;  // flattened from text_encoder.is_decoder
constexpr bool  TEXT_ENCODER_IS_ENCODER_DECODER           = true;  // flattened from text_encoder.is_encoder_decoder
constexpr bool  IS_GATED_ACT                              = false;  // flattened from text_encoder.is_gated_act
constexpr float LAYER_NORM_EPSILON                        = 1.000000e-6f;  // flattened from text_encoder.layer_norm_epsilon
constexpr int   TEXT_ENCODER_LENGTH_PENALTY               = 1;  // flattened from text_encoder.length_penalty
constexpr int   TEXT_ENCODER_MAX_LENGTH                   = 20;  // flattened from text_encoder.max_length
constexpr int   TEXT_ENCODER_MIN_LENGTH                   = 0;  // flattened from text_encoder.min_length
constexpr int   N_POSITIONS                               = 512;  // flattened from text_encoder.n_positions
constexpr int   TEXT_ENCODER_NO_REPEAT_NGRAM_SIZE         = 0;  // flattened from text_encoder.no_repeat_ngram_size
constexpr int   TEXT_ENCODER_NUM_BEAM_GROUPS              = 1;  // flattened from text_encoder.num_beam_groups
constexpr int   TEXT_ENCODER_NUM_BEAMS                    = 1;  // flattened from text_encoder.num_beams
constexpr int   NUM_DECODER_LAYERS                        = 24;  // decoder depth — text_encoder.num_decoder_layers leak fixed (scaffold flattener, 2026-06-04)
constexpr int   NUM_HEADS                                 = 12;  // flattened from text_encoder.num_heads
constexpr int   NUM_LAYERS                                = 12;  // flattened from text_encoder.num_layers
constexpr int   TEXT_ENCODER_NUM_RETURN_SEQUENCES         = 1;  // flattened from text_encoder.num_return_sequences
constexpr bool  TEXT_ENCODER_OUTPUT_ATTENTIONS            = false;  // flattened from text_encoder.output_attentions
constexpr bool  TEXT_ENCODER_OUTPUT_HIDDEN_STATES         = false;  // flattened from text_encoder.output_hidden_states
constexpr bool  OUTPUT_PAST                               = true;  // flattened from text_encoder.output_past
constexpr bool  TEXT_ENCODER_OUTPUT_SCORES                = false;  // flattened from text_encoder.output_scores
constexpr int   TEXT_ENCODER_PAD_TOKEN_ID                 = 0;  // flattened from text_encoder.pad_token_id
constexpr int   RELATIVE_ATTENTION_MAX_DISTANCE           = 128;  // flattened from text_encoder.relative_attention_max_distance
constexpr int   RELATIVE_ATTENTION_NUM_BUCKETS            = 32;  // flattened from text_encoder.relative_attention_num_buckets
constexpr bool  TEXT_ENCODER_REMOVE_INVALID_VALUES        = false;  // flattened from text_encoder.remove_invalid_values
constexpr int   TEXT_ENCODER_REPETITION_PENALTY           = 1;  // flattened from text_encoder.repetition_penalty
constexpr bool  TEXT_ENCODER_RETURN_DICT                  = true;  // flattened from text_encoder.return_dict
constexpr bool  TEXT_ENCODER_RETURN_DICT_IN_GENERATE      = false;  // flattened from text_encoder.return_dict_in_generate
constexpr int   TEXT_ENCODER_TEMPERATURE                  = 1;  // flattened from text_encoder.temperature
constexpr bool  TEXT_ENCODER_TF_LEGACY_LOSS               = false;  // flattened from text_encoder.tf_legacy_loss
constexpr bool  TEXT_ENCODER_TIE_ENCODER_DECODER          = false;  // flattened from text_encoder.tie_encoder_decoder
constexpr bool  TEXT_ENCODER_TIE_WORD_EMBEDDINGS          = true;  // flattened from text_encoder.tie_word_embeddings
constexpr int   TEXT_ENCODER_TOP_K                        = 50;  // flattened from text_encoder.top_k
constexpr int   TEXT_ENCODER_TOP_P                        = 1;  // flattened from text_encoder.top_p
constexpr bool  TEXT_ENCODER_TORCHSCRIPT                  = false;  // flattened from text_encoder.torchscript
constexpr int   TEXT_ENCODER_TYPICAL_P                    = 1;  // flattened from text_encoder.typical_p
constexpr bool  TEXT_ENCODER_USE_BFLOAT16                 = false;  // flattened from text_encoder.use_bfloat16
constexpr bool  TEXT_ENCODER_USE_CACHE                    = true;  // flattened from text_encoder.use_cache
constexpr int   TEXT_ENCODER_VOCAB_SIZE                   = 32128;  // flattened from text_encoder.vocab_size
constexpr int   HEAD_DIM                                  = 64;  // derived: DECODER_HIDDEN_SIZE / NUM_ATTENTION_HEADS (1024/16)
// MusicGen uses absolute positional embeddings (decoder.model.decoder.embed_positions.weights),
// not RoPE. Keep a non-NaN default here so the build sentinel gate passes.
constexpr float ROPE_THETA                                = 10000.0f;  // unused (USES_ROPE=false)
constexpr bool  USES_ROPE                                 = false;  // MusicGen decoder does not use RoPE
constexpr bool  USES_GQA                                  = false;  // derived: num_kv_heads != num_attention_heads

// ── Cross-family aliases ──
// Same semantic dim, different HF naming convention. Both names compile.
constexpr int   N_EMBD              = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int   MODEL_DIM           = HIDDEN_SIZE;  // alias of HIDDEN_SIZE
constexpr int   N_HEAD              = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS
constexpr int   N_LAYER             = NUM_HIDDEN_LAYERS;  // alias of NUM_HIDDEN_LAYERS
constexpr int   N_CTX               = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int   MAX_SEQ_LEN         = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int   MAX_SEQUENCE_LENGTH = MAX_POSITION_EMBEDDINGS;  // alias of MAX_POSITION_EMBEDDINGS
constexpr int   N_VOCAB             = VOCAB_SIZE;  // alias of VOCAB_SIZE
constexpr int   INTERMEDIATE_SIZE   = FFN_DIM;  // alias of FFN_DIM
constexpr int   N_INNER             = FFN_DIM;  // alias of FFN_DIM
constexpr float LAYER_NORM_EPS      = LAYER_NORM_EPSILON;  // alias of LAYER_NORM_EPSILON
constexpr float RMS_NORM_EPS        = LAYER_NORM_EPSILON;  // alias of LAYER_NORM_EPSILON
constexpr float NORM_EPS            = LAYER_NORM_EPSILON;  // alias of LAYER_NORM_EPSILON
constexpr int   NUM_KEY_VALUE_HEADS = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS (no GQA — kv heads == query heads)
constexpr int   NUM_KV_HEADS        = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS (no GQA — kv heads == query heads)
constexpr int   N_KV_HEAD           = NUM_ATTENTION_HEADS;  // alias of NUM_ATTENTION_HEADS (no GQA — kv heads == query heads)

// ── Array dimensions ──
// fixed length 1 (flattened from audio_encoder.target_bandwidths)
constexpr float TARGET_BANDWIDTHS[1] = { 2.2f };

// fixed length 4 (flattened from audio_encoder.upsampling_ratios)
constexpr int UPSAMPLING_RATIOS[4] = { 8, 5, 4, 4 };

// ── Skipped (non-numeric or unsupported) ──
// _commit_hash: type=null
// architectures: array element types not all numeric (len=1)
// audio_encoder._name_or_path: nested type=string
// audio_encoder.architectures: nested array non-numeric (len=1)
// audio_encoder.bad_words_ids: nested type=null
// audio_encoder.begin_suppress_tokens: nested type=null
// audio_encoder.bos_token_id: nested type=null
// audio_encoder.chunk_length_s: nested type=null
// audio_encoder.cross_attention_hidden_size: nested type=null
// audio_encoder.decoder_start_token_id: nested type=null
// audio_encoder.eos_token_id: nested type=null
// audio_encoder.exponential_decay_length_penalty: nested type=null
// audio_encoder.finetuning_task: nested type=null
// audio_encoder.forced_bos_token_id: nested type=null
// audio_encoder.forced_eos_token_id: nested type=null
// audio_encoder.id2label: nested type=object
// audio_encoder.label2id: nested type=object
// audio_encoder.model_type: nested type=string
// audio_encoder.norm_type: nested type=string
// audio_encoder.overlap: nested type=null
// audio_encoder.pad_mode: nested type=string
// audio_encoder.pad_token_id: nested type=null
// audio_encoder.prefix: nested type=null
// audio_encoder.problem_type: nested type=null
// audio_encoder.pruned_heads: nested type=object
// audio_encoder.sep_token_id: nested type=null
// audio_encoder.suppress_tokens: nested type=null
// audio_encoder.task_specific_params: nested type=null
// audio_encoder.tokenizer_class: nested type=null
// audio_encoder.torch_dtype: nested type=string
// audio_encoder.transformers_version: nested type=string
// decoder._name_or_path: nested type=string
// decoder.activation_function: nested type=string
// decoder.architectures: nested type=null
// decoder.bad_words_ids: nested type=null
// decoder.begin_suppress_tokens: nested type=null
// decoder.cross_attention_hidden_size: nested type=null
// decoder.decoder_start_token_id: nested type=null
// decoder.eos_token_id: nested type=null
// decoder.exponential_decay_length_penalty: nested type=null
// decoder.finetuning_task: nested type=null
// decoder.forced_bos_token_id: nested type=null
// decoder.forced_eos_token_id: nested type=null
// decoder.id2label: nested type=object
// decoder.label2id: nested type=object
// decoder.model_type: nested type=string
// decoder.prefix: nested type=null
// decoder.problem_type: nested type=null
// decoder.pruned_heads: nested type=object
// decoder.sep_token_id: nested type=null
// decoder.suppress_tokens: nested type=null
// decoder.task_specific_params: nested type=null
// decoder.tokenizer_class: nested type=null
// decoder.torch_dtype: nested type=null
// decoder.transformers_version: nested type=string
// model_type: type=string
// text_encoder._name_or_path: nested type=string
// text_encoder.architectures: nested array non-numeric (len=1)
// text_encoder.bad_words_ids: nested type=null
// text_encoder.begin_suppress_tokens: nested type=null
// text_encoder.bos_token_id: nested type=null
// text_encoder.cross_attention_hidden_size: nested type=null
// text_encoder.dense_act_fn: nested type=string
// text_encoder.exponential_decay_length_penalty: nested type=null
// text_encoder.feed_forward_proj: nested type=string
// text_encoder.finetuning_task: nested type=null
// text_encoder.forced_bos_token_id: nested type=null
// text_encoder.forced_eos_token_id: nested type=null
// text_encoder.id2label: nested type=object
// text_encoder.label2id: nested type=object
// text_encoder.model_type: nested type=string
// text_encoder.prefix: nested type=null
// text_encoder.problem_type: nested type=null
// text_encoder.pruned_heads: nested type=object
// text_encoder.sep_token_id: nested type=null
// text_encoder.suppress_tokens: nested type=null
// text_encoder.task_specific_params: nested type=object
// text_encoder.tokenizer_class: nested type=null
// text_encoder.torch_dtype: nested type=null
// text_encoder.transformers_version: nested type=string
// torch_dtype: type=string
// transformers_version: type=null

}  // namespace MODEL_CONFIG
