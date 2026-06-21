#pragma once
// Host-side (CPU, fp32) T5 text encoder for facebook/musicgen-small.
//
// MusicGen conditions its audio decoder on a T5 encoder's hidden states
// (cross-attention K/V source). The encoder runs ONCE per generation on a
// short prompt (T ≈ 11 tokens, d_model 768, 12 blocks), so a straightforward
// host implementation is both simpler and faster-to-correctness than OpenCL
// kernels; GPU offload is a later optimization, not a correctness need.
//
// Reference: model_info/transformers_src/modeling_t5.py
//   - T5LayerNorm (RMS, no mean subtraction, no bias), eps=1e-6
//   - T5Attention.forward: NO 1/sqrt(d) scaling (folded into init);
//     relative position bias from block 0 only, shared across blocks
//   - T5LayerFF: pre-norm → wi → ReLU → wo → residual (feed_forward_proj=relu)
//   - encoder.final_layer_norm after the last block
// Then MusicGen's enc_to_dec_proj (768 → 1024, with bias) bridges into the
// decoder's cross-attention dimension (modeling_musicgen.py,
// MusicgenForConditionalGeneration.forward).
//
// Weight keys (fp16 on disk; Weights::get_host_vec decodes to fp32):
//   text_encoder.shared.weight                                   [32128, 768]
//   text_encoder.encoder.block.{B}.layer.0.SelfAttention.{q,k,v,o}.weight [768,768]
//   text_encoder.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight [32,12]
//   text_encoder.encoder.block.{B}.layer.0.layer_norm.weight     [768]
//   text_encoder.encoder.block.{B}.layer.1.DenseReluDense.{wi,wo}.weight
//   text_encoder.encoder.block.{B}.layer.1.layer_norm.weight     [768]
//   text_encoder.encoder.final_layer_norm.weight                 [768]
//   enc_to_dec_proj.{weight [1024,768], bias [1024]}

#include "weights.h"
#include <vector>
#include <cstdint>

// Runs the T5 encoder + enc_to_dec_proj on `ids` (length T).
// Returns row-major [T, 1024] fp32 encoder_hidden_states ready for the
// decoder's cross-attention; empty vector on error (logged).
// When NNOPT_DUMP_LAYERS=1 in the environment, writes per-block fp32 dumps
// layer_dumps/block_layer_{B}.bin ([T,768], pre-proj) so the standard pull +
// cosine tooling can pair them with reference/layers/block_layer_{B}_output.bin.
std::vector<float> t5_encode_host(Weights& weights, const std::vector<int32_t>& ids);
