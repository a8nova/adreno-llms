#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// MusicCoCa text tower — text → 12 style tokens, on device.
//
// This is the component Magenta RT-2 needs to turn a written prompt into the style the LLM is
// conditioned on. It ships only as TFLite upstream (the safetensors releases contain the
// depthformer alone), so its weights are extracted by scripts/extract_musiccoca.py into
// weights/musiccoca.fp16.bin and the forward pass is implemented here against our own kernels.
//
// Every structural fact below is proven, not inferred from names — scripts/musiccoca_oracle.py
// walks the exported graph, reimplements it in NumPy and checks against the TFLite interpreter
// itself: final embedding cosine 1.000000000 and all 12 RVQ tokens exact, per layer >= 0.99993.
// Three things it disproved, each of which silently produces plausible-but-wrong music:
//
//   * the per-layer FCs are emitted K, V, Q — not Q, K, V
//   * logits are NOT scaled by 1/sqrt(head_dim); that is folded into the query weights. The only
//     scaling is the logit cap, 50 * tanh(0.02 * QK^T)
//   * the pooler is a learned-query attention pool (12 heads x 256), not a mean over positions
//
//   ids[n] → embedding [16000,768] + learned positional [128,768]
//   12 × pre-norm layer:
//        LayerNorm → K,V,Q [768,768] → 12 heads × 64
//        logits = 50·tanh(0.02·QKᵀ) → softmax → ·V → O [768,768] → residual
//        LayerNorm → FFN 768→3072→768 (exact erf GELU) → residual
//   learned-query attention pool → 3072→768 → LayerNorm → 768-d embedding
//   residual VQ, 12 stages × [1024,768] → 12 style tokens
//
// The tower runs at the prompt's REAL token count, not the exported 128. Padded positions are
// masked out of every attention and every other op is per-position, so truncating is bit-identical
// (verified) and a 3-token prompt costs 3/128 of the exported graph.
//
// The tokens returned here feed the LLM's style encoder, which already runs on device (see
// Encoder_forward), producing the cross-attention conditioning.
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <string>
#include <vector>

#include "midi.h"
#include "opencl_context.h"
#include "weights.h"

// Maximum sequence the exported tower was frozen at. Prompts longer than this are truncated.
constexpr int kMusicCoCaMaxSeq = 128;
constexpr int kMusicCoCaDim = 768;
constexpr int kMusicCoCaHeads = 12;
constexpr int kMusicCoCaFFN = 3072;
constexpr int kMusicCoCaLayers = 12;
constexpr int kMusicCoCaStyleTokens = 12;
constexpr int kMusicCoCaPoolHeads = 12;
constexpr int kMusicCoCaPoolHeadDim = 256;
constexpr int kMusicCoCaCodebookSize = 1024;

// The conditioning vector the depthformer's style encoder consumes: 12 MusicCoCa style tokens,
// 128 note channels, 1 drum channel and 3 CFG channels, all shifted by NUM_RESERVED_TOKENS+1.
constexpr int kConditioningTokens = 144;

// Text tower forward: SentencePiece ids (leading SOS included, length 1..kMusicCoCaMaxSeq, no
// padding — pass only the real tokens) → 768-d style embedding.
// Returns false and logs on any failure; never returns a partially-computed vector.
bool musiccoca_embed_text(OpenCLContext& cl_ctx, Weights& mc,
                          const std::vector<int32_t>& ids,
                          std::vector<float>& embedding_out);

// Residual VQ: 768-d embedding → 12 style tokens.
bool musiccoca_quantize(OpenCLContext& cl_ctx, Weights& mc,
                        const std::vector<float>& embedding,
                        std::vector<int32_t>& tokens_out);

// 12 style tokens → the 144-token conditioning block Encoder_forward expects.
// `midi` is optional: the default leaves notes and drums unconstrained (-1), so every caller that
// predates live MIDI builds a byte-identical block. CFG defaults to 3.0/3.0/3.0, matching
// magenta_rt's _build_conditioning.
bool musiccoca_conditioning_tokens(const std::vector<int32_t>& style_tokens,
                                   std::vector<int32_t>& tokens_out,
                                   const MidiState* midi = nullptr);

