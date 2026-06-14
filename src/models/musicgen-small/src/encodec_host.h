#pragma once
#include "weights.h"
#include <vector>
#include <cstdint>

// =============================================================================
// Host-side (CPU, fp32) EnCodec 32kHz decoder for facebook/musicgen-small.
//
// Mirrors transformers/models/encodec/modeling_encodec.py (v4.40.0) exactly.
// Line ranges below refer to that file:
//   - EncodecConv1d                     : lines 87-166
//       * _get_extra_padding_for_conv1d : lines 121-133
//       * _pad1d (reflect wrapper)       : lines 135-152
//       * forward (non-causal padding)   : lines 154-166
//   - EncodecConvTranspose1d            : lines 169-209
//       * forward (trim logic)           : lines 191-209
//   - EncodecLSTM (output = lstm(x)+x)  : lines 212-224
//   - EncodecResnetBlock                : lines 227-253
//   - EncodecDecoder.__init__ (module   : lines 310-334
//       list order) / forward
//   - EncodecEuclideanCodebook.decode   : lines 356-358
//   - EncodecVectorQuantization.decode  : lines 379-382
//   - EncodecResidualVectorQuantizer    : lines 410-415
//       .decode (sum over codebooks)
//
// Config (facebook/musicgen-small audio_encoder):
//   upsampling_ratios=[8,5,4,4] num_filters=64 num_residual_layers=1
//   num_lstm_layers=2 hidden_size=128 codebook_dim=128 codebook_size=2048
//   kernel_size=7 last_kernel_size=7 residual_kernel_size=3
//   dilation_growth_rate=2 use_causal_conv=False norm_type=weight_norm
//   compress=2 trim_right_ratio=1.0 pad_mode=reflect use_conv_shortcut=False
//   audio_channels=1 sampling_rate=32000
//
// weight_norm convention (PyTorch default dim=0 for BOTH Conv1d and
// ConvTranspose1d — modeling_encodec calls nn.utils.weight_norm(self.conv)
// with no dim arg). g is indexed by weight axis 0; norm is over the other
// two axes. Conv1d weight=[out,in,k] -> g=[out,1,1], norm over (in,k).
// ConvTranspose1d weight=[in,out,k] -> g=[in,1,1], norm over (out,k).
// =============================================================================

// codes: [num_codebooks][T_frames] EnCodec RVQ token ids (0..2047).
// Returns mono PCM float waveform in [-1,1]; empty vector on error.
std::vector<float> encodec_decode_host(Weights& weights, const std::vector<std::vector<int32_t>>& codes);

class OpenCLContext;

// ── Streaming CPU EnCodec ────────────────────────────────────────────────────
// Processes the SEANet decode in frame chunks while the GPU is still decoding
// (the CPU is idle during decode in pipeline mode; the full CPU EnCodec fits
// inside the decode window → its wall cost disappears). Exactness contract:
// emitted PCM is BYTE-IDENTICAL to encodec_decode_host for the same codes —
// conv windows carry >= the receptive-field margin so interior outputs see
// the exact same inputs; the LSTM is carried statefully (exact); clip edges
// use the true reflect padding. Validated by the byte-compare harness
// (NNOPT_ENC_STREAM=2).
class EncodecStream {
public:
    EncodecStream(Weights& weights, int total_frames);
    ~EncodecStream();
    // codes: [4][total_frames], filled for frames [0, frames_avail).
    // Call repeatedly with growing frames_avail; pass is_last on the final call.
    bool push(const std::vector<std::vector<int32_t>>& codes, int frames_avail, bool is_last);
    const std::vector<float>& pcm() const;
    bool ok() const;
private:
    struct Impl;
    Impl* impl_;
};
// GPU SEANet decode (kernels/encodec.cl): the conv stack runs on the GPU,
// the LSTM stays on the CPU (Phase A; ~1 MB activation roundtrip). NOT
// bit-identical to the host path (fp32 accum, gather-form convT) — validated
// by PCM cosine vs encodec_decode_host. Returns empty on error; the caller
// (main.cpp) falls back to the host path.
std::vector<float> encodec_decode_gpu(OpenCLContext& cl_ctx, Weights& weights,
                                      const std::vector<std::vector<int32_t>>& codes);
