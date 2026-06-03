#pragma once
#include <vector>

// On-device Whisper log-mel front-end (raw 16 kHz mono waveform -> log-mel [80,3000]).
// Pure CPU C++ (no OpenCL) — runs on the phone's CPU; the model runs on the GPU.
// Matches HF WhisperFeatureExtractor bit-for-bit (validated cosine 1.0 vs WhisperProcessor):
//   n_fft=400, hop=160, n_mels=80, 30s=480000 samples, slaney mel filterbank.
//
// mel_filters: row-major [80 * 201] (n_mels x n_freq), loaded from assets/mel_filters.bin.
// Returns row-major [80 * 3000] (n_mels x n_frames), the same layout main.cpp uploads
// as input_features.
std::vector<float> whisper_log_mel(const std::vector<float>& audio,
                                   const std::vector<float>& mel_filters);

// Variable-length log-mel for streaming: emits [80 * n_frames] where n_frames is
// derived from the actual audio length (even, floored ~2s, capped at max_frames /
// 3000). Lets the encoder run on the real window instead of a padded 30s — the
// key to real-time partials. Same per-frame math + Whisper normalization.
//
// min_frames pads SHORT windows up to a context floor (rounded to the 400-frame
// bucket grid so it doesn't spawn a new GEMM/JIT shape). Streaming FINALs use this
// to keep enough zero-padded context that tiny doesn't loop, without paying the
// full fixed 30s encode for a 3s sentence. 0 = no floor (PARTIAL behavior).
std::vector<float> whisper_log_mel_n(const std::vector<float>& audio,
                                     const std::vector<float>& mel_filters,
                                     int max_frames = 3000,
                                     int min_frames = 0);
