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
