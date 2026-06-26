#pragma once
// pipeline.h — end-to-end pocket-tts decode: audio-prompt prime → flow-matching
// generate loop → Mimi decode → 24kHz waveform. See .nnport/PORT_SPEC.md.
#include "gpu_ops.h"
#include <vector>
#include <string>

// Generate `n_frames` audio frames (1920 samples each). noise_std scales the
// flow-matching Gaussian (0 ⇒ deterministic). Returns the concatenated fp32
// waveform. text_ids primes the backbone KV cache after the audio prompt.
std::vector<float> tts_generate(GpuOps& g, int n_frames, float noise_std,
                                const std::vector<int>& text_ids,
                                const std::vector<float>* noise_seq = nullptr,  // [n_frames*32]
                                std::vector<float>* c_out = nullptr,            // collects [n_frames*1024]
                                std::vector<float>* latent_out = nullptr,       // collects [n_frames*32]
                                bool stop_on_eos = false,                       // serve: stop when eos logit fires
                                float eos_threshold = 0.0f,                     // (n_frames is the MAX cap)
                                const std::string& voice_path = "",             // v1 audio_prompt OR v3 KV file ("" = baked)
                                bool voice_is_kv = false);                      // true ⇒ voice_path is a v3 KV cache

// Decode a single latent [32] (normalized space) → 1920 fp32 samples. For
// validating the Mimi decode path in isolation with a known-good latent.
std::vector<float> mimi_decode_one(GpuOps& g, const std::vector<float>& latent32);

// Decode the SAME latent n_frames times through the streaming Mimi state. A
// constant latent should yield a smooth sustained signal; per-frame knocking
// reveals a streaming-state discontinuity bug.
std::vector<float> mimi_decode_seq(GpuOps& g, const std::vector<float>& latent32, int n_frames);

// Streaming decode of a VARYING latent sequence (lats = [N,32]) → N*1920 samples.
std::vector<float> mimi_decode_seq_vec(GpuOps& g, const std::vector<float>& lats, int N);
// Non-streaming decode of the same sequence (one big pass). Must equal the above.
std::vector<float> mimi_decode_nonstream_pub(GpuOps& g, const std::vector<float>& lats, int N);
