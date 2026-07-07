# Benchmark — stable-audio-open-small on motorola razr 2020 (Adreno 620)

Release fp16, fully on-device (T5 conditioning on device CPU, DiT + VAE on GPU),
seed noise from assets/ (deterministic replay — never device RNG).

## Verified run (2026-07-07)

Prompt "warm analog synth chords with soft rain", seed 42, 11 s stereo @ 44.1 kHz, 8 denoise steps:

| Metric | Value |
|---|---|
| conditioning_sec (T5-small, CPU) | 8.32 |
| dit_total_sec (8 steps) | 51.50 |
| dit_step_avg_sec | 6.44 |
| decoder_sec (VAE) | 40.43 |
| dec_host_weight_fold_upload_sec | 3.06 |
| dec_host_download_sec | 1.91 |
| total_inference_sec | **100.75** (RTF ≈ 9.2 for 11 s audio) |
| time_to_first_token_sec | 16.24 |
| peak_cpu_memory_mb | 2429 |
| latent_len | 256 |

Output gate: non-silent (peak 23007 / rms 5587 int16), bit-deterministic per (prompt, seed).

## Notes

- The activation buffer pool in this port exists for correctness, not just speed:
  per-call clCreateBuffer/clReleaseMemObject churn fragmented the Adreno heap and
  caused intermittent CL_OUT_OF_RESOURCES mid-denoise.
- Obvious optimization levers, unexplored: the depth-anything campaign patterns
  (CLBlast binary cache for the one-time compile inside TTFT; per-head/token-major
  GEMM layouts in DiT attention; conv-as-GEMM in the VAE decode — see
  src/models/depth-anything-v2-small/BENCHMARK.md in adreno-llms).
