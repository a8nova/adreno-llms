# KittenTTS Nano 0.1 — Android / OpenCL port (Motorola Razr 2020, Adreno 620)

KittenML's KittenTTS Nano 0.1 text-to-speech (24 kHz) ported to C++/OpenCL for
Adreno 6xx GPUs, running fully on-device. Verified on Motorola Razr 2020
(Adreno 620 v2).

- **Upstream:** [KittenML/kitten-tts-nano-0.1](https://huggingface.co/KittenML/kitten-tts-nano-0.1)
- **Parameters:** ~15 M (nano) — one model, 8 bundled voices
- **Architecture:** StyleTTS2-style — text → phonemes (via espeak-ng) → text
  encoder + bidirectional-LSTM duration/prosody predictor → length regulation →
  decoder generator (harmonic source module + Snake-conv stack) → 24 kHz waveform
- **Precision:** fp32 (see note below)
- **Performance (Adreno 620):** warm `--serve-stream` streaming is **faster than
  real-time** — sustained **RTF 0.84–0.87** on real sentences (0.87 @ 3.4 s
  clause → 0.84 @ 3.6 s clause). A lone short opener is ~RTF 1.0–1.3 (one cold
  chunk that can't amortize). Single-shot reference utterance: **3.47 s for
  3.975 s of audio (RTF 0.87)**, 7.5× over the 26.18 s naive baseline. Full log
  in [BENCHMARK.md](BENCHMARK.md).

> **Why fp32, not fp16?** On this Adreno the hot kernels are *issue-bound on load
> instructions*, not bandwidth — fp16 halves the bytes but not the instruction
> count, and adds `vload_half`→fp32 upconvert cost, so it measures **slower**
> (RTF 1.1–1.5). fp32 is both **exact** (the source module / InstanceNorm
> reductions need fp32 to avoid overflow→NaN) **and fastest** here. The `--release`
> build defaults to fp32; that is the optimized binary. A future int8
> `cl_qcom_dot_product8` path (à la kokoro-82m) is the only remaining lever for
> RTF ~0.5 and is not yet implemented.

## Build (default = fastest)

```bash
./scripts/build.sh              # Release + fp32: THE optimized binary. Always use this.
./scripts/build.sh --debug      # per-op clFinish + layer checks (~1.4x slower) — debugging only
```

On-device streaming requires the espeak-ng static lib vendored at
`src/third_party/espeak-ng/libespeak-ng.a` (cross-compiled arm64-v8a). When
present, CMake defines `NNOPT_TTS_STREAMING` and the `--stream` / `--serve-stream`
modes are enabled; when absent, they report a clean error and the single-shot
build is unaffected.

## Weights

Weights are **not committed** — fetch them from the HuggingFace mirror:

```bash
../../../scripts/fetch_weights.sh kitten-tts
```

This pulls `weights/model.fp16.bin`, `model.fp16.meta.json`, and
`tokenizer_vocab.bin`. (fp32 weights `model.bin` are derived at convert time; the
runtime loads fp32 by default — see the precision note.)

## Run on-device

```bash
# Single-shot WAV (pinned reference ids):
./scripts/run_android.sh

# On-device streaming with host playback (wraps `--stream` via adb exec-out):
./scripts/stream_device.sh "Hello there, streaming on the GPU."

# Persistent streaming REPL (what an app drives): `--serve-stream` keeps the
# model + phonemizer resident and reads one utterance per stdin line (blank line
# to quit). Framing: stderr `KITTEN_PCM_BEGIN <n_samples> 24000` before each
# chunk's raw int16-LE PCM on stdout, `KITTEN_UTT_END` per utterance.
```

## Streaming design

- **On-device G2P** (`src/phonemizer.cpp`): espeak-ng → IPA → KittenTTS's 178-symbol
  vocab (greedy longest-match), wrapped `[0] + ids + [10]` exactly as the upstream
  tokenizer. No Python, no network.
- **Chunking** (`chunk_text`): first chunk kept tiny for low time-to-first-audio;
  later chunks grow on clause boundaries so the playback buffer outruns synthesis.
- **Per-chunk synthesis** reuses the validated single-shot `forward_graph`; the
  decoder's harmonic-source excitation noise is generated deterministically and
  auto-sized per chunk (no fixed RNG fixture needed for arbitrary text).
