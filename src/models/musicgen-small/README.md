# MusicGen-small on Adreno (Android)

Meta's MusicGen-small (text→music) ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 620 / Snapdragon 765G). Runs the full pipeline on-device — text prompt → T5 encoder → autoregressive token LM → EnCodec decoder → **32 kHz waveform**. There is no spectrogram anywhere: audio is generated as discrete tokens, exactly like a language model predicting words, then decoded back to sound.

- **Upstream:** [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small)
- **Architecture (3 components):**
  - **Text encoder:** frozen **T5-base** (12 layers, 12 heads, vocab 32128).
  - **Decoder LM:** MusicGen transformer (24 layers, hidden 1024, 16 heads), autoregressive over **4 EnCodec codebooks × 2048 vocab** with the delayed-codebook pattern; cross-attends to the T5 encoding at every layer.
  - **Codec:** **EnCodec** 32 kHz mono, 4-codebook RVQ, ratios [8,5,4,4] → 640× → **50 Hz** frame rate. Decodes the LM's integer tokens back to a waveform.
- **Precision:** fp16 (single combined `model.fp16.bin` bundles all three components; ~1.18 GB)

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch weights from HuggingFace
../../../scripts/fetch_weights.sh musicgen-small

# 2. Build (release — required for representative perf)
NNOPT_DTYPE=fp16 ./scripts/build.sh --release

# 3. Deploy binary + kernels + weights
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 4. Generate — <max_tokens> is EnCodec frames at 50 Hz (150 ≈ 3 s of audio)
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "80s synthwave with a driving bassline" 150
adb pull /data/local/tmp/musicgen_small_inference/output.wav .
```

## Performance — Razr 2020 / Adreno 620

Release fp16, 250-token (≈5 s) clip, cool device:

| Metric | Value |
|---|---|
| Decode | **~11 tok/s** (22.9 s for 250 tokens) |
| End-to-end (5 s clip) | **~34 s** wall (TTFT ~5 s + decode ~23 s + EnCodec ~6.7 s) |
| **RTF** | **~6.9×** (≈7× slower than real time) |
| Peak GPU mem | ~2.7 GB |

MusicGen is by far the heaviest model in the repo and the Adreno 620 is the entry
tier, so it is **not real-time** — but it's a long way from where bring-up started.
The optimization campaign (full log in [BENCHMARK.md](./BENCHMARK.md)) took decode
from ~8 → **~11.5 tok/s** via **mega-layer fusion** (collapsing the per-layer
LN/QKV/proj/MLP dispatch count — the old per-dispatch `layernorm` overhead was the
bottleneck), an **fp16 image2d (texture) GEMV path**, and a **recordable-queue
replay** of the decode loop that removes per-step host dispatch cost. EnCodec decode
(~6.7 s, much of it readback) and TTFT are the remaining end-to-end costs.

## What runs where

- **GPU (OpenCL on Adreno):** T5 encoder, the full decoder LM (self-attn + cross-attn to T5, MLP, layernorm), per-codebook readout/sampling, and the EnCodec decoder (transposed convs).
- **CPU:** tokenizer / prompt construction, the autoregressive sampling loop orchestration, WAV assembly.

## Caveats / known gaps

- **Speed.** ~6.9× RTF — not real-time on Adreno 620. Remaining e2e costs are EnCodec readback (~4.8 s) and TTFT; on Adreno 660+ the same code runs materially faster.
- **fp16 only.** int8 was explored and dropped (quality regressions on the audio token stream).
- **Single prompt, single turn.** No melody-conditioning or continuation.

## Licensing

- **Code:** Apache 2.0 (this repo).
- **Weights:** MusicGen weights are **CC-BY-NC 4.0 (non-commercial)** — see the [model card](https://huggingface.co/facebook/musicgen-small). Redistributed here for research use with attribution; do not use commercially.

## Layout

```
.
├── BENCHMARK.md          # optimization log + per-kernel profile
├── CMakeLists.txt
├── assets/               # eval fixtures (test_input_ids.bin)
├── kernels/              # OpenCL kernels (.cl)
├── reference/            # PyTorch reference + cosine-validation fixtures
├── scripts/              # build / deploy / run / cos_check
├── src/                  # C++ sources (LM, T5 encoder, EnCodec decoder, ops/)
└── weights/              # fetched from HuggingFace (not committed)
```
