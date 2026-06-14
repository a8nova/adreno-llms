# SeamlessM4T UnitY-small on Adreno (Android)

Meta's **SeamlessM4T UnitY-small** speech-to-speech translation model ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 620, Snapdragon 765G). All compute runs in OpenCL **fp16** (fp32 accumulation inside kernels); only beam/greedy control flow is scalar on the CPU.

- **Upstream:** [facebook/seamless-m4t-unity-small](https://huggingface.co/facebook/seamless-m4t-unity-small) (`.ptl` TorchScript UnitY)
- **Task:** speech in → translated speech out (S2ST); English/Spanish/Portuguese/Hindi/Russian output
- **Precision:** fp16

## Architecture — a cascade, not a single forward pass

Unlike the decoder-LLM ports in this repo, SeamlessM4T is a multi-stage **cascade**, so there is no single `forward()` dispatch. The stages are orchestrated by `src/pipeline.cpp` (`Pipeline::run`):

```
audio ─▶ fbank ─▶ Conformer speech encoder ─▶ text decoder (beam=5)
      ─▶ T2U synthesizer ─▶ unit decoder (greedy) ─▶ CodeHiFiGAN vocoder ─▶ waveform
```

Each stage lives in `src/ops/`; `src/gpu_ops.cpp` is the GPU op layer (all kernel wrappers) over `kernels/*.cl`. (`src/model.cpp` is an unused generated scaffold — the cascade replaces it.)

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch weights from the HuggingFace mirror
../../../scripts/fetch_weights.sh seamless-m4t-unity-small

# 2. Build (fp16) and deploy
NNOPT_DTYPE=fp16 ./scripts/build.sh
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 3. Run — ONE script for every mode:
#   ./scripts/run_android.sh <mode> <input.wav> [lang] [output.wav]
NNOPT_DTYPE=fp16 ./scripts/run_android.sh s2s  samples/tell_me_nearest_gas_station.wav eng out.wav
NNOPT_DTYPE=fp16 ./scripts/run_android.sh asr  samples/what_is_your_name.wav            # → "What is your name?"
NNOPT_DTYPE=fp16 ./scripts/run_android.sh s2tt samples/what_is_your_name.wav spa         # → "Cuál es tu nombre?"
```

`run_android.sh` is the single entrypoint — it pushes the input WAV (any sample rate / channel count; resampled to 16 kHz on device), runs the requested mode on the GPU, and either pulls the synthesized WAV (`s2s`) or prints text (`s2tt`/`asr`/`s2tt_all`). `lang` ∈ `eng` (default) `| spa | por | hin | rus`.

Text modes detokenize via the decoder's own 20,005-entry vocabulary (`tokenizer_vocab.bin`, the SentencePiece `▁` "letter" dictionary embedded in the `.ptl` — **not** the 256k full-text tokenizer). Verified bit-exact against the `.ptl` reference (`"What is your name?"`).

## Performance

Razr 2020 / Adreno 620 / Snapdragon 765G, fp16, warm on-device, 6.0 s English input clip.

| stage | encoder | text_beam | unit | vocoder | **TOTAL** |
|---|---:|---:|---:|---:|---:|
| ms | 5300 | 5000 | 2500 | 6800 | **~20,000** |

Units bit-exact (211/211), waveform cosine **0.999999** vs the `.ptl` reference. Optimized from a 271 s cold baseline (≈13× on the GPU pipeline) — full log, including the per-kernel breakdown and the on-device profiling methodology, in [BENCHMARK.md](./BENCHMARK.md).

## Layout

```
.
├── BENCHMARK.md                # full optimization log
├── CMakeLists.txt
├── kernels/                    # OpenCL kernels (Adreno-tuned)
├── reference/                  # HuggingFace / .ptl parity harness
├── samples/                    # test input clips
├── scripts/                    # build / deploy / run_android (all modes) / cos_check
├── src/                        # C++ inference (pipeline.cpp = the cascade)
└── weights/                    # populated by fetch_weights.sh (not committed)
```

## Licensing

- **Code:** Apache 2.0 (this repo).
- **Weights:** see the upstream [model card](https://huggingface.co/facebook/seamless-m4t-unity-small) for SeamlessM4T's license. Redistributed via `a8nova/adreno-llms-weights` for offline-fetch convenience.

## Examples — running each mode

All modes go through the single runner: `./scripts/run_android.sh <mode> <input.wav> [lang] [output.wav]` (after `./scripts/build.sh` + `./scripts/deploy_android.sh`). Measured on Motorola Razr 2020 (Adreno 620), fp16, warm. English text is bit-exact vs the `.ptl` reference; translation quality reflects the 350M model itself (e.g. it mishears "gas"→"guest" on the second clip).

### 1. S2S — speech → translated speech

```bash
./scripts/run_android.sh s2s samples/tell_me_nearest_gas_station.wav eng out.wav
```
Full cascade (fbank → encoder → text → units → vocoder → waveform). Output: 16 kHz mono WAV. End-to-end ≈ **20 s** for a 6 s clip (encoder 5.3 · text 5.0 · unit 2.5 · vocoder 6.8). Units bit-exact (211/211), waveform cosine 0.999999.

### 2. ASR — transcribe speech → text (same language)

```bash
./scripts/run_android.sh asr samples/what_is_your_name.wav
#  what_is_your_name.wav            ->  What is your name?
#  tell_me_nearest_gas_station.wav  ->  Can you tell me where the closest guest station is?
```
For English input, transcription and translate-to-English are the same path. E2e ≈ **5.4 s** (3 s clip) / **10.5 s** (6 s clip) = encoder + one text decode.

### 3. S2TT — speech → translated text (one target language)

```bash
./scripts/run_android.sh s2tt samples/what_is_your_name.wav spa   # eng|spa|por|hin|rus
#  -> Cuál es tu nombre?
```

### 4. S2TT (all languages) — encode once, decode every language

```bash
./scripts/run_android.sh s2tt_all samples/what_is_your_name.wav
```
The speech encoder is language-independent, so it runs once and is reused across all five text decodes.

`what_is_your_name.wav` (≈3.1 s) — all-5 total **20.7 s** (encoder 3.1 s, paid once):

| lang | text | text ms |
|---|---|---:|
| eng | What is your name? | 2333 |
| spa | Cuál es tu nombre? | 5570 |
| por | Qual é o teu nome? | 3019 |
| hin | आपका नाम क्या है? | 3663 |
| rus | Что такое ваше имя? | 3010 |

`tell_me_nearest_gas_station.wav` (≈6.0 s) — all-5 total **42.3 s** (encoder 5.5 s, paid once):

| lang | text | text ms |
|---|---|---:|
| eng | Can you tell me where the closest guest station is? | 4999 |
| spa | Puedes decirme dónde está la estación de gas más cercana? | 7280 |
| por | Podes dizer-me onde está a estação de convidação mais próxima? | 6295 |
| hin | क्या आप मुे बता सकते हैं कि निकटतम गेस्टेशन कपया है? | 10546 |
| rus | Можешь сказать мне, где находится ближайшая вокзал? | 7601 |
