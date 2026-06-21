# SeamlessM4T UnitY-small on Adreno (Android)

Meta's **SeamlessM4T UnitY-small** speech translation model (speech→speech **and** speech→text) ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 620, Snapdragon 765G). All compute runs in OpenCL **fp16** (fp32 accumulation inside kernels); only beam/greedy control flow is scalar on the CPU.

- **Upstream:** [facebook/seamless-m4t-unity-small](https://huggingface.co/facebook/seamless-m4t-unity-small) (`.ptl` TorchScript UnitY)
- **Tasks:** speech in → translated **speech** (S2ST) or translated **text** (S2TT); same-language **transcription** (ASR); and `s2tt_all` (encode once, decode all five languages). Output languages: English / Spanish / Portuguese / Hindi / Russian.
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

Razr 2020 / Adreno 620 / Snapdragon 765G, fp16, warm on-device. **6.0 s English input clip** (`samples/tell_me_nearest_gas_station.wav`, 6.012 s). Median of **5 warm runs** per language — each a fresh process with an untimed warm-up pass (`NNOPT_WARMUP=1`) so the timed run is steady-state (per-stage `std::chrono`; GPU ~100% busy). RTF = wall ÷ 6.0 s input (lower = faster; > 1.0 = slower than the input is long).

Two latencies per target language — the full **S2ST** cascade (→ speech) and the **S2TT** text half (→ text, which stops after the text decoder and skips T2U → unit-decoder → vocoder):

| target | S2ST (→ speech) | RTF | S2TT (→ text) | RTF |
|---|---:|---:|---:|---:|
| eng | 17.4 s | **2.90×** | 9.3 s | **1.55×** |
| spa | 19.0 s | 3.16× | 11.6 s | 1.92× |
| por | 19.8 s | 3.30× | 10.6 s | 1.76× |
| hin | 26.8 s | 4.46× | 14.8 s | 2.47× |
| rus | 21.5 s | 3.58× | 11.9 s | 1.98× |

**ASR** (transcribe English speech → English text) is the eng S2TT path: **9.3 s, RTF 1.55×** — for English input, transcribe and translate-to-English are the same path.

Per-stage breakdown (eng, ms): fbank 32 · encoder 4377 · text_beam 4941 · mt_feat 202 · synth 80 · unit_greedy 2443 · vocoder 5327 · **TOTAL 17,418**. The fbank + encoder cost (~4.4 s) is language-independent; **all** per-language variation lives in `text_beam` (and, for S2ST, the unit-decoder + vocoder lengths, which grow with the number of units emitted) — Hindi is slowest because the model emits the most tokens.

Units bit-exact vs the `.ptl` reference (eng 211/211, spa 191/191, por 234/234), waveform cosine **0.999999**. Optimized from a 271 s cold baseline (≈15× on the GPU pipeline) — full per-kernel log + methodology in [BENCHMARK.md](./BENCHMARK.md). Measured 2026-06-21.

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
Full cascade (fbank → encoder → text → units → vocoder → waveform). Output: 16 kHz mono WAV. End-to-end ≈ **17.4 s** warm for the 6 s clip (encoder 4.4 · text 4.9 · unit 2.4 · vocoder 5.3). Units bit-exact (eng 211/211), waveform cosine 0.999999.

### 2. ASR — transcribe speech → text (same language)

```bash
./scripts/run_android.sh asr samples/what_is_your_name.wav
#  what_is_your_name.wav            ->  What is your name?
#  tell_me_nearest_gas_station.wav  ->  Can you tell me where the closest guest station is?
```
For English input, transcription and translate-to-English are the same path. E2e ≈ **4.5 s** (3 s clip) / **9.3 s** (6 s clip) = encoder + one text decode.

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

`what_is_your_name.wav` (≈3.1 s) — all-5 total **19.5 s** (encoder 2.2 s, paid once):

| lang | text | text ms |
|---|---|---:|
| eng | What is your name? | 2324 |
| spa | Cuál es tu nombre? | 5533 |
| por | Qual é o teu nome? | 2944 |
| hin | आपका नाम क्या है? | 3558 |
| rus | Что такое ваше имя? | 2931 |

`tell_me_nearest_gas_station.wav` (≈6.0 s) — all-5 total **40.5 s** (encoder 4.4 s, paid once):

| lang | text | text ms |
|---|---|---:|
| eng | Can you tell me where the closest guest station is? | 4919 |
| spa | Puedes decirme dónde está la estación de gas más cercana? | 7157 |
| por | Podes dizer-me onde está a estação de convidação mais próxima? | 6196 |
| hin | क्या आप मुे बता सकते हैं कि निकटतम गेस्टेशन कपया है? | 10393 |
| rus | Можешь сказать мне, где находится ближайшая вокзал? | 7436 |

### 5. Warm server (`--serve`) — ⚠️ experimental

`--serve` keeps the model resident (one 646 MB load) and reads one request per stdin line — `<mode> <lang> <in_wav> <out_wav>` (`<out_wav>` = `-` for text modes) — so requests 2+ skip the cold load. It prints `SEAMLESS_READY` once warm and `SEAMLESS_DONE` after each reply (both on stderr); s2tt/asr emit text on stdout, s2s writes the WAV.

```bash
printf 's2s eng assets/input.wav out.wav\ns2tt spa assets/input.wav -\n' \
  | adb shell "cd $REMOTE_DIR && LD_LIBRARY_PATH=lib:/system/vendor/lib64:\$LD_LIBRARY_PATH \
      ./seamless_m4t_unity_small_inference_fp16 s2s --serve"
```

> ⚠️ **Known bug:** GPU memory is not released between requests, so under repeated requests — especially with varying output lengths — the process aborts with `clReleaseMemObject (-38)` after ~10 runs. Use it for a handful of warm requests only. For batch benchmarking, spawn one fresh process per measurement with `NNOPT_WARMUP=1` (untimed warm-up pass → timed steady-state run) instead.
