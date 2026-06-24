# OpenVoice V2 (voice cloning) on Adreno (Android)

MyShell's **OpenVoice V2** tone-color converter ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 620, Snapdragon 765G). This is the **tone-color converter** half of OpenVoice — it takes an utterance and re-voices it in a target speaker's tone color (it is *not* the base text-to-speech). All compute runs in OpenCL **fp16** (fp32 accumulation inside kernels); only orchestration is scalar on the CPU.

- **Upstream:** [myshell-ai/OpenVoiceV2](https://huggingface.co/myshell-ai/OpenVoiceV2)
- **Task:** speech in → the same speech re-voiced in a target tone color (audio→audio)
- **Precision:** fp16

## Architecture — a fused converter, not a single forward pass

OpenVoice's converter is a VITS-style cascade. The fused **clone** path runs all three stages in one process with a shared weight/kernel cache (no per-stage disk dumps):

```
source audio ─▶ posterior encoder (enc_q, WaveNet) ─▶ normalizing flow ─▶ HiFi-GAN decoder ─▶ re-voiced waveform
                         ▲                                   ▲
                    g_src (source tone color)          g_tgt (target tone color)
```

Each stage lives in `src/ops/` (`enc_q.cpp`, `flow.cpp`, `dec.cpp`); `src/ops/engine.h` is the shared GPU engine over `kernels/*.cl`. The HiFi-GAN decoder is ~87% of the wall.

## Entrypoint — one binary, stage-selected by `argv[1]`

Like the seamless-m4t cascade, there is no single `Model::forward()`. `src/main.cpp` is a thin selector that dispatches a stage by `argv[1]` (default **`clone`**), and each stage's `run_*()` lives in `src/ops/*.cpp` (`run_clone` in `dec.cpp`). One CMake binary (`openvoice_v2_inference_fp16`), built by the standard `build.sh` — CLBlast comes from CMake `FetchContent`. Modes: `clone` (fused enc_q→flow→dec), `enc_q_wn`, `flow`, `dec`, `decfast`, `all`, `bench`, `hwprobe`.

> **Status — `clone` is a real raw-audio→WAV tool (with `--src`), and still a bit-exact benchmark (without).** Given `--src ref.wav` it runs the full front end on a 22.05 kHz mono WAV: CPU linear-spectrogram (`src/ov_stft.h`, mirrors `spectrogram_torch`, cosine 1.0) → `enc_q.pre` Conv1d (513→192) → WN → `proj` → posterior reparam `z = m + ε·τ·exp(logs)` → `flow(g_src)` → `flow(g_tgt, rev)` → `dec` → re-voiced `--out out.wav`. With no `--src` it falls back to seeding enc_q + `z` from the gitignored `reference/layers/*.bin` fixtures (bit-exact, no WAV) for cosine verification. Verified on device: STFT cosine 1.0 vs upstream; identity run (`g_src==g_tgt`, `NNOPT_TAU=0`) reconstructs the source at 0.992 mel-cosine. `--g-src`/`--g-tgt` override the baked embeddings; `NNOPT_TAU` (default 0.3) sets posterior noise. Input is assumed 22.05 kHz mono 16-bit PCM (resampling is the caller's job).
>
> The **`extract`** mode (`extract --in ref.wav --out g.bin`) computes a 256-d tone-color embedding from arbitrary reference audio via the `ref_enc` 2D-CNN+GRU (CPU/fp32, `src/ops/extract.cpp`) — so a clean checkout can now do the full standalone clone with self-extracted embeddings, no fixtures. Verified on device: extractor cosine 1.0 vs a torch `ReferenceEncoder` reference; end-to-end clone with self-extracted g_src/g_tgt preserves content (0.95 mel-cos) while shifting tone color (g_src·g_tgt cos 0.43).

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch weights from the HuggingFace mirror (model + meta; no tokenizer — audio→audio)
../../../scripts/fetch_weights.sh openvoice-v2

# 2. Build (fp16) and deploy
NNOPT_DTYPE=fp16 ./scripts/build.sh
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 3a. Bit-exact fused benchmark (seeds from reference fixtures, no WAV)
NNOPT_DTYPE=fp16 ./scripts/run_android.sh clone     # also: enc_q_wn | flow | dec | all | bench

# 3b. REAL audio→audio: re-voice a 22.05kHz mono WAV into the target tone color.
#     run_android.sh forwards only the mode, so invoke the binary directly:
REMOTE=/data/local/tmp/openvoice_v2_inference
adb push my_source.wav $REMOTE/src.wav
adb shell "cd $REMOTE && NNOPT_TAU=0.3 \
  LD_LIBRARY_PATH=/vendor/lib64:$REMOTE/lib:/system/vendor/lib64 \
  ./openvoice_v2_inference_fp16 clone --src src.wav --out out.wav \
  --g-src assets/g_src.bin --g-tgt assets/g_tgt.bin"
adb pull $REMOTE/out.wav .

# 3c. Extract a tone-color embedding from any reference clip (on-device):
adb push my_reference.wav $REMOTE/ref.wav
adb shell "cd $REMOTE && \
  LD_LIBRARY_PATH=/vendor/lib64:$REMOTE/lib:/system/vendor/lib64 \
  ./openvoice_v2_inference_fp16 extract --in ref.wav --out g_tgt.bin"
# → feed g_tgt.bin back into clone via --g-tgt for full standalone cloning.
```

The source/target tone-color embeddings are `assets/g_src.bin` and `assets/g_tgt.bin` (extract your own from reference audio with the upstream OpenVoice `ToneColorConverter.extract_se`, or — once `extract` mode lands — on-device). `--g-src`/`--g-tgt` override them per run.

## Performance

Razr 2020 / Adreno 620 / Snapdragon 765G, fp16, warm. **19.27 s input clip** (424,960 samples @ 22.05 kHz). RTF = wall ÷ audio duration (lower = faster; > 1.0 = slower than the audio is long). Fused clone, no per-stage dumps:

| stage | wall (s) | share |
|---|---:|---:|
| enc_q (posterior encoder) | ~2.0 | 5% |
| flow | ~3.7 | 10% |
| dec (HiFi-GAN) | ~33 | 85% |
| **total** | **~38.0** | **RTF 2.01×** |

Median **RTF ~2.01×** (best 1.97× on a cool start; thermal swing 1.97–2.03× as the foldable throttles). Peak CPU mem **~192 MB**. Bit-exact vs the reference: **flow cos 1.0000, dec cos 0.9998** (the 0.9998 is fp16 rounding in the decoder). Optimized from a 619-tuned 3.29× baseline via conv1d/convT work-group re-tunes + the fused no-dump clone mode — full per-kernel log in [BENCHMARK.md](./BENCHMARK.md). Measured 2026-06-21.

The remaining lever for a firm sub-2× regardless of temperature is **int8** via `cl_qcom_dot_product8` (work-reduction; this chip has the extension, the 619 tablet did not) — a larger, separate effort with accuracy considerations, not a tuning knob.

## Layout

```
.
├── BENCHMARK.md                # full optimization log
├── CMakeLists.txt              # one binary: openvoice_v2_inference (globs src/*.cpp + src/ops/*.cpp)
├── assets/                     # g_src.bin / g_tgt.bin tone-color embeddings
├── kernels/                    # OpenCL kernels (Adreno-tuned)
├── reference/                  # HuggingFace parity harness (cos verification; layers/ gitignored)
├── scripts/                    # build / deploy_android / run_android (stage modes) / cos_check
├── src/                        # main.cpp = stage selector; src/ops/{enc_q,flow,dec}.cpp + engine.h = the converter
└── weights/                    # populated by fetch_weights.sh (not committed)
```

## Licensing

- **Code:** Apache 2.0 (this repo).
- **Weights:** MIT — see the upstream [model card](https://huggingface.co/myshell-ai/OpenVoiceV2). Redistributed via `a8nova/adreno-llms-weights` for offline-fetch convenience.
