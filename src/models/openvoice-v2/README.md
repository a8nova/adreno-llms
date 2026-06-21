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

> **Status — `clone` is a fused-compute benchmark, not yet a raw-audio→wav path.** `run_clone()` seeds enc_q from cached reference tensors (`reference/layers/enc_q_pre_output.bin` + `enc_q_output.bin`) rather than computing the content encoder from a source WAV, and it prints timing/peak rather than writing an output WAV. That is what makes it bit-exact (it starts from the reference latent). Those `reference/layers/*.bin` fixtures are gitignored, so a **clean checkout cannot run `clone` until they are regenerated** (via the cosine harness / upstream OpenVoice). A raw-audio-in → re-voiced-WAV-out front end (mel/STFT → content encoder → write WAV) is the remaining work to make this a standalone tool.

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch weights from the HuggingFace mirror (model + meta; no tokenizer — audio→audio)
../../../scripts/fetch_weights.sh openvoice-v2

# 2. Build (fp16) and deploy
NNOPT_DTYPE=fp16 ./scripts/build.sh
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 3. Run the fused clone (re-voices the source into the target tone color)
NNOPT_DTYPE=fp16 ./scripts/run_android.sh clone     # also: enc_q_wn | flow | dec | all | bench
```

The source/target tone-color embeddings are `assets/g_src.bin` and `assets/g_tgt.bin` (extract your own from reference audio with the upstream OpenVoice `ToneColorConverter.extract_se`). See the status note above for the reference-tensor input dependency.

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
