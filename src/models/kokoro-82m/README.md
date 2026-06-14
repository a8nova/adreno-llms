# Kokoro-82M — Android / OpenCL port (Motorola Razr 2020, Adreno 620)

Hexgrad's Kokoro-82M text-to-speech (24 kHz) ported to C++/OpenCL for Adreno 6xx GPUs, running fully on-device. Verified on Motorola Razr 2020 (Adreno 620).

- **Upstream:** [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- **Parameters:** 82M (one model, shared across all 54 voices)
- **Architecture:** StyleTTS2-style — text → phonemes (via espeak) → style-conditioned acoustic model → iSTFT decoder → 24 kHz waveform
- **Precision:** fp16
- **Performance:** single-shot **~RTF 1.05** (~2.1 s wall for ~2.0 s audio); gapless `--serve` streaming **~1.0 RTF** (with `NNOPT_DOT8=1`). Full log in [BENCHMARK.md](BENCHMARK.md).

## Build (default = fastest)

```bash
./scripts/build.sh            # Release + fp16: THE optimized binary. Always use this.
```

The defaults produce the fastest code. Variants you should rarely need:

```bash
./scripts/build.sh --debug    # adds per-GEMM clFinish + layer checks (~1.4x slower) — debugging only
NNOPT_DTYPE=fp32 ./scripts/build.sh   # fp32 binary (slow, A/B reference)
./scripts/build.sh --clean    # force full reconfigure (CLBlast rebuild ~5 min)
```

Never benchmark a `--debug` build.

## Deploy & run

```bash
./scripts/fetch_voice.sh                          # REQUIRED first: default voice af_heart (from HF, not in git)
./scripts/fetch_espeak_lang.sh en                 # REQUIRED first: espeak data (from HF, not in git)
./scripts/deploy_android.sh                       # push binary + libs + assets

# Speak arbitrary text — two real text→speech paths:
./scripts/say.sh "Hello from the Razr"            # HOST espeak phonemize → device synth → WAV
adb exec-out "cd $REMOTE_DIR && LD_LIBRARY_PATH=... ./Kokoro_82M_inference_fp16 --stream 'Hello from the Razr' 2>/dev/null" > out.pcm
                                                  # ON-DEVICE espeak G2P → raw int16 PCM (24 kHz) to stdout
```

> ⚠️ **Capturing `--stream` PCM: discard the binary's stderr on the device
> (`2>/dev/null` *inside* the adb command).** `adb exec-out` merges remote
> stdout+stderr into one stream — the binary's diagnostic lines (`STREAM_CHUNKS`,
> per-chunk timing) otherwise get written into the raw PCM as text bytes and play
> as loud clicks at the start/middle/end. The app must read stdout-only too.

> ⚠️ **`run_android.sh "<text>" N` does NOT speak `<text>`.** Without `--stream`
> it loads a fixed pre-phonemized `test_input_ids.bin` and ignores the prompt —
> it's a synthesis/perf probe, not text→speech. Real TTS is `say.sh` (host
> phonemize) or **`--stream`** (fully on-device G2P via espeak; this is what
> exercises `assets/espeak-ng-data/`).

The binary's fast paths are **default-on** — no env vars needed for normal runs.
`say.sh` sets `NNOPT_GPU_FP32_GENERATOR=1` (the audio-correct generator path).

## Voices

The 82M model is **shared across all 54 voices** — a voice is just a small
`[510, 256]` style-embedding tensor. **No voice ships in git**; all 54 live on
HuggingFace (consistent with espeak/weights). Fetch one before deploy:

```bash
./scripts/fetch_voice.sh                 # default af_heart -> assets/voice_pack_af_heart.bin
./scripts/fetch_voice.sh am_michael      # any of the 54 (US af_/am_, GB bf_/bm_, …)
```

Maintainer tooling (publish voices to HF):
```bash
python scripts/convert_voice.py af_bella.pt assets/voice_pack_af_bella.bin   # one .pt -> .bin
./scripts/fetch_voices.sh                # fetch+convert+upload ALL 54 -> kokoro-82m/voices/<name>.bin
```

> Runtime voice **selection** isn't wired yet — the binary hardcodes
> `assets/voice_pack_af_heart.bin` (`main.cpp:216`), so `fetch_voice.sh` copies the
> chosen voice into that slot. A `--voicepack <file>` flag (~10 lines) is the
> remaining piece for an in-app voice picker.

## Languages

Kokoro phonemizes text → phonemes with **espeak-ng** (`src/phonemizer.h`); the
runtime selects the espeak language via `--voice <code>` (default `en-us`).

| Languages | espeak code | Status |
|---|---|---|
| English (US/GB) | `en-us` / `en-gb` | ✅ **base pack on HF** — `fetch_espeak_lang.sh en` (936 KB: shared phoneme core + English; on-device-verified). **Fetch this first.** |
| Spanish, French, Italian, Portuguese, Hindi | `es` `fr` `it` `pt` `hi` | ✅ **native** — espeak is exactly what Kokoro uses; fetch the pack from HF |
| Japanese, Chinese | `ja` `cmn` | ⚠️ **experimental (espeak fallback)** — Kokoro upstream uses dedicated G2P (openjtalk / jieba+pinyin); espeak is only a "close, not identical" fallback, so quality differs. Ear-check before shipping. Proper support = porting a CJK G2P on-device (future work). |

**No espeak data ships in git** (like mms-tts language packs) — it's fetched from
HF. The `en` base pack carries the shared phoneme core; other languages layer on
top (`fetch_espeak_lang.sh` auto-pulls `en` first if the core is missing):

```bash
./scripts/fetch_espeak_lang.sh en      # REQUIRED: shared core + English -> assets/espeak-ng-data/
./scripts/fetch_espeak_lang.sh es      # optional: add Spanish (auto-fetches en if needed)
./scripts/deploy_android.sh            # push to device
# real on-device TTS uses --stream (the text-ignoring run_android.sh won't work here):
adb exec-out "cd \$REMOTE_DIR && LD_LIBRARY_PATH=... ./Kokoro_82M_inference_fp16 --stream 'Hola, ¿qué tal?' --voice es 2>/dev/null" > es.pcm
# maintainer: upload ALL packs (en base + 7 languages) to HF
./scripts/upload_espeak_langs.sh       # -> kokoro-82m/espeak-lang/<code>/
```

English (`en` base) and Spanish (`es`) are **verified on-device via `--stream`** —
text-dependent output, no espeak errors. `fr/it/pt/hi` use the same espeak-native path.

> `espeak-ng-data` is the phonemizer's language data — from the
> [espeak-ng](https://github.com/espeak-ng/espeak-ng) project, **not** the Kokoro
> model repo — so it's **fetched from HF per-language, not committed** (consistent
> with how mms-tts fetches its language packs). The full set is ~19 MB / ~100
> languages; you only pull what you use.

## Useful env toggles (A/B / debugging only)

| Var | Default | Purpose |
|---|---|---|
| `NNOPT_PROFILE=1` | off | per-kernel time table at exit |
| `NNOPT_NONGEN_PROFILE=1` | off | per-stage wall breakdown |
| `NNOPT_CONV_T4X4=0` / `NNOPT_T8=0` | on | revert to legacy scalar conv (A/B) |
| `NNOPT_TEXW=0` | on | disable texture-pipe weights (A/B) |
| `NNOPT_H16CONV=0` / `NNOPT_HMATH=0` | on | disable fp16 conv storage/math (A/B) |
| `NNOPT_SKINNY_GEMM=0` | on | revert linears to CLBlast (A/B; first CLBlast call costs ~1.1 s) |
| `NNOPT_HOST_ISTFT=1` / `NNOPT_HOST_CONV_POST=1` | off | host fp64 fallbacks (precision A/B) |

Removed gates (deleted code, do not set): `NNOPT_HYBRID_RESBLOCKS`,
`NNOPT_HYBRID_FUSED`, `NNOPT_HYBRID_PROFILE` — the May-25 hybrid resblock path
was superseded by the default fp16 texture-weight convs and deleted 2026-06-06.

## Layout

- `src/ops/` — C++/OpenCL ops (`_generator_fp32.cpp` = vocoder generator, the hot 80%)
- `kernels/` — standalone .cl kernels loaded at runtime
- `scripts/` — build / deploy / run / say
- `reference/` — Python reference outputs (`output.wav`, per-layer dumps)
- `BENCHMARK.md` — milestone history, kernel profiles, optimization ledgers

## Validation rules

- Quality gate: on-device A/B via the env toggles above + ear check. Waveform
  cosine vs `reference/output.wav` is INVALID (SineGen random phase); use
  sample count + peak amplitude + HF-ratio instead.
- The PCM epilogue emits the RAW waveform (peak ≈ 0.3, parity with the
  reference). Never add loudness normalization — a 0.8/p99 rescale+clip here
  was the cause of a high-pitched distortion overlay (fixed 2026-06-06).
