# See & Say

A sideloadable Android demo that pairs **SmolVLM-256M** (vision-language)
and **MMS-TTS** (text-to-speech), both running on-device on the Adreno GPU
via hand-written OpenCL kernels (no NNAPI, no LiteRT, no cloud).

> Point the camera, snap, ask a question, hear the answer spoken aloud.
> Plus a dedicated text-to-speech tab with ~1140 downloadable languages.

<!-- Drag-drop a demo .mp4 into a GitHub PR/issue, copy the resulting
     https://github.com/user-attachments/assets/<uuid> URL, and replace
     the placeholder line below. GitHub renders it as an inline player. -->

https://github.com/user-attachments/assets/REPLACE-WITH-VIDEO-UUID

## Prerequisites

- Android NDK (r25+) with aarch64 toolchain
- Android SDK + Gradle (Android Studio installs both)
- ADB with a connected Adreno 6xx device (e.g. Motorola Razr 2020 / SD 765G)
- Python 3.11+ with `transformers`, `safetensors`, `torch`, `numpy` (for weight conversion only)

## Build from scratch

### 1. Build the C++ model binaries

Each model has its own CMake build. Build both in fp16 mode:

```bash
# MMS-TTS
cd src/models/mms-tts
NNOPT_DTYPE=fp16 bash scripts/build.sh --release

# SmolVLM
cd src/models/smolvlm-256m-instruct
NNOPT_DTYPE=fp16 bash scripts/build.sh --release
```

This cross-compiles arm64-v8a binaries via the NDK and links against CLBlast
(fetched automatically by CMake). Output lands in `build/fp16/`.

### 2. Prepare APK assets

```bash
cd examples/see-and-say
bash scripts/prepare_assets.sh
```

This stages the native binaries, OpenCL kernel sources, and uroman
transliteration tables into the Gradle project's `assets/` and `jniLibs/`
directories. **No model weights are bundled** — all MMS-TTS language packs
are downloaded from HuggingFace at runtime.

### 3. Build the APK

```bash
cd examples/see-and-say
./gradlew :app:assembleDebug
```

### 4. Install

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Or use the convenience script:
```bash
bash scripts/install_and_run.sh
```

## Language packs

MMS-TTS language packs (~70 MB each) are hosted on HuggingFace at
`a8nova/adreno-llms-weights` under the `mms-tts/` prefix. The app
downloads them on demand — no languages ship in the APK.

On first launch, the app shows the Language Picker so you can download
at least one language (English, Amharic, Arabic are suggested defaults).

### Converting and uploading new languages

```bash
cd src/models/mms-tts

# Convert specific languages:
python3.11 scripts/convert_all_languages.py --codes eng amh ara

# Or all ~1140:
python3.11 scripts/convert_all_languages.py

# Upload to HuggingFace:
huggingface-cli upload a8nova/adreno-llms-weights packs/ mms-tts/
```

The convert script downloads from Facebook's `mms-tts-<code>` HF repos,
converts to fp16, zips, and generates `languages.json` (the registry the
app fetches to populate the language picker).

## On-device layout

```
/data/data/com.adreno.seeandsay/
├── lib/arm64-v8a/        # jniLibs (extracted by PackageManager)
│   ├── libsmolvlm.so
│   ├── libmmstts.so
│   └── libclblast.so
└── files/
    ├── smolvlm/          # cwd for libsmolvlm.so
    │   ├── weights/      # SmolVLM weights (bundled in APK)
    │   └── kernels/*.cl
    ├── mmstts/           # cwd for libmmstts.so
    │   ├── weights/      # per-language dirs (downloaded at runtime)
    │   │   ├── eng/{model.fp16.bin, model.fp16.meta.json, tokenizer_vocab.bin}
    │   │   └── amh/...
    │   ├── kernels/*.cl
    │   └── assets/uroman/{romanization-table.txt, romanization-auto-table.txt}
    └── .extracted_v15    # asset extraction marker
```

## Architecture

| Tab | Flow |
|---|---|
| Camera | Live preview -> shutter -> frozen JPEG + question -> SmolVLM streams answer -> MMS-TTS speaks it |
| Speak | Text box -> sentence-level streaming TTS with adaptive prebuffer |
| Languages | Browse ~1140 MMS-TTS languages, download/delete packs, wipe all |

Both models run as long-lived `--interactive` subprocesses via `ProcessBuilder`.
The Kotlin app communicates over stdin/stdout/stderr pipes — no JNI, no
fork-per-query. SmolVLM weights are bundled in the APK (~490 MB); MMS-TTS
weights are downloaded per-language (~70 MB each).

## Troubleshooting

- **No TTS languages after install**: Open the Languages tab and download
  at least one language pack. English is the suggested default.
- **Loader stalls at 0%**: Check free space — the APK needs ~550 MB in
  private storage to extract SmolVLM weights + kernels.
- **Black camera preview**: Grant the camera permission when prompted.
- **Motorola Razr cover-display dialog**: Dismiss the one-time opt-in
  dialog on first launch (any choice works).
