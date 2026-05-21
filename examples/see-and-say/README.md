# See & Say

A sideloadable Android demo that pairs **SmolVLM-256M** and **MMS-TTS-Eng**,
both running on-device on the Adreno GPU via hand-written OpenCL kernels
(no NNAPI, no LiteRT, no cloud).

> Point the camera, snap, ask a question, hear the answer spoken aloud.
> Plus a pure text→audio tab for the TTS-only demo path.

This isn't a Play Store app — it's a tangible artifact you can hand to
someone with an Adreno 6xx phone (e.g. Motorola Razr 2020 / Snapdragon 765G).

## What's in here

Both models live as native arm64-v8a binaries from the parent repo:
`src/models/smolvlm-256m-instruct/` and `src/models/mms-tts/`. We don't
rewrite them as JNI libs — we ship them as `lib*.so` under `jniLibs/arm64-v8a/`
and invoke them via `ProcessBuilder`. Android's PackageManager extracts
anything matching `lib*.so` to `nativeLibraryDir`, which is one of the few
app-private locations with `exec` permission on Android 10+.

```
nativeLibraryDir/
  libsmolvlm.so        ← renamed SmolVLM_256M_Instruct_inference_fp16
  libmmstts.so         ← renamed mms_tts_eng_inference_fp16
  libclblast.so        ← shared GEMM backend, stripped
```

Weights, OpenCL kernel source, and uroman tables are bundled as APK assets
and unpacked to the app's private `filesDir/` on first launch (~30–60 s).

## Build & install

```
./scripts/install_and_run.sh        # debug, full build + install
./scripts/install_and_run.sh --skip-prepare   # skip the asset staging step
```

`prepare_assets.sh` expects fp16 builds at:
- `src/models/smolvlm-256m-instruct/build/fp16/`
- `src/models/mms-tts/build/fp16/`

If those don't exist, run `scripts/build.sh` in each model dir first with
`NNOPT_DTYPE=fp16`.

APK is sideload-sized (~650 MB; fp16 weights aren't compressible). Install
with `adb install -r --no-incremental` (the install script handles this).

## On-device layout (after first launch)

```
/data/data/com.adreno.seeandsay/
├── lib/arm64-v8a/        ← jniLibs, populated by PackageManager
└── files/
    ├── smolvlm/          ← cwd for libsmolvlm.so
    │   ├── weights/{model.fp16.bin, model.fp16.meta.json, tokenizer_vocab.bin}
    │   └── kernels/*.cl
    ├── mmstts/           ← cwd for libmmstts.so
    │   ├── weights/{model.fp16.bin, model.fp16.meta.json, tokenizer_vocab.bin}
    │   ├── kernels/*.cl
    │   └── assets/uroman/{romanization-table.txt, chars-to-delete.txt}
    ├── captures/         ← CameraX writes JPEGs here
    └── .extracted_v1     ← first-launch marker
```

Bump the marker (`.extracted_v2` …) when the asset layout changes — the
extractor re-runs on missing flag.

## What it does

| Tab | Flow |
|---|---|
| 📷 Camera | Live preview → shutter → frozen JPEG + question field → "Ask & Speak" runs SmolVLM, streams the answer per token, then synthesizes via MMS-TTS and plays the WAV through the phone speaker. |
| 🔊 Speak | Text box → "Speak" runs MMS-TTS directly, shows audio duration, fwd time, and RTF. |

Only one prompt-and-answer at a time; no multi-turn KV reuse (each invocation
re-loads weights). For a demo this is fine; the first run carries the cold
load tax (~1 s), subsequent runs reuse the warmed-up OpenCL kernel cache
via `nnopt_build_program_cached`.

## Out of scope

No model picker, no language picker, no voice / pitch controls, no
streaming TTS, no multi-turn chat, no settings, no Play Store packaging,
no analytics. Adding any of these is a separate scope from this v1.

## Troubleshooting

- **Motorola Razr cover-display opt-in dialog**: first launch on a Razr
  Foldable triggers a one-time Motorola system dialog asking whether the
  app may be used on the cover display. Dismiss it (any choice works) —
  the prompt won't re-appear and our app will then launch normally.
- **Loader stalls at 0%**: check `adb shell run-as com.adreno.seeandsay df files`
  — APK requires ~700 MB free in private storage to extract.
- **Black camera preview**: grant the camera permission. The app handles
  the permission flow; if the user denies, the rationale screen waits.
