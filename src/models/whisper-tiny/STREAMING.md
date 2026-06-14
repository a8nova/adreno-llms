# Streaming + VAD mode (`--stream`)

Continuous speech-to-text from a live audio stream, for hooking the port into an
interactive app (e.g. the "see and say" demo). Built on the same optimized
mel → encoder → decoder pipeline as the batch path (aggregate RTF 0.529).

## What it does

- Reads **continuous 16 kHz mono float32 PCM from stdin** (no headers, raw samples).
- **Energy VAD** (per-30ms-frame RMS with hysteresis) finds speech and the pauses
  between phrases.
- **Sliding window:** while a phrase is in progress it re-transcribes the growing
  window every `--step-ms` and emits `PARTIAL [t0-t1]: <text>` (live, refining).
- When the VAD sees `--hangover-ms` of trailing silence (end of phrase), or the
  window reaches Whisper's 30 s limit, it emits `FINAL [t0-t1]: <text>` and clears
  the window. On stdin EOF any in-progress phrase is flushed as a `FINAL`.
- **`[t0-t1]` = the phrase's position in the global stdin stream** (seconds since
  the first sample fed). The host counts the samples it writes, so it can (a) drop
  events from audio fed before a UI reset — a FINAL takes seconds to decode and can
  land after the user started a new recording session — and (b) key phrases by
  start offset, making replace-vs-append structural (no text heuristics).
- **FINAL loop guard:** finals decode with a fast ~12 s padded floor; if the text
  repeats a 4-gram verbatim (Whisper-tiny re-reading padded silence — the floor is
  out-of-distribution vs its fixed-30s training), the engine logs `STREAM_REDO` on
  stderr and re-decodes that phrase with the full 30 s pad, which doesn't loop.

Energy VAD naturally segments on sentence-boundary pauses, so each spoken sentence
typically comes out as its own `FINAL`.

## Usage

```bash
# On device (binary reads stdin). Feed a raw f32 PCM file to test:
adb shell "cd $REMOTE_DIR && LD_LIBRARY_PATH=$REMOTE_DIR/lib:/system/vendor/lib64:\$LD_LIBRARY_PATH \
    ./whisper_tiny_inference_fp16 transcribe 96 --stream < bench/sample_01_audio.bin"
```

Output (stdout):
```
PARTIAL [0.60-3.90]:  Nor is Mr. Quilters' manner less interesting than he
FINAL [0.60-5.73]:  Nor is Mr. Quilters' manner less interesting than his matter.
```
`STREAM: ready …` / `STREAM: end …` / `STREAM_SEG` / `STREAM_REDO` diagnostics go
to stderr.

## Deterministic replay testing (no mic needed)

A replay harness pipes the LibriSpeech bench clips into `--stream` on-device,
**paced at 1× real time in 100 ms chunks** — byte-for-byte what the app's mic
capture does — and records timestamped event logs (JSONL). Scenarios cover:
short utterance, two utterances split by a pause, ~29 s continuous speech
(soft/force segmentation), and a stop→start session race. The logs double as
fixtures for the app-side `TranscriptReducer` JVM tests, which assert
append-only committed text, no duplicated 4-grams, LocalAgreement monotonicity,
and reset-fence isolation. (The replay harness and the app-side tests land with
the See & Say streaming integration.)

### Flags

| flag | default | meaning |
|---|---:|---|
| `--stream` | off | enable streaming mode (reads stdin) |
| `--vad-threshold <rms>` | 0.01 | per-frame RMS speech threshold (normalized f32 audio). Raise in noise, lower if speech is missed. |
| `--step-ms <ms>` | 1500 | how often to re-transcribe the growing window (partial cadence) |
| `--hangover-ms <ms>` | 800 | trailing silence that commits a phrase to `FINAL` |
| positional `max_new_tokens` | 96 (stream) | per-phrase token cap |

## Hooking into the Android app

The streaming logic is self-contained (`nnopt_run_stream` / `nnopt_transcribe_window`
in `src/main.cpp`) and the audio interface is deliberately **stdin PCM**, so the app
owns mic capture and just feeds samples. Two integration options:

1. **Process + pipe (fastest to wire):** the app spawns the binary with `--stream`,
   captures mic audio via AAudio/AudioRecord as 16 kHz mono float32, writes the raw
   samples to the child's stdin, and parses `PARTIAL:` / `FINAL:` lines from stdout.
2. **JNI (tighter integration):** lift `nnopt_transcribe_window` + the VAD/window
   loop into a `StreamingTranscriber` class with `feed(const float* samples, int n)`
   returning emitted phrases, and expose it via JNI so the app calls `feed()` with
   each mic buffer. The model-side hook it needs already exists
   (`WhisperBackbone_invalidate_encoder_cache()`), and the core is the same code as
   option 1 — just driven by `feed()` instead of `fread(stdin)`.

## Notes / limits

- **Real-time keep-up:** feeding a file runs faster than real time. For live mic at
  1× speed, each partial re-transcribes the *whole* current window, so very long
  phrases (toward 30 s) make late partials expensive (encoder cost grows with window
  length). VAD silence-commits keep windows short for normal utterance-style speech;
  if you need long continuous dictation, lower `--step-ms` impact by committing more
  aggressively (shorter `--hangover-ms`) or cap window length below 30 s.
- **VAD is energy-based** (zero-dependency). For noisy environments, a neural VAD
  (e.g. Silero ONNX) would be more robust — drop-in replaceable at the VAD step in
  `nnopt_run_stream`.
- **Mission note:** this lives in the port for now; the shared transcribe core and the
  streaming/VAD capability should be folded back into the tool's ASR scaffold
  (`prompts/.../asr.md` + `main.cpp.tmpl`) so every ASR port gets streaming, rather
  than being hand-added per port.
