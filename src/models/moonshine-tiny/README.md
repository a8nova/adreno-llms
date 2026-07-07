# Moonshine-tiny on Adreno (Android)

Useful Sensors' Moonshine-tiny speech-to-text model ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 620 / Snapdragon 765G). Runs the full pipeline on-device — raw 16 kHz waveform → encoder → autoregressive decoder → text — **~3× faster than real time**, with a continuous **streaming + VAD** mode (local-agreement token committing) for live mic input. Unlike Whisper there is no fixed 30 s window: the encoder is variable-length, so short clips don't pay a padding cost.

- **Upstream:** [UsefulSensors/moonshine-tiny](https://huggingface.co/UsefulSensors/moonshine-tiny)
- **Parameters:** ~27M
- **Architecture:** Moonshine encoder-decoder (6 encoder + 6 decoder layers, d=288, 8 heads, vocab 32768), audio enc-dec ASR
- **Precision:** fp16 (~53 MB bundle)

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch weights from HuggingFace (model.fp16.bin + meta + tokenizer_vocab.bin,
#    same 3-file ASR set as whisper-tiny)
../../../scripts/fetch_weights.sh moonshine-tiny

# 2. Build (release — required for representative perf)
NNOPT_DTYPE=fp16 ./scripts/build.sh --release

# 3. Deploy binary + kernels + assets
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 4. Transcribe the bundled test clip (empty prompt slot, 64-token budget)
NNOPT_DTYPE=fp16 ./scripts/run_android.sh '' 64
# expect: "He hoped there would be stew" — token-exact vs PyTorch
```

To transcribe your own audio, push a 16 kHz mono float32 `.bin` over the test asset first:

```bash
adb push my_clip.bin /data/local/tmp/moonshine_tiny_inference/assets/test_waveform.bin
NNOPT_DTYPE=fp16 ./scripts/run_android.sh '' 64
```

> Benchmark from run 2 — run 1 repopulates the on-device kernel binary cache (~+0.5 s TTFT, once).

## Streaming + VAD (live mic)

`--stream` reads continuous 16 kHz mono float32 PCM from stdin, segments speech with an energy VAD over a sliding window, and emits `PARTIAL [t0-t1]:` lines (live, refining) and `FINAL [t0-t1]:` lines (committed). Commits use **local agreement on token ids** — the stable prefix two consecutive windows agree on is committed and never repainted; only the disagreeing tail stays PARTIAL. Same stdin/stdout protocol as the whisper-tiny port's `--stream`, so app integrations (e.g. see-and-say's `WhisperSession`) work unchanged.

```bash
# Pipe a raw f32 PCM file to test the streaming path:
adb shell "cd /data/local/tmp/moonshine_tiny_inference && LD_LIBRARY_PATH=.:/system/vendor/lib64:\$LD_LIBRARY_PATH \
    ./moonshine_tiny_inference_fp16 '' 64 --stream < clip.bin"
```

Tunables: `--vad-threshold`, `--step-ms`, `--hangover-ms`.

## Performance

Razr 2020 / Adreno 620 / Snapdragon 765G, fp16 release build, warm run on verification clip A (2.55 s of speech), measured 2026-07-07:

| Metric | Value |
|---|---:|
| **RTF** | **0.30** (~3× faster than real time) |
| Decode | 24.2 tk/s |
| TTFT | 0.33 s |
| End-to-end | 0.83 s |
| Peak CPU memory | 184.8 MB |

Stable across 9 consecutive runs (decode 23–29 tk/s, e2e 0.72–0.96 s over clips A/B/C — all **token-exact vs PyTorch**, including reproducing PyTorch's own mis-hearings). The [BENCHMARK.md](./BENCHMARK.md) r10 campaign recorded decode up to 59 tk/s / e2e 0.49 s in its DVFS-controlled interleaved window; this device's thermal state swings wall time ~2× between sessions, so treat the table above as the reproducible baseline and the ledger peaks as best-case. Correctness gate: held-out clips in `verification/` are checked id-for-id against `verification/ground_truth.json` — a change that alters any generated id on any clip is reverted.

RTF = processing_time / audio_duration; < 1.0 is faster than real time. Metric definitions (llama-bench / vLLM / MLPerf conventions) and the full optimization log are in [BENCHMARK.md](./BENCHMARK.md).

## Layout

```
.
├── BENCHMARK.md          # optimization log + methodology
├── CMakeLists.txt
├── assets/               # test_waveform.bin eval fixture
├── kernels/              # OpenCL kernels (.cl)
├── scripts/              # build / deploy / run
├── src/                  # C++ sources (main, backbone, ops/)
├── verification/         # held-out clips + PyTorch ground-truth ids
└── weights/              # fetched from HuggingFace (not committed)
```

## Licensing

- **Code:** Apache 2.0 (this repo).
- **Weights:** see the [model card](https://huggingface.co/UsefulSensors/moonshine-tiny).
