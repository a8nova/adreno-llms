# Whisper-tiny on Adreno (Android)

OpenAI's smallest Whisper speech-to-text model ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 620 / Snapdragon 765G). Runs the full pipeline on-device — raw 16 kHz waveform → on-device log-mel → encoder → autoregressive decoder → text — **faster than real time**, with a continuous **streaming + VAD** mode for live mic input.

- **Upstream:** [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)
- **Parameters:** ~37M
- **Architecture:** Whisper encoder-decoder (4 encoder + 4 decoder layers, d=384, 1500-frame encoder), audio enc-dec ASR
- **Precision:** fp16

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch weights from HuggingFace (~73 MB fp16 bundle)
../../../scripts/fetch_weights.sh whisper-tiny

# 2. Build (release — strips per-op debug syncs; required for representative perf)
NNOPT_DTYPE=fp16 ./scripts/build.sh --release

# 3. Deploy binary + kernels + assets
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 4. Transcribe a single 16 kHz mono float32 .bin clip
#    (positional: <prompt-slot> <max-tokens> then flags — `transcribe 96` sets the
#     token budget; bare `--audio clip.bin` would cap at the 16-token default)
NNOPT_DTYPE=fp16 ./scripts/run_android.sh transcribe 96 --audio path/to/clip.bin
```

> A single-clip run pays the one-time CLBlast JIT compile (~4–5 s) in that process,
> so its RTF reads > 1 even though steady-state is ~0.5. Use batch mode
> (`--audio-list <tsv>`, one process for many clips) for representative throughput.

> **Build note:** always benchmark/deploy the `--release` binary. The default debug
> build inserts a `clFinish` after every GEMM (~24/token) and runs ~1.4× slower —
> a uniform per-clip slowdown that is *not* thermal or DVFS. See [BENCHMARK.md](./BENCHMARK.md).

## Streaming + VAD (live mic)

`--stream` reads continuous 16 kHz mono float32 PCM from stdin, uses an energy-based VAD to segment speech, and emits `PARTIAL:` lines (refining, live) and `FINAL:` lines (committed on end-of-speech). The app owns mic capture and feeds samples — ideal for hooking into an interactive demo.

```bash
# Pipe a raw f32 PCM file to test the streaming path:
adb shell "cd $REMOTE_DIR && LD_LIBRARY_PATH=$REMOTE_DIR/lib:/system/vendor/lib64:\$LD_LIBRARY_PATH \
    ./whisper_tiny_inference_fp16 transcribe 96 --stream < clip.bin"
```

Tunables: `--vad-threshold` (0.01), `--step-ms` (1500), `--hangover-ms` (800). Full details + the Android JNI integration path in [STREAMING.md](./STREAMING.md).

## Performance

Razr 2020 / Adreno 620 / Snapdragon 765G, fp16, 10-clip LibriSpeech set, batch mode (`scripts/run_bench_batch.sh`), 3-run warm median.

| Metric | Value |
|---|---:|
| **Aggregate RTF** | **0.529** (faster than real time) |
| Steady-state RTF (JIT excluded) | ~0.485 |
| Transcripts byte-exact | 9 / 10 |
| Long clips (18–29 s) | ~0.40 |

RTF = processing_time / audio_duration; < 1.0 is faster than real time. Optimization journey (3.297 → 0.529, 6.2×): batched-GEMM attention, softmax LDS + KV caches, JIT amortization (batch mode), removing a leftover LayerNorm sync, LDS-tiled mel convs, and a **custom M=1 matvec kernel** replacing CLBlast on the decode path. Full log + per-kernel timings in [BENCHMARK.md](./BENCHMARK.md); architecture notes in [WHISPER_ARCHITECTURE.md](./WHISPER_ARCHITECTURE.md) and [SPECTROGRAM_EXPLAINED.md](./SPECTROGRAM_EXPLAINED.md).

## Layout

```
.
├── BENCHMARK.md          # optimization log + per-kernel profile
├── STREAMING.md          # --stream mode + Android integration
├── WHISPER_ARCHITECTURE.md
├── CMakeLists.txt
├── assets/               # mel filterbank + eval fixtures
├── kernels/              # OpenCL kernels (.cl)
├── reference/            # PyTorch reference + cosine-validation fixtures
├── scripts/              # build / deploy / run / benchmark
├── src/                  # C++ sources (main, backbone, ops/, mel frontend)
└── weights/              # fetched from HuggingFace (not committed)
```

## Licensing

- **Code:** Apache 2.0 (this repo).
- **Weights:** see the [model card](https://huggingface.co/openai/whisper-tiny).
