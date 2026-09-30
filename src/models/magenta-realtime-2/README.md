# Magenta RealTime 2 on Adreno (Android)

Google DeepMind's [Magenta RealTime 2](https://huggingface.co/google/magenta-realtime-2)
(`mrt2_small`, 230M) on the phone GPU through OpenCL: a continuous stream of 48 kHz stereo music,
generated live and steered by text prompts while it plays.

It runs in **real time on Adreno 840** (Snapdragon 8 Elite Gen 5). Each 40 ms frame of audio is
generated in ~31 ms, and a 10-minute session ran with zero dropouts. See [BENCHMARK.md](BENCHMARK.md).

Google ships Magenta RealTime 2 for Apple Silicon (MLX). This is an independent port and is not
affiliated with or endorsed by Google.

## Quickstart

From this directory, with an Adreno 840 device connected over `adb`:

```bash
# 1. Fetch weights from HuggingFace (11 files, ~865 MB)
../../../scripts/fetch_weights.sh magenta-realtime-2

# 2. Build (release, required for representative perf)
NNOPT_DTYPE=fp16 ./scripts/build.sh --release

# 3. Deploy binary + kernels + weights
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 4. Render 10 s of music from a prompt, then pull the WAV
adb shell "cd /data/local/tmp/magenta_realtime_2_inference && \
  LD_LIBRARY_PATH=/data/local/tmp/magenta_realtime_2_inference/lib:/system/vendor/lib64 \
  ./magenta_realtime_2_inference_fp16 --prompt 'lo-fi hip hop' --seconds 10 --out out.wav"
adb pull /data/local/tmp/magenta_realtime_2_inference/out.wav .
```

Blend several styles with weights instead of one prompt: `--blend '0.6:minimal techno|0.4:harp'`.

## Live streaming (`--serve`)

`--serve` keeps one process and its weights resident and reads one request per line on stdin. Each
request generates the next stretch of audio and continues the same piece, so a stream of short
requests plays as one continuous performance. This is how the Edgi app drives live mode.

```
$ ./magenta_realtime_2_inference_fp16 --serve
SERVE_READY                                  (stderr, once weights and kernels are loaded)
frames=10 prompt=cinematic strings           (request: 10 frames = 0.4 s of audio)
SERVE_DONE idx=0 wav=… gen_sec=… ar_sec=… codec_sec=… audio_sec=… rtf=…
```

| Key | Meaning |
|---|---|
| `prompt=` / `blend=` | A written style, or a weighted blend (`0.6:minimal techno\|0.4:harp`). Takes the rest of the line, so put it last |
| `frames=` / `seconds=` | How much audio this request generates (1 frame = 40 ms) |
| `steer=` | `1` (default): a new prompt bends the current piece instead of restarting it |
| `reset=1` | Start a new piece |
| `midi=` | Held MIDI note state for this request: `pitch:state,...[\|drum:v]` |
| `temperature=`, `topk=`, `seed=` | Sampling |
| `chunk=` | Codec batch size in frames (the codec decodes at least 5 frames, 200 ms, at a time) |
| `pipeline=` | Overlap the transformer and the codec. Keep `0` (see Known gaps) |
| `out=` | Output WAV path |

The benchmarked live settings: `frames=10 chunk=10 pipeline=0 temperature=1.1 topk=50 cfgtok=5,0,4 mctail=6`.

## Performance: Adreno 840 (Galaxy S26 Ultra)

| | |
|---|---|
| Time to generate 40 ms of audio | **~31 ms** (78% of the frame budget, 1.28× faster than playback) |
| 10 min 37 s continuous session | **0 dropouts**, no slowdown (77.9% first half, 78.3% second half) |
| Control latency (Edgi's live settings) | ~1.5 s from a prompt change to hearing it |

Full setup, method and raw readings: [BENCHMARK.md](BENCHMARK.md).

## What runs where

Everything below runs on the GPU through OpenCL; the host only tokenizes the prompt (SentencePiece)
and handles requests.

- **MusicCoCa text tower:** prompt → 768-d embedding → 12 style tokens
- **Style encoder:** style tokens (+ MIDI note state) → cross-attention conditioning
- **Temporal transformer:** one step per 40 ms frame, 41-frame attention window
- **Depthformer:** expands each frame into 12 RVQ tokens
- **SpectroStream codec decoder:** RVQ tokens → 48 kHz stereo audio

## Known gaps

1. **Real time is verified on Adreno 840 only.** Adreno 620 (Razr 2020) produces correct audio but
   far below real time. Other GPUs are untested.
2. **`mrt2_small` (230M) only.** `mrt2_base` (2.4B) streams roughly 9× more weights per frame and is
   not a real-time candidate on Adreno 840.
3. **No audio prompts.** Only MusicCoCa's text tower is ported, not its audio tower. Text prompts,
   blends and MIDI note conditioning are supported.
4. **`pipeline=1` corrupts audio** (0.93 cosine against the reference, vs 0.99985 with it off), so
   it stays off.
5. **Control latency is ~1.5 s in the benchmarked settings**, against upstream's ~200 ms. The engine
   can decode in 5-frame (200 ms) pieces; the latency comes from 0.4 s requests plus a 1.5 s
   playback buffer.
6. **int8 transformer weights** (`NNOPT_INT8SCOPE=4`) exist as a speed experiment but have never
   been validated against the reference, so they are off by default.

## Licensing

- **Code:** Apache 2.0 (this repo). Runtime behaviour follows Google's
  [magenta-realtime](https://github.com/magenta/magenta-realtime) reference implementation (Apache 2.0).
- **Weights:** Magenta RealTime 2 by Google DeepMind, licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (see the
  [model card](https://huggingface.co/google/magenta-realtime-2)). The copies on the
  adreno-llms-weights mirror are converted, not retrained: fp16 transformer weights, an fp32 codec,
  and the MusicCoCa text tower and SentencePiece vocab extracted from Google's release. Google's
  usage terms also apply: do not generate content that infringes the rights of others. Google
  claims no rights in the outputs you generate.

## Layout

- `src/main.cpp`: CLI, one-shot and `--serve` modes
- `src/backbone.cpp`: pipeline driver (transformer → depthformer → codec)
- `src/musiccoca.cpp`, `src/spm_tokenizer.cpp`: prompt → style tokens
- `src/midi.cpp`: MIDI note conditioning
- `src/ops/`: one C++ file per module
- `kernels/`: OpenCL kernels
- `reference/`: reference graph and tokens used for parity checks
- `styles/`: fixture style bundles
- `scripts/`: build, deploy, run
- `weights/`: fetched, not committed (see Quickstart)
