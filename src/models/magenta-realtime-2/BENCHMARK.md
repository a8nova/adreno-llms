# Magenta RealTime 2: benchmark on Adreno 840

A single 10-minute live session on a Galaxy S26 Ultra, measured 2026-09-30.

**Result:** real time for the whole session. Each 40 ms frame of audio took ~31 ms to generate
(78% of the frame budget, 1.28× faster than playback), with zero dropouts and no slowdown between
the first and second five minutes.

## Setup

| | |
|---|---|
| Device | Samsung Galaxy S26 Ultra, Snapdragon 8 Elite Gen 5 for Galaxy, **Adreno 840**, Android 16 |
| Where | BrowserStack App Live (remote device farm). The phone was on USB power the whole time; battery still fell 62% → 60% under the load |
| Driver | Edgi 1.0.8 (versionCode 10) release build, which runs this engine in `--serve` mode |
| Model | `mrt2_small` (230M), fp16 transformer weights (`NNOPT_INT8SCOPE=0`), fp16 codec (the default on Adreno 8xx) |
| Requests | `frames=10` (0.4 s of audio per request), `chunk=10 pipeline=0 temperature=1.1 topk=50 cfgtok=5,0,4 mctail=6` |
| Playback | 1.5 s AudioTrack buffer, playback starts once 1.0 s is queued |
| Session | 10 min 37 s continuous; the app's Cinematic pack (10 prompts blended on the 2-D surface); the blend point moved 4 times; UI animations on (they draw on the same GPU) |

## Results

19 readings of the app's FRAME and BUFFER readouts, about every 35 s, plus the Performance panel at
5:02 and 10:06.

| Measure | 0 to 5:02 (10 readings) | 5:52 to 10:06 (9 readings) | Whole run |
|---|---|---|---|
| FRAME (compute time as a share of each 40 ms frame) | 77.9% | 78.3% | **78.1%** (range 75 to 80%) |
| Time to generate 40 ms of audio | 31.2 ms | 31.3 ms | **31.2 ms** |
| Generation speed vs playback | 1.28× | 1.28× | **1.28×** |
| Speed (session, Performance panel) | 1.00× | 1.00× | 1.00× |
| Tokens/s | 300 / 300 | 300 / 300 | 300 / 300 |
| Dropouts (AudioTrack underruns) | 0 | 0 | **0** |
| Buffer (audio queued ahead of the speaker) | median 1.26 s | median 1.40 s | median 1.32 s (1.08 to 1.48 s) |
| Control latency (buffer + ~0.2 s) | | | **~1.5 s** (1.3 to 1.7 s) |

No "catching up" or "phone is hot" warning appeared at any reading.

## Reading the numbers

- **FRAME** is the engine's generation time per request divided by the audio it produced. Under
  100% means generation is faster than playback; 1.28× headroom is 100 / 78.1.
- **Speed and Tokens/s are capped at real time by design.** Once the 1.5 s buffer is full, each
  write blocks until playback frees space, so generation is paced to playback. 1.00× and 300 / 300
  prove the stream kept up; they cannot show headroom. FRAME is the headroom figure.
- **300 tokens/s** is the real-time rate: 25 frames per second × 12 RVQ tokens per frame.
- **Control latency** is how long a prompt change takes to be heard. A change waits for the next
  0.4 s request (0.2 s on average), then plays after the audio already buffered. Upstream quotes
  ~200 ms on Apple Silicon; that comes from generating one frame at a time into a much smaller
  buffer, not from faster generation.

## Not captured

The AR vs codec time split, the slowest single request, and Android's thermal status. They are in
the engine's logcat lines, which the device farm's log viewer could not show live during the run.

## Raw readings

| Play timer | FRAME | Buffer |
|---|---:|---:|
| 0:04 | 77% | 1120 ms |
| 0:26 | 78% | 1280 ms |
| 0:38 | 77% | 1240 ms |
| 1:24 | 78% | 1080 ms |
| 1:46 | 78% | 1240 ms |
| 2:59 | 79% | 1280 ms |
| 3:25 | 79% | 1440 ms |
| 3:59 | 79% | 1360 ms |
| 4:36 | 75% | 1480 ms |
| 5:02 | 79% | 1160 ms |
| 5:52 | 77% | 1480 ms |
| 6:27 | 78% | 1440 ms |
| 7:02 | 78% | 1400 ms |
| 7:36 | 80% | 1360 ms |
| 8:10 | 79% | 1320 ms |
| 8:44 | 78% | 1240 ms |
| 9:18 | 79% | 1160 ms |
| 9:51 | 78% | 1480 ms |
| 10:06 | 78% | 1440 ms |
