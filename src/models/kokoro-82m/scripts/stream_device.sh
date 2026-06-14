#!/bin/bash
# On-device C++ streaming TTS — NO Python. The device binary does G2P
# (espeak-ng) + chunking + synth and streams raw int16 PCM on stdout; this
# wrapper just pipes it to a host player for gapless playback. Replaces the
# old scripts/stream.py.
#
# Usage:
#   scripts/stream_device.sh "Some text to speak." [voice]
#   scripts/stream_device.sh "Text" en-gb
#   NO_PLAY=1 scripts/stream_device.sh "Text"          # write /tmp/kokoro_stream.wav, no playback
#
# Binary stdout is mangled by `adb shell` (LF->CRLF) — we use `adb exec-out`,
# which is binary-clean. Device stderr (per-chunk RTF, diagnostics) is captured
# on-device and pulled back at the end.
set -euo pipefail

ADB="${ADB:-adb}"
REMOTE="${REMOTE:-/data/local/tmp/Kokoro_82M_inference}"
SR=24000

TEXT="${1:?Usage: $0 \"text to speak\" [voice]}"
VOICE="${2:-en-us}"

# Escape single quotes for the nested adb shell command.
ESC_TEXT=$(printf '%s' "$TEXT" | sed "s/'/'\\\\''/g")

DEV_CMD="cd $REMOTE && NNOPT_GPU_FP32_GENERATOR=1 \
LD_LIBRARY_PATH=lib:/system/vendor/lib64:\$LD_LIBRARY_PATH \
./Kokoro_82M_inference_fp16 --stream '$ESC_TEXT' --voice '$VOICE' 2>device_stream.log"

echo "[stream] device synth -> host playback (C++ G2P, no Python)" >&2

# afplay (macOS) can't read a raw PCM stream from stdin — it needs a file. So we
# stream the device PCM through ffmpeg into a WAV, then afplay it. The synth is
# still chunked/streamed on-device; playback starts once synth completes.
# (For gapless PLAYBACK while synth is still running, use the per-chunk variant —
#  ask and I'll wire it.)
$ADB exec-out "$DEV_CMD" \
    | ffmpeg -loglevel error -f s16le -ar $SR -ac 1 -i - -y /tmp/kokoro_stream.wav || true
echo "[stream] wrote /tmp/kokoro_stream.wav" >&2

if [ "${NO_PLAY:-0}" != "1" ]; then
    afplay /tmp/kokoro_stream.wav
fi

# Surface the device-side per-chunk RTF / diagnostics.
$ADB pull "$REMOTE/device_stream.log" /tmp/kokoro_device_stream.log >/dev/null 2>&1 || true
if [ -f /tmp/kokoro_device_stream.log ]; then
    echo "[stream] --- device diagnostics ---" >&2
    grep -E "Phonemizer|STREAM_CHUNK|STREAM_CHUNKS|EXIT_CLEAN|ERROR|FATAL" \
        /tmp/kokoro_device_stream.log >&2 || true
fi
