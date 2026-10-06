#!/bin/bash
# On-device C++ streaming TTS — NO Python. The device binary does G2P
# (espeak-ng) + chunking + synth and streams raw int16 PCM on stdout; this
# wrapper pipes it to a host player. Mirrors kokoro-82m/scripts/stream_device.sh.
#
# Usage:
#   scripts/stream_device.sh "Some text to speak." [espeak-voice]
#   scripts/stream_device.sh "Text" en-gb
#   NO_PLAY=1 scripts/stream_device.sh "Text"        # write /tmp/kitten_stream.wav, no playback
#
# Binary stdout is mangled by `adb shell` (LF->CRLF) — we use `adb exec-out`,
# which is binary-clean. Device stderr (per-chunk RTF, diagnostics) is captured
# on-device and pulled back at the end.
set -euo pipefail

ADB="${ADB:-adb}"
REMOTE="${REMOTE:-/data/local/tmp/kitten_tts_inference}"
BIN="${BIN:-kitten_tts_inference}"   # fp32 = the optimized binary (see README)
SR=24000

TEXT="${1:?Usage: $0 \"text to speak\" [espeak-voice]}"
VOICE="${2:-en-us}"

# Escape single quotes for the nested adb shell command.
ESC_TEXT=$(printf '%s' "$TEXT" | sed "s/'/'\\\\''/g")

# /vendor/lib64 first: on-device libOpenCL.so must be the device's real vendor
# driver (a stub pushed from another device fails clGetPlatformIDs). ./lib
# supplies libclblast.so.
DEV_CMD="cd $REMOTE && \
LD_LIBRARY_PATH=/vendor/lib64:/system/vendor/lib64:./lib:\$LD_LIBRARY_PATH \
./$BIN --stream '$ESC_TEXT' --espeak-voice '$VOICE' 2>device_stream.log"

echo "[stream] device synth -> host playback (on-device C++ G2P, no Python)" >&2

$ADB exec-out "$DEV_CMD" \
    | ffmpeg -loglevel error -f s16le -ar $SR -ac 1 -i - -y /tmp/kitten_stream.wav || true
echo "[stream] wrote /tmp/kitten_stream.wav" >&2

if [ "${NO_PLAY:-0}" != "1" ]; then
    if command -v afplay >/dev/null 2>&1; then afplay /tmp/kitten_stream.wav
    elif command -v ffplay >/dev/null 2>&1; then ffplay -nodisp -autoexit -loglevel error /tmp/kitten_stream.wav
    fi
fi

# Surface the device-side per-chunk RTF / diagnostics.
$ADB pull "$REMOTE/device_stream.log" /tmp/kitten_device_stream.log >/dev/null 2>&1 || true
if [ -f /tmp/kitten_device_stream.log ]; then
    echo "[stream] --- device diagnostics ---" >&2
    grep -E "Phonemizer|STREAM_CHUNK|STREAM_CHUNKS|KITTEN_UTT_END|EXIT_CLEAN|ERROR|FATAL" \
        /tmp/kitten_device_stream.log >&2 || true
fi
