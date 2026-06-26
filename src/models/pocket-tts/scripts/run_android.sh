#!/bin/bash
# Run pocket-tts on the connected Android device and print the benchmark line
# (PHASE steady_per_frame_ms + PHASE audio_s/RTF). Deploy first with
# `NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh`.
#
# Usage:
#   ./scripts/run_android.sh                         # 150 frames, default sentence
#   ./scripts/run_android.sh <n_frames>              # e.g. 50  (≈4 s of audio)
#   ./scripts/run_android.sh <n_frames> <noise_std> <token_ids>
#   ./scripts/run_android.sh 50 0.7 364,1143,287
#
# <token_ids> are comma-separated SentencePiece ids (the `generate` op takes ids,
# not text — the default is "The teacher worked at the ").
set -e
cd "$(dirname "$0")/.."

# pocket-tts ships fp16; match the deployed binary unless overridden.
export NNOPT_DTYPE="${NNOPT_DTYPE:-fp16}"
. ./scripts/remote_dir.env
ADB="${ADB:-adb}"

NFRAMES="${1:-150}"
NOISE="${2:-0.7}"
TOKENS="${3:-364,1143,295,400,278,309,265,260}"   # "The teacher worked at the "

echo "pocket-tts: ${NFRAMES} frames · noise_std=${NOISE} · tokens=${TOKENS}" >&2
echo "" >&2

set +e
$ADB shell "cd ${REMOTE_DIR} && : > dummy.bin && \
  LD_LIBRARY_PATH=lib:/vendor/lib64:/system/vendor/lib64:\$LD_LIBRARY_PATH \
  ./${BINARY_NAME} weights/model.fp16.bin weights/model.fp16.meta.json \
    generate dummy.bin out.bin ${NFRAMES} ${NOISE} ${TOKENS}"
RC=$?
set -e
exit $RC
