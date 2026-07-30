#!/bin/bash
# Run Bonsai inference on the Android device (any size — one dim-generic binary).
# Convention matches adreno-llms qwen2-5-0-5b/scripts/run_android.sh.
# Usage: ./scripts/run_android.sh "<prompt>" [max_tokens] [mode]
#   mode: gen (raw completion, default) | chat (ChatML wrap) | encode

set -e

cd "$(dirname "$0")/.."

ADB="${ADB:-adb}"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/bonsai}"

BONSAI_STORAGE="${BONSAI_STORAGE:-fp32}"
case "$BONSAI_STORAGE" in
    fp32) BIN_SUFFIX="" ;;
    fp16) BIN_SUFFIX="_f16" ;;
    *) echo "ERROR: BONSAI_STORAGE must be fp32 or fp16 (got '$BONSAI_STORAGE')" >&2; exit 1 ;;
esac
BINARY_NAME="bonsai_inference${BIN_SUFFIX}"
# Which size to run. Default 8B; override BONSAI_NNB=bonsai4b.nnb / bonsai1.7b.nnb.
NNB="${BONSAI_NNB:-bonsai8b.nnb}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 \"<prompt>\" [max_tokens] [gen|chat|encode]"
    exit 1
fi

PROMPT="$1"
MAX_TOKENS="${2:-64}"
MODE="${3:-gen}"

echo "Running $NNB on device ($MODE, $MAX_TOKENS tokens)..." >&2

# Escape single quotes for nested adb shell
ESCAPED_PROMPT=$(printf '%s' "$PROMPT" | sed "s/'/'\\\\''/g")

$ADB shell "cd $REMOTE_DIR && \
    LD_LIBRARY_PATH=/vendor/lib64:/system/vendor/lib64:\$LD_LIBRARY_PATH \
    BONSAI_KERNEL_DIR=kernels \
    ./$BINARY_NAME model/$NNB model/tokenizer.json $MODE '$ESCAPED_PROMPT' $MAX_TOKENS"
