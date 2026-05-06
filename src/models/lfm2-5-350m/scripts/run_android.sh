#!/bin/bash
# Run inference on Android device

set -e

cd "$(dirname "$0")/.."

ADB="${ADB:-adb}"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/LFM2.5_350M_Base_inference}"

# Dtype suffix: NNOPT_DTYPE=fp16 picks <name>_inference_fp16, default fp32 picks
# <name>_inference. The Infer tool sets this env when invoked with dtype:"fp16".
NNOPT_DTYPE="${NNOPT_DTYPE:-fp32}"
case "$NNOPT_DTYPE" in
    fp16) NNOPT_BIN_SUFFIX="_fp16" ;;
    fp32|"") NNOPT_BIN_SUFFIX="" ;;
    *)
        echo "ERROR: NNOPT_DTYPE must be fp32 or fp16 (got '$NNOPT_DTYPE')" >&2
        exit 1
        ;;
esac
BINARY_NAME="LFM2.5_350M_Base_inference${NNOPT_BIN_SUFFIX}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 \"<prompt>\" [max_tokens]"
    exit 1
fi

PROMPT="$1"
MAX_TOKENS="${2:-64}"
shift 2 2>/dev/null || true
EXTRA_ARGS="$*"

echo "Running inference on Android device..." >&2
echo "Prompt: $PROMPT" >&2
if [ -n "$EXTRA_ARGS" ]; then echo "Extra args: $EXTRA_ARGS" >&2; fi
echo "" >&2

# Escape single quotes for nested adb shell
ESCAPED_PROMPT=$(printf '%s' "$PROMPT" | sed "s/'/'\\''/g")

# If --token-ids was passed with a local path, translate it to the device-side
# filename. The file itself is pushed by the Deploy tool (incremental, mtime+size
# diff against .deploy_manifest.json) — we do NOT push it here on every Infer.
# Safety fallback: if the device is missing the file (e.g. user ran Infer without
# Deploy first, or manifest got corrupted), push it once so the binary doesn't
# crash on fopen failure.
TOKEN_IDS_DEVICE_ARGS=""
for arg in $EXTRA_ARGS; do
    if [ "$prev_arg" = "--token-ids" ] && [ -f "$arg" ]; then
        # Check if device already has the file with matching size; push only if missing/different.
        LOCAL_SIZE=$(stat -f%z "$arg" 2>/dev/null || stat -c%s "$arg" 2>/dev/null || echo "0")
        REMOTE_SIZE=$($ADB shell "stat -c%s "$REMOTE_DIR/test_input_ids.bin" 2>/dev/null" 2>/dev/null | tr -d '
')
        if [ -z "$REMOTE_SIZE" ] || [ "$LOCAL_SIZE" != "$REMOTE_SIZE" ]; then
            $ADB push "$arg" "$REMOTE_DIR/test_input_ids.bin" 2>/dev/null
        fi
        TOKEN_IDS_DEVICE_ARGS="--token-ids test_input_ids.bin"
        # Remove --token-ids <path> from EXTRA_ARGS, replaced by device path
        EXTRA_ARGS=$(echo "$EXTRA_ARGS" | sed "s|--token-ids $arg|$TOKEN_IDS_DEVICE_ARGS|g")
        break
    fi
    prev_arg="$arg"
done

# Debug layer validation is OFF by default; set NNOPT_DEBUG_LAYERS=1 to enable
# (per-token Sampler + Generated-token stderr noise — useful for debugging,
# drowns out streamed text otherwise).
DEBUG_ENV=""
if [ -n "${NNOPT_DEBUG_LAYERS:-}" ] && [ "${NNOPT_DEBUG_LAYERS}" != "0" ]; then
  DEBUG_ENV="NNOPT_DEBUG_LAYERS=${NNOPT_DEBUG_LAYERS}"
fi

# Layer dump mode (Infer tool sets NNOPT_DUMP_LAYERS=1 for SxSDebug)
DUMP_ENV=""
if [ "${NNOPT_DUMP_LAYERS:-}" = "1" ]; then DUMP_ENV="NNOPT_DUMP_LAYERS=1"; fi

# Execute on device — let stdout and stderr flow through adb to host.
# adb shell merges remote stdout+stderr into host stdout. The Infer tool
# separates them using pattern matching (CHECKPOINT/ERROR/LAYER_CHECK = stderr).
# No device file redirect — no stale logs/device_run_stderr.log to confuse the agent.
set +e
$ADB shell "cd $REMOTE_DIR && $DEBUG_ENV $DUMP_ENV LD_LIBRARY_PATH=$REMOTE_DIR/lib:/system/vendor/lib64:\$LD_LIBRARY_PATH ./$BINARY_NAME '$ESCAPED_PROMPT' $MAX_TOKENS $EXTRA_ARGS"
INFERENCE_EXIT=$?
set -e

exit $INFERENCE_EXIT
