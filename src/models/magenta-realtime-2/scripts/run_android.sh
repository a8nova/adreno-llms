#!/bin/bash
# Run inference on Android device

set -e

cd "$(dirname "$0")/.."

# Phase 3F: share REMOTE_DIR + BINARY_NAME with deploy_android.sh via
# scripts/remote_dir.env. Both scripts cannot drift apart any more.
. ./scripts/remote_dir.env

ADB="${ADB:-adb}"

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
        REMOTE_SIZE=$($ADB shell "stat -c%s "$REMOTE_DIR/test_input_ids.bin" 2>/dev/null" 2>/dev/null | tr -d '')
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

# Debug layer validation is on by default; set NNOPT_DEBUG_LAYERS=0 to disable
DEBUG_ENV="NNOPT_DEBUG_LAYERS=1"
if [ "${NNOPT_DEBUG_LAYERS:-}" = "0" ]; then DEBUG_ENV=""; fi

# Layer dump mode (Infer tool sets NNOPT_DUMP_LAYERS=1 for SxSDebug)
DUMP_ENV=""
if [ "${NNOPT_DUMP_LAYERS:-}" = "1" ]; then DUMP_ENV="NNOPT_DUMP_LAYERS=1"; fi

# Abort-on-fail-layer (Infer tool sets NNOPT_ABORT_ON_FAIL_LAYER=1 by
# default). When a layer dump is FAIL_NAN / FAIL_INF / FAIL_ZEROS,
# debug_utils.h::nnopt_layer_check_impl std::exit(99)s BEFORE the next
# OpenCL kernel runs. Without this, broken op stubs feed zero buffers
# into downstream kernels (sample_prior / flow / vocoder) which then
# crash the Adreno GPU driver hard (kernel panic, device wedged until
# physical USB reconnect). Default ON because the cost of false-positive
# aborts is one extra build iteration; the cost of a GPU wedge is a
# user-initiated USB reconnect + 60s of inactive device probes.
ABORT_ENV=""
if [ "${NNOPT_ABORT_ON_FAIL_LAYER:-1}" != "0" ]; then ABORT_ENV="NNOPT_ABORT_ON_FAIL_LAYER=${NNOPT_ABORT_ON_FAIL_LAYER:-1}"; fi

# Execute on device — let stdout and stderr flow through adb to host.
# adb shell merges remote stdout+stderr into host stdout. The Infer tool
# separates them using pattern matching (CHECKPOINT/ERROR/LAYER_CHECK = stderr).
# No device file redirect — no stale logs/device_run_stderr.log to confuse the agent.
set +e
$ADB shell "cd $REMOTE_DIR && $DEBUG_ENV $DUMP_ENV $ABORT_ENV LD_LIBRARY_PATH=$REMOTE_DIR/lib:/system/vendor/lib64:\$LD_LIBRARY_PATH ./$BINARY_NAME '$ESCAPED_PROMPT' $MAX_TOKENS $EXTRA_ARGS"
INFERENCE_EXIT=$?
set -e

exit $INFERENCE_EXIT
