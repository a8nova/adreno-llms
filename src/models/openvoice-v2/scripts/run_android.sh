#!/bin/bash
# Run the OpenVoice V2 converter on a connected Android device.
#
# Stage-mode runner (mirrors the seamless-m4t cascade runner): argv[1] selects a
# stage, default "clone" (the fused enc_q -> flow -> dec path). Other modes
# (enc_q_wn / flow / dec / decfast / all / bench / hwprobe) are per-stage
# cosine-verification harnesses.
#
#   ./scripts/run_android.sh [mode]      # mode defaults to "clone"

set -e

cd "$(dirname "$0")/.."

# Share REMOTE_DIR + BINARY_NAME with deploy_android.sh via remote_dir.env so
# the two scripts cannot drift apart.
. ./scripts/remote_dir.env

ADB="${ADB:-adb}"
MODE="${1:-clone}"

echo "Running OpenVoice converter on device (mode=$MODE)..." >&2
echo "Remote dir: $REMOTE_DIR  binary: $BINARY_NAME" >&2

# The clone / per-stage verification modes read their enc_q inputs from cached
# reference tensors (reference/layers/*.bin) rather than computing the content
# encoder from raw audio — these are dev-time fixtures and are gitignored, so
# they are NOT present on a clean checkout. Push whatever is available; warn if
# a stage's input is missing so the failure is legible instead of a bare crash.
$ADB shell "mkdir -p $REMOTE_DIR/reference/layers"
for ref in enc_q_pre_output enc_q_output flow_output; do
    if [ -f "reference/layers/$ref.bin" ]; then
        $ADB push "reference/layers/$ref.bin" "$REMOTE_DIR/reference/layers/$ref.bin" >/dev/null
    else
        echo "  note: reference/layers/$ref.bin missing locally (regenerate with the cos harness)" >&2
    fi
done

# Per-stage debug/abort env toggles, useful for the verification modes.
DEBUG_ENV="NNOPT_DEBUG_LAYERS=1"
if [ "${NNOPT_DEBUG_LAYERS:-}" = "0" ]; then DEBUG_ENV=""; fi
DUMP_ENV=""
if [ "${NNOPT_DUMP_LAYERS:-}" = "1" ]; then DUMP_ENV="NNOPT_DUMP_LAYERS=1"; fi
ABORT_ENV=""
if [ "${NNOPT_ABORT_ON_FAIL_LAYER:-1}" != "0" ]; then ABORT_ENV="NNOPT_ABORT_ON_FAIL_LAYER=${NNOPT_ABORT_ON_FAIL_LAYER:-1}"; fi

# Execute on device — stdout/stderr flow through adb to the host.
# NOTE: /vendor/lib64 MUST come before $REMOTE_DIR/lib. The libOpenCL.so we
# deploy (from ~/.nnopt/deps) is a Khronos ICD *loader stub* that enumerates 0
# vendors on this device; if it shadows the real Adreno driver the engine's
# clGetPlatformIDs returns no platforms and run_clone aborts with "no cl".
# Putting the device's vendor driver first makes OpenCL init succeed.
set +e
$ADB shell "cd $REMOTE_DIR && $DEBUG_ENV $DUMP_ENV $ABORT_ENV LD_LIBRARY_PATH=/vendor/lib64:$REMOTE_DIR/lib:/system/vendor/lib64:\$LD_LIBRARY_PATH ./$BINARY_NAME $MODE"
INFERENCE_EXIT=$?
set -e

exit $INFERENCE_EXIT
