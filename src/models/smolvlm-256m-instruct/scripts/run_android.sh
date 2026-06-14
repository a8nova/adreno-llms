#!/bin/bash
# Run SmolVLM-256M inference on the Android device.
#
# Two modes:
#
#   1) One-shot:
#        ./scripts/run_android.sh "<prompt>" [max_tokens] [--image <path>] [--token-ids <path>]
#      Examples:
#        ./scripts/run_android.sh "Describe the image." 64 --image fixtures/qcheck/01_cafe_patio.jpg
#        ./scripts/run_android.sh "Hello" 32
#
#   2) Interactive REPL (multi-turn, KV cache reuses across turns):
#        ./scripts/run_android.sh --interactive
#      Inside the REPL:
#        /image <path>         load an image (host-side path auto-uploads to the device)
#        /reset                drop image + start a fresh conversation
#        /quit                 exit (Ctrl-D / blank line also work)
#        <any text>            follow-up question to the current image
#
# Image paths in BOTH modes:
#   - If the path exists on your laptop, it's auto-pushed to
#     $REMOTE_DIR/images_uploaded/<basename> and the argument is rewritten.
#   - Otherwise it's forwarded as-is (treated as already deployed under
#     $REMOTE_DIR, e.g. fixtures/qcheck/01_cafe_patio.jpg).
#
# Token streaming: stdout/stderr from the device flow directly to your terminal
# — no host-side pipes — so tokens print as they decode. The binary flushes
# after every token.
#
# Env vars:
#   DEVICE=<serial>          select device (defaults to adb get-serialno)
#   NNOPT_DTYPE=fp16|fp32    binary suffix (default fp32; demo uses fp16)
#   NNOPT_DEBUG_LAYERS=0|1   LAYER_CHECK gating (default 1 from this wrapper)
#   NNOPT_PROFILE=1          enable per-stage profile output
#   NNOPT_DUMP_LAYERS=1      dump per-layer outputs for SxSDebug
#   REMOTE_DIR=<path>        on-device deploy root (default /data/local/tmp/SmolVLM_…)

set -e

cd "$(dirname "$0")/.."

ADB="${ADB:-adb}"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/SmolVLM_256M_Instruct_inference}"
REMOTE_UPLOAD_DIR="images_uploaded"

NNOPT_DTYPE="${NNOPT_DTYPE:-fp16}"
case "$NNOPT_DTYPE" in
    fp16) NNOPT_BIN_SUFFIX="_fp16" ;;
    fp32|"") NNOPT_BIN_SUFFIX="" ;;
    *)
        echo "ERROR: NNOPT_DTYPE must be fp32 or fp16 (got '$NNOPT_DTYPE')" >&2
        exit 1
        ;;
esac
BINARY_NAME="SmolVLM_256M_Instruct_inference${NNOPT_BIN_SUFFIX}"

# ── Env vars forwarded to the on-device binary ────────────────────────────
DEBUG_ENV="NNOPT_DEBUG_LAYERS=1"
if [ "${NNOPT_DEBUG_LAYERS:-}" = "0" ]; then DEBUG_ENV="NNOPT_DEBUG_LAYERS=0"; fi
DUMP_ENV=""
if [ "${NNOPT_DUMP_LAYERS:-}" = "1" ]; then DUMP_ENV="NNOPT_DUMP_LAYERS=1"; fi
PROFILE_ENV=""
if [ "${NNOPT_PROFILE:-}" = "1" ]; then PROFILE_ENV="NNOPT_PROFILE=1"; fi
VTIME_ENV=""
if [ "${NNOPT_VISION_TIMING:-}" = "1" ]; then VTIME_ENV="NNOPT_VISION_TIMING=1"; fi

DEVICE_FLAG=""
if [ -n "${DEVICE:-}" ]; then DEVICE_FLAG="-s $DEVICE"; fi

# Ensure upload dir exists on device.
$ADB $DEVICE_FLAG shell "mkdir -p $REMOTE_DIR/$REMOTE_UPLOAD_DIR" </dev/null >/dev/null 2>&1 || true

# Helper: if $1 is a local file, push to device and echo the device-side path;
# else echo $1 unchanged (assume already-deployed device-relative path).
push_if_local() {
    local p="$1"
    if [ -f "$p" ]; then
        local base
        base="$(basename "$p")"
        if $ADB $DEVICE_FLAG push "$p" "$REMOTE_DIR/$REMOTE_UPLOAD_DIR/$base" >/dev/null 2>&1; then
            printf '%s/%s' "$REMOTE_UPLOAD_DIR" "$base"
            return 0
        fi
        echo "ERROR: adb push failed for $p" >&2
        return 1
    fi
    printf '%s' "$p"
}

# ── Parse args: --interactive switches modes ──────────────────────────────
INTERACTIVE=0
for a in "$@"; do
    if [ "$a" = "--interactive" ]; then INTERACTIVE=1; fi
done

# ────────────────────────────────────────────────────────────────────────
# INTERACTIVE MODE — REPL with auto-upload + raw streaming.
# ────────────────────────────────────────────────────────────────────────
if [ $INTERACTIVE -eq 1 ]; then
    if ! $ADB $DEVICE_FLAG shell "[ -x $REMOTE_DIR/$BINARY_NAME ]" </dev/null 2>/dev/null; then
        cat >&2 <<EOF
ERROR: $REMOTE_DIR/$BINARY_NAME missing on device.
Build + deploy first:
    NNOPT_DTYPE=$NNOPT_DTYPE ./scripts/build.sh --release
    NNOPT_DTYPE=$NNOPT_DTYPE ./scripts/deploy_android.sh
EOF
        exit 2
    fi

    echo "run_android.sh: interactive mode (NNOPT_DTYPE=$NNOPT_DTYPE)" >&2
    echo "  /image <path>     load image (local path auto-uploads)" >&2
    echo "  /reset            clear conversation" >&2
    echo "  /quit             exit" >&2
    echo >&2

    FIFO="$(mktemp -u /tmp/rai_in.XXXXXX)"
    mkfifo "$FIFO"
    cleanup() {
        if [ -n "${KEEPALIVE_FD:-}" ]; then eval "exec ${KEEPALIVE_FD}>&-"; fi
        wait 2>/dev/null || true
        rm -f "$FIFO"
    }
    trap cleanup EXIT INT TERM

    # adb shell stdout/stderr stream DIRECTLY to the user's terminal — no
    # pipe-through, so each flushed token appears immediately.
    $ADB $DEVICE_FLAG shell "cd $REMOTE_DIR && $DEBUG_ENV $DUMP_ENV $PROFILE_ENV $VTIME_ENV LD_LIBRARY_PATH=$REMOTE_DIR/lib:/system/vendor/lib64:\$LD_LIBRARY_PATH ./$BINARY_NAME --interactive 2>&1" < "$FIFO" &
    ADB_PID=$!

    exec 9>"$FIFO"
    KEEPALIVE_FD=9

    while IFS= read -r line; do
        case "$line" in
            "/image "*)
                arg="${line#/image }"
                arg="${arg# }"
                if [ -z "$arg" ]; then
                    echo "(usage: /image <path>)" >&2
                    continue
                fi
                dev_path="$(push_if_local "$arg")"
                if [ -z "$dev_path" ]; then continue; fi
                printf '/image %s\n' "$dev_path" >&9
                ;;
            /quit|/exit)
                printf '/quit\n' >&9
                break
                ;;
            "")
                printf '\n' >&9
                break
                ;;
            *)
                printf '%s\n' "$line" >&9
                ;;
        esac
    done

    exec 9>&-
    KEEPALIVE_FD=""
    wait "$ADB_PID" 2>/dev/null || true
    exit 0
fi

# ────────────────────────────────────────────────────────────────────────
# ONE-SHOT MODE — existing path, but with --image auto-upload.
# ────────────────────────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    sed -n '2,30p' "$0" >&2
    exit 1
fi

PROMPT="$1"
MAX_TOKENS="${2:-64}"
shift 2 2>/dev/null || true
EXTRA_ARGS="$*"

# Auto-upload --image <local_path> → rewrite to device-relative path.
# (Same logic as interactive's /image, applied to the CLI flag form.)
NEW_EXTRA=""
prev=""
for arg in $EXTRA_ARGS; do
    if [ "$prev" = "--image" ] && [ -f "$arg" ]; then
        dev_path="$(push_if_local "$arg")"
        if [ -n "$dev_path" ]; then
            NEW_EXTRA="$NEW_EXTRA $dev_path"
        else
            echo "ERROR: failed to upload $arg" >&2
            exit 1
        fi
    else
        NEW_EXTRA="$NEW_EXTRA $arg"
    fi
    prev="$arg"
done
EXTRA_ARGS="${NEW_EXTRA# }"

echo "Running inference on Android device..." >&2
echo "Prompt: $PROMPT" >&2
if [ -n "$EXTRA_ARGS" ]; then echo "Extra args: $EXTRA_ARGS" >&2; fi
echo "" >&2

# Escape for the REMOTE shell's double-quote context. Inside "..." only
# `"`, `\`, `$`, and backtick are special. This handles apostrophes
# transparently (the old single-quote wrapping broke on prompts like "What's").
ESCAPED_PROMPT=$(printf '%s' "$PROMPT" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\$/\\$/g' -e 's/`/\\`/g')

# --token-ids handling: push local file under a stable device name.
TOKEN_IDS_DEVICE_ARGS=""
for arg in $EXTRA_ARGS; do
    if [ "$prev_arg" = "--token-ids" ] && [ -f "$arg" ]; then
        LOCAL_SIZE=$(stat -f%z "$arg" 2>/dev/null || stat -c%s "$arg" 2>/dev/null || echo "0")
        REMOTE_SIZE=$($ADB $DEVICE_FLAG shell "stat -c%s $REMOTE_DIR/test_input_ids.bin 2>/dev/null" 2>/dev/null | tr -d '\r')
        if [ -z "$REMOTE_SIZE" ] || [ "$LOCAL_SIZE" != "$REMOTE_SIZE" ]; then
            $ADB $DEVICE_FLAG push "$arg" "$REMOTE_DIR/test_input_ids.bin" 2>/dev/null
        fi
        TOKEN_IDS_DEVICE_ARGS="--token-ids test_input_ids.bin"
        EXTRA_ARGS=$(echo "$EXTRA_ARGS" | sed "s|--token-ids $arg|$TOKEN_IDS_DEVICE_ARGS|g")
        break
    fi
    prev_arg="$arg"
done

# Stream directly — no pipe → tokens appear as they decode.
set +e
$ADB $DEVICE_FLAG shell "cd $REMOTE_DIR && $DEBUG_ENV $DUMP_ENV $PROFILE_ENV $VTIME_ENV LD_LIBRARY_PATH=$REMOTE_DIR/lib:/system/vendor/lib64:\$LD_LIBRARY_PATH ./$BINARY_NAME \"$ESCAPED_PROMPT\" $MAX_TOKENS $EXTRA_ARGS"
INFERENCE_EXIT=$?
set -e

exit $INFERENCE_EXIT
