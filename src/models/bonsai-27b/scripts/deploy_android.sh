#!/bin/bash
# Deploy Bonsai-27B to an Android device via ADB.
# The .nnb is pushed ONCE and skipped when the size already matches.

set -e

cd "$(dirname "$0")/.."

ADB="${ADB:-adb}"
# NOT /data/local/tmp/bonsai — that is the 4B/8B port's directory, and sharing it
# means whichever deployed last wins (different architecture, different binary).
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/bonsai27b}"

BONSAI_STORAGE="${BONSAI_STORAGE:-fp32}"
case "$BONSAI_STORAGE" in
    fp32) BIN_SUFFIX="";     BUILD_DIR="build" ;;
    fp16) BIN_SUFFIX="_f16"; BUILD_DIR="build/fp16" ;;
    *) echo "ERROR: BONSAI_STORAGE must be fp32 or fp16 (got '$BONSAI_STORAGE')" >&2; exit 1 ;;
esac
BINARY_NAME="bonsai27b_inference${BIN_SUFFIX}"

# Weights live in weights/ (fetched from HF, gitignored). There is exactly one
# 27B .nnb — the 4B/8B names this script used to default to belong to the other
# port and would have deployed nothing this runtime can load.
NNB="${BONSAI_NNB:-bonsai27b.nnb}"
VISION_NNB="${BONSAI_VISION_NNB:-bonsai27b-vision.nnb}"

if ! command -v $ADB &> /dev/null; then
    echo "ERROR: adb not found in PATH"
    exit 1
fi
DEVICES=$($ADB devices | grep -v "List of devices" | grep -c "device$" || true)
if [ "$DEVICES" -eq 0 ]; then
    echo "ERROR: No Android device connected"
    exit 1
fi

echo "Deploying $BINARY_NAME to $REMOTE_DIR ..."
$ADB shell "mkdir -p $REMOTE_DIR/kernels $REMOTE_DIR/model"
$ADB push "$BUILD_DIR/$BINARY_NAME" "$REMOTE_DIR/" >/dev/null
for f in kernels/*.cl; do $ADB push "$f" "$REMOTE_DIR/kernels/" >/dev/null; done
$ADB push weights/tokenizer.json "$REMOTE_DIR/model/" >/dev/null

# model .nnb: push once; skip when remote size matches (1.15 GB 8B / 574 MB 4B /
# 242 MB 1.7B)
LOCAL_SIZE=$(stat -f%z weights/$NNB 2>/dev/null || stat -c%s weights/$NNB)
REMOTE_SIZE=$($ADB shell "stat -c%s $REMOTE_DIR/model/$NNB 2>/dev/null" | tr -d '\r' || true)
if [ "$LOCAL_SIZE" != "$REMOTE_SIZE" ]; then
    echo "Pushing model ($NNB, one-time)..."
    $ADB push weights/$NNB "$REMOTE_DIR/model/"
else
    echo "Model unchanged on device — skipped."
fi
# CLBlast — the vision tower's GEMMs are dynamically linked against it, so a
# vision build that does not find it on the device dies at launch with
# 'library "libclblast.so" not found'. CMake FetchContent builds it under the
# model's build dir; a text-only build simply has none to push.
CLBLAST_LIB=$(find "$BUILD_DIR" -name libclblast.so 2>/dev/null | head -1)
if [ -n "$CLBLAST_LIB" ]; then
    $ADB shell "mkdir -p $REMOTE_DIR/lib"
    $ADB push "$CLBLAST_LIB" "$REMOTE_DIR/lib/libclblast.so" >/dev/null
    echo "  libclblast.so deployed"
fi

# Vision tower, when it was built and fetched. Optional: the engine loads it
# lazily on the first image, so a text-only session never needs it.
if [ -f "weights/$VISION_NNB" ]; then
    V_LOCAL=$(stat -f%z "weights/$VISION_NNB" 2>/dev/null || stat -c%s "weights/$VISION_NNB")
    V_REMOTE=$($ADB shell "stat -c%s $REMOTE_DIR/model/$VISION_NNB 2>/dev/null" | tr -d '\r' || true)
    if [ "$V_LOCAL" != "$V_REMOTE" ]; then
        echo "Pushing $VISION_NNB ($((V_LOCAL / 1048576)) MB) ..."
        $ADB push "weights/$VISION_NNB" "$REMOTE_DIR/model/"
    fi
fi

$ADB shell "chmod +x $REMOTE_DIR/$BINARY_NAME"

echo "==================================="
echo "Deployment complete!"
echo "==================================="
echo "To run: ./scripts/run_android.sh \"<prompt>\" [max_tokens]"
