#!/bin/bash
# Deploy Bonsai to an Android device via ADB (any size — one dim-generic binary).
# Convention matches adreno-llms qwen2-5-0-5b/scripts/deploy_android.sh.
# The .nnb is pushed ONCE and skipped when the size matches.

set -e

cd "$(dirname "$0")/.."

ADB="${ADB:-adb}"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/bonsai}"

BONSAI_STORAGE="${BONSAI_STORAGE:-fp32}"
case "$BONSAI_STORAGE" in
    fp32) BIN_SUFFIX="";     BUILD_DIR="build" ;;
    fp16) BIN_SUFFIX="_f16"; BUILD_DIR="build/fp16" ;;
    *) echo "ERROR: BONSAI_STORAGE must be fp32 or fp16 (got '$BONSAI_STORAGE')" >&2; exit 1 ;;
esac
BINARY_NAME="bonsai_inference${BIN_SUFFIX}"

# Which size to deploy. Weights live in weights/ (fetched from HF, gitignored);
# the shared dim-generic runtime reads whichever .nnb is present. Default 8B;
# override BONSAI_NNB=bonsai4b.nnb / bonsai1.7b.nnb for the smaller bundles.
NNB="${BONSAI_NNB:-bonsai8b.nnb}"

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
$ADB shell "chmod +x $REMOTE_DIR/$BINARY_NAME"

echo "==================================="
echo "Deployment complete!"
echo "==================================="
echo "To run: ./scripts/run_android.sh \"<prompt>\" [max_tokens]"
