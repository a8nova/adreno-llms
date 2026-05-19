#!/bin/bash
# Deploy to Android device via ADB

set -e

cd "$(dirname "$0")/.."

ADB="${ADB:-adb}"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/mms_tts_eng_inference}"

# Dtype: NNOPT_DTYPE=fp16 picks the fp16 binary + fp16 weights, fp32 default.
NNOPT_DTYPE="${NNOPT_DTYPE:-fp32}"
case "$NNOPT_DTYPE" in
    fp16) NNOPT_BIN_SUFFIX="_fp16"; NNOPT_WEIGHTS_BIN="weights/model.fp16.bin"; NNOPT_WEIGHTS_META="weights/model.fp16.meta.json"; NNOPT_BUILD_DIR="build/fp16" ;;
    fp32|"") NNOPT_BIN_SUFFIX=""; NNOPT_WEIGHTS_BIN="weights/model.bin"; NNOPT_WEIGHTS_META="weights/model.meta.json"; NNOPT_BUILD_DIR="build" ;;
    *)
        echo "ERROR: NNOPT_DTYPE must be fp32 or fp16 (got '$NNOPT_DTYPE')" >&2
        exit 1
        ;;
esac
BINARY_NAME="mms_tts_eng_inference${NNOPT_BIN_SUFFIX}"

# Check if ADB is available
if ! command -v $ADB &> /dev/null; then
    echo "ERROR: adb not found in PATH"
    echo "Install Android SDK platform-tools"
    exit 1
fi

# Check for connected device
DEVICES=$($ADB devices | grep -v "List of devices" | grep "device$" | wc -l)
if [ "$DEVICES" -eq 0 ]; then
    echo "ERROR: No Android device connected"
    echo "Connect device and enable USB debugging"
    exit 1
fi

echo "Deploying $NNOPT_DTYPE binary $BINARY_NAME to Android device..."
echo "Remote directory: $REMOTE_DIR"

# Create directories
$ADB shell "mkdir -p $REMOTE_DIR/weights"
$ADB shell "mkdir -p $REMOTE_DIR/kernels"
$ADB shell "mkdir -p $REMOTE_DIR/lib"

# Push binary
echo "Pushing binary..."
$ADB push "$NNOPT_BUILD_DIR/$BINARY_NAME" $REMOTE_DIR/
$ADB shell "chmod +x $REMOTE_DIR/$BINARY_NAME"

# Push OpenCL library if available
OPENCL_LIB="$HOME/.nnopt/deps/opencl/lib/android-arm64-v8a/libOpenCL.so"
if [ -f "$OPENCL_LIB" ]; then
    echo "Pushing OpenCL library..."
    $ADB push "$OPENCL_LIB" $REMOTE_DIR/lib/libOpenCL.so
    echo "  libOpenCL.so deployed to device"
else
    echo "Note: Using system OpenCL library (no cached lib found)"
fi

# Push weights matching the dtype (model.bin OR model.fp16.bin — never both).
echo "Pushing weights ($NNOPT_WEIGHTS_BIN)..."
$ADB push "$NNOPT_WEIGHTS_BIN" $REMOTE_DIR/weights/ 2>/dev/null || echo "Warning: $NNOPT_WEIGHTS_BIN not found"
$ADB push "$NNOPT_WEIGHTS_META" $REMOTE_DIR/weights/ 2>/dev/null || echo "Warning: $NNOPT_WEIGHTS_META not found"
$ADB push weights/tokenizer_vocab.bin $REMOTE_DIR/weights/ 2>/dev/null || true

# Push reference dumps (needed by backbone.cpp fixture loading)
if [ -d "reference" ]; then
    echo "Pushing reference/ directory..."
    $ADB shell "mkdir -p $REMOTE_DIR/reference"
    $ADB push reference $REMOTE_DIR/ >/dev/null
fi

# Push all kernels
echo "Pushing OpenCL kernels..."
if [ -d "kernels" ]; then
    for kernel in kernels/*.cl; do
        if [ -f "$kernel" ]; then
            $ADB push "$kernel" $REMOTE_DIR/kernels/
        fi
    done
fi

echo ""
echo "==================================="
echo "Deployment complete!"
echo "==================================="
echo ""
echo "To run on device:"
echo "  ./scripts/run_android.sh \"Hello I am a language model\" 10"
