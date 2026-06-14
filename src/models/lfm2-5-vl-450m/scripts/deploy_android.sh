#!/bin/bash
# Deploy to Android device via ADB

set -e

cd "$(dirname "$0")/.."

ADB="${ADB:-adb}"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/LFM2.5_VL_450M_inference}"

# Dtype: NNOPT_DTYPE=fp16 (the default) picks the fp16 binary + fp16 weights.
NNOPT_DTYPE="${NNOPT_DTYPE:-fp16}"
case "$NNOPT_DTYPE" in
    fp16) NNOPT_BIN_SUFFIX="_fp16"; NNOPT_WEIGHTS_BIN="weights/model.fp16.bin"; NNOPT_WEIGHTS_META="weights/model.fp16.meta.json"; NNOPT_BUILD_DIR="build/fp16" ;;
    fp32|"") NNOPT_BIN_SUFFIX=""; NNOPT_WEIGHTS_BIN="weights/model.bin"; NNOPT_WEIGHTS_META="weights/model.meta.json"; NNOPT_BUILD_DIR="build" ;;
    *)
        echo "ERROR: NNOPT_DTYPE must be fp32 or fp16 (got '$NNOPT_DTYPE')" >&2
        exit 1
        ;;
esac
BINARY_NAME="LFM2.5_VL_450M_inference${NNOPT_BIN_SUFFIX}"

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

# Push CLBlast — the binary is dynamically linked against libclblast.so (it
# handles the M>1 prefill GEMMs), so without it on the device the executable
# fails to link at launch: 'library "libclblast.so" not found'. CMake's
# FetchContent builds it under the model's build dir.
CLBLAST_LIB="$NNOPT_BUILD_DIR/_deps/clblast-build/libclblast.so"
if [ -f "$CLBLAST_LIB" ]; then
    echo "Pushing CLBlast library..."
    $ADB push "$CLBLAST_LIB" $REMOTE_DIR/lib/libclblast.so
    echo "  libclblast.so deployed to device"
else
    echo "Warning: $CLBLAST_LIB not found — the binary will fail to link at launch."
    echo "         Build first: NNOPT_DTYPE=$NNOPT_DTYPE ./scripts/build.sh --release"
fi

# Push weights matching the dtype. Under fp16, prefer the int8 quantized
# bundle (weights/model.int8.bin + .meta.json — per-row int8 with fp16 scales,
# ~50% smaller and ~2x decode-bandwidth) when present. The runtime probes the
# file path order in src/main.cpp; we just need to push whichever exists.
echo "Pushing weights ($NNOPT_WEIGHTS_BIN)..."
$ADB push "$NNOPT_WEIGHTS_BIN" $REMOTE_DIR/weights/ 2>/dev/null || echo "Warning: $NNOPT_WEIGHTS_BIN not found"
$ADB push "$NNOPT_WEIGHTS_META" $REMOTE_DIR/weights/ 2>/dev/null || echo "Warning: $NNOPT_WEIGHTS_META not found"
if [ "$NNOPT_DTYPE" = "fp16" ] && [ -f "weights/model.int8.bin" ] && [ -f "weights/model.int8.meta.json" ]; then
    echo "Pushing int8 weights (model.int8.bin + model.int8.meta.json)..."
    $ADB push weights/model.int8.bin $REMOTE_DIR/weights/
    $ADB push weights/model.int8.meta.json $REMOTE_DIR/weights/
fi
$ADB push weights/tokenizer_vocab.bin $REMOTE_DIR/weights/ 2>/dev/null || true

# Push all kernels
echo "Pushing OpenCL kernels..."
if [ -d "kernels" ]; then
    for kernel in kernels/*.cl; do
        if [ -f "$kernel" ]; then
            $ADB push "$kernel" $REMOTE_DIR/kernels/
        fi
    done
fi

# Push fixture image(s). run_android.sh / the README quickstart reference them
# by device-relative path (e.g. --image fixtures/sample.jpg), so they must exist
# on-device — otherwise the run fails opening the image on a clean deploy.
if [ -d "fixtures" ]; then
    echo "Pushing fixtures..."
    $ADB shell "mkdir -p $REMOTE_DIR/fixtures"
    for fx in fixtures/*; do
        if [ -f "$fx" ]; then
            $ADB push "$fx" $REMOTE_DIR/fixtures/
        fi
    done
fi

# Diagnostic: push reference/inputs_embeds.bin if present. The binary reads
# it and substitutes its rows for the on-device vision pipeline output,
# isolating LM-stack correctness from vision-pipeline correctness.
if [ -f "reference/inputs_embeds.bin" ]; then
    echo "Pushing reference/inputs_embeds.bin (LM-stack isolation diagnostic)..."
    $ADB shell "mkdir -p $REMOTE_DIR/reference"
    $ADB push reference/inputs_embeds.bin $REMOTE_DIR/reference/
fi

echo ""
echo "==================================="
echo "Deployment complete!"
echo "==================================="
echo ""
echo "To run on device:"
echo "  ./scripts/run_android.sh \"Hello I am a language model\" 10"
