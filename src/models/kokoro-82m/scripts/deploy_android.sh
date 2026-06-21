#!/bin/bash
# Deploy to Android device via ADB

set -e

cd "$(dirname "$0")/.."

ADB="${ADB:-adb}"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/Kokoro_82M_inference}"

# Dtype: NNOPT_DTYPE=fp16 picks the fp16 binary + fp16 weights, fp32 default.
NNOPT_DTYPE="${NNOPT_DTYPE:-fp16}"  # fp16 = the optimized binary; fastest is the default
case "$NNOPT_DTYPE" in
    fp16) NNOPT_BIN_SUFFIX="_fp16"; NNOPT_WEIGHTS_BIN="weights/model.fp16.bin"; NNOPT_WEIGHTS_META="weights/model.fp16.meta.json"; NNOPT_BUILD_DIR="build/fp16" ;;
    fp32|"") NNOPT_BIN_SUFFIX=""; NNOPT_WEIGHTS_BIN="weights/model.bin"; NNOPT_WEIGHTS_META="weights/model.meta.json"; NNOPT_BUILD_DIR="build" ;;
    *)
        echo "ERROR: NNOPT_DTYPE must be fp32 or fp16 (got '$NNOPT_DTYPE')" >&2
        exit 1
        ;;
esac
BINARY_NAME="Kokoro_82M_inference${NNOPT_BIN_SUFFIX}"

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
# kernel_cache: keyed by hash(source + options + device_name), so cache entries
# self-invalidate when kernel source changes. Preserved across deploys to keep
# warm-start benefit (~0.7s saved per invocation on Adreno 620).
$ADB shell "mkdir -p $REMOTE_DIR/kernel_cache"

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

# Push CLBlast shared library if we linked dynamically.
# Even if we *intend* to link CLBlast statically, some toolchains still end up
# producing a DT_NEEDED on libclblast.so depending on how CLBlast was built.
# Deploy it to avoid 'library "libclblast.so" not found' at runtime.
CLBLAST_SO=""
if [ -f "$NNOPT_BUILD_DIR/libclblast.so" ]; then
    CLBLAST_SO="$NNOPT_BUILD_DIR/libclblast.so"
elif [ -f "$NNOPT_BUILD_DIR/lib/libclblast.so" ]; then
    CLBLAST_SO="$NNOPT_BUILD_DIR/lib/libclblast.so"
elif [ -f "$NNOPT_BUILD_DIR/_deps/clblast-build/libclblast.so" ]; then
    CLBLAST_SO="$NNOPT_BUILD_DIR/_deps/clblast-build/libclblast.so"
elif [ -f "build/_deps/clblast-build/libclblast.so" ]; then
    CLBLAST_SO="build/_deps/clblast-build/libclblast.so"
elif [ -f "build/fp16/_deps/clblast-build/libclblast.so" ]; then
    CLBLAST_SO="build/fp16/_deps/clblast-build/libclblast.so"
fi

if [ -n "$CLBLAST_SO" ]; then
    echo "Pushing CLBlast runtime ($CLBLAST_SO)..."
    $ADB push "$CLBLAST_SO" $REMOTE_DIR/lib/libclblast.so
    echo "  libclblast.so deployed to device"
else
    echo "Note: libclblast.so not found in build outputs; assuming CLBlast was statically linked"
fi

# Push weights matching the dtype (model.bin OR model.fp16.bin — never both).
echo "Pushing weights ($NNOPT_WEIGHTS_BIN)..."
$ADB push "$NNOPT_WEIGHTS_BIN" $REMOTE_DIR/weights/ 2>/dev/null || echo "Warning: $NNOPT_WEIGHTS_BIN not found"
$ADB push "$NNOPT_WEIGHTS_META" $REMOTE_DIR/weights/ 2>/dev/null || echo "Warning: $NNOPT_WEIGHTS_META not found"
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

# Push runtime input fixtures (TTS/ASR/VLM modalities publish
# assets/{test_input_ids,duration_noise,prior_noise,test_audio,test_image}.bin
# via GenerateReference; main.cpp reads them on-device from REMOTE_DIR/assets/).
# Without this push, on-device code fails with "required <modality> fixture
# missing under assets/ (did GenerateReference run?)".
if [ -d "assets" ]; then
    $ADB shell "mkdir -p $REMOTE_DIR/assets"
    echo "Pushing input fixtures (assets/)..."
    for asset in assets/*; do
        # Push files AND directories (e.g. espeak-ng-data/ for --stream G2P).
        # `adb push <dir>` is recursive; the old file-only guard silently
        # dropped the espeak data dir.
        if [ -e "$asset" ]; then
            $ADB push "$asset" $REMOTE_DIR/assets/
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
