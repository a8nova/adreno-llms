#!/bin/bash
# Deploy to Android device via ADB

set -e

cd "$(dirname "$0")/.."

# Phase 3F: share REMOTE_DIR + BINARY_NAME with run_android.sh via
# scripts/remote_dir.env. Editing one in isolation can't drift any more.
. ./scripts/remote_dir.env

ADB="${ADB:-adb}"

# NNOPT_DTYPE / NNOPT_BIN_SUFFIX / BINARY_NAME come from remote_dir.env
# (sourced above). Derive deploy-specific paths from those.
case "$NNOPT_DTYPE" in
    fp16) NNOPT_WEIGHTS_BIN="weights/model.fp16.bin"; NNOPT_WEIGHTS_META="weights/model.fp16.meta.json"; NNOPT_BUILD_DIR="build/fp16" ;;
    fp32|"") NNOPT_WEIGHTS_BIN="weights/model.bin"; NNOPT_WEIGHTS_META="weights/model.meta.json"; NNOPT_BUILD_DIR="build" ;;
esac

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

# Phase 3F: atomic deploy. Wipe the volatile subdirs (assets/, kernels/,
# lib/, plus the binary) before pushing fresh. weights/ is preserved
# because the .bin can be hundreds of MB and rarely changes between
# iterations — incremental push of weights/ via mtime-diff still works.
# Closes Kokoro Entry 1.7 finding #7: libclblast.so deploy fragility
# from relying on prior-run residue.
echo "Wiping device-side runtime env (preserves weights/)..."
$ADB shell "rm -rf $REMOTE_DIR/assets $REMOTE_DIR/kernels $REMOTE_DIR/lib $REMOTE_DIR/$BINARY_NAME" || true

# Create directories
$ADB shell "mkdir -p $REMOTE_DIR/weights"
$ADB shell "mkdir -p $REMOTE_DIR/kernels"
$ADB shell "mkdir -p $REMOTE_DIR/lib"

# Push binary
echo "Pushing binary..."
$ADB push "$NNOPT_BUILD_DIR/$BINARY_NAME" $REMOTE_DIR/
$ADB shell "chmod +x $REMOTE_DIR/$BINARY_NAME"

# Push OpenCL library — fail loudly if missing rather than silently
# falling back to the system OpenCL (which doesn't exist on most Android
# devices and produces an opaque "Cannot find libOpenCL.so" load error
# at runtime). Phase 3F finding #7 fix.
OPENCL_LIB="$HOME/.nnopt/deps/opencl/lib/android-arm64-v8a/libOpenCL.so"
if [ -f "$OPENCL_LIB" ]; then
    echo "Pushing OpenCL library..."
    $ADB push "$OPENCL_LIB" $REMOTE_DIR/lib/libOpenCL.so
    echo "  libOpenCL.so deployed to device"
else
    echo "ERROR: libOpenCL.so not found at $OPENCL_LIB" >&2
    echo "  The Android binary needs libOpenCL.so at runtime; the device" >&2
    echo "  rarely ships one. Install via the OpenCL bootstrap (downloads" >&2
    echo "  the prebuilt Adreno OpenCL stub) and re-run deploy:" >&2
    echo "    mkdir -p $HOME/.nnopt/deps/opencl/lib/android-arm64-v8a" >&2
    echo "    # then drop libOpenCL.so into that directory" >&2
    exit 2
fi

# Push CLBlast — the binary is linked against -lclblast and won't load without
# it. CMake FetchContent drops the .so under <build>/_deps/clblast-build/, which
# differs by dtype (build/ vs build/fp16/), so search known locations in order
# and push the first hit. Refuse to deploy silently-missing the file.
CLBLAST_SO=""
for cand in \
    "lib/libclblast.so" \
    "build/libclblast.so" \
    "build/fp16/_deps/clblast-build/libclblast.so" \
    "build/_deps/clblast-build/libclblast.so"; do
    if [ -f "$cand" ]; then CLBLAST_SO="$cand"; break; fi
done
# Last resort: find it anywhere under build/ (newest first).
if [ -z "$CLBLAST_SO" ]; then
    CLBLAST_SO=$(find build -name libclblast.so 2>/dev/null | head -1)
fi
if [ -n "$CLBLAST_SO" ] && [ -f "$CLBLAST_SO" ]; then
    echo "Pushing libclblast.so (from $CLBLAST_SO)..."
    $ADB push "$CLBLAST_SO" $REMOTE_DIR/lib/libclblast.so
else
    echo "ERROR: libclblast.so not found in lib/ or build/ — the binary links" >&2
    echo "  against it and will fail at load. Build first (scripts/build.sh)." >&2
    exit 1
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
        if [ -f "$asset" ]; then
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
