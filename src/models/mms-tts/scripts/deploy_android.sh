#!/bin/bash
# Deploy to Android device via ADB.
#
# MMS-TTS is multi-language: one binary, per-language weight packs under
# weights/<lang>/ and per-language asset packs under assets/<lang>/. This
# script pushes:
#   - the binary + kernels + libOpenCL.so link stub
#   - every weights/<lang>/ subdir that prep_lang.py has populated
#   - every assets/<lang>/ subdir (noise fixtures, test ids)
#   - assets/uroman/ if present (needed for non-Latin scripts; see
#     scripts/fetch_uroman.sh)
#   - reference/ (fixtures for NNOPT_REF_TEST)

set -e

cd "$(dirname "$0")/.."

ADB="${ADB:-adb}"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/mms_tts_eng_inference}"

# Dtype: NNOPT_DTYPE=fp16 (the default) picks the fp16 binary + fp16 weights.
NNOPT_DTYPE="${NNOPT_DTYPE:-fp16}"
case "$NNOPT_DTYPE" in
    fp16) NNOPT_BIN_SUFFIX="_fp16"; NNOPT_WEIGHTS_BIN="model.fp16.bin"; NNOPT_WEIGHTS_META="model.fp16.meta.json"; NNOPT_BUILD_DIR="build/fp16" ;;
    fp32|"") NNOPT_BIN_SUFFIX=""; NNOPT_WEIGHTS_BIN="model.bin"; NNOPT_WEIGHTS_META="model.meta.json"; NNOPT_BUILD_DIR="build" ;;
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

# Create base directories
$ADB shell "mkdir -p $REMOTE_DIR/weights"
$ADB shell "mkdir -p $REMOTE_DIR/assets"
$ADB shell "mkdir -p $REMOTE_DIR/kernels"
$ADB shell "mkdir -p $REMOTE_DIR/lib"

# Push binary
echo "Pushing binary..."
$ADB push "$NNOPT_BUILD_DIR/$BINARY_NAME" $REMOTE_DIR/
$ADB shell "chmod +x $REMOTE_DIR/$BINARY_NAME"

# Push OpenCL library if available
OPENCL_LIB="$HOME/.cache/adreno-llms/opencl/lib/android-arm64-v8a/libOpenCL.so"
if [ -f "$OPENCL_LIB" ]; then
    echo "Pushing OpenCL library..."
    $ADB push "$OPENCL_LIB" $REMOTE_DIR/lib/libOpenCL.so
    echo "  libOpenCL.so deployed to device"
else
    echo "Note: Using system OpenCL library (no cached lib found)"
fi

# Push libclblast.so to keep its libc++ ABI in sync with the binary. A stale
# device-side libclblast.so (older NDK) causes the dynamic linker to fail with
# `cannot locate symbol _ZNKSt6__ndk1...` at process start.
CLBLAST_LIB="$NNOPT_BUILD_DIR/_deps/clblast-build/libclblast.so"
if [ -f "$CLBLAST_LIB" ]; then
    echo "Pushing libclblast.so..."
    $ADB push "$CLBLAST_LIB" $REMOTE_DIR/lib/libclblast.so
else
    echo "Warning: $CLBLAST_LIB not found — device may load a stale libclblast.so"
fi

# Push per-language weight packs. prep_lang.py writes weights/<lang>/{model.fp16.bin,
# model.fp16.meta.json, tokenizer_vocab.bin}. The on-device binary's --lang flag
# selects which subdir to load.
PUSHED_LANGS=()
if [ -d "weights" ]; then
    for lang_dir in weights/*/; do
        [ -d "$lang_dir" ] || continue
        lang=$(basename "$lang_dir")
        if [ ! -f "$lang_dir/$NNOPT_WEIGHTS_BIN" ]; then
            echo "  skipping $lang (no $NNOPT_WEIGHTS_BIN)"
            continue
        fi
        echo "Pushing weights/$lang/ ..."
        $ADB shell "mkdir -p $REMOTE_DIR/weights/$lang"
        $ADB push "$lang_dir/$NNOPT_WEIGHTS_BIN"  $REMOTE_DIR/weights/$lang/
        $ADB push "$lang_dir/$NNOPT_WEIGHTS_META" $REMOTE_DIR/weights/$lang/ 2>/dev/null || true
        $ADB push "$lang_dir/tokenizer_vocab.bin" $REMOTE_DIR/weights/$lang/ 2>/dev/null || true
        PUSHED_LANGS+=("$lang")
    done
fi
if [ "${#PUSHED_LANGS[@]}" -eq 0 ]; then
    echo "Warning: no weight packs found under weights/*/. Run scripts/prep_lang.py <lang> first."
fi

# Push per-language asset packs (noise fixtures + test ids) and the shared
# uroman tables. Uroman is best-effort — required only for non-Latin scripts.
if [ -d "assets" ]; then
    for asset_dir in assets/*/; do
        [ -d "$asset_dir" ] || continue
        name=$(basename "$asset_dir")
        # Skip directories that are empty (e.g. just .gitkeep).
        if [ -z "$(find "$asset_dir" -maxdepth 1 -type f ! -name '.gitkeep' -print -quit 2>/dev/null)" ]; then
            continue
        fi
        echo "Pushing assets/$name/ ..."
        $ADB shell "mkdir -p $REMOTE_DIR/assets/$name"
        for f in "$asset_dir"*; do
            [ -f "$f" ] || continue
            case "$(basename "$f")" in
                .gitkeep) ;;
                *) $ADB push "$f" $REMOTE_DIR/assets/$name/ ;;
            esac
        done
    done
fi

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
if [ "${#PUSHED_LANGS[@]}" -gt 0 ]; then
    echo "Languages deployed: ${PUSHED_LANGS[*]}"
fi
echo ""
echo "To run on device:"
echo "  ./scripts/run_android.sh \"Hello, my name is.\" 1 --lang eng"
