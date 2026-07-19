#!/bin/bash
# Build script for Bonsai-8B (Q1_0 1-bit).
# Cross-compiles for Android via the NDK + CMake. Convention matches
# adreno-llms/src/models/qwen2-5-0-5b/scripts/build.sh.

set -e

cd "$(dirname "$0")/.."

# ------------------------------------------------------------------ args
CLEAN=""
BUILD_TYPE="Release"

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)   CLEAN="yes"; shift ;;
        --debug)   BUILD_TYPE="Debug"; shift ;;
        --release) BUILD_TYPE="Release"; shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--clean] [--debug] [--release]"
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------------ dtype
# Storage dtype for ACTIVATIONS on device. fp32 is the token-exact gate
# configuration (weights are always 1-bit packed regardless); fp16 is the
# P3 performance A/B. fp32 -> build/, binary bonsai_inference;
# fp16 -> build/fp16/, binary bonsai_inference_f16 (-DBONSAI_FP16_STORAGE=1).
BONSAI_STORAGE="${BONSAI_STORAGE:-fp32}"
case "$BONSAI_STORAGE" in
    fp32) BUILD_DIR="build";      BIN_SUFFIX="";     FP16_ARG="" ;;
    fp16) BUILD_DIR="build/fp16"; BIN_SUFFIX="_f16"; FP16_ARG="-DBONSAI_FP16_STORAGE=1" ;;
    *) echo "ERROR: BONSAI_STORAGE must be fp32 or fp16 (got '$BONSAI_STORAGE')" >&2; exit 1 ;;
esac
BINARY_NAME="bonsai_inference${BIN_SUFFIX}"
echo "Build dtype: $BONSAI_STORAGE (build dir: $BUILD_DIR)"

# A previous configure of a different build type carries stale CMakeCache.txt
# entries — force a clean reconfigure when the type changes.
if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
    CACHED_TYPE=$(grep -E "^CMAKE_BUILD_TYPE:" "$BUILD_DIR/CMakeCache.txt" | head -1 | cut -d= -f2)
    if [ -n "$CACHED_TYPE" ] && [ "$CACHED_TYPE" != "$BUILD_TYPE" ]; then
        echo "Build type changed ($CACHED_TYPE -> $BUILD_TYPE) — forcing clean reconfigure"
        CLEAN="yes"
    fi
fi

# ------------------------------------------------------------------ NDK
ANDROID_NDK="${ANDROID_NDK:-}"
if [ -z "$ANDROID_NDK" ] || [ ! -d "$ANDROID_NDK" ]; then
    for NDK_BASE in "$HOME/Library/Android/sdk/ndk" "$HOME/Android/Sdk/ndk"; do
        if [ -d "$NDK_BASE" ]; then
            LATEST=$(ls -1 "$NDK_BASE" 2>/dev/null | sort -V | tail -1)
            if [ -n "$LATEST" ] && [ -d "$NDK_BASE/$LATEST" ]; then
                ANDROID_NDK="$NDK_BASE/$LATEST"
                echo "Auto-discovered NDK: $ANDROID_NDK"
                break
            fi
        fi
    done
fi
if [ -z "$ANDROID_NDK" ] || [ ! -d "$ANDROID_NDK" ]; then
    echo "ERROR: Android NDK not found. Set ANDROID_NDK or install the NDK." >&2
    exit 1
fi
TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
if [ ! -f "$TOOLCHAIN_FILE" ]; then
    echo "ERROR: Android toolchain file not found: $TOOLCHAIN_FILE" >&2
    exit 1
fi
echo "Using Android NDK: $ANDROID_NDK"
echo "Target ABI: arm64-v8a / API 21"

# ------------------------------------------------------------------ OpenCL deps
# Headers + libOpenCL.so link stub. Prefer the repo-standard cache the CI
# populates ($ADRENO_LLMS_CACHE/opencl via scripts/ci/setup_opencl_stub.sh);
# fall back to the legacy nnopt location for local dev. Override with OPENCL_DEPS.
ADRENO_LLMS_CACHE="${ADRENO_LLMS_CACHE:-$HOME/.cache/adreno-llms}"
OPENCL_DEPS="${OPENCL_DEPS:-$ADRENO_LLMS_CACHE/opencl}"
[ -f "$OPENCL_DEPS/include/CL/cl.h" ] || OPENCL_DEPS="$HOME/.nnopt/deps/opencl"
OPENCL_INC="${OPENCL_INC:-$OPENCL_DEPS/include}"
OPENCL_LIB="${OPENCL_LIB:-$OPENCL_DEPS/lib/android-arm64-v8a/libOpenCL.so}"

if [ ! -f "$OPENCL_INC/CL/cl.h" ]; then
    echo "ERROR: OpenCL headers not found at $OPENCL_INC (expected CL/cl.h)" >&2
    echo "  Run scripts/ci/setup_opencl_stub.sh (device-free) or set OPENCL_DEPS." >&2
    exit 1
fi
if [ ! -f "$OPENCL_LIB" ]; then
    echo "ERROR: OpenCL link stub not found at $OPENCL_LIB" >&2
    exit 1
fi
echo "Using OpenCL headers: $OPENCL_INC"
echo "Using OpenCL library: $OPENCL_LIB"

# ------------------------------------------------------------------ configure + build
[ "$CLEAN" = "yes" ] && { echo "Cleaning $BUILD_DIR..."; rm -rf "$BUILD_DIR"; }
mkdir -p "$BUILD_DIR"
PROJECT_ROOT="$(pwd)"
cd "$BUILD_DIR"

echo "Configuring with CMake for Android ($BUILD_TYPE, $BONSAI_STORAGE)..."
cmake "$PROJECT_ROOT" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-21 \
    -DANDROID_STL=c++_static \
    -DOPENCL_INCLUDE_DIR="$OPENCL_INC" \
    -DOPENCL_LIBRARIES="$OPENCL_LIB" \
    $FP16_ARG \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE

echo "Building for Android..."
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --build . -j$CORES

cd "$PROJECT_ROOT"

echo ""
echo "==================================="
echo "Android build complete! ($BUILD_TYPE / $BONSAI_STORAGE)"
echo "Binary: $BUILD_DIR/$BINARY_NAME"
echo "==================================="
echo "To deploy: BONSAI_STORAGE=$BONSAI_STORAGE ./scripts/deploy_android.sh"
