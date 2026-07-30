#!/bin/bash
# Build the CPU reference forward for the DEV MACHINE — no OpenCL, no NDK, no
# device. This is the fast loop for bringing up a new architecture: iterate the
# math in reference_model.h and diff against the llama.cpp oracle in seconds.
#
#   ./scripts/build_host.sh
#   ./build/host/bonsai_ref weights/bonsai1.7b.nnb weights/tokenizer.json gen "The quick brown fox" 16
#
# Debug build (-O0 -g, no -ffast-math) for stepping through a divergence:
#   ./scripts/build_host.sh --debug

set -e

cd "$(dirname "$0")/.."

BUILD_TYPE=Release
for arg in "$@"; do
    case "$arg" in
        --debug) BUILD_TYPE=Debug ;;
        --release) BUILD_TYPE=Release ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done

BUILD_DIR=build/host

cmake -S . -B "$BUILD_DIR" \
    -DBONSAI_HOST_ONLY=1 \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    >/dev/null

cmake --build "$BUILD_DIR" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

echo
echo "==================================="
echo "Host reference build complete! ($BUILD_TYPE)"
echo "Binary: $BUILD_DIR/bonsai_ref"
echo "==================================="
