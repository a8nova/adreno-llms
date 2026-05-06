#!/usr/bin/env bash
# One-time dependency setup for adreno-llms.
#
# Downloads:
#   1. Khronos OpenCL C headers (CL/cl.h, etc.) into the local cache.
#   2. libOpenCL.so from a connected Android device (used as a link-time
#      stub so the inference binary can link against the device's vendor
#      OpenCL ICD; the actual runtime lookup happens via dlopen on device).
#
# Cache lives at:
#   ${ADRENO_LLMS_CACHE:-$HOME/.cache/adreno-llms}/opencl/
#
# Idempotent — re-running is a no-op once both items are present.
#
# CLBlast is NOT downloaded here — each model's CMakeLists.txt fetches +
# builds CLBlast 1.6.3 from source via FetchContent on first cmake configure.

set -euo pipefail

CACHE="${ADRENO_LLMS_CACHE:-$HOME/.cache/adreno-llms}"
OPENCL_DIR="${CACHE}/opencl"
HEADERS_DIR="${OPENCL_DIR}/include/CL"
LIB_DIR="${OPENCL_DIR}/lib/android-arm64-v8a"
LIBCL="${LIB_DIR}/libOpenCL.so"

HEADERS_TAG="${OPENCL_HEADERS_TAG:-v2024.10.24}"
HEADERS_URL="https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/tags/${HEADERS_TAG}.tar.gz"

mkdir -p "${HEADERS_DIR}" "${LIB_DIR}"

# ── 1. OpenCL headers ───────────────────────────────────────────────────
echo ">>> [1/2] OpenCL headers (Khronos ${HEADERS_TAG})"
if [ -f "${HEADERS_DIR}/cl.h" ]; then
    echo "    already present at ${HEADERS_DIR}/cl.h"
else
    echo "    downloading ${HEADERS_URL}"
    TMP=$(mktemp -d)
    trap 'rm -rf "${TMP}"' EXIT
    curl --location --fail-with-body --silent --output "${TMP}/headers.tar.gz" "${HEADERS_URL}"
    tar -xzf "${TMP}/headers.tar.gz" -C "${TMP}"
    SRC_HEADERS_DIR="${TMP}/OpenCL-Headers-${HEADERS_TAG#v}/CL"
    if [ ! -d "${SRC_HEADERS_DIR}" ]; then
        echo "    ERROR: extracted tarball doesn't contain CL/ directory at ${SRC_HEADERS_DIR}" >&2
        exit 1
    fi
    cp "${SRC_HEADERS_DIR}"/*.h "${HEADERS_DIR}/"
    echo "    installed $(ls "${HEADERS_DIR}" | wc -l | tr -d ' ') headers to ${HEADERS_DIR}"
    trap - EXIT
    rm -rf "${TMP}"
fi

# ── 2. Android libOpenCL.so (link stub) ─────────────────────────────────
echo ""
echo ">>> [2/2] Android libOpenCL.so (link stub, pulled from a connected device)"
if [ -f "${LIBCL}" ]; then
    echo "    already present at ${LIBCL}"
else
    if ! command -v adb >/dev/null 2>&1; then
        echo "    ERROR: adb not in PATH" >&2
        echo "    install android-platform-tools, or manually copy any" >&2
        echo "    Android arm64-v8a libOpenCL.so to ${LIBCL}" >&2
        exit 1
    fi
    if [ "$(adb devices | tail -n +2 | grep -c 'device$')" -eq 0 ]; then
        echo "    ERROR: no Android device connected via adb" >&2
        echo "    connect your target device and re-run, or copy a libOpenCL.so to:" >&2
        echo "      ${LIBCL}" >&2
        exit 1
    fi
    PULLED=""
    for path in /vendor/lib64/libOpenCL.so /system/vendor/lib64/libOpenCL.so /system/lib64/libOpenCL.so; do
        if adb shell "[ -f ${path} ]" 2>/dev/null; then
            echo "    pulling from device: ${path}"
            adb pull "${path}" "${LIBCL}" >/dev/null 2>&1 && PULLED="1"
            break
        fi
    done
    if [ -z "${PULLED}" ] || [ ! -f "${LIBCL}" ]; then
        echo "    ERROR: libOpenCL.so not found on device at any standard path" >&2
        echo "    standard paths checked: /vendor/lib64, /system/vendor/lib64, /system/lib64" >&2
        exit 1
    fi
    echo "    saved $(stat -f%z "${LIBCL}" 2>/dev/null || stat -c%s "${LIBCL}") bytes to ${LIBCL}"
fi

echo ""
echo "Deps ready at ${OPENCL_DIR}"
