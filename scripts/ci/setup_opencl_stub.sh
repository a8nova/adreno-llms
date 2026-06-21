#!/usr/bin/env bash
# Device-free OpenCL dependency bootstrap for CI (and any machine with no
# Android device attached). Populates the same cache that build.sh expects:
#
#   $ADRENO_LLMS_CACHE/opencl/include/CL/*.h            (Khronos headers)
#   $ADRENO_LLMS_CACHE/opencl/lib/android-arm64-v8a/libOpenCL.so   (link stub)
#
# The stub libOpenCL.so is built from the Khronos OpenCL-ICD-Loader (cross-
# compiled with the NDK) — a real, license-clean arm64 .so exporting the full
# OpenCL API. It's only a LINK stub: on a real device the vendor ICD is used at
# runtime. This lets `build.sh` compile/link every model in cloud CI without a
# phone (build.sh skips its own device-pull path once these files exist).
#
# Requires: ANDROID_NDK set (or auto-discoverable), cmake, git, curl.
set -euo pipefail

HEADERS_TAG="${OPENCL_HEADERS_TAG:-v2024.10.24}"
ICD_TAG="${OPENCL_ICD_TAG:-v2024.10.24}"
ADRENO_LLMS_CACHE="${ADRENO_LLMS_CACHE:-$HOME/.cache/adreno-llms}"
OPENCL_DIR="${ADRENO_LLMS_CACHE}/opencl"
INC_DIR="${OPENCL_DIR}/include"
LIB_DIR="${OPENCL_DIR}/lib/android-arm64-v8a"
mkdir -p "${INC_DIR}/CL" "${LIB_DIR}"

# ── NDK ──────────────────────────────────────────────────────────────────
ANDROID_NDK="${ANDROID_NDK:-${ANDROID_NDK_LATEST_HOME:-${ANDROID_NDK_HOME:-}}}"
if [ -z "${ANDROID_NDK}" ] || [ ! -d "${ANDROID_NDK}" ]; then
  echo "ERROR: ANDROID_NDK not set / not found. On GitHub runners use ANDROID_NDK_LATEST_HOME." >&2
  exit 1
fi
TOOLCHAIN="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
[ -f "${TOOLCHAIN}" ] || { echo "ERROR: NDK toolchain missing: ${TOOLCHAIN}" >&2; exit 1; }
echo "Using NDK: ${ANDROID_NDK}"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# ── 1. Khronos OpenCL headers ────────────────────────────────────────────
echo ">>> [1/2] OpenCL headers (${HEADERS_TAG})"
if [ -f "${INC_DIR}/CL/cl.h" ]; then
  echo "    already present"
else
  curl -fsSL "https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/tags/${HEADERS_TAG}.tar.gz" \
    -o "${WORK}/headers.tgz"
  tar -xzf "${WORK}/headers.tgz" -C "${WORK}"
  cp "${WORK}/OpenCL-Headers-${HEADERS_TAG#v}/CL/"*.h "${INC_DIR}/CL/"
  echo "    installed $(ls "${INC_DIR}/CL" | wc -l | tr -d ' ') headers"
fi

# ── 2. libOpenCL.so link stub (OpenCL-ICD-Loader, cross-compiled) ────────
echo ">>> [2/2] libOpenCL.so stub (OpenCL-ICD-Loader ${ICD_TAG})"
if [ -f "${LIB_DIR}/libOpenCL.so" ]; then
  echo "    already present"
else
  git clone --depth 1 --branch "${ICD_TAG}" \
    https://github.com/KhronosGroup/OpenCL-ICD-Loader.git "${WORK}/icd"
  cmake -S "${WORK}/icd" -B "${WORK}/icd/build" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=21 \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENCL_ICD_LOADER_HEADERS_DIR="${INC_DIR}" \
    -DOPENCL_ICD_LOADER_BUILD_TESTING=OFF >/dev/null
  cmake --build "${WORK}/icd/build" --target OpenCL -j"$(getconf _NPROCESSORS_ONLN)" >/dev/null
  found="$(find "${WORK}/icd/build" -name 'libOpenCL.so*' -type f | head -1)"
  [ -n "${found}" ] || { echo "ERROR: ICD-loader build produced no libOpenCL.so" >&2; exit 1; }
  cp "${found}" "${LIB_DIR}/libOpenCL.so"
  echo "    built $(du -h "${LIB_DIR}/libOpenCL.so" | cut -f1) stub"
fi

echo ""
echo "OpenCL deps ready under ${OPENCL_DIR}"
