#!/usr/bin/env bash
# Enforce the CLBlast linkage convention: if a model links CLBlast at all, it
# MUST link it DYNAMICALLY (against the shared libclblast.so that is shipped
# ONCE for the whole app), never statically. A static CLBlast bakes ~3 MB of
# duplicated GEMM code into every model binary; the shared lib is stored once.
#
# Device-free: inspects the built binary's dynamic section + CLBlast fingerprint.
#   • DT_NEEDED libclblast.so present            -> dynamic  ✓
#   • CLBlast kernel strings but no DT_NEEDED     -> STATIC   ✗ (fail)
#   • no CLBlast fingerprint at all               -> no GEMM  ✓ (bonsai, moonshine)
#
# Exemption: openvoice-v2 is intentionally --strict self-contained (app-embedded,
# static CLBlast + static libc++), so it is allowed to link CLBlast statically.
set -uo pipefail

BIN="${1:?usage: check_clblast_dynamic.sh <binary> [model-name]}"
MODEL="${2:-$(basename "$BIN")}"

[ -f "$BIN" ] || { echo "✗ ${MODEL}: binary not found: $BIN"; exit 1; }

if [ "$MODEL" = "openvoice-v2" ]; then
    echo "✓ ${MODEL}: exempt (intentionally --strict self-contained / static CLBlast)"
    exit 0
fi

# Locate a readelf that can read arm64 ELF (system binutils works; else the NDK's).
READELF=""
for c in readelf llvm-readelf; do command -v "$c" >/dev/null 2>&1 && { READELF="$c"; break; }; done
if [ -z "$READELF" ] && [ -n "${ANDROID_NDK:-}" ]; then
    READELF=$(ls "$ANDROID_NDK"/toolchains/llvm/prebuilt/*/bin/llvm-readelf 2>/dev/null | head -1)
fi
[ -n "$READELF" ] || { echo "✗ ${MODEL}: no readelf available"; exit 1; }

dyn=$("$READELF" -d "$BIN" 2>/dev/null | grep -ci "NEEDED.*libclblast" || true)
strs=$(strings -a "$BIN" 2>/dev/null | grep -icE "XgemmDirect|CLBlastSgemm|clblast" || true)

if [ "${dyn:-0}" -ge 1 ]; then
    echo "✓ ${MODEL}: CLBlast linked DYNAMICALLY (DT_NEEDED libclblast.so)"
    exit 0
elif [ "${strs:-0}" -gt 20 ]; then
    echo "✗ ${MODEL}: CLBlast is STATICALLY linked (${strs} CLBlast fingerprints, no DT_NEEDED libclblast.so)."
    echo "    Every CLBlast user must link the SHARED libclblast.so (shipped once) instead of baking"
    echo "    a static copy into each binary. In the model's CMakeLists, on Android set"
    echo "    BUILD_SHARED_LIBS ON for the CLBlast FetchContent (see qwen2-5-0-5b/CMakeLists.txt)."
    echo "    If this model is intentionally self-contained, add it to the exemption above."
    exit 1
else
    echo "✓ ${MODEL}: no CLBlast dependency (custom kernels / no dense GEMM)"
    exit 0
fi
