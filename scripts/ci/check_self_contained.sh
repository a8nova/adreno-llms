#!/usr/bin/env bash
#
# CI gate — a model binary must be APP-LINKABLE.
#
# These binaries are packaged into a host app (e.g. Edgi) and run from its
# read-only jniLibs, which ship ONLY a fixed set of shared libraries: the system
# set + libOpenCL.so (+ a single shared libclblast.so). A binary that needs any
# OTHER shared library — or, in --strict mode, that needs libclblast.so or carries
# undefined C++ stdlib symbols it expects from a shared libc++ — will fail at
# runtime with:
#
#     CANNOT LINK EXECUTABLE "lib<model>.so": cannot locate symbol
#     "_ZNKSt6__ndk1...__throw_length_errorEv" referenced by "lib<model>.so"
#
# …even though it links and runs fine via `adb` from /data/local/tmp, because the
# standalone deploy pushes the matching libclblast.so / libc++_shared.so alongside
# it. This bug bit openvoice-v2: it was built against a newer CLBlast/libc++ than
# the libclblast.so committed in the app, so a libc++ symbol the app's clblast
# didn't export went unresolved.
#
# This check is DEVICE-FREE — pure ELF inspection via the NDK's llvm-readelf — so
# it runs on GitHub-hosted runners with no Adreno GPU.
#
# Usage:
#   check_self_contained.sh <binary> [--strict]
#
#   default  : forbids libc++_shared.so / libc++.so and any shared lib outside the
#              system + OpenCL + clblast allowlist (catches accidental new deps).
#   --strict : additionally forbids libclblast.so and ANY undefined C++ stdlib
#              symbol — i.e. the binary must be fully self-contained (static
#              CLBlast + static libc++). Use for app-embedded models that cannot
#              rely on the app shipping a symbol-compatible libclblast.so.
set -euo pipefail

BIN="${1:?usage: check_self_contained.sh <binary> [--strict]}"
STRICT=0
[ "${2:-}" = "--strict" ] && STRICT=1
[ -f "$BIN" ] || { echo "ERROR: no such binary: $BIN"; exit 2; }

# ── Locate the NDK's llvm-readelf (env first, then the standard SDK locations) ──
NDK="${ANDROID_NDK:-${ANDROID_NDK_LATEST_HOME:-${ANDROID_NDK_HOME:-}}}"
if [ -z "$NDK" ] || [ ! -d "$NDK" ]; then
    for base in "$HOME/Library/Android/sdk/ndk" "$HOME/Android/Sdk/ndk"; do
        if [ -d "$base" ]; then NDK="$base/$(ls -1 "$base" | sort -V | tail -1)"; break; fi
    done
fi
READELF="$(ls "$NDK"/toolchains/llvm/prebuilt/*/bin/llvm-readelf 2>/dev/null | head -1 || true)"
[ -x "$READELF" ] || { echo "ERROR: llvm-readelf not found under NDK ('$NDK'); set ANDROID_NDK"; exit 2; }

name="$(basename "$BIN")"
fail=0

# ── (1) NEEDED shared libraries must be a subset of what a host app provides ──
allowed="libOpenCL.so libm.so libdl.so libc.so liblog.so ld-android.so"
[ "$STRICT" -eq 0 ] && allowed="$allowed libclblast.so"
needed="$("$READELF" -d "$BIN" | sed -nE 's/.*\(NEEDED\).*\[(.+)\]/\1/p')"
for lib in $needed; do
    case " $allowed " in
        *" $lib "*) ;;
        *) echo "FAIL [$name]: depends on shared library not shipped by the app: $lib"; fail=1 ;;
    esac
done

# ── (2) --strict: no undefined C++ stdlib symbols (would need a shared libc++) ──
if [ "$STRICT" -eq 1 ]; then
    und_cpp="$("$READELF" --dyn-symbols "$BIN" 2>/dev/null \
        | awk '$7=="UND"{print $8}' \
        | grep -E 'St[0-9]+__ndk1|^_ZN?K?St[0-9]|^_ZTVN?St|^_ZTINSt|^_ZTISt' || true)"
    if [ -n "$und_cpp" ]; then
        echo "FAIL [$name]: undefined C++ stdlib symbols (binary expects a shared libc++/clblast at runtime):"
        echo "$und_cpp" | head -10 | sed 's/^/    /'
        extra=$(($(echo "$und_cpp" | wc -l) - 10)); [ "$extra" -gt 0 ] && echo "    … and $extra more"
        fail=1
    fi
fi

if [ "$fail" -eq 0 ]; then
    echo "OK [$name]: app-linkable${STRICT:+ (strict: self-contained)} — NEEDED: ${needed:-none}"
    exit 0
fi

cat >&2 <<'EOF'

How to fix: build the model with a STATICALLY linked CLBlast and static libc++ so
the binary embeds both and depends on nothing under the app's jniLibs:
  • CMakeLists.txt — for the FetchContent CLBlast, set
      set(BUILD_SHARED_LIBS OFF CACHE BOOL "Static CLBlast: self-contained binary" FORCE)
    (do NOT force it ON for ANDROID), and
  • scripts/build.sh — keep -DANDROID_STL=c++_static.
See src/models/openvoice-v2/CMakeLists.txt for the reference setup.
EOF
exit 1
