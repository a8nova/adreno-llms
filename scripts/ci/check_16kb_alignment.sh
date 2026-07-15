#!/usr/bin/env bash
#
# check_16kb_alignment.sh — CI gate: a model binary must be GOOGLE-PLAY-SHIPPABLE
# on Android 15 (API 35+), i.e. every PT_LOAD segment 16 KB (0x4000)-page aligned.
#
# WHY THIS EXISTS
#   NDK r26d links native .so with 4 KB (0x1000) PT_LOAD alignment by default. Google
#   Play REJECTS an AAB whose bundled .so isn't 16 KB-aligned, and such a lib fails to
#   load on 16 KB-page devices. AGP does not catch it, so a green app build can still be
#   Play-rejected at upload. Each model must link with -Wl,-z,max-page-size=16384; this
#   gate proves it BEFORE the binary is handed to the host app (Edgi), so the failure
#   surfaces in the model repo — where the fix lives — not at the store console.
#
# RELIABLE + DEVICE-FREE
#   Pure ELF program-header parsing (no NDK, no readelf, no device) — the same check
#   Edgi's build.gradle.kts verifyNativeLibAlignment task runs, so the two can't drift.
#   Also asserts the binary is arm64 (EM_AARCH64) — the only ABI Edgi ships.
#
# Usage:
#   check_16kb_alignment.sh <binary|dir> [more...]      # files and/or directories
#   check_16kb_alignment.sh build/fp16/foo_inference_fp16
#   check_16kb_alignment.sh app/src/main/jniLibs/arm64-v8a   # scans *.so in a dir
#
# Exit 0 = all inputs are 16 KB-aligned arm64 ELFs; non-zero = at least one is not.
set -euo pipefail

[ "$#" -ge 1 ] || { echo "usage: check_16kb_alignment.sh <binary|dir> [more...]"; exit 2; }

# Expand any directory arg into the .so files it contains.
FILES=()
for arg in "$@"; do
    if [ -d "$arg" ]; then
        while IFS= read -r f; do FILES+=("$f"); done < <(find "$arg" -maxdepth 1 -name '*.so' | sort)
    elif [ -f "$arg" ]; then
        FILES+=("$arg")
    else
        echo "ERROR: no such file or directory: $arg"; exit 2
    fi
done
[ "${#FILES[@]}" -ge 1 ] || { echo "ERROR: no binaries to check"; exit 2; }

MIN_ALIGN=16384   # 0x4000

python3 - "$MIN_ALIGN" "${FILES[@]}" <<'PY'
import struct, sys

min_align = int(sys.argv[1])
paths = sys.argv[2:]

def inspect(path):
    """Return (ok, arch_ok, max_pt_load_align, detail) for an ELF file."""
    with open(path, "rb") as fh:
        b = fh.read()
    if len(b) < 64 or b[:4] != b"\x7fELF":
        return (False, False, 0, "not an ELF file")
    is64 = b[4] == 2
    little = b[5] == 1
    if not is64:
        return (False, False, 0, "not ELF64 (arm64 required)")
    end = "<" if little else ">"
    machine = struct.unpack_from(end + "H", b, 0x12)[0]
    arch_ok = (machine == 0xB7)  # EM_AARCH64
    phoff = struct.unpack_from(end + "Q", b, 0x20)[0]
    phentsize = struct.unpack_from(end + "H", b, 0x36)[0]
    phnum = struct.unpack_from(end + "H", b, 0x38)[0]
    max_align = 0
    for i in range(phnum):
        o = phoff + i * phentsize
        p_type = struct.unpack_from(end + "I", b, o)[0]
        if p_type == 1:  # PT_LOAD
            align = struct.unpack_from(end + "Q", b, o + 0x30)[0]
            max_align = max(max_align, align)
    ok = arch_ok and max_align >= min_align
    return (ok, arch_ok, max_align, "")

fail = 0
for p in paths:
    try:
        ok, arch_ok, align, detail = inspect(p)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL  {p}: {e}")
        fail = 1
        continue
    name = p.rsplit("/", 1)[-1]
    if detail:
        print(f"  FAIL  {name}: {detail}")
        fail = 1
        continue
    tag = "OK  " if ok else "FAIL"
    arch = "arm64" if arch_ok else "NOT-arm64"
    print(f"  {tag}  0x{align:x}  {arch}  {name}")
    if not ok:
        fail = 1

if fail:
    print()
    print(f"16 KB-alignment check FAILED. Rebuild the flagged binaries with")
    print(f"  -Wl,-z,max-page-size=16384")
    print(f"in the model's CMakeLists.txt target_link_options(<exe> PRIVATE ...) block, then")
    print(f"re-copy into the app's jniLibs. Google Play rejects 4 KB-aligned .so on Android 15.")
    sys.exit(1)

print(f"16 KB-alignment OK: all {len(paths)} binary(ies) are >= 0x{min_align:x} and arm64.")
PY
