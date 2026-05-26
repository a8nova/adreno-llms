#!/bin/bash
# Stages binaries and weights for both on-device models into the Gradle project.
#
# Inputs (built ahead of time via each model's own scripts/build.sh):
#   src/models/smolvlm-256m-instruct/build/fp16/SmolVLM_256M_Instruct_inference_fp16
#   src/models/smolvlm-256m-instruct/build/fp16/_deps/clblast-build/libclblast.so
#   src/models/mms-tts/build/fp16/mms_tts_eng_inference_fp16
#
# Outputs:
#   examples/see-and-say/app/src/main/jniLibs/arm64-v8a/{libsmolvlm,libmmstts,libclblast}.so
#   examples/see-and-say/app/src/main/assets/smolvlm/{weights,kernels}/...
#   examples/see-and-say/app/src/main/assets/mmstts/{weights,kernels,assets/uroman}/...
#
# Binaries are renamed to lib*.so so Android's PackageManager extracts them
# into nativeLibraryDir on install — one of the few app-writable locations
# with exec permission on Android 10+.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$APP_ROOT/../.." && pwd)"

JNI="$APP_ROOT/app/src/main/jniLibs/arm64-v8a"
ASSETS="$APP_ROOT/app/src/main/assets"

SMOLVLM_DIR="$REPO_ROOT/src/models/smolvlm-256m-instruct"
MMSTTS_DIR="$REPO_ROOT/src/models/mms-tts"
SMOLVLM_BIN="$SMOLVLM_DIR/build/fp16/SmolVLM_256M_Instruct_inference_fp16"
MMSTTS_BIN="$MMSTTS_DIR/build/fp16/mms_tts_eng_inference_fp16"
SMOLVLM_CLBLAST="$SMOLVLM_DIR/build/fp16/_deps/clblast-build/libclblast.so"
MMSTTS_CLBLAST="$MMSTTS_DIR/build/fp16/_deps/clblast-build/libclblast.so"

# Optional: uroman romanization tables — only needed for non-Latin scripts
# (Amharic, etc.). English degrades gracefully without them. Point UROMAN_DIR
# at a local checkout of https://github.com/isi-nlp/uroman to bundle them;
# leave it unset to skip non-Latin language packs.
UROMAN_DIR="${UROMAN_DIR:-}"

# NDK strip tool (used to shrink libclblast.so from ~66 MB → ~10 MB). Auto-
# discovered under $ANDROID_NDK when present; override with NDK_STRIP=<path>.
if [[ -z "${NDK_STRIP:-}" && -n "${ANDROID_NDK:-}" ]]; then
    for arch in darwin-x86_64 linux-x86_64; do
        cand="$ANDROID_NDK/toolchains/llvm/prebuilt/$arch/bin/llvm-strip"
        [[ -x "$cand" ]] && NDK_STRIP="$cand" && break
    done
fi
NDK_STRIP="${NDK_STRIP:-llvm-strip}"

say() { printf '\033[1;36m[prepare]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[prepare]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[prepare] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

say "Verifying build artifacts exist..."
[[ -f "$SMOLVLM_BIN" ]] || die "Missing $SMOLVLM_BIN — run \`scripts/build.sh\` in $SMOLVLM_DIR (NNOPT_DTYPE=fp16)"
[[ -f "$MMSTTS_BIN" ]]  || die "Missing $MMSTTS_BIN — run \`scripts/build.sh\` in $MMSTTS_DIR (NNOPT_DTYPE=fp16)"
[[ -f "$SMOLVLM_CLBLAST" || -f "$MMSTTS_CLBLAST" ]] || die "Missing libclblast.so in either model's build/fp16/_deps/clblast-build/"

CLBLAST_SRC="$SMOLVLM_CLBLAST"; [[ -f "$CLBLAST_SRC" ]] || CLBLAST_SRC="$MMSTTS_CLBLAST"

say "Staging native binaries → $JNI"
mkdir -p "$JNI"
install -m 0755 "$SMOLVLM_BIN"     "$JNI/libsmolvlm.so"
install -m 0755 "$MMSTTS_BIN"      "$JNI/libmmstts.so"
install -m 0755 "$CLBLAST_SRC"     "$JNI/libclblast.so"

if [[ -x "$NDK_STRIP" ]]; then
    say "Stripping libclblast.so (debug symbols → smaller APK)"
    "$NDK_STRIP" "$JNI/libclblast.so" || warn "strip failed; continuing with un-stripped lib"
else
    warn "NDK strip not found at $NDK_STRIP — libclblast.so will be ~66 MB unstripped"
fi

# ---- SmolVLM assets ------------------------------------------------------
say "Staging SmolVLM weights + kernels → $ASSETS/smolvlm"
SVDST="$ASSETS/smolvlm"
rm -rf "$SVDST"
mkdir -p "$SVDST/weights" "$SVDST/kernels"

# cp -L follows symlinks (model.fp16.bin and tokenizer_vocab.bin may be
# symlinks pointing into a separate models cache — we need the real bytes).
cp -L "$SMOLVLM_DIR/weights/model.fp16.bin"        "$SVDST/weights/" || die "model.fp16.bin missing"
cp -L "$SMOLVLM_DIR/weights/model.fp16.meta.json"  "$SVDST/weights/" || die "model.fp16.meta.json missing"
cp -L "$SMOLVLM_DIR/weights/tokenizer_vocab.bin"   "$SVDST/weights/" || die "tokenizer_vocab.bin missing"

# Sanity: tokenizer_vocab.bin was a 0-byte symlink in the source dir.
[[ -s "$SVDST/weights/tokenizer_vocab.bin" ]] || die "tokenizer_vocab.bin came across empty — symlink target missing?"

cp "$SMOLVLM_DIR/kernels/"*.cl "$SVDST/kernels/"

# Warmup fixture — a small JPEG we feed the binary at app startup so vision
# kernels compile + the OpenCL program cache is populated. Without this the
# first user-initiated query pays the full cold-load tax.
mkdir -p "$SVDST/fixtures"
if [[ -f "$SMOLVLM_DIR/fixtures/sample.jpg" ]]; then
    cp "$SMOLVLM_DIR/fixtures/sample.jpg" "$SVDST/fixtures/warmup.jpg"
else
    warn "no warmup fixture found at $SMOLVLM_DIR/fixtures/sample.jpg"
fi

# ---- MMS-TTS assets ------------------------------------------------------
# Language weights are NOT bundled — all languages (including English) are
# downloaded from HuggingFace at runtime via LanguageRegistry. Only the
# binary, OpenCL kernels, and uroman tables ship in the APK.
say "Staging MMS-TTS kernels + uroman → $ASSETS/mmstts"
MTDST="$ASSETS/mmstts"
rm -rf "$MTDST"
mkdir -p "$MTDST/kernels" "$MTDST/assets/uroman"

cp "$MMSTTS_DIR/kernels/"*.cl "$MTDST/kernels/"

# Uroman tables. The binary calls uroman::load_tables(romanization-table.txt)
# AND auto-loads `romanization-auto-table.txt` from the same directory if it
# exists (see src/models/mms-tts/src/uroman.cpp:247). The auto-table is ~978 KB
# and contains all the non-trivial transliterations including Amharic Ge'ez.
# Without it, non-Latin chars get dropped and you get tokenized garbage →
# audible noise from the model. We MUST ship both tables for Amharic to work.
#
# If UROMAN_DIR is not set, auto-fetch from GitHub via scripts/fetch_uroman.sh.
if [[ -z "$UROMAN_DIR" ]]; then
    FETCH_SCRIPT="$MMSTTS_DIR/scripts/fetch_uroman.sh"
    FETCHED_DIR="$MMSTTS_DIR/assets/uroman"
    if [[ -s "$FETCHED_DIR/romanization-table.txt" && -s "$FETCHED_DIR/romanization-auto-table.txt" ]]; then
        say "Uroman tables already fetched at $FETCHED_DIR"
    elif [[ -x "$FETCH_SCRIPT" ]]; then
        say "UROMAN_DIR not set — auto-fetching from GitHub..."
        bash "$FETCH_SCRIPT"
    else
        die "UROMAN_DIR not set and $FETCH_SCRIPT not found — cannot bundle uroman tables"
    fi
    UROMAN_DIR="$FETCHED_DIR"
fi

if [[ -f "$UROMAN_DIR/romanization-table.txt" ]]; then
    cp "$UROMAN_DIR/romanization-table.txt" "$MTDST/assets/uroman/"
    if [[ -f "$UROMAN_DIR/romanization-auto-table.txt" ]]; then
        cp "$UROMAN_DIR/romanization-auto-table.txt" "$MTDST/assets/uroman/"
        say "Uroman tables: base + auto (Amharic-capable)."
    else
        die "romanization-auto-table.txt missing — Amharic and other non-Latin languages will not work."
    fi
    [[ -f "$UROMAN_DIR/chars-to-delete.txt" ]] && cp "$UROMAN_DIR/chars-to-delete.txt" "$MTDST/assets/uroman/"
else
    die "romanization-table.txt missing at $UROMAN_DIR — run scripts/fetch_uroman.sh in $MMSTTS_DIR first."
fi

# ---- Summary -------------------------------------------------------------
say "Sizes:"
du -sh "$JNI"/*.so 2>/dev/null | sed 's/^/    /'
du -sh "$SVDST"/* "$MTDST"/* 2>/dev/null | sed 's/^/    /'

TOTAL_BYTES=$(du -sk "$JNI" "$SVDST" "$MTDST" 2>/dev/null | awk '{s+=$1} END {print s}')
say "Total APK payload (KB): $TOTAL_BYTES"

say "Done. Next: cd $APP_ROOT && ./gradlew :app:assembleDebug"
