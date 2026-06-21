#!/bin/bash
# Unified on-device runner for SeamlessM4T UnitY-small (Adreno OpenCL fp16).
# One script for every mode — push input, run on device, return the result.
#
# Usage:
#   ./scripts/run_android.sh <mode> <input.wav> [lang] [output.wav]
#
#   mode    s2s       speech -> translated SPEECH   (writes a WAV)
#           s2tt      speech -> translated TEXT      (prints text)
#           asr       speech -> TEXT, same language  (transcribe; prints text)
#           s2tt_all  speech -> TEXT for ALL langs   (encode once, prints 5 lines)
#   input   PCM16 WAV, any sample rate / channel count (resampled to 16 kHz on device)
#   lang    eng (default) | spa | por | hin | rus     (ignored by s2tt_all)
#   output  s2s only — local path for the synthesized 16 kHz mono WAV (default: output.wav)
#
# Examples:
#   ./scripts/run_android.sh s2s      samples/what_is_your_name.wav spa out.wav
#   ./scripts/run_android.sh asr      samples/what_is_your_name.wav
#   ./scripts/run_android.sh s2tt     samples/what_is_your_name.wav por
#   ./scripts/run_android.sh s2tt_all samples/what_is_your_name.wav
set -e
cd "$(dirname "$0")/.."

NNOPT_DTYPE="${NNOPT_DTYPE:-fp16}"; export NNOPT_DTYPE
. ./scripts/remote_dir.env          # REMOTE_DIR, BINARY_NAME
ADB="${ADB:-adb}"

usage() { sed -n '2,20p' "$0"; exit 1; }
MODE="${1:-}"; IN="${2:-}"; LANG_TGT="${3:-eng}"; OUT="${4:-output.wav}"
[ -n "$MODE" ] && [ -n "$IN" ] || usage
[ -f "$IN" ] || { echo "ERROR: input '$IN' not found" >&2; exit 1; }

echo "Pushing $IN -> device..." >&2
$ADB push "$IN" "$REMOTE_DIR/assets/input.wav" >/dev/null
RUN="cd $REMOTE_DIR && LD_LIBRARY_PATH=$REMOTE_DIR/lib:/system/vendor/lib64:\$LD_LIBRARY_PATH"

case "$MODE" in
  s2s)
    echo "speech->speech ($LANG_TGT) ..." >&2
    $ADB shell "$RUN ./$BINARY_NAME s2s 0 --in assets/input.wav --out output.wav --lang $LANG_TGT"
    $ADB pull "$REMOTE_DIR/output.wav" "$OUT" >/dev/null
    echo "Wrote $OUT"
    ;;
  s2tt|asr|s2tt_all)
    if [ "$MODE" = "s2tt_all" ]; then echo "speech->text (all languages) ..." >&2; else echo "$MODE ($LANG_TGT) ..." >&2; fi
    $ADB shell "$RUN ./$BINARY_NAME $MODE 0 --in assets/input.wav --lang $LANG_TGT"
    ;;
  *)
    echo "ERROR: unknown mode '$MODE'" >&2; usage ;;
esac
