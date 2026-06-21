#!/usr/bin/env bash
# Fetch an espeak language pack (dict + lang config) from the adreno-llms weights
# repo and merge it into assets/espeak-ng-data/, so kokoro can phonemize that
# language. The shared phoneme core ships committed (the English subset); this
# only adds the language-specific files. Deploy afterwards to push to device.
#
# Usage: ./scripts/fetch_espeak_lang.sh <es|fr|it|pt|hi|ja|cmn>
#
# Note: es/fr/it/pt/hi are espeak-native. ja/cmn are best-effort — Kokoro upstream
# uses dedicated G2P for Japanese/Chinese, so phonemes may differ; ear-check.
set -euo pipefail
cd "$(dirname "$0")/.."

HF_REPO="${HF_REPO:-a8nova/adreno-llms-weights}"
DEST="assets/espeak-ng-data"
code="${1:-}"
case "$code" in
  en|es|fr|it|pt|hi|ja|cmn) ;;
  *) echo "usage: $0 <en|es|fr|it|pt|hi|ja|cmn>   (en = base pack w/ shared core; fetch it first)" >&2; exit 1 ;;
esac

case "$code" in ja|cmn) echo "⚠️  '$code' is EXPERIMENTAL: espeak fallback only — Kokoro upstream uses dedicated G2P (openjtalk/jieba). Ear-check the output." >&2 ;; esac

command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' CLI not found" >&2; exit 1; }
mkdir -p "$DEST"
# Every language needs the shared phoneme core, which ships in the `en` base
# pack. Auto-fetch `en` first if it isn't present (unless we ARE fetching en).
if [ "$code" != "en" ] && [ ! -f "$DEST/phondata" ]; then
  echo ">>> shared phoneme core missing — fetching the 'en' base pack first"
  "$0" en
fi

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
echo ">>> fetching espeak-lang/$code from $HF_REPO"
hf download "$HF_REPO" --include "kokoro-82m/espeak-lang/$code/*" --local-dir "$WORK" >/dev/null
SRC="$WORK/kokoro-82m/espeak-lang/$code"
[ -d "$SRC" ] || { echo "ERROR: pack not found on HF (run upload_espeak_langs.sh first)" >&2; exit 1; }

rsync -a "$SRC/" "$DEST/"
echo "    merged $(find "$SRC" -type f | wc -l | tr -d ' ') files into $DEST/"
echo "Done — redeploy ( ./scripts/deploy_android.sh ) to push the '$code' language to the device."
