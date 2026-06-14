#!/usr/bin/env bash
# Build per-language espeak packs (dict + lang config) and upload them to the
# adreno-llms weights repo under kokoro-82m/espeak-lang/<code>/. The shared
# phoneme core (phondata/phonindex/phontab/intonations) is NOT included here —
# it ships committed in assets/espeak-ng-data/ (the English subset). A language
# pack only adds the language-specific dictionary + lang config.
#
# Source = a full espeak-ng-data dir (default: the nnopt kokoro port). Run after
# `hf auth login`. The on-device side fetches a pack with fetch_espeak_lang.sh.
#
# Languages (kokoro voice prefix → espeak): e→es f→fr i→it p→pt h→hi  (espeak-
# native) and j→ja z→cmn (espeak CAN do these but Kokoro upstream uses dedicated
# G2P — treat ja/zh as best-effort; ear-check before shipping).
set -euo pipefail
cd "$(dirname "$0")/.."

HF_REPO="${HF_REPO:-a8nova/adreno-llms-weights}"
ESPEAK_FULL="${ESPEAK_FULL:-/Users/alazarshenkute/Projects/nnopt/ports/kokoro-android-opencl-motorola-razr-2020/assets/espeak-ng-data}"

# code : space-separated relative paths within espeak-ng-data.
# `en` is the SELF-CONTAINED base pack: it carries the shared phoneme core
# (phondata/phonindex/phontab/intonations) that every other (thin) pack needs,
# plus the English dict + lang configs. Fetch `en` first, then layer others.
LANGS=(
  "en:phondata phondata-manifest phonindex phontab intonations en_dict lang/gmw/en lang/gmw/en-029 lang/gmw/en-GB-scotland lang/gmw/en-GB-x-gbclan lang/gmw/en-GB-x-gbcwmd lang/gmw/en-GB-x-rp lang/gmw/en-Shaw lang/gmw/en-US lang/gmw/en-US-nyc"
  "es:es_dict lang/roa/es lang/roa/es-419"
  "fr:fr_dict lang/roa/fr lang/roa/fr-BE lang/roa/fr-CH"
  "it:it_dict lang/roa/it"
  "pt:pt_dict lang/roa/pt lang/roa/pt-BR"
  "hi:hi_dict lang/inc/hi"
  "ja:ja_dict lang/jpx/ja"
  "cmn:cmn_dict lang/sit/cmn lang/sit/cmn-Latn-pinyin"
)

command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' CLI not found" >&2; exit 1; }
hf auth whoami >/dev/null 2>&1 || { echo "ERROR: not logged in — run: hf auth login" >&2; exit 1; }
[ -f "$ESPEAK_FULL/phondata" ] || { echo "ERROR: full espeak-ng-data not at $ESPEAK_FULL (set ESPEAK_FULL=)" >&2; exit 1; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

for entry in "${LANGS[@]}"; do
  code="${entry%%:*}"; files="${entry#*:}"
  pack="$WORK/$code"; mkdir -p "$pack"
  echo ">>> $code"
  case "$code" in ja|cmn) echo "    ⚠️  EXPERIMENTAL: espeak fallback only — Kokoro upstream uses dedicated G2P (openjtalk/jieba); quality differs" ;; esac
  for rel in $files; do
    if [ ! -f "$ESPEAK_FULL/$rel" ]; then echo "    MISSING: $rel" >&2; exit 1; fi
    mkdir -p "$pack/$(dirname "$rel")"
    cp "$ESPEAK_FULL/$rel" "$pack/$rel"
    echo "    + $rel"
  done
  hf upload "$HF_REPO" "$pack" "kokoro-82m/espeak-lang/$code"
done

echo ""
echo "Done — 7 language packs at $HF_REPO/kokoro-82m/espeak-lang/{es,fr,it,pt,hi,ja,cmn}/"
