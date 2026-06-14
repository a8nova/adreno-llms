#!/usr/bin/env bash
# Fetch a Kokoro voice pack (.bin) from the adreno-llms weights repo into the
# runtime's voice slot. Until the --voicepack flag lands, the binary hardcodes
# assets/voice_pack_af_heart.bin, so any chosen voice is copied there.
#
# Usage: ./scripts/fetch_voice.sh [voice_name]    (default: af_heart)
#   e.g. ./scripts/fetch_voice.sh am_michael
# 54 voices exist — see VOICES.md on hexgrad/Kokoro-82M (af_*/am_* US, bf_*/bm_* GB,
# plus es/fr/it/pt/hi/ja/zh which also need the matching espeak language pack).
set -euo pipefail
cd "$(dirname "$0")/.."

HF_REPO="${HF_REPO:-a8nova/adreno-llms-weights}"
name="${1:-af_heart}"
DEST="assets/voice_pack_af_heart.bin"   # the hardcoded runtime slot

command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' CLI not found" >&2; exit 1; }
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

echo ">>> fetching voice '$name' from $HF_REPO"
hf download "$HF_REPO" --include "kokoro-82m/voices/$name.bin" --local-dir "$WORK" >/dev/null
src="$WORK/kokoro-82m/voices/$name.bin"
[ -f "$src" ] || { echo "ERROR: voice '$name' not on HF (run scripts/fetch_voices.sh to publish all 54)" >&2; exit 1; }

mkdir -p assets
cp "$src" "$DEST"
echo "    $name -> $DEST  (redeploy: ./scripts/deploy_android.sh)"
