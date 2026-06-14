#!/usr/bin/env bash
# Download all Kokoro voice packs from hexgrad/Kokoro-82M, convert each to the
# runtime .bin format (scripts/convert_voice.py), and upload them to the
# adreno-llms weights repo under kokoro-82m/voices/<name>.bin.
#
# Run after `hf auth login` (write-scoped token). Public download, authed upload.
# Idempotent — re-run to refresh / resume. ~54 voices × ~510 KB ≈ 28 MB each way.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC_REPO="${SRC_REPO:-hexgrad/Kokoro-82M}"
HF_REPO="${HF_REPO:-a8nova/adreno-llms-weights}"
PY="${PY:-/Users/alazarshenkute/Projects/nnopt/venv/bin/python}"

command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' CLI not found (pip install -U 'huggingface_hub[cli]')" >&2; exit 1; }
"$PY" -c "import torch, numpy" 2>/dev/null || { echo "ERROR: $PY lacks torch/numpy" >&2; exit 1; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
BINS="$WORK/voices_bin"; mkdir -p "$BINS"

echo ">>> [1/3] download voices/*.pt from $SRC_REPO"
hf download "$SRC_REPO" --include "voices/*.pt" --local-dir "$WORK/src" >/dev/null
n=$(ls "$WORK/src/voices/"*.pt 2>/dev/null | wc -l | tr -d ' ')
echo "    got $n voice .pt files"
[ "$n" -gt 0 ] || { echo "ERROR: no voices downloaded" >&2; exit 1; }

echo ">>> [2/3] convert .pt -> .bin"
for pt in "$WORK/src/voices/"*.pt; do
  name="$(basename "$pt" .pt)"
  "$PY" scripts/convert_voice.py "$pt" "$BINS/$name.bin"
done

echo ">>> [3/3] upload to $HF_REPO/kokoro-82m/voices/"
hf auth whoami >/dev/null 2>&1 || { echo "ERROR: not logged in — run: hf auth login" >&2; exit 1; }
hf upload "$HF_REPO" "$BINS" kokoro-82m/voices

code=$(curl -sI -L -o /dev/null -w "%{http_code}" "https://huggingface.co/$HF_REPO/resolve/main/kokoro-82m/voices/af_heart.bin")
echo ""
echo "Done — $n voices at $HF_REPO/kokoro-82m/voices/  (af_heart.bin HEAD: $code)"
