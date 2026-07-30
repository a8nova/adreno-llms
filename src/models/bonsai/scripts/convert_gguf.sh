#!/bin/bash
# GGUF (Q1_0) -> .nnb, for ANY Bonsai size. This is the whole conversion
# pipeline: gguf_parse.py (metadata + tensor table) -> convert_to_nnb.py
# (header + verbatim Q1 blob). Nothing here is size-specific — every
# hyperparameter is derived from the GGUF metadata block.
#
# Usage:
#   ./scripts/convert_gguf.sh weights/Bonsai-1.7B-Q1_0.gguf weights/bonsai1.7b.nnb
#   ./scripts/convert_gguf.sh --dry-run weights/Bonsai-27B-Q1_0.gguf
#
# --dry-run parses + prints the derived meta and stops before writing the blob;
# use it to check a new size is supported before spending the I/O.
#
# Side effects: writes manifest.json + tokenizer.json next to the GGUF. The
# tokenizer.json is PER SIZE (vocabularies differ across Bonsai sizes) — deploy
# the one produced alongside the .nnb you are shipping.

set -e

cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
DRY=""
if [ "$1" = "--dry-run" ]; then DRY="--dry-run"; shift; fi

GGUF="$1"
OUT="$2"

if [ -z "$GGUF" ]; then
    echo "usage: $0 [--dry-run] <model.gguf> [out.nnb]" >&2
    exit 1
fi
if [ ! -f "$GGUF" ]; then
    echo "ERROR: no such file: $GGUF" >&2
    exit 1
fi
if [ -z "$DRY" ] && [ -z "$OUT" ]; then
    echo "ERROR: <out.nnb> required unless --dry-run" >&2
    exit 1
fi

DIR="$(dirname "$GGUF")"

echo "==> parsing $GGUF"
$PY scripts/gguf_parse.py "$GGUF"

echo "==> converting"
$PY scripts/convert_to_nnb.py "$GGUF" "$DIR/manifest.json" \
    "$DIR/tokenizer.json" ${OUT:+"$OUT"} $DRY

if [ -z "$DRY" ]; then
    echo
    echo "wrote $OUT (+ $DIR/tokenizer.json for this size)"
    echo "deploy: BONSAI_NNB=$(basename "$OUT") ./scripts/deploy_android.sh"
fi
