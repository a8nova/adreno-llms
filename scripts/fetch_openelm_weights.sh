#!/usr/bin/env bash
# Fetch + convert OpenELM-270M weights for adreno-llms.
#
# Apple's ASCL forbids redistribution of OpenELM weights, so we don't host
# them in the project's HuggingFace repo. Instead, this script:
#   1. Downloads the small files (tokenizer.json, tokenizer_vocab.bin,
#      model.fp16.meta.json) from the project's HF mirror.
#   2. Downloads model.safetensors directly from apple/OpenELM-270M (HF).
#   3. Runs scripts/convert_openelm_weights.py to produce model.fp16.bin
#      using the layout declared in our meta.json.
#   4. Cleans up the safetensors blob.
#
# Run from anywhere; paths are resolved relative to the repo root.
#
# Override:
#   HF_REPO        target HF mirror (default: a8nova/adreno-llms-weights)
#   APPLE_REPO     upstream weights (default: apple/OpenELM-270M)

set -euo pipefail

HF_REPO="${HF_REPO:-a8nova/adreno-llms-weights}"
HF_BRANCH="${HF_BRANCH:-main}"
APPLE_REPO="${APPLE_REPO:-apple/OpenELM-270M}"
APPLE_BRANCH="${APPLE_BRANCH:-main}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WEIGHTS_DIR="${REPO_ROOT}/src/models/openelm-270m/weights"
SAFETENSORS_TMP="$(mktemp -d)/openelm-270m.safetensors"
trap 'rm -rf "$(dirname "$SAFETENSORS_TMP")"' EXIT

mkdir -p "${WEIGHTS_DIR}"

echo ">>> [1/3] Fetching small files from ${HF_REPO}/openelm-270m"
for f in model.fp16.meta.json tokenizer.json tokenizer_vocab.bin; do
  url="https://huggingface.co/${HF_REPO}/resolve/${HF_BRANCH}/openelm-270m/${f}"
  echo "    ${f}"
  curl --location --continue-at - --fail-with-body --progress-bar \
       --retry 3 --retry-delay 5 \
       --output "${WEIGHTS_DIR}/${f}" "${url}"
done

echo ""
echo ">>> [2/3] Downloading apple/OpenELM-270M safetensors (subject to Apple ASCL)"
SAFE_URL="https://huggingface.co/${APPLE_REPO}/resolve/${APPLE_BRANCH}/model.safetensors"
echo "    ${SAFE_URL}"
curl --location --continue-at - --fail-with-body --progress-bar \
     --retry 3 --retry-delay 5 \
     --output "${SAFETENSORS_TMP}" "${SAFE_URL}"

echo ""
echo ">>> [3/3] Converting to nnopt fp16 layout"
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found in PATH" >&2
  exit 1
fi
python3 "${SCRIPT_DIR}/convert_openelm_weights.py" \
  --safetensors "${SAFETENSORS_TMP}" \
  --meta "${WEIGHTS_DIR}/model.fp16.meta.json" \
  --output "${WEIGHTS_DIR}/model.fp16.bin"

echo ""
echo "Done. weights/ contents:"
ls -lh "${WEIGHTS_DIR}/" | grep -v '^total\|\.gitkeep$'
