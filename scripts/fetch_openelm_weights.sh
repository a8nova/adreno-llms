#!/usr/bin/env bash
# Fetch + convert OpenELM-270M-Instruct weights for adreno-llms.
#
# Apple's ASCL forbids redistribution of OpenELM weights, so we don't host the
# model binary in the project's HuggingFace repo. We pull safetensors directly
# from apple/OpenELM-270M-Instruct and pack them into the nnopt fp16 layout via
# scripts/convert_hf_to_bin.py. The tokenizer companion files (tokenizer.json +
# tokenizer_vocab.bin) ARE redistributable and are pulled from the project repo.
#
# Run from anywhere; paths are resolved relative to the repo root.
#
# Override:
#   APPLE_REPO     upstream weights (default: apple/OpenELM-270M-Instruct)
#   HF_REPO        project repo for tokenizer companions
#                  (default: a8nova/adreno-llms-weights)

set -euo pipefail

APPLE_REPO="${APPLE_REPO:-apple/OpenELM-270M-Instruct}"
APPLE_BRANCH="${APPLE_BRANCH:-main}"
HF_REPO="${HF_REPO:-a8nova/adreno-llms-weights}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WEIGHTS_DIR="${REPO_ROOT}/src/models/openelm-270m/weights"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found in PATH" >&2
  exit 1
fi

mkdir -p "${WEIGHTS_DIR}"

# 1. Convert the model binary locally from Apple's repo (not redistributed).
python3 "${SCRIPT_DIR}/convert_hf_to_bin.py" \
  --hf-repo-id "${APPLE_REPO}" \
  --hf-branch  "${APPLE_BRANCH}" \
  --model-id   "${APPLE_REPO}" \
  --output-dir "${WEIGHTS_DIR}"

# 2. Pull the redistributable tokenizer companions from the project repo.
BASE_URL="https://huggingface.co/${HF_REPO}/resolve/main/openelm-270m-instruct"
for f in tokenizer.json tokenizer_vocab.bin; do
  echo "Fetching ${f} ..."
  curl -fsSL -o "${WEIGHTS_DIR}/${f}" "${BASE_URL}/${f}"
done

echo ""
echo "Done. weights/ contents:"
ls -lh "${WEIGHTS_DIR}/" | grep -v '^total\|\.gitkeep$'
