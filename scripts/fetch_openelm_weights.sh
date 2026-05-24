#!/usr/bin/env bash
# Fetch + convert OpenELM-270M weights for adreno-llms.
#
# Apple's ASCL forbids redistribution of OpenELM weights, so we don't host
# them in the project's HuggingFace repo. We pull safetensors + tokenizer.json
# directly from apple/OpenELM-270M and pack them into the nnopt fp16 layout
# via scripts/convert_weights.py.
#
# Run from anywhere; paths are resolved relative to the repo root.
#
# Override:
#   APPLE_REPO     upstream weights (default: apple/OpenELM-270M)

set -euo pipefail

APPLE_REPO="${APPLE_REPO:-apple/OpenELM-270M}"
APPLE_BRANCH="${APPLE_BRANCH:-main}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WEIGHTS_DIR="${REPO_ROOT}/src/models/openelm-270m/weights"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found in PATH" >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/convert_weights.py" \
  --hf-repo-id "${APPLE_REPO}" \
  --hf-branch  "${APPLE_BRANCH}" \
  --out-dir    "${WEIGHTS_DIR}"

echo ""
echo "Done. weights/ contents:"
ls -lh "${WEIGHTS_DIR}/" | grep -v '^total\|\.gitkeep$'
