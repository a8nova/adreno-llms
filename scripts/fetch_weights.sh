#!/usr/bin/env bash
# Download all weights/ files for one model (or all models) from the public
# HuggingFace mirror via curl. No huggingface-cli / hf_hub dependency.
# Supports resume and an HF_REPO override for forks.
#
# Usage:
#   ./scripts/fetch_weights.sh <model-name>
#   ./scripts/fetch_weights.sh all
#
# Models:
#   lfm2-5-350m  mamba-130m  mamba2-130m  qwen2-5-0-5b  smollm2-135m-instruct
#
# Per model, this fetches every file the runtime needs:
#   weights/model.fp16.bin
#   weights/model.fp16.meta.json
#   weights/tokenizer.json
#   weights/tokenizer_vocab.bin
#
# OpenELM is NOT here: Apple's ASCL forbids redistribution. Use
# scripts/fetch_openelm_weights.sh which pulls from apple/OpenELM-270M
# directly and converts locally.

set -euo pipefail

HF_REPO="${HF_REPO:-a8nova/adreno-llms-weights}"
HF_BRANCH="${HF_BRANCH:-main}"
HF_BASE="https://huggingface.co/${HF_REPO}/resolve/${HF_BRANCH}"

MODELS=(lfm2-5-350m mamba-130m mamba2-130m qwen2-5-0-5b smollm2-135m-instruct)
WEIGHT_FILES=(model.fp16.bin model.fp16.meta.json tokenizer.json tokenizer_vocab.bin)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<EOF
Usage: $0 <model-name | all>

Models: ${MODELS[*]}

Per-model files fetched: ${WEIGHT_FILES[*]}

Environment:
  HF_REPO   override the HuggingFace repo (default: a8nova/adreno-llms-weights)
  HF_BRANCH override the revision (default: main)
EOF
  exit 1
}

[ $# -eq 1 ] || usage

fetch_one() {
  local model="$1"

  if [ ! -d "${REPO_ROOT}/src/models/${model}" ]; then
    echo "ERROR: unknown model '${model}'" >&2
    return 1
  fi

  local weights_dir="${REPO_ROOT}/src/models/${model}/weights"
  mkdir -p "${weights_dir}"

  echo ">>> ${model}"
  for f in "${WEIGHT_FILES[@]}"; do
    local dest="${weights_dir}/${f}"
    local url="${HF_BASE}/${model}/${f}"
    echo "    ${f}"
    curl --location --continue-at - --fail-with-body --progress-bar \
         --retry 3 --retry-delay 5 \
         --output "${dest}" "${url}"
  done
  echo "    done — total $(du -sh "${weights_dir}" | awk '{print $1}')"
}

target="$1"
if [ "${target}" = "all" ]; then
  for m in "${MODELS[@]}"; do fetch_one "${m}"; done
else
  found=""
  for m in "${MODELS[@]}"; do
    if [ "${m}" = "${target}" ]; then found="1"; break; fi
  done
  if [ -z "${found}" ]; then
    echo "ERROR: unknown model '${target}'" >&2
    echo "Known models: ${MODELS[*]}" >&2
    exit 1
  fi
  fetch_one "${target}"
fi

echo ""
echo "All requested weights downloaded."
