#!/usr/bin/env bash
# Download all weights/ files for one model (or all models) from the public
# HuggingFace mirror via curl. No huggingface-cli / hf_hub dependency.
# Supports resume and an HF_REPO override for forks.
#
# Usage:
#   ./scripts/fetch_weights.sh <model-name> [--quant fp16|int8|q4]
#   ./scripts/fetch_weights.sh all          [--quant fp16|int8|q4]
#
# Models:
#   granite-4-0-350m  lfm2-5-350m  lfm2-5-vl-450m  mamba-130m  mamba2-130m
#   qwen2-5-0-5b  smollm2-135m-instruct  whisper-tiny  kokoro-82m
#   musicgen-small  seamless-m4t-unity-small  openvoice-v2
#
# Per model, this fetches the base set the runtime needs:
#   weights/model.fp16.bin
#   weights/model.fp16.meta.json
#   weights/tokenizer.json          (not for whisper-tiny — ASR has no tokenizer.json)
#   weights/tokenizer_vocab.bin
#
# Optional `--quant int8` also fetches model.int8.bin + model.int8.meta.json
# (currently available for lfm2-5-350m, lfm2-5-vl-450m, and smollm2-135m-instruct).
#
# Optional `--quant q4` also fetches model.q4.bin + model.q4.meta.json
# (currently available for lfm2-5-350m only).
#
# The runtime picks which weights to load at run-time via NNOPT_QUANT=int8|q4
# (see each model's README). Having multiple bundles side-by-side on disk is
# fine — they don't conflict.
#
# OpenELM is NOT here: Apple's ASCL forbids redistribution. Use
# scripts/fetch_openelm_weights.sh which pulls from apple/OpenELM-270M
# directly and converts locally.

set -euo pipefail

HF_REPO="${HF_REPO:-a8nova/adreno-llms-weights}"
HF_BRANCH="${HF_BRANCH:-main}"
HF_BASE="https://huggingface.co/${HF_REPO}/resolve/${HF_BRANCH}"


MODELS=(granite-4-0-350m lfm2-5-350m lfm2-5-vl-450m mamba-130m mamba2-130m qwen2-5-0-5b smollm2-135m-instruct whisper-tiny kokoro-82m musicgen-small seamless-m4t-unity-small openvoice-v2)
BASE_FILES=(model.fp16.bin model.fp16.meta.json tokenizer.json tokenizer_vocab.bin)
# whisper-tiny (ASR), musicgen-small (text→music) and seamless-m4t-unity-small
# (speech translation) load tokenizer_vocab.bin directly and have no
# tokenizer.json, so their base set is 3 files.
WHISPER_BASE_FILES=(model.fp16.bin model.fp16.meta.json tokenizer_vocab.bin)
# kokoro-82m (TTS) phonemizes via espeak (assets) and openvoice-v2 (voice
# cloning) is audio-to-audio — neither needs a tokenizer, just model + meta.
KOKORO_BASE_FILES=(model.fp16.bin model.fp16.meta.json)

# Which models currently have which quant variants published on HF. Update when
# new quant bundles are uploaded.
MODELS_WITH_INT8=(lfm2-5-350m lfm2-5-vl-450m smollm2-135m-instruct)
MODELS_WITH_Q4=(lfm2-5-350m)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<EOF
Usage: $0 <model-name | all> [--quant fp16|int8|q4]

Models: ${MODELS[*]}

Default fetch set (always pulled): ${BASE_FILES[*]}

--quant int8 → additionally pulls model.int8.bin + model.int8.meta.json
              (available: ${MODELS_WITH_INT8[*]})
--quant q4   → additionally pulls model.q4.bin + model.q4.meta.json
              (available: ${MODELS_WITH_Q4[*]})
--quant fp16 → default behavior (no extra files)

Environment:
  HF_REPO   override the HuggingFace repo (default: a8nova/adreno-llms-weights)
  HF_BRANCH override the revision (default: main)
EOF
  exit 1
}

[ $# -ge 1 ] || usage

target="$1"
shift

QUANT="fp16"
while [ $# -gt 0 ]; do
  case "$1" in
    --quant)
      QUANT="${2:-}"
      shift 2
      ;;
    --quant=*)
      QUANT="${1#--quant=}"
      shift
      ;;
    *)
      echo "ERROR: unknown arg '$1'" >&2
      usage
      ;;
  esac
done

case "${QUANT}" in
  fp16|int8|q4) ;;
  *) echo "ERROR: --quant must be one of fp16 / int8 / q4 (got '${QUANT}')" >&2; usage ;;
esac

# Return 0 if the given model is in the given array, 1 otherwise.
_in_array() {
  local needle="$1"; shift
  for m in "$@"; do
    [ "${m}" = "${needle}" ] && return 0
  done
  return 1
}

# Build the per-model file list based on --quant.
file_list_for() {
  local model="$1"
  local files
  if [ "${model}" = "kokoro-82m" ] || [ "${model}" = "openvoice-v2" ]; then
    files=("${KOKORO_BASE_FILES[@]}")
  elif [ "${model}" = "whisper-tiny" ] || [ "${model}" = "musicgen-small" ] || [ "${model}" = "seamless-m4t-unity-small" ]; then
    files=("${WHISPER_BASE_FILES[@]}")
  else
    files=("${BASE_FILES[@]}")
  fi
  if [ "${QUANT}" = "int8" ]; then
    if _in_array "${model}" "${MODELS_WITH_INT8[@]}"; then
      files+=(model.int8.bin model.int8.meta.json)
    else
      echo "    note: ${model} has no published int8 bundle; fetching fp16 only" >&2
    fi
  elif [ "${QUANT}" = "q4" ]; then
    if _in_array "${model}" "${MODELS_WITH_Q4[@]}"; then
      files+=(model.q4.bin model.q4.meta.json)
    else
      echo "    note: ${model} has no published q4 bundle; fetching fp16 only" >&2
    fi
  fi
  printf '%s\n' "${files[@]}"
}

# Local folder name -> HF repo subdir. The instruct ports keep their generic
# local folder names but live under a -instruct path on HF for clarity.
hf_subdir_for() {
  case "$1" in
    qwen2-5-0-5b)  echo "qwen2-5-0-5b-instruct" ;;
    lfm2-5-350m)   echo "lfm2-5-350m-instruct" ;;
    openelm-270m)  echo "openelm-270m-instruct" ;;
    *)             echo "$1" ;;
  esac
}

fetch_one() {
  local model="$1"

  if [ ! -d "${REPO_ROOT}/src/models/${model}" ]; then
    echo "ERROR: unknown model '${model}'" >&2
    return 1
  fi

  local weights_dir="${REPO_ROOT}/src/models/${model}/weights"
  mkdir -p "${weights_dir}"

  echo ">>> ${model} (--quant ${QUANT})"
  while IFS= read -r f; do
    local dest="${weights_dir}/${f}"
    local url="${HF_BASE}/$(hf_subdir_for "${model}")/${f}"
    echo "    ${f}"
    curl --location --continue-at - --fail-with-body --progress-bar \
         --retry 3 --retry-delay 5 \
         --output "${dest}" "${url}"
  done < <(file_list_for "${model}")
  echo "    done — total $(du -sh "${weights_dir}" | awk '{print $1}')"
}

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
