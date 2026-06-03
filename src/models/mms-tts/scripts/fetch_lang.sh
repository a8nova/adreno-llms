#!/usr/bin/env bash
# Download pre-built MMS-TTS language packs from HuggingFace and unpack them
# into weights/<lang>/. This is the no-Python path to getting languages on
# device.
#
# Each pack is a ZIP whose entries are
#   weights/<code>/{model.fp16.bin, model.fp16.meta.json, tokenizer_vocab.bin}
# (see scripts/convert_all_languages.py), so unzipping from the port root drops
# them straight into the layout deploy_android.sh expects.
#
# You do NOT need prep_lang.py for normal synthesis: the on-device binary
# tokenizes your text and samples the VITS noise itself, so the weights above
# are all it needs. (prep_lang.py is still how you ADD a language that isn't
# published yet — see the README "Adding a language not yet in the dataset".)
#
# Non-Latin scripts (amh, ara, khm, tha, …) additionally need the uroman
# romanization tables:
#   ./scripts/fetch_uroman.sh
#
# Usage:
#   ./scripts/fetch_lang.sh eng amh ara
#
# Environment:
#   HF_REPO    override the HuggingFace repo (default: a8nova/adreno-llms-weights)
#   HF_BRANCH  override the revision (default: main)

set -euo pipefail

HF_REPO="${HF_REPO:-a8nova/adreno-llms-weights}"
HF_BRANCH="${HF_BRANCH:-main}"
HF_BASE="https://huggingface.co/${HF_REPO}/resolve/${HF_BRANCH}/mms-tts"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <lang-code> [lang-code ...]" >&2
  echo "  e.g. $0 eng amh ara" >&2
  echo "  Browse available codes: https://huggingface.co/${HF_REPO}/tree/main/mms-tts" >&2
  exit 1
fi

command -v unzip >/dev/null 2>&1 || { echo "ERROR: 'unzip' not found in PATH." >&2; exit 1; }

cd "${PORT_ROOT}"
tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

for code in "$@"; do
  zip="${tmpdir}/mms-tts-${code}.zip"
  url="${HF_BASE}/mms-tts-${code}.zip"
  echo ">>> ${code}"
  echo "    downloading ${url}"
  if ! curl --location --fail-with-body --progress-bar \
            --retry 3 --retry-delay 5 \
            --output "${zip}" "${url}"; then
    echo "    ERROR: no published pack for '${code}'." >&2
    echo "           Browse available codes: https://huggingface.co/${HF_REPO}/tree/main/mms-tts" >&2
    exit 1
  fi
  # Entries are weights/<code>/… — unzipping from the port root drops them in place.
  unzip -o -q "${zip}" -d "${PORT_ROOT}"
  echo "    done — weights/${code}/ ($(du -sh "weights/${code}" | awk '{print $1}'))"
done

echo ""
echo "All requested language packs downloaded."
echo "Non-Latin scripts (amh, ara, khm, tha, …) also need: ./scripts/fetch_uroman.sh"
