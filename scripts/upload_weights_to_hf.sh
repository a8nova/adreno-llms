#!/usr/bin/env bash
# Upload all model weights to HuggingFace.
#
# Prerequisites:
#   1. Run `hf auth login` (or legacy `huggingface-cli login`) FIRST and paste
#      a write-scoped token. This script will NOT attempt to log in itself.
#   2. Each model directory under src/models/<m>/weights/ must contain its
#      expected files. Most models use the 4-file set (model.fp16.bin,
#      model.fp16.meta.json, tokenizer.json, tokenizer_vocab.bin); whisper-tiny
#      uses a 3-file set (no tokenizer.json — its runtime reads only
#      tokenizer_vocab.bin). The script validates this before touching HF.
#
# What it uploads (to HF repo a8nova/adreno-llms-weights by default):
#   <model>/model.fp16.bin
#   <model>/model.fp16.meta.json
#   <model>/tokenizer.json          (not for whisper-tiny)
#   <model>/tokenizer_vocab.bin
#   for each of: granite-4-0-350m, lfm2-5-350m, lfm2-5-vl-450m, mamba-130m,
#                mamba2-130m, qwen2-5-0-5b, smollm2-135m-instruct, whisper-tiny,
#                kokoro-82m, musicgen-small, seamless-m4t-unity-small,
#                openelm-270m (companion files only)
#   plus a top-level README.md sourced from scripts/hf_repo_README.md
#
# OpenELM's model.fp16.bin is intentionally skipped — Apple's ASCL forbids
# redistribution (only its companion files are uploaded).
#
# Override: set HF_REPO to upload elsewhere (e.g. for a fork).

set -euo pipefail

HF_REPO="${HF_REPO:-a8nova/adreno-llms-weights}"
MODELS=(granite-4-0-350m lfm2-5-350m lfm2-5-vl-450m mamba-130m mamba2-130m qwen2-5-0-5b smollm2-135m-instruct whisper-tiny kokoro-82m pocket-tts musicgen-small seamless-m4t-unity-small openelm-270m openvoice-v2)
# Push only the model(s) named as args (e.g. `upload_weights_to_hf.sh pocket-tts`); default = all.
[ $# -ge 1 ] && MODELS=("$@")
WEIGHT_FILES=(model.fp16.bin model.fp16.meta.json tokenizer.json tokenizer_vocab.bin)
# OpenELM weights are NOT redistributable (Apple ASCL). For openelm-270m we
# upload only the 3 small files (meta + tokenizer); fetch_openelm_weights.sh
# pulls the actual weights from apple/OpenELM-270M on demand and converts.
OPENELM_WEIGHT_FILES=(model.fp16.meta.json tokenizer.json tokenizer_vocab.bin)
# whisper-tiny is ASR (encoder-decoder), not a causal LM. Its runtime loads
# tokenizer_vocab.bin directly and never reads a tokenizer.json, so its weight
# set is 3 files — the full model.fp16.bin IS redistributable (Apache 2.0).
WHISPER_WEIGHT_FILES=(model.fp16.bin model.fp16.meta.json tokenizer_vocab.bin)
# musicgen-small (text→music) and seamless-m4t-unity-small (speech translation)
# use the same 3-file set. kokoro-82m (TTS) phonemizes via espeak assets and
# openvoice-v2 (voice cloning) is audio-to-audio — neither needs a tokenizer,
# just model + meta.
KOKORO_WEIGHT_FILES=(model.fp16.bin model.fp16.meta.json)
# pocket-tts (TTS): model + tokenizer_vocab + the 8 selectable v1 voices (raw audio_prompt
# the runtime primes). The v3 voices are NOT shipped — their pre-computed KV was made by a
# different model snapshot and is silent on tts_b6369a24 (verified).
POCKET_WEIGHT_FILES=(model.fp16.bin model.fp16.meta.json tokenizer_vocab.bin \
  voices/alba.fp16.bin voices/azelma.fp16.bin voices/cosette.fp16.bin voices/eponine.fp16.bin \
  voices/fantine.fp16.bin voices/javert.fp16.bin voices/jean.fp16.bin voices/marius.fp16.bin)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── pick whichever CLI the user has installed ────────────────────────────
if command -v hf >/dev/null 2>&1; then
  HF=hf
  HF_WHOAMI=( hf auth whoami )
  HF_REPO_CREATE=( hf repo create "$(basename "$HF_REPO")" --repo-type model -y )
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=huggingface-cli
  HF_WHOAMI=( huggingface-cli whoami )
  HF_REPO_CREATE=( huggingface-cli repo create "$(basename "$HF_REPO")" --type model --yes )
else
  echo "ERROR: neither 'hf' nor 'huggingface-cli' found in PATH." >&2
  echo "       Install with: pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi
echo "Using CLI: $HF"
echo "Target HF repo: $HF_REPO"
echo ""

# ── Step 1: auth ─────────────────────────────────────────────────────────
echo ">>> [1/5] Verifying auth"
if ! "${HF_WHOAMI[@]}" >/dev/null 2>&1; then
  echo "ERROR: not logged in to HuggingFace." >&2
  echo "       Run: $HF auth login   (or: huggingface-cli login)" >&2
  exit 1
fi
WHOAMI=$( "${HF_WHOAMI[@]}" 2>/dev/null | head -1 )
echo "    logged in as: $WHOAMI"
echo ""

# ── Step 2: validate local files ─────────────────────────────────────────
echo ">>> [2/5] Validating local weights/ directories"
missing=0
expected=0
for m in "${MODELS[@]}"; do
  if [ "$m" = "openelm-270m" ]; then
    files=("${OPENELM_WEIGHT_FILES[@]}")
  elif [ "$m" = "kokoro-82m" ] || [ "$m" = "openvoice-v2" ]; then
    files=("${KOKORO_WEIGHT_FILES[@]}")
  elif [ "$m" = "pocket-tts" ]; then
    files=("${POCKET_WEIGHT_FILES[@]}")     # model + meta + tokenizer_vocab + 8 v1 voices
  elif [ "$m" = "whisper-tiny" ] || [ "$m" = "musicgen-small" ] || [ "$m" = "seamless-m4t-unity-small" ]; then
    files=("${WHISPER_WEIGHT_FILES[@]}")   # model.fp16.bin + meta + tokenizer_vocab.bin
  else
    files=("${WEIGHT_FILES[@]}")
  fi
  for f in "${files[@]}"; do
    p="${REPO_ROOT}/src/models/${m}/weights/${f}"
    expected=$((expected + 1))
    if [ ! -f "$p" ]; then
      echo "    MISSING: $p" >&2
      missing=$((missing + 1))
    fi
  done
done
if [ "$missing" -gt 0 ]; then
  echo "ERROR: $missing local weight files missing — populate them before uploading." >&2
  exit 1
fi
echo "    all ${expected} local files present."
echo ""

# ── Step 3: create the HF repo (idempotent) ──────────────────────────────
echo ">>> [3/5] Creating HF repo $HF_REPO (idempotent — ok if it already exists)"
"${HF_REPO_CREATE[@]}" 2>&1 | sed 's/^/    /' || echo "    (repo likely already exists, continuing)"
echo ""

# Local folder name -> HF repo subdir. Instruct ports keep generic local folder
# names but live under a -instruct path on HF for clarity (must match
# fetch_weights.sh's hf_subdir_for).
hf_subdir_for() {
  case "$1" in
    qwen2-5-0-5b)  echo "qwen2-5-0-5b-instruct" ;;
    lfm2-5-350m)   echo "lfm2-5-350m-instruct" ;;
    openelm-270m)  echo "openelm-270m-instruct" ;;
    *)             echo "$1" ;;
  esac
}

# ── Step 4: upload each model's weights/ ─────────────────────────────────
echo ">>> [4/5] Uploading weights"
for m in "${MODELS[@]}"; do
  echo "    --- $m ---"
  src_dir="${REPO_ROOT}/src/models/${m}/weights"
  # `hf upload <repo> <local-folder> <repo-prefix>` uploads recursively.
  # Always exclude .gitkeep. For openelm-270m, also exclude model.fp16.bin
  # (Apple ASCL forbids redistribution — users fetch + convert locally).
  exclude_args=( --exclude ".gitkeep" )
  if [ "$m" = "openelm-270m" ]; then
    exclude_args+=( --exclude "model.fp16.bin" --exclude "model.bin" )
  fi
  if [ "$m" = "pocket-tts" ]; then
    # ship only fp16 model + tokenizer + 8 v1 voices; drop fp32 model, fp32 meta,
    # the cached voice_kv, and voices_v3/ (incompatible KV — silent on our model).
    exclude_args+=( --exclude "model.bin" --exclude "model.meta.json" \
                    --exclude "voice_kv.bin" --exclude "voices_v3/*" )
  fi
  "$HF" upload "$HF_REPO" "$src_dir" "$(hf_subdir_for "$m")" "${exclude_args[@]}"
done
echo ""

# ── Step 5: upload the HF repo README ────────────────────────────────────
echo ">>> [5/5] Uploading HF repo README.md"
README_SRC="${SCRIPT_DIR}/hf_repo_README.md"
if [ ! -f "$README_SRC" ]; then
  echo "WARN: $README_SRC missing — skipping README upload." >&2
else
  "$HF" upload "$HF_REPO" "$README_SRC" "README.md"
fi
echo ""

# ── Verify ───────────────────────────────────────────────────────────────
echo ">>> Verifying public URLs (head request on mamba-130m, smallest)"
all_ok=1
for f in "${WEIGHT_FILES[@]}"; do
  url="https://huggingface.co/${HF_REPO}/resolve/main/mamba-130m/${f}"
  status=$(curl --head --location --silent -o /dev/null -w "%{http_code}" "$url")
  if [ "$status" = "200" ]; then
    echo "    200 OK   $f"
  else
    echo "    HTTP $status  $f  ($url)"
    all_ok=0
  fi
done
echo ""

if [ "$all_ok" = "1" ]; then
  echo "✓ Upload complete. Public URLs resolve."
  echo ""
  echo "End-to-end fetch test (optional):"
  echo "  cd ~/Projects/adreno-llms"
  echo "  mkdir -p /tmp/mamba-130m-backup"
  echo "  mv src/models/mamba-130m/weights/{model.fp16.bin,model.fp16.meta.json,tokenizer.json,tokenizer_vocab.bin} /tmp/mamba-130m-backup/"
  echo "  ./scripts/fetch_weights.sh mamba-130m"
  echo "  diff -r src/models/mamba-130m/weights/ /tmp/mamba-130m-backup/   # should be empty"
else
  echo "WARN: some URLs not yet resolving — HF may take ~30s to propagate. Re-run the verify step in a moment."
  exit 1
fi
