#!/bin/bash
#
# scripts/quality_check.sh — end-to-end VLM quality check on 10 different
# (image, prompt) pairs. For each pair:
#   1. Download the image to fixtures/qcheck/
#   2. Stage it as fixtures/sample.jpg
#   3. Run scripts/qcheck_generate_features.py to:
#        - apply the SmolVLM chat-template to the prompt (HF processor)
#        - process the image through the Idefics3 image pipeline (tile splitting + resize + normalize)
#        - run the VISION TOWER + CONNECTOR via the HF model with a forward hook
#        - dump:
#            reference/test_input_ids.bin             (int32 tokens, chat-template-expanded)
#            reference/layers/model_connector_output.bin  (fp32 image features)
#        - print the reference caption (the "ground truth" the C++ binary must match)
#   4. Push test_input_ids.bin + model_connector_output.bin + sample.jpg to the device.
#   5. Run the C++ binary on the device (it auto-loads the .bin files).
#   6. Capture and compare both outputs.
#
# Why does step 3 need Python?
#   The C++ binary's on-device vision pipeline (src/ops/vision_pipeline.cpp) is
#   single-tile only — it produces 576 image tokens. But SmolVLM uses Idefics3
#   multi-tile splitting (up to 13 tiles → 832 image tokens). The chat template
#   expands the image-placeholder count based on tile count, so we need the HF
#   processor to compute the right `input_ids` AND the per-image connector
#   features (832 × 576 fp32). Until the multi-tile pipeline is ported on-device,
#   this Python step is the bridge.
#
# Setup (one-time):
#   python3.11 -m venv /tmp/qcheck-venv
#   source /tmp/qcheck-venv/bin/activate
#   pip install transformers==4.48.3 torch pillow 'numpy<2'
#
# Usage:
#   ./scripts/quality_check.sh                                  # 10 prompts, current binary only
#   ./scripts/quality_check.sh --tokens=20                      # gen length per prompt (default 16)
#   ./scripts/quality_check.sh --baseline-bin=/tmp/old.fp16     # also run a saved baseline binary
#   ./scripts/quality_check.sh --skip-ref                       # don't run Python ref; use stale .bin (debug only)
#
# Output:
#   results/quality_<UTC>.log
#

set -eu
cd "$(dirname "$0")/.."

BASELINE_BIN=""
MAX_NEW_TOKENS=16
SKIP_REF=0
for arg in "$@"; do
  case "$arg" in
    --baseline-bin=*) BASELINE_BIN="${arg#--baseline-bin=}" ;;
    --tokens=*)       MAX_NEW_TOKENS="${arg#--tokens=}" ;;
    --skip-ref)       SKIP_REF=1 ;;
    -h|--help)
      sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p fixtures/qcheck results
RESULTS="results/quality_${STAMP}.log"

ADB="${ADB:-adb}"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/SmolVLM_256M_Instruct_inference}"
BIN_NAME="SmolVLM_256M_Instruct_inference_fp16"
VENV="${VENV:-/tmp/qcheck-venv}"

WANT_BASELINE=0
[ -n "$BASELINE_BIN" ] && WANT_BASELINE=1

if [ "$SKIP_REF" = "0" ] && [ ! -d "$VENV" ]; then
  echo "[qcheck] Python venv at $VENV doesn't exist. Set it up with:" >&2
  echo "         python3.11 -m venv $VENV" >&2
  echo "         source $VENV/bin/activate" >&2
  echo "         pip install transformers==4.48.3 torch pillow 'numpy<2'" >&2
  exit 1
fi

# ─────────────────────────────────────────────
# 10 (image_url, image_filename, prompt) triples. Diverse images via picsum
# (stable, free, no auth) + diverse prompts probing description, counting,
# Q&A, captioning, scene reasoning, color, mood, comparison, and content.
# ─────────────────────────────────────────────
TESTS=(
  "https://picsum.photos/seed/cat/640/480|01_cat.jpg|Describe this image in one sentence."
  "https://picsum.photos/seed/dog/640/480|02_dog.jpg|What is in this picture?"
  "https://picsum.photos/seed/city/640/480|03_city.jpg|Where do you think this photo was taken?"
  "https://picsum.photos/seed/food/640/480|04_food.jpg|What kind of scene is this?"
  "https://picsum.photos/seed/forest/640/480|05_forest.jpg|What dominant colors appear in this image?"
  "https://picsum.photos/seed/beach/640/480|06_beach.jpg|What is the mood of this image?"
  "https://picsum.photos/seed/mountain/640/480|07_mountain.jpg|Is this indoors or outdoors? Explain briefly."
  "https://picsum.photos/seed/flower/640/480|08_flower.jpg|What time of day does this image suggest?"
  "https://picsum.photos/seed/car/640/480|09_car.jpg|Count any visible objects in this image."
  "https://picsum.photos/seed/person/640/480|10_person.jpg|Caption this image as if for a travel blog."
)

# ─────────────────────────────────────────────
# Download images we don't already have.
# ─────────────────────────────────────────────
echo "[qcheck] staging 10 images into fixtures/qcheck/" >&2
for entry in "${TESTS[@]}"; do
  IFS='|' read -r url file _prompt <<<"$entry"
  out="fixtures/qcheck/$file"
  if [ ! -s "$out" ]; then
    echo "[qcheck] downloading $file" >&2
    if ! curl -sSL -A "qcheck/1.0" "$url" -o "$out"; then
      echo "[qcheck] WARN: failed to fetch $url — skipping $file" >&2
      rm -f "$out"
    fi
  fi
done

# Backup the original fixtures/sample.jpg so we can restore on exit.
if [ -f fixtures/sample.jpg ]; then
  cp fixtures/sample.jpg fixtures/.sample.jpg.qcheckbackup
fi
trap '[ -f fixtures/.sample.jpg.qcheckbackup ] && mv fixtures/.sample.jpg.qcheckbackup fixtures/sample.jpg' EXIT

# ─────────────────────────────────────────────
# Validate baseline binary if provided.
# ─────────────────────────────────────────────
if [ "$WANT_BASELINE" = "1" ] && [ ! -x "$BASELINE_BIN" ]; then
  echo "[qcheck] --baseline-bin path is not executable: $BASELINE_BIN" >&2
  exit 1
fi

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
push_to_device() {
  $ADB push -q fixtures/sample.jpg "$REMOTE_DIR/fixtures/sample.jpg" >/dev/null 2>&1 || \
    $ADB push    fixtures/sample.jpg "$REMOTE_DIR/fixtures/sample.jpg" >/dev/null
  $ADB push -q reference/test_input_ids.bin "$REMOTE_DIR/reference/test_input_ids.bin" >/dev/null 2>&1 || \
    $ADB push    reference/test_input_ids.bin "$REMOTE_DIR/reference/test_input_ids.bin" >/dev/null
  $ADB push -q reference/layers/model_connector_output.bin "$REMOTE_DIR/reference/layers/model_connector_output.bin" >/dev/null 2>&1 || \
    $ADB push    reference/layers/model_connector_output.bin "$REMOTE_DIR/reference/layers/model_connector_output.bin" >/dev/null
}

run_python_reference() {
  local image="$1" prompt="$2"
  # Stage the image at fixtures/sample.jpg (the Python helper reads this path
  # only implicitly via the explicit argv — but the C++ binary always reads
  # fixtures/sample.jpg, so we keep them in sync).
  cp "$image" fixtures/sample.jpg
  # shellcheck disable=SC1091
  "$VENV/bin/python3" scripts/qcheck_generate_features.py "$image" "$prompt" "$MAX_NEW_TOKENS" 2>/dev/null | tail -1
}

run_binary_on_device() {
  $ADB shell "cd '$REMOTE_DIR' && LD_LIBRARY_PATH=lib:\$LD_LIBRARY_PATH ./$BIN_NAME 'Describe this image.' $MAX_NEW_TOKENS --image fixtures/sample.jpg 2>/dev/null"
}

push_baseline_binary() {
  $ADB push -q "$BASELINE_BIN" "$REMOTE_DIR/$BIN_NAME" >/dev/null 2>&1 || \
    $ADB push    "$BASELINE_BIN" "$REMOTE_DIR/$BIN_NAME" >/dev/null
  $ADB shell "chmod +x '$REMOTE_DIR/$BIN_NAME'" >/dev/null
}

push_current_binary() {
  local local_bin="build/fp16/$BIN_NAME"
  if [ -x "$local_bin" ]; then
    $ADB push -q "$local_bin" "$REMOTE_DIR/$BIN_NAME" >/dev/null 2>&1 || \
      $ADB push    "$local_bin" "$REMOTE_DIR/$BIN_NAME" >/dev/null
    $ADB shell "chmod +x '$REMOTE_DIR/$BIN_NAME'" >/dev/null
  fi
}

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
{
  echo "VLM quality check — $STAMP"
  echo "max_new_tokens=$MAX_NEW_TOKENS  baseline=$WANT_BASELINE  skip_ref=$SKIP_REF"
  echo ""
} | tee "$RESULTS"

# ─────────────────────────────────────────────
# Run the matrix
# ─────────────────────────────────────────────
i=0
for entry in "${TESTS[@]}"; do
  i=$((i+1))
  IFS='|' read -r _url file prompt <<<"$entry"
  img="fixtures/qcheck/$file"
  [ -s "$img" ] || { echo "[qcheck] skipping missing $file" >&2; continue; }

  printf '%s\n'                                                          \
    "==================================================" \
    "# $i  image: $file"                                  \
    "    prompt: $prompt"                                 \
    "=================================================="   | tee -a "$RESULTS"

  # ── Generate per-image features via Python ref ──
  if [ "$SKIP_REF" = "0" ]; then
    echo "[reference (HF PyTorch)]"                                      | tee -a "$RESULTS"
    ref_text=$(run_python_reference "$img" "$prompt")
    printf '%s\n' "$ref_text"                                            | tee -a "$RESULTS"
    echo ""                                                              | tee -a "$RESULTS"
    push_to_device
  else
    cp "$img" fixtures/sample.jpg
    push_to_device
  fi

  # ── Baseline binary (saved pre-Round-5 binary) ──
  if [ "$WANT_BASELINE" = "1" ]; then
    push_baseline_binary
    echo "[baseline binary]"                                             | tee -a "$RESULTS"
    out=$(run_binary_on_device)
    printf '%s\n' "$out"                                                 | tee -a "$RESULTS"
    echo ""                                                              | tee -a "$RESULTS"
    push_current_binary
  fi

  # ── Current optimized binary ──
  echo "[current (Round-5 optimized binary)]"                            | tee -a "$RESULTS"
  out=$(run_binary_on_device)
  printf '%s\n' "$out"                                                   | tee -a "$RESULTS"
  echo ""                                                                | tee -a "$RESULTS"
done

echo ""                                                                  | tee -a "$RESULTS"
echo "Results saved to: $RESULTS"
