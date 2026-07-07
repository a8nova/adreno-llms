#!/bin/bash
# One-shot arbitrary-prompt generation, FULLY ON DEVICE:
# T5 conditioning (device CPU) + 8-step DiT denoise + VAE decode (device GPU).
# The host only supplies seed noise (deterministic-replay methodology) and
# pulls the clip.
#
#   ./scripts/generate.sh "detuned synthwave arpeggio at dusk" [seed] [name] [seconds]
#
# Output: multi_prompt_results/<name>.wav + <name>.log
set -e
cd "$(dirname "$0")/.."
# Default to fp16 BEFORE sourcing remote_dir.env — it bakes NNOPT_DTYPE into
# BINARY_NAME, and the deployed binary is the fp16 one.
export NNOPT_DTYPE="${NNOPT_DTYPE:-fp16}"
. ./scripts/remote_dir.env

PROMPT="${1:?usage: generate.sh \"<prompt>\" [seed] [name] [seconds] [steps]}"
SEED="${2:-0}"
NAME="${3:-user_prompt}"
SECONDS_TOTAL="${4:-11}"
STEPS="${5:-8}"
ADB="${ADB:-adb}"
PY="${PY:-$HOME/.nnopt/ref_venvs/env_legacy/bin/python3}"
OUT="multi_prompt_results"
mkdir -p "$OUT"

echo "=== [1/3] Seed noise (host, ~2s — no model load) ==="
NOISE_DIR="multi_prompts/noise_seed${SEED}_s${STEPS}"
if [ ! -f "$NOISE_DIR/sigmas.bin" ]; then
    "$PY" reference/gen_noise.py "$NOISE_DIR" "$SEED" "$STEPS"
fi
for f in "$NOISE_DIR"/*.bin; do
    $ADB push "$f" "$REMOTE_DIR/assets/" > /dev/null
done

echo "=== [2/3] Full on-device run: T5 (CPU) + DiT + VAE (GPU) ==="
# Remove any previous output so a failed run can never hand back a stale clip.
$ADB shell "rm -f $REMOTE_DIR/output.wav"
set +e
./scripts/run_android.sh "$PROMPT" 8 --seconds "$SECONDS_TOTAL" 2>&1 \
    | tee "$OUT/$NAME.log" \
    | grep -E "BENCHMARK|conditioning|ERROR|error|not found"
RUN_EXIT=${PIPESTATUS[0]}
set -e
if [ "$RUN_EXIT" -ne 0 ]; then
    echo "DEVICE RUN FAILED (exit $RUN_EXIT) — full log: $OUT/$NAME.log" >&2
    exit "$RUN_EXIT"
fi

echo "=== [3/3] Pull clip ==="
$ADB pull "$REMOTE_DIR/output.wav" "$OUT/$NAME.wav" > /dev/null || {
    echo "No output.wav on device — run did not produce audio. Log: $OUT/$NAME.log" >&2
    exit 1
}
echo "DONE — $OUT/$NAME.wav  (prompt: \"$PROMPT\", seed $SEED, ${SECONDS_TOTAL}s)"
