#!/bin/bash
# Multi-prompt e2e device sweep: for each prompt asset set, push conditioning +
# noise to the device, run the FULL pipeline on-device (8-step DiT + VAE + WAV),
# collect BENCHMARK numbers and the generated clip. 120s cooldown between runs
# so per-prompt numbers are thermally comparable.
set -e
cd "$(dirname "$0")/.."
. ./scripts/remote_dir.env

ADB="${ADB:-adb}"
OUT="${1:-multi_prompt_results}"
COOLDOWN="${COOLDOWN:-120}"
mkdir -p "$OUT"

for d in multi_prompts/p*/; do
    name=$(basename "$d")
    echo "=== $name ==="
    $ADB push "$d"cross_attn_cond.bin "$d"global_embed.bin "$d"init_noise.bin \
              "$d"sigmas.bin "$d"step_noise_"{0,1,2,3,4,5,6,7}".bin \
              $REMOTE_DIR/assets/ > /dev/null 2>&1 || {
        # brace expansion may not survive some adb versions — push individually
        for f in "$d"*.bin; do $ADB push "$f" $REMOTE_DIR/assets/ > /dev/null; done
    }
    sleep "$COOLDOWN"
    NNOPT_DTYPE=fp16 ./scripts/run_android.sh "multi-prompt e2e" 8 \
        > "$OUT/$name.log" 2>&1 || true
    grep -E "BENCHMARK (dit_total|decoder|total_inference)" "$OUT/$name.log" \
        | sed "s/^/[$name] /"
    $ADB pull $REMOTE_DIR/output.wav "$OUT/$name.wav" > /dev/null 2>&1
done
echo "SWEEP DONE — results in $OUT/"
