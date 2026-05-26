#!/usr/bin/env bash
# bench.sh — locked benchmark harness for the Razr 2020 / Adreno 620 port.
#
# Usage:
#   scripts/bench.sh "<label>" [extra notes ...]
#
# What it does:
#   1. Confirms the device is connected (ZY22D5NLGQ by default; override via DEVICE).
#   2. Builds + deploys the fp16 binary (skip with SKIP_BUILD=1, SKIP_DEPLOY=1).
#   3. Runs the inference 3 times capturing the full profile output.
#   4. Picks the run with the median RTF as the representative measurement.
#   5. Appends a new dated section near the top of BENCHMARK.md.
#   6. Saves raw run logs under .bench/ for forensics.
#
# Required: the binary must already print the standard
#   `=== KERNEL PROFILE ===` / `=== GPU TIMELINE ===` / `BENCHMARK …` blocks.

set -euo pipefail

cd "$(dirname "$0")/.."   # port root

LABEL="${1:-unlabeled}"
shift || true
EXTRA_NOTES="$*"

DEVICE="${DEVICE:-ZY22D5NLGQ}"
PROMPT='Hello, my name is'
DTYPE="${NNOPT_DTYPE:-fp16}"
BIN_SUFFIX=""
[ "$DTYPE" = "fp16" ] && BIN_SUFFIX="_fp16"
BINARY="mms_tts_eng_inference${BIN_SUFFIX}"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/mms_tts_eng_inference}"

BENCH_DIR=".bench"
mkdir -p "$BENCH_DIR"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
RUN_DIR="$BENCH_DIR/$TS-$LABEL"
mkdir -p "$RUN_DIR"

COMMIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
COMMIT_SUBJ="$(git log -1 --format='%s' 2>/dev/null || echo '')"
DATE_HUMAN="$(date -u +%Y-%m-%d)"

echo "==> bench.sh label=$LABEL commit=$COMMIT_SHA device=$DEVICE dtype=$DTYPE"

# Device check
if ! adb devices | grep -q "^${DEVICE}[[:space:]]*device$"; then
    echo "ERROR: device $DEVICE not connected. adb devices says:" >&2
    adb devices >&2
    exit 2
fi

# Build
if [ "${SKIP_BUILD:-0}" != "1" ]; then
    echo "==> build"
    NNOPT_DTYPE="$DTYPE" bash scripts/build.sh > "$RUN_DIR/build.log" 2>&1 || {
        echo "ERROR: build failed; see $RUN_DIR/build.log" >&2
        tail -30 "$RUN_DIR/build.log" >&2
        exit 3
    }
fi

# Deploy
if [ "${SKIP_DEPLOY:-0}" != "1" ]; then
    echo "==> deploy"
    NNOPT_DTYPE="$DTYPE" bash scripts/deploy_android.sh > "$RUN_DIR/deploy.log" 2>&1 || {
        echo "ERROR: deploy failed; see $RUN_DIR/deploy.log" >&2
        tail -20 "$RUN_DIR/deploy.log" >&2
        exit 4
    }
fi

run_once() {
    local idx="$1"
    local out="$RUN_DIR/run${idx}.log"
    adb -s "$DEVICE" shell "cd $REMOTE_DIR && NNOPT_PROFILE=1 LD_LIBRARY_PATH=lib:/system/vendor/lib64:\$LD_LIBRARY_PATH ./$BINARY '$PROMPT' 1" > "$out" 2>&1 || true
    grep -E '^BENCHMARK rtf:' "$out" | awk '{print $3}'
}

echo "==> 3 timed runs"
RTFS=()
for i in 1 2 3; do
    rtf="$(run_once "$i")"
    if [ -z "$rtf" ]; then
        echo "ERROR: run $i produced no BENCHMARK rtf line; see $RUN_DIR/run${i}.log" >&2
        tail -40 "$RUN_DIR/run${i}.log" >&2
        exit 5
    fi
    echo "    run$i RTF=$rtf"
    RTFS+=("$rtf:$i")
done

# Pick median run
MEDIAN_LINE="$(printf '%s\n' "${RTFS[@]}" | sort -n -t: -k1,1 | sed -n '2p')"
MEDIAN_RTF="${MEDIAN_LINE%%:*}"
MEDIAN_IDX="${MEDIAN_LINE##*:}"
MIN_RTF="$(printf '%s\n' "${RTFS[@]}" | sort -n -t: -k1,1 | head -1 | awk -F: '{print $1}')"
MAX_RTF="$(printf '%s\n' "${RTFS[@]}" | sort -n -t: -k1,1 | tail -1 | awk -F: '{print $1}')"
MEDIAN_LOG="$RUN_DIR/run${MEDIAN_IDX}.log"

echo "==> median RTF=$MEDIAN_RTF (min $MIN_RTF / max $MAX_RTF) — using run${MEDIAN_IDX}"

# Extract fields from the median run
grep_field() { grep -E "^$1" "$MEDIAN_LOG" | head -1; }
val_after()  { grep_field "$1" | awk '{print $NF}'; }

TOTAL_INF="$(val_after 'BENCHMARK total_inference_sec:')"
AUDIO_DUR="$(val_after 'BENCHMARK audio_duration_sec:')"
VOC_WALL_MS="$(grep -oE 'PHASE vocoder wall: [0-9.]+ ms' "$MEDIAN_LOG" | head -1 | awk '{print $4}')"
KERN_TOTAL="$(grep -oE 'TOTAL kernel runtime \(start->end\):\s+[0-9.]+ ms' "$MEDIAN_LOG" | head -1 | grep -oE '[0-9.]+ ms' | head -1 | awk '{print $1}')"
SPAN_MS="$(grep -oE 'span \(first kernel start -> last kernel end\): [0-9.]+ ms' "$MEDIAN_LOG" | head -1 | awk '{print $(NF-1)}')"
BUSY_PCT="$(grep -oE 'kernel-busy fraction:\s+[0-9.]+%' "$MEDIAN_LOG" | head -1 | grep -oE '[0-9.]+')"
IDLE_MS="$(grep -oE 'total inter-kernel idle \(sum of gaps\):\s+[0-9.]+ ms' "$MEDIAN_LOG" | head -1 | grep -oE '[0-9.]+ ms' | head -1 | awk '{print $1}')"
GAPS_N="$(grep -oE 'across [0-9]+ gaps' "$MEDIAN_LOG" | head -1 | awk '{print $2}')"
AVG_GAP="$(grep -oE 'avg gap:\s+[0-9.]+ ms' "$MEDIAN_LOG" | head -1 | awk '{print $3}')"
PEAK_MEM="$(val_after 'BENCHMARK peak_cpu_memory_mb:')"

# Top 3 kernels by kern_ms (the lines under === KERNEL PROFILE === until the dashes/blank)
TOP_KERNELS="$(awk '
    /=== KERNEL PROFILE/ {in_block=1; next}
    in_block && /^---/ {next}
    in_block && /^label/ {next}
    in_block && NF == 0 {in_block=0; next}
    in_block && /=== TOTAL/ {in_block=0; next}
    in_block { printf "%s %s\n", $1, $2 }
' "$MEDIAN_LOG" | head -3 | awk '{printf "%s%s %s ms · ", (NR>1?"":""), $1, $2; if (NR<3) printf ""}' | sed 's/ · $//')"

# Compose the new entry
NEW_ENTRY=$(cat <<EOF

## $DATE_HUMAN — $LABEL

- **commit:** $COMMIT_SHA  _"$COMMIT_SUBJ"_
- **RTF:** **$MEDIAN_RTF**  (variance over 3 runs: min $MIN_RTF · max $MAX_RTF)
- **wall total:** $TOTAL_INF s  (audio $AUDIO_DUR s)
- **vocoder phase wall:** $VOC_WALL_MS ms
- **kernel time total:** $KERN_TOTAL ms
- **GPU timeline span:** $SPAN_MS ms · **busy $BUSY_PCT %**
- **inter-kernel idle:** $IDLE_MS ms across $GAPS_N gaps · avg gap $AVG_GAP ms
- **top kernels:** $TOP_KERNELS
- **peak CPU memory:** $PEAK_MEM MB
EOF
)

if [ -n "$EXTRA_NOTES" ]; then
    NEW_ENTRY="$NEW_ENTRY
- **notes:** $EXTRA_NOTES"
fi

NEW_ENTRY="$NEW_ENTRY

(raw logs: \`$RUN_DIR/\`)

---
"

# Insert near the top of BENCHMARK.md, just after the first '---' separator (after the preamble).
ENTRY_FILE="$RUN_DIR/entry.md"
printf '%s\n' "$NEW_ENTRY" > "$ENTRY_FILE"

python3 - "$LABEL" "$ENTRY_FILE" <<'PYEOF'
import sys, pathlib
label, entry_path = sys.argv[1], sys.argv[2]
md = pathlib.Path("BENCHMARK.md")
text = md.read_text()
marker = "\n---\n"
i = text.find(marker)
if i < 0:
    raise SystemExit("ERROR: BENCHMARK.md preamble separator missing — expected a '\\n---\\n' line.")
insertion_point = i + len(marker)
new = pathlib.Path(entry_path).read_text()
md.write_text(text[:insertion_point] + new + text[insertion_point:])
print(f"==> appended entry: {label}")
PYEOF

echo "==> done. BENCHMARK.md updated."
