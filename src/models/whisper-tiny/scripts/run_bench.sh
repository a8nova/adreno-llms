#!/bin/bash
# Run all 10 benchmark clips end-to-end on device (raw audio -> transcript + RTF).
# Reads durations from bench/manifest.json; writes bench/results.tsv and prints a table.
cd "$(dirname "$0")/.."
. ./scripts/remote_dir.env

PY=/Users/alazarshenkute/Projects/nnopt/venv/bin/python
OUT=bench/results.tsv
: > "$OUT"
printf "id\tdur_s\tRTF\tproc_s\ttranscript\n" >> "$OUT"

# Pull (id, duration, hf_transcript) rows from the manifest (bash 3.2: no mapfile).
ROWS_FILE=$(mktemp)
"$PY" - > "$ROWS_FILE" <<'PY'
import json
for m in json.load(open("bench/manifest.json")):
    print(f'{m["id"]}\t{m["duration_s"]}\t{m["hf_transcript"]}')
PY

echo "=== Running $(wc -l < "$ROWS_FILE" | tr -d ' ') clips ==="
# Read the loop on FD 3 so the adb shell inside (which consumes stdin) can't eat
# the remaining rows — the bug that made only clip 0 run.
while IFS=$'\t' read -r id dur ref <&3; do
    [ -z "$id" ] && continue
    fn=$(printf "bench/sample_%02d_audio.bin" "$id")
    echo ""
    echo "--- clip $id  (dur=${dur}s)  ref='${ref:0:60}'"
    log=$(NNOPT_DTYPE=fp16 bash scripts/run_android.sh "transcribe" 128 --audio "$fn" --audio-seconds "$dur" 2>/dev/null)
    rtf=$(echo "$log" | grep -oE "RTF: [0-9.]+" | head -1 | awk '{print $2}')
    proc=$(echo "$log" | grep -oE "proc=[0-9.]+s" | head -1 | tr -dc '0-9.')
    txt=$(echo "$log" | grep "GENERATED_TEXT:" | sed 's/GENERATED_TEXT: //')
    printf "  RTF=%s  got='%s'\n" "$rtf" "${txt:0:70}"
    printf "%s\t%s\t%s\t%s\t%s\n" "$id" "$dur" "$rtf" "$proc" "$txt" >> "$OUT"
done 3< "$ROWS_FILE"
rm -f "$ROWS_FILE"

echo ""
echo "=== SUMMARY (bench/results.tsv) ==="
"$PY" - <<'PY'
import csv
rows=list(csv.DictReader(open("bench/results.tsv"),delimiter="\t"))
rtfs=[float(r["RTF"]) for r in rows if r["RTF"]]
tot_dur=sum(float(r["dur_s"]) for r in rows)
tot_proc=sum(float(r["proc_s"]) for r in rows if r["proc_s"])
print(f"clips={len(rows)}  audio_total={tot_dur:.1f}s  proc_total={tot_proc:.1f}s")
print(f"aggregate RTF = {tot_proc/tot_dur:.3f}   mean per-clip RTF = {sum(rtfs)/len(rtfs):.3f}")
print(f"min RTF={min(rtfs):.2f}  max RTF={max(rtfs):.2f}")
PY
