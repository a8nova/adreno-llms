#!/bin/bash
# Batch benchmark (lever #1): transcribe ALL clips in ONE device process via
# --audio-list, so the per-process JIT compile (CLBlast + our .cl kernels, ~4-5s)
# is paid once for clip 0 instead of re-paid every clip. This measures the
# real-deployment steady-state RTF (model loaded once, streaming many clips),
# unlike run_bench.sh which launches a fresh process (and re-JITs) per clip.
set -e
cd "$(dirname "$0")/.."
export NNOPT_DTYPE="${NNOPT_DTYPE:-fp16}"
. ./scripts/remote_dir.env
ADB="${ADB:-adb}"
PY=/Users/alazarshenkute/Projects/nnopt/venv/bin/python

# Build the device-side clip list (wav_path<TAB>duration) from the manifest.
LIST=bench/clips_batch.tsv
"$PY" - "$LIST" <<'PY'
import json, sys
with open(sys.argv[1], "w") as f:
    for m in json.load(open("bench/manifest.json")):
        f.write(f"bench/sample_{m['id']:02d}_audio.bin\t{m['duration_s']}\n")
PY
$ADB push "$LIST" "$REMOTE_DIR/$LIST" >/dev/null

echo "=== Batch-transcribing $(wc -l < "$LIST" | tr -d ' ') clips in ONE process ==="
RAW=bench/batch_raw.txt
$ADB shell "cd $REMOTE_DIR && NNOPT_DEBUG_LAYERS=0 LD_LIBRARY_PATH=$REMOTE_DIR/lib:/system/vendor/lib64:\$LD_LIBRARY_PATH ./$BINARY_NAME 'transcribe' 128 --audio-list $LIST" 2>/dev/null | tee "$RAW"

echo ""
echo "=== SUMMARY (steady-state, JIT amortized) ==="
"$PY" - "$RAW" <<'PY'
import json, re, sys
raw = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
man = json.load(open("bench/manifest.json"))
gen  = [l.split("GENERATED_TEXT:",1)[1].strip() for l in raw if "GENERATED_TEXT:" in l]
rtf  = [float(re.search(r"RTF:\s*([0-9.]+)", l).group(1)) for l in raw if l.strip().startswith("RTF:")]
proc = [float(re.search(r"proc=([0-9.]+)s", l).group(1)) for l in raw if "proc=" in l]
n = min(len(man), len(rtf))
print(f"{'clip':>4} {'dur':>6} {'RTF':>7}  transcript-ok")
tot_dur = tot_proc = 0.0
for i in range(n):
    d = float(man[i]["duration_s"]); tot_dur += d; tot_proc += proc[i]
    ok = "exact" if gen[i].strip() == man[i]["hf_transcript"].strip() else "~drift"
    print(f"{man[i]['id']:>4} {d:>6.2f} {rtf[i]:>7.3f}  {ok}")
print(f"\nclips={n}  audio_total={tot_dur:.1f}s  proc_total={tot_proc:.1f}s")
print(f"AGGREGATE RTF = {tot_proc/tot_dur:.3f}   (clip0 carries the one-time JIT)")
print(f"steady-state (clips 1..N, JIT excluded) = {sum(proc[1:n])/sum(float(man[i]['duration_s']) for i in range(1,n)):.3f}")
PY
