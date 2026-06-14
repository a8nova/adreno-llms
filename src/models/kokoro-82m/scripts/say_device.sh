#!/bin/bash
# Fully on-device demo: synthesize MULTIPLE prompts on the phone GPU (via the
# pinned scripts/run_android.sh — never inline adb inference commands) and
# play each through the PHONE's speaker. The host is only used for misaki G2P
# (text -> ~30 phoneme ids; Kokoro has no on-device G2P) and adb plumbing.
#
# Usage: bash scripts/say_device.sh "First prompt." "Second prompt." ...
set -e
cd "$(dirname "$0")/.."

REMOTE=/data/local/tmp/Kokoro_82M_inference
PY=~/.nnopt/venvs/kokoro_py312/bin/python

if [ $# -eq 0 ]; then
    set -- "The teacher worked at the school for many years."
fi

# 1. Phonemize all prompts on host (the only non-device step) -> /tmp/say_ids_N.bin
$PY - "$@" <<'EOF' 2>/dev/null
import sys, struct, json
from misaki import en
g2p = en.G2P()
with open('model_info/config.json') as f: vocab = json.load(f)['vocab']
for n, text in enumerate(sys.argv[1:]):
    ph, _ = g2p(text)
    ids = [vocab[p] for p in ph if p in vocab]
    final = [0] + ids + [0]
    with open(f'/tmp/say_ids_{n}.bin', 'wb') as f:
        for i in final: f.write(struct.pack('<i', i))
    print(f"[g2p] {n}: {len(final)} ids  «{text}»")
EOF

# 2. Synthesize each prompt on the phone via the pinned run script, then play
#    the result through the phone's speaker.
export NNOPT_GPU_FP32_GENERATOR=1     # audio-correct generator path
export NNOPT_DEBUG_LAYERS=0           # demo mode: no per-layer validation
n=0
for TEXT in "$@"; do
    echo "[device] synthesizing prompt $n: \"$TEXT\""
    ./scripts/run_android.sh "$TEXT" 7 --token-ids /tmp/say_ids_$n.bin 2>&1 \
        | grep -E "BENCHMARK total_inference_sec|TTS_OUTPUT_PCM_SAMPLES" \
        | awk -v n=$n '/PCM/{s=$2} /total/{printf "[device] prompt %s: %.2fs synth, %.2fs audio (RTF %.2f)\n", n, $3, s/24000, $3/(s/24000)}'
    adb shell "cp $REMOTE/output.wav /sdcard/Download/kokoro_say_$n.wav"
    DUR=$(adb shell "toybox stat -c %s $REMOTE/output.wav" | awk '{print ($1-44)/2/24000}')
    echo "[play] prompt $n on phone speaker (${DUR}s)"
    adb shell "am start -a android.intent.action.VIEW -d file:///sdcard/Download/kokoro_say_$n.wav -t audio/wav" >/dev/null 2>&1
    sleep "$(echo "$DUR" | awk '{print $1 + 0.8}')"
    n=$((n+1))
done
echo "[done] all prompts synthesized and played on-device"
