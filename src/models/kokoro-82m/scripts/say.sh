#!/bin/bash
# Convenience runner: phonemize text → generate test_input_ids.bin → run on device → pull WAV.
# Usage: bash scripts/say.sh "Your text here"

set -e
cd "$(dirname "$0")/.."

TEXT="${1:-The teacher worked at the}"
echo "[say] text: $TEXT"

# 1. Phonemize via misaki (deterministic) → assets/test_input_ids.bin
~/.nnopt/venvs/kokoro_py312/bin/python - "$TEXT" <<'EOF'
import sys, struct, json
from misaki import en
g2p = en.G2P()
text = sys.argv[1]
ph, _ = g2p(text)
print(f"[say] phonemes: {ph}")
with open('model_info/config.json') as f: cfg = json.load(f)
vocab = cfg['vocab']
ids = [vocab.get(p) for p in ph]
ids = [i for i in ids if i is not None]
final = [0] + ids + [0]
print(f"[say] {len(final)} ids: {final}")
with open('assets/test_input_ids.bin', 'wb') as f:
    for i in final: f.write(struct.pack('<i', i))
EOF

# 2. Deploy assets (binary already deployed)
adb push assets/test_input_ids.bin /data/local/tmp/Kokoro_82M_inference/assets/ 2>&1 | tail -1

# 3. Run on device with the optimized fp16-conv generator path.
#   NNOPT_GPU_FP32_GENERATOR=1   - fp32-accumulator generator (audio-correct path)
#   The fp16 texture-weight convs (conv1d_ht_t8x4) are default-on via
#   NNOPT_H16CONV/NNOPT_HMATH/NNOPT_TEXW. (The obsolete NNOPT_HYBRID_RESBLOCKS
#   path was deleted 2026-06-06.)
echo "[say] running on device (fp16-conv GPU path)..."
adb shell "cd /data/local/tmp/Kokoro_82M_inference && \
  NNOPT_GPU_FP32_GENERATOR=1 \
  LD_LIBRARY_PATH=/data/local/tmp/Kokoro_82M_inference/lib:/system/vendor/lib64:\$LD_LIBRARY_PATH \
  ./Kokoro_82M_inference_fp16 '$TEXT' 7" 2>&1 | grep -E "BENCHMARK total|TTS_OUTPUT_PCM|FP32|fp32_gen"

# 4. Pull WAV
OUT="/tmp/kokoro_say_$(date +%s).wav"
adb pull /data/local/tmp/Kokoro_82M_inference/output.wav "$OUT" 2>&1 | tail -1
echo "[say] saved: $OUT"

# 5. Quick stats + RTF
~/.nnopt/venvs/kokoro_py312/bin/python -c "
import wave
w = wave.open('$OUT')
n = w.getnframes()
audio_s = n / 24000.0
print(f'[say] {n} samples, {audio_s:.2f}s audio @ 24kHz')
"

# 6. (macOS) auto-play
if [ "$(uname)" = "Darwin" ]; then afplay "$OUT"; fi
