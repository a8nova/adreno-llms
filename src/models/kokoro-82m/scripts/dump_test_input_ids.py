#!/usr/bin/env python3
"""Generate assets/test_input_ids.bin matching the reference prompt.

Uses Kokoro's KPipeline to phonemize 'The teacher worked at the ' with voice
'af_heart' (same as reference/_run_reference.py captured), then writes the
resulting input_ids (including BOS/EOS) as int32 little-endian.

Run with the kokoro_host venv:
    ~/.nnopt/venvs/kokoro_host/bin/python scripts/dump_test_input_ids.py
"""
import struct
import sys
from pathlib import Path

try:
    from kokoro import KModel, KPipeline
except ImportError as e:
    print(f"kokoro not installed in this Python: {e}", file=sys.stderr)
    sys.exit(2)

PROMPT = "The teacher worked at the "
VOICE = "af_heart"

model = KModel(repo_id="hexgrad/Kokoro-82M")
pipe = KPipeline(lang_code="a", model=model)

input_ids = None
for chunk in pipe(PROMPT, voice=VOICE):
    # Try several common attribute names.
    for attr in ("input_ids", "tokens", "ids"):
        if hasattr(chunk, attr):
            v = getattr(chunk, attr)
            if hasattr(v, "tolist"):
                v = v.tolist()
            if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                v = v[0]
            input_ids = list(v)
            break
    if input_ids is not None:
        break
    if hasattr(chunk, "phonemes"):
        # Map phonemes through model.vocab manually.
        ph = chunk.phonemes
        ids = [model.vocab.get(p) for p in ph]
        ids = [i for i in ids if i is not None]
        input_ids = [0, *ids, 0]
        break

if not input_ids:
    print("could not extract input_ids from chunk", file=sys.stderr)
    sys.exit(3)

out_path = Path(__file__).parent.parent / "assets" / "test_input_ids.bin"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "wb") as f:
    for i in input_ids:
        f.write(struct.pack("<i", int(i)))

print(f"wrote {len(input_ids)} ids to {out_path}: {input_ids}")
