#!/usr/bin/env python3
# Shared GENERATED_TEXT golden compare/capture for asr/vlm modalities (models that
# emit a decoded "GENERATED_TEXT: ..." line rather than per-token ids).
# Usage: _cmp_text.py <device-run-log> <modality>
# Env: GOLDEN=<path>, CAPTURE=0|1.  Exit 0=match/captured, 1=mismatch/error.
import json, sys, re, os

log = open(sys.argv[1], errors="ignore").read()
modality = sys.argv[2] if len(sys.argv) > 2 else "asr"
m = re.findall(r"GENERATED_TEXT:(.*)", log)
if not m:
    print("FAIL: no 'GENERATED_TEXT:' line parsed from device output (see log)")
    sys.exit(1)
got = m[-1].rstrip("\n")  # keep leading space (whisper emits ' you'); drop trailing newline

gp = os.environ["GOLDEN"]
if os.environ.get("CAPTURE") == "1":
    json.dump({"modality": modality, "text": got}, open(gp, "w"), indent=2, ensure_ascii=False)
    print(f"CAPTURED golden: text={got!r} -> {gp}")
    sys.exit(0)

if not os.path.exists(gp):
    print(f"FAIL: no golden at {gp} — run with --capture on a known-good build first.")
    sys.exit(1)
exp = json.load(open(gp))["text"]
if got == exp:
    print(f"PASS: text matches golden  {got!r}")
    sys.exit(0)
print("FAIL: device text deviates from golden (regression?)")
print(f"  golden: {exp!r}")
print(f"  device: {got!r}")
sys.exit(1)
