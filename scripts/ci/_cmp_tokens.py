#!/usr/bin/env python3
# Shared token golden compare/capture for text/vlm/asr modalities.
# Usage: _cmp_tokens.py <device-run-log> <modality>
# Env: GOLDEN=<path to golden json>, CAPTURE=0|1
# Parses "Generated token: N" lines from the device run, then either captures them
# as the golden or compares to it. Exit 0=match/captured, 1=mismatch/error.
import json, sys, re, os

log = open(sys.argv[1], errors="ignore").read()
modality = sys.argv[2] if len(sys.argv) > 2 else "text"
got = [int(m) for m in re.findall(r"Generated token:\s*(-?\d+)", log)]
if not got:
    print("FAIL: no 'Generated token:' lines parsed from device output (see log)")
    sys.exit(1)

gp = os.environ["GOLDEN"]
if os.environ.get("CAPTURE") == "1":
    json.dump({"modality": modality, "generated_ids": got}, open(gp, "w"), indent=2)
    print(f"CAPTURED golden: {len(got)} tokens -> {gp}\n  {got}")
    sys.exit(0)

if not os.path.exists(gp):
    print(f"FAIL: no golden at {gp} — run with --capture on a known-good build first.")
    sys.exit(1)
exp = json.load(open(gp))["generated_ids"]
if got == exp:
    print(f"PASS: {len(exp)} tokens match golden  {exp}")
    sys.exit(0)
print("FAIL: device output deviates from golden (regression?)")
print(f"  golden: {exp}")
print(f"  device: {got}")
sys.exit(1)
