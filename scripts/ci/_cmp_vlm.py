#!/usr/bin/env python3
# VLM caption golden compare/capture. The VLM binaries stream the generated caption
# to stdout (no "Generated token:"/"GENERATED_TEXT:" marker), merged with device
# info + BENCHMARK lines via adb. We extract the caption as the text emitted after
# the "Prompt:" line and before the metrics, then golden it (greedy => deterministic).
# Usage: _cmp_vlm.py <device-run-log>   Env: GOLDEN, CAPTURE.  Exit 0/1.
import json, sys, os, re

raw = open(sys.argv[1], errors="ignore").read()
if re.search(r"ERROR|failed to load|stbi_load failed|CANNOT LINK", raw):
    print("FAIL: device run errored (see log) — not capturing"); sys.exit(1)

# Preferred path: some VLMs (lfm2-vl) emit a clean, deterministic "GENERATED_TEXT:"
# line — use it directly. The streamed-token path is interleaved with per-token
# "Sampler:" debug noise and is NOT stable run-to-run, so only fall back to stream
# extraction for models that don't print GENERATED_TEXT (smolvlm).
_gt = re.findall(r"GENERATED_TEXT:(.*)", raw)
if _gt:
    caption = _gt[-1].strip()
    gp = os.environ["GOLDEN"]
    if not caption:
        print("FAIL: empty GENERATED_TEXT"); sys.exit(1)
    if os.environ.get("CAPTURE") == "1":
        json.dump({"modality": "vlm", "text": caption}, open(gp, "w"), indent=2, ensure_ascii=False)
        print(f"CAPTURED golden: caption={caption!r} -> {gp}"); sys.exit(0)
    if not os.path.exists(gp):
        print(f"FAIL: no golden at {gp} — run with --capture first."); sys.exit(1)
    exp = json.load(open(gp))["text"]
    if caption == exp:
        print(f"PASS: caption matches golden  {caption!r}"); sys.exit(0)
    print("FAIL: caption deviates from golden (regression?)")
    print(f"  golden: {exp!r}")
    print(f"  device: {caption!r}")
    sys.exit(1)

lines = raw.splitlines()

NOISE = re.compile(
    r"(^\s*$|^\s+[a-z_]+\s|cl_|qcom|^─|^━|OpenCL|^Device:|^\s*SoC|Memory:|^\s*CUs|"
    r"Clock:|Driver:|Platform|IMGINFO|PerfHint|RecordQ|LmHead|^PHASE |^Loaded |"
    r"image2d|^BENCHMARK|tokens/sec|Running inference|Prompt:|pushed|uploaded|^Extra args|\[opencl\])"
)
STOP = re.compile(r"^(BENCHMARK|PHASE |✓|tokens/sec)")

# start collecting after the last "Prompt:" line
start = max((i for i, l in enumerate(lines) if l.strip().startswith("Prompt:")), default=-1)
cap = []
for l in lines[start + 1:]:
    if STOP.search(l.strip()):
        break
    if NOISE.search(l):
        continue
    if re.search(r"\S\s{2,}\S", l):   # column-aligned device-info table row (prose uses single spaces)
        continue
    cap.append(l)
caption = re.sub(r"\s+", " ", " ".join(cap)).strip()

if not caption:
    print("FAIL: no caption text parsed from device output (see log)"); sys.exit(1)

gp = os.environ["GOLDEN"]
if os.environ.get("CAPTURE") == "1":
    json.dump({"modality": "vlm", "text": caption}, open(gp, "w"), indent=2, ensure_ascii=False)
    print(f"CAPTURED golden: caption={caption!r} -> {gp}"); sys.exit(0)
if not os.path.exists(gp):
    print(f"FAIL: no golden at {gp} — run with --capture first."); sys.exit(1)
exp = json.load(open(gp))["text"]
if caption == exp:
    print(f"PASS: caption matches golden  {caption!r}"); sys.exit(0)
print("FAIL: caption deviates from golden (regression?)")
print(f"  golden: {exp!r}")
print(f"  device: {caption!r}")
sys.exit(1)
