#!/usr/bin/env python3
"""Per-layer cosine check between runtime layer_dumps and reference dumps.

Pure stdlib. Usage:
    python3 scripts/cos_check.py <layer_name> [pass=0]

Reads:
    layer_dumps/<layer>__pass<N>.bin           (runtime, fp16 or fp32)
    layer_dumps/<layer>__pass<N>.bin.meta.json (optional dtype sidecar)
    reference/layers/<layer>_output.bin        (PyTorch reference, always fp32)

Prints cos, per-side mean/std, first few elements, and a size-mismatch line
if shapes disagree (truncates to min size and proceeds).
"""
import json
import math
import struct
import sys
from pathlib import Path


def _read_bin(path: Path, dtype: str):
    data = path.read_bytes()
    fmt = "e" if dtype == "float16" else "f"
    bpe = 2 if dtype == "float16" else 4
    n = len(data) // bpe
    return list(struct.unpack(f"{n}{fmt}", data[: n * bpe]))


def _detect_runtime_dtype(bin_path: Path, default: str) -> str:
    meta = bin_path.with_suffix(bin_path.suffix + ".meta.json")
    if meta.exists():
        try:
            v = json.loads(meta.read_text()).get("dtype")
            if v in ("float16", "float32"):
                return v
        except Exception:
            pass
    return default


def _stats(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    v = sum((x - m) * (x - m) for x in xs) / n
    return m, math.sqrt(v)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: cos_check.py <layer_name> [pass=0]", file=sys.stderr)
        return 2
    layer = sys.argv[1]
    pass_idx = sys.argv[2] if len(sys.argv) > 2 else "0"
    ws = Path.cwd()
    rt_bin = ws / "layer_dumps" / f"{layer}__pass{pass_idx}.bin"
    ref_bin = ws / "reference" / "layers" / f"{layer}_output.bin"
    if not rt_bin.exists():
        print(f"missing runtime dump: {rt_bin}", file=sys.stderr)
        return 1
    if not ref_bin.exists():
        print(f"missing reference dump: {ref_bin}", file=sys.stderr)
        return 1

    # Runtime dtype: read sidecar meta if present; else infer from bits in
    # nnport_config.yaml (fp16 for bits=16 ports, fp32 otherwise). Default
    # fp16 — modern ports run at fp16 by default.
    default_dtype = "float16"
    cfg = ws / "nnport_config.yaml"
    if cfg.exists():
        try:
            text = cfg.read_text()
            if "bits: 32" in text or "target_bits: 32" in text:
                default_dtype = "float32"
        except Exception:
            pass
    rt_dtype = _detect_runtime_dtype(rt_bin, default_dtype)

    rt = _read_bin(rt_bin, rt_dtype)
    ref = _read_bin(ref_bin, "float32")
    if len(rt) != len(ref):
        print(f"SIZE MISMATCH: runtime={len(rt)} reference={len(ref)} (comparing first {min(len(rt), len(ref))})")
    n = min(len(rt), len(ref))
    if n == 0:
        print("empty tensor")
        return 1

    dot = math.fsum(rt[i] * ref[i] for i in range(n))
    norm_rt = math.sqrt(math.fsum(x * x for x in rt[:n]))
    norm_ref = math.sqrt(math.fsum(x * x for x in ref[:n]))
    cos = dot / (norm_rt * norm_ref + 1e-12)

    rt_m, rt_s = _stats(rt[:n])
    ref_m, ref_s = _stats(ref[:n])
    head_rt = [round(x, 4) for x in rt[:5]]
    head_ref = [round(x, 4) for x in ref[:5]]
    print(f"layer={layer} pass={pass_idx} n={n} runtime_dtype={rt_dtype}")
    print(f"cos={cos:.4f}")
    print(f"runtime: mean={rt_m:.4f} std={rt_s:.4f} head={head_rt}")
    print(f"reference: mean={ref_m:.4f} std={ref_s:.4f} head={head_ref}")
    if cos < 0.95:
        print("HINT: cos < 0.95 — the bug is upstream of this layer; do not edit decode/print sites.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
