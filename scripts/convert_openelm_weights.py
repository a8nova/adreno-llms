#!/usr/bin/env python3
"""Convert apple/OpenELM-270M safetensors to the nnopt fp16 binary layout.

The binary format consumed by the C++ runtime is dead simple:
- A flat file of fp16 tensor data, concatenated.
- Each tensor lives at the byte offset declared in model.fp16.meta.json.

This converter reads the meta.json, opens the HF safetensors, and writes each
named tensor at the right offset. Tensor shapes / dtypes are validated so any
upstream model change is surfaced loudly.

Usage:
    python3 convert_openelm_weights.py \\
        --safetensors path/to/apple/OpenELM-270M/model.safetensors \\
        --meta path/to/model.fp16.meta.json \\
        --output path/to/model.fp16.bin

Requires:  pip install safetensors numpy
"""

import argparse
import json
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--safetensors", required=True,
                   help="Path to the HF safetensors file (e.g. apple/OpenELM-270M/model.safetensors)")
    p.add_argument("--meta", required=True,
                   help="Path to model.fp16.meta.json describing the target layout")
    p.add_argument("--output", required=True,
                   help="Path to write the output model.fp16.bin")
    args = p.parse_args()

    try:
        from safetensors import safe_open
        import numpy as np
    except ImportError as e:
        print(f"ERROR: missing Python dependency: {e}", file=sys.stderr)
        print("       pip install safetensors numpy", file=sys.stderr)
        return 1

    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    tensors_meta = meta["tensors"]
    total_bytes = meta["total_bytes"]
    print(f"Target layout: {len(tensors_meta)} tensors, {total_bytes:,} bytes total")

    # Open safetensors and gather available keys for diff-reporting.
    with safe_open(args.safetensors, framework="numpy") as st:
        available = set(st.keys())
        wanted = set(tensors_meta.keys())
        missing = wanted - available
        if missing:
            print(f"ERROR: {len(missing)} tensor(s) declared in meta.json not present in safetensors:",
                  file=sys.stderr)
            for k in sorted(missing)[:20]:
                print(f"  {k}", file=sys.stderr)
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more", file=sys.stderr)
            return 1

        # Pre-allocate the output buffer.
        out = bytearray(total_bytes)

        for i, (name, info) in enumerate(tensors_meta.items()):
            offset = info["offset"]
            expected_shape = list(info["shape"])
            expected_bytes = info["size_bytes"]

            arr = st.get_tensor(name)
            if list(arr.shape) != expected_shape:
                print(
                    f"ERROR: {name}: shape mismatch — meta wants {expected_shape}, "
                    f"safetensors has {list(arr.shape)}",
                    file=sys.stderr,
                )
                return 1

            arr16 = arr.astype(np.float16, copy=False)
            data = arr16.tobytes()
            if len(data) != expected_bytes:
                print(
                    f"ERROR: {name}: size mismatch — meta wants {expected_bytes} bytes, "
                    f"safetensors yields {len(data)} after fp16 cast",
                    file=sys.stderr,
                )
                return 1

            out[offset:offset + expected_bytes] = data
            if (i + 1) % 25 == 0 or i + 1 == len(tensors_meta):
                print(f"  [{i + 1:3d}/{len(tensors_meta)}] {name}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(out)

    written = os.path.getsize(args.output)
    if written != total_bytes:
        print(f"ERROR: wrote {written} bytes, expected {total_bytes}", file=sys.stderr)
        return 1

    print(f"Wrote {args.output} ({written:,} bytes) ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
