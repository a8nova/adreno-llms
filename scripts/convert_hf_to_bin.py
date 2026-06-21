#!/usr/bin/env python3
"""Convert any HuggingFace safetensors model into the nnopt fp16 binary layout.

This is the "first-mile" converter: it GENERATES the flat .bin + meta.json that
the C++/OpenCL runtime loads, straight from upstream safetensors. (The older
scripts/convert_openelm_weights.py only fills bytes into a *pre-existing*
meta.json; this script produces the layout itself.)

The binary format consumed by the runtime is dead simple:
- A flat file of fp16 tensor data, concatenated, contiguous, no padding.
- Each tensor lives at the byte offset declared in model.fp16.meta.json.
- Tensors are laid out in alphabetical order of their names (deterministic), so
  the same model always produces a byte-identical layout.

Handles single-file and sharded safetensors (model.safetensors.index.json), and
F32 / F16 / BF16 / F64 / integer dtypes (everything is cast to fp16). Depends on
numpy only — no `safetensors`/`torch` needed (it parses the container directly),
which sidesteps numpy's lack of a native bf16 dtype.

Usage:
    # Convert a local model dir (or a single .safetensors file):
    python3 convert_hf_to_bin.py \\
        --input  /path/to/HF/model_dir \\
        --model-id Qwen/Qwen2.5-0.5B-Instruct \\
        --output-dir src/models/qwen2-5-0-5b/weights

    # Validate the generator against an existing meta.json (no files written):
    python3 convert_hf_to_bin.py \\
        --input  /path/to/apple/OpenELM-270M \\
        --check-against src/models/openelm-270m/weights/model.fp16.meta.json

Requires:  pip install numpy
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

# safetensors dtype string -> (numpy dtype to read raw bytes as, is_bf16)
_ST_DTYPE = {
    "F64": ("<f8", False),
    "F32": ("<f4", False),
    "F16": ("<f2", False),
    "BF16": ("<u2", True),   # read as uint16, expand to fp32 manually
    "I64": ("<i8", False),
    "I32": ("<i4", False),
    "I16": ("<i2", False),
    "I8": ("<i1", False),
    "U8": ("<u1", False),
    "BOOL": ("<u1", False),
}


def _read_safetensors_header(path):
    """Return (header_dict, data_start_offset) for one .safetensors file."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return header, 8 + n


def _discover_files(input_path):
    """Resolve --input to an ordered, de-duplicated list of .safetensors files."""
    if os.path.isfile(input_path):
        if not input_path.endswith(".safetensors"):
            raise SystemExit(f"ERROR: {input_path} is not a .safetensors file")
        return [input_path]

    if not os.path.isdir(input_path):
        raise SystemExit(f"ERROR: --input not found: {input_path}")

    index = os.path.join(input_path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted({os.path.join(input_path, s) for s in weight_map.values()})
        return shards

    shards = sorted(
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.endswith(".safetensors")
    )
    if not shards:
        raise SystemExit(f"ERROR: no .safetensors found under {input_path}")
    return shards


def _to_fp16(raw, np, dtype_str):
    np_dtype, is_bf16 = _ST_DTYPE[dtype_str]
    arr = np.frombuffer(raw, dtype=np_dtype)
    if is_bf16:
        # bf16 -> fp32: place the 16 bits in the high half of a 32-bit float.
        u32 = arr.astype(np.uint32) << 16
        arr = u32.view(np.float32)
    return arr.astype(np.float16)


def build_layout(input_path, np):
    """Read all tensors, return (records, total_bytes).

    records: list of dicts {name, shape, num_elements, size_bytes, fp16_bytes},
    ordered alphabetically by name (the runtime's expected layout order).
    """
    files = _discover_files(input_path)
    # name -> (file, info)
    tensors = {}
    for fp in files:
        header, data_start = _read_safetensors_header(fp)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise SystemExit(f"ERROR: tensor {name!r} present in multiple shards")
            tensors[name] = (fp, data_start, info)

    records = []
    for name in sorted(tensors):
        fp, data_start, info = tensors[name]
        dtype_str = info["dtype"]
        if dtype_str not in _ST_DTYPE:
            raise SystemExit(f"ERROR: {name}: unsupported dtype {dtype_str}")
        beg, end = info["data_offsets"]
        with open(fp, "rb") as fh:
            fh.seek(data_start + beg)
            raw = fh.read(end - beg)
        arr = _to_fp16(raw, np, dtype_str)
        shape = list(info["shape"])
        nelem = int(np.prod(shape)) if shape else 1
        if arr.size != nelem:
            raise SystemExit(
                f"ERROR: {name}: element count {arr.size} != prod(shape) {nelem}"
            )
        data = arr.tobytes()
        records.append({
            "name": name,
            "shape": shape,
            "num_elements": nelem,
            "size_bytes": len(data),
            "data": data,
        })
    return records


def make_meta(model_id, records):
    offset = 0
    tensors = {}
    for r in records:
        tensors[r["name"]] = {
            "offset": offset,
            "shape": r["shape"],
            "dtype": "float16",
            "num_elements": r["num_elements"],
            "size_bytes": r["size_bytes"],
        }
        offset += r["size_bytes"]
    return {
        "model_id": model_id,
        "format": "binary",
        "layout": "row_major",
        "dtype": "float16",
        "quantization": {"enabled": True, "bits": 16, "method": "float16"},
        "total_bytes": offset,
        "tensors": tensors,
    }


def check_against(meta, ref_path):
    """Compare generated layout to an existing meta.json. Returns #diffs."""
    with open(ref_path, "r", encoding="utf-8") as f:
        ref = json.load(f)
    diffs = 0
    gen_t, ref_t = meta["tensors"], ref["tensors"]
    only_gen = sorted(set(gen_t) - set(ref_t))
    only_ref = sorted(set(ref_t) - set(gen_t))
    for k in only_gen:
        print(f"  + generated has extra tensor: {k}"); diffs += 1
    for k in only_ref:
        print(f"  - reference has tensor not generated: {k}"); diffs += 1
    for k in sorted(set(gen_t) & set(ref_t)):
        for field in ("offset", "shape", "size_bytes", "num_elements"):
            if gen_t[k][field] != ref_t[k][field]:
                print(f"  ~ {k}.{field}: gen={gen_t[k][field]} ref={ref_t[k][field]}")
                diffs += 1
    if meta["total_bytes"] != ref["total_bytes"]:
        print(f"  ~ total_bytes: gen={meta['total_bytes']} ref={ref['total_bytes']}")
        diffs += 1
    return diffs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=None,
                   help="Path to a .safetensors file or a model dir (sharded ok). "
                        "Omit when using --hf-repo-id.")
    p.add_argument("--hf-repo-id", default=None,
                   help="Download safetensors from this HF repo (via the `hf` CLI) "
                        "and convert. Mutually exclusive with --input.")
    p.add_argument("--hf-branch", default="main",
                   help="HF revision/branch to download (default: main)")
    p.add_argument("--model-id", default=None,
                   help="Label stored in meta.json (default: basename of --input)")
    p.add_argument("--output-dir", default=None,
                   help="Write model.fp16.bin + model.fp16.meta.json here")
    p.add_argument("--check-against", default=None, metavar="META",
                   help="Validate generated layout against an existing meta.json; "
                        "writes nothing unless --output-dir is also given")
    args = p.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("ERROR: numpy required — pip install numpy", file=sys.stderr)
        return 1

    if bool(args.input) == bool(args.hf_repo_id):
        print("ERROR: pass exactly one of --input or --hf-repo-id", file=sys.stderr)
        return 1

    tmpdir = None
    if args.hf_repo_id:
        tmpdir = tempfile.mkdtemp(prefix="hf_convert_")
        print(f"Downloading {args.hf_repo_id}@{args.hf_branch} via hf CLI ...")
        try:
            subprocess.run(
                ["hf", "download", args.hf_repo_id, "--revision", args.hf_branch,
                 "--include", "*.safetensors", "*.safetensors.index.json",
                 "--local-dir", tmpdir],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"ERROR: hf download failed ({e}). Install with "
                  f"`pip install -U 'huggingface_hub[cli]'` and `hf auth login`.",
                  file=sys.stderr)
            return 1
        input_path = tmpdir
        model_id = args.model_id or args.hf_repo_id
    else:
        input_path = args.input
        model_id = args.model_id or os.path.basename(os.path.abspath(args.input))

    print(f"Reading: {input_path}")
    records = build_layout(input_path, np)
    meta = make_meta(model_id, records)
    print(f"Layout: {len(records)} tensors, {meta['total_bytes']:,} bytes (fp16)")

    if args.check_against:
        print(f"Checking against {args.check_against} ...")
        diffs = check_against(meta, args.check_against)
        if diffs:
            print(f"MISMATCH: {diffs} difference(s) ✗")
            if not args.output_dir:
                return 1
        else:
            print("MATCH: generated layout is byte-identical ✓")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        bin_path = os.path.join(args.output_dir, "model.fp16.bin")
        meta_path = os.path.join(args.output_dir, "model.fp16.meta.json")
        with open(bin_path, "wb") as f:
            for r in records:
                f.write(r["data"])
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        written = os.path.getsize(bin_path)
        if written != meta["total_bytes"]:
            print(f"ERROR: wrote {written} bytes, expected {meta['total_bytes']}",
                  file=sys.stderr)
            return 1
        print(f"Wrote {bin_path} ({written:,} bytes) ✓")
        print(f"Wrote {meta_path} ✓")

    return 0


if __name__ == "__main__":
    sys.exit(main())
