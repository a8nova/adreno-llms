"""Quantize model.fp16.bin to model.int8.bin with per-row int8 weights + fp16 scales.

Selective scheme: 2D weight matrices >= 64KB get quantized; everything else
(biases, RMSNorm/LayerNorm weights, tiny lookup tables) stays fp16.

Output layout:
  - Quantized 2D tensors: int8 weights stored row-major, identical shape to source.
    Accompanied by a pseudo-tensor "<key>.scales" containing N fp16 scales (one
    per output row).
  - Other tensors: copied as-is.
  - All tensor offsets padded to 128 bytes for OpenCL sub-buffer alignment.

The C++ loader reads dtype==int8 and looks up the corresponding ".scales" key
for the per-row dequantization scale buffer. GEMV/dequant kernels read the int8
weight + fp16 scale to compute fp16 outputs at the same numeric precision as
the original fp16 weights (typical max-abs error ~0.5%).
"""
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent / "weights"
    src_bin = workspace / "model.fp16.bin"
    src_meta_path = workspace / "model.fp16.meta.json"
    out_bin = workspace / "model.int8.bin"
    out_meta_path = workspace / "model.int8.meta.json"

    print(f"[quant] reading {src_meta_path}", file=sys.stderr)
    with open(src_meta_path) as f:
        meta = json.load(f)

    src_data = np.memmap(src_bin, dtype=np.uint8, mode="r")
    src_tensors = meta["tensors"]
    print(f"[quant] {len(src_tensors)} tensors in source", file=sys.stderr)

    new_tensors: dict = {}
    out_chunks: list[bytes] = []
    out_offset = 0
    PAD = 128

    def pad_to_align():
        nonlocal out_offset
        rem = (-out_offset) % PAD
        if rem:
            out_chunks.append(b"\x00" * rem)
            out_offset += rem

    n_quantized = 0
    n_kept_fp16 = 0
    bytes_saved = 0

    # Process in canonical order (sorted by source offset for stability).
    items = sorted(src_tensors.items(), key=lambda kv: kv[1]["offset"])
    for name, info in items:
        shape = info["shape"]
        src_off = info["offset"]
        src_bytes = info["size_bytes"]
        dtype = info["dtype"]

        # Quantize only 2D fp16 weight matrices >= 64KB.
        quantize = (
            len(shape) == 2
            and dtype == "float16"
            and src_bytes >= 64 * 1024
        )

        if quantize:
            N, K = shape
            fp16 = np.frombuffer(
                bytes(src_data[src_off:src_off + src_bytes]),
                dtype=np.float16,
            ).reshape(N, K)
            fp32 = fp16.astype(np.float32)
            absmax = np.abs(fp32).max(axis=1)
            absmax = np.maximum(absmax, 1e-8)
            scales_fp32 = absmax / 127.0
            q = np.round(fp32 / scales_fp32[:, None]).clip(-127, 127).astype(np.int8)
            scales_fp16 = scales_fp32.astype(np.float16)

            pad_to_align()
            new_tensors[name] = {
                "offset": out_offset,
                "shape": shape,
                "dtype": "int8",
                "num_elements": int(N * K),
                "size_bytes": int(N * K),
                "scale_key": name + ".scales",
            }
            out_chunks.append(q.tobytes())
            out_offset += N * K

            pad_to_align()
            new_tensors[name + ".scales"] = {
                "offset": out_offset,
                "shape": [int(N)],
                "dtype": "float16",
                "num_elements": int(N),
                "size_bytes": int(N * 2),
            }
            out_chunks.append(scales_fp16.tobytes())
            out_offset += N * 2

            n_quantized += 1
            bytes_saved += src_bytes - (N * K + N * 2)
        else:
            pad_to_align()
            new_tensors[name] = {**info, "offset": out_offset}
            out_chunks.append(bytes(src_data[src_off:src_off + src_bytes]))
            out_offset += src_bytes
            n_kept_fp16 += 1

    pad_to_align()

    print(
        f"[quant] {n_quantized} tensors quantized, {n_kept_fp16} kept fp16, "
        f"saved {bytes_saved/1e6:.1f}MB; output {out_offset/1e6:.1f}MB "
        f"(source {meta['total_bytes']/1e6:.1f}MB)",
        file=sys.stderr,
    )

    print(f"[quant] writing {out_bin}", file=sys.stderr)
    with open(out_bin, "wb") as f:
        for c in out_chunks:
            f.write(c)

    out_meta = {
        "model_id": meta.get("model_id", ""),
        "format": meta.get("format", ""),
        "layout": meta.get("layout", ""),
        "dtype": "int8",
        "quantization": {
            "method": "int8",
            "per_row_scale": True,
            "scale_dtype": "float16",
            "threshold_bytes": 64 * 1024,
            "shape_requirement": "2d_only",
        },
        "bin_size_bytes": int(out_offset),
        "total_bytes": int(out_offset),
        "tensors": new_tensors,
    }
    with open(out_meta_path, "w") as f:
        json.dump(out_meta, f, indent=2)

    print(f"[quant] OK — {out_bin.name} ({out_offset/1e6:.1f}MB), {out_meta_path.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
