#!/usr/bin/env python3
"""
Per-row symmetric int8 weight quantizer for SmolLM2-135M-Instruct (LLaMA family).

Reads weights/model.fp16.bin + meta.json, writes weights/model.int8.bin + meta.json.

Quantization scheme:
- Target tensors: every nn.Linear weight in the transformer blocks
  (q_proj/k_proj/v_proj/o_proj + gate_proj/up_proj/down_proj). Embedding /
  lm_head stays fp16 in the first pass (lm_head accuracy is sensitive on
  small vocabs; revisit after first measurement). LayerNorm/RMSNorm scales
  pass through unchanged.
- Per output row, symmetric (no zero-point):
      scale[n] = max(|W[n, :]|) / 127
      W_q[n, k] = clip(round(W[n, k] / scale[n]), -127, 127).astype(int8)
- Stored side-by-side: original key holds int8 [N, K]; a new key
  `<name>.scale` holds fp16 [N]. The OpenCL int8 GEMV kernel reads both.

Why symmetric / no zero-point:
- LLaMA-style Linear weights are roughly zero-mean (Xavier init + RMSNorm
  upstream), so symmetric quantization loses negligible accuracy.
- Skipping zero-point removes an int32 correction term from the GEMV inner
  loop, keeping it branch-free and friendly to qcom_dot8_acc.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


QUANTIZE_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


def is_quantizable(name: str, shape: list, quantize_embed: bool) -> bool:
    if len(shape) != 2:
        return False
    if quantize_embed and name == "model.embed_tokens.weight":
        return True
    return any(name.endswith(suf) for suf in QUANTIZE_SUFFIXES)


def _emit_lm_head_alias(arr_fp16: np.ndarray, n_rows: int, n_cols: int):
    """Quantize embed_tokens fp16 weights to int8 + fp16 scales and return both
    as bytes. Used to emit a separate `lm_head.weight` int8 tensor while
    keeping the original `model.embed_tokens.weight` fp16 (preserves the
    fp16 embedding-lookup path under tied-embedding models like SmolLM2)."""
    q = np.empty((n_rows, n_cols), dtype=np.int8)
    scales = np.empty((n_rows,), dtype=np.float16)
    for r in range(n_rows):
        qr, sr = quantize_row_symmetric(arr_fp16[r])
        q[r] = qr
        scales[r] = sr
    return q.tobytes(), scales.tobytes()


def fp16_bytes_to_array(data: bytes, n_elems: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float16, count=n_elems)


def quantize_row_symmetric(w_row: np.ndarray):
    abs_max = float(np.max(np.abs(w_row.astype(np.float32))))
    scale = abs_max / 127.0 if abs_max > 0 else 1.0
    q = np.clip(np.round(w_row.astype(np.float32) / scale), -127, 127).astype(np.int8)
    return q, np.float16(scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default=str(Path(__file__).resolve().parent.parent / "weights"))
    ap.add_argument("--fp16-bin", default=None)
    ap.add_argument("--fp16-meta", default=None)
    ap.add_argument("--out-bin", default=None)
    ap.add_argument("--out-meta", default=None)
    ap.add_argument("--quantize-embed", action="store_true",
                    help="Also quantize model.embed_tokens.weight (tied to lm_head). "
                         "Off by default — first int8 step keeps lm_head fp16 to avoid argmax drift.")
    ap.add_argument("--emit-lm-head-int8", action="store_true",
                    help="Emit a separate `lm_head.weight` int8 tensor (alias of "
                         "embed_tokens) WITHOUT touching model.embed_tokens.weight. "
                         "Lets the lm_head layer use int8 image path while the "
                         "embedding-lookup kernel keeps its fp16 fast path.")
    args = ap.parse_args()

    weights_dir = Path(args.weights_dir)
    fp16_bin = Path(args.fp16_bin) if args.fp16_bin else weights_dir / "model.fp16.bin"
    fp16_meta = Path(args.fp16_meta) if args.fp16_meta else weights_dir / "model.fp16.meta.json"
    out_bin = Path(args.out_bin) if args.out_bin else weights_dir / "model.int8.bin"
    out_meta = Path(args.out_meta) if args.out_meta else weights_dir / "model.int8.meta.json"

    print(f"Reading: {fp16_meta}", file=sys.stderr)
    meta = json.loads(fp16_meta.read_text())
    tensors_in = meta["tensors"]

    print(f"Reading: {fp16_bin}", file=sys.stderr)
    raw = fp16_bin.read_bytes()

    out_tensors = {}
    out_chunks = []
    cur_offset = 0
    quantized_count = 0
    passthrough_count = 0
    total_fp16_quant_input_bytes = 0

    for name in sorted(tensors_in.keys()):
        t = tensors_in[name]
        shape = t["shape"]
        offset = t["offset"]
        nbytes = t["size_bytes"]
        nelems = t["num_elements"]
        dtype = t["dtype"]

        if dtype != "float16":
            raise RuntimeError(f"Expected float16 input, got {dtype} for {name}")
        if offset + nbytes > len(raw):
            raise RuntimeError(f"Tensor {name} out of range")

        arr = fp16_bytes_to_array(raw[offset:offset + nbytes], nelems)

        if is_quantizable(name, shape, args.quantize_embed):
            n_rows, n_cols = shape
            arr = arr.reshape(n_rows, n_cols)
            q = np.empty((n_rows, n_cols), dtype=np.int8)
            scales = np.empty((n_rows,), dtype=np.float16)
            for r in range(n_rows):
                qr, sr = quantize_row_symmetric(arr[r])
                q[r] = qr
                scales[r] = sr

            if quantized_count == 0:
                recon = q[0].astype(np.float32) * float(scales[0])
                err = float(np.max(np.abs(recon - arr[0].astype(np.float32))))
                rel = err / max(1e-6, float(np.max(np.abs(arr[0].astype(np.float32)))))
                print(f"  sanity[{name}] row0: max_abs_err={err:.4g} rel={rel:.2%}", file=sys.stderr)

            q_bytes = q.tobytes()
            scale_bytes = scales.tobytes()
            out_tensors[name] = {
                "offset": cur_offset,
                "shape": list(shape),
                "dtype": "int8",
                "num_elements": int(nelems),
                "size_bytes": len(q_bytes),
            }
            out_chunks.append(q_bytes)
            cur_offset += len(q_bytes)

            out_tensors[name + ".scale"] = {
                "offset": cur_offset,
                "shape": [n_rows],
                "dtype": "float16",
                "num_elements": n_rows,
                "size_bytes": len(scale_bytes),
            }
            out_chunks.append(scale_bytes)
            cur_offset += len(scale_bytes)

            quantized_count += 1
            total_fp16_quant_input_bytes += nbytes
        else:
            out_tensors[name] = {
                "offset": cur_offset,
                "shape": list(shape),
                "dtype": "float16",
                "num_elements": int(nelems),
                "size_bytes": nbytes,
            }
            out_chunks.append(raw[offset:offset + nbytes])
            cur_offset += nbytes
            passthrough_count += 1

    # Optional sidecar: a separate `lm_head.weight` int8 tensor cloned from
    # embed_tokens. Leaves model.embed_tokens.weight fp16 (so the embedding
    # lookup keeps working) while giving the lm_head layer a quantized
    # weight to load by a different key.
    lm_head_emitted = False
    if args.emit_lm_head_int8:
        embed_meta = tensors_in.get("model.embed_tokens.weight")
        if embed_meta is None or embed_meta["dtype"] != "float16":
            print("warn: --emit-lm-head-int8 needs fp16 model.embed_tokens.weight", file=sys.stderr)
        else:
            shape = embed_meta["shape"]
            offset = embed_meta["offset"]
            nbytes = embed_meta["size_bytes"]
            nelems = embed_meta["num_elements"]
            arr = fp16_bytes_to_array(raw[offset:offset + nbytes], nelems).reshape(shape[0], shape[1])
            q_bytes, scale_bytes = _emit_lm_head_alias(arr, shape[0], shape[1])
            out_tensors["lm_head.weight"] = {
                "offset": cur_offset,
                "shape": list(shape),
                "dtype": "int8",
                "num_elements": int(nelems),
                "size_bytes": len(q_bytes),
            }
            out_chunks.append(q_bytes)
            cur_offset += len(q_bytes)
            out_tensors["lm_head.weight.scale"] = {
                "offset": cur_offset,
                "shape": [shape[0]],
                "dtype": "float16",
                "num_elements": shape[0],
                "size_bytes": len(scale_bytes),
            }
            out_chunks.append(scale_bytes)
            cur_offset += len(scale_bytes)
            lm_head_emitted = True
            print(f"  emitted lm_head.weight int8 alias ({(len(q_bytes)+len(scale_bytes))/1e6:.1f} MB)", file=sys.stderr)

    out_meta_dict = {
        "model_id": meta.get("model_id"),
        "format": "binary",
        "layout": "row_major",
        "dtype": "mixed",
        "quantization": {
            "enabled": True,
            "bits": 8,
            "method": "per-row symmetric int8 (linear weights) + fp16 scales + fp16 passthrough (norms/lm_head)",
            "quantized_embed": bool(args.quantize_embed),
            "lm_head_int8_alias": lm_head_emitted,
        },
        "tensors": out_tensors,
    }

    print(f"Writing: {out_bin} ({cur_offset/1e6:.1f} MB total)", file=sys.stderr)
    with out_bin.open("wb") as f:
        for chunk in out_chunks:
            f.write(chunk)

    print(f"Writing: {out_meta}", file=sys.stderr)
    out_meta.write_text(json.dumps(out_meta_dict, indent=2))

    print(f"Quantized {quantized_count} weights, passed through {passthrough_count}", file=sys.stderr)
    print(f"Original fp16 size of quantized weights: {total_fp16_quant_input_bytes/1e6:.1f} MB", file=sys.stderr)
    print(f"Total new file size:                     {cur_offset/1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
