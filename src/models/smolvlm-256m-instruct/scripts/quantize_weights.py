#!/usr/bin/env python3
"""
Per-row symmetric int8 weight quantizer for SmolVLM-256M-Instruct.

Reads weights/model.fp16.bin + meta.json, writes weights/model.int8.bin + meta.json.

Adapted from smollm2-135m-instruct/scripts/quantize_weights.py to cover
SmolVLM's three weight families:

  1) text decoder (LLaMA-style)  — q/k/v/o_proj, gate/up/down_proj
  2) vision encoder (SigLIP)     — q/k/v/out_proj, mlp.fc1/fc2
  3) modality projector (Idefics3) — proj.weight   (576 × 12288)

  4) lm_head.weight              — independent in SmolVLM (NOT tied to
     embed_tokens), so we quantize it directly. embed_tokens stays fp16
     because the embedding-lookup kernel uses the fp16 fast path.

Quantization scheme (per row, symmetric, no zero-point):
    scale[n] = max(|W[n, :]|) / 127
    W_q[n, k] = clip(round(W[n, k] / scale[n]), -127, 127).astype(int8)

Stored side-by-side: original key holds int8 [N, K]; a sibling key
`<name>.scale` holds fp16 [N]. The OpenCL int8 GEMV kernels read both.

Why symmetric / no zero-point:
- These Linear weights are roughly zero-mean (Xavier/LayerNorm upstream).
- Skipping zero-point removes an int32 correction term from the GEMV inner
  loop, keeping it branch-free and friendly to qcom_dot8_acc.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Suffixes (last few path segments) of fp16 weights to quantize.
# Text decoder + vision encoder + projector + lm_head.
QUANTIZE_SUFFIXES = (
    # text decoder (LLaMA)
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
    # vision tower (SigLIP / Idefics3)
    ".self_attn.out_proj.weight",     # vision attention uses out_proj
    ".mlp.fc1.weight",                # vision MLP fc1 (768 → 3072)
    ".mlp.fc2.weight",                # vision MLP fc2 (3072 → 768)
    # projector
    ".modality_projection.proj.weight",
)

# Explicit per-key quantization (overrides suffix list when matched).
# lm_head sits at the model root in SmolVLM, no parent module path.
QUANTIZE_EXACT = (
    "lm_head.weight",
)


def is_quantizable(name: str, shape: list, quantize_embed: bool) -> bool:
    if len(shape) != 2:
        return False
    if quantize_embed and name == "model.text_model.embed_tokens.weight":
        return True
    if name in QUANTIZE_EXACT:
        return True
    return any(name.endswith(suf) for suf in QUANTIZE_SUFFIXES)


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
                    help="Also quantize model.text_model.embed_tokens.weight. "
                         "Off by default — keep fp16 to preserve the embedding "
                         "lookup's fast path. lm_head (independent in SmolVLM) "
                         "is quantized regardless.")
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
    families = {"text": 0, "vision": 0, "projector": 0, "lm_head": 0, "embed": 0}

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

            # one-shot sanity print on the first row of the first tensor
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
            # family bookkeeping
            if "text_model" in name and "embed_tokens" not in name:
                families["text"] += 1
            elif "vision_model" in name:
                families["vision"] += 1
            elif "connector" in name or "modality_projection" in name:
                families["projector"] += 1
            elif name == "lm_head.weight":
                families["lm_head"] += 1
            elif "embed_tokens" in name:
                families["embed"] += 1
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

    out_meta_dict = {
        "model_id": meta.get("model_id"),
        "format": "binary",
        "layout": "row_major",
        "dtype": "mixed",
        "quantization": {
            "enabled": True,
            "bits": 8,
            "method": "per-row symmetric int8 (text + vision Linear weights + projector + lm_head) + fp16 scales + fp16 passthrough (norms / biases / embed_tokens / patch_embedding / position_embedding)",
            "quantized_embed": bool(args.quantize_embed),
            "families": families,
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
    print(f"  families: {families}", file=sys.stderr)
    print(f"Original fp16 size of quantized weights: {total_fp16_quant_input_bytes/1e6:.1f} MB", file=sys.stderr)
    print(f"Total new file size:                     {cur_offset/1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
