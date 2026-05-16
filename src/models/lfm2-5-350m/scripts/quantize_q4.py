#!/usr/bin/env python3
"""
Block-32 symmetric Q4_0 weight quantizer for LFM2.5-350M.

Reads weights/model.fp16.bin + meta.json, writes weights/model.q4.bin + meta.json.

Q4 scheme (block 32, symmetric):
- Group each row into blocks of 32 consecutive elements.
- For each block: scale = max(|values|) / 7.0  (symmetric, range -7..7).
- For each weight: q = clip(round(value/scale), -7, 7) ∈ {-7, …, 7}.
- Storage:  q_stored = q + 8  ∈ {1, …, 15} packed as 4-bit nibbles, 2 per byte.
- Dequant:  value ≈ (q_stored - 8) * scale.

Why scale/7 not /8: the round() can produce ±8 on the boundary; clipping
to ±7 keeps the stored byte in {1..15} and avoids the 0 / -8 boundary
ambiguity for the dequant kernel. Also, simple symmetric matches the int8
quant (per-row absmax/127) so output dynamic range mismatches stay small.

Output tensors per quantized weight `<name>`:
- `<name>`                row-major uint8 [N, K/2] — 2 packed nibbles per byte
- `<name>.scale`          fp16        [N, K/32]   — per-block scale

Same `--quantize-embed` and `--emit-lm-head-int8`-style flag (`--emit-lm-head-q4`)
patterns as the int8 quantizer, so the lm_head can stay fp16 in the embedding
lookup while the lm_head GEMV reads from a separate `lm_head.weight` q4 alias.
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
    ".self_attn.out_proj.weight",
    ".conv.in_proj.weight",
    ".conv.out_proj.weight",
    ".feed_forward.w1.weight",
    ".feed_forward.w2.weight",
    ".feed_forward.w3.weight",
)
BLOCK = 32


def is_quantizable(name: str, shape: list, quantize_embed: bool) -> bool:
    if len(shape) != 2:
        return False
    if quantize_embed and name == "model.embed_tokens.weight":
        return True
    return any(name.endswith(suf) for suf in QUANTIZE_SUFFIXES)


def fp16_bytes_to_array(data: bytes, n_elems: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float16, count=n_elems)


def quantize_row_q4(row_fp32: np.ndarray):
    """Quantize one row [K] into:
      packed:  uint8 [K/2]     packed nibbles (2 weights per byte)
      scales:  float16 [K/32]  per-block scale
    """
    K = row_fp32.shape[0]
    if K % BLOCK != 0:
        raise RuntimeError(f"Q4 requires K divisible by {BLOCK}, got {K}")
    n_blocks = K // BLOCK
    blocks = row_fp32.reshape(n_blocks, BLOCK)
    absmax = np.max(np.abs(blocks), axis=1)         # [n_blocks]
    scales = np.where(absmax > 0, absmax / 7.0, 1.0).astype(np.float32)
    inv = (1.0 / scales).reshape(n_blocks, 1)
    q = np.clip(np.round(blocks * inv), -7.0, 7.0).astype(np.int8)  # [n_blocks, 32]
    q_stored = (q + 8).astype(np.uint8)                              # 1..15
    # Pack 2 nibbles per byte: byte = (q[2i] & 0xF) | ((q[2i+1] & 0xF) << 4)
    flat = q_stored.reshape(-1)                                      # [K]
    low  = flat[0::2] & 0x0F
    high = flat[1::2] & 0x0F
    packed = (low | (high << 4)).astype(np.uint8)                    # [K/2]
    return packed, scales.astype(np.float16)


def _emit_alias_q4(arr_fp16: np.ndarray, n_rows: int, n_cols: int):
    packed_full = np.empty((n_rows, n_cols // 2), dtype=np.uint8)
    scales_full = np.empty((n_rows, n_cols // BLOCK), dtype=np.float16)
    for r in range(n_rows):
        p, s = quantize_row_q4(arr_fp16[r].astype(np.float32))
        packed_full[r] = p
        scales_full[r] = s
    return packed_full.tobytes(), scales_full.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default=str(Path(__file__).resolve().parent.parent / "weights"))
    ap.add_argument("--fp16-bin", default=None)
    ap.add_argument("--fp16-meta", default=None)
    ap.add_argument("--out-bin", default=None)
    ap.add_argument("--out-meta", default=None)
    ap.add_argument("--quantize-embed", action="store_true",
                    help="Also quantize model.embed_tokens.weight.")
    ap.add_argument("--emit-lm-head-q4", action="store_true",
                    help="Emit a separate `lm_head.weight` q4 alias WITHOUT touching "
                         "model.embed_tokens.weight (which keeps fp16 for the lookup path).")
    args = ap.parse_args()

    weights_dir = Path(args.weights_dir)
    fp16_bin = Path(args.fp16_bin) if args.fp16_bin else weights_dir / "model.fp16.bin"
    fp16_meta = Path(args.fp16_meta) if args.fp16_meta else weights_dir / "model.fp16.meta.json"
    out_bin = Path(args.out_bin) if args.out_bin else weights_dir / "model.q4.bin"
    out_meta = Path(args.out_meta) if args.out_meta else weights_dir / "model.q4.meta.json"

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
            if n_cols % BLOCK != 0:
                raise RuntimeError(f"{name}: K={n_cols} not divisible by {BLOCK}")
            arr = arr.reshape(n_rows, n_cols)
            packed = np.empty((n_rows, n_cols // 2), dtype=np.uint8)
            scales = np.empty((n_rows, n_cols // BLOCK), dtype=np.float16)
            for r in range(n_rows):
                p, s = quantize_row_q4(arr[r].astype(np.float32))
                packed[r] = p
                scales[r] = s

            if quantized_count == 0:
                # Sanity: dequant row 0 block 0 and check error vs fp16.
                fp32 = arr[0, :BLOCK].astype(np.float32)
                q_stored = np.zeros(BLOCK, dtype=np.int32)
                for i in range(BLOCK):
                    byte = packed[0, i // 2]
                    q_stored[i] = (byte & 0xF) if (i % 2 == 0) else ((byte >> 4) & 0xF)
                recon = (q_stored.astype(np.float32) - 8.0) * float(scales[0, 0])
                err = float(np.max(np.abs(recon - fp32)))
                rel = err / max(1e-6, float(np.max(np.abs(fp32))))
                print(f"  sanity[{name}] r0b0: max_abs_err={err:.4g} rel={rel:.2%}", file=sys.stderr)

            p_bytes = packed.tobytes()
            s_bytes = scales.tobytes()
            out_tensors[name] = {
                "offset": cur_offset,
                "shape": [n_rows, n_cols // 2],
                "dtype": "q4_packed",
                "num_elements": n_rows * (n_cols // 2),
                "size_bytes": len(p_bytes),
            }
            out_chunks.append(p_bytes); cur_offset += len(p_bytes)
            out_tensors[name + ".scale"] = {
                "offset": cur_offset,
                "shape": [n_rows, n_cols // BLOCK],
                "dtype": "float16",
                "num_elements": n_rows * (n_cols // BLOCK),
                "size_bytes": len(s_bytes),
            }
            out_chunks.append(s_bytes); cur_offset += len(s_bytes)

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

    lm_head_emitted = False
    if args.emit_lm_head_q4:
        embed_meta = tensors_in.get("model.embed_tokens.weight")
        if embed_meta is None or embed_meta["dtype"] != "float16":
            print("warn: --emit-lm-head-q4 needs fp16 model.embed_tokens.weight", file=sys.stderr)
        else:
            shape = embed_meta["shape"]
            offset = embed_meta["offset"]
            nbytes = embed_meta["size_bytes"]
            nelems = embed_meta["num_elements"]
            arr = fp16_bytes_to_array(raw[offset:offset + nbytes], nelems).reshape(shape[0], shape[1])
            p_bytes, s_bytes = _emit_alias_q4(arr, shape[0], shape[1])
            out_tensors["lm_head.weight"] = {
                "offset": cur_offset,
                "shape": [shape[0], shape[1] // 2],
                "dtype": "q4_packed",
                "num_elements": shape[0] * (shape[1] // 2),
                "size_bytes": len(p_bytes),
            }
            out_chunks.append(p_bytes); cur_offset += len(p_bytes)
            out_tensors["lm_head.weight.scale"] = {
                "offset": cur_offset,
                "shape": [shape[0], shape[1] // BLOCK],
                "dtype": "float16",
                "num_elements": shape[0] * (shape[1] // BLOCK),
                "size_bytes": len(s_bytes),
            }
            out_chunks.append(s_bytes); cur_offset += len(s_bytes)
            lm_head_emitted = True
            print(f"  emitted lm_head.weight q4 alias ({(len(p_bytes)+len(s_bytes))/1e6:.1f} MB)", file=sys.stderr)

    out_meta_dict = {
        "model_id": meta.get("model_id"),
        "format": "binary",
        "layout": "row_major",
        "dtype": "mixed",
        "quantization": {
            "enabled": True,
            "bits": 4,
            "block_size": BLOCK,
            "method": (
                "block-32 symmetric q4 (linear weights in conv + attention blocks): "
                "scale = absmax / 7, stored q ∈ {1..15} (q-8 ∈ {-7..7}), 2 nibbles per byte"
            ),
            "quantized_embed": bool(args.quantize_embed),
            "lm_head_q4_alias": lm_head_emitted,
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
