#!/usr/bin/env python3
"""
Bake the SigLIP vision-tower position embedding for IMAGE_SIZE=384 (Fast mode).

Background
----------
SmolVLM's vision encoder ships with a learned [1024, 768] position-embedding
table corresponding to a 32x32 patch grid at IMAGE_SIZE=512 / PATCH_SIZE=16.
At IMAGE_SIZE=384 the grid is 24x24=576, so we bicubically resize the table
to match. This is exactly what HF Idefics3 does at runtime; we do it once,
offline, so the runtime stays a plain Embedding lookup.

The runtime (vision_pipeline.cpp) keeps BOTH variants and picks by --image-size:
    model.vision_model.embeddings.position_embedding.weight       (32x32, 512-mode)
    model.vision_model.embeddings.position_embedding.weight_384   (24x24 padded, 384-mode)

So this script APPENDS `weight_384` as a new tensor at the end of the .bin
(leaving every existing tensor + its offset untouched) and adds the matching
meta entry. The new tensor uses the same [1024, 768] slot: the first 576 rows
are the interpolated 24x24 grid, the trailing 448 rows are zero padding (never
indexed at runtime — the 24x24 patchify only emits position_ids 0..575).

NOTE: an earlier version of this script overwrote `.weight` in place, which
matched an older runtime that swapped the single table per mode. The current
runtime needs both tables present simultaneously, hence the append.

Run once:
    python3 scripts/rebake_pos_embed_384.py
Idempotent: re-running replaces an existing weight_384 instead of appending a
second one. Backs up the .bin and .meta.json once. Pass --restore to undo.
"""
import argparse, json, os, shutil, struct, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "weights" / "model.fp16.meta.json"
BIN  = ROOT / "weights" / "model.fp16.bin"
BAK  = BIN.with_suffix(".bin.bak")
META_BAK = Path(str(META) + ".bak")

POS_KEY  = "model.vision_model.embeddings.position_embedding.weight"
POS384_KEY = "model.vision_model.embeddings.position_embedding.weight_384"

def _cubic_kernel(x: np.ndarray, a: float = -0.5) -> np.ndarray:
    """Catmull-Rom (a=-0.5) bicubic convolution kernel — matches
    torch.nn.functional.interpolate(mode='bicubic')."""
    ax = np.abs(x)
    w = np.zeros_like(ax)
    m1 = ax < 1.0
    m2 = (ax >= 1.0) & (ax < 2.0)
    w[m1] = (a + 2) * ax[m1] ** 3 - (a + 3) * ax[m1] ** 2 + 1.0
    w[m2] = a * ax[m2] ** 3 - 5 * a * ax[m2] ** 2 + 8 * a * ax[m2] - 4 * a
    return w

def _bicubic_axis(src: np.ndarray, axis: int, new_n: int) -> np.ndarray:
    """Bicubic resize along a single axis. src axis has length old_n.
    align_corners=False, sampling formula:
        s = (i + 0.5) * old_n/new_n - 0.5
    For each output i: 4 taps at floor(s)-1..floor(s)+2 with cubic weights.
    """
    src = np.moveaxis(src, axis, 0)
    old_n = src.shape[0]
    s = (np.arange(new_n, dtype=np.float64) + 0.5) * (old_n / new_n) - 0.5
    s0 = np.floor(s).astype(np.int32)
    frac = (s - s0).astype(np.float32)
    # Tap offsets relative to s0: -1, 0, +1, +2
    # Distance from sample point to each tap (signed): tap - frac
    offsets = np.array([-1, 0, 1, 2], dtype=np.float32)
    # For each output row i, kernel weights for the 4 taps
    # weights[i, k] = cubic(offsets[k] - frac[i])
    w = _cubic_kernel(offsets[None, :] - frac[:, None])   # [new_n, 4]
    # Normalize: torch's bicubic does NOT normalize, but rounding errors
    # in float32 are tiny; we still skip normalization to match exactly.
    indices = s0[:, None] + np.array([-1, 0, 1, 2], dtype=np.int32)[None, :]  # [new_n, 4]
    indices = np.clip(indices, 0, old_n - 1)
    # Gather: out[i, ...] = sum_k w[i,k] * src[indices[i,k], ...]
    gathered = src[indices]                                # [new_n, 4, ...]
    w_b = w.reshape((new_n, 4) + (1,) * (gathered.ndim - 2)).astype(np.float32)
    out = (gathered * w_b).sum(axis=1)                     # [new_n, ...]
    return np.moveaxis(out, 0, axis)

def bicubic_resize(src: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """src: [H, W, C] float32 -> [new_h, new_w, C] float32.

    Matches torch.nn.functional.interpolate(x, size=(new_h,new_w),
    mode='bicubic', align_corners=False). Same kernel HF uses in
    Idefics3VisionEmbeddings.interpolate_pos_encoding.
    """
    H, W, _ = src.shape
    if H != new_h:
        src = _bicubic_axis(src, axis=0, new_n=new_h)
    if W != new_w:
        src = _bicubic_axis(src, axis=1, new_n=new_w)
    return src

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true", help="restore from .bak")
    ap.add_argument("--src-grid", type=int, default=32, help="source grid side (512/16=32)")
    ap.add_argument("--dst-grid", type=int, default=24, help="target grid side (384/16=24)")
    args = ap.parse_args()

    if args.restore:
        if not BAK.exists():
            sys.exit(f"no backup at {BAK}")
        shutil.copy2(BAK, BIN)
        if META_BAK.exists():
            shutil.copy2(META_BAK, META)
            print(f"restored {META} from {META_BAK}")
        print(f"restored {BIN} from {BAK}")
        return

    meta = json.loads(META.read_text())
    tensors = meta.get("tensors", meta)
    if POS_KEY not in tensors:
        sys.exit(f"{POS_KEY} not found in {META}")
    entry = tensors[POS_KEY]
    offset = int(entry["offset"])
    shape  = list(entry["shape"])
    dtype  = entry["dtype"]
    if dtype != "float16":
        sys.exit(f"expected float16, got {dtype}")
    if shape != [args.src_grid * args.src_grid, 768]:
        sys.exit(f"shape {shape} doesn't match expected [{args.src_grid**2}, 768]")
    H = W = args.src_grid
    C = shape[1]
    nbytes = H * W * C * 2

    # Back up the pristine .bin + meta ONCE so --restore works.
    if not BAK.exists():
        print(f"backing up {BIN} -> {BAK}")
        shutil.copy2(BIN, BAK)   # copy2 follows the symlink → real bytes
    if not META_BAK.exists():
        shutil.copy2(META, META_BAK)

    # Read the source 32x32 table from the CURRENT bin (the .weight tensor is
    # never modified, so reading from BIN is safe and idempotent).
    with open(BIN, "rb") as f:
        f.seek(offset)
        raw = f.read(nbytes)
    src_f16 = np.frombuffer(raw, dtype=np.float16).reshape(H, W, C).astype(np.float32)
    print(f"loaded position embedding [{H},{W},{C}] from offset {offset}")

    new_H = new_W = args.dst_grid
    print(f"bicubic interpolate (Catmull-Rom, align_corners=False) -> [{new_H},{new_W},{C}]")
    dst_f32 = bicubic_resize(src_f16, new_H, new_W)

    dst_f16 = dst_f32.astype(np.float16).reshape(new_H * new_W, C)
    print(f"  stats new: min={dst_f32.min():.4f} max={dst_f32.max():.4f} mean={dst_f32.mean():.4f}")
    print(f"  stats old: min={src_f16.min():.4f} max={src_f16.max():.4f} mean={src_f16.mean():.4f}")

    # Pack into a [1024, 768] slot: first 576 rows are the interpolated 24x24
    # grid, trailing 448 rows are zeros (never indexed —24x24 emits ids 0..575).
    padded = np.zeros((H * W, C), dtype=np.float16)
    padded[: new_H * new_W] = dst_f16
    new_bytes = padded.tobytes()

    # ── Append weight_384 as a NEW tensor at EOF (idempotent) ──────────────
    # model.fp16.bin may be a symlink into an archive (e.g. v2/) — never mutate
    # the target in place. Materialize a private real file, then append.
    if BIN.is_symlink():
        real = BIN.resolve()
        print(f"{BIN.name} is a symlink -> {real}; materializing a private copy")
        tmp = BIN.with_suffix(".bin.real")
        shutil.copy2(real, tmp)     # 512MB copy of resolved bytes
        BIN.unlink()
        tmp.rename(BIN)

    cur_size = BIN.stat().st_size
    if POS384_KEY in tensors:
        # Idempotent re-run: overwrite the existing weight_384 in place.
        new_off = int(tensors[POS384_KEY]["offset"])
        if int(tensors[POS384_KEY]["size_bytes"]) != len(new_bytes):
            sys.exit("existing weight_384 has unexpected size; restore + re-run")
        with open(BIN, "r+b") as f:
            f.seek(new_off)
            f.write(new_bytes)
        print(f"overwrote existing {POS384_KEY} ({len(new_bytes)} B) at offset {new_off}")
    else:
        new_off = cur_size
        with open(BIN, "ab") as f:
            f.write(new_bytes)
        tensors[POS384_KEY] = {
            "offset": new_off,
            "shape": [H * W, C],
            "dtype": "float16",
            "num_elements": H * W * C,
            "size_bytes": len(new_bytes),
        }
        print(f"appended {POS384_KEY} ({len(new_bytes)} B) at offset {new_off}")

    # Keep top-level size fields consistent (loader ignores them, but tools read them).
    final_size = BIN.stat().st_size
    if "bin_size_bytes" in meta: meta["bin_size_bytes"] = final_size
    if "total_bytes" in meta:    meta["total_bytes"] = final_size
    if "bin_sha256" in meta:
        import hashlib
        h = hashlib.sha256()
        with open(BIN, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        meta["bin_sha256"] = h.hexdigest()
    META.write_text(json.dumps(meta, indent=2))
    print(f"updated {META} (bin now {final_size} bytes)")
    print("done.")

if __name__ == "__main__":
    main()
