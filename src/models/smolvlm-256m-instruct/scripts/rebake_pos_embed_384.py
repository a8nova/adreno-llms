#!/usr/bin/env python3
"""
Rebake the SigLIP vision-tower position embedding for IMAGE_SIZE=384.

Background
----------
SmolVLM's vision encoder ships with a learned [1024, 768] position-embedding
table corresponding to a 32x32 patch grid at IMAGE_SIZE=512 / PATCH_SIZE=16.
At IMAGE_SIZE=384 the grid is 24x24=576, so we bilinearly resize the table
to match. This is exactly what HF Idefics3 does at runtime; we just do it
once, offline, so the runtime stays a plain Embedding lookup.

We keep the on-disk slot at [1024, 768] (1,572,864 bytes) — only the first
576 rows are ever indexed at runtime, the trailing 448 rows are zero-padded
dead space. This avoids rewriting offsets for every tensor after it.

Run once:
    python3 scripts/rebake_pos_embed_384.py
Re-run is idempotent w.r.t. a fresh copy; we back up to .bak the first time
and refuse to clobber a backup. Pass --restore to undo.
"""
import argparse, json, os, shutil, struct, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "weights" / "model.fp16.meta.json"
BIN  = ROOT / "weights" / "model.fp16.bin"
BAK  = BIN.with_suffix(".bin.bak")

POS_KEY = "model.vision_model.embeddings.position_embedding.weight"

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

    if not BAK.exists():
        print(f"backing up {BIN} -> {BAK}")
        shutil.copy2(BIN, BAK)
    else:
        print(f"backup exists at {BAK} — reading from BACKUP to stay idempotent")

    src_path = BAK if BAK.exists() else BIN
    with open(src_path, "rb") as f:
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

    # Pack into [1024, 768] slot: first 576 rows are the new interpolated
    # values, trailing 448 rows are zeros (never indexed at runtime — the
    # patchify pipeline only emits position_ids 0..575 for a 24x24 grid).
    padded = np.zeros((H * W, C), dtype=np.float16)
    padded[: new_H * new_W] = dst_f16

    with open(BIN, "r+b") as f:
        f.seek(offset)
        f.write(padded.tobytes())
    print(f"wrote {padded.nbytes} bytes back to {BIN} at offset {offset}")
    print("done.")

if __name__ == "__main__":
    main()
