#!/usr/bin/env python3
"""Convert a Kokoro voice pack (.pt from hexgrad/Kokoro-82M/voices/) into the
raw float32 .bin the on-device runtime loads.

A Kokoro voice is a [510, 1, 256] float32 style-embedding tensor (510 = token-
length index, 256 = style dim; the model picks row = len(tokens)). The runtime
(`main.cpp`) loads it as a flat 510*256 float32 buffer — see the
`vp.size() != 510*256` check. This writes exactly that: [510,256] little-endian
float32, byte-identical to the bundled `assets/voice_pack_af_heart.bin`.

Usage:
    python convert_voice.py <voice.pt> <out.bin>
"""
import sys
import numpy as np
import torch

EXPECT_ROWS, STYLE_DIM = 510, 256


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    pt_path, bin_path = sys.argv[1], sys.argv[2]

    v = torch.load(pt_path, map_location="cpu", weights_only=True)
    arr = v.detach().to(torch.float32).reshape(EXPECT_ROWS, STYLE_DIM).contiguous().numpy()
    assert arr.size == EXPECT_ROWS * STYLE_DIM, f"unexpected size {arr.size}"
    arr.astype("<f4").tofile(bin_path)
    print(f"  {pt_path} -> {bin_path} ({arr.size * 4} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
