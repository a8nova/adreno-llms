"""Generate single-tile ground-truth input_ids for tokenizer parity testing.

This script forces the SmolVLM/Idefics3 processor to run in single-tile mode
(`do_image_splitting=False`), so the on-device builder in src/tokenizer.cpp
(build_vlm_prompt) can be validated id-for-id against the HF processor's
expansion of a (image, prompt) pair.

Output:
  reference/expected_prompt_ids.bin   int32 little-endian, one id per element

Usage:
  python reference/_gen_prompt_ids.py \\
    --model HuggingFaceTB/SmolVLM-256M-Instruct \\
    --image fixtures/sample.jpg \\
    --prompt "What is in this image?"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--image", required=True, help="path to test image")
    ap.add_argument("--prompt", required=True, help="user text prompt")
    ap.add_argument("--out", default=None, help="output bin path (default: reference/expected_prompt_ids.bin)")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_path = Path(args.out) if args.out else script_dir / "expected_prompt_ids.bin"

    processor = AutoProcessor.from_pretrained(args.model)

    # Force single-tile mode: only the global image view (one 512x512 tile)
    # so the placeholder run is exactly 64 image tokens, matching the
    # on-device pixel_shuffle output rows.
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_image_splitting = False

    image = Image.open(args.image).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": args.prompt},
        ],
    }]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt")

    ids = inputs["input_ids"][0].tolist()
    print(f"[gen_prompt_ids] sequence length: {len(ids)}", file=sys.stderr)
    # Quick sanity: count IMAGE_TOKEN (49190) — must equal 64 for single-tile.
    n_img = sum(1 for i in ids if i == 49190)
    print(f"[gen_prompt_ids] image-token count: {n_img}", file=sys.stderr)
    if n_img != 64:
        print(f"[gen_prompt_ids] WARNING: expected 64 image tokens for single-tile, got {n_img}. "
              f"Processor's do_image_splitting flag may not have stuck.", file=sys.stderr)

    np.asarray(ids, dtype=np.int32).tofile(str(out_path))
    print(f"[gen_prompt_ids] wrote {len(ids)} ids to {out_path}", file=sys.stderr)

    # Also dump a short textual preview to make debugging quick.
    preview = ids[:8] + ["..."] + ids[-8:]
    print(f"[gen_prompt_ids] preview: {preview}", file=sys.stderr)


if __name__ == "__main__":
    main()
