#!/usr/bin/env python3
"""
qcheck_generate_features.py — for ONE (image, prompt) pair:
  - run the SmolVLM-256M processor (manually stacking tile tensors to avoid the
    BatchFeature shape-mismatch ValueError seen on transformers >=4.46)
  - dump the resulting input_ids to reference/test_input_ids.bin (int32 little-endian)
  - run the vision tower + connector via forward hooks
  - dump the connector output to reference/layers/model_connector_output.bin (fp32)
  - print the HF reference caption + the deterministic generated token ids

The C++ binary auto-loads both .bin files when present.

Usage:
  scripts/qcheck_generate_features.py <image_path> <prompt> <max_new_tokens>
"""

import os
import sys
import struct
import warnings
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

warnings.filterwarnings("ignore")

def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    image_path = Path(sys.argv[1])
    prompt = sys.argv[2]
    max_new_tokens = int(sys.argv[3])
    if not image_path.exists():
        print(f"image not found: {image_path}", file=sys.stderr)
        return 2

    project_dir = Path(__file__).resolve().parent.parent
    out_layers = project_dir / "reference" / "layers"
    out_layers.mkdir(parents=True, exist_ok=True)
    out_token_ids = project_dir / "reference" / "test_input_ids.bin"
    out_connector = out_layers / "model_connector_output.bin"

    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    print(f"[qcheck-py] loading processor + model ({model_id})...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    # Load image; force RGB.
    img = Image.open(str(image_path)).convert("RGB")
    print(f"[qcheck-py] image: {img.size} mode={img.mode}", file=sys.stderr)

    # Build the chat-template-formatted prompt (image-placeholder expansion).
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Call processor WITHOUT return_tensors="pt" to avoid the BatchFeature
    # auto-batching failure (Idefics3 image_processor returns a list of
    # variable-length tile lists per image). Manually stack tile pixel_values
    # into a single 5-D tensor before forward.
    out = processor(text=prompt_text, images=[img])

    # `pixel_values` shape after manual stack: [batch=1, n_tiles, C, H, W].
    pv = out["pixel_values"]
    # pv is list[list[np.ndarray | torch.Tensor]] in the new transformers; the
    # inner list is one tensor per tile of one image.
    pv_list = pv[0] if isinstance(pv, list) else pv
    if isinstance(pv_list, list):
        # Each element is a (C, H, W) per-tile tensor.
        tile_tensors = []
        for t in pv_list:
            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t)
            tile_tensors.append(t)
        pixel_values = torch.stack(tile_tensors, dim=0).unsqueeze(0).float()
    else:
        # Already a tensor / ndarray.
        if isinstance(pv_list, np.ndarray):
            pv_list = torch.from_numpy(pv_list)
        pixel_values = pv_list.unsqueeze(0).float() if pv_list.dim() == 4 else pv_list.float()
    print(f"[qcheck-py] pixel_values: {tuple(pixel_values.shape)}", file=sys.stderr)

    # pixel_attention_mask: same per-tile-list shape; reshape to [B, T, H, W].
    pam = out.get("pixel_attention_mask")
    pam_t = None
    if pam is not None:
        pam_list = pam[0] if isinstance(pam, list) else pam
        if isinstance(pam_list, list):
            tile_masks = [torch.from_numpy(m) if isinstance(m, np.ndarray) else m for m in pam_list]
            pam_t = torch.stack(tile_masks, dim=0).unsqueeze(0)
        else:
            if isinstance(pam_list, np.ndarray):
                pam_list = torch.from_numpy(pam_list)
            pam_t = pam_list.unsqueeze(0) if pam_list.dim() == 3 else pam_list

    input_ids = torch.tensor(out["input_ids"][0], dtype=torch.long).unsqueeze(0)
    attention_mask = torch.tensor(out["attention_mask"][0], dtype=torch.long).unsqueeze(0)
    print(f"[qcheck-py] input_ids: {tuple(input_ids.shape)}", file=sys.stderr)

    # Persist token ids — the C++ main.cpp auto-loads this.
    with open(out_token_ids, "wb") as f:
        f.write(input_ids[0].numpy().astype("<i4").tobytes())

    # Hook the connector output so we can dump it to disk.
    captured = {}
    def hook(_module, _inputs, output):
        captured["connector_output"] = output.detach().cpu().float()
    h = model.model.connector.register_forward_hook(hook)

    # Generate. Pass through pixel_attention_mask if available.
    gen_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                      pixel_values=pixel_values,
                      max_new_tokens=max_new_tokens, do_sample=False)
    if pam_t is not None:
        gen_kwargs["pixel_attention_mask"] = pam_t
    with torch.no_grad():
        gen_ids = model.generate(**gen_kwargs)
    h.remove()

    if "connector_output" not in captured:
        print("[qcheck-py] WARNING: connector hook didn't fire — model.connector may not match", file=sys.stderr)
    else:
        co = captured["connector_output"]
        # The C++ binary expects a flat fp32 stream [N, D] where D=HIDDEN_SIZE=576.
        # connector_output may be shape [B, N, D] or already [N, D].
        if co.dim() == 3:
            co = co.reshape(-1, co.shape[-1])
        with open(out_connector, "wb") as f:
            f.write(co.numpy().astype("<f4").tobytes())
        print(f"[qcheck-py] wrote {out_connector} ({co.shape[0]} tokens × {co.shape[1]} dims)", file=sys.stderr)

    # Print the reference output (just the generated suffix).
    new_ids = gen_ids[0, input_ids.shape[1]:].tolist()
    ref_text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    print(ref_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
