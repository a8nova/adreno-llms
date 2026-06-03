"""Capture the merged text+vision `inputs_embeds` tensor that the LFM2-VL
language stack consumes at prefill, and write it as a binary the C++ port
can read.

This is a one-shot diagnostic: if the C++ language stack is correct, swapping
on-device vision features for these PyTorch-computed features should make
the binary produce the SAME caption as the reference. If the captions still
diverge, the bug is in the language stack (decoder / RMSNorm / attention / LM head).

Output:
  reference/inputs_embeds.bin       fp32 little-endian, [seq_len, hidden] row-major
  reference/inputs_embeds.meta.json {seq_len, hidden, dtype, image_token_id}
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _install_torch_compat_shims():
    """Bridge torch 2.2 -> 2.4 API gaps (transformers 5.x calls
    torch.is_autocast_enabled(device_type) which doesn't exist on 2.2)."""
    try:
        try:
            from packaging.version import parse as _vparse
        except Exception:
            from distutils.version import LooseVersion as _vparse
        tv = _vparse(str(torch.__version__).split("+")[0])
        if tv < _vparse("2.4"):
            _orig_iae = torch.is_autocast_enabled
            torch.is_autocast_enabled = (lambda *a, **k: bool(_orig_iae()))
            if not hasattr(torch, "get_autocast_dtype"):
                def _gad(device_type=None):
                    if device_type == "cuda" and hasattr(torch, "get_autocast_gpu_dtype"):
                        return torch.get_autocast_gpu_dtype()
                    if device_type == "cpu" and hasattr(torch, "get_autocast_cpu_dtype"):
                        return torch.get_autocast_cpu_dtype()
                    return torch.float32
                torch.get_autocast_dtype = _gad
            _orig_sae = torch.set_autocast_enabled
            def _sae(*a, **k):
                enabled = k.get("enabled", a[-1] if a else False)
                return _orig_sae(bool(enabled))
            torch.set_autocast_enabled = _sae
    except Exception:
        pass


_install_torch_compat_shims()

from PIL import Image
import transformers
from transformers import AutoConfig, AutoProcessor


def main() -> int:
    project_dir = Path(__file__).resolve().parent.parent
    out_dir = project_dir / "reference"
    image_path = project_dir / "fixtures" / "sample.jpg"
    model_id = "LiquidAI/LFM2.5-VL-450M"
    prompt = "Describe this image."

    print(f"[capture] image={image_path}", file=sys.stderr)
    print(f"[capture] loading {model_id}", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(model_id)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.float32, trust_remote_code=False
    )
    model.eval()

    # The LFM2VL wrapper sometimes leaves lm_head untied — apply the same
    # generic tie pattern used by _run_reference.py so this capture matches.
    try:
        in_emb = model.get_input_embeddings()
        out_emb = model.get_output_embeddings()
        tie_flag = bool(getattr(model.config, "tie_word_embeddings", False) or
                        getattr(getattr(model.config, "text_config", None), "tie_word_embeddings", False))
        if (tie_flag and in_emb is not None and out_emb is not None
                and hasattr(in_emb, "weight") and hasattr(out_emb, "weight")
                and tuple(in_emb.weight.shape) == tuple(out_emb.weight.shape)
                and in_emb.weight.data_ptr() != out_emb.weight.data_ptr()):
            out_emb.weight = in_emb.weight
            print("[capture] tied lm_head <- embed_tokens", file=sys.stderr)
    except Exception as e:
        print(f"[capture] tie warn: {e}", file=sys.stderr)

    image = Image.open(str(image_path)).convert("RGB")
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt")

    captured = {}

    def pre_hook(module, args, kwargs):
        if "inputs_embeds" in captured:
            return  # only keep the prefill (first call)
        # Lfm2DecoderLayer.forward signature: (hidden_states, ...)
        if args:
            t = args[0]
        else:
            t = kwargs.get("hidden_states") or kwargs.get("inputs_embeds")
        if torch.is_tensor(t):
            captured["inputs_embeds"] = t.detach().to("cpu", dtype=torch.float32).contiguous()

    # Hook the first language-model decoder layer. The merged
    # (text + vision-spliced) inputs_embeds enters this layer at prefill.
    lm = model.model.language_model  # nn.Module containing layers[]
    target = lm.layers[0]
    handle = target.register_forward_pre_hook(pre_hook, with_kwargs=True)

    with torch.no_grad():
        _ = model(**inputs, use_cache=False)
    handle.remove()

    t = captured["inputs_embeds"]
    print(f"[capture] inputs_embeds shape={tuple(t.shape)} dtype={t.dtype}", file=sys.stderr)
    # Squeeze batch dim if present.
    if t.dim() == 3 and t.size(0) == 1:
        t = t[0]
    assert t.dim() == 2, f"expected [seq_len, hidden], got {tuple(t.shape)}"
    seq_len, hidden = int(t.size(0)), int(t.size(1))

    arr = t.numpy().astype(np.float32, copy=False)
    bin_path = out_dir / "inputs_embeds.bin"
    arr.tofile(str(bin_path))
    print(f"[capture] wrote {bin_path} ({arr.nbytes} bytes, [{seq_len}, {hidden}] fp32)", file=sys.stderr)

    meta = {
        "produced_by": "capture_inputs_embeds.py",
        "model_id": model_id,
        "seq_len": seq_len,
        "hidden": hidden,
        "dtype": "float32",
        "image_token_id": int(getattr(cfg, "image_token_id", 396)),
    }
    (out_dir / "inputs_embeds.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[capture] OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
