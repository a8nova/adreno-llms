"""Capture per-layer intermediates from the PyTorch reference vision pipeline
for byte-for-byte validation of the C++ port. Writes fp32 binaries under
reference/vision_dumps/ with a meta JSON listing shapes/dtypes.

Captures:
  - per-tile RGB after resize+normalize (input to patchify)
  - per-tile patch_embeds (after patch_embedding.weight @ patches + bias)
  - per-tile position-resized embeddings (pre-encoder input)
  - per-tile encoder layer 0 output, layer 5 output, layer 11 output
  - per-tile post_layernorm output
  - projector output per-tile (after pixel_unshuffle + linear_1 + GELU + linear_2)
  - final image_features (concatenated)
  - input_ids (from chat-template)
  - inputs_embeds post-masked-scatter (already captured by capture_inputs_embeds.py
    but we re-capture here so all dumps come from one consistent run)
  - tile_grid, spatial_shapes
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import transformers
from transformers import AutoConfig, AutoProcessor

def _shim():
    try:
        from packaging.version import parse as _vparse
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

_shim()

def save_t(out_dir: Path, name: str, tensor: torch.Tensor, meta: dict):
    arr = tensor.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
    (out_dir / f"{name}.bin").write_bytes(arr.tobytes(order="C"))
    meta[name] = {"shape": list(arr.shape), "dtype": "float32"}
    print(f"[capture] {name} shape={list(arr.shape)} bytes={arr.nbytes}", file=sys.stderr)

def _patch_model_dir(src: Path) -> Path:
    """Workaround: tokenizer_config.json has extra_special_tokens as a list
    (newer transformers format); older transformers expects a dict."""
    import os, shutil, json
    work = Path("/tmp/lfm2vl_model_patched")
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    for f in os.listdir(src):
        os.symlink(os.path.realpath(src / f), work / f)
    tc_path = work / "tokenizer_config.json"
    tc_path.unlink()
    tc = json.loads((src / "tokenizer_config.json").read_text())
    tc["extra_special_tokens"] = {}
    tc["tokenizer_class"] = "PreTrainedTokenizerFast"
    tc_path.write_text(json.dumps(tc, indent=2))
    return work

def main() -> int:
    project_dir = Path(__file__).resolve().parent.parent
    out_dir = project_dir / "reference" / "vision_dumps"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = project_dir / "fixtures" / "sample.jpg"
    model_id = "LiquidAI/LFM2.5-VL-450M"
    prompt = "Describe this image."

    print(f"[capture] image={image_path}", file=sys.stderr)
    from huggingface_hub import snapshot_download
    snap_path = Path(snapshot_download(model_id))
    work_path = _patch_model_dir(snap_path)
    print(f"[capture] using patched dir {work_path}", file=sys.stderr)
    from transformers import AutoImageProcessor, PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(work_path))
    image_processor = AutoImageProcessor.from_pretrained(str(work_path))
    print(f"[capture] tokenizer={type(tokenizer).__name__} img_proc={type(image_processor).__name__}", file=sys.stderr)
    cfg = AutoConfig.from_pretrained(str(work_path))
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        str(work_path), torch_dtype=torch.float32
    )
    model.eval()

    # Tie lm_head if needed
    try:
        in_emb = model.get_input_embeddings(); out_emb = model.get_output_embeddings()
        tie_flag = bool(getattr(model.config, "tie_word_embeddings", False) or
                        getattr(getattr(model.config, "text_config", None), "tie_word_embeddings", False))
        if tie_flag and in_emb is not None and out_emb is not None and \
           tuple(in_emb.weight.shape) == tuple(out_emb.weight.shape) and \
           in_emb.weight.data_ptr() != out_emb.weight.data_ptr():
            out_emb.weight = in_emb.weight
    except Exception:
        pass

    image = Image.open(str(image_path)).convert("RGB")
    print(f"[capture] image_size HxW = {image.size[1]}x{image.size[0]}", file=sys.stderr)
    img_outputs = image_processor(images=[image], return_tensors="pt")
    pixel_values = img_outputs["pixel_values"]
    spatial_shapes = img_outputs.get("spatial_shapes")
    pixel_attention_mask = img_outputs.get("pixel_attention_mask")
    # Build input_ids manually with image-token expansion
    # Find how many image tokens we need based on spatial_shapes
    image_token_id = int(getattr(cfg, "image_token_id", 396))
    # Number of image tokens = num_tiles * tokens_per_tile + thumbnail_tokens
    # tokens_per_tile = (tile_size/patch_size/downsample_factor)^2 = (512/16/2)^2 = 256
    # thumbnail_tokens = ceil(h/16/2)*ceil(w/16/2) (h,w from last spatial_shape)
    # The HF processor expansion handles this; emulate by counting from pixel_attention_mask
    N_total_patches = int(pixel_attention_mask.sum().item())  # over all tiles
    # After unshuffle: each spatial_shape (h,w) produces (h/2)*(w/2) tokens
    n_per_image = []
    for h, w in spatial_shapes.tolist():
        n_per_image.append((h // 2) * (w // 2))
    n_image_tokens = sum(n_per_image)
    print(f"[capture] spatial_shapes={spatial_shapes.tolist()} -> tokens per tile={n_per_image} total={n_image_tokens}", file=sys.stderr)
    # Build prompt: <|startoftext|><|im_start|>user\n<image>*N <text><|im_end|>\n<|im_start|>assistant\n
    bos = tokenizer.convert_tokens_to_ids("<|startoftext|>")
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    img_token = image_token_id
    newline = tokenizer.encode("\n", add_special_tokens=False)
    text_ids = tokenizer.encode(prompt, add_special_tokens=False)
    user_ids = tokenizer.encode("user", add_special_tokens=False)
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
    # Simple expansion: <bos><im_start>user\n<image>*N<text><im_end>\n<im_start>assistant\n
    ids = [bos, im_start] + user_ids + newline + [img_token]*n_image_tokens + text_ids + [im_end] + newline + [im_start] + assistant_ids + newline
    iids = torch.tensor([ids], dtype=torch.long)
    inputs = {"input_ids": iids, "pixel_values": pixel_values, "spatial_shapes": spatial_shapes, "pixel_attention_mask": pixel_attention_mask}

    meta = {"produced_by": "capture_vision_dumps.py", "model_id": model_id,
            "image_path": str(image_path.relative_to(project_dir)),
            "image_HxW": [image.size[1], image.size[0]],
            "prompt": prompt, "dtype": "float32"}

    # ── Save pixel_values, spatial_shapes, pixel_attention_mask from processor
    pv = inputs["pixel_values"]
    sps = inputs["spatial_shapes"]
    pam = inputs["pixel_attention_mask"]
    iids = inputs["input_ids"]
    print(f"[capture] pixel_values shape={tuple(pv.shape)} (N_tiles+thumb, max_patches, 768)", file=sys.stderr)
    print(f"[capture] spatial_shapes={sps.tolist()}", file=sys.stderr)
    print(f"[capture] input_ids len={iids.shape[-1]}", file=sys.stderr)
    save_t(out_dir, "pixel_values", pv, meta)
    save_t(out_dir, "spatial_shapes", sps.to(torch.float32), meta)
    save_t(out_dir, "pixel_attention_mask", pam.to(torch.float32), meta)
    save_t(out_dir, "input_ids", iids.to(torch.float32), meta)

    captured = {}

    # ── Hook the vision tower embedding output (post pos_embed)
    vt = model.model.vision_tower.vision_model
    emb = vt.embeddings
    def post_emb_hook(mod, inputs, output):
        captured["post_pos_embed"] = output.detach().to("cpu", dtype=torch.float32).contiguous()
    h0 = emb.register_forward_hook(post_emb_hook)

    # ── Hook encoder layers 0, 5, 11
    enc = vt.encoder
    layer_idxs_to_capture = [0, 5, 11]
    enc_handles = []
    for li in layer_idxs_to_capture:
        def make_hook(idx):
            def _h(mod, inp, out):
                # SigLIP layers return tuple; first elt is hidden_states
                t = out[0] if isinstance(out, tuple) else out
                captured[f"encoder_layer_{idx}_out"] = t.detach().to("cpu", dtype=torch.float32).contiguous()
            return _h
        enc_handles.append(enc.layers[li].register_forward_hook(make_hook(li)))

    # ── Hook post_layernorm
    def pln_hook(mod, inp, out):
        captured["post_layernorm"] = out.detach().to("cpu", dtype=torch.float32).contiguous()
    h_pln = vt.post_layernorm.register_forward_hook(pln_hook)

    # ── Hook patch_embedding output (pre pos)
    def pe_hook(mod, inp, out):
        captured["patch_embeds_no_pos"] = out.detach().to("cpu", dtype=torch.float32).contiguous()
    h_pe = emb.patch_embedding.register_forward_hook(pe_hook)

    # ── Hook multimodal projector input (post unpadding+reshape, pre unshuffle)
    proj = model.model.multi_modal_projector
    def proj_pre(mod, inputs):
        x = inputs[0]
        captured.setdefault("projector_inputs", []).append(x.detach().to("cpu", dtype=torch.float32).contiguous())
    h_proj_pre = proj.register_forward_pre_hook(proj_pre)
    def proj_post(mod, inp, out):
        captured.setdefault("projector_outputs", []).append(out.detach().to("cpu", dtype=torch.float32).contiguous())
    h_proj_post = proj.register_forward_hook(proj_post)

    # ── Hook first language-model decoder layer input (post masked_scatter)
    def lm_pre(mod, args, kwargs):
        if "inputs_embeds_merged" in captured: return
        t = args[0] if args else kwargs.get("hidden_states") or kwargs.get("inputs_embeds")
        if torch.is_tensor(t):
            captured["inputs_embeds_merged"] = t.detach().to("cpu", dtype=torch.float32).contiguous()
    lm = model.model.language_model
    h_lm = lm.layers[0].register_forward_pre_hook(lm_pre, with_kwargs=True)

    with torch.no_grad():
        _ = model(**inputs, use_cache=False)

    for h in [h0, h_pln, h_pe, h_proj_pre, h_proj_post, h_lm] + enc_handles:
        h.remove()

    # Save captured
    save_t(out_dir, "patch_embeds_no_pos", captured["patch_embeds_no_pos"], meta)
    save_t(out_dir, "post_pos_embed", captured["post_pos_embed"], meta)
    for li in layer_idxs_to_capture:
        save_t(out_dir, f"encoder_layer_{li}_out", captured[f"encoder_layer_{li}_out"], meta)
    save_t(out_dir, "post_layernorm", captured["post_layernorm"], meta)
    # Projector inputs/outputs are per-tile (list); concat in batch order
    if "projector_inputs" in captured and len(captured["projector_inputs"]) > 0:
        for i, t in enumerate(captured["projector_inputs"]):
            save_t(out_dir, f"projector_input_{i}", t, meta)
        for i, t in enumerate(captured["projector_outputs"]):
            save_t(out_dir, f"projector_output_{i}", t, meta)
    save_t(out_dir, "inputs_embeds_merged", captured["inputs_embeds_merged"], meta)

    # Also count image tokens for splice validation
    image_token_id = int(getattr(cfg, "image_token_id", 396))
    n_image_tokens = int((iids == image_token_id).sum().item())
    meta["image_token_id"] = image_token_id
    meta["n_image_tokens"] = n_image_tokens
    meta["seq_len"] = int(iids.shape[-1])
    print(f"[capture] image_token_id={image_token_id} count={n_image_tokens}", file=sys.stderr)

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[capture] wrote {out_dir}/meta.json", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
