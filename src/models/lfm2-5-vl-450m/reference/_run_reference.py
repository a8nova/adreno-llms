# Reference for VLM port: LiquidAI/LFM2.5-VL-450M
#
# Generates the HuggingFace reference output the C++ binary must match.
# Produces:
#   reference/reference_tokens.json          - canonical token + metadata bundle (produced_by stamp)
#   reference/reference_caption_tokens.json  - bare token id list (back-compat for vlm/eval.ts)
#   reference/reference_caption_text.txt    - decoded caption (diagnostic)
#   reference/layers/<dump_name>_output.bin  - raw fp32 dumps, one per graph.json node
#   reference/layers/manifest.json           - _capture_meta map: dump_name -> {module_path, shape, dtype}
#
# This script is invoked by GenerateReference (TS tool). It is NOT meant
# to be hand-edited: the template lives in nnopt under
# extensions/cli/prompts/templates/references/vlm.tmpl and any local
# edits are clobbered on the next GenerateReference call.
#
# argv (positional only; matches generateReferenceTs.ts spawn convention):
#   1: model_id                 e.g. "HuggingFaceTB/SmolVLM-256M-Instruct"
#   2: project_dir              absolute path to the port workspace
#   3: prompt                   text prompt (the image-token is added per the chat template)
#   4: max_new_tokens           int
#   5: source_path              path to local modeling source (unused for VLM today)
#   6: tokenizer_repo           tokenizer override repo (unused for VLM today)
#
# Image input: read {project_dir}/fixtures/sample.jpg (staged by GenerateReference).
# If absent, fall back to a runtime-generated 256x256 solid-gray JPEG at
# reference/_auto_dummy.jpg so the pipeline does not crash on a missing fixture.

import json
import sys
from pathlib import Path

import numpy as np
import torch


def _install_torch_compat_shims():
    """Bridge torch API signature changes when the installed transformers
    expects a newer torch than the platform can provide. Tier 2 resolver
    picks the newest transformers whose static is_torch_available() passes;
    modeling code may still call torch 2.4+ APIs (torch.is_autocast_enabled
    gained a device_type arg in 2.4). Generic across architectures."""
    try:
        try:
            from packaging.version import parse as _vparse
        except Exception:
            from distutils.version import LooseVersion as _vparse
        tv = _vparse(str(torch.__version__).split("+")[0])
        if tv < _vparse("2.4"):
            try:
                _orig_iae = torch.is_autocast_enabled
                torch.is_autocast_enabled = (lambda *a, **k: bool(_orig_iae()))
            except Exception:
                pass
            try:
                if not hasattr(torch, "get_autocast_dtype"):
                    def _gad(device_type=None):
                        if device_type == "cuda" and hasattr(torch, "get_autocast_gpu_dtype"):
                            return torch.get_autocast_gpu_dtype()
                        if device_type == "cpu" and hasattr(torch, "get_autocast_cpu_dtype"):
                            return torch.get_autocast_cpu_dtype()
                        return torch.float32
                    torch.get_autocast_dtype = _gad
            except Exception:
                pass
            try:
                _orig_sae = torch.set_autocast_enabled
                def _sae(*a, **k):
                    enabled = k.get("enabled", a[-1] if a else False)
                    return _orig_sae(bool(enabled))
                torch.set_autocast_enabled = _sae
            except Exception:
                pass
    except Exception:
        pass


_install_torch_compat_shims()

from PIL import Image
import transformers
from transformers import AutoConfig, AutoProcessor


def _resolve_image_path(project_dir: Path) -> Path:
    # Hard-fail when the user fixture is missing. A silent gray-dummy fallback
    # used to be the prior behavior, but the C++ binary on-device runs against
    # whatever was actually deployed under fixtures/ — almost never that
    # gray dummy. Reference and binary then "diverge" on pixels rather than
    # porting correctness, and the agent chases a fake bug. Surface the
    # missing-fixture condition loudly so the agent stages the real image.
    staged = project_dir / "fixtures" / "sample.jpg"
    if staged.exists():
        return staged
    raise FileNotFoundError(
        f"VLM reference requires a real image fixture but none was found at {staged}. "
        f"Stage the test image at fixtures/sample.jpg (the same file the C++ binary "
        f"will receive at Infer time) and re-run GenerateReference. Do NOT substitute "
        f"a placeholder — reference and on-device runs must operate on identical pixels."
    )


def _build_name_index(root: torch.nn.Module):
    """Build a lookup from graph-node id -> live module.

    ExtractGraph is allowed to root paths at an inner submodule (e.g.
    Idefics3Model) while AutoModelForVision2Seq loads the outer wrapper
    (Idefics3ForConditionalGeneration). Rather than guess at the wrapper
    prefix, we index every named_modules name by its suffix segments.

    For each live module `full_name` (e.g. "model.text_model.embed_tokens"),
    we record every dotted suffix ("text_model.embed_tokens",
    "embed_tokens"). Lookup picks the longest unique match. Numeric
    segments inside dotted ids stay intact — named_modules already emits
    them as "...layers.0.self_attn..." which is exactly the format
    graph.json uses.
    """
    suffix_to_full: dict = {}
    ambiguous: set = set()
    for full_name, mod in root.named_modules():
        if not full_name:
            continue
        parts = full_name.split(".")
        for i in range(len(parts)):
            suffix = ".".join(parts[i:])
            if suffix in suffix_to_full and suffix_to_full[suffix] is not mod:
                ambiguous.add(suffix)
            else:
                suffix_to_full[suffix] = mod
    return suffix_to_full, ambiguous


def _walk_module_path(name_index, dotted: str):
    """Resolve a graph.json node id against the named_modules suffix index."""
    return name_index.get(dotted)


def _load_graph(project_dir: Path):
    gp = project_dir / ".nnport" / "graph.json"
    if not gp.exists():
        return []
    try:
        data = json.loads(gp.read_text())
    except Exception:
        return []
    nodes = data.get("nodes") if isinstance(data, dict) else None
    if not isinstance(nodes, list):
        return []
    out = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = n.get("id") or n.get("name") or n.get("module_path")
        if isinstance(nid, str) and nid:
            out.append({"id": nid, "class": n.get("class") or n.get("op") or ""})
    return out


def _register_hooks(model: torch.nn.Module, nodes, dumps: dict):
    handles = []
    captured = []
    skipped = []
    name_index, ambiguous = _build_name_index(model)
    if ambiguous:
        print(
            f"[reference] WARN: {len(ambiguous)} graph node ids ambiguous in named_modules; "
            f"first 5 = {sorted(ambiguous)[:5]}",
            file=sys.stderr,
        )
    for node in nodes:
        nid = node["id"]
        target = _walk_module_path(name_index, nid)
        if target is None or not isinstance(target, torch.nn.Module):
            skipped.append(nid)
            continue
        dump_name = nid.replace(".", "_")

        def make_hook(_dump_name: str, _module_path: str):
            def _hook(_mod, _inp, output):
                # AUTOREGRESSIVE PREFILL-ONLY CAPTURE.
                # The C++ binary dumps each layer's pass=0 output at PREFILL
                # shape ([1, seq_len, hidden]) and pass>=1 at decode-step
                # shape ([1, 1, hidden]). model.generate() fires this hook
                # 1 + max_new_tokens times — once for prefill, once per
                # decode step. Capturing every call overwrites the prefill
                # tensor with the last decode-step capture (seq_len=1),
                # which is what produced the SmolVLM SxS "ratio=5" alignment
                # failure: ref=[1,1,hidden]=576 elems vs C++ pass0=
                # [seq,hidden]=2880 elems for a 5-token prompt.
                #
                # Fix: keep the FIRST capture per layer (the prefill). Drop
                # subsequent decode-step calls. This makes ref/`*__pass0.bin`
                # alignment correct for any autoregressive model — text,
                # VLM, ASR-AR, TTS-AR all benefit.
                if _dump_name in dumps:
                    return
                t = output
                # Unwrap composite outputs in order of specificity. HF modules
                # routinely return ModelOutput dataclasses (BaseModelOutput,
                # BaseModelOutputWithPooling, etc.) whose primary tensor lives
                # at .last_hidden_state, .logits, or .hidden_states. The
                # previous "tuple-only" unwrap left these unhooked — for
                # SmolVLM that meant Idefics3Encoder (the wrapper around 12
                # encoder layers) had no reference dump, blocking PortNode
                # indefinitely. (May-2026 fix.)
                if isinstance(t, tuple) and len(t) > 0:
                    t = t[0]
                elif hasattr(t, "last_hidden_state") and torch.is_tensor(getattr(t, "last_hidden_state")):
                    t = t.last_hidden_state
                elif hasattr(t, "logits") and torch.is_tensor(getattr(t, "logits")):
                    t = t.logits
                elif hasattr(t, "hidden_states") and isinstance(getattr(t, "hidden_states"), (tuple, list)) \
                        and len(t.hidden_states) > 0 and torch.is_tensor(t.hidden_states[-1]):
                    t = t.hidden_states[-1]
                elif isinstance(t, dict):
                    for _k in ("last_hidden_state", "logits", "hidden_states"):
                        _v = t.get(_k)
                        if torch.is_tensor(_v):
                            t = _v
                            break
                        if isinstance(_v, (tuple, list)) and _v and torch.is_tensor(_v[-1]):
                            t = _v[-1]
                            break
                if not torch.is_tensor(t):
                    return
                arr = t.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
                dumps[_dump_name] = {
                    "module_path": _module_path,
                    "shape": list(arr.shape),
                    "dtype": "float32",
                    "data": arr,
                }
            return _hook

        handles.append(target.register_forward_hook(make_hook(dump_name, nid)))
        captured.append(nid)
    return handles, captured, skipped


def main() -> int:
    if len(sys.argv) < 5:
        print(
            "usage: _run_reference.py <model_id> <project_dir> <prompt> <max_new_tokens> [source_path] [tokenizer_repo]",
            file=sys.stderr,
        )
        return 2

    model_id = sys.argv[1]
    project_dir = Path(sys.argv[2]).resolve()
    prompt = sys.argv[3]
    max_new_tokens = int(sys.argv[4])
    # sys.argv[5] = source_path (unused for VLM today)
    # sys.argv[6] = tokenizer_repo (unused for VLM today)

    out_dir = project_dir / "reference"
    layers_dir = out_dir / "layers"
    out_dir.mkdir(parents=True, exist_ok=True)
    layers_dir.mkdir(parents=True, exist_ok=True)

    image_path = _resolve_image_path(project_dir)
    print(f"[reference] Image: {image_path}", file=sys.stderr)
    print(f"[reference] Loading processor + model: {model_id}", file=sys.stderr)

    processor = AutoProcessor.from_pretrained(model_id)

    # Resolve the correct AutoModel class via the shared resolver — VLMs can
    # be Vision2Seq, ImageTextToText, or something custom via auto_map.
    sys.path.insert(0, str(project_dir / ".nnport"))
    try:
        from _nnopt_auto_class_resolver import resolve_auto_class
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        cls_name = resolve_auto_class(model_id, cfg)
    except Exception as e:
        print(f"[reference] resolver failed, defaulting to AutoModelForVision2Seq: {e}", file=sys.stderr)
        cls_name = "AutoModelForVision2Seq"
    print(f"[reference] Loader class: {cls_name}", file=sys.stderr)
    loader_cls = getattr(transformers, cls_name)
    model = loader_cls.from_pretrained(
        model_id, torch_dtype=torch.float32, trust_remote_code=False
    )
    model.eval()

    # Force weight tying when the config requests it. Some VLM wrapper
    # classes return get_output_embeddings()=None or skip the tie inside
    # PreTrainedModel.post_init() (transformers 5.0.0 regression for several
    # VLM families), which leaves `lm_head.weight` randomly initialized when
    # the checkpoint relied on tying. The symptom is fluent-but-random
    # generation. The fix is GENERIC: ask HF for the input/output embedding
    # modules through the documented `get_input_embeddings`/
    # `get_output_embeddings` API, and if both exist with matching shapes,
    # share the underlying parameter. Works for any (text + VL) model where
    # the wrapper exposes those methods. Falls back to model.tie_weights()
    # for classes that handle tying correctly themselves.
    try:
        _tie_flag = False
        for _cfg_obj in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
            if _cfg_obj is not None and getattr(_cfg_obj, "tie_word_embeddings", False):
                _tie_flag = True
                break
        if _tie_flag:
            try:
                _in_emb = model.get_input_embeddings()
            except Exception:
                _in_emb = None
            try:
                _out_emb = model.get_output_embeddings()
            except Exception:
                _out_emb = None
            if (
                _in_emb is not None and _out_emb is not None
                and hasattr(_in_emb, "weight") and hasattr(_out_emb, "weight")
                and tuple(_in_emb.weight.shape) == tuple(_out_emb.weight.shape)
                and _in_emb.weight.data_ptr() != _out_emb.weight.data_ptr()
            ):
                _out_emb.weight = _in_emb.weight
                print(
                    f"[reference] manually tied {type(_out_emb).__name__}.weight ← "
                    f"{type(_in_emb).__name__}.weight shape={tuple(_in_emb.weight.shape)}",
                    file=sys.stderr,
                )
            else:
                # Fall back to the canonical tie helper if API exposure is
                # already correct OR if shapes don't match (skip in the
                # latter case — caller will see lm_head MISSING in the load
                # report).
                try:
                    model.tie_weights()
                    print(f"[reference] tie_weights() called", file=sys.stderr)
                except Exception:
                    pass
    except Exception as _tie_err:
        print(f"[reference] WARN: weight-tying step failed: {_tie_err}", file=sys.stderr)

    image = Image.open(str(image_path)).convert("RGB")

    if hasattr(processor, "apply_chat_template"):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    else:
        prompt_text = f"<image>{prompt}"

    inputs = processor(text=prompt_text, images=[image], return_tensors="pt")

    nodes = _load_graph(project_dir)
    dumps: dict = {}
    handles, captured, skipped = _register_hooks(model, nodes, dumps)
    if nodes:
        print(
            f"[reference] graph.json: {len(nodes)} nodes; hooked={len(captured)}; skipped={len(skipped)}",
            file=sys.stderr,
        )
        if skipped:
            print(f"[reference] skipped (path did not resolve): {skipped[:10]}", file=sys.stderr)
    else:
        print(
            "[reference] no .nnport/graph.json found - skipping per-node hooks. "
            "Run ExtractGraph to enable layer-level dumps.",
            file=sys.stderr,
        )

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        caption_token_ids = generated[0, prompt_len:].tolist()

    for h in handles:
        h.remove()

    caption_text = processor.decode(caption_token_ids, skip_special_tokens=True)

    # ── VLM splice metadata ────────────────────────────────────────────────
    # The C++ port needs the EXACT input_ids the processor emitted (with
    # image placeholder tokens already inserted by the chat template) so
    # that the device-side prefill runs on the same sequence the reference
    # ran on. Without this, the C++ side tokenizes the raw text prompt
    # only — 4 tokens with no image placeholders — and splice has nowhere
    # to inject the image features, producing on-the-wire divergence at
    # the very first embedding boundary.
    input_ids_full = inputs["input_ids"][0].tolist()
    # Resolve image_token_id from the processor / model config across families.
    image_token_id = None
    for path_chain in (
        ("image_token_id",),
        ("image_token", "id"),
        ("config", "image_token_id"),
    ):
        obj = processor
        for k in path_chain:
            obj = getattr(obj, k, None)
            if obj is None:
                break
        if isinstance(obj, int):
            image_token_id = obj
            break
    if image_token_id is None:
        # Fall back to the underlying model config (Idefics3/SmolVLM exposes it here).
        image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    image_placeholder_positions = []
    if isinstance(image_token_id, int):
        image_placeholder_positions = [i for i, t in enumerate(input_ids_full) if t == image_token_id]

    # Persist the spliced input ids as a deterministic binary that the C++
    # main.cpp can read via read_input_ids_bin() — same path it already uses
    # for text-only ports.
    import numpy as _np
    _np.asarray(input_ids_full, dtype=_np.int32).tofile(str(out_dir / "test_input_ids.bin"))

    # Canonical bundle (produced_by stamp is load-bearing - GenerateReference
    # validates it and refuses fabricated reference_tokens.json files).
    (out_dir / "reference_tokens.json").write_text(json.dumps({
        "produced_by": "_run_reference.py",
        "model_id": model_id,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "image_path": str(image_path.relative_to(project_dir)) if image_path.is_relative_to(project_dir) else str(image_path),
        "generated_ids": caption_token_ids,
        "reference_text": caption_text,
        # Splice integration metadata — consumed by the C++ port's
        # model_forward_graph to splice image_features_ into the LM input
        # sequence at the right positions. See vlm.md for the callsite.
        "input_ids": input_ids_full,
        "input_seq_len": len(input_ids_full),
        "image_token_id": image_token_id,
        "image_placeholder_positions": image_placeholder_positions,
        "num_image_tokens": len(image_placeholder_positions),
    }, indent=2))

    # Back-compat list consumed by src/modalities/vlm/eval.ts::evaluateCaptionTokenMatch.
    (out_dir / "reference_caption_tokens.json").write_text(json.dumps(caption_token_ids))
    (out_dir / "reference_caption_text.txt").write_text(caption_text)

    capture_meta = {}
    # Build the manifest in the same shape that reference-script.tmpl emits so
    # generateReferenceTs.ts::regenerateDumpSpecFromManifest can derive
    # .nnport/dump_spec.json from it. The TS regenerator walks every top-level
    # `<name>_output` key — without those, dump_spec.json comes out empty and
    # the runtime "loaded dump_spec.txt (0 entries)" path makes SxS comparisons
    # meaningless. (May 2026: SmolVLM port spent ~30 min debugging this.)
    manifest_entries = {}
    for dump_name, info in dumps.items():
        arr = info["data"]
        (layers_dir / f"{dump_name}_output.bin").write_bytes(arr.tobytes())
        flat = arr.reshape(-1).astype(np.float32, copy=False)
        def _safe_stat(v):
            fv = float(v)
            return fv if np.isfinite(fv) else None
        capture_meta[dump_name] = {
            "module_path": info["module_path"],
            "shape": info["shape"],
            "dtype": info["dtype"],
        }
        manifest_entries[f"{dump_name}_output"] = {
            "shape": info["shape"],
            "mean": _safe_stat(flat.mean()) if flat.size else None,
            "std":  _safe_stat(flat.std())  if flat.size else None,
            "min":  _safe_stat(flat.min())  if flat.size else None,
            "max":  _safe_stat(flat.max())  if flat.size else None,
            "numel": int(flat.size),
            "tensor_kind": "activation",
        }
    (layers_dir / "manifest.json").write_text(json.dumps({
        "produced_by": "_run_reference.py",
        **manifest_entries,
        "_capture_meta": capture_meta,
        "_captured_layers": list(capture_meta.keys()),
    }, indent=2))

    print(
        f"[reference] OK - {len(caption_token_ids)} caption tokens, "
        f"{len(capture_meta)} layer dumps",
        file=sys.stderr,
    )
    print(f"[reference] caption: {caption_text!r}", file=sys.stderr)

    # Result document for generateReferenceTs.ts::JSON.parse(stdout.trim()).
    # Mirrors the shape the text reference template emits — fields here gate
    # the success/partial/failure branches in the parser (see ~line 2314).
    result = {
        "success": True,
        "model_id": model_id,
        "input_text": prompt,
        "reference_text": caption_text,
        "generated_text": caption_text,
        "num_tokens_generated": len(caption_token_ids),
        "num_layers_captured": len(capture_meta),
        "num_intermediate_captured": 0,
        "captured_layers": list(capture_meta.keys()),
        "reference_dir": str(out_dir),
        "image_path": str(image_path),
    }
    sys.stdout.write(json.dumps(result))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
