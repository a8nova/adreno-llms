#!/usr/bin/env python3
"""TorchScript/.ptl end-to-end reference template — GENERIC across any
scripted-archive port (script version 2026-06-03-encdec-generic-forward-v4-waveform-capture).

The architecture of a TorchScript port is the scripted graph itself —
there is no transformers loader, no AutoConfig, no modeling_*.py. This
script loads the archive with torch.jit.load and captures the END-TO-END
forward as ground truth.

AGENT INSTRUCTIONS (read before editing):
  1. model_info/ptl_graph.json holds the module tree, every parameter and
     buffer shape, and the scripted forward() source for each submodule.
     READ IT before changing anything here — it is the reference code for
     this port.
  2. TorchScript modules do NOT support register_forward_hook. To capture
     intermediates, call submodules DIRECTLY (e.g.
     `m.model.encoder(feats, lens)`) and _write_bin each tensor, then add
     the name to MANIFEST_LAYERS. The forward() source in ptl_graph.json
     tells you each submodule's exact call signature.
  3. If the automatic input builder below cannot satisfy the forward
     schema, fill FORWARD_ARGS_OVERRIDE with the exact positional args.
     Your edits to this file are PRESERVED across GenerateReference calls.
  4. Autoregressive submodules (decoders, vocoders driven step-by-step)
     need one capture per generation step — capture the first step only
     unless Evaluate asks for more.
"""

import json
import os
import sys


def _progress(msg: str):
    sys.stderr.write(f"PROGRESS: {msg}\n")
    sys.stderr.flush()


def _write_bin(path: str, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.tofile(path)


# ── AGENT-EDIT SECTION ──────────────────────────────────────────────────
# Set to a tuple/list of the exact positional args for model(*args) when
# the schema-driven builder below cannot construct them. Example for a
# speech-translation wrapper taking (waveform, tgt_lang):
#   FORWARD_ARGS_OVERRIDE = lambda wav, fixture: (wav, fixture.get("tgt_lang", "eng"))
# (receives the loaded fixture tensor + the contract input_fixture dict).
FORWARD_ARGS_OVERRIDE = None
# Extra layer captures: dict of dump_name -> callable(model, forward_args,
# outputs) returning a torch.Tensor. Filled in by the agent as the port
# progresses (see AGENT INSTRUCTIONS above).
EXTRA_CAPTURES = {}
# ────────────────────────────────────────────────────────────────────────


def _find_archive(model_id: str, project_dir: str):
    """Locate the TorchScript archive: workspace first, then the nnopt
    model cache, then HuggingFace download. Returns a path or None."""
    candidates = []
    minfo = os.path.join(project_dir, "model_info")
    for root in (minfo, project_dir):
        if os.path.isdir(root):
            for f in sorted(os.listdir(root)):
                if f.endswith(".ptl") or f.endswith(".torchscript.pt"):
                    candidates.append(os.path.join(root, f))
    cache_root = os.path.expanduser("~/.nnopt/models")
    for slug in (model_id.replace("/", "--"), model_id.replace("/", "-")):
        d = os.path.join(cache_root, slug)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".ptl") or f.endswith(".torchscript.pt"):
                    candidates.append(os.path.join(d, f))
    if candidates:
        return candidates[0]
    # Last resort: list the HF repo and download any .ptl sibling.
    try:
        from huggingface_hub import HfApi, hf_hub_download

        files = HfApi().list_repo_files(model_id)
        ptls = [f for f in files if f.endswith(".ptl")]
        if ptls:
            _progress(f"Downloading {ptls[0]} from HF (not in any local cache)")
            return hf_hub_download(model_id, ptls[0])
    except Exception as e:
        _progress(f"HF archive lookup failed: {e}")
    return None


def _load_audio_fixture(path: str, target_sr: int):
    """Load a PCM WAV into a mono float32 torch tensor at target_sr."""
    import numpy as np
    import torch
    import wave as _wave

    with _wave.open(path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise RuntimeError(f"fixture {path}: only 16-bit PCM WAV supported (got sampwidth={sw})")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)
    wav = torch.from_numpy(pcm.copy())
    if sr != target_sr:
        try:
            import torchaudio

            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
            _progress(f"Resampled fixture {sr} Hz -> {target_sr} Hz via torchaudio")
        except Exception:
            # Linear-interp fallback — adequate for a reference fixture.
            import numpy as _np

            n_out = int(round(len(pcm) * target_sr / sr))
            x_old = _np.linspace(0.0, 1.0, num=len(pcm), endpoint=False)
            x_new = _np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            wav = torch.from_numpy(_np.interp(x_new, x_old, pcm).astype(_np.float32))
            _progress(f"Resampled fixture {sr} Hz -> {target_sr} Hz via linear interp (torchaudio unavailable)")
    return wav


def _build_forward_args(model, fixture_tensor, fixture_dict):
    """Schema-driven input construction. Introspects the scripted forward
    signature: Tensor args get the fixture tensor; str args are filled
    from the contract input_fixture by name (lang-ish names default to
    'eng'); args with defaults are left to their defaults."""
    try:
        schema = model.forward.schema
    except Exception:
        return (fixture_tensor,)
    args = []
    for a in schema.arguments[1:]:  # skip self
        t = str(a.type)
        has_default = a.default_value is not None
        if "Tensor" in t and not args:
            args.append(fixture_tensor)
        elif t == "str":
            v = fixture_dict.get(a.name)
            if v is None and "lang" in a.name.lower():
                v = fixture_dict.get("tgt_lang", "eng")
            if v is not None:
                args.append(v)
            elif has_default:
                break  # defaults from here on
            else:
                raise RuntimeError(
                    f"forward() arg '{a.name}: str' has no default and no value in "
                    f"contract input_fixture — set FORWARD_ARGS_OVERRIDE or add "
                    f"'{a.name}' to reference_protocol.input_fixture"
                )
        elif has_default:
            break
        else:
            raise RuntimeError(
                f"forward() arg '{a.name}: {t}' cannot be auto-filled — set FORWARD_ARGS_OVERRIDE "
                f"(see ptl_graph.json for the forward source)"
            )
    return tuple(args)


def _call_forward_with_shape_retries(model, fwd_args):
    """The forward schema names a Tensor arg but not its rank. Audio models
    variously take [T], [C,T] or [B,1,T] waveforms (e.g. kaldi fbank inside a
    scripted forward needs [C,T] and raises 'slice() cannot be applied to a
    0-dim tensor' on [T]). Retry mechanical rank variants on shape errors —
    generic across models, no shape vocabulary."""
    import torch

    variants = [tuple(fwd_args)]
    if fwd_args and isinstance(fwd_args[0], torch.Tensor):
        t = fwd_args[0]
        rest = tuple(fwd_args[1:])
        variants.append((t.unsqueeze(0),) + rest)
        if t.dim() == 1:
            variants.append((t.unsqueeze(0).unsqueeze(0),) + rest)
        if t.dim() >= 2:
            variants.append((t.squeeze(0),) + rest)
    last_err = None
    for i, va in enumerate(variants):
        try:
            with torch.no_grad():
                out = model(*va)
            if i > 0:
                _progress(f"forward() accepted input rank {va[0].dim()} (retry {i})")
            return out, va
        except RuntimeError as e:
            msg = str(e)
            if any(k in msg for k in ("0-dim", "dimension", "shape", "size", "slice", "expected", "rank")):
                last_err = e
                _progress(f"forward() shape retry {i + 1}/{len(variants)}: {msg.strip().splitlines()[-1][:160]}")
                continue
            raise
    raise last_err


def _flatten_outputs(out, prefix="e2e_out"):
    """Normalize a forward result (tensor | str | tuple | list | dict) to an
    ordered list of (name, value)."""
    import torch

    items = []
    if isinstance(out, (tuple, list)):
        for i, v in enumerate(out):
            items.extend(_flatten_outputs(v, f"{prefix}_{i}"))
    elif isinstance(out, dict):
        for k, v in out.items():
            items.extend(_flatten_outputs(v, f"{prefix}_{k}"))
    elif isinstance(out, torch.Tensor) or isinstance(out, str):
        items.append((prefix, out))
    elif out is not None:
        items.append((prefix, str(out)))
    return items


def main():
    model_id = sys.argv[1]
    project_dir = sys.argv[2]

    ref_dir = os.path.join(project_dir, "reference")
    layers_dir = os.path.join(ref_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)

    try:
        import numpy as np
        import torch
    except Exception as e:
        print(json.dumps({"success": False, "error": f"Import failed: {e}"}))
        return

    torch.manual_seed(42)
    np.random.seed(42)

    # ── Contract ─────────────────────────────────────────────────────────
    contract = {}
    try:
        cpath = os.path.join(project_dir, ".nnport", "contract.json")
        if os.path.exists(cpath):
            contract = json.load(open(cpath))
    except Exception:
        pass
    fixture_dict = contract.get("reference_protocol", {}).get("input_fixture", {})
    modality_in = contract.get("modality_in", []) or []
    modality_out = contract.get("modality_out", []) or []
    sr = int(
        contract.get("reference_protocol", {}).get("output_artifact", {}).get("sample_rate_hz")
        or 16000
    )
    pre = {p.get("name"): p.get("params", {}) for p in contract.get("preprocessors", []) or []}
    in_sr = int(pre.get("resample", {}).get("target_sr") or sr)

    # ── Locate + load the archive ────────────────────────────────────────
    archive = _find_archive(model_id, project_dir)
    if not archive:
        print(json.dumps({"success": False, "error": f"No TorchScript archive (.ptl) found for {model_id} in workspace, ~/.nnopt/models, or the HF repo"}))
        return
    _progress(f"Loading TorchScript archive: {archive}")
    try:
        model = torch.jit.load(archive, map_location="cpu")
    except Exception as e1:
        try:
            from torch.jit.mobile import _load_for_lite_interpreter

            model = _load_for_lite_interpreter(archive)
            _progress("Loaded via lite-interpreter fallback")
        except Exception as e2:
            print(json.dumps({"success": False, "error": f"torch.jit.load failed ({e1}); lite interpreter failed ({e2})"}))
            return
    try:
        model.eval()
    except Exception:
        pass

    # ── Build inputs ─────────────────────────────────────────────────────
    fixture_tensor = None
    if "audio" in modality_in:
        audio_rel = fixture_dict.get("audio", "model_info/test_audio.wav")
        audio_path = audio_rel if os.path.isabs(audio_rel) else os.path.join(project_dir, audio_rel)
        if not os.path.exists(audio_path):
            print(json.dumps({"success": False, "error": f"Audio fixture missing at {audio_path} — place a real spoken WAV there"}))
            return
        fixture_tensor = _load_audio_fixture(audio_path, in_sr)
        _progress(f"Audio fixture: {audio_path} -> {tuple(fixture_tensor.shape)} @ {in_sr} Hz")
    else:
        # Text-in TorchScript ports: tokenized prompt from PortTokenizer's
        # artifact if present; otherwise the agent supplies tensors via
        # FORWARD_ARGS_OVERRIDE.
        prompt = fixture_dict.get("prompt", "Hello, my name is")
        _progress(f"Text fixture prompt: {prompt!r} (tokenization is port-specific — use FORWARD_ARGS_OVERRIDE if needed)")

    # ── Input-feature fixture for the C++ runner ─────────────────────────
    # graph_mode_main.cpp loads assets/test_input_features.bin (float32 mel)
    # for audio encoder-decoders — without it the encoder runs on zeros and
    # the sampler collapses to id=0. Compute features with the CONTRACT's
    # log_mel params (kaldi-style fbank — what speech models' internal
    # frontends use), and keep the raw waveform alongside for ports whose
    # C++ computes mel itself.
    assets_dir = os.path.join(project_dir, "assets")
    feature_fixtures = []
    if fixture_tensor is not None:
        os.makedirs(assets_dir, exist_ok=True)
        raw_np = fixture_tensor.detach().cpu().numpy().astype("float32")
        _write_bin(os.path.join(assets_dir, "test_audio_raw.bin"), raw_np)
        feature_fixtures.append(
            {"name": "test_audio_raw", "shape": list(raw_np.shape), "dtype": "float32",
             "source": f"fixture WAV @ {in_sr} Hz", "deploy_path": "assets/test_audio_raw.bin"}
        )
        if "log_mel" in pre:
            mel_p = pre.get("log_mel", {})
            n_mels = int(mel_p.get("n_mels") or 80)
            hop = int(mel_p.get("hop") or 160)
            win = int(mel_p.get("win") or 400)
            try:
                import torchaudio

                feats = torchaudio.compliance.kaldi.fbank(
                    # kaldi expects [C, T] in int16 range — reshape is
                    # rank-agnostic (fixture may be [T] or [1, T] depending
                    # on what the forward-call retry ladder settled on).
                    fixture_tensor.reshape(1, -1) * 32768.0,
                    num_mel_bins=n_mels,
                    frame_shift=hop * 1000.0 / in_sr,
                    frame_length=win * 1000.0 / in_sr,
                    sample_frequency=float(in_sr),
                )  # [T_mel, n_mels]
                feats_np = feats.cpu().numpy().astype("float32")
                _write_bin(os.path.join(assets_dir, "test_input_features.bin"), feats_np)
                _write_bin(os.path.join(layers_dir, "input_features_output.bin"), feats_np)
                feature_fixtures.append(
                    {"name": "test_input_features", "shape": list(feats_np.shape), "dtype": "float32",
                     "source": f"torchaudio kaldi fbank (n_mels={n_mels}, hop={hop}, win={win})",
                     "deploy_path": "assets/test_input_features.bin"}
                )
                _progress(f"Mel fixture: test_input_features.bin {feats_np.shape}")
            except Exception as e:
                _progress(f"WARNING: mel feature fixture skipped (torchaudio: {e}) — C++ runner will need raw-audio frontend or agent-provided features")

    if FORWARD_ARGS_OVERRIDE is not None:
        fwd_args = FORWARD_ARGS_OVERRIDE(fixture_tensor, fixture_dict)
    else:
        fwd_args = _build_forward_args(model, fixture_tensor, fixture_dict)
    _progress(f"forward() args: {[type(a).__name__ for a in fwd_args]}")

    # ── End-to-end forward (with mechanical input-rank retries) ──────────
    out, fwd_args = _call_forward_with_shape_retries(model, fwd_args)

    items = _flatten_outputs(out)
    if not items:
        print(json.dumps({"success": False, "error": "forward() returned no capturable outputs"}))
        return

    # ── Persist outputs ──────────────────────────────────────────────────
    manifest_layers = {}
    text_out = None
    wav_out = None
    for name, val in items:
        if isinstance(val, str):
            if text_out is None:
                text_out = val
            continue
        arr = val.detach().cpu().contiguous().float().numpy().astype("float32")
        _write_bin(os.path.join(layers_dir, f"{name}_output.bin"), arr)
        manifest_layers[name] = {"output_shape": list(arr.shape), "output_dtype": "float32"}
        if wav_out is None and "audio" in modality_out and arr.ndim <= 2 and arr.size > sr // 4:
            wav_out = arr.reshape(-1)

    for name, fn in EXTRA_CAPTURES.items():
        try:
            t = fn(model, fwd_args, out)
            arr = t.detach().cpu().contiguous().float().numpy().astype("float32")
            _write_bin(os.path.join(layers_dir, f"{name}_output.bin"), arr)
            manifest_layers[name] = {"output_shape": list(arr.shape), "output_dtype": "float32"}
        except Exception as e:
            _progress(f"EXTRA_CAPTURES[{name}] failed: {e}")

    if text_out is not None:
        with open(os.path.join(ref_dir, "reference_text.txt"), "w") as f:
            f.write(text_out + "\n")
        _progress(f"Reference text: {text_out!r}")
    if wav_out is not None:
        import wave as _wave

        pcm16 = (np.clip(wav_out, -1.0, 1.0) * 32767.0).astype("<i2")
        with _wave.open(os.path.join(ref_dir, "output.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm16.tobytes())
        _progress(f"Reference WAV: {len(wav_out)} samples @ {sr} Hz")

    manifest = {
        "_nnport_capture_version": "torchscript_e2e_v1",
        "_captured_layers": list(manifest_layers.keys()),
        "layers": manifest_layers,
    }
    with open(os.path.join(layers_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Top-level stage graph (P6) ───────────────────────────────────────
    # TorchScript can't be hooked for intermediates, but its top-level children
    # ARE the pipeline stages and are enumerable. Emit them as forward_graph.json
    # nodes so the stage-plan derivation gives this .ptl its REAL multi-stage
    # decomposition instead of one box. Control-flow is detected generically: a
    # child exposing generate() is an autoregressive search; the rest are
    # feed-forward. No model names — pure structural inspection.
    try:
        stage_nodes = []
        for _order, (_cname, _child) in enumerate(model.named_children()):
            _is_ar = callable(getattr(_child, "generate", None))
            stage_nodes.append({
                "module_path": _cname,
                "dump_name": _cname,
                "op": type(_child).__name__,
                "order": _order,
                "control_flow": "autoregressive" if _is_ar else "feed_forward",
                "param_tensors": sum(1 for _ in _child.parameters()),
            })
        if len(stage_nodes) >= 2:
            with open(os.path.join(ref_dir, "forward_graph.json"), "w") as f:
                json.dump({"_nnport_capture_version": "torchscript_e2e_v1", "nodes": stage_nodes}, f, indent=2)
            _progress(
                "Emitted forward_graph.json with %d top-level stages: %s"
                % (len(stage_nodes), ", ".join("%s[%s]" % (n["module_path"], n["control_flow"]) for n in stage_nodes))
            )
    except Exception as e:
        _progress(f"stage-graph emission skipped: {e}")

    with open(os.path.join(ref_dir, "io_contract.json"), "w") as f:
        json.dump(
            {
                "version": 1,
                "model_class": type(model).__name__,
                "archive": os.path.basename(archive),
                "sample_rate_hz": sr,
                "input_sample_rate_hz": in_sr,
                "forward_arg_types": [type(a).__name__ for a in fwd_args],
                "output_artifact": {
                    "kind": "wav" if wav_out is not None else "text",
                    "sample_rate_hz": sr,
                },
                "_nnport_capture_version": "torchscript_e2e_v1",
            },
            f,
            indent=2,
        )

    # PRESERVE synthesized pipeline-stage nodes: GenerateScaffold derives
    # per-stage nodes from ptl_graph.json (module_path != None) and merges
    # them into forward_graph.json — re-running this script must not wipe
    # them or the per-stage op scaffolds lose their graph entries.
    fg_path = os.path.join(ref_dir, "forward_graph.json")
    preserved_nodes = []
    try:
        if os.path.exists(fg_path):
            old = json.load(open(fg_path))
            preserved_nodes = [
                n for n in old.get("nodes", [])
                if n.get("op") != "TorchScriptE2E" and n.get("module_path")
            ]
    except Exception:
        preserved_nodes = []
    e2e_nodes = [
        {
            "op": "TorchScriptE2E",
            "dump_name": name,
            "module_path": None,
            "output_shape": meta["output_shape"],
            "order": len(preserved_nodes) + i,
            "weight_prefix": None,
            "is_pre_hook": False,
        }
        for i, (name, meta) in enumerate(manifest_layers.items())
    ]
    with open(fg_path, "w") as f:
        json.dump(
            {
                "version": 1,
                "model_class": type(model).__name__,
                "nodes": preserved_nodes + e2e_nodes,
                "_note": "E2E captures + scaffold-synthesized pipeline stages. Extend reference coverage via EXTRA_CAPTURES in _run_reference.py using submodule calls from model_info/ptl_graph.json.",
                "_nnport_capture_version": "torchscript_e2e_v1",
            },
            f,
            indent=2,
        )

    with open(os.path.join(ref_dir, "reference_tokens.json"), "w") as f:
        json.dump(
            {
                "produced_by": "_run_reference.py (torchscript_e2e)",
                "text": text_out,
                "generated_ids": [],
            },
            f,
        )

    print(
        json.dumps(
            {
                "success": True,
                "archive": os.path.basename(archive),
                "captured": list(manifest_layers.keys()),
                "text": text_out,
                "wav_samples": int(len(wav_out)) if wav_out is not None else 0,
            }
        )
    )


if __name__ == "__main__":
    main()
