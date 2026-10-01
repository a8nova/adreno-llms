#!/usr/bin/env python3
# NNOPT MLX reference runner.
#
# Loads an MLX (Apple array framework) model per the discovered loader recipe
# (_llm_loader.json), runs it deterministically, and dumps golden tensors in
# nnopt's contract:
#   reference/reference_tokens.json   (produced_by stamp gates the TS cache)
#   reference/layers/<module_path>_{input,output}.bin   (fp32, C-order)
#   reference/layers/manifest.json    (_captured_layers, _capture_meta.module_path, ...)
#
# Per-layer capture mechanism (proven on real mlx.nn): monkeypatch __call__ on
# every DISTINCT class in the module tree, keyed by id(module) -> the dotted
# module_path that named_modules() returns (the same namespace the runtime
# localizer pairs on). If the model is an exported .mlxfn graph with no
# nn.Module tree, no captures appear and we fall back to an end-to-end-only
# reference (logged, never silent).
#
# Spawn contract (set by generateReferenceTs):
#   argv = [script, model_id, output_dir, prompt, max_new, source_path, tokenizer_repo]
import sys, os, json, traceback

REFERENCE_SCRIPT_VERSION = "2026-06-16-auto-dep-install-retry-v6"


def progress(msg):
    print(msg, flush=True)


def fail(failure_class, error, tb=None):
    print(json.dumps({
        "success": False,
        "failure_class": failure_class,
        "error": str(error),
        "stderr_tail": (tb or traceback.format_exc())[-2000:],
    }))


def main():
    model_id    = sys.argv[1] if len(sys.argv) > 1 else ""
    output_dir  = sys.argv[2] if len(sys.argv) > 2 else "."
    prompt      = sys.argv[3] if len(sys.argv) > 3 else ""
    max_new     = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 64
    source_path = sys.argv[5] if len(sys.argv) > 5 else ""

    ref_dir = os.path.join(output_dir, "reference")
    layers_dir = os.path.join(ref_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)

    try:
        import mlx.core as mx
    except Exception as e:
        return fail("mlx_not_installed", f"import mlx.core failed: {e}")
    try:
        import mlx.nn as nn
    except Exception:
        nn = None
    import numpy as np

    # ── Loader recipe (from loader-discovery) ──
    loader_path = os.path.join(output_dir, "_llm_loader.json")
    if not os.path.exists(loader_path):
        return fail("mlx_no_loader",
                    "MLX runner requires _llm_loader.json (loader-discovery produced no recipe).")
    with open(loader_path) as f:
        loader = json.load(f)

    # Inject the MLX-only download filter (written by nnopt at port-init) so any
    # snapshot_download(model_id) in the loader recipe pulls just the chosen
    # variant's weights + resources (the variant's flat checkpoint + .mlxfn +
    # codec/style resources), never the OTHER variant's giant checkpoint. Patch
    # BEFORE exec'ing the recipe's imports/load.
    #
    # A model's weights+resources can total several GB and the plain-HTTPS path
    # (Xet disabled by nnopt for stall-safety) is slow. huggingface_hub's outer
    # "Fetching N files" tqdm bar only repaints when a whole file finishes, so a
    # single multi-hundred-MB file can be silent for minutes and trip nnopt's
    # 7-min inactivity watchdog mid-download. Emit a heartbeat every 60s during
    # the download so a slow-but-progressing fetch is never mistaken for a hang.
    try:
        ap_path = os.path.join(output_dir, ".nnport", "mlx_allow_patterns.json")
        if os.path.exists(ap_path):
            with open(ap_path) as _apf:
                allow = (json.load(_apf) or {}).get("allow_patterns") or []
            if allow:
                import threading
                import huggingface_hub as _hf
                _orig_snap = _hf.snapshot_download
                def _filtered_snap(*a, **k):
                    k.setdefault("allow_patterns", allow)
                    _stop = threading.Event()
                    def _hb():
                        _n = 0
                        while not _stop.wait(60):
                            _n += 60
                            progress(f"[mlx] still downloading model files... ({_n}s elapsed)")
                    _t = threading.Thread(target=_hb, daemon=True)
                    _t.start()
                    try:
                        return _orig_snap(*a, **k)
                    finally:
                        _stop.set()
                _hf.snapshot_download = _filtered_snap
                progress(f"[mlx] snapshot_download filtered to: {allow}")
    except Exception as _e:
        progress(f"[mlx] could not install download filter (continuing): {_e}")

    # Make the cloned source repo importable.
    if source_path and os.path.isdir(source_path):
        sys.path.insert(0, source_path)
        parent = os.path.dirname(source_path.rstrip("/"))
        if parent:
            sys.path.insert(0, parent)

    g = {"mx": mx, "nn": nn, "np": np, "os": os, "sys": sys,
         "prompt": prompt, "max_new": max_new, "model_id": model_id,
         "source_path": source_path, "output_dir": output_dir}

    # extra_setup + imports
    for key in ("extra_setup", "import_statement", "tokenizer_import"):
        stmt = (loader.get(key) or "").strip()
        if stmt:
            try:
                exec(stmt, g)
            except Exception as e:
                return fail("mlx_loader_import_error", f"{key} failed: {e}")

    # Load model
    try:
        g["model"] = eval(loader["model_load"], g)
    except Exception as e:
        return fail("mlx_model_load_error", f"model_load failed: {e}")
    model = g["model"]

    # Tokenizer (optional — many MLX audio/codec models tokenize internally)
    tload = (loader.get("tokenizer_load") or "").strip()
    if tload:
        try:
            g["tokenizer"] = eval(tload, g)
        except Exception as e:
            progress(f"tokenizer_load failed (continuing; model may tokenize internally): {e}")
    g.setdefault("tokenizer", None)

    # ── Per-class __call__ capture keyed by named_modules() ──
    captures, order, capture_meta = {}, [], {}
    instrumentation_kind = "e2e-only"
    patched_originals = {}
    # Collect nn.Module ROOTS to instrument. CRITICAL: many model wrappers are NOT
    # themselves an nn.Module — the real MLX module tree lives in a sub-attribute
    # (e.g. magenta_rt's MagentaRT2Mlx is a plain object whose 3274-module
    # depthformer lives under `._sampler`). Checking only the top-level
    # `model.named_modules()` then returns 0 modules → e2e-only → the port has no
    # per-layer reference to implement against. So when the top-level isn't an
    # nn.Module, scan its attributes ONE level for nn.Module roots and instrument
    # those, prefixing paths with the attribute name so dumps stay uniquely keyed.
    _roots = []
    if nn is not None:
        if isinstance(model, nn.Module):
            _roots.append(("", model))
        else:
            for _name in dir(model):
                if _name.startswith("__"):
                    continue
                try:
                    _v = getattr(model, _name)
                except Exception:
                    continue
                if isinstance(_v, nn.Module):
                    _roots.append((_name.lstrip("_"), _v))
            if _roots:
                progress(f"[mlx-capture] top-level {type(model).__name__} is not an nn.Module; "
                         f"instrumenting nn.Module sub-roots: {[r[0] for r in _roots]}")
    if _roots:
        try:
            path_index, classes = {}, set()
            for _prefix, _root in _roots:
                for mp, mod in _root.named_modules():
                    _full = (_prefix + "." + mp) if (_prefix and mp) else (_prefix or mp or "<root>")
                    path_index[id(mod)] = _full
                    classes.add(type(mod))

            def _first_array(x, _depth=0):
                # Many modules return/accept NOT a bare mx.array but a tuple/list/
                # dict (e.g. attention → (out, kv_cache); a block → {"x": ...}).
                # Recursively pull the first mx.array so per-layer dumps are DENSE
                # (without this, ~108/115 nodes had None I/O → no .bin to validate).
                if isinstance(x, mx.array):
                    return x
                if _depth < 4:
                    if isinstance(x, (list, tuple)):
                        for _e in x:
                            _r = _first_array(_e, _depth + 1)
                            if _r is not None:
                                return _r
                    elif isinstance(x, dict):
                        for _e in x.values():
                            _r = _first_array(_e, _depth + 1)
                            if _r is not None:
                                return _r
                return None

            def make_traced(orig):
                def call(self, *args, **kw):
                    out = orig(self, *args, **kw)
                    mp = path_index.get(id(self))
                    if mp and mp != "<root>" and mp not in captures:
                        inp = _first_array(args[0]) if args else None
                        outp = _first_array(out)
                        if outp is not None:
                            try:
                                mx.eval(outp)         # materialize at THIS boundary (MLX is lazy)
                            except Exception:
                                pass
                        captures[mp] = {"input": inp, "output": outp}
                        order.append(mp)
                        capture_meta[mp] = {
                            "module_path": mp,
                            "module_class": type(self).__name__,
                            "qualname_in_parent": mp.rsplit(".", 1)[-1],
                            "is_pre_hook": False,
                        }
                    return out
                return call

            for cls in classes:
                patched_originals[cls] = cls.__call__
                cls.__call__ = make_traced(cls.__call__)
            instrumentation_kind = "class-call-wrap"
            progress(f"[mlx-capture] patched {len(classes)} classes across {len(path_index)} modules")
        except Exception as e:
            progress(f"[mlx-capture] per-class setup failed ({e}); running e2e-only")
            patched_originals = {}
    else:
        progress("[mlx-capture] no nn.Module tree (likely an exported .mlxfn graph) — e2e-only")

    # ── Run inference deterministically ──
    inf = (loader.get("inference_call") or "").strip()
    if not inf:
        return fail("mlx_no_inference_call", "loader recipe missing inference_call")
    out_obj = None
    try:
        try:
            out_obj = eval(inf, g)               # single expression
        except SyntaxError:
            exec(inf, g)                         # multi-statement routine assigning `result`
            out_obj = g.get("result")
    except Exception as e:
        return fail("mlx_inference_error", f"inference_call failed: {e}")
    finally:
        for cls, orig in patched_originals.items():
            try:
                cls.__call__ = orig
            except Exception:
                pass

    # ── Helpers ──
    def to_np(x):
        try:
            if isinstance(x, mx.array):
                # Cast to float32 IN MLX first: numpy cannot represent bfloat16
                # (the depthformer runs in bf16), so `np.array(bf16_array)` raises
                # and the tensor is silently dropped — only fp32 layers (~7/115)
                # survived. astype(float32) in MLX makes every captured boundary
                # dumpable.
                x = x.astype(mx.float32)
                mx.eval(x)
                return np.array(x, dtype=np.float32)
        except Exception:
            pass
        # Audio/codec models often wrap their output in a container object
        # (e.g. magenta_rt's Waveform with a `.samples` ndarray) rather than
        # returning a bare array. Unwrap the common audio sample attributes so
        # the end-to-end waveform reference is actually written instead of None.
        for attr in ("samples", "audio", "waveform", "wav", "data", "array"):
            try:
                v = getattr(x, attr, None)
                if v is not None and not callable(v):
                    arr = np.asarray(v, dtype=np.float32)
                    if arr.size > 0:
                        return arr
            except Exception:
                pass
        try:
            return np.asarray(x, dtype=np.float32)
        except Exception:
            return None

    # ── End-to-end output (waveform / logits / tokens) ──
    out_arr = to_np(out_obj)
    if out_arr is None and isinstance(out_obj, (list, tuple)):
        for item in out_obj:                     # TTS-style list of tuples/segments
            cand = to_np(item)
            if cand is not None:
                out_arr = cand
                break
    e2e_written = False
    if out_arr is not None:
        out_arr.reshape(-1).astype(np.float32).tofile(os.path.join(layers_dir, "output_output.bin"))
        e2e_written = True

    # ── Per-layer dumps + manifest ──
    # `manifest_layers` is the nested form; `flat_entries` is the FLAT form the
    # TS dump_spec regenerator (regenerateDumpSpecFromManifest) consumes — it
    # iterates top-level keys ending in `_output`/`_input` with numel+shape and
    # tags each comparison=primary (SxS gates per-layer cosines here). Without
    # these flat keys dump_spec stays empty → 0 primary entries → "SxS runs
    # blind" even though the .bin dumps exist.
    manifest_layers = {}
    flat_entries = {}
    for mp in order:
        cap = captures.get(mp, {})
        ia, oa = to_np(cap.get("input")), to_np(cap.get("output"))
        if ia is not None:
            ia.reshape(-1).astype(np.float32).tofile(os.path.join(layers_dir, f"{mp}_input.bin"))
        if oa is not None:
            oa.reshape(-1).astype(np.float32).tofile(os.path.join(layers_dir, f"{mp}_output.bin"))
        manifest_layers[mp] = {
            "input_shape": list(ia.shape) if ia is not None else [],
            "output_shape": list(oa.shape) if oa is not None else [],
            "input_dtype": "float32",
            "output_dtype": "float32",
            "_capture_meta": capture_meta.get(mp, {"module_path": mp}),
        }
        if oa is not None:
            flat_entries[f"{mp}_output"] = {
                "numel": int(oa.size), "shape": list(oa.shape), "tensor_kind": "activation",
            }
        if ia is not None:
            flat_entries[f"{mp}_input"] = {
                "numel": int(ia.size), "shape": list(ia.shape), "tensor_kind": "activation",
            }

    captured_layers = list(order)
    capture_mode = "per_layer"
    if not captured_layers:
        # Honest sentinel so the TS cache key reflects degraded (e2e-only)
        # capture and never masquerades as a full per-layer run.
        instrumentation_kind = "e2e-only"
        capture_mode = "e2e_only"
        captured_layers = ["output"]
        progress("[mlx-capture] WARNING: 0 per-layer captures — emitting END-TO-END reference only; "
                 "per-layer cosine localization is DISABLED for this model.")

    manifest = {
        "_captured_layers": captured_layers,
        "_capture_order": order,
        "_capture_meta": capture_meta,
        "_nnport_capture_version": REFERENCE_SCRIPT_VERSION,
        "_framework": "mlx",
        "_instrumentation_kind": instrumentation_kind,
        "_capture_mode": capture_mode,
        "layers": manifest_layers,
    }
    # Merge the FLAT `<mp>_output`/`<mp>_input` entries at top level so the TS
    # dump_spec regenerator finds them (it scans top-level keys, not `layers`).
    manifest.update(flat_entries)
    with open(os.path.join(layers_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ── forward_graph.json (Phase 3A artifact the scaffold + Build consume) ──
    # The scaffold wires src/ops/backbone.cpp::model_forward_graph from these
    # nodes, and Build/Evaluate pair runtime↔reference dumps by module_path.
    # Without it the agent has "no layer graph to implement from" even when the
    # per-layer .bin dumps exist. Built from the SAME captured order/meta/shapes.
    if capture_mode == "per_layer":
        fg_nodes = []
        for i, mp in enumerate(order):
            meta = capture_meta.get(mp, {})
            ml = manifest_layers.get(mp, {})
            fg_nodes.append({
                "op": meta.get("module_class", "Module"),
                "dump_name": mp,
                "module_path": mp,
                "input_shape": ml.get("input_shape", []),
                "output_shape": ml.get("output_shape", []),
                "order": i,
                "weight_prefix": mp,
                "is_pre_hook": False,
                "reference_only": True,
            })
        forward_graph = {
            "version": "1.0",
            "model_class": type(model).__name__,
            "nodes": fg_nodes,
            "_nnport_capture_version": REFERENCE_SCRIPT_VERSION,
            "_raw_node_count_before_dedup": len(order),
            "_framework": "mlx",
        }
        with open(os.path.join(ref_dir, "forward_graph.json"), "w") as f:
            json.dump(forward_graph, f, indent=2)
        progress(f"[mlx-capture] wrote forward_graph.json ({len(fg_nodes)} nodes)")

    # ── reference_tokens.json (produced_by stamp gates the TS cache) ──
    try:
        import mlx as _mlx_mod
        _mlx_ver = getattr(_mlx_mod, "__version__", "unknown")
    except Exception:
        _mlx_ver = "unknown"
    ref_tokens = {
        "produced_by": "_run_reference.py",
        "script_version": REFERENCE_SCRIPT_VERSION,
        "framework": "mlx",
        "mlx_version": _mlx_ver,
        "python_version": sys.version.split()[0],
        "model_id": model_id,
        "prompt": prompt,
        "max_new_tokens": max_new,
        "instrumentation_kind": instrumentation_kind,
        "reference_text": "",
    }
    with open(os.path.join(ref_dir, "reference_tokens.json"), "w") as f:
        json.dump(ref_tokens, f, indent=2)

    progress(f"[mlx] reference complete: {len(order)} layers captured "
             f"({instrumentation_kind}); e2e_output={'yes' if e2e_written else 'no'}")
    print(json.dumps({
        "success": True,
        "framework": "mlx",
        "instrumentation_kind": instrumentation_kind,
        "num_layers_captured": len(order),
        "captured_layers": order[:50],
        "e2e_output_written": e2e_written,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        fail("mlx_runner_crash", e)
