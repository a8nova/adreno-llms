#!/usr/bin/env python3
"""Generate reference output from a PyTorch/HuggingFace model."""
import sys, os, json, struct, traceback, math

def main():
    model_id   = sys.argv[1]
    output_dir = sys.argv[2]
    prompt     = sys.argv[3] if len(sys.argv) > 3 else "The teacher worked at the "
    max_new    = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    source_path = sys.argv[5] if len(sys.argv) > 5 else ""
    tokenizer_repo = sys.argv[6] if len(sys.argv) > 6 else ""

    # Normalize model_id: strip HuggingFace URLs to bare repo ID
    for prefix in ["https://huggingface.co/", "http://huggingface.co/",
                    "https://www.huggingface.co/", "http://www.huggingface.co/"]:
        if model_id.startswith(prefix):
            model_id = model_id[len(prefix):]
            break
    # Strip URL path suffixes and trailing slashes
    for suffix in ["/resolve/main", "/tree/main", "/blob/main"]:
        if model_id.endswith(suffix):
            model_id = model_id[:-len(suffix)]
    model_id = model_id.strip("/")

    ref_dir = os.path.join(output_dir, "reference")
    layers_dir = os.path.join(ref_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)

    def progress(msg):
        sys.stderr.write(f"PROGRESS: {msg}\n")
        sys.stderr.flush()

    # ─── Reference adaptation patches (injected by SxS via the ledger) ───────
    # _LAYOUT_OVERRIDES: dump_name → permutation. When a dump is captured we
    # apply tensor.permute(*perm).contiguous() before reshape so the bytes
    # come out in C++ row-major order.
    # _FORM_TRANSFORMS: list of named monkey-patches applied BEFORE the model
    # is loaded. Each entry maps to a small Python shim below.
    _LAYOUT_OVERRIDES = {}
    _FORM_TRANSFORMS = [{"transform":"disable_attr","params":{"module":"transformers.models.lfm2.modeling_lfm2","attr":"is_causal_conv1d_available","value":False,"value_kind":"callable_returning"}},{"transform":"disable_attr","params":{"module":"modular_lfm2","attr":"is_causal_conv1d_available","value":False,"value_kind":"callable_returning"}}]

    def _apply_layout_override(_t, _name):
        """If a permutation is registered for this dump name, materialize the
        permuted-and-contiguous tensor. Otherwise force a contiguous copy so
        the dump bytes always reflect logical row-major over the tensor's
        actual shape (was a latent bug on hook outputs that returned views)."""
        try:
            import torch as _torch
        except Exception:
            return _t
        _perm = _LAYOUT_OVERRIDES.get(_name)
        if _perm is not None and isinstance(_t, _torch.Tensor) and _t.dim() == len(_perm):
            return _t.permute(*_perm).contiguous()
        if isinstance(_t, _torch.Tensor):
            return _t.contiguous()
        return _t

    def _apply_form_transforms(_specs):
        """Patch reference internals based on requested form transforms.
        Runs BEFORE the model is loaded. The interpreter is architecture-
        blind: every transform is a generic operation parameterized by
        `params`. SxS discovers what to patch by reading the model's
        mirrored source; this code only knows how to APPLY the discovery.

        Currently supported transforms:
          disable_attr — set <params.module>.<params.attr> = <params.value>.
                         Used to flip module-level fast-path flags off so
                         reference takes the slow/decomposed kernel branch.
        """
        if not _specs:
            return
        import importlib as _importlib
        for _spec in _specs:
            _name = _spec.get("transform") if isinstance(_spec, dict) else None
            _params = _spec.get("params", {}) if isinstance(_spec, dict) else {}
            try:
                if _name == "disable_attr":
                    _module = _params.get("module")
                    _attr = _params.get("attr")
                    _value = _params.get("value", False)
                    _value_kind = _params.get("value_kind", "literal")
                    if not _module or not _attr:
                        progress(f"FORM_PATCH: disable_attr missing module/attr — skipped")
                        continue
                    try:
                        _mod = _importlib.import_module(_module)
                    except Exception as _ie:
                        progress(f"FORM_PATCH: disable_attr import {_module} failed: {_ie}")
                        continue
                    # callable_returning: replace with a function that always
                    # returns the literal — needed for HF availability checks
                    # like is_mamba_ssm_available which are functions, not
                    # bools. Setting them to a constant breaks any caller that
                    # actually invokes them.
                    if _value_kind == "callable_returning":
                        _const = _value
                        _new = (lambda _c: (lambda *_a, **_kw: _c))(_const)
                    else:
                        _new = _value
                    try:
                        _prev = getattr(_mod, _attr, "<unset>")
                        setattr(_mod, _attr, _new)
                        progress(f"FORM_PATCH: disable_attr {_module}.{_attr} ({_value_kind}): {_prev!r} -> {_value!r}")
                    except Exception as _se:
                        progress(f"FORM_PATCH: disable_attr setattr {_module}.{_attr} failed: {_se}")
                else:
                    progress(f"FORM_PATCH: unknown transform {_name!r} — skipped")
            except Exception as _e:
                progress(f"FORM_PATCH error: {_e}")

    _apply_form_transforms(_FORM_TRANSFORMS)
    # ─────────────────────────────────────────────────────────────────────────

    # The TypeScript preflight is the sole gate for installing torch/transformers.
    # It installs only into the isolated reference_venv, never the user's active
    # env. If imports fail here, something is broken upstream — hard-error out
    # rather than silently pip-installing into whatever interpreter we ended up
    # with.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        result = {
            "success": False,
            "error": (
                f"Import failed inside reference venv: {e}. "
                f"The TypeScript preflight should have installed torch + transformers. "
                f"Python: {sys.executable}, version: {sys.version}."
            ),
        }
        print(json.dumps(result))
        return

    # Check CUDA availability upfront — if the model needs GPU-only deps, report early
    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        progress("WARNING: CUDA not available. Models requiring mamba_ssm/flash_attn/triton will fail.")
        progress("If this model needs GPU-only deps, run GenerateReference on a CUDA machine instead.")

    progress(f"Loading model: {model_id}")

    # If source_path is provided, install the source package first so custom
    # model classes are available to from_pretrained / AutoModel.
    # CRITICAL: if the install fails, we ITERATE — diagnose missing deps,
    # install them, and retry. The whole point of having local source is to
    # make it work, not bail on the first error.
    if source_path and os.path.exists(source_path):
        import subprocess, re as _re
        progress(f"Installing source package from {source_path}")
        setup_py = os.path.join(source_path, "setup.py")
        pyproject = os.path.join(source_path, "pyproject.toml")
        has_setup = os.path.exists(setup_py) or os.path.exists(pyproject)

        if has_setup:
            MAX_INSTALL_RETRIES = 3
            for attempt in range(1, MAX_INSTALL_RETRIES + 1):
                try:
                    out = subprocess.check_output(
                        [sys.executable, "-m", "pip", "install", "-e", source_path, "-q"],
                        stderr=subprocess.STDOUT,
                        timeout=120,
                    ).decode()
                    progress("Source package installed")
                    break
                except subprocess.CalledProcessError as e:
                    pip_output = (e.output or b"").decode()
                    progress(f"Install attempt {attempt}/{MAX_INSTALL_RETRIES} failed")

                    # Parse missing dependencies from pip error output
                    missing = set()
                    # "No matching distribution found for <pkg>"
                    for m in _re.finditer(r"No matching distribution found for ([\w_-]+)", pip_output):
                        missing.add(m.group(1).split("[")[0].split(">")[0].split("<")[0].split("=")[0].split("!")[0])
                    # "Could not find a version that satisfies the requirement <pkg>"
                    for m in _re.finditer(r"Could not find a version.*requirement ([\w_-]+)", pip_output):
                        missing.add(m.group(1))
                    # "ModuleNotFoundError: No module named '<pkg>'"
                    for m in _re.finditer(r"No module named ['\"](\w+)", pip_output):
                        missing.add(m.group(1))

                    if missing and attempt < MAX_INSTALL_RETRIES:
                        progress(f"Missing deps detected: {', '.join(missing)} — installing...")
                        for dep in missing:
                            try:
                                subprocess.check_call(
                                    [sys.executable, "-m", "pip", "install", dep, "-q"],
                                    stderr=subprocess.STDOUT,
                                    timeout=60,
                                )
                                progress(f"  Installed {dep}")
                            except Exception:
                                progress(f"  Could not install {dep}")
                    elif attempt < MAX_INSTALL_RETRIES:
                        # No specific dep found — try --no-deps as last resort
                        progress("No specific missing dep found — trying --no-deps install")
                        try:
                            subprocess.check_call(
                                [sys.executable, "-m", "pip", "install", "-e", source_path, "--no-deps", "-q"],
                                stderr=subprocess.STDOUT,
                                timeout=120,
                            )
                            progress("Source package installed (--no-deps)")
                            break
                        except Exception:
                            progress("--no-deps install also failed")
                    else:
                        progress(f"Source install failed after {MAX_INSTALL_RETRIES} attempts, falling back to sys.path")
                        sys.path.insert(0, source_path)
                except subprocess.TimeoutExpired:
                    progress(f"Install attempt {attempt} timed out")
                    if attempt >= MAX_INSTALL_RETRIES:
                        progress("Source install timed out, falling back to sys.path")
                        sys.path.insert(0, source_path)
                except Exception as e:
                    progress(f"Install attempt {attempt} error: {e}")
                    if attempt >= MAX_INSTALL_RETRIES:
                        sys.path.insert(0, source_path)
        else:
            # No setup.py/pyproject.toml — just add to sys.path
            sys.path.insert(0, source_path)
            progress(f"Added {source_path} to sys.path (no setup.py/pyproject.toml)")

    # ══════════════════════════════════════════════════════════════════
    # ── SOURCE PATH: pip install already ran in TS preflight.
    # Find the installed package, import it, call from_pretrained(model_id).
    # No regex. No file parsing. Just: find package → import → load.
    # ══════════════════════════════════════════════════════════════════
    if source_path:
        try:
            import numpy as np

            src_model = None
            tokenizer_info = None

            # ══════════════════════════════════════════════════════════════
            # PRIORITY: Check for LLM-analyzed loader instructions
            # Written by the Colab setup cell from analyzeRepoWithLLM()
            # ══════════════════════════════════════════════════════════════
            llm_loader_path = os.path.join(output_dir, "_llm_loader.json")
            if os.path.exists(llm_loader_path):
                progress("Found LLM-analyzed model loading instructions")
                with open(llm_loader_path) as f:
                    llm_info = json.load(f)
                try:
                    # Execute the LLM-provided import and load
                    progress(f"Executing: {llm_info['import_statement']}")
                    exec(llm_info["import_statement"], globals())
                    progress(f"Executing: {llm_info['model_load']}")
                    import torch as _torch
                    src_model = eval(llm_info["model_load"])
                    progress(f"Model loaded via LLM instructions: {type(src_model).__name__}")

                    # Load tokenizer from LLM instructions
                    if llm_info.get("tokenizer_import"):
                        exec(llm_info["tokenizer_import"], globals())
                    if llm_info.get("tokenizer_load"):
                        tokenizer_info = {"llm": True}
                        from transformers import AutoTokenizer
                        # Parse the tokenizer load to extract repo ID
                        import re as _tok_re
                        tok_match = _tok_re.search(r"from_pretrained\(['\"](.*?)['\"", llm_info["tokenizer_load"])
                        if tok_match:
                            tok_repo = tok_match.group(1)
                            progress(f"Loading tokenizer: {tok_repo}")

                    # Apply extra setup
                    if llm_info.get("extra_setup"):
                        exec(llm_info["extra_setup"])

                    if llm_info.get("device") == "cuda":
                        src_model = src_model.cuda()
                    src_model.eval()
                except Exception as llm_err:
                    progress(f"LLM loader failed: {llm_err}")
                    # Try to fix common errors and retry
                    err_str = str(llm_err)
                    if "cannot import name" in err_str and "transformers" in err_str:
                        # Transformers version mismatch — downgrade
                        progress("Fixing: transformers version mismatch — installing compatible version...")
                        import subprocess as _fix_sub
                        _fix_sub.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers==4.44.0"], timeout=120)
                        # Clear transformers modules and retry
                        for mod_name in list(sys.modules.keys()):
                            if "transformers" in mod_name:
                                del sys.modules[mod_name]
                        try:
                            progress(f"Retrying: {llm_info['import_statement']}")
                            exec(llm_info["import_statement"], globals())
                            progress(f"Retrying: {llm_info['model_load']}")
                            import torch as _torch
                            src_model = eval(llm_info["model_load"])
                            progress(f"SUCCESS on retry: {type(src_model).__name__}")
                            if llm_info.get("device") == "cuda":
                                src_model = src_model.cuda()
                            src_model.eval()
                        except Exception as retry_err:
                            progress(f"Retry also failed: {retry_err}")
                            src_model = None
                    else:
                        progress(f"Error not auto-fixable: {err_str}")
                        src_model = None

            # ══════════════════════════════════════════════════════════════
            # FALLBACK: Read the repo's OWN scripts to find how to load the
            # model (only if LLM loader didn't work above).
            # ══════════════════════════════════════════════════════════════
            import re as _re, importlib as _importlib

            if source_path not in sys.path:
                sys.path.insert(0, source_path)

            # Scan repo scripts for from_pretrained patterns
            search_dirs = ["evals", "examples", "scripts", "tests", "demo", "inference", "benchmarks"]
            py_files = []
            for d in search_dirs:
                dir_path = os.path.join(source_path, d)
                if os.path.isdir(dir_path):
                    for f in os.listdir(dir_path):
                        if f.endswith(".py"):
                            py_files.append(os.path.join(dir_path, f))
            # Also top-level .py files (not setup.py)
            for f in os.listdir(source_path):
                if f.endswith(".py") and f not in ("setup.py", "conftest.py", "__init__.py"):
                    py_files.append(os.path.join(source_path, f))

            progress(f"Scanning {len(py_files)} script files for model loading patterns...")

            # Find from_pretrained calls and their imports
            loader_info = None
            tokenizer_info = None
            all_loaders = []  # collect all candidates
            all_tokenizers = []
            for py_file in py_files:
                try:
                    content = open(py_file).read()
                    # Find: SomeClass.from_pretrained("model-id")
                    fp_matches = _re.findall(r'(\w+)\.from_pretrained\(["\']([ ^"\'\ ]+)["\']', content)
                    for cls_name, fp_model_id in fp_matches:
                        if cls_name == "AutoTokenizer":
                            all_tokenizers.append({"class": cls_name, "model_id": fp_model_id, "file": py_file})
                            continue
                        if cls_name.startswith("Auto"):
                            continue  # Skip AutoModel — we want the specific class
                        # Find the import for this class
                        imp_match = _re.search(
                            rf"from\s+([\w.]+)\s+import\s+.*\b{cls_name}\b", content
                        )
                        if imp_match:
                            info = {
                                "module": imp_match.group(1),
                                "class": cls_name,
                                "model_id": fp_model_id,
                                "file": py_file,
                            }
                            all_loaders.append(info)
                            progress(f"Found: from {info['module']} import {cls_name} → .from_pretrained(\"{fp_model_id}\")")
                except Exception:
                    continue

            # Pick the loader whose model_id matches what the user passed (weights repo)
            if all_loaders:
                # Prefer exact model_id match, then partial match, then first
                for info in all_loaders:
                    if model_id and model_id in info["model_id"]:
                        loader_info = info
                        break
                if not loader_info:
                    for info in all_loaders:
                        if model_id and info["model_id"] in model_id:
                            loader_info = info
                            break
                if not loader_info:
                    loader_info = all_loaders[0]

                progress(f"Selected: {loader_info['class']}.from_pretrained(\"{loader_info['model_id']}\")")

                # Find the tokenizer that's near this loader in the same file
                if all_tokenizers:
                    # Prefer tokenizer from the same file and near the selected model
                    same_file = [t for t in all_tokenizers if t["file"] == loader_info["file"]]
                    tokenizer_info = same_file[-1] if same_file else all_tokenizers[0]

            # ══════════════════════════════════════════════════════════════
            # STEP 2: Use the discovered loader to load the model
            # (only if LLM loader didn't already load it)
            # ══════════════════════════════════════════════════════════════
            if src_model is None and loader_info:
                try:
                    mod = _importlib.import_module(loader_info["module"])
                    cls = getattr(mod, loader_info["class"])
                    # Use model_id from CLI args (weights repo), fall back to what the script uses
                    load_id = model_id or loader_info["model_id"]
                    progress(f"Loading: {loader_info['class']}.from_pretrained({load_id})")
                    src_model = cls.from_pretrained(load_id)
                    progress(f"SUCCESS: Model loaded via {loader_info['class']}")
                except Exception as e:
                    progress(f"Loader from script failed: {e}")
                    # Try with the script's own model_id as fallback
                    if model_id != loader_info["model_id"]:
                        try:
                            progress(f"Retrying with script's model_id: {loader_info['model_id']}")
                            src_model = cls.from_pretrained(loader_info["model_id"])
                            progress(f"SUCCESS with script's model_id")
                        except Exception as e2:
                            progress(f"  Also failed: {e2}")

            if src_model is None and not loader_info:
                progress("No model loading pattern found in repo scripts — trying generic scan...")
                # Fallback: walk packages looking for Model classes with from_pretrained
                import pkgutil
                for entry in os.listdir(source_path):
                    if src_model is not None:
                        break
                    pkg_path = os.path.join(source_path, entry)
                    if not os.path.isdir(pkg_path) or entry.startswith(".") or entry.startswith("_"):
                        continue
                    try:
                        for importer, modname, ispkg in pkgutil.walk_packages(
                            path=[pkg_path], prefix=entry + ".", onerror=lambda x: None
                        ):
                            if src_model is not None:
                                break
                            try:
                                mod = _importlib.import_module(modname)
                                for attr_name in dir(mod):
                                    obj = getattr(mod, attr_name, None)
                                    if (isinstance(obj, type) and
                                        hasattr(obj, "from_pretrained") and
                                        attr_name[0].isupper() and
                                        "Model" in attr_name):
                                        progress(f"Trying {modname}.{attr_name}.from_pretrained({model_id})")
                                        try:
                                            src_model = obj.from_pretrained(model_id)
                                            progress(f"SUCCESS: {attr_name}")
                                        except Exception as e:
                                            progress(f"  Failed: {e}")
                            except Exception:
                                continue
                    except Exception:
                        continue

            if src_model is not None:
                progress("Source model loaded — running reference generation")

                # ── Tokenizer ──
                # NEVER silently substitute a different tokenizer for the model's
                # actual one — see the long comment in the AutoModel branch below
                # for context on the LFM2.5 GPT-2-fallback gibberish incident.
                from transformers import AutoTokenizer
                tokenizer = None
                _src_tok_errors = []
                if tokenizer_info:
                    try:
                        progress(f"Loading tokenizer from repo script: {tokenizer_info['model_id']}")
                        tokenizer = AutoTokenizer.from_pretrained(tokenizer_info["model_id"])
                    except Exception as tok_err:
                        _src_tok_errors.append(f"tokenizer_info={tokenizer_info['model_id']}: {tok_err}")
                        progress(f"  Tokenizer from script failed: {tok_err}")
                if tokenizer is None:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                    except Exception as e:
                        _src_tok_errors.append(f"AutoTokenizer({model_id}): {e}")
                if tokenizer is None:
                    # Direct PreTrainedTokenizerFast fallback for tokenizer_class
                    # declarations the local transformers version can't resolve.
                    try:
                        from transformers import PreTrainedTokenizerFast
                        from huggingface_hub import hf_hub_download
                        _tok_json_path = hf_hub_download(model_id, "tokenizer.json")
                        _kwargs = {"tokenizer_file": _tok_json_path}
                        try:
                            _tcfg_path = hf_hub_download(model_id, "tokenizer_config.json")
                            with open(_tcfg_path) as _tcf:
                                _tcfg = json.load(_tcf)
                            for _stk in ("bos_token", "eos_token", "pad_token", "unk_token"):
                                _val = _tcfg.get(_stk)
                                if isinstance(_val, str):
                                    _kwargs[_stk] = _val
                                elif isinstance(_val, dict) and isinstance(_val.get("content"), str):
                                    _kwargs[_stk] = _val["content"]
                        except Exception:
                            pass
                        tokenizer = PreTrainedTokenizerFast(**_kwargs)
                        progress(f"Loaded tokenizer via direct PreTrainedTokenizerFast: {_tok_json_path}")
                    except Exception as e:
                        _src_tok_errors.append(f"PreTrainedTokenizerFast direct: {e}")
                if tokenizer is None:
                    result = {
                        "success": False,
                        "error": (
                            f"Could not load tokenizer for source model {model_id}. "
                            f"Refusing to silently fall back to a different tokenizer. "
                            f"Attempts: " + " | ".join(_src_tok_errors)
                        ),
                        "failure_category": "tokenizer_load_failed",
                        "tokenizer_load_errors": _src_tok_errors,
                    }
                    print(json.dumps(result))
                    return

                # ── Generate reference text (greedy) ──
                progress("Generating reference text")
                generated_text = ""
                if hasattr(src_model, "generate"):
                    for chunk in src_model.generate(prompt, max_tokens=max_new, temperature=0.0):
                        generated_text += chunk
                elif hasattr(src_model, "generate_tokens"):
                    try:
                        import mlx.core as mx
                        input_ids_gen = mx.array(tokenizer.encode(prompt))[None, :]
                    except ImportError:
                        import torch
                        input_ids_gen = torch.tensor([tokenizer.encode(prompt)])
                    for tok_id in src_model.generate_tokens(input_ids_gen, max_tokens=max_new, temperature=0.0):
                        generated_text += tokenizer.decode([int(tok_id)])
                elif hasattr(src_model, "__call__"):
                    # Direct forward pass + greedy decode
                    try:
                        import mlx.core as mx
                        ids = mx.array(tokenizer.encode(prompt))[None, :]
                    except ImportError:
                        import torch
                        ids = torch.tensor([tokenizer.encode(prompt)])
                    for _ in range(max_new):
                        logits = src_model(ids)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        next_id = int(logits[0, -1].argmax())
                        generated_text += tokenizer.decode([next_id])
                        try:
                            ids = mx.array([[next_id]])
                        except Exception:
                            ids = torch.tensor([[next_id]])
                else:
                    result = {"success": False, "error": "Source model has no generate() or __call__() method"}
                    print(json.dumps(result))
                    return

                full_text = prompt + generated_text
                progress(f"Generated: {full_text[:100]}...")

                # Token IDs
                input_ids = tokenizer.encode(prompt)
                generated_ids = tokenizer.encode(full_text)

                # Save reference_text.txt
                with open(os.path.join(ref_dir, "reference_text.txt"), "w") as f:
                    f.write(full_text)

                # Save test_input_ids.bin
                input_ids_np = np.array(input_ids, dtype=np.int32)
                with open(os.path.join(ref_dir, "test_input_ids.bin"), "wb") as f:
                    f.write(input_ids_np.tobytes())

                # ── Layer captures via __call__ wrapping ──
                progress("Capturing per-layer tensors")
                layer_container = None
                layer_container_attr = None
                MLX_LAYER_PATHS = [
                    "backbone.layers", "model.layers", "transformer.h",
                    "gpt_neox.layers", "transformer.layers",
                ]
                for attr in MLX_LAYER_PATHS:
                    obj = src_model
                    try:
                        for part in attr.split("."):
                            obj = getattr(obj, part)
                        layer_container = obj
                        layer_container_attr = attr
                        progress(f"Found layer container: {attr} ({len(layer_container)} layers)")
                        break
                    except (AttributeError, TypeError):
                        continue

                captured_layers = []
                num_captured = 0
                if layer_container is not None:
                    captures = {}
                    originals = {}
                    for i, layer in enumerate(layer_container):
                        name = f"layer_{i}"
                        originals[name] = layer.__call__
                        def make_wrapper(n, orig):
                            def wrapper(*args, **kwargs):
                                out = orig(*args, **kwargs)
                                inp_t = args[0] if args else None
                                out_t = out[0] if isinstance(out, tuple) else out
                                captures[n] = {"input": inp_t, "output": out_t}
                                return out
                            return wrapper
                        layer.__call__ = make_wrapper(name, layer.__call__)

                    # Run one forward pass for captures
                    progress("Running forward pass for layer captures")
                    try:
                        import mlx.core as mx
                        cap_ids = mx.array(input_ids)[None, :]
                    except ImportError:
                        import torch
                        cap_ids = torch.tensor([input_ids])
                    try:
                        if hasattr(src_model, "prefill"):
                            src_model.prefill(cap_ids)
                        elif hasattr(src_model, "__call__"):
                            src_model(cap_ids)
                        try:
                            mx.eval()
                        except Exception:
                            pass
                    except Exception as fwd_err:
                        progress(f"Forward pass failed: {fwd_err}")

                    # Restore originals
                    for i, layer in enumerate(layer_container):
                        name = f"layer_{i}"
                        if name in originals:
                            layer.__call__ = originals[name]

                    # Save captures
                    manifest_layers = {}
                    for name, cap in sorted(captures.items()):
                        try:
                            def to_np(t):
                                if t is None:
                                    return None
                                try:
                                    import mlx.core as mx
                                    return np.array(t.astype(mx.float32))
                                except Exception:
                                    pass
                                try:
                                    return t.detach().float().cpu().numpy()
                                except Exception:
                                    return np.array(t, dtype=np.float32)
                            inp_arr = to_np(cap["input"])
                            out_arr = to_np(cap["output"])
                            if inp_arr is not None:
                                inp_path = os.path.join(layers_dir, f"{name}_input.bin")
                                inp_arr.tofile(inp_path)
                            if out_arr is not None:
                                out_path = os.path.join(layers_dir, f"{name}_output.bin")
                                out_arr.tofile(out_path)
                            manifest_layers[name] = {
                                "input_shape": list(inp_arr.shape) if inp_arr is not None else [],
                                "output_shape": list(out_arr.shape) if out_arr is not None else [],
                                "input_dtype": "float32",
                                "output_dtype": "float32",
                            }
                            captured_layers.append(name)
                            num_captured += 1
                        except Exception as save_err:
                            progress(f"  Could not save {name}: {save_err}")

                    # Write manifest
                    manifest = {
                        "_captured_layers": captured_layers,
                        "_layer_container": layer_container_attr or "unknown",
                        "_nnport_capture_version": "2026-05-02-no-silent-tokenizer-fallback-v1",
                        "_framework": "source",
                        "layers": manifest_layers,
                    }
                    with open(os.path.join(layers_dir, "manifest.json"), "w") as f:
                        json.dump(manifest, f, indent=2)
                else:
                    progress("WARNING: no layer container found for MLX model")

                # Build test_prompts[] — varied inputs for FinalizePort's tokenizer
                # verification gate. Each entry holds Python-computed input_ids
                # that the C++ tokenizer.encode() must reproduce byte-exactly.
                # Best-effort: prompts the tokenizer rejects are dropped (non-ASCII
                # models without those chars in vocab, etc.).
                _tp_candidates = [
                    prompt,
                    "Hello",
                    "Hello, world!",
                    "The quick brown fox jumps over the lazy dog.",
                    "  leading and   multiple   whitespace  ",
                    "Numbers 0 1 2 3 and code def f(x): return x**2",
                    "Email: user@example.com (2024-01-01)",
                    "\n\tnewlines and\ttabs\n",
                ]
                _test_prompts = []
                _seen = set()
                for _tp in _tp_candidates:
                    if _tp in _seen:
                        continue
                    _seen.add(_tp)
                    try:
                        _ids = tokenizer.encode(_tp)
                        if isinstance(_ids, list):
                            _test_prompts.append({"prompt": _tp, "input_ids": list(_ids)})
                    except Exception:
                        pass

                # Save reference_tokens.json
                ref_tokens = {
                    "model_id": model_id,
                    "prompt": prompt,
                    "max_new_tokens": max_new,
                    "input_ids": input_ids,
                    "generated_ids": generated_ids,
                    "reference_text": full_text,
                    "generated_text": generated_text,
                    "framework": "source",
                    "test_prompts": _test_prompts,
                }
                with open(os.path.join(ref_dir, "reference_tokens.json"), "w") as f:
                    json.dump(ref_tokens, f, indent=2)

                progress(f"Source reference generation complete: {num_captured} layers captured")

                # Detect gibberish
                _gen_toks = generated_ids[len(input_ids):]
                if len(_gen_toks) > 0 and len(set(_gen_toks)) <= 1:
                    progress(f"WARNING: Gibberish — all generated tokens identical (id={_gen_toks[0]})")
                    result = {
                        "success": False,
                        "error": f"Gibberish output: all {len(_gen_toks)} tokens are id={_gen_toks[0]}. Wrong tokenizer?",
                        "gibberish": True,
                    }
                    print(json.dumps(result))
                    return

                result = {
                    "success": True,
                    "reference_text": full_text,
                    "generated_text": generated_text,
                    "num_tokens_generated": len(generated_ids) - len(input_ids),
                    "num_layers_captured": num_captured,
                    "captured_layers": captured_layers,
                    "input_ids_count": len(input_ids),
                    "total_ids_count": len(generated_ids),
                    "framework": "source",
                }
                print(json.dumps(result))
                return

            else:
                # If we had LLM instructions but still couldn't load, report the error
                # clearly — do NOT fall through to HF AutoModel
                if os.path.exists(llm_loader_path):
                    result = {
                        "success": False,
                        "error": (
                            "LLM-analyzed model loading failed and no fallback found. "
                            "The model class could not be imported or loaded. "
                            "Check the Colab logs for the specific error."
                        ),
                    }
                    print(json.dumps(result))
                    return
                progress("No model loader found in source — falling back to PyTorch path")
        except Exception as src_err:
            if source_path and os.path.exists(os.path.join(output_dir, "_llm_loader.json")):
                result = {
                    "success": False,
                    "error": f"Source-driven loading failed: {src_err}",
                }
                print(json.dumps(result))
                return
            progress(f"Source-driven path failed: {src_err} — falling back to PyTorch")

    # ══════════════════════════════════════════════════════════════════
    # ── PYTORCH PATH (original) — only used when NO source_path ──
    # ══════════════════════════════════════════════════════════════════
    tokenizer = None
    model = None

    # ── Tokenizer loading ──
    # CRITICAL: NEVER silently substitute a different tokenizer for the
    # model's actual one. Doing so produces gibberish reference output and
    # breaks every downstream gate (test_input_ids.bin will hold IDs from
    # the wrong vocab; SxS will compare against forward-pass garbage; the
    # entire port is unconvergeable).
    #
    # Past breakage we are explicitly preventing here:
    #   LFM2.5-350M-Base (vocab_size=65536, tokenizer_class=TokenizersBackend
    #   which needs transformers>=5.x). On older transformers, AutoTokenizer
    #   raised, the silent GPT-2 fallback engaged, test_input_ids.bin was
    #   populated with [464, 4701, 3111, 379, 262, 220] (GPT-2 BPE for
    #   "The teacher worked at the "), and the LFM2.5 forward pass produced
    #   "indign teacher Department Department Department" gibberish.
    #
    # Priority: 1) LLM-discovered tokenizer_repo, 2) model_id direct via
    # AutoTokenizer, 3) direct PreTrainedTokenizerFast load of the model's
    # own tokenizer.json (handles unknown tokenizer_class declarations).
    # 4) HARD FAIL — never substitute.
    _tokenizer_load_errors = []

    if tokenizer_repo:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
            progress(f"Loaded tokenizer from LLM-discovered repo: {tokenizer_repo}")
        except Exception as e:
            _tokenizer_load_errors.append(f"tokenizer_repo={tokenizer_repo}: {type(e).__name__}: {e}")
            progress(f"LLM-discovered tokenizer {tokenizer_repo} failed: {e}")

    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            progress(f"Loaded tokenizer from {model_id}")
        except Exception as e:
            _tokenizer_load_errors.append(f"AutoTokenizer({model_id}): {type(e).__name__}: {e}")
            progress(f"AutoTokenizer({model_id}) failed: {e}")

    if tokenizer is None:
        # Direct PreTrainedTokenizerFast fallback. Most failures from
        # AutoTokenizer above are because the tokenizer_config declares a
        # class string the local transformers can't resolve (e.g.
        # "TokenizersBackend" requiring transformers>=5.x). The underlying
        # tokenizer.json is still loadable via the fast tokenizer directly.
        try:
            from transformers import PreTrainedTokenizerFast
            from huggingface_hub import hf_hub_download
            progress(f"Trying direct PreTrainedTokenizerFast for {model_id}")
            _tok_json_path = hf_hub_download(model_id, "tokenizer.json")
            _kwargs = {"tokenizer_file": _tok_json_path}
            try:
                _tcfg_path = hf_hub_download(model_id, "tokenizer_config.json")
                with open(_tcfg_path) as _tcf:
                    _tcfg = json.load(_tcf)
                for _stk in ("bos_token", "eos_token", "pad_token", "unk_token"):
                    _val = _tcfg.get(_stk)
                    if isinstance(_val, str):
                        _kwargs[_stk] = _val
                    elif isinstance(_val, dict) and isinstance(_val.get("content"), str):
                        _kwargs[_stk] = _val["content"]
            except Exception as _tc_err:
                progress(f"  tokenizer_config.json fetch optional, skipped: {_tc_err}")
            tokenizer = PreTrainedTokenizerFast(**_kwargs)
            progress(f"Loaded tokenizer via direct PreTrainedTokenizerFast: {_tok_json_path}")
        except Exception as e:
            _tokenizer_load_errors.append(f"PreTrainedTokenizerFast direct: {type(e).__name__}: {e}")
            progress(f"Direct PreTrainedTokenizerFast failed: {e}")

    if tokenizer is None:
        _err_msg = (
            f"Failed to load tokenizer for {model_id}. "
            f"Refusing to silently substitute a different tokenizer "
            f"(would produce gibberish reference output and break the entire port). "
            f"Attempts: " + " | ".join(_tokenizer_load_errors)
        )
        if any("TokenizersBackend" in _e or "does not exist or is not currently imported" in _e
               for _e in _tokenizer_load_errors):
            _err_msg += (
                ". The model's tokenizer_config.json declares a tokenizer_class "
                "(e.g. 'TokenizersBackend') that requires transformers>=5.x. "
                "Upgrade transformers in the reference_venv: "
                "pip install 'transformers>=5.0' (may also require torch>=2.4). "
                "The TS preflight should detect this and provision the venv accordingly."
            )
        result = {
            "success": False,
            "error": _err_msg,
            "failure_category": "tokenizer_load_failed",
            "tokenizer_load_errors": _tokenizer_load_errors,
        }
        print(json.dumps(result))
        sys.exit(1)

    # ── Vocab-size sanity check ──
    # Catches the wrong-tokenizer-loaded case BEFORE the gibberish reference
    # output is generated. Two failure conditions:
    #   1. tokenizer vocab > model vocab + 256: encoded IDs would fall outside
    #      the embedding table → out-of-range index → forward-pass crash.
    #   2. tokenizer vocab < model vocab * 0.85: tokenizer is much smaller
    #      than the model expects. Catches the silent-fallback class
    #      (e.g. GPT-2 vocab=50257 vs LFM2.5 model vocab=65536, ratio=0.77).
    # Allows model_vocab > tokenizer_vocab modestly because many models pad
    # the embedding matrix for SIMD alignment (LFM2.5: tok=64400, model=65536).
    try:
        _config_path = os.path.join(output_dir, "model_info", "config.json")
        _expected_vocab = None
        if os.path.exists(_config_path):
            with open(_config_path) as _cf:
                _cfg = json.load(_cf)
            _expected_vocab = _cfg.get("vocab_size")
        _actual_vocab = getattr(tokenizer, "vocab_size", None)
        if _actual_vocab is None:
            try:
                _actual_vocab = len(tokenizer.get_vocab())
            except Exception:
                _actual_vocab = None
        if _expected_vocab and _actual_vocab:
            _ev = int(_expected_vocab)
            _av = int(_actual_vocab)
            _bad_too_large = _av > _ev + 256
            _bad_too_small = _av < int(_ev * 0.85)
            if _bad_too_large or _bad_too_small:
                _why = "tokenizer vocab exceeds model embedding range" if _bad_too_large else (
                    "tokenizer vocab is far smaller than model expects "
                    "(strong signal that the wrong tokenizer was loaded — "
                    "e.g. silent GPT-2 fallback for a 64k-vocab model)"
                )
                _err_msg = (
                    f"Tokenizer/model vocab mismatch: model expects {_ev}, "
                    f"loaded tokenizer has {_av}. {_why}. Feeding mismatched-vocab "
                    f"IDs to the model produces gibberish that looks like a math bug."
                )
                result = {
                    "success": False,
                    "error": _err_msg,
                    "failure_category": "tokenizer_vocab_mismatch",
                    "expected_vocab_size": _ev,
                    "actual_vocab_size": _av,
                }
                print(json.dumps(result))
                sys.exit(1)
        progress(f"Tokenizer vocab check: expected={_expected_vocab} actual={_actual_vocab} OK")
    except SystemExit:
        raise
    except Exception as _vs_err:
        progress(f"Tokenizer vocab sanity check skipped (non-fatal): {_vs_err}")

    # PyTorch AutoModel fallback (only if source path didn't load above)
    if model is None:
        try:
            progress(f"Trying AutoModel.from_pretrained({model_id})")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32, trust_remote_code=True
            )
            model.eval()
            progress(f"Loaded model via AutoModel ({type(model).__name__})")
        except Exception as e:
            progress(f"AutoModel failed: {e}")

    if model is None:
        # Determine WHY it failed — check if it's a CUDA dep issue
        cuda_deps = ["mamba_ssm", "flash_attn", "causal_conv1d", "triton"]
        missing_cuda_deps = []
        for dep in cuda_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_cuda_deps.append(dep)

        if missing_cuda_deps and not has_cuda:
            error_msg = (
                f"CANNOT LOAD MODEL ON THIS HARDWARE. "
                f"Missing GPU-only dependencies: {', '.join(missing_cuda_deps)}. "
                f"These packages require NVIDIA CUDA which is not available on this machine. "
                f"Run GenerateReference on a machine with CUDA, then copy the reference/ directory here."
            )
        else:
            error_msg = (
                f"Failed to load model: could not find a working loader for {model_id}. "
                f"Tried: AutoModelForCausalLM, source-specific loading."
            )

        result = {
            "success": False,
            "error": error_msg,
            "requires_cuda": bool(missing_cuda_deps and not has_cuda),
            "missing_deps": missing_cuda_deps,
        }
        print(json.dumps(result))
        sys.exit(1)

    if tokenizer is None:
        result = {
            "success": False,
            "error": "Could not load tokenizer. Provide tokenizer_repo parameter.",
        }
        print(json.dumps(result))
        sys.exit(1)

    # ── Capture PyTorch's actual config consumption to reference/config_used.json ──
    # This is the ground-truth source for "what value did PyTorch use at runtime"
    # for every config field. Build's findConfigDriftViolations gate diffs this
    # against src/model_config.h to catch silent-sentinel bugs (e.g. ROPE_THETA
    # = 0.0f when the real value is nested under config.rope_parameters).
    # Best-effort: any missing field is just None.
    try:
        _cfg = getattr(model, "config", None)
        if _cfg is not None:
            try:
                _cfg_dict = _cfg.to_dict()
            except Exception:
                _cfg_dict = {}
            _cfg_used = {
                "config_dict": _cfg_dict,
                "rope_theta_used": getattr(_cfg, "rope_theta", None),
                "rope_parameters": getattr(_cfg, "rope_parameters", None),
                "rope_scaling": getattr(_cfg, "rope_scaling", None),
                "head_dim_used": getattr(_cfg, "head_dim", None) or (
                    (getattr(_cfg, "hidden_size", 0) // max(1, getattr(_cfg, "num_attention_heads", 1)))
                    if getattr(_cfg, "hidden_size", None) and getattr(_cfg, "num_attention_heads", None)
                    else None
                ),
                "rms_norm_eps": getattr(_cfg, "rms_norm_eps", getattr(_cfg, "layer_norm_eps", None)),
                "norm_eps_used": getattr(_cfg, "norm_eps", None),
                "intermediate_size": getattr(_cfg, "intermediate_size", None),
                "hidden_size": getattr(_cfg, "hidden_size", None),
                "num_attention_heads": getattr(_cfg, "num_attention_heads", None),
                "num_key_value_heads": getattr(_cfg, "num_key_value_heads", None),
                "num_hidden_layers": getattr(_cfg, "num_hidden_layers", None),
                "vocab_size": getattr(_cfg, "vocab_size", None),
                "max_position_embeddings": getattr(_cfg, "max_position_embeddings", None),
                "tie_word_embeddings": getattr(_cfg, "tie_word_embeddings", None),
            }
            _cfg_path = os.path.join(ref_dir, "config_used.json")
            with open(_cfg_path, "w") as _f:
                json.dump(_cfg_used, _f, indent=2, default=str)
            progress(f"Wrote {_cfg_path} (PyTorch's runtime config snapshot)")
    except Exception as _e:
        progress(f"WARN: could not capture config_used.json: {_e}")

    progress("Tokenizing prompt")
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Save input token IDs as int32 binary
    input_ids_np = input_ids[0].numpy().astype("int32")
    input_ids_path = os.path.join(ref_dir, "test_input_ids.bin")
    with open(input_ids_path, "wb") as f:
        f.write(input_ids_np.tobytes())

    # Register hooks on first transformer block to capture per-layer tensors
    progress("Registering layer hooks")
    layer_captures = {}
    hooks = []

    # Per-capture execution-order map. Counter increments on every capture
    # event (forward hook OR sys.settrace local). Hooks update the value
    # on each fire (last-write-wins is fine — module hooks fire once per
    # forward); trace updates only on FIRST sight of each var (so the
    # position reflects when the variable first appears in the function).
    # Read by sxsDebugTs.ts as manifest["_capture_order"]; without this,
    # intermediate-trace captures collapse to alphabetical order and SxS
    # picks the wrong "first divergent" sub-op.
    capture_order = {}
    _capture_seq = [0]

    # Architecture-blind layer-container discovery. A model is a directed
    # graph of named modules; the transformer-block list is, by definition,
    # the longest nn.ModuleList whose children are all the same type. Works
    # for any HF transformer (Llama-family model.layers, GPT-2 transformer.h,
    # GPT-NeoX gpt_neox.layers, BART model.decoder.layers, Mamba backbone.layers,
    # and anything we haven't seen) without per-family enumeration.
    layer_container = None
    layer_container_attr = None
    _layer_candidates = []
    for _name, _module in model.named_modules():
        if isinstance(_module, torch.nn.ModuleList) and len(_module) > 0:
            _child_types = {type(_c).__name__ for _c in _module}
            if len(_child_types) == 1:
                _layer_candidates.append((len(_module), _name, _module))
    if _layer_candidates:
        # Prefer the longest list; ties broken by shortest dotted path
        # (closer to the root — almost always the real transformer-block list).
        _layer_candidates.sort(key=lambda x: (-x[0], len(x[1])))
        layer_container = _layer_candidates[0][2]
        layer_container_attr = _layer_candidates[0][1]
        progress(f"Found layer container (generic): {layer_container_attr} ({len(layer_container)} layers)")
    if layer_container is None:
        progress("WARNING: no nn.ModuleList of identical-typed children found — model has non-standard structure")

    # Capture metadata is populated AT HOOK REGISTRATION TIME (not inside the
    # hook closure — the hook fires during forward, but we want the static
    # module topology which is available when we register). Drives layer
    # contract dump_sites[] descriptions in scaffoldTs.ts::writeLayerContracts.
    # Architecture-blind: every nn.Module has __class__.__name__ and is
    # reachable via named_modules() → qualified path.
    capture_meta = {}

    def _path_parts(path_str):
        return path_str.split(".") if path_str else []

    def _qualname_in_parent(path_str):
        parts = _path_parts(path_str)
        return parts[-1] if parts else ""

    def _parent_module(root, path_str):
        parts = _path_parts(path_str)
        if not parts:
            return None
        obj = root
        for p in parts[:-1]:
            if p.isdigit():
                try:
                    obj = obj[int(p)]
                except Exception:
                    return None
            else:
                obj = getattr(obj, p, None)
                if obj is None:
                    return None
        return obj

    def record_meta(sname, mod, module_path, is_pre_hook=False):
        try:
            parent = _parent_module(model, module_path) if module_path else None
            capture_meta[sname] = {
                "module_path": module_path or None,
                "module_class": type(mod).__name__ if mod is not None else None,
                "parent_class": type(parent).__name__ if parent is not None else None,
                "qualname_in_parent": _qualname_in_parent(module_path),
                "is_pre_hook": is_pre_hook,
            }
        except Exception:
            # Metadata is diagnostic only — never break capture on failure.
            capture_meta[sname] = {
                "module_path": module_path or None,
                "module_class": None,
                "parent_class": None,
                "qualname_in_parent": "",
                "is_pre_hook": is_pre_hook,
            }

    def make_hook(sname):
        def hook_fn(mod, inp, out):
            try:
                inp_tensor = inp[0] if isinstance(inp, tuple) else inp
                out_tensor = out[0] if isinstance(out, tuple) else out
                if isinstance(inp_tensor, torch.Tensor) and isinstance(out_tensor, torch.Tensor):
                    layer_captures[sname] = {
                        "input": inp_tensor.detach().float().cpu(),
                        "output": out_tensor.detach().float().cpu(),
                    }
                    capture_order[sname] = _capture_seq[0]
                    _capture_seq[0] += 1
            except Exception:
                pass
        return hook_fn

    # Hook embedding layer at model root level.
    # IMPORTANT: We hook the raw token embedding module (wte / embed_tokens)
    # as "embedding_wte" for diagnostics, but the CRITICAL capture is
    # "embedding" which must be the COMBINED embedding (token + position)
    # that feeds into the first transformer block. Without this, SxSDebug
    # compares the C++ combined embedding (wte+wpe) against just wte,
    # causing a persistent ~0.45 cosine mismatch that no code change can fix.
    #
    # Architecture-blind discovery: the token embedding is, by construction,
    # the nn.Embedding module whose weight has the largest first dim (vocab).
    # Works for any HF model without per-family path enumeration.
    _emb_candidates = []
    for _name, _module in model.named_modules():
        if isinstance(_module, torch.nn.Embedding):
            _vocab = _module.weight.shape[0] if _module.weight is not None else 0
            _emb_candidates.append((_vocab, _name, _module))
    if _emb_candidates:
        _emb_candidates.sort(key=lambda x: -x[0])
        _emb_vocab, _emb_name, _emb_module = _emb_candidates[0]
        hooks.append(_emb_module.register_forward_hook(make_hook("embedding_wte")))
        record_meta("embedding_wte", _emb_module, _emb_name)
        progress(f"Hooked raw token embedding (generic): {_emb_name} vocab={_emb_vocab} (as embedding_wte)")

    # Capture COMBINED embedding (wte + wpe + dropout) by hooking the first
    # transformer block's input. This is what the C++ embedding layer produces
    # and what SxSDebug should compare against.
    def make_pre_hook(sname):
        def hook_fn(mod, inp):
            try:
                inp_tensor = inp[0] if isinstance(inp, tuple) else inp
                if isinstance(inp_tensor, torch.Tensor):
                    layer_captures[sname] = {
                        "input": inp_tensor.detach().float().cpu(),
                        "output": inp_tensor.detach().float().cpu(),
                    }
                    capture_order[sname] = _capture_seq[0]
                    _capture_seq[0] += 1
            except Exception:
                pass
        return hook_fn

    if layer_container is not None and len(layer_container) > 0:
        hooks.append(layer_container[0].register_forward_pre_hook(make_pre_hook("embedding")))
        # The pre-hook captures the input to block 0 — i.e., the combined
        # embedding regardless of how the model assembled it (wte+wpe for
        # GPT-2, embed_tokens for Llama, etc.).
        record_meta(
            "embedding",
            layer_container[0],
            (f"{layer_container_attr}.0" if layer_container_attr else "layer_container.0"),
            is_pre_hook=True,
        )
        progress("Hooked block 0 pre-hook for combined embedding capture")

    # Hook final norm (the layer after all transformer blocks, before lm_head).
    # Without this capture, SxSDebug cannot tell the user whether the block
    # stack output matches PyTorch — any bug between "last block" and "logits"
    # is invisible.
    #
    # Architecture-blind discovery: the final norm is the *Norm module that
    # is OUTSIDE the layer container (so it sees the stack output). We pick
    # the one whose qualified name does NOT start with the layer-container
    # path. If multiple candidates remain, prefer the deepest one (closest
    # to lm_head topologically).
    _norm_class_substrings = ("LayerNorm", "RMSNorm", "Norm")
    def _is_norm(m):
        n = type(m).__name__
        return any(s in n for s in _norm_class_substrings)
    _final_norm_module = None
    _final_norm_path = None
    if layer_container_attr is not None:
        _layer_prefix = layer_container_attr + "."
        _norm_candidates = [
            (_name, _module) for _name, _module in model.named_modules()
            if _is_norm(_module) and _name and not _name.startswith(_layer_prefix)
        ]
        if _norm_candidates:
            # Deepest path = most dotted segments — usually the immediate
            # pre-lm_head norm (e.g. transformer.ln_f, model.norm, etc).
            _norm_candidates.sort(key=lambda x: -len(x[0].split(".")))
            _final_norm_path, _final_norm_module = _norm_candidates[0]
    if _final_norm_module is not None:
        hooks.append(_final_norm_module.register_forward_hook(make_hook("final_norm")))
        record_meta("final_norm", _final_norm_module, _final_norm_path)
        progress(f"Hooked final norm (generic): {_final_norm_path}")
    else:
        progress("WARNING: no final norm module found outside the layer container")

    # Hook lm_head (the final output projection that produces logits). On
    # tied-weight models (GPT-2, Mamba2) this points at the embedding module,
    # but the hook still fires at forward-time on the logits path so the
    # captured tensor is the pre-softmax logits. Without this capture, the
    # model's actual OUTPUT cannot be compared against PyTorch.
    LM_HEAD_PATHS = [
        "lm_head",                    # HuggingFace standard
        "embed_out",                  # GPT-NeoX
        "output_proj",                # some custom models
        "score",                      # some classifier heads
    ]
    for lh_attr in LM_HEAD_PATHS:
        obj = model
        try:
            for part in lh_attr.split("."):
                obj = getattr(obj, part)
            hooks.append(obj.register_forward_hook(make_hook("lm_head")))
            record_meta("lm_head", obj, lh_attr)
            progress(f"Hooked lm_head: {lh_attr}")
            break
        except AttributeError:
            continue

    if layer_container is not None:
        # Hook ALL layers with indexed names so SxSDebug can match
        # mixer_0 → mixer_0_output.bin, mixer_1 → mixer_1_output.bin, etc.
        # These are embedded models so total reference data is small.
        num_layers = len(layer_container)
        progress(f"Hooking all {num_layers} layers with indexed names")

        # Resolve the layer_container's own dotted path once so we can
        # prefix it onto every captured module. Used for meta recording only.
        _prefix = (layer_container_attr + ".") if layer_container_attr else ""

        for idx in range(num_layers):
            layer = layer_container[idx]

            # Hook the whole layer as "layer_{idx}"
            hooks.append(layer.register_forward_hook(make_hook(f"layer_{idx}")))
            record_meta(f"layer_{idx}", layer, f"{_prefix}{idx}")

            for name, module in layer.named_children():
                # Direct children = composite layers (attn, mlp, mixer, norm, etc.)
                # Name them with index: "mixer_0", "norm_0", "mixer_1", "norm_1", ...
                safe_name = name.replace(".", "_").replace("/", "_")
                if not safe_name:
                    continue

                dump_name = f"{safe_name}_{idx}"
                hooks.append(module.register_forward_hook(make_hook(dump_name)))
                record_meta(dump_name, module, f"{_prefix}{idx}.{name}")

            # For block 0 ONLY, capture at PRIMITIVE boundaries — Linear / Norm /
            # Embedding outputs (and Linear inputs via pre-hooks). These shapes
            # are deterministic from MODEL_CONFIG and every C++ port naturally
            # produces a tensor at the same stage — output of pytorch_linear for
            # post-Linear, output of an RMSNorm/LayerNorm kernel for post-Norm,
            # input to the next Linear for pre-Linear.
            #
            # Previously this block hooked EVERY sub-module via named_modules(),
            # which meant composite modules like MultiHeadCausalAttention had
            # their internal shapes demanded as comparison targets (e.g.,
            # [B, kv_heads, S, hd] pre-GQA-expansion K) — a shape the C++ code
            # naturally doesn't produce at the structurally-equivalent point.
            # Agents then spend infinite cycles trying to "align" dumps that
            # are fundamentally incompatible with their kernel fusion choices.
            #
            # Primitive-only capture is architecture-blind: every transformer
            # decoder has Linear/Norm/Embedding. Captures under this scheme are
            # tagged with comparison="primary" in dump_spec.json and drive SxS
            # alignment checks. Block-level captures (layer_i, mlp_i, attn_i,
            # final_norm, lm_head, embedding) remain but are tagged
            # comparison="inspection" so mismatches don't gate progress.
            def _is_primitive(mod):
                try:
                    if isinstance(mod, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                        return True
                    cls_name = type(mod).__name__
                    # RMSNorm variants (LlamaRMSNorm, GemmaRMSNorm, etc.) aren't in nn
                    # but always end in "Norm" by HF convention.
                    return cls_name.endswith("RMSNorm") or cls_name.endswith("LayerNorm")
                except Exception:
                    return False

            if idx == 0:
                _primitive_hooked = 0
                for sub_name, sub_mod in layer.named_modules():
                    if sub_name == "":
                        continue
                    if sub_name.count(".") > 3:
                        continue  # depth cap for pathological nesting
                    if not _is_primitive(sub_mod):
                        continue
                    safe_sub = sub_name.replace(".", "_").replace("/", "_")
                    if not safe_sub:
                        continue
                    # Post-hook: primitive's output tensor (the C++ kernel result).
                    post_name = f"block0_sub_{safe_sub}_out"
                    hooks.append(sub_mod.register_forward_hook(make_hook(post_name)))
                    record_meta(post_name, sub_mod, f"{_prefix}0.{sub_name}")
                    _primitive_hooked += 1
                    # Pre-hook on Linear: captures the Linear's INPUT. For out_proj
                    # this is the post-softmax attention context; for down_proj
                    # this is the post-activation MLP tensor. For every other
                    # Linear it's the post-norm or post-previous-linear input.
                    # Uniform pre-hooks avoid architecture-specific detection.
                    if isinstance(sub_mod, nn.Linear):
                        pre_name = f"block0_sub_{safe_sub}_in"
                        hooks.append(sub_mod.register_forward_pre_hook(make_pre_hook(pre_name)))
                        record_meta(pre_name, sub_mod, f"{_prefix}0.{sub_name}", is_pre_hook=True)
                        _primitive_hooked += 1
                progress(f"Hooked block 0 primitives ({_primitive_hooked} captures at Linear/Norm/Embedding boundaries)")

    # ── Intermediate-local capture (sys.settrace per target class.forward) ──
    # PyTorch forward hooks only fire at nn.Module boundaries. When the agent
    # needs to bisect INSIDE a single module's forward (e.g. MambaMixer's
    # dt_raw / B / C / scan_out are tensor locals, not submodule outputs),
    # we monkey-patch the target class's forward to install a frame-local
    # tracer that captures named locals at function-return.
    #
    # Driver: (model metadata)'s intermediate_dumps[].
    # Architecture-blind — agent declares which class+var pairs it wants
    # via the RequestIntermediateDumps tool. First invocation per class
    # only (block 0 in the iteration order PyTorch uses).
    intermediate_captured = {}
    # Per-name state for shape-change detection. When a Python local is
    # reassigned to a tensor of a different shape mid-method (e.g. OpenELM's
    # `values` is split [kv_heads,S,D] then repeat_interleave'd [q_heads,S,D]),
    # the current "last-binding-wins" rule means SxS sees ONE reference shape
    # and any C++ NNOPT_LAYER_CHECK that emits a buffer of the OTHER shape
    # produces an alignment failure that no kernel edit can fix. Track every
    # distinct shape and save prior bindings under sibling names `<name>__vN`
    # (1-indexed in source order). The canonical `<name>` still gets the LAST
    # binding (downstream contract unchanged); the siblings let SxS resolve a
    # size-mismatched C++ dump against the binding that matches its shape.
    intermediate_last_shape = {}
    intermediate_alt_count = {}
    # Set of distinct shape-tuples already saved as a sibling for each name.
    # Without dedup, every loop iteration / per-layer re-entry that re-binds
    # the var creates ANOTHER __vN file under the same shape — for OpenELM
    # that produced 23 sibling files for one var across 16 attention layers,
    # most of which were duplicates. Cap to MAX 8 distinct shapes per name
    # to bound disk growth on pathological models.
    intermediate_shapes_seen = {}
    INTERMEDIATE_SIBLING_CAP = 8
    intermediate_dump_specs = []
    # Auto-capture: when a contract has pytorch_reference.class set but no
    # explicit intermediate_dumps, capture EVERY tensor local in that class's
    # forward and emit dumps under <category_prefix>_<varname>. This makes
    # sub-op coverage automatic — agents don't have to call
    # RequestIntermediateDumps for routine cases. Generic, model-agnostic.
    _auto_capture_classes = {}  # class_name -> {"prefix": str, "category": str}
    _CATEGORY_DUMP_PREFIX = {
        "attention":  "block0_sub_attn",
        "mlp":        "block0_sub_mlp",
        "layer_norm": "block0_sub_norm",
        "embedding":  "block0_sub_embed",
        "lm_head":    "block0_sub_lm_head",
        "ssm":        "block0_sub_ssm",
        "moe":        "block0_sub_moe",
    }
    # Per-contract weight_keys, indexed by the contract's class_name. Used
    # below to resolve the live PyTorch class via parameter ownership rather
    # than fragile substring matching.
    _contract_weight_keys = {}
    contracts_dir = os.path.join(output_dir, ".nnport", "layer_contracts")
    if os.path.isdir(contracts_dir):
        for fname in sorted(os.listdir(contracts_dir)):
            if not fname.endswith(".json") or fname == "INDEX.json":
                continue
            try:
                with open(os.path.join(contracts_dir, fname)) as f:
                    _c = json.load(f)
            except Exception:
                continue
            _dumps = _c.get("intermediate_dumps") or []
            # The contract's class_name is nnopt's INTERNAL category label
            # (e.g. "Ssm") which does NOT match the live PyTorch class
            # (e.g. "MambaMixer"). settrace must target the real PyTorch
            # class name; fall back to class_name only if the field is
            # absent (older contracts).
            _pyt_ref = _c.get("pytorch_reference") or {}
            _pyt_cls = _pyt_ref.get("class") if isinstance(_pyt_ref, dict) else None
            _cls_name = _pyt_cls or _c.get("class_name") or fname[:-5]
            _cat = (_c.get("category") or "").lower()
            # Stash the contract's weight_keys alongside the class name so we
            # can resolve to the LIVE PyTorch class via parameter ownership
            # (mechanical, naming-convention-independent) below.
            _wkeys = _c.get("weight_keys") or {}
            _contract_weight_keys.setdefault(_cls_name, {}).update(_wkeys)
            # Explicit specs (if any).
            for d in _dumps:
                _n = d.get("name")
                _v = d.get("var")
                if _n and _v:
                    intermediate_dump_specs.append({
                        "class_name": _cls_name,
                        "name": _n,
                        "var": _v,
                    })
            # Auto mode — additive. Fires whenever a contract has
            # pytorch_reference.class set AND its category has a canonical
            # dump-tag prefix. Captures every tensor local that doesn't have
            # an explicit spec; explicit names win on collision. lm_head /
            # embedding skipped — their forward is trivial and adds noise.
            if _pyt_cls and _cat in _CATEGORY_DUMP_PREFIX and _cat not in ("lm_head", "embedding"):
                _auto_capture_classes[_cls_name] = {
                    "prefix": _CATEGORY_DUMP_PREFIX[_cat],
                    "category": _cat,
                }

    _intermediate_warnings = []
    _intermediate_capture_active = False
    _seen_tensor_locals = {}
    if intermediate_dump_specs or _auto_capture_classes:
        progress(f"Intermediate-local capture: {len(intermediate_dump_specs)} explicit var(s) + {len(_auto_capture_classes)} auto class(es)")
        _by_class = {}
        for _s in intermediate_dump_specs:
            _by_class.setdefault(_s["class_name"], []).append(_s)

        # Map requested class NAME → live class OBJECT(s) on the model.
        # We use class identity (target_class_objs) as the trace filter so
        # methods of the target class — forward, slow_forward, helper methods,
        # nested utility methods — are all traced. Matching by name is too
        # weak when the agent specified the contract's class_name (e.g. "Ssm")
        # but the live PyTorch class is named differently (e.g. "MambaMixer");
        # we already remapped via pytorch_reference.class above, but we still
        # match by class identity here for full robustness.
        _all_target_names = set(_by_class.keys()) | set(_auto_capture_classes.keys())
        _class_objs_by_name = {}
        # Index live classes by name (for exact-match path) and named modules
        # by parameter key (for the mechanical weight-ownership resolution).
        _live_class_by_name = {}
        for _mod in model.modules():
            _cn = type(_mod).__name__
            if _cn not in _live_class_by_name:
                _live_class_by_name[_cn] = type(_mod)
        # Build {full_param_name: owning_module} so that for any weight key
        # the contract names (e.g. "h.0.attn.c_proj.weight") we can find
        # which live module owns it and read its real class type. This is
        # the only naming-convention-independent way: PyTorch's own module
        # tree owns the weights, the safetensors keys are ground truth.
        _param_owner = {}
        try:
            _named_modules = dict(model.named_modules())
            for _pname, _ in model.named_parameters():
                # Strip the trailing ".weight"/".bias"/etc. to get module path.
                if "." in _pname:
                    _mpath = _pname.rsplit(".", 1)[0]
                else:
                    _mpath = ""
                _owner = _named_modules.get(_mpath)
                if _owner is not None:
                    _param_owner[_pname] = _owner
        except Exception as _e:
            progress(f"WARN: param-owner index build failed: {_e}")

        # Resolution order (each step strictly mechanical, no string heuristics):
        #   1) exact name match against any live class
        #   2) weight-key ownership: for each templated weight_key in the
        #      contract, substitute {i}=0..N until a real param key matches;
        #      take type(owner_module).__name__
        #   3) fail loudly — caller surfaces this as success=false.
        def _resolve_via_weight_keys(req_name):
            wkeys = _contract_weight_keys.get(req_name) or {}
            if not wkeys:
                return None
            for _role, _tmpl in wkeys.items():
                if not isinstance(_tmpl, str) or "{i}" not in _tmpl:
                    # Not templated — try the literal key.
                    _candidate_keys = [_tmpl] if isinstance(_tmpl, str) else []
                else:
                    # Try a generous range of layer indices. The first hit wins.
                    _candidate_keys = [_tmpl.replace("{i}", str(_i)) for _i in range(0, 256)]
                for _ck in _candidate_keys:
                    _owner = _param_owner.get(_ck)
                    if _owner is not None:
                        return type(_owner)
            return None

        for _cn in list(_all_target_names):
            if _cn in _live_class_by_name:
                _class_objs_by_name[_cn] = _live_class_by_name[_cn]
                continue
            _resolved_cls = _resolve_via_weight_keys(_cn)
            if _resolved_cls is not None:
                _resolved_name = _resolved_cls.__name__
                _class_objs_by_name[_cn] = _resolved_cls
                progress(
                    f"INFO: resolved contract class '{_cn}' -> live PyTorch class "
                    f"'{_resolved_name}' via weight_keys parameter ownership "
                    f"(naming-convention-independent)."
                )
                # Remap _by_class so specs filed under the contract name are
                # also applied when the tracer fires on instances of the
                # resolved class — and rewrite class_name on each spec so
                # missed-var reporting below references the real class.
                if _cn in _by_class and _resolved_name != _cn:
                    _by_class.setdefault(_resolved_name, []).extend(_by_class[_cn])
                    for _s in _by_class[_resolved_name]:
                        _s["class_name"] = _resolved_name

        # Build target_class_specs: class object -> list of spec dicts.
        _target_class_specs = {}
        for _cn, _cls_obj in _class_objs_by_name.items():
            _target_class_specs[_cls_obj] = list(_by_class.get(_cn, []))
        # Build target_class_auto: class object -> {"prefix": ..., "category": ...}
        _target_class_auto = {}
        for _cn, _info in _auto_capture_classes.items():
            _cls_obj = _class_objs_by_name.get(_cn)
            if _cls_obj is not None:
                _target_class_auto[_cls_obj] = _info

        _missing_classes = _all_target_names - set(_class_objs_by_name.keys())
        if _missing_classes:
            _msg = (
                f"intermediate_dumps requested for unknown class(es) — not present on this model: "
                f"{sorted(_missing_classes)}. Available classes on this model include: "
                f"{sorted({type(m).__name__ for m in model.modules()})[:20]}"
            )
            _intermediate_warnings.append(_msg)
            progress(f"WARN: {_msg}")

        if _target_class_specs or _target_class_auto:
            _captured_classes_done = set()

            def _intermediate_tracer(frame, event, arg):
                # Trace ANY method call whose self argument is an instance
                # of a target class. This catches forward AND slow_forward
                # AND any helper method on the same class — HF Mamba's
                # slow_forward is the actual work; forward is just a
                # CUDA-vs-slow-path dispatcher and patching forward alone
                # captures nothing.
                if event != "call":
                    return None
                _self = frame.f_locals.get("self")
                if _self is None:
                    return None
                _cls = type(_self)
                _specs = _target_class_specs.get(_cls) or []
                _auto_info = _target_class_auto.get(_cls)
                if not _specs and _auto_info is None:
                    return None
                if _cls in _captured_classes_done:
                    return None
                _auto_prefix = _auto_info["prefix"] if _auto_info else None
                def _frame_tracer(f, ev, a):
                    # Capture on BOTH line and return events. LAST-tensor-binding
                    # wins per name: when source reassigns a local mid-method
                    # (e.g. HF Mamba's discrete_time_step is bound twice —
                    # first as the post-transpose dt_proj output, then as the
                    # post-softplus tensor), the value that flows downstream
                    # — and that the C++ port produces at the corresponding
                    # NNOPT_LAYER_CHECK boundary — is the FINAL binding.
                    # Capturing the first binding produces a reference tensor
                    # that is mathematically different from what the C++
                    # kernel emits, causing perpetual SxS failure at cos < 0.5
                    # even with correct math. Overwriting on every event is
                    # strictly more general: for single-binding vars
                    # (B/C/ssm_parameters/gate/...) last == first, no behavior
                    # change; for reassigned vars (discrete_time_step/hidden_states/
                    # scan_output) we now align with the C++ contract.
                    if ev == "line" or ev == "return":
                        for _spec in _specs:
                            _v = f.f_locals.get(_spec["var"])
                            if isinstance(_v, torch.Tensor):
                                try:
                                    _was_new = _spec["name"] not in intermediate_captured
                                    _captured_t = _v.detach().float().cpu()
                                    _captured_t = _apply_layout_override(_captured_t, _spec["name"])
                                    _new_shape = list(_captured_t.shape)
                                    _spec_name = _spec["name"]
                                    # Sibling-on-shape-change: when the var was already captured
                                    # with a DIFFERENT shape (real reassignment, not a no-op
                                    # line event), persist the prior binding under
                                    # f"{name}__v{N}" before overwriting. This lets SxS resolve
                                    # a C++ dump whose shape matches an EARLIER binding instead
                                    # of dead-ending on alignment failure. We use list(shape)
                                    # equality (cheap) rather than tensor identity, because the
                                    # tracer fires once per source line and the value may be
                                    # the same Tensor object on consecutive lines.
                                    if (not _was_new) and intermediate_last_shape.get(_spec_name) != _new_shape:
                                        # Save the PRIOR binding under a sibling name ONLY IF its
                                        # shape hasn't already been saved as a sibling. Without
                                        # this dedup, oscillation between two shapes (e.g. across
                                        # multiple layers in a generation loop) produces a long
                                        # tail of duplicate __vN files. Use a tuple key.
                                        _prior_shape = intermediate_last_shape.get(_spec_name)
                                        _prior_key = tuple(_prior_shape) if _prior_shape is not None else None
                                        _seen = intermediate_shapes_seen.setdefault(_spec_name, set())
                                        if _prior_key is not None and _prior_key not in _seen and len(_seen) < INTERMEDIATE_SIBLING_CAP:
                                            _alt_n = intermediate_alt_count.get(_spec_name, 0) + 1
                                            intermediate_alt_count[_spec_name] = _alt_n
                                            _alt_name = f"{_spec_name}__v{_alt_n}"
                                            intermediate_captured[_alt_name] = intermediate_captured[_spec_name]
                                            capture_order[_alt_name] = _capture_seq[0]
                                            _capture_seq[0] += 1
                                            _seen.add(_prior_key)
                                    intermediate_captured[_spec_name] = _captured_t
                                    intermediate_last_shape[_spec_name] = _new_shape
                                    intermediate_shapes_seen.setdefault(_spec_name, set()).add(tuple(_new_shape))
                                    if _was_new:
                                        # Record execution order on FIRST sight only.
                                        # Last-binding-wins for the VALUE (post-softplus
                                        # etc.); first-sight-wins for the ORDER (so
                                        # SxS bisects in real source-line order).
                                        # Key matches the bare ref-basename
                                        # (sxsDebugTs strips "_output" before lookup).
                                        capture_order[_spec_name] = _capture_seq[0]
                                        _capture_seq[0] += 1
                                except Exception:
                                    pass
                        # Auto mode: capture every tensor local under
                        # <prefix>_<varname>. Skip dunders, "self", and any
                        # var that already has an explicit spec (explicit
                        # wins on naming).
                        if _auto_prefix is not None:
                            try:
                                _explicit_vars = {s["var"] for s in _specs}
                                for _vname, _vval in list(f.f_locals.items()):
                                    if _vname.startswith("_") or _vname == "self":
                                        continue
                                    if _vname in _explicit_vars:
                                        continue
                                    if not isinstance(_vval, torch.Tensor):
                                        continue
                                    _name = _auto_prefix + "_" + _vname
                                    _was_new = _name not in intermediate_captured
                                    _captured_t = _vval.detach().float().cpu()
                                    _captured_t = _apply_layout_override(_captured_t, _name)
                                    _new_shape = list(_captured_t.shape)
                                    if (not _was_new) and intermediate_last_shape.get(_name) != _new_shape:
                                        _prior_shape = intermediate_last_shape.get(_name)
                                        _prior_key = tuple(_prior_shape) if _prior_shape is not None else None
                                        _seen = intermediate_shapes_seen.setdefault(_name, set())
                                        if _prior_key is not None and _prior_key not in _seen and len(_seen) < INTERMEDIATE_SIBLING_CAP:
                                            _alt_n = intermediate_alt_count.get(_name, 0) + 1
                                            intermediate_alt_count[_name] = _alt_n
                                            _alt_name = f"{_name}__v{_alt_n}"
                                            intermediate_captured[_alt_name] = intermediate_captured[_name]
                                            capture_order[_alt_name] = _capture_seq[0]
                                            _capture_seq[0] += 1
                                            _seen.add(_prior_key)
                                    intermediate_captured[_name] = _captured_t
                                    intermediate_last_shape[_name] = _new_shape
                                    intermediate_shapes_seen.setdefault(_name, set()).add(tuple(_new_shape))
                                    if _was_new:
                                        capture_order[_name] = _capture_seq[0]
                                        _capture_seq[0] += 1
                            except Exception:
                                pass
                        if ev == "return":
                            # Record what tensor locals existed at return — used
                            # to give actionable feedback when capture missed a name.
                            try:
                                for _k, _val in f.f_locals.items():
                                    if isinstance(_val, torch.Tensor):
                                        _seen_tensor_locals.setdefault(_cls.__name__, set()).add(_k)
                            except Exception:
                                pass
                            # Mark the class done after one complete return:
                            #  - pure auto (no explicit specs): one return = all locals seen
                            #  - explicit-only: wait until every explicit spec captured
                            #  - additive (both): explicit needs to all be captured;
                            #    auto already captured everything else during the same trace
                            if _specs:
                                if all(s["name"] in intermediate_captured for s in _specs):
                                    _captured_classes_done.add(_cls)
                            elif _auto_prefix is not None:
                                _captured_classes_done.add(_cls)
                    return _frame_tracer
                return _frame_tracer

            _prev_tracer = sys.gettrace()
            sys.settrace(_intermediate_tracer)
            _intermediate_capture_active = True

    # Forward pass for layer captures
    progress("Running forward pass for layer captures")
    try:
        with torch.no_grad():
            outputs = model(input_ids)
    finally:
        if _intermediate_capture_active:
            try:
                sys.settrace(_prev_tracer)
            except Exception:
                sys.settrace(None)

    if intermediate_dump_specs:
        progress(f"Captured {len(intermediate_captured)} intermediate local tensor(s) out of {len(intermediate_dump_specs)} requested")
        # Surface any var the agent requested but which never appeared as a
        # local tensor in any traced frame. Most common cause: the var name
        # doesn't exist in this PyTorch source (typo), or it exists in
        # a method we didn't reach (e.g. the model took the CUDA fast path
        # which bypasses slow_forward).
        _requested_names = {s["name"] for s in intermediate_dump_specs}
        _missed = sorted(_requested_names - set(intermediate_captured.keys()))
        if _missed:
            # Map missed dump-names back to their requested var names + class for
            # an actionable error. The agent's most common failure mode is passing
            # C++ dump-label names instead of actual Python locals.
            _missed_by_class = {}
            for _s in intermediate_dump_specs:
                if _s["name"] in _missed:
                    _missed_by_class.setdefault(_s["class_name"], []).append(_s["var"])
            _detail_lines = []
            for _cn, _vars in _missed_by_class.items():
                _seen = sorted(_seen_tensor_locals.get(_cn, set())) if intermediate_dump_specs else []
                _seen_str = ", ".join(_seen) if _seen else "<none observed>"
                _detail_lines.append(
                    f"  class {_cn}: requested locals NOT FOUND = {sorted(_vars)}; "
                    f"tensor locals that DID exist at return = [{_seen_str}]"
                )
            _msg = (
                "intermediate_dumps NOT captured. The names below are not Python locals in "
                "the traced class. STOP — do not edit C++ blindly. Re-Read "
                "model_info/transformers_src/modeling_<arch>.py, locate the class's "
                ".slow_forward (or .forward if no slow_forward), and copy identifiers "
                "VERBATIM from the LHS of '=' assignments. Then call "
                "RequestIntermediateDumps again with corrected var_names.\n"
                + "\n".join(_detail_lines)
            )
            _intermediate_warnings.append(_msg)
            progress(f"WARN: {_msg}")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Tensor-kind classifier (architecture-blind, purely statistical).
    # Runs once per captured tensor. Results drive metric selection in SxSDebug
    # (KL for probabilities, Hamming for discrete, cos for activations).
    # Degrades safely to "activation" if the stats are ambiguous.
    def classify_tensor_kind(tensor, shape):
        try:
            import numpy as _np
            flat = tensor.reshape(-1).numpy().astype("float32")
            n = flat.size
            if n == 0:
                return "activation"

            # discrete: small unique-value cardinality OR all-integer values.
            # Covers argmax / class-index / routing-selection tensors.
            uniq = _np.unique(flat[: min(n, 4096)])
            if uniq.size <= 16 or _np.all(_np.floor(flat) == flat):
                if uniq.size <= 256:
                    return "discrete"

            mn = float(flat.min())
            mx = float(flat.max())

            # probability: values in [0,1], last-dim rows (approximately) sum to 1.
            # Covers post-softmax attention weights and classifier-head probabilities.
            if mn >= -1e-6 and mx <= 1.0 + 1e-6 and len(shape) >= 1:
                last_dim = shape[-1] if shape[-1] > 0 else n
                if last_dim > 1 and n % last_dim == 0:
                    # Sum rows over the last dim; require rows to ~sum to 1.
                    try:
                        rows = flat.reshape(-1, last_dim)
                        # Sample up to 64 rows to keep this cheap.
                        sample = rows if rows.shape[0] <= 64 else rows[:64]
                        sums = sample.sum(axis=1)
                        if _np.all(_np.abs(sums - 1.0) < 1e-3):
                            return "probability"
                    except Exception:
                        pass

            return "activation"
        except Exception:
            return "activation"

    # Save layer captures
    manifest = {}
    for sname, data in layer_captures.items():
        for kind in ["input", "output"]:
            tensor = data[kind]
            # Apply per-dump layout override (or at minimum force contiguous —
            # hook outputs may be transposed views, and reshape().numpy() of
            # a view materializes bytes in the logical (post-view) shape
            # without round-tripping through contiguous, which historically
            # caused silent layout drift for any module returning a view.
            tensor = _apply_layout_override(tensor, f"{sname}_{kind}")
            flat = tensor.reshape(-1).numpy().astype("float32")
            bin_path = os.path.join(layers_dir, f"{sname}_{kind}.bin")
            with open(bin_path, "wb") as f:
                f.write(flat.tobytes())
            # JSON spec doesn't allow Infinity/NaN. Causal-mask tensors are
            # filled with -inf so flat.mean() == -inf — Python's json.dump
            # happily writes "-Infinity" but JavaScript's JSON.parse rejects
            # it, which silently breaks the TS-side manifest reader (dump_spec
            # regen). Sanitize to None for any non-finite stat.
            def _safe_stat(v):
                fv = float(v)
                return fv if math.isfinite(fv) else None
            manifest[f"{sname}_{kind}"] = {
                "shape": list(tensor.shape),
                "mean": _safe_stat(flat.mean()),
                "std": _safe_stat(flat.std()),
                "min": _safe_stat(flat.min()),
                "max": _safe_stat(flat.max()),
                "numel": int(flat.shape[0]),
                "tensor_kind": classify_tensor_kind(tensor, list(tensor.shape)),
            }

    # Save intermediate-local captures (settrace results). Each writes a
    # *_output.bin file so SxSDebug's existing buildSubOpChain (which scans
    # block0_sub_*_output.bin) finds them with no matcher changes.
    for cap_name, _tensor in intermediate_captured.items():
        try:
            _flat = _tensor.reshape(-1).numpy().astype("float32")
            _bin_path = os.path.join(layers_dir, f"{cap_name}_output.bin")
            with open(_bin_path, "wb") as f:
                f.write(_flat.tobytes())
            def _safe_stat2(v):
                fv = float(v)
                return fv if math.isfinite(fv) else None
            manifest[f"{cap_name}_output"] = {
                "shape": list(_tensor.shape),
                "mean": _safe_stat2(_flat.mean()),
                "std": _safe_stat2(_flat.std()),
                "min": _safe_stat2(_flat.min()),
                "max": _safe_stat2(_flat.max()),
                "numel": int(_flat.shape[0]),
                "tensor_kind": classify_tensor_kind(_tensor, list(_tensor.shape)),
                "_is_intermediate": True,
            }
        except Exception as _e:
            progress(f"WARN: failed to save intermediate capture {cap_name}: {_e}")

    # Create aliases so C++ layer names resolve to reference files.
    # Names are now indexed (e.g. "attn_0", "self_attn_3"), so we strip
    # the trailing "_N" index, look up the base in ALIASES, then re-append
    # the index to produce indexed aliases like "attention_0".
    import re as _re
    import shutil as _shutil

    # Simple base-name aliases. Each entry makes "BASE_N" files also resolvable
    # under "TARGET_N". Keys here are the CAPTURED pytorch-module name; values
    # are additional names we emit copies under so C++ can find them.
    ALIASES = {
        "attn": "attention",
        "self_attn": "attention",
        "self_attention": "attention",
        "ln_1": "layer_norm",
        "ln_2": "layer_norm_2",
        "ln_f": "final_layer_norm",
        "input_layernorm": "layer_norm",
        "post_attention_layernorm": "layer_norm_2",
    }

    # Multi-target aliases: one captured name → N alias names. Used where
    # different model families + C++ scaffolds use different words for the
    # SAME module. "ffn" (OpenELM naming) is also reachable via "mlp" (C++
    # naming) and "feed_forward" / "feedforward" (other HF variants).
    MULTI_ALIASES = {
        "ffn": ["mlp", "feed_forward", "feedforward"],
        "mlp": ["ffn", "feed_forward", "feedforward"],
        "feed_forward": ["mlp", "ffn"],
        "feedforward": ["mlp", "ffn"],
        "mixer": ["attn", "attention"],
    }

    def _emit_alias(src_name: str, dst_name: str, lmap: dict, kinds=("input", "output")) -> None:
        lmap[src_name] = dst_name
        for kind in kinds:
            src = os.path.join(layers_dir, f"{src_name}_{kind}.bin")
            dst = os.path.join(layers_dir, f"{dst_name}_{kind}.bin")
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    _shutil.copy2(src, dst)
                except Exception:
                    pass

    layer_map = {}
    for sname in list(layer_captures.keys()):
        # 1. Single-target alias (ALIASES table). Either direct hit ("embedding")
        #    or strip trailing "_N" index, look up base, re-append index.
        alias = ALIASES.get(sname)
        idx_suffix = ""
        base_name = sname
        if not alias:
            m = _re.match(r'^(.+?)_(d+)$', sname)
            if m:
                base_name, idx_str = m.group(1), m.group(2)
                alias = ALIASES.get(base_name)
                idx_suffix = f"_{idx_str}"
        if alias:
            _emit_alias(sname, alias + idx_suffix, layer_map)

        # 2. Multi-target aliases — emit one copy per target name.
        base_for_multi = base_name
        for target in MULTI_ALIASES.get(base_for_multi, []):
            _emit_alias(sname, target + idx_suffix, layer_map)

    # 3. Token-replacement aliases for block0_sub_* primitives. PyTorch module
    #    naming uses 'ffn' but C++ agent naming often uses 'mlp' -- so
    #    block0_sub_ffn_proj_1_out should ALSO be reachable as block0_sub_mlp_proj_1_out.
    #    Covers every primitive capture inside a renamed composite parent.
    TOKEN_REPLACEMENTS = [
        ("ffn", "mlp"),
        ("mlp", "ffn"),
        ("self_attn", "attn"),
        ("mixer", "attn"),
    ]
    for sname in list(layer_captures.keys()):
        for from_tok, to_tok in TOKEN_REPLACEMENTS:
            # Replace only as a whole word, bracketed by _ or start/end.
            pat = _re.compile(r"(^|_)" + _re.escape(from_tok) + r"(_|$)")
            if pat.search(sname):
                aliased = pat.sub(lambda m, t=to_tok: m.group(1) + t + m.group(2), sname)
                if aliased != sname:
                    _emit_alias(sname, aliased, layer_map)

    # Namespace-prefix aliases: the C++ scaffold often names its dumps with
    # a container prefix (e.g. "backbone_layer_0.bin" instead of "layer_0.bin").
    # Emit aliased reference files on disk so SxSDebug's tier-1 direct match
    # finds them without needing any change to the matcher.
    NAMESPACE_PREFIXES = ["backbone_", "transformer_", "model_"]
    for sname in list(layer_captures.keys()):
        for prefix in NAMESPACE_PREFIXES:
            prefixed = prefix + sname
            if prefixed == sname:
                continue
            _emit_alias(sname, prefixed, layer_map)

    manifest["_layer_map"] = layer_map
    manifest["_captured_layers"] = list(layer_captures.keys())
    # Authoritative execution-order map. Covers BOTH module-hook captures
    # (forward / pre-forward) and sys.settrace intermediate-local captures
    # under one counter, recording the order each name FIRST appeared.
    # sxsDebugTs.ts reads this to order the sub-op chain — without it,
    # intermediate captures fall back to alphabetical sort and SxS reports
    # the LAST sub-op as the first divergence (because alphabetical order
    # puts e.g. "contextualized_states" before "discrete_time_step" / "gate"
    # even though it's actually the final output of the mixer).
    manifest["_capture_order"] = capture_order
    # Per-capture module metadata: {module_path, module_class, parent_class,
    # qualname_in_parent, is_pre_hook}. Drives layer_contracts/*.json
    # dump_sites[] so the agent knows EXACTLY what PyTorch tensor each dump
    # name corresponds to, not just numel/shape. Architecture-blind: derived
    # from type(mod).__name__ + named_modules() paths.
    manifest["_capture_meta"] = {
        k: v for k, v in capture_meta.items() if k in layer_captures
    }
    # Version marker — read by the TS cache check to invalidate stale caches
    # when the embedded script's capture logic changes.
    manifest["_nnport_capture_version"] = "2026-05-02-no-silent-tokenizer-fallback-v1"
    manifest["_layer_container_attr"] = layer_container_attr or "<not_found>"

    manifest_path = os.path.join(layers_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    progress(f"Captured {len(layer_captures)} layer tensors")

    # ── Decode-pass capture (passes 1..D) ──────────────────────────────────
    # Run D greedy decode-step forwards (use_cache=True, seq_q=1, threading
    # past_key_values across iterations) and capture module outputs as
    # <name>__pass<N>_output.bin for N in 1..D. The C++ port's NNOPT_LAYER_CHECK
    # fires on every forward — pass=0 dumps match the prefill captures above,
    # pass>=1 dumps match these decode captures. Without per-pass reference
    # binaries, decode-path bugs (KV-cache offsets, start_pos arithmetic,
    # RoPE drift, sampler row indexing, causal mask shape) misclassify as
    # "prefill weight-loading" failures because the prefill-only reference
    # has no decode-shape ground truth.
    #
    # D=4 catches:
    #   pass=1 → wpe[0]-only / RoPE-pos-0 (any "ignores start_pos" bug)
    #   pass=2 → off-by-one start_pos that happened to alias pass=1 by chance
    #   pass=3,4 → KV cache growth bugs (need ≥2 cache entries to manifest)
    #
    # Failure-tolerant: any exception breaks the loop, logs a warning, and
    # leaves whatever passes were already captured. The rest of the reference
    # (prefill captures, generated text) is unaffected. Encoder-only models
    # (BERT, no use_cache support) cleanly skip via the TypeError fallback.
    decode_passes_target = 4
    pass_layer_rosters = {}  # pass_idx → list[str]
    # Per-pass per-name shape/numel so dump_spec.json can validate decode-shape
    # writes alongside prefill ones. Schema:
    #   { "<name>": { "1": {"numel": 768, "shape": [1,1,768]}, "2": {...}, ... } }
    pass_shapes_per_name = {}
    pkv = None
    next_id = None
    seq_so_far = int(input_ids.shape[1])
    try:
        # Initial prefill to seed pkv + first next_id. Hooks are still
        # registered and will fire — but layer_captures was already saved
        # above for pass=0, so we clear it before the loop body uses it.
        with torch.no_grad():
            try:
                prefill_out = model(input_ids, use_cache=True)
            except TypeError:
                # Encoder-only / models that don't accept use_cache.
                raise RuntimeError("model.forward does not accept use_cache — decode capture skipped")
            pkv = getattr(prefill_out, "past_key_values", None)
            if pkv is None:
                raise RuntimeError("model returned no past_key_values — decode capture skipped (encoder-only?)")
            next_logits = prefill_out.logits[0, -1, :]
            next_id = int(next_logits.argmax().item())

        for pass_idx in range(1, decode_passes_target + 1):
            progress(f"Running decode-step forward (pass={pass_idx}/{decode_passes_target}) for layer captures")
            # Reset capture buffers so this pass's writes are isolated from
            # prior passes. layer_captures came from prefill on first iter and
            # from previous pass on later iters.
            layer_captures.clear()
            try:
                intermediate_captured.clear()
            except Exception:
                pass

            decode_input = torch.tensor([[next_id]])
            # Some HF models require cache_position in decode mode (transformers
            # >=4.41 GPT-2 / Llama families). Pass it when accepted; older models
            # that don't accept it fall back via TypeError below.
            decode_kwargs = {"past_key_values": pkv, "use_cache": True}
            try:
                decode_kwargs["cache_position"] = torch.tensor([seq_so_far])
            except Exception:
                pass

            with torch.no_grad():
                try:
                    decode_out = model(decode_input, **decode_kwargs)
                except TypeError:
                    decode_kwargs.pop("cache_position", None)
                    decode_out = model(decode_input, **decode_kwargs)

            # Thread state across iterations so pass=2..D see growing KV cache.
            new_pkv = getattr(decode_out, "past_key_values", None)
            if new_pkv is not None:
                pkv = new_pkv
            next_id = int(decode_out.logits[0, -1, :].argmax().item())
            seq_so_far += 1

            # Save this pass's captures with pass-tagged filenames.
            pass_layers_list = []
            for sname, data in layer_captures.items():
                try:
                    tensor = _apply_layout_override(data["output"], f"{sname}_output")
                    flat = tensor.reshape(-1).numpy().astype("float32")
                    bin_path = os.path.join(layers_dir, f"{sname}__pass{pass_idx}_output.bin")
                    with open(bin_path, "wb") as f:
                        f.write(flat.tobytes())
                    pass_layers_list.append(sname)
                    # Per-pass shape/numel so dump_spec can validate decode writes.
                    try:
                        shape_list = list(tensor.shape)
                        numel_val = int(tensor.numel())
                        pass_shapes_per_name.setdefault(sname, {})[str(pass_idx)] = {
                            "numel": numel_val,
                            "shape": shape_list,
                        }
                    except Exception:
                        pass
                except Exception as _save_err:
                    progress(f"  pass{pass_idx} save failed for {sname}: {_save_err}")
            # Same for intermediate-local captures (sys.settrace closure).
            for cap_name, _tensor in intermediate_captured.items():
                try:
                    _flat = _tensor.reshape(-1).numpy().astype("float32")
                    _bin_path = os.path.join(layers_dir, f"{cap_name}__pass{pass_idx}_output.bin")
                    with open(_bin_path, "wb") as f:
                        f.write(_flat.tobytes())
                    if cap_name not in pass_layers_list:
                        pass_layers_list.append(cap_name)
                    try:
                        shape_list = list(_tensor.shape)
                        numel_val = int(_tensor.numel())
                        pass_shapes_per_name.setdefault(cap_name, {})[str(pass_idx)] = {
                            "numel": numel_val,
                            "shape": shape_list,
                        }
                    except Exception:
                        pass
                except Exception as _save_err:
                    progress(f"  pass{pass_idx} save failed for {cap_name}: {_save_err}")

            pass_layer_rosters[pass_idx] = pass_layers_list
            progress(f"  captured {len(pass_layers_list)} layer tensors at pass={pass_idx}")

        # Update manifest with per-pass rosters so SxSDebug knows which
        # binaries exist. Re-read + re-write to avoid clobbering everything
        # else. Keep _pass1_captured_layers for backward compatibility with
        # SxS code that hasn't been upgraded yet.
        try:
            with open(manifest_path, "r") as f:
                _man = json.load(f)
        except Exception:
            _man = {}
        if 1 in pass_layer_rosters:
            _man["_pass1_captured_layers"] = pass_layer_rosters[1]
        _man["_pass_captured_layers"] = {
            str(k): v for k, v in pass_layer_rosters.items()
        }
        _man["_decode_passes_captured"] = sorted(pass_layer_rosters.keys())
        # _pass_shapes drives the dump_spec.json passes map, which lets the C++
        # debug_utils size-aware-overwrite filter accept decode-shape writes
        # for the same name as prefill (e.g. embedding 4608 prefill + 768 decode).
        _man["_pass_shapes"] = pass_shapes_per_name
        with open(manifest_path, "w") as f:
            json.dump(_man, f, indent=2)
        progress(f"Captured {len(pass_layer_rosters)} decode passes "
                 f"({sum(len(v) for v in pass_layer_rosters.values())} pass-tagged tensors total)")
    except Exception as _decode_err:
        captured_so_far = sorted(pass_layer_rosters.keys())
        if captured_so_far:
            # Persist whatever we got before the failure.
            try:
                with open(manifest_path, "r") as f:
                    _man = json.load(f)
            except Exception:
                _man = {}
            if 1 in pass_layer_rosters:
                _man["_pass1_captured_layers"] = pass_layer_rosters[1]
            _man["_pass_captured_layers"] = {
                str(k): v for k, v in pass_layer_rosters.items()
            }
            _man["_decode_passes_captured"] = captured_so_far
            _man["_pass_shapes"] = pass_shapes_per_name
            with open(manifest_path, "w") as f:
                json.dump(_man, f, indent=2)
            progress(f"decode capture truncated at pass={captured_so_far[-1]}: {_decode_err}")
        else:
            progress(f"decode capture skipped (no passes captured): {_decode_err}")

    # Greedy autoregressive decode
    progress(f"Generating {max_new} tokens (greedy)")
    generated_ids = input_ids[0].tolist()
    cur_ids = input_ids
    with torch.no_grad():
        for step in range(max_new):
            logits = model(cur_ids).logits
            next_id = logits[0, -1, :].argmax().item()
            generated_ids.append(next_id)
            cur_ids = torch.cat([cur_ids, torch.tensor([[next_id]])], dim=1)
            progress(f"Token {step + 1}/{max_new}: {next_id}")

    # Decode
    reference_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_only = tokenizer.decode(generated_ids[len(input_ids[0]):], skip_special_tokens=True)

    # Save reference files
    ref_text_path = os.path.join(ref_dir, "reference_text.txt")
    with open(ref_text_path, "w") as f:
        f.write(reference_text)

    # Build test_prompts[] — varied inputs for FinalizePort's tokenizer
    # verification gate. Each entry holds Python-computed input_ids
    # that the C++ tokenizer.encode() must reproduce byte-exactly.
    _tp_candidates = [
        prompt,
        "Hello",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "  leading and   multiple   whitespace  ",
        "Numbers 0 1 2 3 and code def f(x): return x**2",
        "Email: user@example.com (2024-01-01)",
        "\n\tnewlines and\ttabs\n",
    ]
    _test_prompts = []
    _seen = set()
    for _tp in _tp_candidates:
        if _tp in _seen:
            continue
        _seen.add(_tp)
        try:
            _ids = tokenizer.encode(_tp)
            if isinstance(_ids, list):
                _test_prompts.append({"prompt": _tp, "input_ids": list(_ids)})
        except Exception:
            pass

    ref_tokens_path = os.path.join(ref_dir, "reference_tokens.json")
    ref_data = {
        "input_ids": input_ids[0].tolist(),
        "generated_ids": generated_ids,
        "prompt": prompt,
        "reference_text": reference_text,
        "generated_text": generated_only,
        "max_new_tokens": max_new,
        "model_id": model_id,
        "test_prompts": _test_prompts,
    }
    with open(ref_tokens_path, "w") as f:
        json.dump(ref_data, f, indent=2)

    # Tokenizer SxS captures intentionally NOT emitted. Tokenizer is verified
    # end-to-end at FinalizePort (release-binary diff of
    # layer_dumps/tokenizer_encode.json vs reference_tokens.json.input_ids),
    # NOT during SxS convergence — convergence runs with --token-ids and
    # bypasses src/tokenizer.cpp::encode entirely. The previous block emitted
    # tokenizer_(encode|decode)_case{N}_output.bin and pollluted
    # manifest._capture_order, which sent the agent to src/tokenizer.cpp on
    # every layer-math divergence. FinalizePort's verifyTokenizerEncode owns
    # tokenizer correctness.

    progress(f"Input: {prompt}")
    progress(f"Generated: {generated_only}")

    # ── Detect gibberish output (Fix 5, 2026-05-03 — strengthened) ──
    # Three orthogonal criteria; ANY trips → fail. Defense-in-depth so
    # different degenerate-output shapes (single-token loop, short-cycle loop,
    # decode-failure binary garbage) all surface as a hard failure rather than
    # producing a useless reference that downstream SxS can't bisect against.
    gen_token_ids = generated_ids[len(input_ids[0]):]
    N_gen = len(gen_token_ids)
    is_gibberish = False
    gibberish_reason = ""

    if N_gen >= 4:
        # 1. Tail-window collapse: last min(8, N) tokens have ≤ 2 unique
        #    values. Catches single-token loops AND short-cycle loops
        #    (e.g. "X Y X Y X Y" or "Department Department Department De").
        tail_window = min(8, N_gen)
        tail = gen_token_ids[-tail_window:]
        tail_unique = set(tail)
        if len(tail_unique) <= 2:
            is_gibberish = True
            gibberish_reason = (
                f"Last {tail_window} generated tokens have only {len(tail_unique)} unique "
                f"value(s): {sorted(tail_unique)}. Model collapsed into a short repetition "
                f"loop — likely wrong tokenizer, broken weights, or untrained head."
            )
        else:
            # 2. Whole-window low diversity: unique-token ratio < 0.4.
            #    Secondary catch for cases where the loop pattern is longer
            #    than 8 tokens but still repetitive overall.
            unique_n = len(set(gen_token_ids))
            ratio = unique_n / N_gen
            if ratio < 0.4:
                is_gibberish = True
                gibberish_reason = (
                    f"Only {unique_n} unique tokens out of {N_gen} (ratio={ratio:.2f} < 0.4). "
                    f"Output is degenerate — likely wrong tokenizer or untrained head."
                )

    # 3. Decoded-text control-char / replacement-char check (independent of
    #    ID stats — catches binary garbage from a tokenizer-decode mismatch
    #    that still produces ID-diverse output).
    if not is_gibberish and generated_only:
        bad_chars = [
            c for c in generated_only
            if c == "\ufffd" or (ord(c) < 32 and c not in "\n\t\r")
        ]
        if len(bad_chars) >= 2:
            is_gibberish = True
            gibberish_reason = (
                f"Decoded text contains {len(bad_chars)} control/replacement char(s) — "
                f"tokenizer decode mismatch."
            )

    if is_gibberish:
        progress(f"WARNING: Gibberish detected — {gibberish_reason}")
        result = {
            "success": False,
            "error": f"Reference generation produced gibberish output: {gibberish_reason}",
            "generated_text": generated_only,
            "reference_text": reference_text,
            "gibberish": True,
            "actionable_advice": "The tokenizer may be wrong. Check that the correct tokenizer is being used for this model.",
        }
        print(json.dumps(result))
        sys.exit(1)

    # Group intermediate captures by base name so the result can render a
    # human-readable summary (e.g. "block0_sub_attn_values: 3 distinct shapes
    # — [12,7,64], [3,7,64], [4,7,64]"). Without this the user has only a
    # bare count and can't tell whether siblings actually fired.
    _intermediate_grouped = {}
    for _cap_name, _cap_t in intermediate_captured.items():
        _base = _cap_name.split("__v")[0] if "__v" in _cap_name else _cap_name
        _entry = _intermediate_grouped.setdefault(_base, {"shapes": [], "names": []})
        _entry["shapes"].append(list(_cap_t.shape))
        _entry["names"].append(_cap_name)
    # Stable order: by base name.
    _intermediate_summary = []
    for _base in sorted(_intermediate_grouped.keys()):
        _g = _intermediate_grouped[_base]
        _intermediate_summary.append({
            "base": _base,
            "num_distinct_shapes": len(_g["shapes"]),
            "shapes": _g["shapes"],
            "names": _g["names"],
        })

    result = {
        "success": True,
        "input_text": prompt,
        "generated_text": generated_only,
        "reference_text": reference_text,
        "reference_dir": ref_dir,
        "num_tokens_generated": max_new,
        "num_layers_captured": len(layer_captures),
        "num_intermediate_captured": len(intermediate_captured),
        "intermediate_warnings": _intermediate_warnings,
        "intermediate_summary": _intermediate_summary,
        "manifest_path": os.path.join(layers_dir, "manifest.json"),
        "input_ids_count": len(input_ids[0]),
        "total_ids_count": len(generated_ids),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"success": False, "error": str(e)}))
