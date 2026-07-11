#!/usr/bin/env python3
"""Generate reference output from a PyTorch/HuggingFace model."""
import sys, os, json, struct, traceback, math


def _install_torch_compat_shims():
    """Bridge torch API signature changes when running on older torch than
    the installed transformers expects. The Tier 2 resolver picks the newest
    transformers whose torch-floor check passes for the platform's torch
    ceiling; transformers may still call newer torch API surfaces from its
    model code (e.g. torch.is_autocast_enabled gained a device_type arg in
    torch 2.4). Without these shims those calls raise TypeError mid-forward.
    Shims are guarded on installed_torch < threshold so they no-op when the
    real API exists. Generic across architectures."""
    try:
        import torch
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


def _resolve_input_audio(source_path, output_dir):
    """Best-effort audio fixture for audio-driven reference capture. Voice
    converters / vocoders consume AUDIO, not a text prompt, so the generic
    text `prompt` is useless for them. Resolution order (generic, no model name):
      1. a workspace-provided fixture under reference/input_fixtures/ or reference/,
      2. a small sample shipped in the source repo (resources/ samples/ assets/
         demo/ examples/ test_audio/ — OpenVoice ships resources/*.mp3),
      3. None (caller falls back to the text path).
    """
    import os as _os, glob as _glob
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    cands = []
    for d in (_os.path.join(output_dir or "", "reference", "input_fixtures"),
              _os.path.join(output_dir or "", "reference")):
        if d and _os.path.isdir(d):
            for f in sorted(_os.listdir(d)):
                if f.lower().endswith(exts):
                    cands.append(_os.path.join(d, f))
    for sub in ("resources", "samples", "assets", "demo", "examples", "test_audio", "."):
        base = _os.path.join(source_path or "", sub)
        if _os.path.isdir(base):
            for f in sorted(_glob.glob(_os.path.join(base, "*"))):
                if f.lower().endswith(exts):
                    cands.append(f)
    return cands[0] if cands else None


def main():
    model_id   = sys.argv[1]
    output_dir = sys.argv[2]
    prompt     = sys.argv[3] if len(sys.argv) > 3 else "The teacher worked at the "
    max_new    = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    source_path = sys.argv[5] if len(sys.argv) > 5 else ""
    tokenizer_repo = sys.argv[6] if len(sys.argv) > 6 else ""

    # Phase 3G — deterministic seeds.
    # Without these, any reference run that consumes RNG (SineGen rand_ini
    # in HiFi-GAN vocoders, dropout during eval=False slip-ups, sampler
    # noise) produces a different byte sequence each invocation. Per-layer
    # cosine validation in evaluateTs.ts then has a noise floor below 1.0
    # in those regions for spurious reasons. Pinning to 42 makes reference
    # dumps reproducible: re-running yields byte-equal layer .bin files.
    # Closes Kokoro Entry 8 finding #13.
    import random as _random
    try:
        import torch as _torch_seed
        _torch_seed.manual_seed(42)
        if hasattr(_torch_seed, "cuda") and _torch_seed.cuda.is_available():
            _torch_seed.cuda.manual_seed_all(42)
    except Exception:
        pass
    try:
        import numpy as _np_seed
        _np_seed.random.seed(42)
    except Exception:
        pass
    _random.seed(42)

    # Normalize model_id: strip HuggingFace URLs to bare repo ID — but ONLY when
    # it is an HF-style id. An absolute local SOURCE PATH (e.g. a downloaded
    # checkpoint dir "/Users/.../model") must be left intact: the old
    # unconditional `.strip("/")` chewed off the leading slash, turning the path
    # into an invalid repo id ("Incorrect path_or_model_id") and breaking every
    # local-source port.
    if not (os.path.isabs(model_id) or os.path.exists(model_id)):
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
    _FORM_TRANSFORMS = []

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
        import transformers
        from transformers import AutoConfig, AutoTokenizer
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
        # Walk UP from source_path to find pyproject.toml / setup.py — many
        # repos put the package config at the parent of the actual source
        # subdirectory. E.g. for `kokoro/kokoro/` (Python module dir), the
        # pyproject.toml is at `kokoro/`. Without this walk-up, pip-install
        # silently no-ops and we fall to sys.path mode (no real install,
        # missing transitive deps like misaki / loguru / etc.).
        install_root = None
        probe = source_path
        for _ in range(3):  # check source_path, parent, grandparent
            if (os.path.exists(os.path.join(probe, "pyproject.toml"))
                or os.path.exists(os.path.join(probe, "setup.py"))
                or os.path.exists(os.path.join(probe, "setup.cfg"))):
                install_root = probe
                break
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        setup_py = os.path.join(install_root, "setup.py") if install_root else None
        pyproject = os.path.join(install_root, "pyproject.toml") if install_root else None
        has_setup = install_root is not None
        if has_setup and install_root != source_path:
            progress(f"Found package config at {install_root} (parent of source_path)")
        # Override source_path used for the actual install call below.
        install_path = install_root or source_path

        if has_setup:
            MAX_INSTALL_RETRIES = 3
            # 600s — heavy packages with phonemizer / audio transitives can
            # exceed 120s on slower networks (kokoro pulls misaki[en] which
            # pulls phonemizer + espeak wrappers). 120s timing out then
            # falling to sys.path leaves the package unimportable.
            INSTALL_TIMEOUT = 600
            for attempt in range(1, MAX_INSTALL_RETRIES + 1):
                try:
                    out = subprocess.check_output(
                        [sys.executable, "-m", "pip", "install", "-e", install_path, "-q"],
                        stderr=subprocess.STDOUT,
                        timeout=INSTALL_TIMEOUT,
                    ).decode()
                    progress("Source package installed")
                    # pip install -e writes an `__editable__.*.pth` file under
                    # site-packages. Python reads .pth files only at interpreter
                    # startup, so the CURRENT process has stale sys.path. Append
                    # install_path explicitly + invalidate import caches so the
                    # very next `from <pkg> import ...` resolves to the freshly
                    # installed package instead of raising ModuleNotFoundError.
                    if install_path not in sys.path:
                        sys.path.insert(0, install_path)
                    import importlib as _imp
                    _imp.invalidate_caches()
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
                                [sys.executable, "-m", "pip", "install", "-e", install_path, "--no-deps", "-q"],
                                stderr=subprocess.STDOUT,
                                timeout=INSTALL_TIMEOUT,
                            )
                            progress("Source package installed (--no-deps)")
                            if install_path not in sys.path:
                                sys.path.insert(0, install_path)
                            import importlib as _imp
                            _imp.invalidate_caches()
                            break
                        except Exception:
                            progress("--no-deps install also failed")
                    else:
                        progress(f"Source install failed after {MAX_INSTALL_RETRIES} attempts, falling back to sys.path")
                        # Use install_path (the dir containing pyproject/setup.py),
                        # NOT source_path. sys.path needs the PARENT of the package
                        # so `from <pkg> import ...` resolves to install_path/<pkg>/.
                        sys.path.insert(0, install_path)
                except subprocess.TimeoutExpired:
                    progress(f"Install attempt {attempt} timed out")
                    if attempt >= MAX_INSTALL_RETRIES:
                        progress("Source install timed out, falling back to sys.path")
                        sys.path.insert(0, install_path)
                except Exception as e:
                    progress(f"Install attempt {attempt} error: {e}")
                    if attempt >= MAX_INSTALL_RETRIES:
                        sys.path.insert(0, install_path)
        else:
            # No setup.py/pyproject.toml found anywhere up the tree —
            # add source_path to sys.path so single-file modules still resolve.
            sys.path.insert(0, source_path)
            progress(f"Added {source_path} to sys.path (no setup.py/pyproject.toml found in source or parents)")

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
                # Wrapper loaders (e.g. OpenVoice ToneColorConverter) are NOT
                # nn.Modules: they have no .eval(); the real module is held at
                # .model / .net. Put whichever object supports eval() into eval
                # mode; never crash if the loaded object lacks it.
                def _safe_eval(_m):
                    for _o in (_m, getattr(_m, "model", None), getattr(_m, "net", None)):
                        if _o is not None and hasattr(_o, "eval"):
                            try: _o.eval()
                            except Exception: pass
                    return _m
                try:
                    # Execute the LLM-provided import and load
                    progress(f"Executing: {llm_info['import_statement']}")
                    exec(llm_info["import_statement"], globals())
                    progress(f"Executing: {llm_info['model_load']}")
                    import torch as _torch
                    # Handle BOTH forms the LLM might emit:
                    #   1. expression — "KModel(repo_id='X')"          → eval
                    #   2. assignment — "model = KModel(repo_id='X')"  → exec + lookup
                    # Brittle eval-only logic broke for Kokoro where the LLM
                    # discovered a statement-form load. This is generic across
                    # any non-transformers model whose load API is multi-step.
                    _ml = llm_info["model_load"].strip()
                    import re as _ml_re
                    _assign_match = _ml_re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", _ml)
                    if _assign_match:
                        _var_name = _assign_match.group(1)
                        _load_ns = {"torch": _torch}
                        exec(_ml, globals(), _load_ns)
                        src_model = _load_ns.get(_var_name)
                        if src_model is None:
                            raise RuntimeError(
                                f"LLM load executed '{_ml}' but variable '{_var_name}' was not assigned"
                            )
                    else:
                        src_model = eval(_ml)
                    progress(f"Model loaded via LLM instructions: {type(src_model).__name__}")

                    # Load tokenizer from LLM instructions
                    if llm_info.get("tokenizer_import"):
                        exec(llm_info["tokenizer_import"], globals())
                    if llm_info.get("tokenizer_load"):
                        tokenizer_info = {"llm": True}
                        from transformers import AutoTokenizer
                        # Parse the tokenizer load to extract repo ID
                        import re as _tok_re
                        tok_match = _tok_re.search(r"from_pretrained\(['\"](.*?)['\"]", llm_info["tokenizer_load"])
                        if tok_match:
                            tok_repo = tok_match.group(1)
                            progress(f"Loading tokenizer: {tok_repo}")

                    # Apply extra setup. The LLM follows ML convention and
                    # references the variable as `model` (e.g. "model.eval()").
                    # Our internal name is `src_model`. Alias both names in
                    # the exec namespace so the LLM's idiomatic output works
                    # without it having to know our internal naming. After
                    # exec, sync any rebinding back to src_model.
                    if llm_info.get("extra_setup"):
                        _setup_ns = {"model": src_model, "src_model": src_model, "torch": _torch}
                        exec(llm_info["extra_setup"], globals(), _setup_ns)
                        if "model" in _setup_ns and _setup_ns["model"] is not src_model:
                            src_model = _setup_ns["model"]

                    # ── LLM-provided inference_call: generic path ─────────────────
                    # When the LLM supplied an `inference_call`, the model has its
                    # own end-to-end inference API (TTS pipelines like Kokoro, ASR
                    # encoder-decoders, multi-modal wrappers, etc.). Execute it
                    # in a namespace with `model`, `tokenizer`, `prompt`, `max_new`
                    # bound; capture the result. The standard tokenize+generate
                    # loop below is SKIPPED for these models — the result is
                    # written to `reference/inference_output.json` (truncated /
                    # serialized best-effort) and `reference/reference_tokens.json`
                    # gets a minimal stub so downstream tools have something to
                    # read. Per-layer hook dumps still happen via the existing
                    # hook registration earlier in the script.
                    _llm_inference_call = (llm_info.get("inference_call") or "").strip()
                    if _llm_inference_call and src_model is not None:
                        progress(f"Executing LLM-provided inference_call: {_llm_inference_call[:140]}")
                        # Tokenizer may be None for TTS-like models; pass it through.
                        # `tokenizer` is not yet loaded at this point in the script
                        # — it gets loaded later. For inference_call mode the LLM
                        # already encodes the tokenization step in the expression
                        # (or skips it). Pre-populate `tokenizer` from the LLM
                        # info if a tokenizer_load is present; otherwise None.
                        _ic_tokenizer = None
                        _ic_tok_load = (llm_info.get("tokenizer_load") or "").strip()
                        if _ic_tok_load and not _ic_tok_load.startswith("tokenizer = None"):
                            try:
                                if llm_info.get("tokenizer_import"):
                                    exec(llm_info["tokenizer_import"], globals())
                                _tok_ns = {"torch": _torch}
                                exec(f"_tok = {_ic_tok_load.split('=', 1)[-1].strip() if '=' in _ic_tok_load and not _ic_tok_load.startswith('tokenizer =') else _ic_tok_load}", globals(), _tok_ns)
                                _ic_tokenizer = _tok_ns.get("_tok")
                            except Exception as _tok_err:
                                progress(f"  Tokenizer load skipped/failed for inference_call mode: {_tok_err}")
                        # ── Generic forward-hook registration ─────────────
                        # Register a forward hook on every named module so the
                        # inference_call populates reference/layers/*_output.bin.
                        # The cache validator gates resume on layer count > 0;
                        # without dumps we'd loop forever. Cap depth to avoid
                        # 1000s of dumps in deep models. Tensor extraction is
                        # generic (Tensor / tuple[0] / ModelOutput.{last_hidden_state,logits}).
                        os.makedirs(layers_dir, exist_ok=True)
                        _ic_hooks = []
                        _ic_capture_count = [0]
                        _ic_layer_names = []
                        def _ic_extract_tensor(x):
                            if isinstance(x, _torch.Tensor): return x
                            if isinstance(x, (tuple, list)) and len(x) > 0:
                                return _ic_extract_tensor(x[0])
                            for _a in ("last_hidden_state", "logits", "audio"):
                                if hasattr(x, _a):
                                    v = getattr(x, _a)
                                    if isinstance(v, _torch.Tensor): return v
                            return None
                        def _ic_make_hook(name):
                            def _ic_hook(mod, inp, out):
                                try:
                                    t = _ic_extract_tensor(out)
                                    if t is None: return
                                    safe = name.replace(".", "_").replace("/", "_") or "root"
                                    arr = t.detach().float().cpu().contiguous().numpy()
                                    bin_path = os.path.join(layers_dir, f"{safe}_output.bin")
                                    # Cap each dump to the first _MAX_DUMP_ELEMS floats. Per-layer
                                    # cosine compares the OVERLAPPING PREFIX (forwardGraphValidator's
                                    # flatCosine uses min(len_ref, len_cpp)), so a capped reference is
                                    # still a valid convergence target — and it shrinks the reference
                                    # tarball ~9x (vocoder activations are waveform-length × channels =
                                    # huge), which matters on slow links where a 715MB pull stalls.
                                    _MAX_DUMP_ELEMS = 262144  # 1 MB float32 per layer
                                    _flat = arr.reshape(-1)
                                    (_flat[:_MAX_DUMP_ELEMS] if _flat.size > _MAX_DUMP_ELEMS else _flat).tofile(bin_path)
                                    _ic_capture_count[0] += 1
                                    # Record the REAL PyTorch class (type(mod).__name__) so the
                                    # forward_graph nodes carry their true op_class (Attention,
                                    # MimiDecoderBlock, LayerNorm, ...) instead of a generic
                                    # "captured_layer". Without this, Phase 3B groups EVERY node
                                    # into one shared <Class>.cpp (= monolithic). The module is
                                    # available here as `mod`.
                                    try:
                                        _ic_cls = type(mod).__name__
                                    except Exception:
                                        _ic_cls = "captured_layer"
                                    _ic_layer_names.append({"name": safe, "op": _ic_cls, "shape": list(arr.shape), "dtype": str(arr.dtype)})
                                except Exception:
                                    pass
                            return _ic_hook
                        # Walk model.named_modules(); cap depth to 3 to bound dump count.
                        # Wrapper loaders (OpenVoice ToneColorConverter etc.) are not
                        # nn.Modules — resolve the real module held at .model / .net.
                        _hook_root = src_model
                        if not hasattr(_hook_root, "named_modules"):
                            _hook_root = getattr(src_model, "model", None) or getattr(src_model, "net", None) or src_model
                        _ic_modules = list(_hook_root.named_modules()) if hasattr(_hook_root, "named_modules") else []
                        for _mn, _mod in _ic_modules:
                            if not _mn: continue  # skip root
                            if _mn.count(".") > 3: continue
                            try:
                                _ic_hooks.append(_mod.register_forward_hook(_ic_make_hook(_mn)))
                            except Exception:
                                pass
                        progress(f"Registered {len(_ic_hooks)} forward hooks for inference_call layer capture")

                        try:
                            # Audio-driven models (voice converters, vocoders) take
                            # audio, not the text `prompt` — resolve a fixture so the
                            # inference_call has something real to consume.
                            _input_audio = _resolve_input_audio(source_path, output_dir)
                            if _input_audio:
                                progress(f"inference_call audio fixture: {_input_audio}")
                            try:
                                import numpy as _np
                            except Exception:
                                _np = None
                            _ic_ns = {
                                "model": src_model, "src_model": src_model,
                                "tokenizer": _ic_tokenizer,
                                "prompt": prompt, "max_new": max_new,
                                "torch": _torch, "np": _np, "os": os,
                                "source_path": source_path, "output_dir": output_dir,
                                "input_audio": _input_audio, "audio_path": _input_audio,
                            }
                            # Support BOTH a single expression (eval) and a
                            # multi-statement routine (exec). A voice converter's
                            # forward is multi-step (extract speaker embedding from
                            # `input_audio`, then convert) and is NOT one expression.
                            # Multi-statement snippets assign their output to `result`.
                            try:
                                _ic_result = eval(_llm_inference_call, globals(), _ic_ns)
                            except SyntaxError:
                                exec(_llm_inference_call, globals(), _ic_ns)
                                _ic_result = _ic_ns.get("result", _ic_ns.get("_ic_result"))

                            # ── Reference WAV write (TTS / audio modalities) ───────
                            # The inference_call result is whatever the LLM-derived
                            # snippet returns. Robust extractor handles:
                            #   - torch.Tensor (any dim — flattens to 1D)
                            #   - object with .audio attribute (single chunk)
                            #   - object with .waveform attribute (HF TTS convention)
                            #   - list/tuple/generator of objects with .audio
                            #     (Kokoro KPipeline returns this)
                            # Without this, llm_inference_call modalities silently
                            # leave a 44-byte empty-header WAV behind and Evaluate
                            # has nothing to score against. (2026-05-23, kokoro port.)
                            try:
                                _ic_out_kind = (((_ic_loader_info or {}).get("output_artifact") or {}).get("kind")
                                                if isinstance(globals().get("_ic_loader_info"), dict) else None)
                            except Exception:
                                _ic_out_kind = None
                            # Always try TTS WAV write for llm_inference_call ports;
                            # extractor returns None for non-audio results so the
                            # write step no-ops naturally.
                            try:
                                import wave as _wave
                                import struct as _struct

                                def _ic_resolve_sample_rate():
                                    """Discovery chain: contract → model_info config → 24000 (TTS default)."""
                                    # 1. Contract: reference_protocol.output_artifact.sample_rate_hz
                                    try:
                                        _cp = os.path.join(project_dir, ".nnport", "contract.json")
                                        if os.path.exists(_cp):
                                            with open(_cp) as _f:
                                                _c = json.load(_f)
                                            _sr = (((_c.get("reference_protocol") or {})
                                                    .get("output_artifact") or {}).get("sample_rate_hz"))
                                            if isinstance(_sr, (int, float)) and _sr > 0:
                                                return int(_sr)
                                    except Exception:
                                        pass
                                    # 2. HF convention: model_info/config.json::sampling_rate
                                    try:
                                        _mp = os.path.join(project_dir, "model_info", "config.json")
                                        if os.path.exists(_mp):
                                            with open(_mp) as _f:
                                                _mc = json.load(_f)
                                            for _k in ("sampling_rate", "sample_rate", "audio_sample_rate"):
                                                _v = _mc.get(_k)
                                                if isinstance(_v, (int, float)) and _v > 0:
                                                    return int(_v)
                                    except Exception:
                                        pass
                                    # 3. Modern-TTS default (Kokoro/StyleTTS use 24kHz).
                                    return 24000

                                def _ic_extract_audio_tensor(_res):
                                    """Coerce arbitrary inference_call results into a 1D float32 tensor."""
                                    if isinstance(_res, _torch.Tensor):
                                        out = _res
                                    elif hasattr(_res, "audio") and isinstance(getattr(_res, "audio"), _torch.Tensor):
                                        out = _res.audio
                                    elif hasattr(_res, "waveform") and isinstance(getattr(_res, "waveform"), _torch.Tensor):
                                        out = _res.waveform
                                    elif (hasattr(_res, "__iter__")
                                          and not isinstance(_res, (str, bytes, dict, _torch.Tensor))):
                                        try:
                                            _items = list(_res)
                                        except Exception:
                                            _items = []
                                        _chunks = []
                                        for _it in _items:
                                            for _attr in ("audio", "waveform"):
                                                _v = getattr(_it, _attr, None)
                                                if isinstance(_v, _torch.Tensor):
                                                    _chunks.append(_v.detach().float().cpu().flatten()); break
                                            else:
                                                if isinstance(_it, _torch.Tensor):
                                                    _chunks.append(_it.detach().float().cpu().flatten())
                                        if not _chunks:
                                            return None
                                        out = _torch.cat(_chunks)
                                    else:
                                        return None
                                    out = out.detach().float().cpu().contiguous()
                                    while out.dim() > 1:
                                        if out.shape[0] == 1:
                                            out = out[0]
                                        else:
                                            out = out.flatten(); break
                                    return out

                                _ic_audio = _ic_extract_audio_tensor(_ic_result)
                                if _ic_audio is not None and _ic_audio.numel() > 0:
                                    _ic_sr = _ic_resolve_sample_rate()
                                    _ic_pcm = []
                                    for _s in _ic_audio.numpy().tolist():
                                        if _s > 1.0: _s = 1.0
                                        if _s < -1.0: _s = -1.0
                                        _ic_pcm.append(int(round(_s * 32767.0)))
                                    _ic_wav_path = os.path.join(ref_dir, "output.wav")
                                    with _wave.open(_ic_wav_path, "wb") as _wf:
                                        _wf.setnchannels(1)
                                        _wf.setsampwidth(2)
                                        _wf.setframerate(_ic_sr)
                                        _wf.writeframes(_struct.pack("<%dh" % len(_ic_pcm), *_ic_pcm))
                                    progress(f"Wrote reference output.wav: {_ic_wav_path} ({len(_ic_pcm)} samples @ {_ic_sr}Hz)")
                                else:
                                    progress(f"Reference WAV: extractor returned no audio from inference_call result (type={type(_ic_result).__name__}); WAV left untouched")
                            except Exception as _wav_err:
                                progress(f"Reference WAV write failed (non-fatal): {_wav_err}")

                            # Detach all hooks immediately after to avoid stray captures.
                            for _h in _ic_hooks:
                                try: _h.remove()
                                except Exception: pass
                            progress(f"Layer captures written: {_ic_capture_count[0]} files in {layers_dir}")
                            # Manifest so SxSDebug / autoBisect can discover what was captured.
                            try:
                                with open(os.path.join(layers_dir, "manifest.json"), "w") as _mfh:
                                    json.dump({
                                        "_nnport_capture_version": "1",
                                        "_captured_layers": [ln["name"] for ln in _ic_layer_names],
                                        "produced_by": "_run_reference.py:llm_inference_call",
                                        "layers": {ln["name"]: {"shape": ln["shape"], "dtype": ln["dtype"], "passes": [0]} for ln in _ic_layer_names},
                                    }, _mfh, indent=2)
                            except Exception as _mf_err:
                                progress(f"manifest write failed (non-fatal): {_mf_err}")
                            # forward_graph.json — synthesize from the captured
                            # layers so Evaluate's per-node cosine validator can
                            # run for NON-TEXT models (it keys exclusively on this
                            # file). Each hooked module output is a graph node;
                            # order = capture order; dump_name/module_path = the
                            # sanitized module name (the C++ scaffold emits matching
                            # NNOPT_LAYER_CHECK dumps from these). Without this file,
                            # audio/tensor ports have NO convergence signal at all.
                            # Generic across any inference_call modality.
                            try:
                                # Build nodes from the .bin files that actually
                                # exist on disk (ground truth) — robust to any
                                # _ic_layer_names bookkeeping divergence (a 2nd
                                # RUNREF pass can re-init the names list while the
                                # files remain). Fall back to the names list only
                                # if no files are found.
                                import glob as _glob
                                _bins = sorted(_glob.glob(os.path.join(layers_dir, "*_output.bin")))
                                # Map sanitized dump_name → REAL PyTorch class recorded at capture.
                                _name_to_op = {
                                    _ln["name"]: _ln.get("op")
                                    for _ln in _ic_layer_names
                                    if isinstance(_ln, dict) and _ln.get("name") and _ln.get("op")
                                }
                                if _bins:
                                    _fg_nodes = [
                                        {
                                            "order": _i,
                                            "dump_name": os.path.basename(_b)[:-len("_output.bin")],
                                            "module_path": os.path.basename(_b)[:-len("_output.bin")],
                                            "op": _name_to_op.get(os.path.basename(_b)[:-len("_output.bin")], "captured_layer"),
                                        }
                                        for _i, _b in enumerate(_bins)
                                    ]
                                else:
                                    _fg_nodes = [
                                        {"order": _i, "dump_name": _ln["name"], "module_path": _ln["name"], "op": _ln.get("op", "captured_layer"), "output_shape": _ln["shape"]}
                                        for _i, _ln in enumerate(_ic_layer_names)
                                    ]
                                with open(os.path.join(ref_dir, "forward_graph.json"), "w") as _fgh:
                                    json.dump({
                                        "_produced_by": "_run_reference.py:llm_inference_call",
                                        "nodes": _fg_nodes,
                                    }, _fgh, indent=2)
                                progress(f"forward_graph.json written: {len(_fg_nodes)} nodes (enables per-layer-cosine convergence for non-text ports)")
                            except Exception as _fg_err:
                                progress(f"forward_graph.json write failed (non-fatal): {_fg_err}")
                            progress(f"inference_call returned: type={type(_ic_result).__name__}")
                            # Best-effort: serialize a small descriptor of the result.
                            _ic_desc = {"type": type(_ic_result).__name__}
                            try:
                                if hasattr(_ic_result, "shape"):
                                    _ic_desc["shape"] = list(_ic_result.shape)
                                if hasattr(_ic_result, "dtype"):
                                    _ic_desc["dtype"] = str(_ic_result.dtype)
                                if isinstance(_ic_result, (list, tuple)):
                                    _ic_desc["length"] = len(_ic_result)
                            except Exception:
                                pass
                            with open(os.path.join(ref_dir, "inference_output.json"), "w") as _of:
                                json.dump(_ic_desc, _of, indent=2)
                            # Minimal reference_tokens.json so downstream tools
                            # (Evaluate / SxSDebug) have something to read.
                            with open(os.path.join(ref_dir, "reference_tokens.json"), "w") as _rtf:
                                json.dump({
                                    "prompt": prompt,
                                    "max_new_tokens": max_new,
                                    "model_id": model_id,
                                    "produced_by": "_run_reference.py",   # cache-validator literal
                                    "produced_via": "llm_inference_call",  # informational
                                    "inference_call": _llm_inference_call,
                                    "result_descriptor": _ic_desc,
                                    "input_ids": [0],   # non-empty so cache `hasIds` check passes
                                    "generated_ids": [0],
                                    "generated_text": "",
                                    "reference_text": "",
                                }, _rtf, indent=2)
                            print(json.dumps({
                                "success": True,
                                "produced_by": "llm_inference_call",
                                "inference_call": _llm_inference_call,
                                "result_descriptor": _ic_desc,
                            }))
                            return
                        except Exception as _ic_err:
                            import traceback as _ic_tb
                            _ic_msg = f"inference_call failed: {type(_ic_err).__name__}: {_ic_err}"
                            progress(_ic_msg)
                            # Echo via the standard "LLM loader failed:" prefix so
                            # the TS Tier 3 LLM-retry loop picks it up and asks
                            # the LLM for a corrected inference_call.
                            print(f"LLM loader failed: {_ic_msg}", file=sys.stderr)
                            print(_ic_tb.format_exc(), file=sys.stderr)
                            print(json.dumps({
                                "success": False,
                                "error": _ic_msg,
                                "failure_category": "llm_inference_call_failed",
                            }))
                            return

                    # Device selection — the LLM often returns "cuda" (most ML
                    # repos default to it), but the host may have no CUDA at
                    # all (e.g. macOS / Apple Silicon). Pick the best
                    # available device generically.
                    _req_dev = (llm_info.get("device") or "cpu").lower()
                    if _req_dev == "cuda" and not _torch.cuda.is_available():
                        if _torch.backends.mps.is_available():
                            progress("LLM requested cuda; CUDA unavailable on this host — using mps")
                            _req_dev = "mps"
                        else:
                            progress("LLM requested cuda; CUDA unavailable on this host — using cpu")
                            _req_dev = "cpu"
                    if _req_dev == "cuda":
                        src_model = src_model.cuda()
                    elif _req_dev == "mps":
                        try: src_model = src_model.to("mps")
                        except Exception as _mps_err:
                            progress(f"mps .to() failed ({_mps_err}); falling back to cpu")
                            src_model = src_model.to("cpu")
                    # CPU is the implicit default; no .to() needed.
                    _safe_eval(src_model)
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
                            # Same generic device fallback as primary path.
                            _req_dev_r = (llm_info.get("device") or "cpu").lower()
                            if _req_dev_r == "cuda" and not _torch.cuda.is_available():
                                _req_dev_r = "mps" if _torch.backends.mps.is_available() else "cpu"
                            if _req_dev_r == "cuda":
                                src_model = src_model.cuda()
                            elif _req_dev_r == "mps":
                                try: src_model = src_model.to("mps")
                                except Exception:
                                    src_model = src_model.to("cpu")
                            _safe_eval(src_model)
                        except Exception as retry_err:
                            progress(f"Retry also failed: {retry_err}")
                            src_model = None
                    elif "No module named" in err_str:
                        # Custom packages installed --no-deps (or with a broken
                        # pinned requirements.txt that won't build) import a CHAIN
                        # of light runtime deps at module load — OpenVoice's
                        # api.py top-level-imports se_extractor → librosa/soundfile/
                        # unidecode/... So one missing module surfaces at a time.
                        # Install each into the REF VENV (sys.executable — NOT the
                        # system python) and retry the load, one module per loop,
                        # until the import chain is satisfied. Generic — no model name.
                        import subprocess as _fix_sub, re as _fix_re
                        _PIP_NAME = {  # import name → pip package when they differ
                            "cv2": "opencv-python", "sklearn": "scikit-learn",
                            "PIL": "Pillow", "yaml": "PyYAML", "skimage": "scikit-image",
                            "soundfile": "soundfile", "librosa": "librosa",
                        }
                        _MAX_DEP_FIXES = 25
                        for _dep_attempt in range(_MAX_DEP_FIXES):
                            _mm = _fix_re.search(r"No module named ['\"]([\w\.]+)", err_str)
                            if not _mm:
                                break
                            _mod = _mm.group(1).split(".")[0]
                            _pkg = _PIP_NAME.get(_mod, _mod)
                            progress(f"Auto-installing missing module '{_mod}' (pip: {_pkg}) into ref venv [{_dep_attempt+1}/{_MAX_DEP_FIXES}]...")
                            try:
                                _fix_sub.check_call([sys.executable, "-m", "pip", "install", "-q", _pkg], timeout=600)
                            except Exception as _pip_e:
                                progress(f"pip install {_pkg} failed: {_pip_e}")
                                break
                            for _mn in [m for m in list(sys.modules.keys()) if m.split(".")[0] == _mod]:
                                del sys.modules[_mn]
                            try:
                                exec(llm_info["import_statement"], globals())
                                import torch as _torch
                                src_model = eval(llm_info["model_load"])
                                _rd = (llm_info.get("device") or "cpu").lower()
                                if _rd == "cuda" and not _torch.cuda.is_available():
                                    _rd = "mps" if _torch.backends.mps.is_available() else "cpu"
                                if _rd == "cuda":
                                    src_model = src_model.cuda()
                                elif _rd == "mps":
                                    try: src_model = src_model.to("mps")
                                    except Exception: src_model = src_model.to("cpu")
                                _safe_eval(src_model)
                                progress(f"SUCCESS after auto-installing {_dep_attempt+1} missing module(s): {type(src_model).__name__}")
                                break
                            except Exception as _next_err:
                                err_str = str(_next_err)
                                if "No module named" in err_str:
                                    continue  # next module in the import chain
                                progress(f"Load failed after dep install (non-module error): {err_str}")
                                src_model = None
                                break
                        else:
                            progress(f"Gave up after {_MAX_DEP_FIXES} dep installs; last error: {err_str}")
                            src_model = None
                        if src_model is None:
                            progress(f"Error not auto-fixable: {err_str}")
                    elif "unexpected keyword argument" in err_str:
                        # The LLM sometimes hallucinates a constructor kwarg the
                        # real class doesn't accept (e.g. ToneColorConverter with
                        # enable_watermark=). Strip the offending kwarg(s) from the
                        # model_load expression mechanically and retry — no LLM
                        # round-trip. Generic across any spurious-kwarg load error.
                        import re as _kw_re, torch as _torch
                        _ml_fixed = llm_info["model_load"]
                        for _kw_attempt in range(8):
                            _km = _kw_re.search(r"unexpected keyword argument ['\"](\w+)['\"]", err_str)
                            if not _km:
                                break
                            _bad = _km.group(1)
                            _ml_fixed = _kw_re.sub(r",?\s*\b" + _bad + r"\s*=\s*[^,()]+", "", _ml_fixed, count=1)
                            progress(f"Stripping hallucinated kwarg '{_bad}' and retrying load...")
                            try:
                                src_model = eval(_ml_fixed)
                                _safe_eval(src_model)
                                llm_info["model_load"] = _ml_fixed
                                progress(f"SUCCESS after stripping kwarg(s): {type(src_model).__name__}")
                                break
                            except Exception as _kw_err:
                                err_str = str(_kw_err)
                                if "unexpected keyword argument" in err_str:
                                    continue
                                progress(f"Load failed after kwarg strip (other error): {err_str}")
                                src_model = None
                                break
                        else:
                            src_model = None
                        if src_model is None:
                            progress(f"Error not auto-fixable: {err_str}")
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
                        "_nnport_capture_version": "2026-06-25-local-source-path-no-strip-v9",
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
                # produced_by + version stamps let the TS cache layer
                # distinguish legit-script-produced files from agent-fabricated
                # stubs. The TS user-provided cache path requires produced_by
                # to short-circuit; without it the script re-runs.
                _tx_ver = "unknown"
                _torch_ver = "unknown"
                try:
                    import transformers as _tx_mod
                    _tx_ver = getattr(_tx_mod, "__version__", "unknown")
                except Exception:
                    pass
                try:
                    import torch as _torch_mod
                    _torch_ver = getattr(_torch_mod, "__version__", "unknown")
                except Exception:
                    pass
                ref_tokens = {
                    "produced_by": "_run_reference.py",
                    "script_version": "2026-06-25-local-source-path-no-strip-v9",
                    "transformers_version": _tx_ver,
                    "pytorch_version": _torch_ver,
                    "python_version": sys.version.split()[0] if hasattr(sys, "version") else "unknown",
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
    # Priority: 0) local .nnport/tokenizer.json (PortTokenizer output, no network),
    # 1) LLM-discovered tokenizer_repo, 2) model_id direct via AutoTokenizer,
    # 3) direct PreTrainedTokenizerFast load of the model's own tokenizer.json
    # (handles unknown tokenizer_class declarations). 4) HARD FAIL — never substitute.
    _tokenizer_load_errors = []

    # Tier-0: local .nnport/tokenizer.json — required for repos like
    # apple/OpenELM-270M that don't ship tokenizer assets on HuggingFace.
    # output_dir IS the project directory; .nnport is a subdirectory of it.
    _nnport_dir = os.path.join(output_dir, ".nnport")
    _local_tok_json = os.path.join(_nnport_dir, "tokenizer.json")
    if tokenizer is None and os.path.exists(_local_tok_json):
        try:
            from transformers import PreTrainedTokenizerFast as _PTTF
            tokenizer = _PTTF(tokenizer_file=_local_tok_json)
            progress(f"Loaded tokenizer from local .nnport/tokenizer.json")
        except Exception as _e:
            _tokenizer_load_errors.append(f"local .nnport/tokenizer.json: {type(_e).__name__}: {_e}")
            progress(f"Local tokenizer.json load failed: {_e}")

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

    # ── Patch transformers CVE-2025-32434 check upstream of torch.load ────────
    # transformers >= 4.50 calls check_torch_load_is_safe() BEFORE invoking
    # torch.load and refuses .bin checkpoints on torch < 2.6. A monkey-patch
    # of torch.load itself is too late — the check fires first. Patch the
    # transformers function directly. Safe in our isolated reference venv:
    # we trust the HF source and only need CPU inference for ground-truth dumps.
    try:
        import importlib as _il
        for _modpath in (
            "transformers.utils.import_utils",
            "transformers.modeling_utils",
            "transformers.utils",
        ):
            try:
                _mod = _il.import_module(_modpath)
                if hasattr(_mod, "check_torch_load_is_safe"):
                    setattr(_mod, "check_torch_load_is_safe", lambda *a, **kw: None)
            except Exception:
                pass
    except Exception as _cve_err:
        progress(f"CVE check patch skipped (non-fatal): {_cve_err}")

    # ── Probe checkpoint shapes + alias-normalize config (generic across archs) ─
    # Two failure classes are fixed here, both purely mechanical:
    #
    # (1) Alias mismatch: HF Config classes default hidden_size/num_heads/etc.
    #     to power-of-2 values when the model author used a different field name
    #     in config.json (e.g. Mamba uses d_model, GPT-2 uses n_embd). Read raw
    #     config.json, propagate the author-chosen value across all aliases.
    #
    # (2) Derived dims: some checkpoint dims aren't in config.json at all —
    #     they're computed in the model code (e.g. Mamba's num_heads = d_inner /
    #     head_dim). Probe checkpoint key shapes (safetensors header or .bin via
    #     mmap=True torch.load) and OVERRIDE config values from observed shapes.
    #     Checkpoint truth wins over config.json defaults wins over class defaults.
    def _probe_checkpoint_shapes(model_id, hf_token=None):
        """Read parameter shapes from the HF checkpoint without instantiating
        the model. Returns dict[str, tuple[int]]. Tries safetensors first
        (zero-copy header), falls back to .bin via mmap-load, then to sharded
        index files. Best-effort: returns {} if nothing readable."""
        from huggingface_hub import hf_hub_download as _hub_dl
        # Single-file safetensors
        try:
            from safetensors import safe_open as _safe_open
            _path = _hub_dl(model_id, "model.safetensors", token=hf_token)
            shapes = {}
            with _safe_open(_path, framework="pt") as _f:
                for _k in _f.keys():
                    shapes[_k] = tuple(_f.get_slice(_k).get_shape())
            if shapes:
                return shapes
        except Exception:
            pass
        # Sharded safetensors
        try:
            from safetensors import safe_open as _safe_open
            _idx_path = _hub_dl(model_id, "model.safetensors.index.json", token=hf_token)
            with open(_idx_path) as _f:
                _idx = json.load(_f)
            shapes = {}
            _file_to_keys = {}
            for _k, _fn in _idx.get("weight_map", {}).items():
                _file_to_keys.setdefault(_fn, []).append(_k)
            for _fn, _keys in _file_to_keys.items():
                _shard_path = _hub_dl(model_id, _fn, token=hf_token)
                with _safe_open(_shard_path, framework="pt") as _sf:
                    for _k in _keys:
                        shapes[_k] = tuple(_sf.get_slice(_k).get_shape())
            if shapes:
                return shapes
        except Exception:
            pass
        # pytorch_model.bin via mmap'd torch.load (header only would require
        # a separate pickle parser; mmap=True keeps memory low even for big files)
        for _bin_name in ("pytorch_model.bin", "pytorch_model.bin.index.json"):
            try:
                if _bin_name.endswith(".index.json"):
                    _idx_path = _hub_dl(model_id, _bin_name, token=hf_token)
                    with open(_idx_path) as _f:
                        _idx = json.load(_f)
                    shapes = {}
                    _file_to_keys = {}
                    for _k, _fn in _idx.get("weight_map", {}).items():
                        _file_to_keys.setdefault(_fn, []).append(_k)
                    for _fn, _keys in _file_to_keys.items():
                        _shard_path = _hub_dl(model_id, _fn, token=hf_token)
                        try:
                            _sd = torch.load(_shard_path, map_location="cpu", weights_only=False, mmap=True)
                        except TypeError:
                            _sd = torch.load(_shard_path, map_location="cpu", weights_only=False)
                        for _k in _keys:
                            if _k in _sd:
                                shapes[_k] = tuple(_sd[_k].shape)
                        del _sd
                    if shapes:
                        return shapes
                else:
                    _path = _hub_dl(model_id, _bin_name, token=hf_token)
                    try:
                        _sd = torch.load(_path, map_location="cpu", weights_only=False, mmap=True)
                    except TypeError:
                        _sd = torch.load(_path, map_location="cpu", weights_only=False)
                    shapes = {_k: tuple(_v.shape) for _k, _v in _sd.items()}
                    del _sd
                    if shapes:
                        return shapes
            except Exception:
                continue
        return {}

    def _derive_overrides_from_shapes(shapes, raw_cfg=None):
        """Match canonical key patterns to canonical config field names.
        Generic across architectures — each pattern is an observation about
        what tensor name corresponds to what config concept across HF models.
        Also inverts known shape formulas (e.g. Mamba2 conv_dim = intermediate +
        2*n_groups*state_size) to derive params that aren't in config.json."""
        import re as _re
        overrides = {}
        max_layer_idx = -1
        # Per-mixer Mamba2 shapes for shape-inversion derivation later
        mixer_conv1d = None
        mixer_norm = None
        for _key, _shape in shapes.items():
            # Token embedding: [vocab_size, hidden_size]
            # Pattern matches embedding(s) (singular Mamba style + plural HF style),
            # embed_tokens (Llama), wte (GPT-2), tok_embeddings (LLaMA fork), word_embeddings (BERT/GPT-NeoX)
            if _re.search(r"(embedding|embeddings|embed_tokens|wte|tok_embeddings|word_embeddings).weight$", _key):
                if len(_shape) == 2:
                    overrides.setdefault("vocab_size", _shape[0])
                    for _h in ("hidden_size", "d_model", "n_embd", "model_dim", "dim", "embed_dim"):
                        overrides.setdefault(_h, _shape[1])
            # Final norm: [hidden_size]
            elif _re.search(r"(norm_f|final_layer_norm|ln_f|model.norm).weight$", _key):
                if len(_shape) == 1:
                    for _h in ("hidden_size", "d_model", "n_embd"):
                        overrides.setdefault(_h, _shape[0])
            # LM head: [vocab_size, hidden_size]
            elif _re.search(r"(lm_head|output_projection).weight$", _key):
                if len(_shape) == 2:
                    overrides.setdefault("vocab_size", _shape[0])
            # Mamba SSM heads: A_log, D, dt_bias have shape [num_heads]
            elif _re.search(r".(A_log|D|dt_bias)$", _key):
                if len(_shape) == 1:
                    for _h in ("num_heads", "n_heads", "nheads"):
                        overrides.setdefault(_h, _shape[0])
            # Mamba mixer norm: [intermediate_size] (= d_inner in state-spaces)
            elif _re.search(r".mixer.norm.weight$", _key) and len(_shape) == 1:
                if mixer_norm is None:
                    mixer_norm = _shape[0]
                for _i in ("intermediate_size", "d_intermediate", "d_inner", "ffn_dim"):
                    overrides.setdefault(_i, _shape[0])
            # Mamba mixer conv1d: [conv_dim, 1, d_conv] = [intermediate + 2*n_groups*state_size, 1, d_conv]
            elif _re.search(r".mixer.conv1d.weight$", _key) and len(_shape) >= 1:
                if mixer_conv1d is None:
                    mixer_conv1d = _shape[0]
            # ── SwiGLU / MLP intermediate derivation (cross-family) ──
            # Mirrors extensions/cli/src/tools/nnport/dimensionsAudit.ts::DIM_EVIDENCE.
            # Whichever canonical MLP key matches first gives intermediate_size from
            # weight shape — overriding config (e.g. LFM2 stores raw pre-SwiGLU-adjust
            # value, but weight is the post-adjustment dim). Family-agnostic.
            # Factor-2 entries handle merged gate+up projections (Phi-3 gate_up_proj,
            # OpenELM proj_1).
            elif _re.search(r"(\.mlp\.(gate_proj|up_proj|c_fc|fc1)|\.feed_forward\.(w1|w3)|\.dense_h_to_4h)\.weight$", _key) and len(_shape) == 2:
                for _i in ("intermediate_size", "d_intermediate", "d_inner", "ffn_dim", "n_inner", "ffn_hidden_size"):
                    overrides.setdefault(_i, _shape[0])
            elif _re.search(r"(\.mlp\.gate_up_proj|\.proj_1)\.weight$", _key) and len(_shape) == 2 and _shape[0] % 2 == 0:
                # Merged gate+up: shape[0] = 2 * intermediate
                for _i in ("intermediate_size", "d_intermediate", "d_inner", "ffn_dim", "n_inner", "ffn_hidden_size"):
                    overrides.setdefault(_i, _shape[0] // 2)
            # Attention K/V projections: [num_kv_heads * head_dim, hidden_size]
            # (skipping derivation here — too arch-specific without head_dim)
            # Layer index discovery
            _m = (
                _re.search(r".layers.(d+).", _key)
                or _re.search(r".h.(d+).", _key)
                or _re.search(r".blocks.(d+).", _key)
            )
            if _m:
                _idx = int(_m.group(1))
                if _idx > max_layer_idx:
                    max_layer_idx = _idx
        if max_layer_idx >= 0:
            for _l in ("num_hidden_layers", "n_layer", "n_layers", "num_layers"):
                overrides.setdefault(_l, max_layer_idx + 1)
        # ── Mamba2 n_groups inversion ──
        # HF transformers' Mamba2Config defaults n_groups=8; state-spaces' mamba_ssm
        # defaults ngroups=1. config.json never spells this out — it lives only in the
        # original library's defaults. The checkpoint's conv1d.weight shape encodes it
        # uniquely once intermediate_size and state_size are known:
        #     conv_dim = intermediate_size + 2 * n_groups * state_size
        # state_size: prefer raw_cfg (or its ssm_cfg.d_state), else default 128 (the
        # universal Mamba2 default, identical between HF and state-spaces).
        if mixer_conv1d is not None and mixer_norm is not None:
            _state_size = None
            if isinstance(raw_cfg, dict):
                _state_size = raw_cfg.get("state_size") or raw_cfg.get("d_state")
                _ssm = raw_cfg.get("ssm_cfg")
                if _state_size is None and isinstance(_ssm, dict):
                    _state_size = _ssm.get("d_state") or _ssm.get("state_size")
            if not _state_size or _state_size <= 0:
                _state_size = 128
            _diff = mixer_conv1d - mixer_norm
            if _diff > 0 and _diff % (2 * _state_size) == 0:
                _n_groups = _diff // (2 * _state_size)
                for _g in ("n_groups", "ngroups", "num_groups"):
                    overrides.setdefault(_g, _n_groups)
                # Also propagate state_size so HF picks up our assumed value
                for _s in ("state_size", "d_state", "ssm_state_size"):
                    overrides.setdefault(_s, _state_size)
        return overrides

    # ── Build remapped state_dict for known checkpoint↔HF key naming mismatches ──
    # Some original-library checkpoints use slightly different parameter names
    # than HF's auto-loaded model class expects. Without remapping, from_pretrained
    # silently leaves those layers randomly initialized — model loads but produces
    # gibberish. Concrete: state-spaces models use .embedding.weight (singular);
    # transformers.models.mamba2 expects .embeddings.weight (plural) — embed
    # layer ends up random and every token collapses to id 0.
    # Generic mechanism: a list of (regex, replacement) — easy to extend.
    _KEY_REMAPS = (
        # state-spaces (mamba/mamba2/etc) → HF transformers
        (r".embedding.weight$", ".embeddings.weight"),
    )

    # Cache shape-probe results so we don't re-download for the remap check.
    _checkpoint_shapes = {}
    _remapped_sd = None  # populated by _build_remapped_state_dict() if needed

    def _build_remapped_state_dict():
        """Pre-load checkpoint state_dict, apply KEY_REMAPS, return remapped dict.
        Returns None when no remap pattern matches (caller should fall back to
        standard from_pretrained path which loads from disk directly)."""
        import re as _re
        try:
            _candidates = list(_checkpoint_shapes.keys()) if _checkpoint_shapes else []
        except Exception:
            _candidates = []
        _will_remap = False
        for _src_pat, _ in _KEY_REMAPS:
            if any(_re.search(_src_pat, _k) for _k in _candidates):
                _will_remap = True
                break
        if not _will_remap:
            return None
        from huggingface_hub import hf_hub_download as _hub_dl
        _sd = None
        try:
            from safetensors import safe_open as _safe_open
            _path = _hub_dl(model_id, "model.safetensors")
            _sd = {}
            with _safe_open(_path, framework="pt") as _f:
                for _k in _f.keys():
                    _sd[_k] = _f.get_tensor(_k)
        except Exception:
            pass
        if _sd is None:
            try:
                _path = _hub_dl(model_id, "pytorch_model.bin")
                try:
                    _sd = torch.load(_path, map_location="cpu", weights_only=False, mmap=True)
                except TypeError:
                    _sd = torch.load(_path, map_location="cpu", weights_only=False)
            except Exception:
                return None
        _new_sd = {}
        _remapped_count = 0
        for _k, _v in _sd.items():
            _new_k = _k
            for _src_pat, _dst in _KEY_REMAPS:
                _candidate = _re.sub(_src_pat, _dst, _new_k)
                if _candidate != _new_k:
                    _new_k = _candidate
                    _remapped_count += 1
            _new_sd[_new_k] = _v
        progress(f"Built remapped state_dict ({_remapped_count} key(s) renamed for HF naming compat)")
        return _new_sd

    _normalized_config = None
    try:
        from transformers import AutoConfig as _AutoConfig
        from huggingface_hub import hf_hub_download as _hf_hub_download
        try:
            _raw_cfg_path = _hf_hub_download(model_id, "config.json")
            with open(_raw_cfg_path) as _rcf:
                _raw_cfg = json.load(_rcf)
        except Exception:
            _raw_cfg = None

        if _raw_cfg:
            _normalized_config = _AutoConfig.from_pretrained(model_id, trust_remote_code=True)

            # ── Lift nested config blocks (state-spaces ssm_cfg/attn_cfg, etc) ──
            # state-spaces models store derived params inside nested dicts that HF's
            # flat Config classes never read. Lifting them onto the cfg object via
            # setattr makes them visible to any HF model class that reads
            # config.<field> directly (PretrainedConfig has no nested-dict awareness).
            # Build a flat view of raw_cfg with nested keys lifted to top level.
            _flat_raw_cfg = dict(_raw_cfg)
            for _nested_key in ("ssm_cfg", "attn_cfg", "rope_scaling", "quantization_config"):
                _nested = _raw_cfg.get(_nested_key)
                if isinstance(_nested, dict):
                    for _nk, _nv in _nested.items():
                        if _nk not in _flat_raw_cfg:
                            _flat_raw_cfg[_nk] = _nv
                        try:
                            setattr(_normalized_config, _nk, _nv)
                        except Exception:
                            pass

            # Alias groups: each group maps to ONE canonical concept; whichever
            # alias the author wrote in config.json (or a nested block) wins, then
            # propagated to all aliases on the cfg object.
            _ALIAS_GROUPS = (
                ("hidden_size", "d_model", "n_embd", "model_dim", "dim", "embed_dim"),
                ("num_attention_heads", "num_heads", "n_head", "n_heads"),
                ("intermediate_size", "n_inner", "ffn_dim", "d_intermediate", "ffn_hidden_size", "d_inner"),
                ("num_hidden_layers", "n_layer", "n_layers", "num_layers"),
                ("num_key_value_heads", "n_kv_heads", "num_kv_heads"),
                ("max_position_embeddings", "n_positions", "n_ctx", "seq_len"),
                # Mamba/SSM-specific (HF default ngroups=8, state-spaces=1; head_dim=64; state_size=128)
                ("n_groups", "ngroups", "num_groups"),
                ("head_dim", "headdim", "d_head"),
                ("state_size", "d_state", "ssm_state_size"),
                ("chunk_size", "ssm_chunk_size"),
                ("expand", "expand_factor"),
                ("conv_kernel", "d_conv"),
            )
            _author_chose = {}
            for _aliases in _ALIAS_GROUPS:
                for _a in _aliases:
                    if _a in _flat_raw_cfg and _flat_raw_cfg[_a] is not None:
                        _author_chose[_aliases] = _flat_raw_cfg[_a]
                        break

            for _aliases, _val in _author_chose.items():
                for _a in _aliases:
                    setattr(_normalized_config, _a, _val)

            if _author_chose:
                _summary = ", ".join(f"{aliases[0]}={v}" for aliases, v in _author_chose.items())
                progress(f"Normalized config aliases from raw config.json (incl. nested blocks): {_summary}")

            # Now probe checkpoint shapes and OVERRIDE config from observed truth.
            # Checkpoint shape wins over alias-mapped config wins over class default.
            # Pass the flat raw_cfg so the derivation can read explicit hints
            # (e.g. raw_cfg.ssm_cfg.d_state) before falling back to model defaults.
            try:
                progress(f"Probing checkpoint shapes for {model_id} ...")
                _shapes = _probe_checkpoint_shapes(model_id)
                if _shapes:
                    # Cache for downstream use (key-remap detection)
                    _checkpoint_shapes = _shapes
                    _shape_overrides = _derive_overrides_from_shapes(_shapes, raw_cfg=_flat_raw_cfg)
                    if _shape_overrides:
                        for _ck, _cv in _shape_overrides.items():
                            setattr(_normalized_config, _ck, _cv)
                        _shape_summary = ", ".join(f"{k}={v}" for k, v in _shape_overrides.items())
                        progress(f"Applied checkpoint-derived config overrides: {_shape_summary}")
                    else:
                        progress(f"Checkpoint shapes probed ({len(_shapes)} keys), no canonical patterns matched")
                else:
                    progress("Checkpoint shape probe found nothing readable (non-fatal)")
            except Exception as _shape_err:
                progress(f"Checkpoint shape probe skipped (non-fatal): {_shape_err}")

            # Build a key-remapped state_dict if any KEY_REMAPS would fire.
            # When None, attempts use from_pretrained's standard disk-loading path.
            try:
                _remapped_sd = _build_remapped_state_dict()
            except Exception as _rmp_err:
                progress(f"State_dict remap skipped (non-fatal): {_rmp_err}")
                _remapped_sd = None
    except Exception as _norm_err:
        progress(f"Config alias normalization skipped (non-fatal): {_norm_err}")
        _normalized_config = None

    def _load_kwargs(extra=None):
        kw = dict(trust_remote_code=True, torch_dtype=torch.float32)
        if _normalized_config is not None:
            kw["config"] = _normalized_config
        if _remapped_sd is not None:
            kw["state_dict"] = _remapped_sd
        if extra:
            kw.update(extra)
        return kw

    # Resolve the correct AutoModelForXxx class (auto_map -> mapping table ->
    # pipeline_tag -> try-chain). Used as the loader for the three attempts
    # below instead of a hardcoded AutoModelForCausalLM — VLMs / ASR / TTS
    # all need different loader classes.
    _RESOLVED_LOADER = None
    _RESOLVED_CLS_NAME = "AutoModelForCausalLM"
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.join(output_dir, ".nnport"))
        from _nnopt_auto_class_resolver import resolve_auto_class as _resolve_auto_class
        try:
            _cfg_for_resolver = _normalized_config
            if _cfg_for_resolver is None:
                _cfg_for_resolver = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            _RESOLVED_CLS_NAME = _resolve_auto_class(model_id, _cfg_for_resolver)
            _RESOLVED_LOADER = getattr(transformers, _RESOLVED_CLS_NAME)
            progress(f"Resolved loader class: {_RESOLVED_CLS_NAME}")
        except Exception as _re_err:
            progress(f"AutoModel resolver failed (falling back to CausalLM): {_re_err}")
            _RESOLVED_LOADER = transformers.AutoModelForCausalLM
            _RESOLVED_CLS_NAME = "AutoModelForCausalLM"
    except Exception as _imp_err:
        progress(f"Resolver import failed (falling back to CausalLM): {_imp_err}")
        _RESOLVED_LOADER = transformers.AutoModelForCausalLM
        _RESOLVED_CLS_NAME = "AutoModelForCausalLM"

    # PyTorch AutoModel fallback (only if source path didn't load above)
    _automodel_last_exc = None
    _attempt_outcomes = []  # one short string per attempt — surfaced verbatim in stderr tail
    if model is None:
        # Attempt 1: prefer safetensors to sidestep CVE-2025-32434 torch.load gate.
        # NOTE: kwarg is torch_dtype (NOT dtype) — universal across transformers
        # 4.40 through 5.x. The shorter dtype alias was added in 4.45.
        try:
            progress(f"Trying {_RESOLVED_CLS_NAME}.from_pretrained({model_id}) [safetensors]")
            model = _RESOLVED_LOADER.from_pretrained(
                model_id, use_safetensors=True, **_load_kwargs()
            )
            model.eval()
            progress(f"Loaded model via AutoModel with safetensors ({type(model).__name__})")
            _attempt_outcomes.append("attempt1[safetensors]: OK")
        except Exception as e_safe:
            _automodel_last_exc = e_safe
            _safe_msg = str(e_safe)
            _attempt_outcomes.append(f"attempt1[safetensors]: FAIL ({type(e_safe).__name__}: {_safe_msg[:140]})")
            # Only log if it's not just "no safetensors file found" — that's expected for old repos.
            if not any(x in _safe_msg.lower() for x in ("safetensor", "no file named", "not found")):
                progress(f"AutoModel (safetensors) failed: {_safe_msg}")

    if model is None:
        # Attempt 2: patch torch.load to allow old pickle-format weights.
        # CVE-2025-32434: PyTorch >= 2.6 / transformers >= 4.50 block loading .bin
        # checkpoints by default. This is safe in our isolated reference venv —
        # we trust the HuggingFace source and only need CPU inference here.
        import torch as _torch
        _orig_torch_load = _torch.load
        def _permissive_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _orig_torch_load(*args, **kwargs)
        _torch.load = _permissive_load
        try:
            progress(f"Trying {_RESOLVED_CLS_NAME}.from_pretrained({model_id}) [permissive load for old-format weights]")
            model = _RESOLVED_LOADER.from_pretrained(model_id, **_load_kwargs())
            model.eval()
            progress(f"Loaded model via AutoModel ({type(model).__name__})")
            _attempt_outcomes.append("attempt2[permissive_torch_load]: OK")
        except Exception as e:
            _automodel_last_exc = e
            _attempt_outcomes.append(f"attempt2[permissive_torch_load]: FAIL ({type(e).__name__}: {str(e)[:140]})")
            progress(f"AutoModel failed: {e}")
        finally:
            _torch.load = _orig_torch_load  # always restore

    if model is None:
        # Attempt 3 (last-ditch): tolerate size_mismatch.
        # Catches padded-vocab quirks (e.g. Mamba2 checkpoint vocab=50288 from
        # pad_vocab_size_multiple=16 vs config vocab_size=50277) and other minor
        # checkpoint-vs-config dim drift. This loads what fits and zero-inits
        # the rest — produces a usable reference for comparison purposes.
        # NOTE: torch.load patch from Attempt 2 is restored by now; re-patch.
        import torch as _torch
        _orig_torch_load2 = _torch.load
        def _permissive_load2(*args, **kwargs):
            kwargs['weights_only'] = False
            return _orig_torch_load2(*args, **kwargs)
        _torch.load = _permissive_load2
        try:
            progress(f"Trying {_RESOLVED_CLS_NAME}.from_pretrained({model_id}) [ignore_mismatched_sizes — last-ditch]")
            model = _RESOLVED_LOADER.from_pretrained(
                model_id, **_load_kwargs({"ignore_mismatched_sizes": True})
            )
            model.eval()
            progress(f"Loaded model via AutoModel with ignore_mismatched_sizes ({type(model).__name__})")
            progress("WARNING: loaded with ignore_mismatched_sizes=True — some weights may be zero-initialized")
            _attempt_outcomes.append("attempt3[ignore_mismatched_sizes]: OK_WITH_WARNINGS")
        except Exception as e:
            _automodel_last_exc = e
            _attempt_outcomes.append(f"attempt3[ignore_mismatched_sizes]: FAIL ({type(e).__name__}: {str(e)[:140]})")
            progress(f"AutoModel ignore_mismatched_sizes attempt failed: {e}")
        finally:
            _torch.load = _orig_torch_load2  # always restore

    # ── Attempt summary — always emit, regardless of outcome ──
    # Stderr tail UI shows only the last few lines; this guarantees per-attempt
    # diagnostics survive truncation so failures are debuggable from the CLI alone.
    if _attempt_outcomes:
        progress("LOAD_ATTEMPT_SUMMARY: " + " | ".join(_attempt_outcomes))
    else:
        progress("LOAD_ATTEMPT_SUMMARY: no attempts ran (model loaded via source path)")

    if model is None:
        # Determine WHY it failed — prioritise the actual exception before guessing CUDA.
        _exc_str = str(_automodel_last_exc) if _automodel_last_exc else ""
        _is_cve_error = (
            "CVE-2025-32434" in _exc_str or
            "vulnerability issue in" in _exc_str or
            "vulnerability issue in" in _exc_str.lower()
        )

        # causal_conv1d / mamba_ssm etc. are fast-path CUDA ops but the slow
        # CPU path works without them — their absence is NOT a loading failure.
        # Only call this CUDA-required when the exception itself demands CUDA.
        cuda_deps = ["mamba_ssm", "flash_attn", "triton"]  # NOT causal_conv1d (has CPU fallback)
        missing_cuda_deps = []
        for dep in cuda_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_cuda_deps.append(dep)

        _exc_demands_cuda = (
            "cuda" in _exc_str.lower() and
            any(x in _exc_str for x in ("require", "not available", "CUDA error")) and
            not _is_cve_error
        )

        if _is_cve_error:
            error_msg = (
                f"torch.load security gate (CVE-2025-32434) blocked loading {model_id}. "
                f"The model ships pickle-format .bin weights which PyTorch >= 2.6 refuses to load. "
                f"Fix options: (1) downgrade torch to < 2.6 in your venv: pip install 'torch<2.6'; "
                f"(2) if the model has a safetensors version, use that repo instead; "
                f"(3) this is NOT a CUDA/GPU requirement — the model can run on CPU once loaded."
            )
        elif missing_cuda_deps and _exc_demands_cuda and not has_cuda:
            error_msg = (
                f"CANNOT LOAD MODEL ON THIS HARDWARE. "
                f"Missing GPU-only dependencies: {', '.join(missing_cuda_deps)}. "
                f"These packages require NVIDIA CUDA which is not available on this machine. "
                f"Run GenerateReference on a machine with CUDA, then copy the reference/ directory here."
            )
        else:
            error_msg = (
                f"Failed to load model: could not find a working loader for {model_id}. "
                f"Last error: {_exc_str[:300] if _exc_str else 'unknown'}. "
                f"Tried: safetensors load, permissive torch.load, AutoModelForCausalLM."
            )

        result = {
            "success": False,
            "error": error_msg,
            "requires_cuda": bool(missing_cuda_deps and _exc_demands_cuda and not has_cuda),
            "is_cve_error": _is_cve_error,
            "missing_deps": missing_cuda_deps,
            "attempt_outcomes": _attempt_outcomes,
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

    # ── Modality dispatch: audio (Whisper, SeamlessM4T, Speech2Text) vs text ──
    # For audio-input models, the encoder consumes `input_features` (mel
    # spectrogram), not `input_ids`. We detect via model.config.model_type
    # and run the AutoProcessor on a small standard audio fixture
    # (hf-internal-testing/librispeech_asr_dummy[0]), then write the
    # captured input_features tensor as `test_input_features.bin` for the
    # C++ binary to load as a fixture (no on-device WAV → mel needed in
    # first port; the agent can add an on-device STFT/mel-filterbank
    # pipeline in a follow-up if realtime audio in is required).
    _model_type = getattr(getattr(model, "config", None), "model_type", "")
    _is_audio_model = _model_type in (
        "whisper", "seamless_m4t", "seamless_m4t_v2", "speech_to_text", "speech_to_text_2",
    )
    input_features = None
    if _is_audio_model:
        progress(f"Audio-input model detected (model_type={_model_type}); using AutoProcessor + librispeech sample")
        try:
            from transformers import AutoProcessor as _AutoProcessor
            _processor = _AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except Exception as _proc_err:
            progress(f"WARN: AutoProcessor.from_pretrained failed ({_proc_err}); audio pipeline may not work")
            _processor = None
        try:
            from datasets import load_dataset as _load_dataset
            _ds = _load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            _sample = _ds[0]["audio"]
            _audio_array = _sample["array"]
            _audio_sr = _sample["sampling_rate"]
            progress(f"Loaded librispeech sample: {len(_audio_array)} samples @ {_audio_sr}Hz")
        except Exception as _ds_err:
            progress(f"WARN: librispeech_asr_dummy fetch failed ({_ds_err}); falling back to 1s silence")
            import numpy as _np
            _audio_array = _np.zeros(16000, dtype=_np.float32)
            _audio_sr = 16000
        if _processor is not None:
            try:
                # padding="max_length" forces the processor to pad the audio
                # array to the encoder's expected length (30 seconds → 3000
                # mel frames for Whisper). Without this, short audio (e.g.
                # the 1s silence fallback) produces a 10-frame mel tensor
                # that the encoder refuses with
                #   ValueError: Whisper expects the mel input features to be of length 3000, but found 10.
                _feat = _processor(_audio_array, sampling_rate=_audio_sr, return_tensors="pt", padding="max_length")
                input_features = _feat.get("input_features")
                if input_features is None:
                    input_features = _feat.get("input_values")  # some processors use input_values
                progress(f"Processor produced input_features: shape={tuple(input_features.shape)} dtype={input_features.dtype}")
            except Exception as _feat_err:
                progress(f"WARN: processor() failed ({_feat_err}); input_features will be None")
        # For ASR / audio encoder-decoder models, the decoder doesn't take
        # the user-typed prompt — it autoregresses from the forced decoder
        # prefix declared by generation_config (Whisper: [sot, lang, task,
        # notimestamps]; SeamlessM4T similar). The C++ binary must
        # autoregress from the SAME prefix to match PyTorch token-by-token.
        # If we wrote the text-tokenization of `prompt` here, the C++ port
        # would start decoding from arbitrary text tokens, never matching
        # PyTorch's model.generate(input_features=…) output regardless of
        # encoder/decoder math correctness.
        _forced_prefix = []
        try:
            _gen_cfg = getattr(model, "generation_config", None) or getattr(model, "config", None)
            _dec_start = getattr(_gen_cfg, "decoder_start_token_id", None)
            if _dec_start is None:
                _dec_start = getattr(getattr(model, "config", None), "decoder_start_token_id", None)
            if _dec_start is not None:
                _forced_prefix.append(int(_dec_start))
            # Whisper-style forced_decoder_ids: list of (index, token_id) tuples,
            # sorted by index. Concatenate token ids in index order after the
            # decoder_start_token_id.
            _fdi = getattr(_gen_cfg, "forced_decoder_ids", None)
            if _fdi:
                try:
                    _pairs = sorted(((int(i), int(t)) for i, t in _fdi), key=lambda p: p[0])
                    _forced_prefix.extend(t for _, t in _pairs)
                except Exception:
                    pass
        except Exception:
            _forced_prefix = []
        if _forced_prefix:
            input_ids = torch.tensor([_forced_prefix], dtype=torch.long)
            progress(f"Audio model: forced decoder prefix = {_forced_prefix} (will be saved to test_input_ids.bin)")
        else:
            # Fallback: a single decoder_start_token_id of 0 lets the binary
            # at least start somewhere. Better than an empty tensor.
            input_ids = torch.tensor([[0]], dtype=torch.long)
            progress("WARN: audio model has no decoder_start_token_id or forced_decoder_ids; using [0]")
        # Save mel features as a binary fixture.
        if input_features is not None:
            try:
                import numpy as _np
                _feat_np = input_features[0].numpy().astype("float32")  # [n_mels, T_mel] or similar
                _feat_path = os.path.join(ref_dir, "test_input_features.bin")
                with open(_feat_path, "wb") as _f:
                    _f.write(_feat_np.tobytes())
                _assets_dir = os.path.join(output_dir, "assets")
                os.makedirs(_assets_dir, exist_ok=True)
                with open(os.path.join(_assets_dir, "test_input_features.bin"), "wb") as _f:
                    _f.write(_feat_np.tobytes())
                progress(f"Saved test_input_features.bin: shape={_feat_np.shape}")
            except Exception as _save_err:
                progress(f"WARN: failed to save test_input_features.bin: {_save_err}")
    else:
        progress("Tokenizing prompt")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

    # Save input token IDs as int32 binary. Phase 3E writes to BOTH
    # reference/test_input_ids.bin (cosine-check + --token-ids replay)
    # AND assets/test_input_ids.bin (the deploy_path declared in
    # io_contract.json; pushed to device by deploy_android.sh's
    # assets/* wildcard). Without the assets/ copy, the device side has
    # no canonical phoneme/token sequence and the C++ tokenizer's output
    # can drift from PyTorch's by trailing-space / BOS / whitespace
    # quirks (Kokoro Entry 4: trailing space → T=25 vs ref T=24,
    # time-shifted every downstream layer).
    input_ids_np = input_ids[0].numpy().astype("int32")
    input_ids_path = os.path.join(ref_dir, "test_input_ids.bin")
    with open(input_ids_path, "wb") as f:
        f.write(input_ids_np.tobytes())
    try:
        _assets_dir = os.path.join(output_dir, "assets")
        os.makedirs(_assets_dir, exist_ok=True)
        with open(os.path.join(_assets_dir, "test_input_ids.bin"), "wb") as f:
            f.write(input_ids_np.tobytes())
    except Exception as _e:
        progress(f"WARN: failed to copy test_input_ids.bin to assets/: {_e}")

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

    # Authoritative forward call graph captured in hook-fire order. One
    # node per module-level hook firing. The scaffolder
    # (extensions/cli/src/tools/nnport/scaffoldTs.ts) reads this to emit
    # src/ops/op_<dump_name>.cpp + src/backbone.cpp regardless of modality;
    # the Build phase reads it to refuse on C++ call-sequence divergence.
    # Per Phase 3 redesign: every port consumes forward_graph.json instead
    # of a per-modality template — see Entry 1.8 of
    # ~/.claude/plans/kokoro-port-journal.md.
    forward_graph = []

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
        # Collect ALL containers with >=2 layers for multi-stack models
        # (encoder-decoder: Whisper, T5, SeamlessM4T, BART). These get
        # registered AFTER the single-container block below with prefixed
        # dump names ("decoder_layer_0", "encoder_layer_0", etc.) to keep
        # both stacks in forward_graph.json.
        _additional_containers = []  # list of (prefix, container, attr)
        for _n_layers, _name, _module in _layer_candidates[1:]:
            if _name == layer_container_attr:
                continue
            # Derive a prefix from the path segment before ".layers" /
            # ".layer" / ".h" / ".blocks". For "model.decoder.layers" → "decoder".
            # Used to namespace dump_names so encoder + decoder don't collide.
            import re as _re_aux
            _m = _re_aux.match(r"^(.*?)\.(?:layers|layer|h|blocks)$", _name)
            _prefix_seg = _m.group(1).split(".")[-1] if _m else _name.split(".")[-1]
            _additional_containers.append((_prefix_seg, _module, _name))
            progress(f"Additional layer container found: {_name} ({_n_layers} layers, prefix='{_prefix_seg}')")
    else:
        _additional_containers = []
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

    def _compute_weight_prefix(mod, module_path):
        # Return module_path if this module has own parameters (the prefix
        # every param of this module shares in state_dict), else None for
        # container modules with no direct params. Per-op subagents read
        # this from forward_graph.json so they know which weight keys to
        # load via weights.get_buffer(...) without grepping the full
        # state_dict.
        if mod is None or not module_path:
            return None
        try:
            for _ in mod.named_parameters(recurse=False):
                return module_path
        except Exception:
            return None
        return None

    # Extract a torch.Tensor from arbitrary forward-hook output. HuggingFace
    # models return dataclasses (BaseModelOutputWithPast, CausalLMOutputWithPast,
    # ModelOutput) instead of bare tensors, so the naive isinstance(out, Tensor)
    # check silently drops captures for OpenELMModel / OpenELMForCausalLM /
    # LlamaModel / etc. — the canonical dump_name (e.g. transformer_output.bin)
    # never gets written, PortNode hard-blocks, agent stalls. Generic across
    # any HF model: try a Tensor directly, then tuple[0], then dataclass field
    # priority (last_hidden_state > logits > prediction_scores), then any
    # tensor-valued attribute, then .to_tuple()[0]. Returns None on no tensor.
    def _extract_tensor(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return _extract_tensor(x[0])
        # ModelOutput / dataclass paths.
        for _attr in ("last_hidden_state", "logits", "prediction_scores", "hidden_states", "encoder_last_hidden_state"):
            if hasattr(x, _attr):
                v = getattr(x, _attr)
                if isinstance(v, torch.Tensor):
                    return v
                if isinstance(v, (tuple, list)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    return v[0]
        # Any Tensor field in __dict__ (last-resort, mostly for custom output classes).
        try:
            d = vars(x)
        except Exception:
            d = None
        if isinstance(d, dict):
            for _k, _v in d.items():
                if isinstance(_v, torch.Tensor):
                    return _v
        # HF ModelOutput.to_tuple() — last resort.
        if hasattr(x, "to_tuple"):
            try:
                t = x.to_tuple()
                if isinstance(t, tuple) and len(t) > 0 and isinstance(t[0], torch.Tensor):
                    return t[0]
            except Exception:
                pass
        return None

    def make_hook(sname, reference_only=False):
        def hook_fn(mod, inp, out):
            try:
                # AUTOREGRESSIVE PREFILL-ONLY CAPTURE.
                # model.generate() fires this hook 1 + max_new_tokens times
                # per layer (1 prefill at full seq_len, N decode steps at
                # seq_len=1). Without this guard the saved tensor is the
                # LAST decode-step capture ([1,1,hidden]) which mismatches
                # the C++ binary's pass=0 dump ([1, seq_len, hidden]) by a
                # factor of seq_len — SxS then reports "ratio=seq_len"
                # alignment failure for every layer. First-write-wins keeps
                # the prefill tensor, which is what __pass0 compares against.
                if sname in layer_captures:
                    return
                inp_tensor = _extract_tensor(inp)
                out_tensor = _extract_tensor(out)
                if isinstance(out_tensor, torch.Tensor):
                    # Some modules (root LM-head, embedding pre-hooks) only have
                    # a meaningful output; treat input as best-effort.
                    in_record = (
                        inp_tensor.detach().float().cpu()
                        if isinstance(inp_tensor, torch.Tensor)
                        else out_tensor.detach().float().cpu()
                    )
                    layer_captures[sname] = {
                        "input": in_record,
                        "output": out_tensor.detach().float().cpu(),
                    }
                    capture_order[sname] = _capture_seq[0]
                    _capture_seq[0] += 1
                    try:
                        forward_graph.append({
                            "op": type(mod).__name__ if mod is not None else "Unknown",
                            "dump_name": sname,
                            "module_path": capture_meta.get(sname, {}).get("module_path"),
                            "input_shape": list(in_record.shape),
                            "output_shape": list(out_tensor.shape),
                            "order": _capture_seq[0] - 1,
                            "weight_prefix": _compute_weight_prefix(
                                mod, capture_meta.get(sname, {}).get("module_path")
                            ),
                            "is_pre_hook": False,
                            # Reference-side diagnostic with no runtime counterpart
                            # (e.g. token-only "embedding_wte"): the C++ produces
                            # only the COMBINED embedding, validated under
                            # "embedding". The per-node validator skips such nodes
                            # when no runtime dump exists instead of flagging a
                            # structurally-missing op.
                            "reference_only": reference_only,
                        })
                    except Exception:
                        # forward_graph is diagnostic; never break capture on error.
                        pass
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
        hooks.append(_emb_module.register_forward_hook(make_hook("embedding_wte", reference_only=True)))
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
                    try:
                        forward_graph.append({
                            "op": type(mod).__name__ if mod is not None else "Unknown",
                            "dump_name": sname,
                            "module_path": capture_meta.get(sname, {}).get("module_path"),
                            "input_shape": list(inp_tensor.shape),
                            "output_shape": list(inp_tensor.shape),
                            "order": _capture_seq[0] - 1,
                            "weight_prefix": _compute_weight_prefix(
                                mod, capture_meta.get(sname, {}).get("module_path")
                            ),
                            "is_pre_hook": True,
                        })
                    except Exception:
                        pass
            except Exception:
                pass
        return hook_fn

    # ── Choose the layer stack that the TOKEN EMBEDDING actually feeds ──
    # The combined-embedding pre-hook must sit on block 0 of the stack that
    # CONSUMES the token embeddings, not just the longest/first ModuleList.
    # On encoder-decoder models (Whisper, T5, BART, SeamlessM4T) the encoder
    # and decoder stacks are the SAME length, so the generic "longest container"
    # pick can land on the ENCODER — whose layer-0 input is the audio/vision
    # frontend output ([1,1500,384] for Whisper), totally unrelated to the token
    # embedding. Comparing the C++ decoder embedding against that gives cos≈0 and
    # a phantom first-divergence that no code change can fix. Pick the candidate
    # whose dotted path shares the longest prefix with the token-embedding module
    # path (_emb_name) — that is the decoder stack for enc-dec, and the sole
    # stack for decoder-only models (GPT-2/Llama), so the behavior is unchanged
    # there. Falls back to layer_container if no token embedding was found.
    _emb_feed_container = layer_container
    _emb_feed_attr = layer_container_attr
    if _layer_candidates and _emb_candidates:
        def _dotted_prefix_len(_a, _b):
            _n = 0
            for _x, _y in zip(_a.split("."), _b.split(".")):
                if _x == _y:
                    _n += 1
                else:
                    break
            return _n
        _best = None
        for _n_layers, _cand_name, _cand_mod in _layer_candidates:
            # higher prefix-match with the token embedding wins; tie → more
            # layers; tie → shorter (closer-to-root) path.
            _key = (_dotted_prefix_len(_cand_name, _emb_name), _n_layers, -len(_cand_name))
            if _best is None or _key > _best[0]:
                _best = (_key, _cand_mod, _cand_name)
        if _best is not None:
            _emb_feed_container, _emb_feed_attr = _best[1], _best[2]
            if _emb_feed_attr != layer_container_attr:
                progress(f"Combined-embedding pre-hook retargeted to token-embedding-fed stack: {_emb_feed_attr} (was {layer_container_attr})")

    if _emb_feed_container is not None and len(_emb_feed_container) > 0:
        hooks.append(_emb_feed_container[0].register_forward_pre_hook(make_pre_hook("embedding")))
        # The pre-hook captures the input to block 0 — i.e., the combined
        # embedding regardless of how the model assembled it (wte+wpe for
        # GPT-2, embed_tokens for Llama, decoder embed_tokens for Whisper).
        record_meta(
            "embedding",
            _emb_feed_container[0],
            (f"{_emb_feed_attr}.0" if _emb_feed_attr else "layer_container.0"),
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

    # ── Multi-stack capture for encoder-decoder models ─────────────────────
    # Whisper, T5, BART, SeamlessM4T, Pegasus, MBart, etc. have TWO layer
    # ModuleLists (encoder.layers + decoder.layers) — the single-container
    # block above only hooks one of them. For each ADDITIONAL container,
    # register the same per-layer + per-child hooks with a path-derived
    # prefix on the dump_name (so encoder and decoder don't collide).
    # Without this, forward_graph.json silently drops half the model and
    # backbone.cpp's multi-wrapper loop has nothing to wire on one side.
    for _extra_prefix, _extra_container, _extra_attr in _additional_containers:
        _extra_num = len(_extra_container)
        progress(f"Hooking additional stack '{_extra_prefix}' with {_extra_num} layers (attr={_extra_attr})")
        _extra_path_prefix = (_extra_attr + ".") if _extra_attr else ""
        for _idx2 in range(_extra_num):
            _layer2 = _extra_container[_idx2]
            # Per-layer wrapper hook.
            _dn2 = f"{_extra_prefix}_layer_{_idx2}"
            hooks.append(_layer2.register_forward_hook(make_hook(_dn2)))
            record_meta(_dn2, _layer2, f"{_extra_path_prefix}{_idx2}")
            # Per-direct-child hook (self_attn, encoder_attn, mlp, norms).
            for _cname, _cmod in _layer2.named_children():
                _safe = _cname.replace(".", "_").replace("/", "_")
                if not _safe:
                    continue
                _cdn = f"{_extra_prefix}_{_safe}_{_idx2}"
                hooks.append(_cmod.register_forward_hook(make_hook(_cdn)))
                record_meta(_cdn, _cmod, f"{_extra_path_prefix}{_idx2}.{_cname}")

    # ── Encoder frontend OUTPUT capture (audio enc-dec: Whisper / Seamless) ──
    # The frontend (mel→hidden Conv1D stack on Whisper) sits BEFORE the
    # encoder.layers ModuleList, so the per-layer hooks above never capture it.
    # Then the FIRST reference node is encoder layer 0, and a wrong frontend
    # shows up only as a degraded layer-0 cosine — the agent cannot isolate or
    # validate the frontend and circles a sub-1.0 ceiling forever (the Whisper
    # 2026-06-01 port stalled at cos~0.88 for exactly this reason).
    #
    # Capture the frontend OUTPUT = input to block 0 of each non-token-fed
    # stack, mirroring the decoder "embedding" pre-hook. This is the post-conv,
    # post-gelu, post-pos-embed hidden state — the ONE tensor the C++ frontend
    # produces and the layer stack consumes, so it is directly comparable. (We
    # deliberately do NOT hook conv1/conv2 individually: the reference captures
    # the pre-activation Conv1d output, but the C++ frontend fuses gelu into
    # each conv, so a per-conv compare would mismatch on the gelu and mislead.)
    # Architecture-blind: driven by topology, not family names.
    import re as _re_fe
    _strip_list = _re_fe.compile(r"\.(?:layers|layer|h|blocks)$")
    _stacks_fe = [(layer_container_attr, layer_container)] + [
        (_a, _c) for (_p, _c, _a) in _additional_containers
    ]
    for _sattr, _scont in _stacks_fe:
        if _scont is None or len(_scont) == 0 or _scont is _emb_feed_container:
            continue
        _root = _strip_list.sub("", _sattr) if _sattr else "encoder"
        _fe_out = (_root.replace(".", "_").replace("/", "_") or "encoder") + "_frontend_out"
        if _fe_out in capture_meta:
            continue
        hooks.append(_scont[0].register_forward_pre_hook(make_pre_hook(_fe_out)))
        record_meta(_fe_out, _scont[0], _root, is_pre_hook=True)
        progress(f"Hooked encoder-frontend-output pre-hook for stack '{_sattr}' as {_fe_out}")

    # ── Graph-mode capture: hook EVERY node in .nnport/graph.json ───────────
    # Architecture-agnostic: every model that's been graph-extracted has a
    # graph.json with primitive-level node definitions. We walk it and
    # register a forward hook for each node whose module_path resolves on
    # the live model. This replaces the per-architecture hand-patching the
    # agent has had to do (BLOOM's transformer.h.<i>.self_attention.query_key_value
    # hooks added by hand were the canonical example).
    #
    # Dump names match PortNode's canonical-name convention: node.id with
    # '.' and '/' → '_'. So a node like 'transformer.h.5.self_attention.query_key_value'
    # produces 'transformer_h_5_self_attention_query_key_value_output.bin',
    # which PortNode's existence check at portNode.ts::referenceDumpPaths
    # finds without any _capture_meta walk.
    #
    # The block-level layer_i / mlp_i / attn_i hooks above remain (they're
    # cheap, and SxS bisection at the BloomBlock granularity is still useful
    # for triage). The graph.json walk adds the per-primitive captures
    # PortNode actually needs for SxS during the per-class port loop.
    import json as _json
    _graph_json_path = os.path.join(output_dir, ".nnport", "graph.json")
    _graph_hooked = 0
    _graph_skipped_unreachable = 0
    _graph_skipped_already_hooked = 0
    # Track which dump_names we've registered in THIS pass so we don't
    # double-hook the same primitive (registering the same hook twice causes
    # two .bin writes per forward pass — second wins, harmless but wasteful).
    _already_hooked_dump_names = set()
    if os.path.exists(_graph_json_path):
        try:
            with open(_graph_json_path, "r") as _gf:
                _graph = _json.load(_gf)
            _graph_nodes = _graph.get("nodes", []) if isinstance(_graph, dict) else []
            progress(f"Graph mode: walking {len(_graph_nodes)} graph.json nodes for additive hook registration")
            for _n in _graph_nodes:
                if not isinstance(_n, dict):
                    continue
                _mp = _n.get("module_path") or _n.get("id")
                if not isinstance(_mp, str) or not _mp:
                    continue
                # Skip the root model node (module_path == "" or matches
                # top-level class). These aren't capturable as hooks.
                if _mp in ("", "model"):
                    continue
                # Sanitize dump_name with the SAME rule PortNode applies to
                # node.id: '.' and '/' → '_'. PortNode's referenceDumpPaths
                # helper looks for "<dump_name>_output.bin" — must match.
                _node_id = _n.get("id") or _mp
                _dump_name = _node_id.replace(".", "_").replace("/", "_")
                if not _dump_name:
                    continue
                if _dump_name in _already_hooked_dump_names:
                    _graph_skipped_already_hooked += 1
                    continue
                # Attribute-walk to resolve model.<module_path>. Handles
                # numeric segments (ModuleList indexing, e.g. transformer.h.5
                # resolves transformer → ModuleList → [5]).
                _resolved = model
                _ok = True
                for _part in _mp.split("."):
                    try:
                        if _part.isdigit():
                            _resolved = _resolved[int(_part)]
                        else:
                            _resolved = getattr(_resolved, _part)
                    except (AttributeError, IndexError, TypeError):
                        _ok = False
                        break
                if not _ok or _resolved is None:
                    _graph_skipped_unreachable += 1
                    continue
                # Skip if the resolved object isn't an nn.Module — composite
                # nodes (BloomModel, BloomForCausalLM, BloomBlock) ARE
                # nn.Modules, so they DO get hooked. Dropout modules also get
                # hooked but produce identity-passthrough dumps; harmless.
                try:
                    _is_mod = hasattr(_resolved, "register_forward_hook")
                except Exception:
                    _is_mod = False
                if not _is_mod:
                    _graph_skipped_unreachable += 1
                    continue
                try:
                    hooks.append(_resolved.register_forward_hook(make_hook(_dump_name)))
                    record_meta(_dump_name, _resolved, _mp)
                    _already_hooked_dump_names.add(_dump_name)
                    _graph_hooked += 1
                except Exception as _ge:
                    _graph_skipped_unreachable += 1
            progress(
                f"Graph-mode hooks: {_graph_hooked} registered, "
                f"{_graph_skipped_already_hooked} skipped (already hooked by block-level pass), "
                f"{_graph_skipped_unreachable} skipped (module_path unreachable or non-module)"
            )
        except Exception as _ge:
            progress(f"Graph-mode hook walk failed: {_ge} — falling back to block-level + block0-primitives only")

    # ── Per-stage oracle capture: hook each Port-IR stage's module_path ──────
    # (NNOPT_PORT_ENGINE_DESIGN_PROMPT §8 / FIX_PLAN fix b.) DesignPort writes
    # .nnport/port_ir.json — a typed DAG of the model's STAGES (encoder, decoder
    # step, vocoder, ...). For every multi-stage / non-HF model the block-level
    # and graph-mode passes above can miss the exact stage boundaries the C++
    # port is decomposed into, so StageTest has no per-stage oracle to compare
    # against and the agent loops blind on an end-to-end-only score. We hook the
    # LIVE eager module (the same one the loader already returned — non-HF is NOT
    # non-hookable) at each stage's module_path and let the existing flush write
    # reference/layers/<stageId>_{input,output}.bin — the exact files StageTest
    # and Evaluate's auto-bisect read. PURELY ADDITIVE: these hooks attach to the
    # same model alongside every pass above and write under NEW stage-id keys;
    # nothing here changes existing capture. Best-effort throughout — a missing,
    # partial, or unresolvable IR never breaks reference generation (an
    # un-refined skeleton whose module_path is still a bare stage name simply
    # resolves to nothing and is skipped).
    _port_ir_path = os.path.join(output_dir, ".nnport", "port_ir.json")
    if os.path.exists(_port_ir_path):
        try:
            with open(_port_ir_path, "r") as _irf:
                _ir = json.load(_irf)
            _ir_stages = _ir.get("stages", []) if isinstance(_ir, dict) else []
            _ir_hooked = 0
            _ir_skipped = 0
            for _stage in _ir_stages:
                try:
                    _sid = _stage.get("id")
                    _mpath = _stage.get("module_path")
                    if not _sid or not _mpath:
                        _ir_skipped += 1
                        continue
                    # Already captured under this exact name by a pass above.
                    if _sid in capture_meta:
                        continue
                    # Resolve the stage's module on the live eager model. Prefer
                    # get_submodule (dotted path); fall back to a getattr/index
                    # walk that tolerates numeric ModuleList indices.
                    _smod = None
                    try:
                        _smod = model.get_submodule(_mpath)
                    except Exception:
                        _obj = model
                        for _part in str(_mpath).split("."):
                            if not _part:
                                continue
                            if _part.isdigit() and hasattr(_obj, "__getitem__"):
                                try:
                                    _obj = _obj[int(_part)]
                                except Exception:
                                    _obj = None
                            else:
                                _obj = getattr(_obj, _part, None)
                            if _obj is None:
                                break
                        _smod = _obj
                    if _smod is None or not hasattr(_smod, "register_forward_hook"):
                        _ir_skipped += 1
                        continue
                    hooks.append(_smod.register_forward_hook(make_hook(_sid)))
                    record_meta(_sid, _smod, _mpath)
                    _ir_hooked += 1
                except Exception:
                    _ir_skipped += 1
            progress(
                f"Port-IR stage oracle: hooked {_ir_hooked} stage module(s), "
                f"skipped {_ir_skipped} (unresolved module_path or duplicate)"
            )
        except Exception as _ire:
            progress(f"Port-IR stage-oracle hook walk failed (non-fatal): {_ire}")

    # ── Intermediate-local capture (sys.settrace per target class.forward) ──
    # PyTorch forward hooks only fire at nn.Module boundaries. When the agent
    # needs to bisect INSIDE a single module's forward (e.g. MambaMixer's
    # dt_raw / B / C / scan_out are tensor locals, not submodule outputs),
    # we monkey-patch the target class's forward to install a frame-local
    # tracer that captures named locals at function-return.
    #
    # Driver: .nnport/layer_contracts/<Class>.json's intermediate_dumps[].
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
            #
            # OPT-IN ONLY (NNOPT_INTERMEDIATE_DUMPS=1). User evidence: ports
            # converged BETTER without sub-op dumps. The auto-emitted
            # block0_sub_* names cost ~80-160 manual NNOPT_LAYER_CHECK calls
            # per port to instrument, generate name-collision bugs in SxS
            # ("embedding 8960→5376" in yesterday's trace was actually a
            # collision against block0_sub_attn_q_norm_input), and produce
            # the noisy "missing producer instrumentation" diagnostic that
            # the agent thrashes against. Explicit intermediate_dumps from
            # RequestIntermediateDumps still work either way — they populate
            # intermediate_dump_specs above.
            _auto_intermediates_enabled = os.environ.get("NNOPT_INTERMEDIATE_DUMPS", "").strip() in ("1", "true", "yes", "on")
            if _auto_intermediates_enabled and _pyt_cls and _cat in _CATEGORY_DUMP_PREFIX and _cat not in ("lm_head", "embedding"):
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

        # ── Per-instance submodule hooks for contract classes ─────────────
        # Robust per-layer captures that don't depend on sys.settrace (which
        # only fires for the FIRST instance of each class — block 0 only —
        # and breaks for torch.compile'd code, fragile to forward-source
        # changes, can't see local variables that only exist transiently).
        #
        # PyTorch forward hooks always fire on __call__, capture exact I/O,
        # work for any architecture, and crucially yield SEPARATE captures
        # for EVERY instance — so models with variable per-layer dims (e.g.
        # OpenELM, where num_query_heads / num_kv_heads / head_dim differ
        # by block index) get per-layer reference tensors with the correct
        # per-layer shapes. Block-0-only sys.settrace captures cannot match
        # block-5's tensor shapes, which is the exact failure mode that
        # was making OpenELM unportable.
        #
        # Naming is deterministic from {category, layer_idx, submodule_name}:
        #   block{i}_{prefix_tail}_input              ← whole-instance input
        #   block{i}_{prefix_tail}_output             ← whole-instance output
        #   block{i}_{prefix_tail}_{submod}_input     ← per-submodule input
        #   block{i}_{prefix_tail}_{submod}_output    ← per-submodule output
        # where prefix_tail is e.g. "sub_attn" for category="attention".
        # The C++ scaffold's NNOPT_LAYER_CHECK calls already use this
        # convention; the references now line up by name + layer index.
        _per_instance_hooked = 0
        _per_instance_summary = {}
        for _cn, _cls_obj in _class_objs_by_name.items():
            _auto_info = _target_class_auto.get(_cls_obj)
            _category = _auto_info["category"] if _auto_info else None
            if not _category:
                # Fall back: look up category from the contract for this class.
                # _by_class entries are spec dicts; we re-read the contract category
                # from disk indirectly via _CATEGORY_DUMP_PREFIX which is already
                # populated by category. If still unset, skip — without category
                # we can't form the dump prefix.
                pass
            _full_prefix = _CATEGORY_DUMP_PREFIX.get(_category) if _category else None
            if not _full_prefix:
                continue
            # "block0_sub_attn" → "sub_attn" (we re-prefix per layer index)
            _bare_tail = _full_prefix[len("block0_"):] if _full_prefix.startswith("block0_") else _full_prefix
            _instance_count = 0
            for _mod_name, _mod in model.named_modules():
                if not isinstance(_mod, _cls_obj):
                    continue
                # Layer index = first integer-only path component (HF convention:
                # "transformer.layers.5.attn", "model.layers.5.mlp", etc.).
                _layer_idx = None
                for _part in _mod_name.split("."):
                    if _part.isdigit():
                        _layer_idx = int(_part)
                        break
                if _layer_idx is None:
                    continue
                # Whole-instance input/output. Captures the boundary that the
                # C++ port crosses when it calls Attention::forward() / MLP::forward().
                _inst_in = f"block{_layer_idx}_{_bare_tail}_input"
                _inst_out = f"block{_layer_idx}_{_bare_tail}_output"
                hooks.append(_mod.register_forward_pre_hook(make_pre_hook(_inst_in)))
                hooks.append(_mod.register_forward_hook(make_hook(_inst_out)))
                record_meta(_inst_in, _mod, _mod_name, is_pre_hook=True)
                record_meta(_inst_out, _mod, _mod_name)
                _per_instance_hooked += 2
                # Direct children — these are the qkv_proj / out_proj / q_norm /
                # k_norm / pos_embedding for attention, gate_proj / up_proj /
                # down_proj for MLP, etc. Defined statically in __init__, named
                # by Python attribute name. No filtering by primitive type:
                # every child is a structurally-meaningful boundary.
                for _child_name, _child_mod in _mod.named_children():
                    _safe = _child_name.replace(".", "_").replace("/", "_")
                    if not _safe:
                        continue
                    _child_in = f"block{_layer_idx}_{_bare_tail}_{_safe}_input"
                    _child_out = f"block{_layer_idx}_{_bare_tail}_{_safe}_output"
                    hooks.append(_child_mod.register_forward_pre_hook(make_pre_hook(_child_in)))
                    hooks.append(_child_mod.register_forward_hook(make_hook(_child_out)))
                    record_meta(_child_in, _child_mod, f"{_mod_name}.{_child_name}", is_pre_hook=True)
                    record_meta(_child_out, _child_mod, f"{_mod_name}.{_child_name}")
                    _per_instance_hooked += 2
                _instance_count += 1
            if _instance_count > 0:
                _per_instance_summary[_cn] = _instance_count
        if _per_instance_summary:
            progress(
                f"Hooked {_per_instance_hooked} per-instance submodule captures across "
                f"classes {_per_instance_summary}. These give per-layer reference tensors "
                f"so SxS works for models with variable per-layer dims (OpenELM-style)."
            )

        # ── Graph-aware coverage pass (Phase C4) ────────────────────────
        # PortNode hard-blocks if reference/layers/<dump_name>_output.bin
        # is missing for the requested node. Architecture-aware hooks above
        # cover the common cases (Attention/MLP per layer, embedding, lm_head,
        # final_norm) but custom nodes — root model class, RoPE module, KV
        # cache wrappers — are easily missed. To guarantee the agent can
        # always iterate, we read graph.json (if it exists) and register a
        # forward hook for EVERY node whose dump_name is not already covered.
        # This is mechanical: the hook just dumps the post-hook output to
        # <dump_name>_output.bin. No source interpretation, no regex.
        try:
            _graph_path = os.path.join(output_dir, ".nnport", "graph.json")
            if os.path.exists(_graph_path):
                with open(_graph_path, "r") as _gf:
                    _graph = json.load(_gf)
                _existing_dump_names = set()
                # Collect dump_names already targeted by the architecture-
                # aware passes above. We can't introspect record_meta directly
                # without growing API; instead we conservatively re-register
                # — register_forward_hook is idempotent in effect (multiple
                # hooks each dump, last writer wins on the same path), and
                # the file write is the same target.
                _named = dict(model.named_modules())
                _graph_hooked = 0
                for _node in (_graph.get("nodes") or []):
                    _nid = _node.get("id")
                    _dname = _node.get("dump_name")
                    if not _nid or not _dname:
                        continue
                    # Skip the "__root__" sentinel — model-level output is
                    # already captured implicitly by lm_head + final_norm
                    # paths and re-running model() at the root is redundant.
                    if _nid == "__root__":
                        continue
                    _mod = _named.get(_nid)
                    if _mod is None:
                        continue
                    if _dname in _existing_dump_names:
                        continue
                    # Pass _dname as the sname; the writer downstream appends
                    # _input.bin / _output.bin. Doubling the suffix here was
                    # the cause of block0_sub_attn_norm_output_output.bin.
                    hooks.append(_mod.register_forward_hook(make_hook(_dname)))
                    record_meta(_dname, _mod, _nid)
                    _existing_dump_names.add(_dname)
                    _graph_hooked += 1
                if _graph_hooked > 0:
                    progress(
                        f"Hooked {_graph_hooked} additional graph-node captures from .nnport/graph.json "
                        f"(Phase C4 coverage). Every PortNode-requestable node now has a reference dump."
                    )
        except Exception as _e:
            progress(f"graph-aware coverage pass failed (non-fatal): {_e}")

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
            if getattr(getattr(model, "config", None), "is_encoder_decoder", False):
                # ANY encoder-decoder model — architecture-blind. Covers Whisper
                # (audio-in → text), MusicGen / Bark (text-in → audio tokens),
                # T5 / BART (text-in → text). A bare model(input_ids) forward
                # crashes on all of these because the DECODER half receives no
                # decoder_input_ids and transformers calls ones_like(None).
                # Generic fix: feed the encoder input under the model's OWN
                # main_input_name, and supply a one-step decoder start token.
                _dec_start = getattr(model.config, "decoder_start_token_id", None)
                if _dec_start is None:
                    _dec_start = getattr(getattr(model, "generation_config", None), "decoder_start_token_id", None)
                if _dec_start is None:
                    _dec_start = getattr(model.config, "bos_token_id", None)
                if _dec_start is None:
                    _dec_start = getattr(model.config, "pad_token_id", 0) or 0
                # Some decoders consume one row PER codebook (MusicGen's audio
                # LM): decoder_input_ids is (batch*num_codebooks, 1), not (1, 1).
                # Read num_codebooks generically from the (possibly nested)
                # config; None for ordinary single-stream decoders.
                _num_cb = (
                    getattr(getattr(model.config, "decoder", None), "num_codebooks", None)
                    or getattr(model.config, "num_codebooks", None)
                )
                if _num_cb:
                    _dec_input = torch.full((int(_num_cb), 1), int(_dec_start), dtype=torch.long)
                else:
                    _dec_input = torch.tensor([[int(_dec_start)]])
                _main_name = getattr(model, "main_input_name", "input_ids")
                if _main_name == "input_features" and input_features is not None:
                    outputs = model(input_features=input_features, decoder_input_ids=_dec_input)
                else:
                    outputs = model(input_ids=input_ids, decoder_input_ids=_dec_input)
            else:
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
    manifest["_nnport_capture_version"] = "2026-06-25-local-source-path-no-strip-v9"
    manifest["_layer_container_attr"] = layer_container_attr or "<not_found>"

    manifest_path = os.path.join(layers_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    progress(f"Captured {len(layer_captures)} layer tensors")

    # ── forward_graph.json — authoritative call graph ─────────────────────
    # One node per module-level hook firing, in execution order. The
    # scaffolder (extensions/cli/src/tools/nnport/scaffoldTs.ts) reads
    # this to emit src/ops/op_<dump_name>.cpp + src/backbone.cpp; the
    # Build phase reads it to refuse on C++ call-sequence divergence.
    # Per Phase 3 redesign: every port consumes this artifact regardless
    # of modality. Path is reference/forward_graph.json — alongside the
    # reference/ root, NOT inside layers/.
    forward_graph_path = os.path.join(os.path.dirname(layers_dir), "forward_graph.json")
    # Dedupe by module_path. The reference template registers hooks at
    # multiple levels (per-layer wrapper, per-named-child, per-block-0
    # primitive) so the same nn.Module instance fires multiple hooks per
    # forward call, producing N forward_graph entries for ONE actual
    # computation. backbone.cpp consumers (scaffolder, divergence gate)
    # need ONE call per real module. Keep the first hook to fire for each
    # module_path — that's the layer-children naming, which is the most
    # uniform across all 30 transformer layers. The other dump_names'
    # reference .bin files remain on disk for SxS sub-op debugging.
    seen_module_paths = set()
    deduped_forward_graph = []
    for _node in forward_graph:
        _mp = _node.get("module_path")
        # If no module_path (host helpers), fall back to dump_name as key.
        _key = _mp if _mp else _node.get("dump_name")
        if not _key:
            continue
        if _key in seen_module_paths:
            continue
        seen_module_paths.add(_key)
        deduped_forward_graph.append(_node)
    # Re-stamp execution order on the deduped list so backbone.cpp sees
    # contiguous 0..N-1 indices.
    for _i, _node in enumerate(deduped_forward_graph):
        _node["order"] = _i
    try:
        with open(forward_graph_path, "w") as _fg:
            json.dump({
                "version": 1,
                "model_class": type(model).__name__,
                "nodes": deduped_forward_graph,
                "_nnport_capture_version": "2026-06-25-local-source-path-no-strip-v9",
                "_raw_node_count_before_dedup": len(forward_graph),
            }, _fg, indent=2)
        progress(
            f"Wrote forward_graph.json with {len(deduped_forward_graph)} unique-module nodes "
            f"(deduped from {len(forward_graph)} hook firings)"
        )
    except Exception as _e:
        progress(f"WARNING: failed to write forward_graph.json: {_e}")

    # ── io_contract.json — runtime I/O + fixture + seed contract ─────────
    # Phase 3E: scaffolder reads this to thread sample_rate_hz into
    # main.cpp, derive deploy.sh fixture list, and seed any RNG. Replaces
    # hardcoded 16 kHz / hardcoded duration_noise+prior_noise fixtures
    # that broke Kokoro (deterministic, 24 kHz). See Kokoro Entries 1.5
    # findings #2, #3, #11 for the failure mode this retires.
    #
    # Architecture-blind: sample_rate_hz comes from model.config when the
    # model advertises one (sampling_rate / sample_rate / audio_sample_rate);
    # null otherwise (text LM). input_fixtures lists every .bin asset the
    # runtime needs on-device — for a text LM that's just test_input_ids.
    _io_contract_sample_rate = None
    try:
        _cfg_obj = getattr(model, "config", None)
        for _sr_attr in ("sampling_rate", "sample_rate", "audio_sample_rate"):
            _sr_val = getattr(_cfg_obj, _sr_attr, None) if _cfg_obj is not None else None
            if isinstance(_sr_val, (int, float)) and _sr_val > 0:
                _io_contract_sample_rate = int(_sr_val)
                break
    except Exception:
        _io_contract_sample_rate = None

    _io_contract_path = os.path.join(os.path.dirname(layers_dir), "io_contract.json")
    # Build input_fixtures list. For audio encoder-decoder models we emit a
    # second fixture entry for the mel features so main.cpp knows to load
    # assets/test_input_features.bin (analogous to how it loads
    # test_input_ids.bin via the contract). Without this entry, the agent
    # has no signal that the .bin file exists and ends up zero-filling the
    # input_features buffer — the encoder then runs on silence and the
    # decoder sampler collapses to id=0 regardless of how correct the
    # generated math is.
    _ic_fixtures = [
        {
            "name": "test_input_ids",
            "shape": [int(_d) for _d in input_ids.shape],
            "dtype": "int32",
            "source": "audio model: forced decoder prefix (decoder_start_token_id + forced_decoder_ids)" if _is_audio_model else "tokenizer(prompt, return_tensors='pt')['input_ids']",
            "deploy_path": "assets/test_input_ids.bin",
        },
    ]
    if _is_audio_model and input_features is not None:
        try:
            _feat_shape = [int(_d) for _d in input_features.shape]
        except Exception:
            _feat_shape = list(getattr(input_features, "shape", []))
        _ic_fixtures.append({
            "name": "test_input_features",
            "shape": _feat_shape,
            "dtype": "float32",
            "source": "AutoProcessor(librispeech_sample, padding='max_length').input_features",
            "deploy_path": "assets/test_input_features.bin",
            "consumer": "encoder",
            "note": "MUST be loaded by main.cpp and passed via ForwardDispatch::set_input_features. Do NOT zero-fill; the deployed binary at the deploy_path contains the real mel spectrogram.",
        })
    try:
        _io_contract = {
            "version": 1,
            "model_class": type(model).__name__,
            "sample_rate_hz": _io_contract_sample_rate,
            "input_fixtures": _ic_fixtures,
            "output_artifact": {
                "kind": "text",
                "sample_rate_hz": _io_contract_sample_rate,
            },
            "rng_seeds": {
                # text-LM reference is deterministic given input_ids; no
                # seed-dependent path. Modalities that DO need seeds
                # (mms_tts duration_noise/prior_noise) write their own
                # io_contract.json with the real seeds.
                "deterministic": True,
            },
            "exact_input_sequence": input_ids_np.tolist(),
            "_nnport_capture_version": "2026-06-25-local-source-path-no-strip-v9",
        }
        with open(_io_contract_path, "w") as _ic:
            json.dump(_io_contract, _ic, indent=2)
        progress(
            f"Wrote io_contract.json (sample_rate_hz="
            f"{_io_contract_sample_rate}, "
            f"input_ids={len(_io_contract['exact_input_sequence'])} tokens)",
        )
    except Exception as _e:
        progress(f"WARNING: failed to write io_contract.json: {_e}")

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
    # Audio encoder-decoder models (Whisper, SeamlessM4T, etc.) cannot run
    # this decode-capture loop with the text-style positional `input_ids`
    # invocation below — Whisper interprets the first positional arg as
    # `input_features` (mel spec) and raises ValueError when it's not
    # shape (n_mels, 3000). Per-decode-step captures for audio models are
    # a follow-up; first port validates against the prefill hooks only.
    if _is_audio_model:
        decode_passes_target = 0
        progress("Skipping decode-pass capture for audio encoder-decoder model (first-port: prefill captures only)")
    try:
        if _is_audio_model:
            # Skip the entire decode-capture seed+loop for audio models —
            # the prefill model(input_ids, use_cache=True) below would feed
            # text token ids into a model that expects mel input_features
            # and ValueError out before reaching the loop.
            raise RuntimeError("decode capture skipped for audio encoder-decoder model")
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
    _is_enc_dec = getattr(getattr(model, "config", None), "is_encoder_decoder", False)
    _main_name = getattr(model, "main_input_name", "input_ids")
    if _is_enc_dec:
        # ANY encoder-decoder model — architecture-blind. A text-LM
        # `model(input_ids).logits` decode loop is WRONG here: the decoder gets
        # no decoder_input_ids, so Whisper (audio-in) misreads input_ids as a mel
        # and MusicGen/Bark (text-in -> audio) crash inside their codec
        # ('NoneType' object has no attribute 'shape' in EnCodec). model.generate()
        # runs the encoder once + the correct decoder loop + decoder_start for
        # every architecture. Feed the encoder input under the model's OWN
        # main_input_name (input_features for audio-in, input_ids for text-in).
        with torch.no_grad():
            if _main_name == "input_features" and input_features is not None:
                _gen_out = model.generate(input_features=input_features, max_new_tokens=int(max_new))
            else:
                _gen_out = model.generate(input_ids=input_ids, max_new_tokens=int(max_new))
        if hasattr(_gen_out, "sequences"):
            _gen_seq = _gen_out.sequences[0]
        else:
            _gen_seq = _gen_out[0]
        # Output is text tokens (Whisper/T5) — integer dtype — OR a decoded audio
        # WAVEFORM (MusicGen/Bark .generate() returns float audio_values, NOT
        # token ids). Int-casting a float waveform truncates every sample in
        # [-1,1] to 0 and silently destroys the end-to-end audio reference.
        # Route on dtype instead: float → save reference/output.wav (the
        # contract's output_artifact) + layers/waveform_output.bin (Evaluate's
        # audio amplitude/cosine gate input); integer → token-id path.
        if getattr(_gen_seq, "dtype", None) is not None and _gen_seq.dtype.is_floating_point:
            _wf = _gen_seq.detach().float().reshape(-1)
            _sr = (
                getattr(getattr(getattr(model, "config", None), "audio_encoder", None), "sampling_rate", None)
                or getattr(getattr(model, "config", None), "sampling_rate", None)
                or 32000
            )
            try:
                import wave as _wave
                _pcm = (_wf.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy().tobytes()
                _wav_path = os.path.join(ref_dir, "output.wav")
                with _wave.open(_wav_path, "wb") as _w:
                    _w.setnchannels(1)
                    _w.setsampwidth(2)
                    _w.setframerate(int(_sr))
                    _w.writeframes(_pcm)
                progress(f"Reference WAV: wrote {_wav_path} ({_wf.numel()} samples @ {int(_sr)}Hz)")
            except Exception as _wav_err:
                progress(f"Reference WAV write failed: {_wav_err}")
            try:
                _wf.numpy().astype("float32").tofile(os.path.join(layers_dir, "waveform_output.bin"))
                progress(f"Captured waveform_output.bin ({_wf.numel()} float32 samples)")
            except Exception as _wfd_err:
                progress(f"waveform_output.bin dump failed: {_wfd_err}")
            generated_ids = []
            reference_text = ""
            generated_only = ""
            progress(f"Generated via model.generate (encoder-decoder, main_input={_main_name}): float waveform, {_wf.numel()} sample(s)")
        else:
            try:
                generated_ids = [int(t) for t in _gen_seq.reshape(-1).tolist()]
            except Exception:
                generated_ids = []
            try:
                reference_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            except Exception:
                reference_text = ""
            generated_only = reference_text
            progress(f"Generated via model.generate (encoder-decoder, main_input={_main_name}): {len(generated_ids)} value(s)")
    else:
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
    # produced_by + version stamps let the TS cache layer distinguish
    # legit-script-produced files from agent-fabricated stubs. The TS
    # user-provided cache path requires produced_by to short-circuit;
    # without it the script re-runs.
    _tx_ver_pt = "unknown"
    _torch_ver_pt = "unknown"
    try:
        import transformers as _tx_mod_pt
        _tx_ver_pt = getattr(_tx_mod_pt, "__version__", "unknown")
    except Exception:
        pass
    try:
        import torch as _torch_mod_pt
        _torch_ver_pt = getattr(_torch_mod_pt, "__version__", "unknown")
    except Exception:
        pass
    ref_data = {
        "produced_by": "_run_reference.py",
        "script_version": "2026-06-25-local-source-path-no-strip-v9",
        "transformers_version": _tx_ver_pt,
        "pytorch_version": _torch_ver_pt,
        "python_version": sys.version.split()[0] if hasattr(sys, "version") else "unknown",
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

    # Audio/codec-output models (MusicGen, Bark, ...) emit EnCodec/codec
    # codebook tokens, NOT text. The text-LM gibberish heuristics below
    # (tail-window collapse, unique-token ratio, control-char decode) are
    # MEANINGLESS on codec tokens: the codebook delay-pattern legitimately
    # front-pads each stream with a repeated pad id, so a perfectly valid
    # audio reference looks like a "collapsed repetition loop" and the gate
    # exits 1. Per the port contract (I2_REFERENCE_COHERENCE) audio reference
    # validity is verified per-dump at Evaluate time, not by this pre-flight
    # text gate. Detect audio-output architecture-blind and skip the gate.
    _gib_cfg = getattr(model, "config", None)
    _output_is_audio = (
        getattr(_gib_cfg, "model_type", "") in ("musicgen", "musicgen_melody", "bark")
        or hasattr(model, "audio_encoder")
        or getattr(getattr(_gib_cfg, "decoder", None), "num_codebooks", None) is not None
        or getattr(_gib_cfg, "num_codebooks", None) is not None
    )
    if _output_is_audio:
        progress(
            "Audio/codec-output model — skipping text-LM gibberish gate "
            "(codec delay-pattern tokens are not text; reference validity is "
            "verified per-dump at Evaluate time)."
        )

    if N_gen >= 4 and not _output_is_audio:
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
    if not is_gibberish and generated_only and not _output_is_audio:
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
