#!/usr/bin/env python3
# NNOPT ONNX reference runner (docs/onnx-support/03_DESIGN.md §2).
#
# Loads the port's .onnx graph, SEEDS every in-graph RNG node (fresh session
# per run = byte-equal reruns — proven on the StyleTTS2/iSTFTNet harmonic
# noise nodes), exposes EVERY node output as a graph output, runs ONE
# deterministic pass on CPUExecutionProvider with graph optimization
# DISABLED, and dumps golden tensors in nnopt's contract:
#
#   reference/reference_tokens.json    (produced_by stamp gates the TS cache)
#   reference/reference_text.txt       (the input text, for audio ports)
#   reference/layers/<module_path>_output.bin       (fp32, C-order)
#   reference/layers/layer_0_{input,output}.bin     (cache-validator pair)
#   reference/layers/manifest.json     (_captured_layers, _capture_meta, flat
#                                       primary entries the dump_spec
#                                       regenerator walks)
#   reference/forward_graph.json       (module_path-keyed, topological order)
#   reference/io_contract.json         (sample_rate_hz, input fixtures, seeds)
#   reference/output.wav + layers/waveform_output.bin   (float terminal out)
#   assets/test_input_ids.bin          (int32 pinned ids — runtime input)
#   assets/rng_<name>.bin              (each seeded RNG node's output — the
#                                       C++ loads these, NEVER device RNG)
#
# Node-name identity: onnx_module_path() below is the VERBATIM Python mirror
# of src/tools/nnport/onnxNameMap.ts::onnxModulePath — the runner, the weight
# converter, and the TS pairing side must agree byte-for-byte on every name
# (ledger F21). Change one, change all three.
#
# Spawn contract (set by generateReferenceTs):
#   argv = [script, model_id, output_dir, prompt, max_new, source_path, tokenizer_repo]
import sys, os, json, re, subprocess, traceback, warnings

# Keep stderr clean for the TS failure-class matcher: per-dump stats on
# empty/degenerate tensors emit numpy RuntimeWarnings that are pure noise.
warnings.filterwarnings("ignore", category=RuntimeWarning)

REFERENCE_SCRIPT_VERSION = "2026-07-02-nonhf-stage-input-capture-v10"
ONNX_OP_CLASS_TABLE = json.loads('''{"primary":["MatMul","Gemm","MatMulInteger","Conv","ConvInteger","ConvTranspose","LSTM","GRU","LayerNormalization","InstanceNormalization","BatchNormalization","GroupNormalization","Softmax","Tanh","Sigmoid","Relu","LeakyRelu","PRelu","Elu","Gelu","HardSigmoid","HardSwish","Softplus","Clip","Sin","Cos","Exp","Log","Sqrt","Atan","Erf","Abs","Neg","Reciprocal","Floor","Ceil","Round","Mul","Add","Div","Sub","Pow","Min","Max","Sum","ReduceMean","ReduceSum","ReduceMax","ReduceMin","ReduceProd","ReduceL2","Gather","Resize","Upsample","RandomNormalLike","RandomUniformLike","RandomNormal","RandomUniform","Multinomial"],"inspection":["Shape","Cast","CastLike","Reshape","Unsqueeze","Squeeze","Transpose","Concat","Split","Slice","Range","ConstantOfShape","Constant","Expand","Where","Equal","Greater","GreaterOrEqual","Less","LessOrEqual","Not","And","Or","Xor","Pad","TopK","CumSum","ScatterND","ScatterElements","GatherElements","GatherND","Identity","Tile","Flatten","ArgMax","ArgMin","NonZero","Trilu","OneHot","Size"],"decompose":["DynamicQuantizeLinear","DequantizeLinear","QuantizeLinear","DynamicQuantizeLSTM"],"control_flow":["If","Loop","Scan","SequenceEmpty","SequenceInsert","SequenceAt","SequenceLength","SequenceConstruct","SplitToSequence","ConcatFromSequence"]}''')
RNG_SEED = 12345.0
DUMP_BUDGET_BYTES = int(os.environ.get("NNOPT_ONNX_DUMP_BUDGET", str(4 * 1024**3)))


def progress(msg):
    print(msg, flush=True)


def fail(failure_class, error, tb=None):
    detail = {
        "success": False,
        "failure_class": failure_class,
        "error": str(error),
        "stderr_tail": (tb or traceback.format_exc())[-2000:],
    }
    print(str(error), file=sys.stderr)
    print(json.dumps(detail))


# ── The Q3 name mapping — VERBATIM mirror of onnxNameMap.ts (F21) ──────────
def onnx_module_path(value_name, node_index=0):
    s = value_name or ""
    if s.startswith("/"):
        s = s[1:]
    s = s.replace("/", ".")
    s = re.sub(r"[^0-9A-Za-z_.]", "_", s)
    s = re.sub(r"[._]{2,}", lambda m: "." if "." in m.group(0) else "_", s)
    s = re.sub(r"^[._]+|[._]+$", "", s)
    return s or ("unnamed_%d" % node_index)


def canonical_dump_name(module_path):
    # Mirror of dumpNaming.ts::canonicalDumpName — dots/slashes → underscores.
    return re.sub(r"[./]", "_", module_path)


def classify_op(op_type, out_dtype):
    """tensor_kind + coverage class from the TS-rendered op table (§3.3).
    Integer/bool outputs are inspection plumbing regardless of op."""
    if op_type in ONNX_OP_CLASS_TABLE.get("control_flow", []):
        return "control_flow"
    if op_type in ONNX_OP_CLASS_TABLE.get("decompose", []):
        return "decompose"
    if op_type in ONNX_OP_CLASS_TABLE.get("inspection", []):
        return "inspection"
    if op_type in ONNX_OP_CLASS_TABLE.get("primary", []):
        if out_dtype is not None and ("int" in str(out_dtype) or "bool" in str(out_dtype)):
            return "inspection"
        return "primary"
    return "unsupported"


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def write_wav_int16(path, samples, sample_rate):
    import numpy as np
    import wave
    pcm = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm16.tobytes())


def phonemize_text(text, tok):
    """Text → phoneme string. In-venv phonemizer first, espeak-ng CLI
    subprocess fallback (both proven; the arm64 espeakng-loader wheel is
    broken on some hosts, so NEVER assume the in-process library loads)."""
    lang = tok.get("language") or "en-us"
    errs = []
    try:
        from phonemizer import phonemize
        out = phonemize(
            text,
            language=lang,
            backend="espeak",
            with_stress=bool(tok.get("with_stress", True)),
            preserve_punctuation=bool(tok.get("preserve_punctuation", True)),
        )
        if out and out.strip():
            return out.strip(), None
        errs.append("phonemizer returned empty output")
    except Exception as e:
        errs.append("phonemizer: %s: %s" % (type(e).__name__, str(e)[:200]))
    for exe in ("espeak-ng", "espeak"):
        try:
            r = subprocess.run(
                [exe, "--ipa", "-q", "-v", lang, text],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode == 0 and r.stdout.strip():
                return " ".join(r.stdout.split()), None
            errs.append("%s exited %d: %s" % (exe, r.returncode, (r.stderr or "")[:200]))
        except FileNotFoundError:
            errs.append("%s not on PATH" % exe)
        except Exception as e:
            errs.append("%s: %s" % (exe, str(e)[:200]))
    remediation = (
        "espeak phonemization unavailable. Fix ONE of: "
        "(a) pip install phonemizer-fork espeakng-loader into the onnx venv "
        "(the espeakng-loader wheel is broken on some arm64 hosts); "
        "(b) install the espeak-ng CLI (brew install espeak-ng / apt install espeak-ng) "
        "matching the host arch so the CLI fallback works."
    )
    return None, "; ".join(errs) + ". " + remediation


def text_to_ids(text, usage, vocab):
    """Text → framed id list via the §7.1 chain (phoneme_vocab.json symbols)."""
    tok = (usage.get("tokenization") or {})
    kind = tok.get("kind") or (vocab.get("phonemizer", {}) or {}).get("kind") or "unknown"
    symbols = vocab.get("symbols") or tok.get("symbol_table")
    if not symbols:
        return None, "no_symbols", (
            "No phoneme/symbol table available: .nnport/phoneme_vocab.json has no "
            "'symbols' and onnx_usage.json has no tokenization.symbol_table. "
            "Run AnalyzeModel's usage discovery, or supply the vocab via "
            "PortTokenizer --tokenizer-vocab."
        )
    if kind == "espeak_ipa":
        phonemes, err = phonemize_text(text, {**tok, **(vocab.get("phonemizer") or {})})
        if phonemes is None:
            return None, "espeak_unavailable", err
        chars = phonemes
    elif kind == "chars":
        chars = text
    else:
        return None, "onnx_run_failed", (
            "tokenization.kind=%r is not runnable by the ONNX reference runner "
            "(supported: espeak_ipa, chars)." % kind
        )
    sym2id = {s: i for i, s in enumerate(symbols)}
    ids = [sym2id[c] for c in chars if c in sym2id]
    if not ids:
        return None, "onnx_run_failed", (
            "Tokenization produced 0 ids — none of the %d phonemized characters "
            "appear in the %d-symbol table." % (len(chars), len(symbols))
        )
    framing = vocab.get("framing") or tok.get("bos_eos_framing") or [0, 0]
    return [int(framing[0])] + ids + [int(framing[1])], None, None


def main():
    model_id    = sys.argv[1] if len(sys.argv) > 1 else ""
    output_dir  = sys.argv[2] if len(sys.argv) > 2 else "."
    prompt      = sys.argv[3] if len(sys.argv) > 3 else ""

    ref_dir = os.path.join(output_dir, "reference")
    layers_dir = os.path.join(ref_dir, "layers")
    assets_dir = os.path.join(output_dir, "assets")
    nnport_dir = os.path.join(output_dir, ".nnport")
    for d in (layers_dir, assets_dir, nnport_dir):
        os.makedirs(d, exist_ok=True)

    try:
        import numpy as np
    except Exception as e:
        return fail("missing_dependency", "import numpy failed: %s" % e)
    try:
        import onnx
        from onnx import helper
    except Exception as e:
        return fail("onnx_import_missing", "import onnx failed: %s" % e)
    try:
        import onnxruntime as ort
    except Exception as e:
        return fail("onnxruntime_import_missing", "import onnxruntime failed: %s" % e)

    # ── Resolve the graph (AnalyzeModel's summary is the pointer) ──────────
    summary = read_json(os.path.join(nnport_dir, "onnx_graph_summary.json"), {})
    usage   = read_json(os.path.join(nnport_dir, "onnx_usage.json"), {})
    vocab   = read_json(os.path.join(nnport_dir, "phoneme_vocab.json"), {})
    model_file = summary.get("model_file") or ""
    if not model_file or not os.path.exists(model_file):
        return fail("onnx_model_file_missing",
                    "onnx_graph_summary.json names model_file=%r which does not exist. "
                    "Run AnalyzeModel first (it downloads the .onnx and writes the summary)." % model_file)

    try:
        model = onnx.load(model_file)  # external data loads automatically when sidecars exist
    except Exception as e:
        msg = str(e)
        if "external data" in msg.lower() or ".onnx_data" in msg or ".data" in msg:
            return fail("onnx_external_data_missing",
                        "onnx.load failed on external data — the .onnx references a sidecar "
                        "tensor file that is not next to it: %s" % msg[:400])
        return fail("onnx_run_failed", "onnx.load failed: %s" % msg[:400])

    graph = model.graph
    rng_ops = {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike", "Multinomial"}

    # ── Determinize: seed every RNG node (F22) ─────────────────────────────
    rng_nodes_seeded = {}
    for node in graph.node:
        if node.op_type in rng_ops:
            for i in range(len(node.attribute) - 1, -1, -1):
                if node.attribute[i].name == "seed":
                    del node.attribute[i]
            node.attribute.append(helper.make_attribute("seed", RNG_SEED))
            rng_nodes_seeded[node.output[0]] = RNG_SEED
    if rng_nodes_seeded:
        progress("[onnx] seeded %d RNG node(s) with seed=%s" % (len(rng_nodes_seeded), RNG_SEED))
    seeded_bytes = model.SerializeToString()

    # ── Name map + per-node metadata (topological order per ONNX spec) ─────
    producers = {}          # value name → (node_index, node)
    for idx, node in enumerate(graph.node):
        for out in node.output:
            if out:
                producers[out] = (idx, node)

    init_names = {i.name for i in graph.initializer}
    graph_input_names = [i.name for i in graph.input if i.name not in init_names]
    terminal_names = [o.name for o in graph.output]

    def value_dtype(name):
        for vi in list(graph.value_info) + list(graph.output) + list(graph.input):
            if vi.name == name and vi.type.HasField("tensor_type"):
                return onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type).lower()
        return None

    # Every node's FIRST output is its dump identity; extra outputs dump too.
    all_value_names, seen = [], set()
    for node in graph.node:
        for out in node.output:
            if out and out not in seen:
                seen.add(out)
                all_value_names.append(out)
    for t in terminal_names:
        if t not in seen:
            seen.add(t)
            all_value_names.append(t)

    name_map, taken = {}, {}
    for i, vn in enumerate(all_value_names):
        base = onnx_module_path(vn, i)
        n = taken.get(base, 0)
        mp = base if n == 0 else "%s__c%d" % (base, n + 1)
        if n > 0:
            progress("[onnx] WARNING: value-name collision: %r → %r" % (vn, mp))
        taken[base] = n + 1
        name_map[vn] = mp

    # Persist the value-name map so later tools re-derive identity without
    # re-parsing the model (merged with the converter's initializer section).
    nm_path = os.path.join(nnport_dir, "onnx_name_map.json")
    nm = read_json(nm_path, {}) or {}
    nm["values"] = name_map
    nm["rng_nodes_seeded"] = {name_map.get(k, k): v for k, v in rng_nodes_seeded.items()}
    with open(nm_path, "w") as f:
        json.dump(nm, f, indent=2)

    # ── Expose every node output as a graph output ─────────────────────────
    existing_outputs = set(terminal_names)
    added = 0
    for vn in all_value_names:
        if vn not in existing_outputs:
            graph.output.append(helper.make_empty_tensor_value_info(vn))
            added += 1
    patched_path = os.path.join(ref_dir, "_model_all_outputs.onnx")
    onnx.save(model, patched_path)
    progress("[onnx] exposed %d intermediate outputs (total fetched: %d)" % (added, len(graph.output)))

    # ── Build feeds from the io contract (§4.3 roles) ───────────────────────
    input_roles = usage.get("input_roles") or {}
    scalar_defaults = usage.get("scalar_defaults") or {}
    aux_assets = usage.get("aux_assets") or {}

    sess_probe = None
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_probe = ort.InferenceSession(seeded_bytes, so, providers=["CPUExecutionProvider"])
    except Exception as e:
        return fail("onnx_run_failed", "InferenceSession(original) failed: %s" % str(e)[:500])
    sess_inputs = {i.name: i for i in sess_probe.get_inputs()}

    ids = None
    feeds = {}
    fixtures_declared = []
    for name, info in sess_inputs.items():
        role = input_roles.get(name, "")
        np_dtype = {"tensor(int64)": "int64", "tensor(int32)": "int32",
                    "tensor(float)": "float32", "tensor(double)": "float64",
                    "tensor(bool)": "bool"}.get(info.type, "float32")
        if role == "token_ids":
            got, fclass, err = text_to_ids(prompt, usage, vocab)
            if got is None:
                return fail(fclass, err)
            ids = got
            feeds[name] = np.asarray([ids], dtype=np_dtype)
            progress("[onnx] input %r ← %d token ids (%s)" % (name, len(ids), np_dtype))
        elif role.endswith("_scalar") or role == "rate_scalar":
            val = float(scalar_defaults.get(name, 1.0))
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in (info.shape or [1])]
            feeds[name] = np.full(shape, val, dtype=np_dtype)
            progress("[onnx] input %r ← scalar %s" % (name, val))
        else:
            asset = None
            for _file, meta in aux_assets.items():
                if (meta or {}).get("feeds_input") == name:
                    asset = meta
                    break
            if asset is None or not asset.get("default"):
                return fail("onnx_input_unresolved",
                            "Graph input %r (type %s, role %r) has no resolvable source: "
                            "not token ids, not a scalar, and no aux asset in "
                            ".nnport/onnx_usage.json declares feeds_input=%r." % (name, info.type, role, name))
            fix_rel = asset["default"]
            fix_path = os.path.join(output_dir, fix_rel)
            if not os.path.exists(fix_path):
                return fail("onnx_input_unresolved",
                            "Aux fixture %s (feeds input %r) is missing on disk — "
                            "AnalyzeModel's aux-asset conversion did not produce it." % (fix_rel, name))
            arr = np.fromfile(fix_path, dtype=np.float32)
            shape = asset.get("shape") or [d if isinstance(d, int) and d > 0 else -1 for d in (info.shape or [])]
            try:
                arr = arr.reshape(shape)
            except Exception:
                pass
            feeds[name] = arr.astype(np_dtype)
            fixtures_declared.append({
                "name": name, "shape": list(arr.shape), "dtype": str(arr.dtype),
                "source": fix_rel, "deploy_path": fix_rel, "role": role or "aux",
            })
            progress("[onnx] input %r ← fixture %s %s" % (name, fix_rel, list(arr.shape)))

    if ids is None:
        return fail("onnx_input_unresolved",
                    "No graph input carries role 'token_ids' in .nnport/onnx_usage.json "
                    "input_roles (%r) — the runner cannot feed the prompt." % (input_roles,))

    # Publish the pinned ids (int32 — the runtime fixture format).
    np.asarray(ids, dtype=np.int32).tofile(os.path.join(assets_dir, "test_input_ids.bin"))

    # ── Run: fresh seeded session, all outputs, opt disabled (one pass) ────
    fetch_names = [o.name for o in graph.output]

    def fresh_run(names):
        so2 = ort.SessionOptions()
        so2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(patched_path, so2, providers=["CPUExecutionProvider"])
        return sess.run(names, feeds)

    # Dump-budget estimate from static shapes; unknown-shaped values are
    # uncounted (the budget is a guard, not an exact accountant).
    est = 0
    try:
        inferred = onnx.shape_inference.infer_shapes(onnx.load(patched_path))
        for vi in list(inferred.graph.value_info) + list(inferred.graph.output):
            tt = vi.type.tensor_type
            if tt.elem_type and all(d.HasField("dim_value") for d in tt.shape.dim) and len(tt.shape.dim) > 0:
                n = 1
                for d in tt.shape.dim:
                    n *= d.dim_value
                est += n * 4
    except Exception:
        pass

    fetched = {}
    if est > DUMP_BUDGET_BYTES:
        chunks = max(2, (est // DUMP_BUDGET_BYTES) + 1)
        size = (len(fetch_names) + chunks - 1) // chunks
        progress("[onnx] estimated dumps %.1f MB > budget — %d chunked fresh-session runs"
                 % (est / 1e6, chunks))
        for c in range(0, len(fetch_names), size):
            names = fetch_names[c:c + size]
            try:
                vals = fresh_run(names)
            except Exception as e:
                return fail("onnx_run_failed", "chunked run failed: %s" % str(e)[:500])
            fetched.update(dict(zip(names, vals)))
    else:
        progress("[onnx] single all-outputs pass (%d fetches)..." % len(fetch_names))
        try:
            vals = fresh_run(fetch_names)
        except Exception as e:
            return fail("onnx_run_failed", "all-outputs run failed: %s" % str(e)[:500])
        fetched = dict(zip(fetch_names, vals))

    # ── e2e sanity (adjusted from design §2.1.8 — empirical correction) ────
    # The design asked for byte-equality between the dumped (patched,
    # DISABLE_ALL) run and the unpatched default-opt run. The proving model
    # refutes that TWICE over: (a) optimization shifts duration rounding in
    # dynamic-length TTS graphs (different waveform LENGTH), and (b) even at
    # the SAME opt level, ORT's RNG draw depends on graph scheduling, so a
    # patched and unpatched graph draw different noise (observed 100800 vs
    # 102600 samples, both self-deterministic). What IS provable — and what
    # F22 actually requires — is that the DUMPED graph variant reruns
    # byte-equal in a fresh session: the C++ replays the dumped run's RNG
    # fixtures, so self-consistency of that exact variant is the invariant.
    #
    # HARD gate: fresh-session rerun of the PATCHED model, terminals must be
    # byte-equal (catches unseeded/nondeterministic RNG). When the graph has
    # NO RNG nodes, the unpatched same-opt-level comparison is also hard
    # (exposing intermediates must not change deterministic math).
    try:
        rerun_out = fresh_run(terminal_names)
        for tname, oval in zip(terminal_names, rerun_out):
            a = np.asarray(fetched[tname])
            b = np.asarray(oval)
            if a.shape != b.shape or a.tobytes() != b.tobytes():
                return fail("onnx_opt_level_divergence",
                            "Terminal output %r is NOT deterministic across fresh sessions of "
                            "the dumped model — an RNG source is unseeded (shapes %s vs %s). "
                            "The reference cannot be replayed; refusing to publish it."
                            % (tname, a.shape, b.shape))
        progress("[onnx] determinism check passed (fresh-session rerun byte-equal)")
    except SystemExit:
        raise
    except Exception as e:
        return fail("onnx_run_failed", "determinism rerun failed: %s" % str(e)[:400])

    # DIAGNOSTIC (recorded in the manifest, never fails for RNG-bearing
    # graphs): unpatched same-opt-level + default-opt comparisons.
    opt_divergence = {"checked": False}
    try:
        per_out = {}
        for label, so_level in (("unpatched_disable_all", "disable"), ("unpatched_default_opt", "default")):
            so_x = ort.SessionOptions()
            if so_level == "disable":
                so_x.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            sess_x = ort.InferenceSession(seeded_bytes, so_x, providers=["CPUExecutionProvider"])
            outs = sess_x.run(terminal_names, feeds)
            for tname, oval in zip(terminal_names, outs):
                a = np.asarray(fetched[tname])
                b = np.asarray(oval)
                rec = {"shape_dumped": list(a.shape), "shape_other": list(b.shape),
                       "byte_equal": bool(a.shape == b.shape and a.tobytes() == b.tobytes())}
                if a.shape == b.shape and a.dtype.kind == "f":
                    d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                    rec["max_abs_delta"] = d if np.isfinite(d) else None
                per_out.setdefault(label, {})[tname] = rec
                if not rec["byte_equal"]:
                    if not rng_nodes_seeded and label == "unpatched_disable_all":
                        return fail("onnx_opt_level_divergence",
                                    "Terminal %r differs between the patched and unpatched graph at "
                                    "the same opt level on an RNG-FREE model — exposing intermediates "
                                    "changed deterministic math; the dumped reference is invalid."
                                    % tname)
                    progress("[onnx] NOTE: %s divergence on %r (%s vs %s) — recorded; the "
                             "reference stays pinned to the dumped variant"
                             % (label, tname, list(a.shape), list(b.shape)))
        opt_divergence = {"checked": True, "outputs": per_out}
    except SystemExit:
        raise
    except Exception as e:
        opt_divergence = {"checked": False, "error": str(e)[:300]}

    # ── Dump + manifest emission (§audit-B byte-compatible) ────────────────
    def stats(a):
        try:
            fa = a.astype(np.float64)
            vals = {"mean": float(np.mean(fa)), "std": float(np.std(fa)),
                    "min": float(np.min(fa)), "max": float(np.max(fa))}
            return {k: (v if np.isfinite(v) else None) for k, v in vals.items()}
        except Exception:
            return {"mean": None, "std": None, "min": None, "max": None}

    manifest = {}
    inspection_stats = {}  # nested under "layers" — NEVER top-level: a real ONNX
    #                        mp can itself end in "_output", and the TS dump_spec
    #                        regenerator scans top-level *_output keys (observed:
    #                        206 of kitten's inspection values would leak in as
    #                        phantom primary entries).
    flat_entries = {}
    capture_meta = {}
    captured_layers = []
    capture_order = []
    skipped_non_tensor = []
    first_primary_mp = None

    for i, vn in enumerate(all_value_names):
        if vn not in fetched:
            continue
        v = fetched[vn]
        mp = name_map[vn]
        if not hasattr(v, "shape"):  # sequence/map-typed output (SplitToSequence etc.)
            skipped_non_tensor.append(mp)
            continue
        arr = np.asarray(v)
        node_idx, node = producers.get(vn, (None, None))
        op_type = node.op_type if node is not None else "GraphOutput"
        kind = classify_op(op_type, str(arr.dtype))
        is_terminal = vn in terminal_names
        if op_type in rng_ops:
            # RNG fixture: the C++ replays this exact noise (F22).
            arr.astype(np.float32).reshape(-1).tofile(
                os.path.join(assets_dir, "rng_%s.bin" % canonical_dump_name(mp)))
        tensor_kind = ("activation" if (kind == "primary" or is_terminal)
                       else ("ids" if arr.dtype.kind in "iub" else "inspection"))
        f32 = arr.astype(np.float32).reshape(-1)
        f32.tofile(os.path.join(layers_dir, "%s_output.bin" % mp))
        entry = {"shape": list(arr.shape), "numel": int(arr.size),
                 "tensor_kind": tensor_kind}
        entry.update(stats(arr))
        parent = None
        if node is not None and node.input:
            pi = producers.get(node.input[0])
            parent = pi[1].op_type if pi else None
        capture_meta[mp] = {
            "module_path": mp,
            "module_class": op_type,
            "parent_class": parent,
            "qualname_in_parent": mp.rsplit(".", 1)[-1],
            "is_pre_hook": False,
        }
        capture_order.append(mp)
        if kind == "primary" or is_terminal:
            captured_layers.append(mp)
            flat_entries["%s_output" % mp] = entry
            if first_primary_mp is None:
                first_primary_mp = mp
        else:
            inspection_stats[mp] = entry  # fetchable for SxS inspection

    # Cache-validator pair: graph input as capture index 0.
    np.asarray(ids, dtype=np.float32).tofile(os.path.join(layers_dir, "layer_0_input.bin"))
    if first_primary_mp is not None:
        f0 = np.asarray(fetched[[k for k, v in name_map.items() if v == first_primary_mp][0]])
        f0.astype(np.float32).reshape(-1).tofile(os.path.join(layers_dir, "layer_0_output.bin"))

    # ── Audio routing: float terminal → wav + waveform_output.bin ──────────
    sample_rate = usage.get("sample_rate_hz") or 24000
    e2e_written = False
    for tname in terminal_names:
        arr = np.asarray(fetched[tname])
        if arr.dtype.kind == "f" and arr.size > 16:
            arr.astype(np.float32).reshape(-1).tofile(os.path.join(layers_dir, "waveform_output.bin"))
            write_wav_int16(os.path.join(ref_dir, "output.wav"), arr, sample_rate)
            e2e_written = True
            progress("[onnx] terminal %r → output.wav (%d samples @ %d Hz) + waveform_output.bin"
                     % (tname, arr.size, sample_rate))
            break

    manifest.update({
        "layers": inspection_stats,
        "_captured_layers": captured_layers,
        "_capture_order": capture_order,
        "_capture_meta": capture_meta,
        "_layer_map": {mp: canonical_dump_name(mp) for mp in captured_layers},
        "_nnport_capture_version": REFERENCE_SCRIPT_VERSION,
        "_framework": "onnx",
        "_instrumentation_kind": "onnx-node-outputs",
        "_rng_nodes_seeded": {name_map.get(k, k): v for k, v in rng_nodes_seeded.items()},
        "_skipped_non_tensor": skipped_non_tensor,
        "_opt_level_divergence": opt_divergence,
    })
    manifest.update(flat_entries)
    with open(os.path.join(layers_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ── forward_graph.json — the GraphProto IS the graph (§3.2/§6.1) ───────
    fg_nodes, seen_mp = [], set()
    for i, node in enumerate(graph.node):
        vn = next((o for o in node.output if o), None)
        if vn is None:
            continue
        mp = name_map.get(vn)
        if mp is None or mp in seen_mp:
            continue
        seen_mp.add(mp)
        prefixes = [onnx_module_path(x) for x in node.input if x in init_names]
        weight_prefix = ""
        if prefixes:
            parts = prefixes[0].split(".")
            for p in prefixes[1:]:
                op = p.split(".")
                k = 0
                while k < min(len(parts), len(op)) and parts[k] == op[k]:
                    k += 1
                parts = parts[:k]
            weight_prefix = ".".join(parts)
        out_dtype = None
        if vn in fetched and hasattr(fetched[vn], "shape"):
            out_dtype = str(np.asarray(fetched[vn]).dtype)
        kind = classify_op(node.op_type, out_dtype)
        fg_nodes.append({
            "op": node.op_type,
            "dump_name": mp,
            "module_path": mp,
            "order": i,
            "weight_prefix": weight_prefix,
            "tensor_kind": "activation" if kind == "primary" else kind,
            "reference_only": kind != "primary",
            "output_shape": (list(np.asarray(fetched[vn]).shape)
                             if vn in fetched and hasattr(fetched[vn], "shape") else None),
        })
    forward_graph = {
        "version": 1,
        "model_class": summary.get("producer") or "OnnxGraph",
        "nodes": fg_nodes,
        "_nnport_capture_version": REFERENCE_SCRIPT_VERSION,
        "_raw_node_count_before_dedup": len(graph.node),
        "_framework": "onnx",
    }
    with open(os.path.join(ref_dir, "forward_graph.json"), "w") as f:
        json.dump(forward_graph, f, indent=2)
    progress("[onnx] wrote forward_graph.json (%d nodes, %d primary)"
             % (len(fg_nodes), len(captured_layers)))

    # ── io_contract.json ────────────────────────────────────────────────────
    fixtures = [{"name": "input_ids", "shape": [1, len(ids)], "dtype": "int32",
                 "source": "assets/test_input_ids.bin",
                 "deploy_path": "assets/test_input_ids.bin", "role": "token_ids"}]
    fixtures += fixtures_declared
    for vn in rng_nodes_seeded:
        mp = name_map.get(vn, onnx_module_path(vn))
        if vn in fetched and hasattr(fetched[vn], "shape"):
            a = np.asarray(fetched[vn])
            fixtures.append({
                "name": "rng_%s" % canonical_dump_name(mp),
                "shape": list(a.shape), "dtype": "float32",
                "source": "assets/rng_%s.bin" % canonical_dump_name(mp),
                "deploy_path": "assets/rng_%s.bin" % canonical_dump_name(mp),
                "role": "rng",
            })
    io_contract = {
        "version": 1,
        "model_class": summary.get("producer") or "OnnxGraph",
        "sample_rate_hz": int(sample_rate) if e2e_written else None,
        "input_fixtures": fixtures,
        "output_artifact": {"kind": "wav" if e2e_written else "tensor",
                            "sample_rate_hz": int(sample_rate) if e2e_written else None},
        "rng_seeds": {"deterministic": True,
                      "onnx_node_seeds": {name_map.get(k, k): v for k, v in rng_nodes_seeded.items()}},
        "exact_input_sequence": [int(x) for x in ids],
    }
    with open(os.path.join(ref_dir, "io_contract.json"), "w") as f:
        json.dump(io_contract, f, indent=2)

    # ── reference_tokens.json + reference_text.txt ──────────────────────────
    ref_tokens = {
        "produced_by": "_run_reference.py",
        "script_version": REFERENCE_SCRIPT_VERSION,
        "framework": "onnx",
        "onnxruntime_version": ort.__version__,
        "onnx_version": onnx.__version__,
        "python_version": sys.version.split()[0],
        "model_id": model_id,
        "prompt": prompt,
        "input_ids": [int(x) for x in ids],
        "generated_ids": [],
        "reference_text": prompt,
        "generated_text": "",
        "instrumentation_kind": "onnx-node-outputs",
    }
    with open(os.path.join(ref_dir, "reference_tokens.json"), "w") as f:
        json.dump(ref_tokens, f, indent=2)
    with open(os.path.join(ref_dir, "reference_text.txt"), "w") as f:
        f.write(prompt)

    progress("[onnx] reference complete: %d dumps (%d primary), %d RNG fixtures, e2e=%s"
             % (len(capture_order), len(captured_layers), len(rng_nodes_seeded),
                "wav" if e2e_written else "none"))
    print(json.dumps({
        "success": True,
        "framework": "onnx",
        "instrumentation_kind": "onnx-node-outputs",
        "num_layers_captured": len(captured_layers),
        "num_dumps_total": len(capture_order),
        "captured_layers": captured_layers[:50],
        "rng_nodes_seeded": len(rng_nodes_seeded),
        "e2e_output_written": e2e_written,
        "reference_text": prompt,
        "input_ids_count": len(ids),
    }))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        fail("onnx_runner_crash", e)
