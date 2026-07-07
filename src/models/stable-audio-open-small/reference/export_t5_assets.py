#!/usr/bin/env python3
# T5-on-device groundwork — exports everything the C++ T5 encoder port needs:
#
#   1. PROBE: empirically diff conditioner output vs bare T5 encoder output so
#      the C++ implementation replicates the exact wrapper math (projection?
#      masking? how the 65-token cross_attn_cond is assembled) without reading
#      stable_audio_tools source.
#   2. WEIGHTS: T5 encoder + seconds-embedder state_dicts -> packed fp16 bin +
#      offset manifest (weights/t5_encoder.fp16.bin + .meta.json, same style as
#      the main model pack).
#   3. TOKENIZER: SentencePiece pieces + scores -> weights/t5_tokenizer.bin
#      (flat: count, then per piece [u16 len][bytes][f32 score]) for a device
#      side unigram/viterbi tokenizer.
#   4. REFERENCE: per-block hidden-state dumps for the smoke-test prompt ->
#      reference/t5_layers/*.bin so the C++ port validates layer-by-layer.

import os, sys, json, struct
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'stable-audio-tools')
sys.path.insert(0, os.path.abspath(SRC))
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
WDIR = os.path.join(ROOT, "weights")
RDIR = os.path.join(ROOT, "reference", "t5_layers")
os.makedirs(RDIR, exist_ok=True)

MODEL_ID = "stabilityai/stable-audio-open-small"
PROMPT = "solo acoustic guitar fingerpicking, folk melody"
SECONDS_TOTAL = 11
MAX_LEN = 64

def log(m):
    print(f"PROGRESS: {m}", flush=True)

def to_np(t):
    return t.detach().cpu().float().numpy()

def main():
    from stable_audio_tools import get_pretrained_model
    log(f"Loading model: {MODEL_ID}")
    model, cfg = get_pretrained_model(MODEL_ID)
    model = model.to("cpu").eval().requires_grad_(False)

    cond_prompt = model.conditioner.conditioners["prompt"]
    cond_secs = model.conditioner.conditioners["seconds_total"]

    # The T5 encoder is a PLAIN attribute (.model), deliberately unregistered
    # as a submodule — which is exactly why it never entered the checkpoint.
    t5 = getattr(cond_prompt, "model", None)
    if t5 is None or "T5" not in t5.__class__.__name__:
        for name, mod in cond_prompt.named_modules():
            if mod.__class__.__name__ in ("T5EncoderModel", "T5Model"):
                t5 = mod
                break
    assert t5 is not None and "T5" in t5.__class__.__name__, \
        f"no T5 module found (got {type(t5).__name__})"
    log(f"T5 encoder: {t5.__class__.__name__}, proj_out={cond_prompt.proj_out.__class__.__name__}")
    tok = getattr(cond_prompt, "tokenizer", None)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("t5-base", model_max_length=MAX_LEN)
    log(f"tokenizer: {tok.__class__.__name__}, vocab={tok.vocab_size}")

    # ---------- 1. PROBE ----------
    enc = tok(PROMPT, truncation=True, max_length=MAX_LEN, padding="max_length",
              return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    log(f"token ids ({int(mask.sum())} real of {ids.shape[1]}): {ids[0].tolist()}")

    with torch.no_grad():
        raw = t5(input_ids=ids, attention_mask=mask).last_hidden_state  # [1,64,768]
        cond = model.conditioner([{"prompt": PROMPT, "seconds_total": SECONDS_TOTAL}],
                                 device="cpu")
        di = model.get_conditioning_inputs(cond)
    cross = di["cross_attn_cond"]           # [1,65,768]
    glob = di["global_cond"]

    probe = {"prompt": PROMPT, "seconds_total": SECONDS_TOTAL,
             "input_ids": ids[0].tolist(), "n_real_tokens": int(mask.sum()),
             "raw_t5_shape": list(raw.shape), "cross_shape": list(cross.shape),
             "global_shape": list(glob.shape)}

    # Does cross[:, :64] equal raw T5 output (identity proj), a masked version,
    # or a projected version?
    a, b = cross[0, :MAX_LEN], raw[0]
    probe["cross_vs_raw_cos"] = float(torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0))
    probe["cross_vs_raw_maxabs"] = float((a - b).abs().max())
    masked = raw[0] * mask[0].unsqueeze(-1)
    probe["cross_vs_rawmasked_maxabs"] = float((a - masked).abs().max())
    pad_rows = a[mask[0] == 0]
    probe["cross_pad_rows_maxabs"] = float(pad_rows.abs().max()) if pad_rows.numel() else None

    # Is token 65 the seconds embedding, and is global_cond the same vector?
    secs_out = cond["seconds_total"]
    probe["seconds_cond_shapes"] = [list(t.shape) for t in secs_out]
    sec_vec = secs_out[0].reshape(-1)[:768]
    probe["tok65_vs_seconds_maxabs"] = float((cross[0, MAX_LEN] - sec_vec).abs().max())
    probe["global_vs_seconds_maxabs"] = float((glob.reshape(-1) - sec_vec).abs().max())

    # Any extra learned modules in the conditioners beyond t5 itself?
    probe["prompt_conditioner_params"] = {
        k: list(v.shape) for k, v in cond_prompt.state_dict().items()
        if not k.startswith(("model.", "t5."))} or "none-outside-t5"
    probe["seconds_conditioner_params"] = {
        k: list(v.shape) for k, v in cond_secs.state_dict().items()}

    with open(os.path.join(ROOT, "reference", "t5_probe.json"), "w") as f:
        json.dump(probe, f, indent=2)
    log(f"PROBE: cross[:64] vs raw-T5 maxabs={probe['cross_vs_raw_maxabs']:.3e} "
        f"masked-maxabs={probe['cross_vs_rawmasked_maxabs']:.3e} "
        f"pad-rows-maxabs={probe['cross_pad_rows_maxabs']}")
    log(f"PROBE: tok65-vs-seconds={probe['tok65_vs_seconds_maxabs']:.3e} "
        f"global-vs-seconds={probe['global_vs_seconds_maxabs']:.3e}")

    # ---------- 2. WEIGHTS ----------
    def pack(state, path_bin, path_meta, note):
        # Same meta schema as model.fp16.meta.json so src/weights.cpp's
        # parser loads this pack unchanged.
        meta, off = {"model_id": note, "format": "packed", "dtype": "float16",
                     "tensors": {}}, 0
        with open(path_bin, "wb") as f:
            for k, v in state.items():
                arr = v.detach().cpu().to(torch.float16).numpy()
                f.write(arr.tobytes())
                meta["tensors"][k] = {"offset": off, "shape": list(v.shape),
                                      "dtype": "float16",
                                      "num_elements": int(arr.size),
                                      "size_bytes": int(arr.nbytes)}
                off += arr.nbytes
        meta["total_bytes"] = off
        with open(path_meta, "w") as f:
            json.dump(meta, f, indent=2)
        log(f"packed {len(state)} tensors, {off/1e6:.1f} MB -> {os.path.basename(path_bin)}")

    pack(t5.state_dict(),
         os.path.join(WDIR, "t5_encoder.fp16.bin"),
         os.path.join(WDIR, "t5_encoder.fp16.meta.json"),
         "t5-base encoder (stable-audio prompt conditioner)")
    pack(cond_secs.state_dict(),
         os.path.join(WDIR, "seconds_embedder.fp16.bin"),
         os.path.join(WDIR, "seconds_embedder.meta.json"),
         "seconds_total NumberConditioner")

    # Seconds-embedding lookup table: the number conditioner only ever sees
    # an integer duration, so precompute all 257 embeddings (min_val=0,
    # max_val=256) in fp32. Device does a table lookup — the tiny MLP never
    # needs porting and the result is bit-identical to the reference.
    rows = []
    with torch.no_grad():
        for s in range(257):
            out = cond_secs([s], "cpu")
            rows.append(to_np(out[0]).reshape(-1)[:768])
    tab = np.stack(rows).astype(np.float32)          # [257, 768]
    tab.tofile(os.path.join(WDIR, "seconds_table.bin"))
    chk = float(np.abs(tab[SECONDS_TOTAL] - to_np(sec_vec)).max())
    log(f"seconds_table.bin: {tab.shape} fp32 ({tab.nbytes/1e3:.0f} KB), "
        f"row[{SECONDS_TOTAL}] vs probe maxabs={chk:.3e}")

    # ---------- 3. TOKENIZER ----------
    # SentencePiece unigram vocab (piece strings + log-prob scores) extracted
    # from the fast tokenizer's JSON backend — same table sp_model would give,
    # no sentencepiece dependency. Device side runs metaspace + viterbi.
    backend = json.loads(tok._tokenizer.to_str())
    tmodel = backend["model"]
    assert tmodel["type"] == "Unigram", f"expected Unigram, got {tmodel['type']}"
    vocab = tmodel["vocab"]  # [[piece, score], ...] indexed by id
    pad_id, eos_id, unk_id = tok.pad_token_id, tok.eos_token_id, tmodel["unk_id"]
    with open(os.path.join(WDIR, "t5_tokenizer.bin"), "wb") as f:
        f.write(struct.pack("<iiii", len(vocab), pad_id, eos_id, unk_id))
        for piece, score in vocab:
            pb = piece.encode("utf-8")
            f.write(struct.pack("<Hf", len(pb), float(score)))
            f.write(pb)
    log(f"tokenizer: {len(vocab)} pieces -> t5_tokenizer.bin "
        f"(pad={pad_id} eos={eos_id} unk={unk_id})")

    # ---------- 4. REFERENCE DUMPS ----------
    dumps = {}
    def hook(name):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            dumps[name] = to_np(t)
        return fn
    hs = []
    for name, mod in t5.named_modules():
        cls = mod.__class__.__name__
        if cls == "T5Block":
            hs.append(mod.register_forward_hook(hook(f"{name.replace('.', '_')}_out")))
        if name.endswith("final_layer_norm") and "block" not in name:
            hs.append(mod.register_forward_hook(hook("encoder_final_layer_norm_out")))
        if cls in ("Embedding",) and name in ("shared", "encoder.embed_tokens"):
            hs.append(mod.register_forward_hook(hook("embed_tokens_out")))
    with torch.no_grad():
        t5(input_ids=ids, attention_mask=mask)
    for h in hs:
        h.remove()
    for k, v in dumps.items():
        v.tofile(os.path.join(RDIR, f"{k}.bin"))
    np.array(ids[0], dtype=np.int32).tofile(os.path.join(RDIR, "input_ids.bin"))
    to_np(cross).tofile(os.path.join(RDIR, "cross_attn_cond_target.bin"))
    to_np(glob).tofile(os.path.join(RDIR, "global_embed_target.bin"))
    with open(os.path.join(RDIR, "manifest.json"), "w") as f:
        json.dump({"prompt": PROMPT, "max_length": MAX_LEN,
                   "dumps": {k: list(v.shape) for k, v in dumps.items()}}, f, indent=2)
    log(f"reference: {len(dumps)} layer dumps -> reference/t5_layers/")
    log("DONE")

if __name__ == "__main__":
    main()
