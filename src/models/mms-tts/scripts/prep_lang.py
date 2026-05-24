#!/usr/bin/env python3
"""One-shot offline prep for any facebook/mms-tts-<lang> checkpoint.

Usage:
  scripts/prep_lang.py <lang_code> ["<sample text>"]

  prep_lang.py amh "ሰላም፣ እንደምን አደርክ?"
  prep_lang.py eng "Hello, my name is Alazar."
  prep_lang.py deu "Hallo, wie geht es dir?"

Produces under weights/<lang>/ and assets/<lang>/:
  weights/<lang>/model.bin        + model.meta.json        (fp32)
  weights/<lang>/model.fp16.bin   + model.fp16.meta.json   (fp16)
  weights/<lang>/tokenizer_vocab.bin                       (VTMV format)
  assets/<lang>/test_input_ids.bin                         (int32 ids)
  assets/<lang>/duration_noise.bin                         (fp32 noise fixture)
  assets/<lang>/prior_noise.bin                            (fp32 noise fixture)

The on-device binary runs with `--lang <code>` and consumes these.
Nothing in this script runs at inference time — the Android binary is
pure C++ (tokenizer + uroman + VITS forward + WAV write, all on-device).

Run-time deps: pip install huggingface_hub transformers safetensors torch numpy uroman
(uroman is only needed for non-Latin-script languages: amh, ara, khm, ...)
"""
import hashlib, json, os, struct, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel
from safetensors import safe_open

if len(sys.argv) < 2:
    print(__doc__, file=sys.stderr)
    sys.exit(2)

LANG = sys.argv[1]
HF_ID = f"facebook/mms-tts-{LANG}"
# Default sample text per known language; user-supplied text overrides.
_DEFAULT_TEXT = {
    "amh": "ሰላም፣ ስሜ አላዛር ነው፣ የተወለድኩት በአዲስ አበባ ነው።",
    "eng": "Hello, my name is.",
    "deu": "Hallo, wie geht es dir?",
    "fra": "Bonjour, je m'appelle Pierre.",
    "spa": "Hola, me llamo Maria.",
}
TEXT = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_TEXT.get(LANG, "Hello, my name is.")

PORT = Path(__file__).resolve().parent.parent
WEIGHTS = PORT / "weights" / LANG
ASSETS = PORT / "assets" / LANG
WEIGHTS.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

VOCAB_MAGIC = 0x564D5456
VOCAB_VERSION = 1


def find_blank_id(vocab):
    for k in ("_", "<blank>", "BLANK"):
        if k in vocab:
            return vocab[k]
    if 0 in vocab.values():
        return 0
    return -1


def find_pad_id(vocab, tok_config):
    # HF's MMS-TTS tokenizers store the actual pad token under tokenizer_config.json:
    # `pad_token` is the literal character (e.g. "c" for amh) — NOT a sentinel
    # like "<pad>". The vocab maps that character → its integer id. The
    # interleaved "blank" inside `add_blank=True` mode is THIS pad id, not the
    # token named "_". Getting this wrong scrambles every other token.
    pt = tok_config.get("pad_token") if isinstance(tok_config, dict) else None
    if pt and pt in vocab:
        return vocab[pt]
    for k in ("<pad>", "[PAD]", "PAD"):
        if k in vocab:
            return vocab[k]
    return -1


def vocab_is_latin_only(vocab):
    for tok in vocab.keys():
        for ch in tok:
            if ord(ch) > 0x7F:
                return False
    return True


def write_vocab_bin(path: Path, vocab: dict, tok_config: dict, add_blank: bool, is_uroman: bool):
    blank_id = find_blank_id(vocab)
    pad_id = find_pad_id(vocab, tok_config)
    unk_id = vocab.get("<unk>", -1)
    with path.open("wb") as f:
        f.write(struct.pack("<I", VOCAB_MAGIC))
        f.write(struct.pack("<I", VOCAB_VERSION))
        f.write(struct.pack("<I", len(vocab)))
        f.write(struct.pack("<i", pad_id))
        f.write(struct.pack("<i", unk_id))
        f.write(struct.pack("<i", blank_id))
        f.write(struct.pack("<B", 1 if add_blank else 0))
        f.write(struct.pack("<B", 1 if is_uroman else 0))
        f.write(struct.pack("<H", 0))
        for tok, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
            utf8 = tok.encode("utf-8")
            f.write(struct.pack("<iI", tid, len(utf8)))
            f.write(utf8)


def convert_weights():
    from huggingface_hub import snapshot_download
    local = Path(snapshot_download(repo_id=HF_ID, allow_patterns=[
        "*.safetensors", "pytorch_model.bin", "config.json",
        "vocab.json", "tokenizer_config.json", "special_tokens_map.json",
    ], max_workers=1))
    config = json.loads((local / "config.json").read_text())
    vocab = json.loads((local / "vocab.json").read_text())
    tok_config_path = local / "tokenizer_config.json"
    tok_config = json.loads(tok_config_path.read_text()) if tok_config_path.exists() else {}

    # --- fp32 model.bin ---
    bin_path = WEIGHTS / "model.bin"
    tensors = {}
    offset = 0
    sha = hashlib.sha256()
    safetensors_files = sorted(local.glob("*.safetensors"))
    with bin_path.open("wb") as bf:
        if safetensors_files:
            for st in safetensors_files:
                with safe_open(st, framework="pt") as f:
                    for name in f.keys():
                        t = f.get_tensor(name)
                        if not t.dtype.is_floating_point:
                            continue
                        arr = t.detach().to("cpu").float().numpy()
                        raw = arr.tobytes(order="C")
                        bf.write(raw)
                        sha.update(raw)
                        tensors[name] = {
                            "offset": offset, "shape": list(arr.shape),
                            "dtype": "float32",
                            "num_elements": int(arr.size),
                            "size_bytes": len(raw),
                        }
                        offset += len(raw)
        else:
            sd = torch.load(local / "pytorch_model.bin", map_location="cpu", weights_only=True)
            for name, t in sd.items():
                if not t.is_floating_point():
                    continue
                arr = t.float().numpy()
                raw = arr.tobytes(order="C")
                bf.write(raw)
                sha.update(raw)
                tensors[name] = {
                    "offset": offset, "shape": list(arr.shape),
                    "dtype": "float32",
                    "num_elements": int(arr.size),
                    "size_bytes": len(raw),
                }
                offset += len(raw)

    meta = {
        "model_id": HF_ID, "format": "binary", "layout": "row_major",
        "dtype": "float32",
        "quantization": {"enabled": False, "bits": 32, "method": "float32"},
        "bin_sha256": sha.hexdigest(),
        "bin_size_bytes": offset, "tensors": tensors, "total_bytes": offset,
    }
    (WEIGHTS / "model.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  fp32 model.bin: {offset/1024/1024:.1f} MB, {len(tensors)} tensors")

    # --- fp16 model.fp16.bin (uniform fp16 conversion) ---
    fp16_bin = WEIGHTS / "model.fp16.bin"
    fp16_tensors = {}
    cursor = 0
    sha16 = hashlib.sha256()
    with bin_path.open("rb") as src, fp16_bin.open("wb") as dst:
        # iterate in offset order to read sequentially
        ordered = sorted(tensors.items(), key=lambda kv: kv[1]["offset"])
        for name, info in ordered:
            src.seek(info["offset"])
            raw32 = src.read(info["size_bytes"])
            arr32 = np.frombuffer(raw32, dtype=np.float32).reshape(info["shape"]) if info["shape"] else np.frombuffer(raw32, dtype=np.float32)
            arr16 = arr32.astype(np.float16)
            raw16 = arr16.tobytes(order="C")
            dst.write(raw16)
            sha16.update(raw16)
            fp16_tensors[name] = {
                "offset": cursor, "shape": info["shape"],
                "dtype": "float16",
                "num_elements": info["num_elements"],
                "size_bytes": len(raw16),
            }
            cursor += len(raw16)
    meta16 = {
        "model_id": HF_ID, "format": "binary", "layout": "row_major",
        "dtype": "float16",
        "quantization": {"enabled": True, "bits": 16, "method": "float16"},
        "bin_sha256": sha16.hexdigest(),
        "bin_size_bytes": cursor, "tensors": fp16_tensors, "total_bytes": cursor,
    }
    (WEIGHTS / "model.fp16.meta.json").write_text(json.dumps(meta16, indent=2))
    print(f"  fp16 model.fp16.bin: {cursor/1024/1024:.1f} MB")

    # --- tokenizer_vocab.bin ---
    is_latin = vocab_is_latin_only(vocab)
    write_vocab_bin(WEIGHTS / "tokenizer_vocab.bin", vocab, tok_config,
                    add_blank=bool(tok_config.get("add_blank", True)),
                    is_uroman=bool(tok_config.get("is_uroman", is_latin)))
    print(f"  tokenizer_vocab.bin: {len(vocab)} entries, latin_vocab={is_latin}")

    return config, vocab


def make_fixtures(config):
    tok = AutoTokenizer.from_pretrained(HF_ID)
    tok_out = tok(TEXT, return_tensors="pt")
    input_ids = tok_out["input_ids"]
    T = int(input_ids.shape[1])
    print(f"  prompt tokenized: T={T} ids -> {input_ids[0].tolist()[:24]}{'...' if T>24 else ''}")
    ids_np = input_ids[0].to(torch.int32).cpu().numpy()
    (ASSETS / "test_input_ids.bin").write_bytes(ids_np.tobytes())

    torch.manual_seed(0)
    # duration_noise: (1, 2, T)
    duration_noise = torch.randn((1, 2, T), dtype=torch.float32).cpu().numpy().astype(np.float32)
    (ASSETS / "duration_noise.bin").write_bytes(duration_noise.tobytes())

    # prior_noise: (1, flow_size=192, T_frames_max). The on-device code
    # pads the buffer with deterministic noise if our buffer is shorter
    # than runtime T_frames, but we ship a generous capacity so the
    # fixture-driven path matches the reference distribution as much as
    # possible.
    flow_size = int(config.get("flow_size", 192))
    T_frames_max = 4096
    prior_noise = torch.randn((1, flow_size, T_frames_max), dtype=torch.float32).cpu().numpy().astype(np.float32)
    (ASSETS / "prior_noise.bin").write_bytes(prior_noise.tobytes())
    print(f"  fixtures: duration_noise=(1,2,{T}), prior_noise=(1,{flow_size},{T_frames_max})")


def main():
    print(f"[1/2] Converting weights {HF_ID} -> {WEIGHTS}")
    config, vocab = convert_weights()
    # Fixture generation requires transformers' VitsTokenizer which calls
    # `return_tensors="pt"` (needs PyTorch ≥ 2.4). It also produces files the
    # on-device interactive binary doesn't need (the runtime generates its
    # own GaussianRng noise). Skip when batch-converting all 1100+ languages
    # by setting NNOPT_SKIP_FIXTURES=1 in the environment.
    import os as _os
    if _os.environ.get("NNOPT_SKIP_FIXTURES") == "1":
        print("[2/2] Skipping fixtures (NNOPT_SKIP_FIXTURES=1)")
    else:
        print(f"[2/2] Building input fixtures for: {TEXT!r}")
        make_fixtures(config)
    print("Done.")


if __name__ == "__main__":
    main()
