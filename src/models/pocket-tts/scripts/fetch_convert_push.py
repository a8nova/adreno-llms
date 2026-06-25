#!/usr/bin/env python3
"""Download kyutai/pocket-tts from HuggingFace, convert to the Adreno port's flat
fp16 layout, and (optionally) push the result to your own HF repo.

WHAT IT PRODUCES (in --out-dir):
    model.fp16.bin         213 model tensors + bos_before_voice, cast BF16->fp16,
                           cumulative offsets, no padding (voice-AGNOSTIC).
    model.fp16.meta.json   schema the C++ weights.cpp reads: tensors[name] =
                           {offset, shape, dtype, num_elements, size_bytes}.
    voices/<name>.fp16.bin  per-voice audio_prompt [125,1024] as raw fp16 (one
                           small file per voice — the "voice").
    tokenizer.model        SentencePiece model (copied through).

SOURCE FACTS (verified against kyutai/pocket-tts):
  - tts_b6369a24.safetensors = 213 BF16 tensors; names == port names (flow_lm.*/mimi.*).
  - audio_prompt is NOT in the model; it is the voice. v1 embeddings/<name>.safetensors
    hold exactly {audio_prompt: [1,125,1024]}.
  - flow_lm.bos_before_voice is voice-independent; it is not in the safetensors, so we
    carry it over from your existing converted model.fp16.bin (a fixed [1024] vector).

REQUIRES:  pip install torch safetensors huggingface_hub numpy
           huggingface-cli login            # kyutai/pocket-tts is a GATED repo —
                                            # accept the license on the model page first
USAGE:
  # convert only (no upload):
  python3 scripts/fetch_convert_push.py --out-dir /tmp/pocket_out
  # convert + push to your repo (model + all voices):
  python3 scripts/fetch_convert_push.py --out-dir /tmp/pocket_out \
          --push-repo <your-username>/pocket-tts-adreno --repo-type model
"""
from __future__ import annotations
import argparse, json, struct, sys
from pathlib import Path

SRC_REPO   = "kyutai/pocket-tts"
MODEL_FILE = "tts_b6369a24.safetensors"      # 6-layer English model (the port's model)
VOICE_DIR  = "embeddings"                      # v1 voices = {audio_prompt:[1,125,1024]}
TOKENIZER  = "tokenizer.model"


def _import_deps():
    try:
        import torch, numpy as np
        from safetensors.torch import safe_open
        from huggingface_hub import hf_hub_download, list_repo_files, HfApi
        return torch, np, safe_open, hf_hub_download, list_repo_files, HfApi
    except ImportError as e:
        sys.exit(f"missing dep: {e}\n  pip install torch safetensors huggingface_hub numpy")


def f16_bytes(t, np):
    """torch tensor (any dtype) -> contiguous little-endian fp16 bytes + shape."""
    import torch
    arr = t.detach().to(torch.float16).cpu().contiguous().numpy()
    return arr.tobytes(), list(t.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--port-weights", default=str(Path(__file__).resolve().parents[1] / "weights"),
                    help="existing converted weights dir (source of bos_before_voice)")
    ap.add_argument("--push-repo", default=None, help="HF repo id to upload to, e.g. you/pocket-tts-adreno")
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    args = ap.parse_args()

    torch, np, safe_open, hf_hub_download, list_repo_files, HfApi = _import_deps()
    out = Path(args.out_dir); (out / "voices").mkdir(parents=True, exist_ok=True)

    # 1) ── model: 213 BF16 tensors -> flat fp16 + meta ───────────────────────────
    print(f"downloading {SRC_REPO}/{MODEL_FILE} ...")
    mpath = hf_hub_download(SRC_REPO, MODEL_FILE)
    blob, meta_tensors, off = bytearray(), {}, 0
    with safe_open(mpath, framework="pt") as f:
        keys = list(f.keys())
        for name in keys:
            b, shape = f16_bytes(f.get_tensor(name), np)
            ne = 1
            for d in shape: ne *= d
            meta_tensors[name] = {"offset": off, "shape": shape, "dtype": "float16",
                                  "num_elements": ne, "size_bytes": len(b)}
            blob += b; off += len(b)
    print(f"  packed {len(keys)} model tensors ({off/1e6:.1f} MB)")

    # 2) ── bos_before_voice: voice-independent, carried from existing weights ─────
    pmeta = json.load(open(Path(args.port_weights) / "model.fp16.meta.json"))
    pbin  = open(Path(args.port_weights) / "model.fp16.bin", "rb").read()
    t = pmeta["tensors"]["flow_lm.bos_before_voice"]
    b = pbin[t["offset"]: t["offset"] + t["size_bytes"]]
    meta_tensors["flow_lm.bos_before_voice"] = {"offset": off, "shape": t["shape"], "dtype": "float16",
                                                "num_elements": t["num_elements"], "size_bytes": len(b)}
    blob += b; off += len(b)

    (out / "model.fp16.bin").write_bytes(bytes(blob))
    json.dump({"model_id": SRC_REPO, "format": "binary", "layout": "row_major", "dtype": "float16",
               "quantization": {"enabled": True, "bits": 16, "method": "float16"},
               "tensors": meta_tensors, "total_bytes": off},
              open(out / "model.fp16.meta.json", "w"))
    print(f"  wrote model.fp16.bin ({off/1e6:.1f} MB, {len(meta_tensors)} tensors, voice-agnostic)")

    # 3) ── voices: every v1 embedding's audio_prompt -> voices/<name>.fp16.bin ────
    voices = [f for f in list_repo_files(SRC_REPO) if f.startswith(VOICE_DIR + "/") and f.endswith(".safetensors")]
    print(f"converting {len(voices)} voices ...")
    for vf in sorted(voices):
        name = Path(vf).stem
        with safe_open(hf_hub_download(SRC_REPO, vf), framework="pt") as f:
            if "audio_prompt" not in f.keys():
                print(f"  ! {name}: no audio_prompt (KV-cache format) — skipping"); continue
            b, shape = f16_bytes(f.get_tensor("audio_prompt"), np)
        (out / "voices" / f"{name}.fp16.bin").write_bytes(b)
        print(f"  {name}: audio_prompt {shape} -> voices/{name}.fp16.bin")

    # 4) ── tokenizer passthrough ─────────────────────────────────────────────────
    try:
        (out / TOKENIZER).write_bytes(open(hf_hub_download(SRC_REPO, TOKENIZER), "rb").read())
        print(f"  copied {TOKENIZER}")
    except Exception as e:
        print(f"  (tokenizer: {e})")

    # 5) ── push ──────────────────────────────────────────────────────────────────
    if args.push_repo:
        api = HfApi()
        print(f"creating + uploading to {args.repo_type}:{args.push_repo} ...")
        api.create_repo(args.push_repo, repo_type=args.repo_type, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.push_repo, repo_type=args.repo_type)
        print(f"  pushed: https://huggingface.co/{args.push_repo}")
    else:
        print("conversion done (no --push-repo given; nothing uploaded).")


if __name__ == "__main__":
    main()
