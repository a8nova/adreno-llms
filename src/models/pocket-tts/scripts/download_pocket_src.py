#!/usr/bin/env python3
"""STEP 1 of 3 — download the raw kyutai/pocket-tts source (English 6-layer model +
voices + tokenizer) for the Adreno port. Convert (step 2) and upload (step 3) are
separate scripts. Resumable: re-run anytime; partial files continue, not restart.

kyutai/pocket-tts is a GATED repo — log in first:
    hf auth login          # (or: huggingface-cli login) — paste a READ token

Run:
    python3 scripts/download_pocket_src.py                       # → .src_download/
    python3 scripts/download_pocket_src.py --out-dir /tmp/pkt    # custom dir

Downloads (~240 MB total):
    tts_b6369a24.safetensors   236 MB  the 6-layer English model (the port's model)
    embeddings/*.safetensors    8 × ~512 KB  v1 voices (alba, marius, cosette, …)
    tokenizer.model             58 KB   SentencePiece unigram tokenizer
"""
import argparse, os, shutil, sys

REPO = "kyutai/pocket-tts"
MODEL_FILE = "tts_b6369a24.safetensors"
TOKENIZER = "tokenizer.model"
VOICE_DIR = "embeddings"   # v1 voices = {audio_prompt:[1,125,1024]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".src_download", help="where to stage the files")
    args = ap.parse_args()
    try:
        from huggingface_hub import hf_hub_download, list_repo_files, whoami
    except ImportError:
        sys.exit("pip install huggingface_hub  (then: hf auth login)")

    try:
        who = whoami()["name"]
    except Exception:
        sys.exit("Not logged in. Run `hf auth login` and paste a READ token "
                 "(kyutai/pocket-tts is a gated repo).")
    print(f"logged in as {who}; listing {REPO} …")

    try:
        files = list_repo_files(REPO)
    except Exception as e:
        sys.exit(f"Can't list {REPO} — accept the gated-repo terms on its HF page first.\n  {e}")

    voices = sorted(f for f in files if f.startswith(VOICE_DIR + "/") and f.endswith(".safetensors"))
    want = [MODEL_FILE, TOKENIZER] + voices
    out = args.out_dir
    os.makedirs(os.path.join(out, "voices"), exist_ok=True)

    print(f"downloading {len(want)} files → {out}/ (resumable)…")
    total = 0
    for f in want:
        p = hf_hub_download(REPO, f)                       # cached + resumes partials
        dst = os.path.join(out, "voices", os.path.basename(f)) if f.startswith(VOICE_DIR + "/") \
              else os.path.join(out, f)
        shutil.copy(p, dst)
        sz = os.path.getsize(dst); total += sz
        print(f"  ✓ {f:42s} {sz/1e6:7.1f} MB")
    print(f"\nDONE — {total/1e6:.0f} MB in {out}/")
    print("  model:    ", os.path.join(out, MODEL_FILE))
    print("  tokenizer:", os.path.join(out, TOKENIZER))
    print(f"  voices:    {len(voices)} in {out}/voices/")
    print("\nNext: STEP 2 — convert (BF16→fp16 + tokenizer_vocab.bin).")

if __name__ == "__main__":
    main()
