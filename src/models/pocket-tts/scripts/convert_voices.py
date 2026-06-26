#!/usr/bin/env python3
"""STEP 2 (voices) — convert pocket-tts voices to the raw fp16 the C++ port loads.

Two voice formats, two on-device load paths:
  v1  embeddings/<name>.safetensors  = {audio_prompt:[1,125,1024]}     (8 voices)
      → voices/<name>.fp16.bin       = audio_prompt [125,1024] fp16
        (runtime PRIMES it: bos + 125 frames = 126-pos KV; the verified path)
  v3  embeddings_v3/<name>.safetensors = pre-computed KV cache          (21 voices)
      → voices_v3/<name>.kv.bin      = 6 layers × [K[125,1024], V[125,1024]] fp16
        (runtime LOADS it directly, offset=125, no bos — same byte layout as
         weights/voice_kv.bin, head-major [pos, head*64+dim])

Usage:  python3 scripts/convert_voices.py [--out-dir .]
Then:   scripts/upload_weights_to_hf.sh  (or hf upload) pushes voices/ + voices_v3/
"""
import argparse, os, numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open

REPO = "kyutai/pocket-tts"
NLAYERS = 6

def f16(t):  # torch/np tensor → contiguous fp16 bytes
    return np.ascontiguousarray(np.asarray(t, dtype=np.float32)).astype(np.float16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    files = list_repo_files(REPO)
    v1 = sorted(f for f in files if f.startswith("embeddings/") and f.endswith(".safetensors"))
    v3 = sorted(f for f in files if f.startswith("embeddings_v3/") and f.endswith(".safetensors"))

    # ── v1: audio_prompt → voices/<name>.fp16.bin ────────────────────────────
    od1 = os.path.join(a.out_dir, "voices"); os.makedirs(od1, exist_ok=True)
    print(f"v1: {len(v1)} voices → voices/")
    for f in v1:
        name = os.path.basename(f).replace(".safetensors", "")
        with safe_open(hf_hub_download(REPO, f), framework="np") as sf:
            if "audio_prompt" not in sf.keys():
                print(f"  ! {name}: no audio_prompt, skip"); continue
            ap_t = sf.get_tensor("audio_prompt").reshape(-1, 1024)   # [N,1024], N varies per voice
        f16(ap_t).tofile(os.path.join(od1, f"{name}.fp16.bin"))
        print(f"  ✓ {name}.fp16.bin  [{ap_t.shape[0]},1024]")

    # ── v3: KV cache → voices_v3/<name>.kv.bin (6 × [K[125,1024], V[125,1024]]) ─
    od3 = os.path.join(a.out_dir, "voices_v3"); os.makedirs(od3, exist_ok=True)
    print(f"v3: {len(v3)} voices → voices_v3/")
    for f in v3:
        name = os.path.basename(f).replace(".safetensors", "")
        out = os.path.join(od3, f"{name}.kv.bin")
        with safe_open(hf_hub_download(REPO, f), framework="np") as sf, open(out, "wb") as w:
            npos = None
            for L in range(NLAYERS):
                c = sf.get_tensor(f"transformer.layers.{L}.self_attn/cache")  # [2,1,N,16,64]
                npos = c.shape[2]
                K = c[0, 0].reshape(npos, 1024)   # head-major [pos, head*64+dim]
                V = c[1, 0].reshape(npos, 1024)
                f16(K).tofile(w); f16(V).tofile(w)
        print(f"  ✓ {name}.kv.bin  (6 layers, {npos} pos)")
    print("\nDONE.  v1 → voices/   v3 → voices_v3/")

if __name__ == "__main__":
    main()
