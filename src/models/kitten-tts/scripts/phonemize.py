#!/usr/bin/env python3
"""Text -> kitten-tts phoneme token ids (int32 .bin), via espeak-ng.

KittenTTS uses a 178-symbol IPA table (assets/phoneme_vocab.tsv) and wraps the
phoneme string with the "$" boundary symbol, matching the KittenTTS GitHub
wrapper's tokenizer.

Usage:  phonemize.py "some text" out.bin
Validate: phonemize.py --check   (reproduces the reference utterance's 70 ids)
"""
import json
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOCAB = ROOT / "assets" / "phoneme_vocab.tsv"


def load_vocab():
    v = {}
    for line in VOCAB.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        sym, idx = parts[0], parts[1]
        v[sym] = int(idx)
    return v


def phonemize(text):
    # -q quiet, --ipa produce IPA, -x/-v en-us US English voice.
    out = subprocess.run(
        ["espeak-ng", "-q", "--ipa=1", "-v", "en-us", text],
        capture_output=True, text=True, check=True,
    ).stdout
    # espeak emits one line per clause; join and normalize whitespace.
    return " ".join(out.split("\n")).strip()


def to_ids(phonemes, vocab):
    """Greedy longest-match over the symbol table, since some IPA symbols are
    multi-codepoint (diphthongs, length marks, tie bars)."""
    ids, unknown = [], []
    maxlen = max(len(k) for k in vocab)
    i = 0
    while i < len(phonemes):
        for L in range(min(maxlen, len(phonemes) - i), 0, -1):
            chunk = phonemes[i : i + L]
            if chunk in vocab:
                ids.append(vocab[chunk])
                i += L
                break
        else:
            unknown.append(phonemes[i])
            i += 1
    return ids, unknown


def build(text, vocab):
    ph = phonemize(text)
    ids, unknown = to_ids(ph, vocab)
    # framing = [bos, pad, eos] = [0, 10, 0] per .nnport/phoneme_vocab.json;
    # the shipped reference ids are bos=0 ... pad=10 (verified against
    # reference/reference_tokens.json). src/tokenizer.cpp applies the same
    # framing on device.
    ids = [0] + ids + [10]
    return ph, ids, unknown


def main():
    vocab = load_vocab()

    if len(sys.argv) >= 2 and sys.argv[1] == "--check":
        ref = json.loads((ROOT / "reference" / "reference_tokens.json").read_text())
        text = ref["prompt"]
        ph, ids, unknown = build(text, vocab)
        want = ref["input_ids"]
        print(f"text     : {text}")
        print(f"phonemes : {ph}")
        print(f"mine     : {len(ids)} ids  {ids[:14]}")
        print(f"reference: {len(want)} ids  {want[:14]}")
        if unknown:
            print(f"unmapped symbols: {sorted(set(unknown))}")
        print("MATCH" if ids == want else "DIFFERS")
        return 0 if ids == want else 1

    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    text, out = sys.argv[1], sys.argv[2]
    ph, ids, unknown = build(text, vocab)
    if unknown:
        print(f"  warning: {len(unknown)} unmapped symbol(s): {sorted(set(unknown))}")
    Path(out).write_bytes(struct.pack(f"{len(ids)}i", *ids))
    print(f"  {len(ids):3d} ids -> {out}   [{ph[:60]}...]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
