#!/usr/bin/env python3
"""Convert the pocket-tts SentencePiece unigram tokenizer.model into the flat
binary `tokenizer_vocab.bin` that src/tokenizer.cpp reads (no protobuf at runtime).

Layout (little-endian):
  u32 magic  = 0x4B4F5450 ('PTOK')
  u32 version= 1
  u32 vocab_size
  u32 unk, bos, eos, pad        # special ids
  u32 byte_fallback[256]        # id of "<0xHH>" per byte, 0xFFFFFFFF if absent
  repeat vocab_size times:      # id = sequence index
    f32  score
    u16  byte_len
    u8[] piece (UTF-8)

Usage:  python3 scripts/convert_tokenizer.py <tokenizer.model> weights/tokenizer_vocab.bin
"""
import sys, struct, sentencepiece as spm

def main():
    src, dst = sys.argv[1], sys.argv[2]
    sp = spm.SentencePieceProcessor(); sp.Load(src)
    V = sp.GetPieceSize()
    bf = [0xFFFFFFFF] * 256
    blob = bytearray()
    for i in range(V):
        p = sp.IdToPiece(i)
        if len(p) == 6 and p.startswith("<0x") and p.endswith(">"):
            try: bf[int(p[3:5], 16)] = i
            except ValueError: pass
        pb = p.encode("utf-8")
        blob += struct.pack("<fH", sp.GetScore(i), len(pb)) + pb
    with open(dst, "wb") as f:
        f.write(struct.pack("<III", 0x4B4F5450, 1, V))
        f.write(struct.pack("<IIII", sp.unk_id(), sp.bos_id(), sp.eos_id(), sp.pad_id()))
        f.write(struct.pack("<256I", *bf))
        f.write(blob)
    print(f"wrote {dst}: vocab={V} unk={sp.unk_id()} bos={sp.bos_id()} eos={sp.eos_id()} bytes={len(blob)}")

if __name__ == "__main__":
    main()
