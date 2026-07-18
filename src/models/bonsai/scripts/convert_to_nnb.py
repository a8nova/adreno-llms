#!/usr/bin/env python3
"""One-time GGUF -> .nnb conversion (the ONLY code that ever reads GGUF).

.nnb container (little-endian):
  [0:8)    magic 'NNB1BIT\\0'
  [8:16)   u64 header_json_len
  [16:16+len)  header JSON:
     { "meta": {...model hyperparams + chat template...},
       "tensors": { name: {"dims":[...], "kind":"q1"|"f32",
                            "offset":..., "nbytes":...} } }
  data blob, each tensor 64-byte aligned, offsets relative to blob start.

Q1 tensor bytes are copied VERBATIM from the GGUF (18-byte units:
fp16 scale + 128 LSB-first sign bits — see model/Q1_0_LAYOUT.md).
F32 norm tensors copied verbatim. Weights are never dequantized.
"""
import json
import struct
import sys
from pathlib import Path

def main(gguf_path, manifest_path, tokenizer_path, out_path):
    man = json.loads(Path(manifest_path).read_text())
    tok = json.loads(Path(tokenizer_path).read_text())
    kv = man['kv']
    meta = {
        'arch': 'qwen3',
        'hidden': 4096, 'layers': 36, 'heads': 32, 'kv_heads': 8,
        'head_dim': 128, 'ffn': 12288, 'vocab': 151669,
        'rms_eps': 1e-6, 'rope_theta': 1000000.0,
        'yarn_factor': kv.get('qwen3.rope.scaling.factor', 4.0),
        'yarn_orig_ctx': kv.get('qwen3.rope.scaling.original_context_length', 16384),
        'eos': 151645, 'pad': 151643, 'add_bos': False,
        'chat_template': tok.get('tokenizer.chat_template', ''),
    }
    tensors = {}
    blob_off = 0
    order = sorted(man['tensors'], key=lambda t: t['offset'])
    for t in order:
        kind = 'q1' if t['type'] == 41 else 'f32'
        # trim alignment-padding slack from the offset-delta nbytes
        numel = t['numel']
        nbytes = (numel // 128) * 18 if kind == 'q1' else numel * 4
        assert nbytes <= t['nbytes'] <= nbytes + man['alignment'], t['name']
        blob_off = (blob_off + 63) // 64 * 64
        tensors[t['name']] = {'dims': t['dims'], 'kind': kind,
                              'offset': blob_off, 'nbytes': nbytes}
        blob_off += nbytes
    hdr = json.dumps({'meta': meta, 'tensors': tensors}).encode()
    data_start = man['data_start']
    with open(gguf_path, 'rb') as src, open(out_path, 'wb') as dst:
        dst.write(b'NNB1BIT\x00')
        dst.write(struct.pack('<Q', len(hdr)))
        dst.write(hdr)
        base = dst.tell()
        for t in order:
            info = tensors[t['name']]
            dst.seek(base + info['offset'])
            src.seek(data_start + t['offset'])
            remaining = info['nbytes']
            while remaining:
                chunk = src.read(min(remaining, 64 << 20))
                dst.write(chunk)
                remaining -= len(chunk)
    print(f'wrote {out_path}: {Path(out_path).stat().st_size/1e9:.3f} GB, '
          f'{len(tensors)} tensors')

if __name__ == '__main__':
    main(*sys.argv[1:5])
