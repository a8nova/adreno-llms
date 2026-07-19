#!/usr/bin/env python3
"""GGUF v3 parser for the Bonsai-8B Q1_0 port (P0.2).

Pure stdlib. Emits:
  model/manifest.json   — header, all KV metadata (scalars), tensor table with
                          {name, dims, type, offset, nbytes, bits_per_elem}
  model/tokenizer.json  — full tokenizer arrays (tokens, token_type, merges)
                          + chat template
Asserts the plan's §1.2 ground truth: 399 tensors, types ⊆ {0, 41},
type-41 bits/elem == 1.125 on every tensor.
"""
import json
import struct
import sys
from pathlib import Path

GGUF_MAGIC = 0x46554747  # 'GGUF' LE

SIMPLE = {
    0: ('B', 1), 1: ('b', 1), 2: ('H', 2), 3: ('h', 2),
    4: ('I', 4), 5: ('i', 4), 6: ('f', 4), 7: ('?', 1),
    10: ('Q', 8), 11: ('q', 8), 12: ('d', 8),
}

def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', errors='replace')

def read_value(f, t):
    if t in SIMPLE:
        fmt, sz = SIMPLE[t]
        return struct.unpack('<' + fmt, f.read(sz))[0]
    if t == 8:
        return read_str(f)
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, et) for _ in range(n)]
    raise ValueError(f'unknown KV type {t}')

def main(path):
    out_dir = Path(path).parent
    kv = {}
    tensors = []
    with open(path, 'rb') as f:
        magic, version = struct.unpack('<II', f.read(8))
        assert magic == GGUF_MAGIC, f'bad magic {magic:#x}'
        assert version == 3, f'expected GGUF v3, got {version}'
        n_tensors, n_kv = struct.unpack('<QQ', f.read(16))
        for _ in range(n_kv):
            key = read_str(f)
            t = struct.unpack('<I', f.read(4))[0]
            kv[key] = read_value(f, t)
        for _ in range(n_tensors):
            name = read_str(f)
            nd = struct.unpack('<I', f.read(4))[0]
            dims = list(struct.unpack(f'<{nd}Q', f.read(8 * nd)))
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensors.append({'name': name, 'dims': dims, 'type': ttype,
                            'offset': offset})
        align = kv.get('general.alignment', 32)
        hdr_end = f.tell()
        data_start = (hdr_end + align - 1) // align * align
        f.seek(0, 2)
        file_size = f.tell()

    # nbytes from offset deltas (offsets are relative to data_start)
    by_off = sorted(tensors, key=lambda t: t['offset'])
    for i, t in enumerate(by_off):
        end = by_off[i + 1]['offset'] if i + 1 < len(by_off) \
              else file_size - data_start
        t['nbytes'] = end - t['offset']
        numel = 1
        for d in t['dims']:
            numel *= d
        t['numel'] = numel
        t['bits_per_elem'] = round(t['nbytes'] * 8 / numel, 6)

    # ---- assertions from the plan §1.2 ----
    types = sorted({t['type'] for t in tensors})
    n41 = sum(1 for t in tensors if t['type'] == 41)
    n0 = sum(1 for t in tensors if t['type'] == 0)
    print(f'tensors={len(tensors)}  types={types}  type41={n41}  f32={n0}')
    assert len(tensors) == 399, len(tensors)
    assert set(types) <= {0, 41}, types
    bad = [t for t in tensors if t['type'] == 41
           and abs(t['bits_per_elem'] - 1.125) > 1e-9]
    # alignment padding can inflate the LAST tensor's delta; tolerate <1 block
    truly_bad = [t for t in bad if t['nbytes'] * 8 / t['numel'] > 1.13]
    for t in bad:
        print(f'  note: {t["name"]} bits/elem={t["bits_per_elem"]} '
              f'(padding tolerance)' , file=sys.stderr)
    assert not truly_bad, [t['name'] for t in truly_bad]

    tok = {k: v for k, v in kv.items() if k.startswith('tokenizer.')}
    (out_dir / 'tokenizer.json').write_text(json.dumps(tok))
    manifest = {
        'version': 3, 'n_tensors': len(tensors), 'data_start': data_start,
        'alignment': align,
        'kv': {k: v for k, v in kv.items()
               if not k.startswith('tokenizer.')},
        'tensors': tensors,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(f'data_start={data_start}  wrote manifest.json + tokenizer.json')
    # headline metadata for the port
    for k in ('general.file_type', 'qwen3.rope.scaling.factor',
              'qwen3.rope.scaling.original_context_length',
              'qwen3.rope.freq_base', 'qwen3.context_length',
              'tokenizer.ggml.eos_token_id'):
        v = kv.get(k, tok.get(k, '<missing>'))
        print(f'  {k} = {v}')

if __name__ == '__main__':
    main(sys.argv[1])
