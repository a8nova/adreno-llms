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

EVERY hyperparameter is derived from the GGUF metadata — nothing is
hardcoded per size. The device runtime is dim-generic and reads this
header, so one converter + one binary serve every Bonsai size
(1.7B / 4B / 8B). Run with --dry-run to print the derived meta without
writing the blob.
"""
import json
import struct
import sys
from pathlib import Path

# Architectures this converter can DESCRIBE. Emitting a header is not the same
# as being able to execute it: the runtime refuses any arch it has no kernels
# for (see `ARCH_HAS_KERNELS` in src/model.cpp). Conversion is deliberately
# allowed to run ahead of the kernels so a new arch can be inspected, diffed
# against the llama.cpp oracle, and developed against.
#   qwen3  — dense decoder: fused QKV, QK-norm, GQA + KV cache, SwiGLU MLP.
#   qwen35 — hybrid stack (Qwen3.5 / Qwen3.6, incl. Bonsai-27B): 3 Gated-DeltaNet
#            linear-attention blocks per 1 gated-attention block. NO KERNELS YET.
SUPPORTED_ARCH = {'qwen3', 'qwen35'}

# Per-layer tensors the forward pass dereferences by name. A GGUF missing any
# of these would fault at upload time with a bare "missing tensor"; check up
# front so the failure names the size, not the tensor.
REQUIRED_BLK = ['attn_q.weight', 'attn_k.weight', 'attn_v.weight',
                'attn_output.weight', 'ffn_gate.weight', 'ffn_up.weight',
                'ffn_down.weight', 'attn_norm.weight', 'ffn_norm.weight',
                'attn_q_norm.weight', 'attn_k_norm.weight']

# qwen35 per-layer tensor sets, keyed by layer kind. 'L' = Gated-DeltaNet
# linear-attention block, 'F' = gated full-attention block. Both carry the
# same SwiGLU MLP + the two norms.
QWEN35_COMMON = ['attn_norm.weight', 'post_attention_norm.weight',
                 'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight']
QWEN35_LINEAR = ['attn_qkv.weight', 'attn_gate.weight', 'ssm_conv1d.weight',
                 'ssm_a', 'ssm_dt.bias', 'ssm_alpha.weight', 'ssm_beta.weight',
                 'ssm_norm.weight', 'ssm_out.weight']
QWEN35_FULL = ['attn_q.weight', 'attn_k.weight', 'attn_v.weight',
               'attn_output.weight', 'attn_q_norm.weight', 'attn_k_norm.weight']


def _kv(kv, arch, suffix, default=None, required=True):
    """Read `<arch>.<suffix>` from the GGUF KV block."""
    key = f'{arch}.{suffix}'
    if key in kv:
        return kv[key]
    if required and default is None:
        raise SystemExit(f'FATAL: GGUF metadata missing required key {key!r}')
    return default


def classify_qwen35_layers(layers, by_name, interval):
    """Per-layer kind string, e.g. 'LLLFLLLF...'.

    Derived from the tensors actually present (a block with `ssm_conv1d` IS a
    linear-attention block) and cross-checked against full_attention_interval,
    so a model that breaks the 3:1 pattern is caught rather than assumed.
    """
    kinds = []
    for L in range(layers):
        has_ssm = f'blk.{L}.ssm_conv1d.weight' in by_name
        has_q = f'blk.{L}.attn_q.weight' in by_name
        if has_ssm and not has_q:
            kinds.append('L')
        elif has_q and not has_ssm:
            kinds.append('F')
        else:
            raise SystemExit(
                f'FATAL: blk.{L} is neither a clean linear nor full block '
                f'(ssm_conv1d={has_ssm}, attn_q={has_q})')
    if interval:
        expect = ''.join('F' if (L % interval) == interval - 1 else 'L'
                         for L in range(layers))
        if ''.join(kinds) != expect:
            print(f'  NOTE: layer pattern deviates from '
                  f'full_attention_interval={interval}')
    return ''.join(kinds)


def derive_qwen35(kv, meta, by_name):
    """Extra header fields the hybrid stack needs, on top of the common ones."""
    a = 'qwen35'
    interval = int(kv.get(f'{a}.full_attention_interval', 0))
    meta['full_attn_interval'] = interval
    meta['layer_types'] = classify_qwen35_layers(meta['layers'], by_name,
                                                 interval)
    # Gated attention: partial RoPE over the first `rope_dim_count` dims only,
    # split into multimodal sections. The dense path rotates the whole head.
    meta['rope_dim_count'] = int(kv.get(f'{a}.rope.dimension_count',
                                        meta['head_dim']))
    meta['rope_sections'] = list(kv.get(f'{a}.rope.dimension_sections', []))
    meta['attn_output_gate'] = f'blk.{meta["layer_types"].find("F")}' \
                               '.attn_gate.weight' in by_name or \
                               any(n.endswith('.attn_gate.weight')
                                   for n in by_name)
    # Gated-DeltaNet (linear attention) dims.
    meta['ssm_conv_kernel'] = int(_kv(kv, a, 'ssm.conv_kernel'))
    meta['ssm_state_size'] = int(_kv(kv, a, 'ssm.state_size'))
    meta['ssm_group_count'] = int(_kv(kv, a, 'ssm.group_count'))
    meta['ssm_time_step_rank'] = int(_kv(kv, a, 'ssm.time_step_rank'))
    meta['ssm_inner_size'] = int(_kv(kv, a, 'ssm.inner_size'))
    return meta


def check_qwen35_tensors(meta, by_name):
    errs = []
    for L, kind in enumerate(meta['layer_types']):
        want = QWEN35_COMMON + (QWEN35_LINEAR if kind == 'L' else QWEN35_FULL)
        for suf in want:
            if f'blk.{L}.{suf}' not in by_name:
                errs.append(f'missing tensor blk.{L}.{suf} ({kind}-block)')
        if len(errs) > 8:
            break
    for n in ('token_embd.weight', 'output_norm.weight'):
        if n not in by_name:
            errs.append(f'missing tensor {n}')
    if errs:
        raise SystemExit('FATAL: qwen35 tensor set incomplete:\n  '
                         + '\n  '.join(errs[:12]))


def derive_meta(kv, tok, tensors_by_name):
    arch = kv.get('general.architecture')
    if arch not in SUPPORTED_ARCH:
        raise SystemExit(
            f'FATAL: unsupported architecture {arch!r} '
            f'(known: {sorted(SUPPORTED_ARCH)}).')

    hidden = int(_kv(kv, arch, 'embedding_length'))
    layers = int(_kv(kv, arch, 'block_count'))
    heads = int(_kv(kv, arch, 'attention.head_count'))
    kv_heads = int(_kv(kv, arch, 'attention.head_count_kv', default=heads,
                       required=False))
    # Qwen3 decouples head_dim from hidden/heads (e.g. 1.7B: 2048/16 = 128 but
    # 4B: 2560/32 = 80 while the real head_dim is 128). Always prefer the
    # explicit key; fall back to the q_norm row length, which IS head_dim.
    head_dim = _kv(kv, arch, 'attention.key_length', default=None,
                   required=False)
    if head_dim is None:
        qn = tensors_by_name.get('blk.0.attn_q_norm.weight')
        if qn is None:
            raise SystemExit('FATAL: no attention.key_length and no '
                             'blk.0.attn_q_norm.weight to infer head_dim from')
        head_dim = qn['dims'][0]
    head_dim = int(head_dim)

    ffn = int(_kv(kv, arch, 'feed_forward_length'))
    # vocab = ROW COUNT of the embedding matrix, which is what the logits GEMV
    # and the argmax are sized by. This can be SMALLER than the tokenizer's
    # token list (Bonsai-8B: 151669 rows vs a 151936-entry tokenizer), so
    # never take it from len(tokenizer.ggml.tokens).
    emb = tensors_by_name.get('token_embd.weight')
    if emb is None:
        raise SystemExit('FATAL: GGUF has no token_embd.weight')
    vocab = int(emb['dims'][1])

    rms_eps = float(_kv(kv, arch, 'attention.layer_norm_rms_epsilon',
                        default=1e-6, required=False))
    rope_theta = float(_kv(kv, arch, 'rope.freq_base', default=1000000.0,
                           required=False))
    # YaRN is optional: absent scaling keys mean plain RoPE. The host RopeYarn
    # treats factor <= 1 as a no-op, so 1.0 is the neutral default.
    yarn_factor = float(kv.get(f'{arch}.rope.scaling.factor', 1.0))
    yarn_orig_ctx = int(kv.get(f'{arch}.rope.scaling.original_context_length',
                               kv.get(f'{arch}.context_length', 32768)))

    eos = tok.get('tokenizer.ggml.eos_token_id')
    if eos is None:
        raise SystemExit('FATAL: tokenizer.ggml.eos_token_id missing')
    pad = tok.get('tokenizer.ggml.padding_token_id',
                  tok.get('tokenizer.ggml.bos_token_id', eos))

    meta = {
        'arch': arch,
        'hidden': hidden, 'layers': layers, 'heads': heads,
        'kv_heads': kv_heads, 'head_dim': head_dim, 'ffn': ffn, 'vocab': vocab,
        'rms_eps': rms_eps, 'rope_theta': rope_theta,
        'yarn_factor': yarn_factor, 'yarn_orig_ctx': yarn_orig_ctx,
        'eos': int(eos), 'pad': int(pad),
        'add_bos': bool(tok.get('tokenizer.ggml.add_bos_token', False)),
        'chat_template': tok.get('tokenizer.chat_template', ''),
    }
    if arch == 'qwen35':
        meta = derive_qwen35(kv, meta, tensors_by_name)
    return meta


def check_runtime_constraints(meta, tensors_by_name):
    """Fail loudly on shapes the Q1 kernels cannot execute."""
    if meta['arch'] == 'qwen35':
        # The hybrid stack has its own tensor set; the dense checks below do
        # not apply (no per-layer attn_q on linear blocks, head_dim 256, etc.).
        check_qwen35_tensors(meta, tensors_by_name)
        return
    errs = []
    # q1_gemv.cl consumes 128-weight units along K with no tail handling.
    for label, K in (('hidden', meta['hidden']), ('ffn', meta['ffn']),
                     ('attn_out (heads*head_dim)',
                      meta['heads'] * meta['head_dim'])):
        if K % 128:
            errs.append(f'{label}={K} is not a multiple of 128 (Q1_0 group size)')
    if meta['heads'] % meta['kv_heads']:
        errs.append(f"heads={meta['heads']} not divisible by "
                    f"kv_heads={meta['kv_heads']} (GQA grouping)")
    if meta['head_dim'] % 2:
        errs.append(f"head_dim={meta['head_dim']} must be even (RoPE pairs)")
    for L in range(meta['layers']):
        for suf in REQUIRED_BLK:
            n = f'blk.{L}.{suf}'
            if n not in tensors_by_name:
                errs.append(f'missing tensor {n}')
        if len(errs) > 8:
            break
    for n in ('token_embd.weight', 'output_norm.weight'):
        if n not in tensors_by_name:
            errs.append(f'missing tensor {n}')
    if errs:
        raise SystemExit('FATAL: model shape unsupported by the Q1 kernels:\n  '
                         + '\n  '.join(errs[:12]))


def main(argv):
    dry = '--dry-run' in argv
    argv = [a for a in argv if a != '--dry-run']
    if len(argv) < 3:
        raise SystemExit('usage: convert_to_nnb.py <model.gguf> <manifest.json> '
                         '<tokenizer.json> <out.nnb> [--dry-run]')
    gguf_path, manifest_path, tokenizer_path = argv[0], argv[1], argv[2]
    out_path = argv[3] if len(argv) > 3 else None
    if out_path is None and not dry:
        raise SystemExit('usage: need <out.nnb> unless --dry-run')

    man = json.loads(Path(manifest_path).read_text())
    tok = json.loads(Path(tokenizer_path).read_text())
    kv = man['kv']
    by_name = {t['name']: t for t in man['tensors']}

    meta = derive_meta(kv, tok, by_name)
    check_runtime_constraints(meta, by_name)

    print('derived meta (from GGUF metadata, nothing hardcoded):')
    for k, v in meta.items():
        if k == 'chat_template':
            print(f'  {k} = <{len(v)} chars>')
        elif k == 'layer_types' and len(v) > 24:
            nl, nf = v.count('L'), v.count('F')
            print(f'  {k} = {v[:24]}... ({nl} linear / {nf} full)')
        else:
            print(f'  {k} = {v}')
    # tied embeddings: no output.weight => logits reuse token_embd (model.cpp)
    print(f"  tied_embeddings = {'output.weight' not in by_name}")
    if meta['arch'] == 'qwen35':
        print('\n  *** qwen35: header only — NO DEVICE KERNELS EXIST YET. ***\n'
              '  The runtime refuses to load this arch. Converting is allowed so\n'
              '  the layout can be inspected and diffed against the llama.cpp\n'
              '  oracle while the Gated-DeltaNet kernels are built.')

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
    if dry:
        print(f'--dry-run: would write {len(tensors)} tensors, '
              f'{blob_off/1e9:.3f} GB blob')
        return
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
    main(sys.argv[1:])
