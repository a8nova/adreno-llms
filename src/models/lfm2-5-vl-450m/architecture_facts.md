# Architecture Facts — lfm2_vl

**architecture_class**: `other` — controls which scaffold tier nnopt emits (transformer → KV cache + batched-heads attention; ssm → state cache + selective-scan; rnn/other → universal hygiene only).

Extracted from config.json and (if available) modeling_*.py. This is the agent's FIRST source of truth for FFN/Attention/Norm math. Read it BEFORE writing any C++ layer.

## Code-path-selecting config flags — CRITICAL, READ FIRST

These config keys switch between distinct forward-pass paths in the PyTorch reference. Implementing the wrong branch produces a binary that compiles, runs, and emits non-zero output but has near-zero cosine vs reference. Match every flag below; do not assume the default.

- **model_type**: `"lfm2_vl"`
- **use_image_special_tokens**: `true`
- **use_thumbnail**: `true`

## Config-derived facts
- model_type: lfm2_vl
- num_layers: 16
- hidden_size: 1024
- num_heads: 16
- num_kv_heads: 8
- intermediate_size: 4608
- position_embedding: n/a
- rope_theta: n/a
- tie_word_embeddings: undefined

## Source: model_info/transformers_src/modeling_lfm2_vl.py

## Source: model_info/transformers_src/__init__.py

## Source: model_info/transformers_src/configuration_lfm2_vl.py

## Source: model_info/transformers_src/modular_lfm2_vl.py

## C++/OpenCL Implementation Guide

### FFN / MLP
Detected variant: **standard MLP**

PyTorch reference pattern:
```python
out = proj_2(act(proj_1(x)))
```

No chunk/gate — single linear + activation + linear. No special layout concerns.

### Attention
- **GQA detected** (num_q_heads=16, num_kv_heads=8). Each KV head is shared by 2 Q heads. Broadcast K/V across query-head groups before SDPA.

- **Head-major vs token-major.** SDPA typically produces out_heads as `[num_q_heads, M, head_dim]` (head-major). PyTorch reshapes to `[M, num_q_heads * head_dim]` (token-major) BEFORE out_proj. If you skip this transpose, out_proj reads strides incorrectly and cos at block0_sub_attn_out_proj ≈ 0. Transpose explicitly (kernel or per-row copies) before the GEMM.

- **CLBlast ldb for out_proj.** weight is stored `[out_features, in_features]` row-major. With `TransposeB=kYes` in row-major, `ldb` MUST be `in_features` (the physical row length of the stored matrix), NOT `out_features`. Using `out_features` gives cos ≈ 0 at out_proj.

### Memory-layout gotchas (always read)
- Row-major tensors CANNOT be split on the last dim via contiguous byte offsets.
  Any op that does `chunk`/`split`/`unbind` on `dim=-1` needs a per-row split kernel
  or separate GEMMs with sliced weights.
- CLBlast row-major with `TransposeB=kYes`: ldb = physical row length of the STORED matrix.
  If weight stored as `[N, K]`, ldb=K. Using N here silently reads wrong strides.
- Sub-buffer origins must be aligned to `CL_DEVICE_MEM_BASE_ADDR_ALIGN` bits on most
  mobile GPUs (usually 1024). Small-I FFN sub-buffers may silently return zeros on Adreno.
- `clEnqueueCopyBuffer` is byte-linear — it copies a contiguous byte range, not a
  strided submatrix. For transposes/reshapes use a kernel.
- `LAYER_DUMP` naming MUST match the PyTorch sub-module attribute name exactly
  (dots → underscores). SxSDebug pairs dumps to refs by this name — a typo means
  "no has_dump" and the tool can't bisect inside the block.
