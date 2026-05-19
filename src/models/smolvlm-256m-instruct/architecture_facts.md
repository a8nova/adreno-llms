# Architecture Facts — idefics3

**architecture_class**: `other` — controls which scaffold tier nnopt emits (transformer → KV cache + batched-heads attention; ssm → state cache + selective-scan; rnn/other → universal hygiene only).

Extracted from config.json and (if available) modeling_*.py. This is the agent's FIRST source of truth for FFN/Attention/Norm math. Read it BEFORE writing any C++ layer.

## Config-derived facts
- model_type: idefics3
- num_layers: 30
- hidden_size: 576
- num_heads: 9
- num_kv_heads: 3
- intermediate_size: 1536
- position_embedding: n/a
- rope_theta: n/a
- tie_word_embeddings: undefined

## Source: model_info/transformers_src/modeling_idefics3.py

### FFN / Feed-forward: Idefics3VisionMLP
```python
class Idefics3VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
```

### FFN / Feed-forward: Idefics3SimpleMLP
```python
class Idefics3SimpleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**2)
        output_size = config.text_config.hidden_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.proj(x)


# Copied from transformers.models.idefics2.modeling_idefics2.Idefics2EncoderLayer with Idefics2->Idefics3
```

### Attention: Idefics3VisionAttention
```python
class Idefics3VisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Ignore copy
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        queries = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        values = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.siglip.modeling_siglip.SiglipMLP with Siglip->Idefics3Vision
```

### Normalization: Idefics3RMSNorm
```python
class Idefics3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Idefics3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
```

## Source: model_info/transformers_src/configuration_idefics3.py

## C++/OpenCL Implementation Guide

### FFN / MLP
Detected variant: **standard MLP**

PyTorch reference pattern:
```python
out = proj_2(act(proj_1(x)))
```

No chunk/gate — single linear + activation + linear. No special layout concerns.

### Attention
- **GQA detected** (num_q_heads=9, num_kv_heads=3). Each KV head is shared by 3 Q heads. Broadcast K/V across query-head groups before SDPA.

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
