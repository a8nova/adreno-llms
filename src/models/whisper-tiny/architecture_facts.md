# Architecture Facts — whisper

**architecture_class**: `other` — controls which scaffold tier nnopt emits (transformer → KV cache + batched-heads attention; ssm → state cache + selective-scan; rnn/other → universal hygiene only).

Extracted from config.json and (if available) modeling_*.py. This is the agent's FIRST source of truth for FFN/Attention/Norm math. Read it BEFORE writing any C++ layer.

## ACTIVATION — CRITICAL, AGENT MUST HARDCODE THIS ONE

- Config field: `activation_function` = "gelu"
- Canonical:    **GELU(erf)**
- Formula:      `gelu(x) = 0.5*x*(1 + erf(x / sqrt(2)))`
- Your FFN/MLP kernel MUST compute GELU(erf). Do NOT write a different activation based on the model name, the scaffold's default kernel name, or conventions from similar models. READ THE CONFIG FIELD ABOVE.
- If your generated code contains any of these tokens it is WRONG: `silu`, `sigmoid`, `relu`
- Common trap: scaffolds often ship a kernel called `gelu_mul` or `silu_mul` — the kernel NAME does not determine the formula it computes. Verify the kernel body, not the kernel name.

## Code-path-selecting config flags — CRITICAL, READ FIRST

These config keys switch between distinct forward-pass paths in the PyTorch reference. Implementing the wrong branch produces a binary that compiles, runs, and emits non-zero output but has near-zero cosine vs reference. Match every flag below; do not assume the default.

- **model_type**: `"whisper"`
- **use_cache**: `true`

## Config-derived facts
- model_type: whisper
- num_layers: 4
- hidden_size: 384
- num_heads: 0
- num_kv_heads: 0
- intermediate_size: 0
- position_embedding: n/a
- rope_theta: n/a
- tie_word_embeddings: undefined

## Source: model_info/transformers_src/modeling_whisper.py

### Attention: WhisperAttention
```python
class WhisperAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        layer_idx: int | None = None,
        config: WhisperConfig | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        if layer_idx is None and is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        # TODO: we need a refactor so that the different attention modules can get their specific kwargs
        # ATM, we have mixed things encoder, decoder, and encoder-decoder attn
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Scaling is susceptible to floating point arithmetics' inprecisions
        # which can lead to different results (this is dependent from model
        # to model, e.g. whisper is one such case). We therefore keep the
        # original order of scaling to follow the original implementation
        # and enforce no scaling (1.0) in the attention call below.
        query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()

        # Check is encoder-decoder model is being used. Otherwise we'll get `DynamicCache`
        if past_key_values is not None and isinstance(past_key_values, EncoderDecoderCache):
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_values.is_updated[self.layer_idx] = True
                past_key_values = past_key_values.cross_attention_cache
            else:
                past_key_values = past_key_values.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_values and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_values.layers[self.layer_idx].keys
            value_states = past_key_values.layers[self.layer_idx].values
        else:
            # Use the query's batch dimension for kv view so that a different-batch
            # encoder output (e.g. in tests) gets absorbed into the sequence axis,
            # preserving backward-compatible behaviour.
            kv_shape = (input_shape[0], -1, self.num_heads, self.head_dim)
            key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
            value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            output_attentions=output_attentions,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->Whisper, MBART->WHISPER
```

## Source: model_info/transformers_src/__init__.py

## Source: model_info/transformers_src/configuration_whisper.py

## Source: model_info/transformers_src/tokenization_whisper.py

## C++/OpenCL Implementation Guide

### FFN / MLP
Detected variant: **standard MLP**

PyTorch reference pattern:
```python
out = proj_2(act(proj_1(x)))
```

No chunk/gate — single linear + activation + linear. No special layout concerns.

### Attention
- **RoPE detected.** Verify: (a) inv_freq = theta^(-2i/head_dim), (b) HF default uses `cat([theta, theta], dim=-1)` so sin/cos tables duplicate across halves — NOT `[theta[0], theta[0], theta[1], theta[1], ...]`. (c) rotate_half splits last dim in two and applies `[-x2, x1]` — FIRST half negated-SECOND, SECOND half unchanged-FIRST. Apply to Q and K only, not V.

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
