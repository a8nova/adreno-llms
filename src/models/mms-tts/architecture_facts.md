# Architecture Facts — vits

**architecture_class**: `other` — controls which scaffold tier nnopt emits (transformer → KV cache + batched-heads attention; ssm → state cache + selective-scan; rnn/other → universal hygiene only).

Extracted from config.json and (if available) modeling_*.py. This is the agent's FIRST source of truth for FFN/Attention/Norm math. Read it BEFORE writing any C++ layer.

## ACTIVATION — CRITICAL, AGENT MUST HARDCODE THIS ONE

- Config field: `hidden_act` = "relu"
- Canonical:    **ReLU**
- Formula:      `relu(x) = max(0, x)`
- Your FFN/MLP kernel MUST compute ReLU. Do NOT write a different activation based on the model name, the scaffold's default kernel name, or conventions from similar models. READ THE CONFIG FIELD ABOVE.
- If your generated code contains any of these tokens it is WRONG: `gelu`, `silu`, `sigmoid`
- Common trap: scaffolds often ship a kernel called `gelu_mul` or `silu_mul` — the kernel NAME does not determine the formula it computes. Verify the kernel body, not the kernel name.

## Code-path-selecting config flags — CRITICAL, READ FIRST

These config keys switch between distinct forward-pass paths in the PyTorch reference. Implementing the wrong branch produces a binary that compiles, runs, and emits non-zero output but has near-zero cosine vs reference. Match every flag below; do not assume the default.

- **model_type**: `"vits"`
- **use_bias**: `true`
- **use_stochastic_duration_prediction**: `true`

## Config-derived facts
- model_type: vits
- num_layers: 6
- hidden_size: 192
- num_heads: 2
- num_kv_heads: 2
- intermediate_size: 768
- position_embedding: n/a
- rope_theta: n/a
- tie_word_embeddings: undefined
- activation_fn_name (config): relu

## Source: model_info/transformers_src/modeling_vits.py

### FFN / Feed-forward: VitsFeedForward
```python
class VitsFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_1 = nn.Conv1d(config.hidden_size, config.ffn_dim, config.ffn_kernel_size)
        self.conv_2 = nn.Conv1d(config.ffn_dim, config.hidden_size, config.ffn_kernel_size)
        self.dropout = nn.Dropout(config.activation_dropout)

        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        if config.ffn_kernel_size > 1:
            pad_left = (config.ffn_kernel_size - 1) // 2
            pad_right = config.ffn_kernel_size // 2
            self.padding = [pad_left, pad_right, 0, 0, 0, 0]
        else:
            self.padding = None

    def forward(self, hidden_states, padding_mask):
        hidden_states = hidden_states.permute(0, 2, 1)
        padding_mask = padding_mask.permute(0, 2, 1)

        hidden_states = hidden_states * padding_mask
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)

        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states * padding_mask
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)

        hidden_states = self.conv_2(hidden_states)
        hidden_states = hidden_states * padding_mask

        hidden_states = hidden_states.permute(0, 2, 1)
        return hidden_states
```

### Attention: VitsAttention
```python
class VitsAttention(nn.Module):
    """Multi-headed attention with relative positional representation."""

    def __init__(self, config: VitsConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.window_size = config.window_size

        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.embed_dim}"
                f" and `num_attention_heads`: {self.num_heads})."
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)

        if self.window_size:
            self.emb_rel_k = nn.Parameter(torch.randn(1, self.window_size * 2 + 1, self.head_dim) * self.scaling)
            self.emb_rel_v = nn.Parameter(torch.randn(1, self.window_size * 2 + 1, self.head_dim) * self.scaling)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling

        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if self.window_size is not None:
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, src_len)
            relative_logits = torch.matmul(query_states, key_relative_embeddings.transpose(-2, -1))
            rel_pos_bias = self._relative_position_to_absolute_position(relative_logits)
            attn_weights += rel_pos_bias

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        if self.window_size is not None:
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, src_len)
            relative_weights = self._absolute_position_to_relative_position(attn_probs)
            rel_pos_bias = torch.matmul(relative_weights, value_relative_embeddings)
            attn_output += rel_pos_bias

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        if pad_length > 0:
            relative_embeddings = nn.functional.pad(relative_embeddings, [0, 0, pad_length, pad_length, 0, 0])

        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        return relative_embeddings[:, slice_start_position:slice_end_position]

    def _relative_position_to_absolute_position(self, x):
        batch_heads, length, _ = x.size()

        # Concat columns of pad to shift from relative to absolute indexing.
        x = nn.functional.pad(x, [0, 1, 0, 0, 0, 0])

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch_heads, length * 2 * length])
        x_flat = nn.functional.pad(x_flat, [0, length - 1, 0, 0])

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch_heads, length + 1, 2 * length - 1])
        x_final = x_final[:, :length, length - 1 :]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch_heads, length, _ = x.size()

        # Pad along column
        x = nn.functional.pad(x, [0, length - 1, 0, 0, 0, 0])
        x_flat = x.view([batch_heads, length * (2 * length - 1)])

        # Add 0's in the beginning that will skew the elements after reshape
        x_flat = nn.functional.pad(x_flat, [length, 0, 0, 0])
        x_final = x_flat.view([batch_heads, length, 2 * length])[:, :, 1:]
        return x_final
```

## Source: model_info/transformers_src/__init__.py

## Source: model_info/transformers_src/configuration_vits.py

## Source: model_info/transformers_src/tokenization_vits.py

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
