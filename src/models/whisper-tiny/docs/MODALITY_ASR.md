# Modality: ASR (audio → text, encoder-decoder)

You are porting an automatic speech recognition model — encoder-decoder
transformer with audio input. Canonical example: Whisper-tiny
(`openai/whisper-tiny`). Other models in this family: SeamlessM4T-UnitY,
Speech2Text, MMS-ASR.

## Input shape

- **The C++ binary does NOT do on-device WAV → mel conversion in the first
  port.** The reference template ships `assets/test_input_features.bin`
  (mel spectrogram, fp32, shape `[n_mels, T_mel]` — e.g. `[80, 3000]` for
  Whisper). The binary loads this fixture and feeds it directly to the
  encoder. The agent can add an on-device STFT + mel filterbank pipeline
  in a follow-up port for realtime audio input.
- Read `reference/io_contract.json::input_fixtures[]` for the exact
  fixture path, shape, and dtype.

## Architecture shape — what's different from causal LM

| Aspect | Causal LM (SmolLM2, Llama) | ASR (Whisper) |
|---|---|---|
| Stacks | 1 (decoder-only) | 2 (encoder + decoder) |
| Encoder runs | — | ONCE per audio input |
| Decoder runs | N times (one per generated token) | N times (one per generated token) |
| Attention in decoder | self-attn (with KV cache) | self-attn (KV cache) + cross-attn (encoder K/V) |
| Cross-attention K/V source | — | Encoder hidden states (cached, NOT re-projected per step) |
| Initial input to decoder | First text token | `decoder_start_token_id` (e.g. `<|startoftranscript|>` = 50258 for Whisper) |

The scaffold detects this shape automatically (model_class contains
"Conditional"/"EncoderDecoder"/"Whisper"/"Seamless" etc.) and emits:
- `Model::encode(input_features)` — runs encoder once, caches `encoder_hidden_states_`
- `Model::decode(input_ids, start_pos)` — runs decoder once per step using cached encoder output
- A `Model::forward(input_ids, input_features, start_pos)` overload that
  combines them for backwards-compat with the causal-LM main.cpp loop
- `src/backbone.cpp::model_forward_graph` takes `cl_mem input_features`
  as an extra parameter when enc-dec is detected

The first version of `Model::encode` and `Model::decode` ships as
`NNOPT_ERROR(...)` sentinels — your job is to restructure backbone.cpp
to expose `encoder_forward_graph` and `decoder_forward_graph` separately,
then implement these methods to call them.

## Restructuring backbone.cpp for enc-dec

The default scaffold emits a single `model_forward_graph` that runs both
encoder and decoder. That's wasteful (encoder re-runs per decode step)
but correct. To make it efficient, split into two functions:

```cpp
// Run the encoder loop once, return the hidden states.
cl_mem encoder_forward_graph(
    OpenCLContext& cl_ctx, Weights& weights,
    cl_mem input_features, int feat_seq_len)
{
    cl_mem x = input_features;
    // <pre-encoder ops: Conv1D × 2 subsample, positional add>
    // <encoder layer loop: 4 × WhisperEncoderLayer_forward>
    // <encoder final norm>
    return x;  // [batch, encoder_seq_len, hidden] — caller owns
}

// Run one decode step with the cached encoder hidden states.
std::vector<float> decoder_forward_graph(
    OpenCLContext& cl_ctx, Weights& weights,
    const std::vector<int32_t>& decoder_input_ids,
    cl_mem encoder_hidden_states,
    int start_pos)
{
    // <token embedding + positional add>
    // <decoder layer loop: 4 × WhisperDecoderLayer_forward, threading encoder_hidden_states>
    // <decoder final norm>
    // <lm_head projection → vocab logits>
    // <readout last token's logits → vector<float>>
    return logits;
}
```

`Model::encode()` calls `encoder_forward_graph`; `Model::decode()` calls
`decoder_forward_graph` with `encoder_hidden_states_` from the Model member.

## Cross-attention wiring (THE critical detail)

`WhisperAttention` is the SAME C++ class used for BOTH self-attention and
cross-attention. The discriminator is `encoder_hidden_states`:

```cpp
cl_mem WhisperAttention_forward(
    OpenCLContext&, Weights&, cl_command_queue,
    cl_mem input,            // Q source (always from this layer's input)
    int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states,  // ← THIS toggles the mode
    const char* weight_prefix)
{
    const bool is_cross = (encoder_hidden_states != nullptr);
    cl_mem q = pytorch_linear(input, wp + ".q_proj.weight");
    cl_mem k, v;
    if (is_cross) {
        // Cross-attention: K/V from encoder_hidden_states, NOT cached per-step.
        // For efficiency, the agent can compute encoder_k/encoder_v ONCE in
        // Model::encode and stash them on Model; here we re-project every
        // call (slower but correct).
        k = pytorch_linear(encoder_hidden_states, wp + ".k_proj.weight");
        v = pytorch_linear(encoder_hidden_states, wp + ".v_proj.weight");
        // No KV cache writes for cross-attn.
    } else {
        // Self-attention: K/V from input, append to KV cache.
        k = pytorch_linear(input, wp + ".k_proj.weight");
        v = pytorch_linear(input, wp + ".v_proj.weight");
        // Append k, v to *k_cache_inout, *v_cache_inout at start_pos.
        // Then read full cached K/V for the scores.
    }
    // Standard scaled-dot-product: scores = Q @ K^T, softmax, attn = scores @ V.
    // Output projection.
    return out_proj(attn);
}
```

WhisperDecoderLayer calls `WhisperAttention_forward` twice per layer:
once for `self_attn` (encoder_hidden_states=nullptr) and once for
`encoder_attn` (encoder_hidden_states=the cached encoder output).
The weight prefix tells the function WHICH attention to load:

```cpp
// Inside WhisperDecoderLayer_forward(wp = "model.decoder.layers.0"):
cl_mem x_self = WhisperAttention_forward(..., /*encoder_hidden_states=*/nullptr,
                                          (wp + ".self_attn").c_str());
cl_mem x_cross = WhisperAttention_forward(..., /*encoder_hidden_states=*/encoder_hidden_states,
                                           (wp + ".encoder_attn").c_str());
```

## Generation loop (main.cpp restructure)

The causal-LM main.cpp template calls `model.forward(prompt_ids)` per
step. For Whisper:

### LOAD THE MEL FIXTURE — NEVER ZERO-FILL `input_features`

The reference pipeline produces `assets/test_input_features.bin` (a
float32 mel spectrogram, shape `[n_mels, T_mel]`, e.g. `[80, 3000]` for
Whisper). It is declared in `reference/io_contract.json::input_fixtures`
under `name="test_input_features"`. **You MUST load that file** in
`main.cpp` and pass it via `ForwardDispatch::set_input_features`. The
default text-LM main.cpp has no audio path — if you leave it as-is, the
encoder runs on uninitialised / zero buffers and the decoder sampler
collapses to `id=0` every step regardless of how correct the math is.

Use the existing `read_input_ids_bin(...)` helper as a template — copy
it to a `read_float_bin(...)` variant that loads a float32 binary into
a `std::vector<float>`, upload to OpenCL, hand to `ForwardDispatch`:

```cpp
static bool read_float_bin(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);
    if (n <= 0 || (n % sizeof(float)) != 0) return false;
    out.resize(static_cast<size_t>(n) / sizeof(float));
    f.read(reinterpret_cast<char*>(out.data()), n);
    return f.good() || f.eof();
}

// In main.cpp:
std::vector<float> mel;
if (!read_float_bin("assets/test_input_features.bin", mel)) {
    NNOPT_ERROR("missing assets/test_input_features.bin");
    return 1;
}
// Convert to nnopt_storage_t (fp16 in fp16 builds) before upload.
// Whisper expects [n_mels, T_mel] = [80, 3000].
cl_mem feats_buf = upload_feats_storage(cl_ctx, mel);  // implement this — copy upload pattern from weights.cpp
ForwardDispatch::set_input_features(feats_buf);
std::vector<float> logits = model.forward(prompt_ids);
```

### DECODE FROM THE FORCED PREFIX

The reference pipeline saves the forced decoder prefix (e.g.
`[50258, 50259, 50359, 50363]` for Whisper-tiny English transcription)
to `assets/test_input_ids.bin`. Use that as the starting `prompt_ids` —
do NOT re-tokenize a text prompt for audio models. The decoder is not
a continuation engine; it autoregresses from the prefix conditioned on
the encoder hidden states.

```cpp
auto features = load_float_bin("assets/test_input_features.bin");
cl_mem features_buf = upload_to_device(features, [n_mels, T_mel]);

// Encode ONCE
cl_mem encoder_hidden = model.encode(features_buf);

// Decode loop — start with decoder_start_token_id
std::vector<int32_t> generated = { MODEL_CONFIG::DECODER_START_TOKEN_ID };  // 50258 for Whisper
for (int step = 0; step < max_new_tokens; step++) {
    auto logits = model.decode(generated, /*start_pos=*/step);
    int next = sampler.sample(logits, generated);
    if (next == MODEL_CONFIG::EOS_TOKEN_ID) break;
    generated.push_back(next);
}

std::string text = tokenizer.decode(generated, /*skip_special_tokens=*/true);
std::cout << "GENERATED_TEXT: " << text << std::endl;
```

`MODEL_CONFIG::DECODER_START_TOKEN_ID` and `MODEL_CONFIG::EOS_TOKEN_ID`
are auto-generated from `config.json::decoder_start_token_id` and
`config.json::eos_token_id`. For Whisper-tiny the start token is 50258
(`<|startoftranscript|>`), which itself encodes the language and task
selection via the immediately-following tokens (`<|en|>`, `<|transcribe|>`,
`<|notimestamps|>`). Use the generation_config defaults — don't hardcode.

## Whisper-specific tokens

| Token | ID (Whisper-tiny) | Role |
|---|---|---|
| `<|startoftranscript|>` | 50258 | First decoder token; signals "begin generating" |
| `<|en|>` | 50259 | Language hint (forced via `model.generation_config.forced_decoder_ids`) |
| `<|transcribe|>` | 50359 | Task = transcription (vs `<|translate|>` for translation) |
| `<|notimestamps|>` | 50363 | Disables timestamp prediction |
| `<|endoftext|>` | 50257 | EOS — stop generating |

For the first port, you can either:
- Hardcode the forced prefix `[50258, 50259, 50359, 50363]` and start
  the generation loop from there
- Read `model_info/config.json::forced_decoder_ids` and apply at decode
  time per Whisper's standard generate() behavior

## Encoder frontend (Conv1d×2 + positional)

Whisper's encoder starts with two `nn.Conv1d` layers (kernel 3, stride 1
then 3, stride 2) that downsample the mel input from `T_mel` to `T_mel/2`.
Then the result is permuted from `[hidden, T_mel/2]` to `[T_mel/2, hidden]`
and the learned `model.encoder.embed_positions.weight[T_mel/2, hidden]`
is added. THEN the encoder layer loop runs.

**The current scaffolder does NOT emit this frontend automatically.** You
write it as `src/ops/WhisperEncoderFrontend.cpp` and call it from
`src/backbone.cpp` BEFORE the encoder-layer loop:

```cpp
// In backbone.cpp, before the encoder loop:
cl_mem x = WhisperEncoderFrontend_forward(
    cl_ctx, weights, queue,
    input_features,
    /*seq_len=*/MODEL_CONFIG::MAX_SOURCE_POSITIONS * 2,  // T_mel = 3000
    /*layer_idx=*/-1, /*start_pos=*/0,
    nullptr, nullptr, nullptr,
    "model.encoder");
// Then loop over encoder layers using x as input.
```

Write a custom OpenCL kernel (`kernels/whisper_encoder_frontend.cl`)
with the conv1d + GELU + add_positional ops. PyTorch Conv1d weight
layout is `[out_channels, in_channels, kernel_size]`. CLBlast Gemm
won't apply directly; dispatch a custom 2D NDRange kernel
`(gws = [hidden, T_out])` instead.

## Cross-attention K/V sequence length

The universal `<Class>_forward` signature has ONE `seq_len` parameter,
which the decoder passes as the *decoder* sequence length (e.g. 4). In
cross-attention `WhisperSdpaAttention_forward(..., encoder_hidden_states, "model.decoder.layers.N.encoder_attn")`,
K and V come from `encoder_hidden_states` which has length
`MAX_SOURCE_POSITIONS = 1500` — NOT the decoder seq_len. If you use
`Tk = Tq` in cross-attention, you read only the first `Tq` (e.g. 4) of
the 1500 encoder frames; the decoder ignores 99.7% of the audio and the
sampler collapses on every step.

When the call is cross-attention (detect via
`encoder_hidden_states != nullptr` AND `wp` contains `encoder_attn`),
override `Tk`:

```cpp
const bool is_cross = (encoder_hidden_states != nullptr);
const int Tk = is_cross ? MODEL_CONFIG::MAX_SOURCE_POSITIONS : Tq;
```

Allocate K/V/scores buffers using this `Tk`. The pytorch_linear for
K/V projection must use `Tk` rows of `kv_src`, not `Tq`.

## Decoder self-attention causal mask

WhisperDecoder uses standard causal masking for self-attention (token i
can only attend to positions ≤ i). When `encoder_hidden_states ==
nullptr` (self-attn path) you MUST apply a causal mask before softmax,
setting `scores[h, i, j] = -INFINITY` for `j > i`. Without the mask,
even prefill produces wrong activations (positions see future tokens)
and decoder cosines diverge at the very first attention layer.

The cross-attention path does NOT use a causal mask — every decoder
position can attend to every encoder frame.

## out_proj bias

`WhisperAttention.out_proj` has `bias=True`. Don't omit the bias add at
the end of attention — it produces consistent ~0.001 RMS drift on
attention output that compounds across 4 decoder layers into a
non-negligible cosine gap on the lm_head input.

## Final decoder layer_norm

After the 4 decoder layers and BEFORE the lm_head, PyTorch applies one
more LayerNorm: `hidden = model.decoder.layer_norm(hidden)`. This
weight exists in the safetensors as `model.decoder.layer_norm.weight`.
If you skip this call in `backbone.cpp` (jumping straight from the
decoder layer loop to the lm_head matmul), logits magnitudes are wrong
and the sampler picks a low-id token (often `0`) for every step.

```cpp
// In backbone.cpp, between the decoder layer loop and the lm_head:
cl_mem x_final = LayerNorm_forward(
    cl_ctx, weights, queue,
    x, seq_len, /*layer_idx=*/-1, /*start_pos=*/0,
    nullptr, nullptr, nullptr,
    "model.decoder.layer_norm");
if (x) clReleaseMemObject(x);
x = x_final;
// THEN: pytorch_linear(... x, embed_tokens.weight, logits_buf) for tied lm_head.
```

## Tokenizer

Whisper uses the GPT-2 BPE tokenizer extended with the special tokens
above. `PortTokenizer` handles this generically — the produced
`weights/tokenizer_vocab.bin` includes the special tokens. The agent's
`tokenizer.cpp::decode()` should pass `skip_special_tokens=true` for
clean transcript output.

## Output

The C++ binary writes `GENERATED_TEXT: <transcript>` to stdout (same
convention as causal LMs). Evaluate compares against
`reference/reference_text.txt` (which holds the librispeech reference
transcript from the test fixture). Convergence criterion: token match
≥95% against the reference, same as text LMs.

## Common failure modes for ASR ports

1. **Forgetting cross-attention.** WhisperDecoderLayer needs TWO
   attention calls per layer (self + cross). If only self_attn is
   wired, the decoder ignores the audio entirely and generates a
   plausible-sounding but unrelated transcript.
2. **Wrong K/V source in cross-attention.** Q is always from the
   current decoder hidden state. K/V come from `encoder_hidden_states`
   in cross-attn (NOT from input or KV cache).
3. **Decoder positional embedding wrong for `start_pos > 0`.** Whisper
   uses learned positional embeddings (not RoPE). The `start_pos`
   parameter tells `WhisperEmbedding_forward` which positional row to
   look up. `(void)start_pos` makes every decode step look up row 0 →
   immediate divergence after the first generated token.
4. **Skipping `<|notimestamps|>` token.** Without it the model emits
   timestamp tokens (`<|0.00|>`, etc.) interleaved with text, breaking
   the transcript reader. Either hardcode the forced prefix or read
   `forced_decoder_ids` from config.
5. **Conv1D stride bug.** Whisper's second Conv1D has stride=2; the
   first has stride=1. Mixing them up halves or doubles the encoder
   sequence length, which time-shifts every cross-attention call.
