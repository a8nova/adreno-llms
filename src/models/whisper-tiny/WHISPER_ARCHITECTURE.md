# Whisper-tiny — Architecture & On-Device Dataflow

A reading guide for **this** Android/OpenCL port: what Whisper is, how the model is
shaped, what "max tokens" means for audio, and the exact path a sound clip takes
through the C++ code from `main()` to a printed transcript.

Companion doc: `benchmark.md` (performance, profiling, and the optimization roadmap).

---

## 0. TL;DR

- Whisper is an **encoder–decoder Transformer** for speech-to-text. The **encoder** "listens" to the whole audio clip once and turns it into a fixed grid of features. The **decoder** then **writes the transcript one token at a time**, repeatedly looking back at the encoder features.
- Audio is **not** fed as a waveform to the Transformer. It is first turned into a **log-mel spectrogram** (a picture of frequencies over time), 80 mel bins × 3000 time frames for a 30s window.
- whisper-tiny is small: **4 encoder layers + 4 decoder layers, width 384, 6 heads**. The encoder always processes a **1500-frame** sequence; the decoder vocabulary is **51,865** tokens.
- `max_tokens` is the **cap on how many text tokens the decoder is allowed to generate** for the clip — it has nothing to do with the audio length directly. See §3.

---

## 1. High-level: what Whisper does

```
   raw audio (16 kHz mono PCM)
        │
        ▼
   ┌─────────────────┐     "turn sound into a picture"
   │  log-mel front  │     80 frequency bins × time frames
   │  end (FFT etc.) │
   └─────────────────┘
        │  mel spectrogram  [80, 3000]   (30s window, zero-padded)
        ▼
   ┌─────────────────┐     "listen to the whole clip, once"
   │    ENCODER      │     2 conv layers downsample time 3000→1500,
   │  (4 layers)     │     then 4 Transformer layers (bidirectional)
   └─────────────────┘
        │  audio features  [1500, 384]   ← computed ONCE per clip, then cached
        ▼
   ┌─────────────────┐     "write the transcript, one token at a time,
   │    DECODER      │      each step peeking at the audio features"
   │  (4 layers)     │     causal self-attn + cross-attn to encoder
   └─────────────────┘
        │  logits over 51,865 tokens  (for the next token)
        ▼
   pick best token → append → feed back in → repeat until <|endoftext|>
        │
        ▼
   token ids → de-tokenize → "Mr. Quilter is the apostle of..."
```

Two mental models that matter for performance:

1. **The encoder runs once; the decoder runs in a loop.** For a clip that produces
   25 tokens, the encoder runs **1×** and the decoder runs **25×**. That's why a
   one-time 16.7s encoder dominates short clips but amortizes on long ones (see
   `benchmark.md` §1).
2. **The decoder is autoregressive** — token *N* depends on tokens *0..N-1*, so it
   is an inherently serial loop, not a single big matmul.

---

## 2. Detailed architecture (whisper-tiny exact shapes)

Config constants (from `src/model_config.h`):

| Constant | Value | Meaning |
|----------|-------|---------|
| `NUM_MEL_BINS` | 80 | frequency bins in the spectrogram |
| `MAX_SOURCE_POSITIONS` | 1500 | encoder sequence length (audio frames after conv downsample) |
| `HIDDEN_SIZE` (d_model) | 384 | model width (every token/frame is a 384-vector) |
| `ENCODER_LAYERS` | 4 | encoder Transformer blocks |
| `DECODER_LAYERS` | 4 | decoder Transformer blocks |
| `ENCODER_ATTENTION_HEADS` | 6 | → head_dim = 384/6 = **64** |
| `DECODER_ATTENTION_HEADS` | 6 | head_dim = 64 |
| `ENCODER_FFN_DIM` / `DECODER_FFN_DIM` | 1536 | feed-forward inner width (= 4×384) |
| `MAX_TARGET_POSITIONS` | 448 | max tokens the decoder can ever produce |
| `VOCAB_SIZE` | 51865 | output token vocabulary |

### 2.1 Front-end: audio → log-mel spectrogram

Not a neural net — pure signal processing, done on-device in `src/mel_frontend.cpp`:

```
waveform [N samples @ 16kHz]
  → frame into overlapping windows (n_fft=400, hop=160)
  → FFT per window → power spectrum
  → multiply by 80 "mel" triangular filters (slaney scale)
  → log10, clamp, normalize
  → log-mel  [80 mel bins, T frames]   (T≈3000 for a 30s window)
```

Validated at cosine 1.0 vs HuggingFace `WhisperProcessor`. Cost ≈ 0.14s (negligible).

### 2.2 Encoder (runs once per clip)

```
mel [80, 3000]
  │
  ├─ Conv1d(80→384, k=3, stride=1) + GELU         ┐ "stem": project mel to width 384
  ├─ Conv1d(384→384, k=3, stride=2) + GELU        ┘ and downsample time 3000 → 1500
  │
  ├─ + sinusoidal positional embedding             [1500, 384]
  │
  ├─ 4× EncoderLayer:                              (pre-norm Transformer, BIDIRECTIONAL)
  │     residual = x
  │     x = LayerNorm(x)
  │     x = SelfAttention(x)        ← Q,K,V all from x; NO causal mask (sees whole clip)
  │     x = residual + x
  │     residual = x
  │     x = LayerNorm(x)
  │     x = fc2(GELU(fc1(x)))       ← MLP: 384 → 1536 → 384
  │     x = residual + x
  │
  └─ final LayerNorm                               → encoder_hidden_states [1500, 384]
```

The 4 encoder self-attentions each build a **1500×1500 attention matrix per head**
— this is the single most expensive thing in the whole pipeline (see `benchmark.md`).

### 2.3 Decoder (runs once per generated token)

The decoder is primed with Whisper's **forced prompt**:
`<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`, then it generates
real text tokens.

```
input token ids  (the prompt + everything generated so far)
  │
  ├─ Embedding(token) + positional embedding        [seq_len, 384]
  │
  ├─ 4× DecoderLayer:                                (pre-norm Transformer, CAUSAL)
  │     residual = x
  │     x = LayerNorm(x)
  │     x = SelfAttention(x, causal)   ← Q,K,V from x; causal mask (can't see future)
  │     x = residual + x
  │     residual = x
  │     x = LayerNorm(x)
  │     x = CrossAttention(x, encoder_hidden_states)  ← Q from x; K,V from ENCODER
  │     x = residual + x                                  (this is where it "reads" the audio)
  │     residual = x
  │     x = LayerNorm(x)
  │     x = fc2(GELU(fc1(x)))          ← MLP 384 → 1536 → 384
  │     x = residual + x
  │
  ├─ final LayerNorm                                 [seq_len, 384]
  │
  └─ lm_head (= embed_tokens.weightᵀ, tied)          → logits [seq_len, 51865]
                                                         we use the LAST row → next token
```

**Self-attention vs cross-attention — the key distinction:**
- *Self-attention*: the decoder attends to **its own previous tokens** (the text). Causal.
- *Cross-attention*: the decoder attends to the **encoder's audio features**. This is the bridge from sound to text. Its K and V depend only on the fixed audio → cacheable (this port now caches them, opt #7).

---

## 3. What does `max_tokens` mean for an audio input?

When you run:

```
./scripts/run_android.sh "transcribe" 128 --audio clip.bin --audio-seconds 5.86
                                       ^^^
                                    max_tokens
```

`max_tokens` (128 here) is **the maximum number of text tokens the decoder loop is
allowed to emit** for this clip. It is a *safety cap on the generation loop*, not a
description of the audio.

The decode loop (in `src/main.cpp`) is literally:

```cpp
for (int step = 0; step < max_new_tokens; ++step) {
    logits = model.forward(prompt_ids);   // run decoder over current tokens
    next   = sample(logits);              // pick the most likely next token
    if (next == end_of_text) break;       // <-- normal exit
    prompt_ids.push_back(next);           // append, feed back next step
}
```

So generation stops at **whichever comes first**:
- the model emits `<|endoftext|>` (50257) — the normal case, or
- it hits `max_tokens` (a hard cap so a misbehaving model can't loop forever).

For our 5.86s clip the model naturally produces ~25 tokens and stops on `<|endoftext|>`,
well under the 128 cap. A token is roughly a word-piece, so spoken length loosely
predicts token count, but the cap is just an upper bound.

**Why I sometimes run `max_tokens=4` while profiling:** it forces the loop to stop
after 4 tokens. Since the **encoder still runs in full** (it runs once, before the
loop produces any token), a 4-token run is **~all encoder, almost no decode** — a
clean way to isolate and measure the encoder's cost without waiting for the whole
transcript. It does **not** truncate the audio; it truncates the *text output*.

A useful rule for this port:

```
total_work(clip)  ≈  1 × encoder_cost   +   num_tokens × decoder_cost
                       └ fixed per clip ┘     └ grows with transcript length ┘
```

---

## 4. Dataflow of THIS C++ port (start → finish)

File map of the hot path:

| Stage | File | Key symbol |
|-------|------|-----------|
| entry / decode loop | `src/main.cpp` | `main()` |
| mel front-end | `src/mel_frontend.cpp` | log-mel extraction |
| graph orchestration | `src/backbone.cpp` | `_asr_forward_graph()` |
| encoder stem | `src/ops/WhisperEncoderFrontend.cpp` | conv×2 + GELU + pos |
| encoder/decoder block | `src/ops/WhisperEncoderLayer.cpp`, `WhisperDecoderLayer.cpp` | |
| attention | `src/ops/WhisperSdpaAttention.cpp` + `kernels/attn.cl` | self & cross |
| matmuls | `src/utils.cpp` (`pytorch_linear`) → CLBlast | all GEMMs |
| sampling / detokenize | `src/sampler.cpp`, `src/tokenizer.cpp` | argmax + BPE decode |

### 4.1 End-to-end sequence

```
main()  [src/main.cpp]
  │
  1. parse args: prompt, max_tokens, --audio file, --audio-seconds
  2. load tokenizer (weights/tokenizer_vocab.bin)
  3. read raw waveform (--audio clip.bin)
  4. MEL FRONT-END  ─────────────────────────────► log-mel [80, T]   (mel_frontend.cpp)
  5. upload mel → cl_mem _feats_buf  (fp16)
  6. ForwardDispatch::set_input_features(_feats_buf)
  │
  7. DECODE LOOP  (for step = 0 .. max_tokens):
  │     │
  │     └─ model.forward(prompt_ids)  ──► _asr_forward_graph()  [backbone.cpp]
  │           │
  │           │  ┌─ ENCODER (only on step 0; cached after) ─────────────────┐
  │           │  │  WhisperEncoderFrontend_forward: conv→conv→GELU→+pos       │
  │           │  │  for L in 0..3: WhisperEncoderLayer (LN→selfAttn→+res      │
  │           │  │                  →LN→fc1→GELU→fc2→+res)                     │
  │           │  │  final encoder LayerNorm                                   │
  │           │  │  encoder_hidden_states [1500,384]  ──► s_enc_cache         │
  │           │  └────────────────────────────────────────────────────────────┘
  │           │     (step 1+ : cache hit, encoder skipped entirely)
  │           │
  │           │  DECODER (every step):
  │           │  Embedding_forward(token ids) + positions     [seq_len, 384]
  │           │  for L in 0..3: WhisperDecoderLayer_forward(
  │           │        x, encoder_hidden_states, "model.decoder.layers.L")
  │           │          ├─ LN → SelfAttention(causal)      → +res
  │           │          ├─ LN → CrossAttention(enc states) → +res   ← K/V cached (opt #7)
  │           │          └─ LN → fc1 → GELU → fc2           → +res
  │           │  final decoder LayerNorm
  │           │  lm_head GEMM (tied embed weight)            → logits [seq_len, 51865]
  │           │  return last row of logits (host float vector)
  │           ▼
  │     8. sample(logits) → next token   (argmax; first 3 positions force <|en|>/<|transcribe|>/<|notimestamps|>)
  │     9. if next == <|endoftext|> break;  else append & stream partial text
  │
  10. print GENERATED_TEXT + RTF line
```

### 4.2 Inside one attention call (`WhisperSdpaAttention_forward`)

This is where 93% of GPU time lives. For each call (per head, conceptually):

```
input [T, 384]
  ├─ Q = Linear(input) · q_proj  → [Tq, 384] → pack to [H=6, Tq, 64]   (always)
  ├─ K = Linear(kv_src) · k_proj → [Tk, 384] → pack to [6, Tk, 64]     (cross: cached, opt #7)
  ├─ V = Linear(kv_src) · v_proj → [Tk, 384] → pack to [6, Tk, 64]     (cross: cached, opt #7)
  │     kv_src = input (self-attn)  OR  encoder_hidden_states (cross-attn)
  │
  ├─ scale Q by 1/sqrt(64)
  ├─ attn_scores:  scores[h,i,j] = Σ_d Q[h,i,d]·K[h,j,d]   → [6, Tq, Tk]   ◄── 76% of GPU time
  ├─ attn_softmax: row-wise softmax over Tk                                 ◄── 9%
  ├─ attn_wsum:    ctx[h,i,d] = Σ_j probs[h,i,j]·V[h,j,d]  → [6, Tq, 64]   ◄── 8%
  ├─ transpose [6,Tq,64] → [Tq, 384]
  └─ out = Linear(ctx) · out_proj  → [Tq, 384]
```

For the **encoder** Tq=Tk=1500 → the score matrix is 1500×1500×6 ≈ 13.5M values, each
a 64-long dot product. The current kernels compute this naively (one GPU thread per
output value, re-reading K from global memory) — that's the optimization target in
`benchmark.md` §4.

### 4.3 Two caches in this port (and the one that's missing)

| Cache | What it stores | Status |
|-------|----------------|--------|
| **Encoder-output cache** (`backbone.cpp`) | `encoder_hidden_states [1500,384]`, keyed by the mel buffer pointer | ✅ in place — encoder runs once per clip, not per token (13.7× win, already banked) |
| **Cross-attention K/V cache** (`WhisperSdpaAttention.cpp`) | packed `k_htd`/`v_htd` per decoder layer, keyed by weight prefix | ✅ added (opt #7) — cross-attn K/V projected once per clip |
| **Decoder self-attention KV cache** | per-step K/V for already-generated tokens | ❌ **not yet** — see below |

**The missing one matters.** Today the decoder is called with `k_cache_inout = nullptr`
everywhere, and `model.forward(prompt_ids)` is handed the **entire** token prefix each
step. So to generate token *N*, the decoder reprocesses tokens *0..N-1* from scratch —
self-attention is recomputed over the whole growing sequence every step. That makes
total decode work **O(N²)** in the number of output tokens. A persistent self-attention
KV cache (append one row per step) would make each step **O(N)** and is optimization
**#12** in `benchmark.md`. It's the structural reason decode cost climbs on long
transcripts.

---

## 5. Where the time goes (links to benchmark.md)

| Phase | Cost (clip 0, 5.86s) | Bottleneck |
|-------|----------------------|------------|
| mel front-end | ~0.14s | none (fine) |
| **encoder** (once) | **~16.7s** | `attn_scores` on 1500×1500 — 76% of all GPU time |
| **decode** (25 steps) | **~10.7s** | host/launch overhead + O(N²) self-attn recompute |

The roadmap to faster-than-real-time (aggregate RTF < 1.0) is in `benchmark.md` §4.
The headline: the encoder's attention is a batched matmul written as a naive kernel
running at <1% of the GPU's GEMM throughput — route it through CLBlast (like every
other matmul in this port already is) and the encoder collapses from ~16.7s to a
few seconds.
