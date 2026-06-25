# Pocket-TTS — Android / OpenCL port (Motorola Razr 2020, Adreno 620)

Kyutai's Pocket-TTS (flow-matching text-to-speech, 24 kHz) ported to C++/OpenCL for Adreno 6xx
GPUs, running fully on-device. Verified on Motorola Razr 2020 (Adreno 620 v2).

- **Upstream:** [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) (gated repo)
- **Parameters:** ~100M
- **Architecture:** flow-matching TTS — text → SentencePiece tokens → **FlowLM backbone**
  (6-layer transformer, d=1024, 16 heads, RoPE, persistent KV cache) → **flow_net** denoiser
  (SimpleMLPAdaLN) → **Mimi decoder** (quantizer → upsample → 2 transformer layers → SEANet
  vocoder) → 24 kHz waveform. The voice is a baked-in `audio_prompt [125,1024]` conditioning
  (voice-agnostic model + per-voice file; see `scripts/fetch_convert_push.py`).
- **Precision:** fp16, fp32 accumulate. **No int8.**
- **Performance (Adreno 620, stock governor):** steady-state **~130–150 ms per 80 ms frame
  ≈ 1.6–2.0× slower than real-time**; bit-exact vs the fp32 reference (cosine 1.0). The decode
  is **host-bound** on CLBlast's per-call dispatch cost (~6 ms/call × 19 Mimi GEMMs), not GPU
  compute (GPU busy ≈ 64 ms/frame). See [BENCHMARK.md](BENCHMARK.md) and `OPTIMIZATION_PLAN.md`
  for the full journey (8.7× over the 738 ms/frame baseline) and why crossing real-time at fp16
  needs int8 or a from-scratch tuned GEMM.

## Build (default = fastest)

```bash
./scripts/build.sh        # Release + fp16 (default): THE optimized binary. Always use this.
```

> The NDK auto-discovery can pick the wrong toolchain; export it explicitly:
> `export ANDROID_NDK="$HOME/Library/Android/sdk/ndk/android-ndk-r26d"`.
> CLBlast is statically linked into the binary — nothing extra to stage or deploy.

## Deploy & run

```bash
./scripts/deploy_android.sh        # push binary + libs + kernels + weights (fp16 default)

# Generate speech (end-to-end, all on GPU). argv: <weights.bin> <meta.json> generate <in> <out> <n_frames> <noise_std> <token_ids>
adb shell "cd /data/local/tmp/pocket_tts_inference && LD_LIBRARY_PATH=lib \
  ./pocket_tts_inference_fp16 weights/model.fp16.bin weights/model.fp16.meta.json \
  generate dummy.bin out.bin 50 0.7 364,1143,287"
# → out.bin = fp32 waveform, 1920 samples/frame @ 24 kHz (50 frames ≈ 4 s)
```

Env flags (perf/diagnostics, all default-off):

- `NNOPT_PROFILE=1` — per-kernel GPU timing + host enqueue-cost breakdown (`dump_summary`).
- `NNOPT_RECORD=1` — record/replay decode via `cl_qcom_recordable_queues` (records the backbone;
  bit-exact, ~wall-neutral on this device — see BENCHMARK.md for why).
- `NNOPT_RECORD_MIMI=1` — also record Mimi (reference only; slower — the driver won't record
  CLBlast's fast `Xgemm`, only the ~100× slower `XgemmDirect`).

## Weights & voices

The converted fp16 weights are **already published** on HuggingFace
(`a8nova/adreno-llms-weights/pocket-tts`) — just **fetch** them, no conversion or gated
access needed:

```bash
./scripts/fetch_weights.sh pocket-tts          # → weights/  (public mirror, no login)
```

Published set (≈211 MB):
- `model.fp16.bin` + `model.fp16.meta.json` — the English 6-layer model. The default
  voice **alba** is baked in (single-voice works out of the box).
- `tokenizer_vocab.bin` — SentencePiece unigram vocab (see `scripts/convert_tokenizer.py`).
- `voices/<name>.fp16.bin` — the **8 selectable v1 voices** (alba, azelma, cosette, eponine,
  fantine, javert, jean, marius), each a raw `audio_prompt` the runtime primes. Select one at
  run time (`generate … <voice.fp16.bin>`, or the serve `@voice <path>` command).
- NOT shipped: the v2/v3 voice sets. Their *pre-computed KV* was made by a different model
  snapshot and is **silent** on `tts_b6369a24` (verified — the loader is correct; the data is
  incompatible).

### Maintainer: regenerate + re-upload (gated source)
Only needed to rebuild the published weights. `kyutai/pocket-tts` is GATED — `hf auth login`
first and accept its terms, then:

```bash
python3 scripts/download_pocket_src.py                              # raw model + voices + tokenizer
python3 scripts/convert_voices.py --out-dir weights                # v1 audio_prompts → voices/
python3 scripts/convert_tokenizer.py <tokenizer.model> weights/tokenizer_vocab.bin
../../../scripts/upload_weights_to_hf.sh pocket-tts                 # push model + tokenizer + voices/
```
(`fetch_convert_push.py` is the older all-in-one converter; the split scripts above are current.)
