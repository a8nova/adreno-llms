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
NNOPT_DTYPE=fp16 ./scripts/build.sh        # Release + fp16: THE optimized binary. Always use this.
```

Variants you should rarely need:

```bash
NNOPT_DTYPE=fp32 ./scripts/build.sh         # fp32 binary (slow, A/B reference)
```

> The NDK auto-discovery can pick the wrong toolchain; export it explicitly:
> `export ANDROID_NDK="$HOME/Library/Android/sdk/ndk/android-ndk-r26d"`.
> Before deploy, stage the CLBlast lib: `cp build/fp16/_deps/clblast-build/libclblast.so build/libclblast.so`.

## Deploy & run

```bash
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh        # push binary + libs + kernels + weights

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

Not in git. Convert + upload with the helper (needs `huggingface-cli login`; upstream is gated):

```bash
python3 scripts/fetch_convert_push.py --out-dir /tmp/pocket_out \
        --push-repo a8nova/adreno-llms-weights --repo-type model
```

This produces `model.fp16.bin` (voice-agnostic), `model.fp16.meta.json`, `voices/<name>.fp16.bin`
(per-voice `audio_prompt`), and copies the SentencePiece `tokenizer.model`.

## Edgi integration

Registered as a TTS model (`task: speak`, `modality: tts`, `architecture: pocket-tts`) in
`Edgi/app/src/main/assets/catalog.json` and `FakeEngine`'s supported set, alongside Kokoro and
MMS-TTS. It surfaces in the Speak variant picker and runs through the generic `Engine.speak()`
contract (text → 24 kHz `AudioPcm`). Update the catalog's `voices/`+`tokenizer.model` SHA-256s
once the weights are uploaded to HuggingFace.
