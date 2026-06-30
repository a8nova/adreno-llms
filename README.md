# adreno-llms 📱⚡ Lightning-fast inference on Qualcomm Adreno 6xx GPUs
> Autonomous kernel generation for Android embedded targets. Built by **NNOpt** — contact a8nova@gmail.com for early access.

### **📲 Try it now:** [Edgi on Google Play](https://play.google.com/store/apps/details?id=com.edgi.app) — the full multimodal stack in one app, running fully on-device.


https://github.com/user-attachments/assets/c5723e58-6bc7-4fbc-921b-59388e26f2c9


## ✨ NEW: six new modalities on-device — Vision, Speech, Listening, Music, Translation & Voice cloning

Beyond text generation, six new model types now run **fully on-device** on Adreno 6xx:

- **👁️ Vision (VLM)** — [SmolVLM-256M-Instruct](src/models/smolvlm-256m-instruct/) and [LFM2.5-VL-450M](src/models/lfm2-5-vl-450m/): image + text in, text out.
- **🗣️ Speech (TTS)** — [MMS-TTS](src/models/mms-tts/), [Kokoro-82M](src/models/kokoro-82m/) and [Pocket-TTS](src/models/pocket-tts/): text in, speech out.
- **🎧 Listening (ASR)** — [Whisper-tiny](src/models/whisper-tiny/): speech in, text out, with real-time streaming transcription.
- **🎵 Music (text→music)** — [MusicGen-small](src/models/musicgen-small/): text prompt in, music out.
- **🌐 Translation (S2ST/S2TT)** — [SeamlessM4T UnitY-small](src/models/seamless-m4t-unity-small/): speech in → translated speech or text out (English/Spanish/Portuguese/Hindi/Russian).
- **🎙️ Voice cloning (tone-color conversion)** — [OpenVoice V2](src/models/openvoice-v2/): speech in → the same speech re-voiced in a target speaker's tone color, fused single-pass clone at ~2× real-time.

<!-- Drop your demo .mp4 into a GitHub PR/issue, copy the resulting
     https://github.com/user-attachments/assets/<uuid> URL, and replace
     the placeholder line below. GitHub renders it inline. -->


https://github.com/user-attachments/assets/d7f1d479-bd01-4003-8a22-18a253fb0b90



## 📊 Models

All ports run on the **Motorola Razr 2020** (Snapdragon 765G / **Adreno 620**, 3.9 GB GPU global memory). Peak CPU mem = highest `peak_cpu_memory_mb` reading across the run — the actual host-process working set during inference (weights + KV cache + activations + driver overhead).

### Language models

Decode tok/s = warm 3-run median, greedy (`--temperature 0`), 32-token generation. Measured 2026-05-18.

| Model | Precision | Params | Architecture | Decode tok/s | TTFT (s) | Peak CPU mem (MB) | Notes |
|---|:-:|---:|---|---:|---:|---:|---|
| [Mamba2-130M](src/models/mamba2-130m/) | fp16 | 130M | SSD | **24.26** | 1.61 | 946 | State-space duality |
| [Mamba-130M](src/models/mamba-130m/) | fp16 | 130M | SSM | 21.52 | 1.62 | 686 | No attention |
| [SmolLM2-135M-Instruct](src/models/smollm2-135m-instruct/) | fp16 | 135M | LLaMA + GQA | **24.40** | 1.67 | 923 | Instruct-tuned; 61% of ceiling |
| [SmolLM2-135M-Instruct](src/models/smollm2-135m-instruct/) | **int8** | 135M | LLaMA + GQA | 24.21 | **0.91** | **670** | Per-row symmetric int8; −27% memory at tied tok/s |
| [OpenELM-270M-Instruct](src/models/openelm-270m/) | fp16 | 270M | LLaMA-style + tied lm_head | 14.65 | 2.00 | 1371 | 78.9% of 10 GB/s ceiling |
| [LFM2.5-230M](src/models/lfm2-5-350m/) | fp16 | 230M | Hybrid conv+attn | 13.81 | 1.89 | 1170 | Smallest LFM2.5 (14 layers); runs on the runtime-parameterized lfm2-5-350m port |
| [LFM2.5-230M](src/models/lfm2-5-350m/) | **int8** | 230M | Hybrid conv+attn | 18.07 | 0.57 | 783 | +30.8% vs fp16 |
| [LFM2.5-230M](src/models/lfm2-5-350m/) | **Q4** | 230M | Hybrid conv+attn | **20.37** | **0.52** | **585** | +47.5% vs fp16; smallest footprint in the LFM family |
| [LFM2.5-350M](src/models/lfm2-5-350m/) | fp16 | 350M | Hybrid conv+attn | 11.43 | 2.21 | 1666 | Liquid AI hybrid; 58% of texture ceiling |
| [LFM2.5-350M](src/models/lfm2-5-350m/) | **int8** | 350M | Hybrid conv+attn | 13.67 | **0.81** | 1015 | +19.6% vs fp16 |
| [LFM2.5-350M](src/models/lfm2-5-350m/) | **Q4** | 350M | Hybrid conv+attn | **14.54** | **0.79** | **719** | +27.2% vs fp16; ALU-bound nibble unpack |
| [Granite-4.0-350M](src/models/granite-4-0-350m/) | fp16 | 350M | Dense decoder + GQA | 10.19 | 2.41 | 2580 | IBM Granite (instruct); 71% of 10 GB/s ceiling |
| [Qwen2.5-0.5B-Instruct](src/models/qwen2-5-0-5b/) | fp16 | 500M | LLaMA + GQA | 10.36 | 3.66 | 2720 | Largest in the repo; 70% of 14 GB/s ceiling |

### Vision-language

Workload: `"Describe this image."` + sample JPEG. TTFT includes image preprocessing, vision tower, projector, and weight loading. Measured 2026-05-19 (SmolVLM) / 2026-06-02 (LFM2-VL).

| Model | Precision | Params | Architecture | Prefill tok/s | Decode tok/s | TTFT (s) | Peak CPU mem (MB) | Notes |
|---|:-:|---:|---|---:|---:|---:|---:|---|
| [SmolVLM-256M-Instruct](src/models/smolvlm-256m-instruct/) | fp16 | 256M | SigLIP + LLaMA (GQA) | 82.5 | **10.20** | 14.0 | 1227 | 29% of realistic BW ceiling; REPL with prewarm: 13.3 tok/s |
| [LFM2.5-VL-450M](src/models/lfm2-5-vl-450m/) | fp16 | 450M | SigLIP-2 (12L bidir) + LFM2 hybrid (conv+attn) | 47.6 | **10.00** | 128.6 | 2197 | Multi-tile (up to 10× 512² + thumb); 1797-token prompt; bidir attn dominates TTFT |
| [LFM2.5-VL-450M](src/models/lfm2-5-vl-450m/) | **int8** | 450M | SigLIP-2 (12L bidir) + LFM2 hybrid (conv+attn) | 46.4 | 9.7 | 130.4 | **2071** | Per-row symmetric; -50% disk; matches PyTorch fp32 ref byte-for-byte (first 7 tokens) |


### Text-to-speech

RTF = wall time / audio duration (lower is better; RTF ≤ 1.0 = real-time). Measured 2026-06-08 (Kokoro streaming), 2026-05-20 (MMS-TTS), 2026-06-25 (Pocket-TTS).

| Model | Precision | Params | Architecture | RTF | Audio | Wall (s) | Peak CPU mem (MB) | Notes |
|---|:-:|---:|---|---:|---:|---:|---:|---|
| [MMS-TTS](src/models/mms-tts/) | fp16 | 36M | VITS (enc + flow + HiFi-GAN) | **1.3** | 7.8 s | ~10.1 | 686 | ~1100 languages; per-op cosine ≥ 0.996 vs HF reference |
| [Kokoro-82M](src/models/kokoro-82m/) | fp16 | 82M | StyleTTS2 (text enc + duration + iSTFT decoder) | **0.93** | 2.0 s | ~2.1 | 783 | warm `--stream` / `--serve` streaming **faster than real-time** — sustained RTF **0.92–1.01** (int8 dot8 paths default-on, gapless); cold single-shot ~1.05 RTF (~2.1 s wall) |
| [Pocket-TTS](src/models/pocket-tts/) | fp16 | ~100M | flow-matching (FlowLM backbone + flow_net denoiser + Mimi SEANet decoder) | **1.47** | 4.0 s | ~5.9 | 513 | 24 kHz; baked `audio_prompt` default + 8 selectable v1 voices; steady-state **115 ms / 80 ms frame**; host/dispatch-bound on CLBlast Mimi GEMMs; bit-exact vs fp32 reference (cos 1.0) |

### Speech-to-text (ASR)

RTF = processing time / audio duration (lower is better; < 1.0 = faster than real time). 10-clip LibriSpeech set, batch mode (model loaded once, JIT amortized), 3-run warm median. Measured 2026-06-14.

| Model | Precision | Params | Architecture | RTF | Audio | Wall (s) | Peak CPU mem (MB) | Notes |
|---|:-:|---:|---|---:|---:|---:|---:|---|
| [Whisper-tiny](src/models/whisper-tiny/) | fp16 | 37M | Whisper encoder-decoder (4 enc + 4 dec, d=384) | **0.53** | 109.8 s | ~58.2 | 458 | faster than real time; 9/10 byte-exact; long clips (18–29 s) ~0.40; live streaming + VAD mode |

### Music generation

> The heaviest model in the repo — a 590M composite (T5 + token LM + EnCodec). After mega-layer fusion + an fp16 texture GEMV path it decodes at **~11 tok/s**, but at **RTF ~6.9×** it's still ~7× slower than real time on the Adreno 620 (the entry tier); faster Adreno tiers close the gap.

| Model | Precision | Params | Architecture | Decode tok/s | Output | Notes |
|---|:-:|---:|---|---:|---|---|
| [MusicGen-small](src/models/musicgen-small/) | fp16 | ~590M | T5 enc + 24-layer token LM + EnCodec | **11.0** | 32 kHz mono | RTF ~6.9× (a 5 s clip ≈ 34 s wall); peak ~2.7 GB; mega-fused decode + fp16 texture path |

### Speech translation (S2ST / S2TT)

> A multi-stage **cascade**, not a single forward pass: fbank → Conformer encoder → text decoder (beam-5) → T2U synthesizer → unit decoder (greedy) → CodeHiFiGAN vocoder. Speech in → translated speech (S2ST) or text (S2TT / ASR) out. On a realistic **6.0 s** English clip (warm): **S2ST RTF ~2.9×** (full cascade, ~17.7 s) and **S2TT / ASR RTF ~1.5×** (~9.2 s) — the text modes stop after the text decoder, skipping the T2U → unit-decoder → vocoder synthesis half (~8 s of the wall). Correctness held fixed: units bit-exact (211/211), waveform cos ≥ 0.999999 vs the `.ptl` reference.

| Model | Precision | Params | Architecture | RTF (warm) | Modes | Notes |
|---|:-:|---:|---|---:|---|---|
| [SeamlessM4T UnitY-small](src/models/seamless-m4t-unity-small/) | fp16 | ~323M | Conformer enc + text dec (beam-5) + T2U + unit dec + CodeHiFiGAN | **2.9** / **1.5** | s2s / s2tt / asr | 6.0 s input, warm: **S2ST 2.9×** (speech out, ~17.7 s) vs **S2TT/ASR 1.5×** (text out, ~9.2 s) — text modes skip T2U+unit-decoder+vocoder; eng/spa/por/hin/rus output |

### Voice cloning (tone-color conversion)

> Audio in → the same utterance re-voiced in a target speaker's tone color (the OpenVoice V2 converter, not the base TTS). A single fused forward — posterior encoder → normalizing flow → HiFi-GAN decoder — runs in one process with a shared weight/kernel cache (no per-stage disk dumps). The HiFi-GAN decoder is ~87% of the wall. RTF = wall / audio (lower better). Measured 2026-06-21 on a 19.27 s clip, warm 3-run median.

| Model | Precision | Params | Architecture | RTF | Audio | Wall (s) | Peak CPU mem (MB) | Notes |
|---|:-:|---:|---|---:|---:|---:|---:|---|
| [OpenVoice V2](src/models/openvoice-v2/) | fp16 | ~32M | VITS converter (posterior enc + flow + HiFi-GAN dec) | **2.01** | 19.27 s | ~38.8 | 192 | fused single-pass clone; enc_q ~2.05 s / flow ~3.7 s / dec ~33 s; bit-exact vs reference (flow cos 1.0000, dec 0.9998); thermal swing 1.97–2.03× on the foldable |


State-of-the-art small language models running on **Adreno 6xx GPUs** — the GPU class found in mid-range Android phones. Pure C++/OpenCL inference, cross-compiled on macOS, deployed via `adb` to `/data/local/tmp/`.

Each model ships with **hand-written OpenCL kernels tuned for Adreno's WG=64 wave size, vec4 fp16 ALUs, and image-cache texture path:**

- **`gemv_m1.cl`** — cooperative-WG=64 GEMV with vec4 fp16 + fp32 accumulator + `__local`-mem tree reduce. The decode-path workhorse, replacing CLBlast HGemm at every M=1 site (q/k/v/o-proj, gate/up/down-proj, lm_head). Hand-unrolled K-specialized variants (`_k768`, `_k1280`, `_k1536`) plus a generic-K **`_no4_coalesced`** path (vec4-strided across the warp for coalesced 256-byte DRAM reads) that hits 9–12 GB/s on Adreno 620. Adding the K=1280 specialization + coalesced generic to OpenELM-270M was the single largest unlock — **2.3× decode-throughput jump (4.60 → 10.59 tok/s)** before any other host-side work landed.
- **`block_fused.cl`** — fused-residual GEMV variants (`gemv_m1_no4_radd`, `gemv_m1_k1280_no4_radd`, `gemv_m1_no4_radd_coalesced`) that write `hidden[i] += sum` in a single launch, killing the separate `element_add` kernel after attn out-proj and MLP down-proj.
- **`attention.cl`** — grouped-query attention scores + softmax + out-proj.
- **`mlp.cl` / `mlp_swiglu.cl`** — fused gate × silu × up.
- **`rope.cl`** — rotary position embedding.
- **`rmsnorm.cl` / `layernorm.cl`** — cooperative-WG=64 row reduction with vec4 fp16, replacing the "1 thread per row" baseline.
- **`embedding.cl`** — embedding lookup.
- **`argmax.cl`** — GPU-side argmax over the vocab. Skips a 50K–150K-element fp16→fp32 readback + host scan per decode token at greedy `temperature=0`.
- **Mamba-specific:** `selective_scan.cl` (the SSM recurrence), `causal_conv1d.cl`, `ssm.cl` (`split_xz`, `split_xproj`, `bias_add_rows` glue).
- **Mamba2-specific:** `mamba_rms_norm_gated.cl` (the SSD-block gated norm), `lm_head.cl`.
- **LFM2-specific:** `convolution.cl` (short conv), `operator_norm.cl`.

CLBlast 1.6.3 handles the M>1 prefill GEMMs; the custom `gemv_m1` path takes over for M=1 decode.

Decode-loop techniques that compound on top of the kernels:

- **Chained cl_mem decode + ping-pong async readback** (OpenELM-270M, mamba2): the next iteration's embedding reads its token id directly from the previous iteration's GPU-side `argmax_result_` buffer (no host roundtrip). Async `clEnqueueReadBuffer(CL_FALSE)` of the int32 token is overlapped with the next forward's enqueue via a two-slot ping-pong. Closed a 33% wall-vs-GPU gap, **+1.38× on OpenELM (10.59 → 14.62 tok/s)**.
- **Persistent activation buffers** for every per-layer scratch: qkv, q/k norm output, attention scores, ctx, projection output, MLP gate, MLP outputs — plus the residual-stream embedding output, RMSNorm outputs, and lm_head logits at the model level. Eliminates per-decode-token `clCreateBuffer`/`clReleaseMemObject` round-trips entirely on the hot path.
- **GPU-side argmax** (`argmax.cl`): single-WG cooperative tree-reduction over the vocab, returning a 4-byte int32 instead of a 50K–150K-element fp16 readback + host scan.
- **Pre-built RoPE tables sized to MAX_CONTEXT_LENGTH at init** — eliminates the per-decode-step host-side trig recompute + buffer regrowth.

## 🚀 Quickstart

```bash
git clone https://github.com/a8nova/adreno-llms.git
cd adreno-llms

# Pick any model — example: SmolLM2-135M-Instruct (fp16)
./scripts/fetch_weights.sh smollm2-135m-instruct

cd src/models/smollm2-135m-instruct
NNOPT_DTYPE=fp16 ./scripts/build.sh --release
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "Once upon a time" 64
```

Tokens stream to stdout as they decode.

### Quantized variants (smaller, faster on some devices)

LFM2.5-350M and SmolLM2-135M-Instruct ship int8 and (for LFM2) Q4 weight bundles alongside fp16. Same binary, runtime-switchable via `NNOPT_QUANT`:

```bash
# Fetch the quantized bundle in addition to fp16
./scripts/fetch_weights.sh lfm2-5-350m --quant q4              # fp16 + Q4
./scripts/fetch_weights.sh smollm2-135m-instruct --quant int8  # fp16 + int8

# Build + deploy as usual (deploy pushes every weight bundle present locally)
cd src/models/lfm2-5-350m
NNOPT_DTYPE=fp16 ./scripts/build.sh --release
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# Run — NNOPT_QUANT picks which weight file the runtime loads
NNOPT_DTYPE=fp16 NNOPT_QUANT=q4 ./scripts/run_android.sh "..." 32 --temperature 0
```

Or run `--quant fp16` (the default — no extra files) and generate the quantized bundles locally with `python3 scripts/quantize_weights.py` / `quantize_q4.py` inside each port. See each model's README for the precision spread + measured tok/s.

## 🎯 Hardware target

- **Verified:** Motorola Razr 2020, Adreno 620, Snapdragon 765G, Android 11+. Also tested on Samsung Galaxy Tab A9+ (Adreno 619 v2, Snapdragon 695) — see LFM2's BENCHMARK.md for cross-device numbers.
- **Should work:** any arm64-v8a Android device with Adreno 6xx-class GPU (610, 612, 618, 619, 620, 630, 640, 650, 660, 680, 690, 695). Performance characteristics vary — the kernels are tuned around 768-thread Adreno 6xx wave behavior.
- **Probably works (untested):** Adreno 7xx flagships should run; numbers will be higher but kernels aren't tuned for them.
- **Won't hit these numbers:** Mali, PowerVR, Apple GPUs. The OpenCL code will run but the optimizations won't transfer.

## 🔧 Prerequisites

| Dependency | Version | How to get it |
|---|---|---|
| Android NDK | r25+ (any recent) | `sdkmanager 'ndk;26.1.10909125'` or Android Studio SDK Manager |
| CMake | 3.18+ | `brew install cmake` |
| OpenCL headers + Android `libOpenCL.so` link stub | Khronos | Run `scripts/setup_deps.sh` once — downloads headers and pulls `libOpenCL.so` from your connected device into `~/.cache/adreno-llms/` |
| CLBlast | 1.6.3 | Pinned via CMake `FetchContent` — built on first `build.sh` |
| Python | 3.10+ | For tokenizer parity checks |
| `adb` | platform-tools | `brew install --cask android-platform-tools` |

Cross-compiled on macOS (canonical path). Linux host should work — `build.sh` is bash-portable, the Android NDK toolchain is host-agnostic — but is currently untested.

## 📁 Repository layout

```
adreno-llms/
├── README.md, LICENSE, .gitignore
├── scripts/
│   ├── fetch_weights.sh          # downloads converted weights for one model
│   ├── fetch_openelm_weights.sh  # special: pulls + converts from apple/OpenELM-270M-Instruct
│   ├── convert_hf_to_bin.py      # universal HF safetensors → fp16 .bin + meta.json converter
│   └── verify_clean_checkout.sh  # pre-push leak scanner
└── src/models/<model>/
    ├── README.md                 # per-model quickstart
    ├── BENCHMARK.md              # optimization log + roofline + numbers
    ├── CMakeLists.txt
    ├── kernels/                  # OpenCL kernels (Adreno-tuned)
    ├── reference/                # Python HF reference parity
    ├── scripts/                  # build / deploy / run
    ├── src/                      # C++ inference
    └── weights/                  # tokenizer + meta (committed); model.bin (downloaded)
```

## Model-specific kernel tuning

The OpenCL kernels in this repo are **tuned for the exact model dimensions shipped** (hidden sizes, head counts, intermediate sizes, conv kernel widths, etc.). Tile sizes, workgroup layouts, shared-memory allocation, and vectorized load patterns are all calibrated to those specific tensor shapes and the Adreno 6xx wave architecture.

**What this means in practice:** if you swap in a checkpoint with different dimensions (e.g., a 500M variant where we ship a 256M), the kernels may hit suboptimal tile boundaries, waste shared memory, or fail on hardcoded shape assertions. You won't get the same speedups — and in some cases it won't run at all.

For MMS-TTS this isn't a concern — all ~1100 language checkpoints share the same VITS architecture and parameter count; only the weights differ.

For decoder-only LMs and vision-language models, each model size needs its own kernel tuning pass. If you want optimized kernels for a model size not currently in the repo, reach out: **a8nova@gmail.com**.

## 🤝 Contributing

## 📜 License

- **Code:** Apache 2.0 — see [LICENSE](LICENSE).
- **Weights:** carry their own upstream licenses, listed per-model:
  - mamba / mamba2 / Qwen2.5 / SmolLM2 / SmolVLM / Granite — Apache 2.0
  - LFM2.5 — Liquid AI's open license
  - MMS-TTS / SeamlessM4T — CC-BY-NC 4.0 (Meta; non-commercial)
  - **OpenELM — Apple Sample Code License (NOT redistributed in this repo; fetch script pulls from Apple's HF repo)**
