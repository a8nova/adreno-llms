# adreno-llms 📱⚡

### ⚡ Lightning-fast inference on Qualcomm Adreno 6xx GPUs ⚡


https://github.com/user-attachments/assets/8ed4d05b-150b-45f7-953c-f1260c053c42


> All code in this repo was ported and optimized by **NNOpt** — a coding agent for porting and optimizing neural networks for Android embedded targets. **Contact a8nova@gmail.com for early access.**

State-of-the-art small language models running on **Adreno 6xx GPUs** — the GPU class found in mid-range Android phones. Pure C++/OpenCL inference, cross-compiled on macOS, deployed via `adb` to `/data/local/tmp/`.

Each model ships with **hand-written OpenCL kernels tuned for Adreno's WG=64 wave size, vec4 fp16 ALUs, and image-cache texture path:**

- **`gemv_m1.cl`** — cooperative-WG=64 GEMV with vec4 fp16 + fp32 accumulator + `__local`-mem tree reduce. The decode-path workhorse, replacing CLBlast HGemm at every M=1 site (q/k/v/o-proj, gate/up/down-proj, lm_head). Hand-unrolled K-specialized variants (`_k768`, `_k1280`, `_k1536`) plus a generic-K **`_no4_coalesced`** path (vec4-strided across the warp for coalesced 256-byte DRAM reads) that hits 9–12 GB/s on Adreno 618. Adding the K=1280 specialization + coalesced generic to OpenELM-270M was the single largest unlock — **2.3× decode-throughput jump (4.60 → 10.59 tok/s)** before any other host-side work landed.
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

## 📊 Models

Decode tok/s = 5-run warm median (OpenELM-270M is 10-run), fp16, greedy, 32-token generation, measured 2026-05-07.

| Model | Params | Architecture | Device | Decode tok/s | TTFT (s) | Notes |
|---|---:|---|---|---:|---:|---|
| [Mamba2-130M](src/models/mamba2-130m/) | 130M | SSD | Razr 2020 (Adreno 618) | 23.18 | 1.53 | State-space duality |
| [Mamba-130M](src/models/mamba-130m/) | 130M | SSM | Razr 2020 (Adreno 618) | 22.15 | 1.60 | No attention |
| [SmolLM2-135M-Instruct](src/models/smollm2-135m-instruct/) | 135M | LLaMA + GQA | Razr 2020 (Adreno 618) | 23.65 | 1.53 | Instruct-tuned; 61% of ceiling |
| [OpenELM-270M](src/models/openelm-270m/) | 270M | LLaMA-style + tied lm_head | Razr 2020 (Adreno 618) | **14.81** (32 tok) / **15.22** (40 tok) | 1.93 | **3.31× over prior 4.47**; 78.9% of 10 GB/s ceiling at 40-token median |
| [LFM2.5-350M-Base](src/models/lfm2-5-350m/) | 350M | Hybrid conv+attn | Razr 2020 (Adreno 618) | 10.20 | 3.72 | Liquid AI hybrid |
| [Qwen2.5-0.5B](src/models/qwen2-5-0-5b/) | 500M | LLaMA + GQA | Razr 2020 (Adreno 618) | **10.41** | 2.59 | Largest in the repo; chained cl_mem decode + fused QKV → 70% of 14 GB/s ceiling |

## 🚀 Quickstart

```bash
git clone https://github.com/a8nova/adreno-llms.git
cd adreno-llms

# Pick any model — example: SmolLM2-135M-Instruct
./scripts/fetch_weights.sh smollm2-135m-instruct

cd src/models/smollm2-135m-instruct
NNOPT_DTYPE=fp16 ./scripts/build.sh --release
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "Once upon a time" 64
```

Tokens stream to stdout as they decode.

## 🎯 Hardware target

- **Verified:** Motorola Razr 2020, Adreno 618, Snapdragon 765G, Android 11+.
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
│   ├── fetch_openelm_weights.sh  # special: pulls + converts from apple/OpenELM-270M
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

## 🤖 How was this ported?

By **NNOpt**, a coding agent that takes a HuggingFace repo URL and a target device and produces a port like the ones in this repo — kernel selection, weight layout, fp32/fp16 paths, deploy scripts, benchmark harness, all of it. None of this was hand-written.

If you have a model you want running on Adreno, Snapdragon, Mali, or any Android device with this kind of polish, **email a8nova@gmail.com** for early access.

## 🤝 Contributing

## 📜 License

- **Code:** Apache 2.0 — see [LICENSE](LICENSE).
- **Weights:** carry their own upstream licenses, listed per-model:
  - mamba / mamba2 / Qwen2.5 / SmolLM2 — Apache 2.0
  - LFM2.5 — Liquid AI's open license
  - **OpenELM — Apple Sample Code License (NOT redistributed in this repo; fetch script pulls from Apple's HF repo)**

## 📧 Contact

For NNOpt early access, custom ports, or commercial licensing: **a8nova@gmail.com**.
