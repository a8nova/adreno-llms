# adreno-llms

> All code in this repo was ported and optimized by **NNOpt** — a coding agent for porting and optimizing neural networks for Android embedded targets. **Contact a8nova@gmail.com for early access.**

State-of-the-art small language models, hand-tuned for **Adreno 6xx GPUs on non-flagship Android phones**. Pure C++/OpenCL with CLBlast — no llama.cpp, no Vulkan, no NNAPI. Cross-compiled on macOS, deploys via `adb` to `/data/local/tmp/`.

## Models

Decode tok/s = 5-run warm median, fp16, greedy, 32-token generation on Motorola Razr 2020 (Adreno 618), measured 2026-05-06.

| Model | Params | Architecture | Decode tok/s | TTFT (s) | Notes |
|---|---:|---|---:|---:|---|
| [Mamba2-130M](src/models/mamba2-130m/) | 130M | SSD | 23.18 | 1.53 | State-space duality |
| [Mamba-130M](src/models/mamba-130m/) | 130M | SSM | 22.15 | 1.60 | No attention |
| [SmolLM2-135M-Instruct](src/models/smollm2-135m-instruct/) | 135M | LLaMA + GQA | 14.57 | 1.61 | Instruct-tuned |
| [LFM2.5-350M-Base](src/models/lfm2-5-350m/) | 350M | Hybrid conv+attn | 10.20 | 3.72 | Liquid AI hybrid |
| [Qwen2.5-0.5B](src/models/qwen2-5-0-5b/) | 500M | LLaMA + GQA | 8.45 | 2.59 | Largest in the repo; ~80% of memory ceiling |
| [OpenELM-270M](src/models/openelm-270m/) | 270M | LLaMA-style | 4.47 | 2.13 | Apple weights — fetch script only |

## Quickstart

```bash
git clone https://github.com/<you>/adreno-llms.git
cd adreno-llms

# Pick any model — example: SmolLM2-135M-Instruct
./scripts/fetch_weights.sh smollm2-135m-instruct

cd src/models/smollm2-135m-instruct
NNOPT_DTYPE=fp16 ./scripts/build.sh --release
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "Once upon a time" 64
```

Tokens stream to stdout as they decode.

## Hardware target

- **Verified:** Motorola Razr 2020, Adreno 618, Snapdragon 765G, Android 11+.
- **Should work:** any arm64-v8a Android device with Adreno 6xx-class GPU (610, 612, 618, 619, 620, 630, 640, 650, 660, 680, 690, 695). Performance characteristics vary — the kernels are tuned around 768-thread Adreno 6xx wave behavior.
- **Probably works (untested):** Adreno 7xx flagships should run; numbers will be higher but kernels aren't tuned for them.
- **Won't hit these numbers:** Mali, PowerVR, Apple GPUs. The OpenCL code will run but the optimizations won't transfer.

## Prerequisites

| Dependency | Version | How to get it |
|---|---|---|
| Android NDK | r25+ (any recent) | `sdkmanager 'ndk;26.1.10909125'` or Android Studio SDK Manager |
| CMake | 3.18+ | `brew install cmake` |
| OpenCL headers | 1.2 (Khronos) | Auto-fetched by build into `~/.nnopt/deps/opencl/` |
| CLBlast | 1.6.3 | Pinned via CMake `FetchContent` — built on first `build.sh` |
| Python | 3.10+ | For tokenizer parity checks |
| `adb` | platform-tools | `brew install --cask android-platform-tools` |

Cross-compiled on macOS (canonical path). Linux host should work — `build.sh` is bash-portable, the Android NDK toolchain is host-agnostic — but is currently untested.

## Tokenizer correctness

Each model ships a Python reference script at `src/models/<m>/reference/_run_reference.py`. Run it to capture HuggingFace's expected output, then diff against the C++ binary's stdout.

| Model | Reference parity script | Test inputs | Status |
|---|---|---|---|
| smollm2-135m-instruct | ✓ | ✓ | Re-runnable |
| mamba-130m | ✓ | ✓ | Re-runnable |
| mamba2-130m | ✓ | ✓ | Re-runnable |
| openelm-270m | — | — | **Not currently re-runnable** (regenerate locally if needed) |
| lfm2-5-350m | ✓ | ✓ | Re-runnable |
| qwen2-5-0-5b | ✓ | ✓ | Re-runnable |

## Repository layout

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

## How was this ported?

By **NNOpt**, a coding agent that takes a HuggingFace repo URL and a target device and produces a port like the ones in this repo — kernel selection, weight layout, fp32/fp16 paths, deploy scripts, benchmark harness, all of it. None of this was hand-written.

If you have a model you want running on Adreno, Snapdragon, Mali, or any Android device with this kind of polish, **email a8nova@gmail.com** for early access.

## Contributing

## License

- **Code:** Apache 2.0 — see [LICENSE](LICENSE).
- **Weights:** carry their own upstream licenses, listed per-model:
  - mamba / mamba2 / Qwen2.5 / SmolLM2 — Apache 2.0
  - LFM2.5 — Liquid AI's open license
  - **OpenELM — Apple Sample Code License (NOT redistributed in this repo; fetch script pulls from Apple's HF repo)**

## Contact

For NNOpt early access, custom ports, or commercial licensing: **a8nova@gmail.com**.
