# Depth-Anything-V2-Small on Adreno (Android)

Depth Anything V2 (small) — monocular depth estimation, single RGB image → per-pixel depth — ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android. Verified on Motorola Razr 2020 (Adreno 620 / Snapdragon 765G). The first dense-vision model in this repo: no tokenizer, no decode loop — one forward pass over a 518×686 image produces the full depth map on-device at **~5.2 s/frame**, numerically matching PyTorch to cosine **0.99999**.

- **Upstream:** [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
- **Parameters:** ~25M
- **Architecture:** DINOv2 ViT-S encoder + DPT decoder head, dense vision forward pass
- **Precision:** fp16 (~48 MB bundle)

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch weights from HuggingFace (model.fp16.bin + meta; no tokenizer —
#    input is a preprocessed pixel tensor)
../../../scripts/fetch_weights.sh depth-anything-v2-small

# 2. Build (release — required for representative perf)
NNOPT_DTYPE=fp16 ./scripts/build.sh --release

# 3. Deploy binary + kernels + assets
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 4. Run on the bundled 518×686 test image (assets/test_pixel_values.bin)
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "Describe this image." 0 --temperature 0
# expect: DEPTH_MAP: elems=355348 min=0.808105 max=5.09375 mean=2.819 nan_inf=0
```

`scripts/cos_check.py` compares the device depth map against the PyTorch reference (`reference/predicted_depth.bin`, mean 2.8508) — measured cosine **0.999991**.

## Performance

Razr 2020 / Adreno 620 / Snapdragon 765G, fp16 release build, 518×686 input, measured 2026-07-07:

| Metric | Value |
|---|---:|
| **Warm** | **5.24 s/frame** |
| Single-shot (cold process) | ~5.8 s/frame |
| Peak CPU memory | 1042 MB |
| Cosine vs PyTorch | 0.999991 |

Token-LM metrics (prefill/decode tk/s, TTFT) don't apply to this dense forward pass — per-frame wall time is the latency metric. The optimization campaign took it **40.0 → 5.24 s/frame (7.6×)**: CLBlast on-disk program binary cache (one-time compile 3.3 s → 0.15 s), attention as per-head CLBlast GEMM on token-major slices, conv-as-GEMM via im2col, a stride==K conv_transpose specialization (10.7× on that op), and workgroup softmax. Every change has an env kill-switch (`NNOPT_ATTN_HEADLOOP/CONV_GEMM/CONVT_SK/IM2COL_V4/SOFTMAX_WG/CLBLAST_CACHE=0`). Peak CPU memory includes the im2col/CLBlast host buffers the campaign added; the pre-campaign port measured 215 MB at 40 s/frame. Full ledger in [BENCHMARK.md](./BENCHMARK.md).

> **Kernel layout note:** kernels are one-file-per-op, built as one `cl_program` per file. This is load-bearing on Adreno — a fat kernel in a shared program cuts wave occupancy for every kernel in it (see the "register footprint trap" in [BENCHMARK.md](./BENCHMARK.md)).

## Layout

```
.
├── BENCHMARK.md          # optimization ledger + per-kernel profile
├── CMakeLists.txt
├── assets/               # test_pixel_values.bin eval fixture
├── kernels/              # OpenCL kernels (.cl), one file per op
├── reference/            # PyTorch reference depth map
├── scripts/              # build / deploy / run / cos_check
├── src/                  # C++ sources
└── weights/              # fetched from HuggingFace (not committed)
```

## Licensing

- **Code:** Apache 2.0 (this repo).
- **Weights:** see the [model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf).
