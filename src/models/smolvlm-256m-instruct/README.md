# SmolVLM-256M-Instruct on Adreno (Android)

HuggingFace's smallest vision-language model ported to C++/OpenCL for Adreno 6xx GPUs on Android. Bring-up running on Motorola Razr 2020 (Adreno 620 / Snapdragon 765G).

- **Upstream:** [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
- **Parameters:** 256M (vision tower + projector + 135M LM)
- **Architecture:** SigLIP-based vision encoder → linear projector → LLaMA-style LM with GQA
- **Precision:** fp16

This port is at the bring-up / perf-investigation stage — see `BENCHMARK.md` for the current decode/prefill numbers. Decode is the open optimization target.

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 0. One-time Python deps for the converter (skip if already installed)
pip install safetensors numpy torch huggingface_hub

# 1. Convert weights from upstream (SmolVLM bins aren't on the HF mirror yet,
#    so we pull safetensors + tokenizer.json from HuggingFaceTB/SmolVLM-256M-Instruct
#    and pack them into the nnopt layout. --rebake-pos-embed-384 resizes the
#    SigLIP position embedding from the upstream 32x32 grid to 24x24 for
#    IMAGE_SIZE=384.)
../../../scripts/convert_weights.py \
    --hf-repo-id HuggingFaceTB/SmolVLM-256M-Instruct \
    --out-dir weights \
    --rebake-pos-embed-384

# 2. Build (first run auto-invokes scripts/setup_deps.sh to fetch OpenCL
#    headers + pull libOpenCL.so from your connected Android device).
NNOPT_DTYPE=fp16 ./scripts/build.sh

# 3. Deploy
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 4. Run with an image + prompt
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "Describe this image." 64 --image fixtures/sample.jpg
```

## Layout

```
smolvlm-256m-instruct/
├── BENCHMARK.md          perf log (decode/prefill, optimizations)
├── CMakeLists.txt
├── src/                  C++ source (model, ops, vision tower, LM)
├── kernels/              OpenCL kernels
├── scripts/              build, deploy, run, quality-check helpers
├── fixtures/             sample image + qcheck data for quality regressions
├── reference/            ground-truth token ids and captions
└── weights/              fetched per-model (gitignored)
```

## License

Same license as adreno-llms. Upstream model weights are Apache-2.0 (SmolVLM).
