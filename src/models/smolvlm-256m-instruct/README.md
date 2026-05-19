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
# 1. Fetch weights (when published to the HF mirror; until then convert locally
#    from HuggingFaceTB/SmolVLM-256M-Instruct).
../../../scripts/fetch_weights.sh smolvlm-256m-instruct

# 2. Build
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
