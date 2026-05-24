# MMS-TTS on Adreno (Android)

Facebook's [MMS-TTS](https://huggingface.co/facebook/mms-tts) family ported to C++/OpenCL for Adreno 6xx GPUs on Android. Verified end-to-end on Motorola Razr 2020 (Adreno 620 / Snapdragon 765G) for **English** and **Amharic**, but the same binary handles any of the ~1100 MMS languages — only the per-language weights and vocab swap.

- **Upstream:** [facebook/mms-tts-\<lang\>](https://huggingface.co/facebook/mms-tts) (`facebook/mms-tts-eng`, `facebook/mms-tts-amh`, …)
- **Architecture:** VITS — text encoder (6 transformer layers, hidden=192) + stochastic duration predictor + residual coupling flow + HiFi-GAN decoder
- **Precision:** fp16

Inference is **100% on-device C++**. The Python script in `scripts/prep_lang.py` only runs once on your laptop to convert HuggingFace weights into the on-device `model.fp16.bin` / `tokenizer_vocab.bin` format. Same role `scripts/fetch_weights.sh` plays for other models in this repo.

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 0. One-time Python deps for the offline weight conversion step
pip install huggingface_hub transformers safetensors torch numpy

# 1. One-time per language: download from HF and convert to the on-device
#    format. Writes weights/<lang>/{model.fp16.bin, model.fp16.meta.json,
#    tokenizer_vocab.bin} and assets/<lang>/{test_input_ids, *_noise}.bin.
python3 scripts/prep_lang.py eng "Hello, my name is."
python3 scripts/prep_lang.py amh "ሰላም፣ እንደምን አደርክ?"

# 1b. ONLY for non-Latin scripts (amh, ara, khm, tha, …): fetch the uroman
#     romanization tables into assets/uroman/. Latin-script langs skip this.
./scripts/fetch_uroman.sh

# 2. Build (first run auto-invokes scripts/setup_deps.sh to fetch OpenCL
#    headers + pull libOpenCL.so from your connected Android device).
#    Single binary handles all languages at runtime via --lang.
NNOPT_DTYPE=fp16 ./scripts/build.sh

# 3. Deploy. Pushes the binary + every weights/<lang>/ subdir that
#    prep_lang.py has populated + assets/<lang>/ + assets/uroman/.
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 4. Run — text goes in, /tmp/tts_out.wav comes out, plays automatically
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "Hello, my name is."        1 --lang eng && afplay /tmp/tts_out.wav
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "ሰላም፣ እንደምን አደርክ?" 1 --lang amh && afplay /tmp/tts_out.wav
```

The on-device binary does tokenization (uroman → char vocab), the full VITS forward graph, and WAV output. No Python at runtime.

## Per-op verification

Element-wise cosine vs HuggingFace Python reference (`NNOPT_REF_TEST=1`, captured on Razr 2020 fp16):

| op                  | cosine     | notes |
|---------------------|------------|-------|
| `text_encoder`      | 1.000000   | embed + 6× transformer + project |
| `duration_predictor`| 1.000000   | real `VitsStochasticDurationPredictor` inverse pass, controlled noise |
| `sample_prior`      | 1.000000   | affine reparameterization |
| `flow_inverse`      | 1.000000   | 4× residual coupling, weight_norm WaveNet |
| `vocoder`           | 0.996865   | HiFi-GAN; remainder is fp16 quantization noise |

Run the matrix any time with:
```bash
NNOPT_REF_TEST=1 NNOPT_DTYPE=fp16 ./scripts/run_android.sh "x" 1
```

## Languages

Any [MMS-TTS language code](https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html) — Latin-script (eng, deu, fra, spa, ...) and non-Latin (amh, ara, khm, tha, ...) both work. Non-Latin scripts go through uroman romanization before the on-device char tokenizer; uroman tables ship in `assets/uroman/` (loaded once at process start).

### Pre-built packs (used by the see-and-say example)

Ready-to-download packs (~69 MB each: `model.fp16.bin` + `tokenizer_vocab.bin` + metadata) live in this HF dataset:

  https://huggingface.co/datasets/a8nova/mms-tts-language-packs

The see-and-say Android picker lists every language with a pack present there. Languages without a pack stay greyed out until one is uploaded.

### Adding a language not yet in the dataset

1. **Convert + zip locally.** From this directory:
   ```bash
   python3 scripts/convert_all_languages.py --codes <code>
   ```
   This calls `prep_lang.py <code>` under the hood, zips the runtime files into `packs/mms-tts-<code>.zip`, and regenerates `packs/languages.json`. (For bulk conversion, drop `--codes` to walk the full 1140-code list in `scripts/mms_tts_languages.json` — it's resumable and skips codes whose zip already exists.)

2. **Upload to HF.** One-time `huggingface-cli login`, then:
   ```bash
   huggingface-cli upload a8nova/mms-tts-language-packs packs/ . --repo-type dataset
   ```

The picker UI in the app picks up the new language on next launch — no code change, no app release.

## Layout

```
mms-tts/
├── CMakeLists.txt
├── src/                  C++ source (tokenizer, model, ops, vocoder, ...)
│   └── ops/              text_encoder, duration_predictor, flow_inverse, vocoder, ...
├── kernels/              OpenCL kernels (.cl)
├── scripts/
│   ├── build.sh          NDK cross-compile
│   ├── deploy_android.sh adb push binary + weights/<lang>/ + assets/<lang>/
│   ├── run_android.sh    adb shell run + pull output.wav
│   ├── prep_lang.py      one-time HF → on-device weight converter
│   ├── bench.sh          locked perf harness
│   └── cos_check.py      offline layer-by-layer comparison vs reference dumps
├── reference/            small ground-truth fixtures for NNOPT_REF_TEST
└── weights/              fetched per-language by prep_lang.py (gitignored)
    └── <lang>/           model.fp16.bin, tokenizer_vocab.bin, model.fp16.meta.json
```

## License

- This port's code: same license as adreno-llms.
- `facebook/mms-tts-*` weights: **CC-BY-NC 4.0**. The `prep_lang.py` script downloads to your machine using your HF auth; no weights are committed to or redistributed from this repo.
- `assets/uroman/` data: MIT (isi-nlp/uroman); preserve their LICENSE in any distribution.
