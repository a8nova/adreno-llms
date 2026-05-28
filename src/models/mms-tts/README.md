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

## Sample runs

10 sentences per language, synthesized on Motorola Razr 2020 (Adreno 620). Audio files in [`samples/`](samples/).

### English

| # | Text | RTF | Audio (s) | Wall (s) |
|--:|------|----:|----------:|----------:|
| 1 | Hello, my name is Sarah and I live in New York. | 1.58 | 3.57 | 5.64 |
| 2 | The weather today is absolutely beautiful. | 1.53 | 3.04 | 4.67 |
| 3 | Can you please tell me how to get to the train station? | 1.54 | 3.31 | 5.11 |
| 4 | Once upon a time, in a land far away, there lived a young princess. | 1.51 | 4.21 | 6.35 |
| 5 | I would like to order a large coffee with milk please. | 1.58 | 3.58 | 5.66 |
| 6 | The quick brown fox jumps over the lazy dog. | 1.54 | 3.68 | 5.65 |
| 7 | Technology has changed the way we communicate with each other. | 1.59 | 4.56 | 7.25 |
| 8 | Good morning everyone, welcome to the presentation. | 1.53 | 3.31 | 5.06 |
| 9 | She opened the door slowly and peeked inside the dark room. | 1.54 | 4.08 | 6.28 |
| 10 | Mathematics is the language of the universe. | 1.59 | 2.98 | 4.74 |

**Average RTF: 1.55** (min 1.51, max 1.59). Peak memory: 528 MB.

### Amharic (አማርኛ)

| # | Text | RTF | Audio (s) | Wall (s) |
|--:|------|----:|----------:|----------:|
| 1 | ሰላም፣ እንደምን አደርክ? | 1.49 | 2.18 | 3.24 |
| 2 | ኢትዮጵያ ውብ ሀገር ናት። | 1.52 | 2.35 | 3.59 |
| 3 | ዛሬ የአየር ሁኔታው በጣም ጥሩ ነው። | 1.56 | 2.85 | 4.45 |
| 4 | ወደ ትምህርት ቤት እየሄድኩ ነው። | 1.59 | 2.75 | 4.39 |
| 5 | ቡና ማግኘት እፈልጋለሁ። | 1.49 | 2.70 | 4.02 |
| 6 | አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት። | 1.50 | 4.11 | 6.17 |
| 7 | ልጆቹ በአትክልት ቦታ ይጫወታሉ። | 1.59 | 3.12 | 4.96 |
| 8 | መጽሐፍ ማንበብ በጣም ጠቃሚ ነው። | 1.59 | 2.48 | 3.94 |
| 9 | እባክዎ ይህንን ያስረዱኝ። | 1.53 | 2.48 | 3.80 |
| 10 | ሙዚቃ ማዳመጥ ደስ ይለኛል። | 1.50 | 2.93 | 4.39 |

**Average RTF: 1.54** (min 1.49, max 1.59). Peak memory: 498 MB.

### Arabic (العربية)

| # | Text | RTF | Audio (s) | Wall (s) |
|--:|------|----:|----------:|----------:|
| 1 | مرحبا، كيف حالك اليوم؟ | 1.41 | 3.92 | 5.54 |
| 2 | الطقس جميل جدا في هذا اليوم. | 1.45 | 3.82 | 5.53 |
| 3 | أريد أن أطلب فنجان قهوة كبير من فضلك. | 1.45 | 5.47 | 7.95 |
| 4 | التكنولوجيا غيرت الطريقة التي نتواصل بها. | 1.43 | 5.04 | 7.23 |
| 5 | كان يا ما كان في قديم الزمان. | 1.43 | 3.42 | 4.89 |
| 6 | العلم نور والجهل ظلام. | 1.42 | 3.33 | 4.73 |
| 7 | أحب القراءة والسفر واكتشاف أماكن جديدة. | 1.47 | 5.57 | 8.17 |
| 8 | صباح الخير، أتمنى لكم يوما سعيدا. | 1.41 | 4.70 | 6.63 |
| 9 | الرياضيات هي لغة الكون. | 1.45 | 3.18 | 4.60 |
| 10 | فتحت الباب ببطء ونظرت إلى الداخل. | 1.48 | 5.18 | 7.66 |

**Average RTF: 1.44** (min 1.41, max 1.48). Peak memory: 607 MB.

## Languages

Any [MMS-TTS language code](https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html) — Latin-script (eng, deu, fra, spa, ...) and non-Latin (amh, ara, khm, tha, ...) both work. Non-Latin scripts go through uroman romanization before the on-device char tokenizer; uroman tables ship in `assets/uroman/` (loaded once at process start).

### Pre-built packs (used by the see-and-say example)

Ready-to-download packs (~69 MB each: `model.fp16.bin` + `tokenizer_vocab.bin` + metadata) live under the `mms-tts/` prefix of the project's weights repo:

  https://huggingface.co/a8nova/adreno-llms-weights/tree/main/mms-tts

The see-and-say Android picker lists every language with a pack present there. Languages without a pack stay greyed out until one is uploaded.

### Adding a language not yet in the dataset

1. **Convert + zip locally.** From this directory:
   ```bash
   python3 scripts/convert_all_languages.py --codes <code>
   ```
   This calls `prep_lang.py <code>` under the hood, zips the runtime files into `packs/mms-tts-<code>.zip`, and regenerates `packs/languages.json`. (For bulk conversion, drop `--codes` to walk the full 1140-code list in `scripts/mms_tts_languages.json` — it's resumable and skips codes whose zip already exists.)

2. **Upload to HF.** One-time `huggingface-cli login`, then:
   ```bash
   huggingface-cli upload a8nova/adreno-llms-weights packs/ mms-tts/
   ```
   (No `--repo-type` flag — `adreno-llms-weights` is the same model repo that hosts all the other ports' weights; `mms-tts/` is the destination subfolder inside it.)

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
