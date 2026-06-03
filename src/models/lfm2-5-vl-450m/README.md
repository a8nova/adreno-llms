# LFM2.5-VL-450M on Adreno (Android)

Liquid AI's compact multimodal foundation model, ported to C++/OpenCL for Adreno 6xx GPUs on non-flagship Android devices. End-to-end vision-language inference on a 1.0 GHz mobile GPU.

- **Upstream:** [LiquidAI/LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)
- **Parameters:** 450M (66M SigLIP-2 vision tower + 16M projector + 350M LFM2 hybrid LM)
- **Architecture:**
  - **Vision tower:** SigLIP-2 base, 12 bidirectional transformer layers, hidden=768, 12 heads, head_dim=64. Multi-tile image splitting (up to 10× 512×512 tiles + thumbnail). Bilinear NaFlex position embedding.
  - **Multimodal projector:** pixel-unshuffle 2×2 → linear (3072→2048) → GELU(erf) → linear (2048→1024).
  - **Language model:** LFM2.5-350M hybrid (10 short-conv layers + 6 full-attention layers, GQA 16/8, RoPE θ=1e6, RMSNorm, tied lm_head, vocab=65536).
- **Precisions:** fp16, int8 (per-row symmetric)

## Choose a variant

3-run median, greedy (`--temperature 0`), 30-token caption of `fixtures/sample.jpg` (1024×704 black-and-white cat).

| Variant | TTFT (s) — Razr 2020 / Adreno 620 | Decode tok/s | Weight file | Peak CPU mem | Quality |
|---|---:|---:|---:|---:|---|
| **fp16** | 128.6 | 10.0 | 856 MB | 2.20 GB | Reference (greedy match w/ HF processor + custom LM) |
| **int8** | 130.4 | 9.7 | **428 MB** | **2.07 GB** | **Byte-for-byte first 7 generated tokens match PyTorch fp32 reference** (quantization noise canceled fp16 drift) |

The int8 variant is the recommended default — it cuts disk by half, peak GPU memory by 130 MB, and produces caption tokens that match the upstream PyTorch reference byte-for-byte for the first ~7 tokens.

## Quickstart

From this directory, with an Android device connected over `adb`:

```bash
# 1. Fetch weights from HuggingFace (incremental; safe to re-run with a
#    different --quant to add another bundle alongside the existing one).
../../../scripts/fetch_weights.sh lfm2-5-vl-450m                # fp16 only (862 MB)
../../../scripts/fetch_weights.sh lfm2-5-vl-450m --quant int8   # fp16 + int8 (1.30 GB)

# 2. Build (single Release binary, both variants supported at runtime)
NNOPT_DTYPE=fp16 ./scripts/build.sh --release

# 3. Deploy binary + every weight bundle that exists locally
NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh

# 4. Run with the fixture image
NNOPT_DTYPE=fp16 ./scripts/run_android.sh "Describe this image." 30 \
    --image fixtures/sample.jpg --temperature 0 --seed 42
```

Expected output (int8, deterministic):

```
GENERATED_TEXT: This photograph captures a striking black and white cat with
piercing green eyes, gazing directly at the camera. The cat's fur is
predominantly white,
```

### Re-generating the int8 bundle locally

If you have the fp16 bundle and would rather not download the int8 file, regenerate it (~10 s, needs Python 3 + numpy):

```bash
python3 scripts/quantize_int8.py    # → weights/model.int8.bin + .meta.json
```

The script reads `weights/model.fp16.bin` and writes the new bundle alongside. Selective scheme: every 2-D weight matrix ≥64 KB gets quantized; biases, LayerNorm/RMSNorm gammas, and tiny lookup tables stay fp16.

## What runs where

- **GPU (OpenCL on Adreno):** image normalize + patchify, full SigLIP-2 encoder (LayerNorm, Q/K/V/out projections with bias, bidirectional tiled flash attention, MLP fc1/fc2 with GELU-tanh), bilinear position resize, pixel-unshuffle, projector (linear + GELU-erf + linear), `masked_scatter` splice, all LM layers (RMSNorm, conv blocks, full-attention blocks with RoPE, MLP w1/w3/w2, fused residual+norm, lm_head GEMV, on-GPU argmax for greedy decode).
- **CPU:** multi-tile image splitting (smart_resize + crop_image_to_patches via stb_image_resize2), chat-template construction with image-placeholder expansion, kernel orchestration.

For int8, weights are stored quantized on disk and in GPU memory; the GEMV kernels read them as `image2d_t` bytes (via `cl_khr_image2d_from_buffer`), multiply against per-row fp16 scales, accumulate in fp32, and store fp16 outputs. **At M=1 (decode) CLBlast is bypassed entirely** — custom kernels in `kernels/gemv_m1_int8_image.cl` handle the path. At M>1 (prefill) int8 weights are dequantized to an fp16 scratch buffer before CLBlast HGemm; on the SigLIP side weights stay on-device through the pipeline.

## Verifying byte-for-byte against PyTorch

The reference dumps (`reference/capture_vision_dumps.py`) generate fp32 PyTorch outputs for every intermediate tensor (patch_embedding, post-position-embedding, encoder layers 0/5/11, post-LayerNorm, projector per-tile in/out, final merged `inputs_embeds`). They're produced by:

```bash
# Needs Python 3.11 + transformers 4.57.6 + torchvision 0.17 + numpy<2
python3 reference/capture_vision_dumps.py
```

The tokenizer chat-template + image-token expansion produces 1797 input IDs that match the HF processor byte-for-byte for the fixture. The int8 generated tokens match the PyTorch fp32 greedy reference for the first 7 tokens, then drift slightly due to fp16 numerical accumulation across 12 vision-tower layers.

## Performance — Razr 2020 / Adreno 620

Greedy (`--temperature 0`), seed 42, prompt `"Describe this image."` with `fixtures/sample.jpg` (1024×704), 30-token caption, 3-run warm median measured 2026-06-02.

| Variant | TTFT (s) | Decode tok/s | Total 30-tok (s) | Peak CPU mem (MB) |
|---|---:|---:|---:|---:|
| fp16 | 128.6 | 10.0 | 131.6 | 2197 |
| int8 | **130.4** | 9.7 | **133.5** | **2071** |

### TTFT breakdown (int8, profiled, 130 s)

| Phase | Time | % | Notes |
|---|---:|---:|---|
| CPU image processor (resize + tile split) | ~50 ms | 0.04% | stb_image_resize2 triangle filter |
| Vision tower (7 tiles × 12 SigLIP layers) | ~95 s | 73% | Bidirectional N²=1024² flash attention dominates |
| Multimodal projector | ~0.5 s | 0.4% | 7 per-tile linear-GELU-linear stacks |
| `masked_scatter` splice | ~40 ms | <0.1% | 1783 per-position fp16 row copies |
| LM prefill (1797 tokens × 16 layers) | ~25 s | 19% | Flash attention + CLBlast HGEMM |
| lm_head + GPU argmax | <50 ms | <0.1% | Tiled image2d no8 GEMV + 2-pass argmax |
| Misc (host overhead, sync) | ~9 s | 7% | Unprofilable directly |

### Where the time is actually going

The SigLIP bidirectional attention kernel runs 84 times (12 layers × 7 tiles) at ~840 ms each — that's **76% of GPU time** and about 78% of the Adreno 620 fp16 ALU peak (already close to roofline). The remaining ~32 s of unprofiled time is CLBlast HGEMM for the vision projections (Q/K/V/out/fc1/fc2 per layer per tile) — the int8 dequant step adds ~3.6 s here.

The decode hot path is at 9.7 tok/s = 6.5 GB/s effective bandwidth = 46% of the practical 14 GB/s streaming ceiling on this device.

## Caveats / known gaps

- **Prefill is slow.** At 130 s TTFT this model is not suitable for interactive captioning on Adreno 620 — the SigLIP encoder dominates and is ALU-bound. On Adreno 660+ (2 SPs, faster fp16 ALU) the same code typically runs 2–3× faster; ports for that class of device are out of scope here.
- **int8 prefill uses dequant-to-fp16 + CLBlast.** A custom int8-aware HGEMM kernel would skip the dequant pass and shave 25–35 s off TTFT but is significant additional work (~600 LOC).
- **Single image, single turn.** Multi-turn conversations and multi-image inputs are not wired up. The tokenizer only emits the single-user-turn-with-one-image format.
- **No KV cache across queries.** Each new query re-runs the full vision tower. Caching `image_features` per (image hash, projector version) would make follow-up text-only turns instant.
- **fp16 numerical drift in vision tower.** Compared to the PyTorch fp32 reference, the first 7 decode tokens match byte-for-byte under int8 (quant noise happens to cancel fp16 drift); under fp16 they diverge by token 1 but stay semantically valid.

## Files of note

- `src/ops/op_siglip_encoder.cpp` — SigLIP encoder (per-tile and batched paths) + projector
- `src/ops/lfm2_vl_image_processor.cpp` — multi-tile image processor (smart_resize, crop_image_to_patches, thumbnail)
- `src/tokenizer.cpp` — chat-template + image-placeholder expansion in `encode_with_image()`
- `src/ops/op_lfm2_full_attention_block.cpp` — LM full-attention block with tiled flash attention prefill (`lfm2_flash_attn_prefill` and `lfm2_flash_attn_decode` kernels)
- `src/ops/op_lfm2_conv_block.cpp` — LM short-conv block with cache
- `kernels/lfm2_ops.cl` — primary kernel file (LayerNorm, RMSNorm, swiglu, GELU, attention, pixel-unshuffle, bilinear pos resize, …)
- `kernels/siglip_attn_image.cl` — image2d-backed bidirectional flash attention (separate program for register isolation)
- `kernels/gemv_m1_int8_image.cl` — int8 image2d-backed GEMVs for decode (separate program)
- `kernels/gemv_m1_image.cl` — fp16 image2d-backed GEMVs for decode
- `kernels/argmax.cl` — 2-pass GPU argmax for greedy decode
- `scripts/quantize_int8.py` — fp16→int8 weight quantizer
- `reference/capture_vision_dumps.py` — PyTorch reference dump generator for byte-for-byte validation
