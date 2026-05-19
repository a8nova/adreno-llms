# MODALITY: TTS (Text → Audio, single-shot VITS family)

You are porting a single-shot text-to-speech model (VITS architecture
family — MMS-TTS, Piper, YourTTS, etc.). The forward graph runs ONCE
per generation: text → expanded latents → waveform. No autoregressive
decode loop, no KV cache, no decode-step concept.

This block tells you (1) the seven forward-graph steps, (2) where each
one must be wired in C++, (3) the TTS-specific failure modes the
agent has hit historically, and (4) the Build-time gates that enforce
correctness pre-deploy.

## VITS forward graph — seven steps

```
INPUT
  input_ids        [B, T_chars]        int32   ← from assets/test_input_ids.bin
  duration_noise   [B, 2, T_chars]     float32 ← from assets/duration_noise.bin
  prior_noise      [B, 192, T_chars']  float32 ← from assets/prior_noise.bin

  Note: T_chars' may equal T_chars OR T_frames depending on transformers
  version. The reference template captures the EXACT shape; load the bin
  file with that shape; do not assume.

FORWARD
  (1) op_text_encoder(input_ids)
        → encoder_hidden    [B, T_chars, 192]
        → stats             [B, T_chars, 384]   (concat of prior_mean + prior_logvar)

  (2) op_duration_predictor(encoder_hidden, duration_noise)
        → log_durations     [B, T_chars]

  (3) HOST: host_compute_durations(log_durations, speaking_rate)
        → durations[T_chars]   int (ceil(exp(log_durations) * speaking_rate))
        → T_frames = sum(durations)
        → char_idx[T_frames]   int    (cumsum lookup table)

  (4) op_length_regulator(stats, char_idx, T_frames)
        → expanded_stats    [B, T_frames, 384]

  (5) op_sample_prior(expanded_stats, prior_noise, noise_scale)
        → z_prior           [B, 192, T_frames]   (channels-first after this point)

  (6) op_flow_inverse(z_prior)                                # 4 affine coupling layers
        → z_latent          [B, 192, T_frames]

  (7) op_vocoder(z_latent)
        → waveform          [B, T_audio]    where T_audio = T_frames * hop_length (256)
```

The C++ main writes the waveform to `output.wav` via `write_wav(...)`
at sample_rate 16000 (from `model.config.sampling_rate`).

## Forward-graph integration — THE failure mode VLM hit

VLM convergence was blocked for ~50 cycles on one bug:
`splice_image_tokens` was defined as a kernel + host wrapper but had
**zero callers in `model_forward_graph`**. The agent had emitted all
the code; the integration call was missing. See
[[feedback_vlm_splice_callsite]].

The TTS analog of this would be: emitting `op_vocoder` and `op_flow_inverse`
without calling them from `Model::forward_graph`. The Build-time gate
`tts_forward_graph_incomplete` (in `buildTs.ts`) refuses to build when
any of the seven `op_*` calls are absent from
`src/ops/backbone.cpp::model_forward_graph` (or `src/model.cpp` if your
scaffold uses that filename). You will see a refusal message listing
exactly which call is missing.

The canonical `Model::forward_graph` shape — copy it verbatim into your
backbone.cpp and only adjust naming if your scaffold differs:

```cpp
int Model::forward_graph(
    const std::vector<int32_t>& input_ids,
    const std::vector<float>&   duration_noise,
    const std::vector<float>&   prior_noise,
    std::vector<int16_t>&       out_pcm_int16) {

    const int T_chars = (int)input_ids.size();

    auto [encoder_hidden, stats] = op_text_encoder(input_ids, T_chars);
    NNOPT_LAYER_CHECK("text_encoder_out",   queue, encoder_hidden, T_chars * 192);
    NNOPT_LAYER_CHECK("text_encoder_stats", queue, stats,          T_chars * 384);

    cl_mem log_durations = op_duration_predictor(encoder_hidden, duration_noise, T_chars);
    NNOPT_LAYER_CHECK("log_durations", queue, log_durations, T_chars);

    std::vector<int> durations(T_chars);
    int T_frames = host_compute_durations(queue, log_durations, durations, speaking_rate_);

    cl_mem expanded_stats = op_length_regulator(stats, durations, T_chars, T_frames);
    NNOPT_LAYER_CHECK("expanded_stats", queue, expanded_stats, T_frames * 384);

    cl_mem z_prior  = op_sample_prior(expanded_stats, prior_noise, T_frames, noise_scale_);
    NNOPT_LAYER_CHECK("z_prior", queue, z_prior, T_frames * 192);

    cl_mem z_latent = op_flow_inverse(z_prior, T_frames);
    NNOPT_LAYER_CHECK("z_latent", queue, z_latent, T_frames * 192);

    cl_mem waveform_dev = op_vocoder(z_latent, T_frames);
    int T_audio = T_frames * 256;
    NNOPT_LAYER_CHECK("waveform", queue, waveform_dev, T_audio);

    std::vector<float> waveform_f32(T_audio);
    clEnqueueReadBuffer(queue, waveform_dev, CL_TRUE, 0,
                        T_audio * sizeof(float), waveform_f32.data(),
                        0, nullptr, nullptr);
    out_pcm_int16.resize(T_audio);
    for (int i = 0; i < T_audio; ++i) {
        float s = waveform_f32[i];
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
        out_pcm_int16[i] = (int16_t)(s * 32767.0f);
    }
    return 0;
}
```

The seven `NNOPT_LAYER_CHECK` calls map 1:1 to SxS dump points the
reference template publishes. If your code converges layer-by-layer in
this order, the WAV is correct.

## Fixture contract — load, never sample

VITS samples random tensors inside its forward. Reference Python and
C++ runtime MUST consume the same bytes; otherwise every cosine
compare past `text_encoder` is noise, not signal. This is the unique
TTS gotcha — it does not exist in any other modality.

Three files the reference publishes and the C++ runtime loads:

| File | Shape | dtype | Source |
|---|---|---|---|
| `assets/test_input_ids.bin` | `[T_chars]` | int32 | output of `AutoTokenizer(...).input_ids` |
| `assets/duration_noise.bin` | `[2 * T_chars]` (or whatever the predictor needs) | float32 | first `torch.randn` call captured during reference forward |
| `assets/prior_noise.bin`    | `[192 * T_chars]` or `[192 * T_frames]` | float32 | second RNG call captured |

In `main.cpp`:

```cpp
auto input_ids       = load_int32_bin("assets/test_input_ids.bin");
auto duration_noise  = load_float_bin("assets/duration_noise.bin");
auto prior_noise     = load_float_bin("assets/prior_noise.bin");
```

The Build gate `tts_main_missing_fixture_loads` refuses if any of these
three calls is absent from `src/main.cpp`. The gate
`tts_rng_in_runtime` refuses if `src/ops/duration_predictor.cpp` or
`src/ops/sample_prior.cpp` contains a host-side RNG call
(`std::normal_distribution`, `cl_rng`, etc.).

For first-port convergence: **never sample. Always load.** Production
RNG (re-seeded per-request) is a follow-up after convergence.

## Length regulation — the one host-device round-trip

Step (3) above runs on the CPU, not the GPU. Why: `T_frames` is
data-dependent. We can't allocate buffers for steps 4–7 until we know
how many frames the duration predictor wants. The flow:

1. After `op_duration_predictor` returns, `clEnqueueReadBuffer` the
   `log_durations[T_chars]` tensor back to host.
2. Compute `durations[i] = ceil(exp(log_durations[i]) * speaking_rate)`
   in plain C++ — small loop, ~3 ms total on Adreno 619.
3. Compute `T_frames = sum(durations)`.
4. Build `char_idx[T_frames]` lookup table (each frame's source character).
5. Upload `char_idx` to a small device buffer (~256 bytes typical).
6. Pass `T_frames` + `char_idx` into `op_length_regulator`.

DO NOT try to do this on the device. A naive single-kernel gather would
require a device-side cumsum which is awkward and slow on Adreno; the
host round-trip is faster AND simpler.

## HiFi-GAN vocoder — common bugs

The vocoder is the heavy hitter (~27M params of the 37M total) and the
source of most kernel-level bugs.

- **Forgetting `tanh` activation at the end.** HiFi-GAN's final Conv1d
  is followed by `tanh`. Omitting it produces cos ≥ 0.95 at the
  pre-tanh layer but audibly clipped/blown-out output. The
  `tanh_inplace` kernel exists in `hifigan_residual_block.cl` — call it.
- **Wrong upsample stride.** VITS-eng uses strides
  `[8, 8, 2, 2]` (cumulative 256 = hop_length). Other VITS variants
  use `[10, 8, 2, 2]` or similar. Read `model.config.upsample_rates`.
- **Bias added twice in dilated residual blocks.** Each
  HiFi-GAN ResBlock1 has 2 conv1ds per dilation branch; the SECOND
  conv1d has no bias (or its bias is folded). Read PyTorch source.
- **Channel-first vs channel-last layout.** VITS internally uses NCL
  (channels first). All `conv_1d` / `conv_transpose_1d` kernels in
  this scaffold assume NCL. Don't transpose unnecessarily.

## Benchmark + profiler — six preservation sites

When you restructure main.cpp (you probably won't for TTS, since
there's no prefill/decode split), preserve:

1. `bench.mark_inference_start()` — after arg parsing
2. `bench.mark_prefill_start()` — before `model.forward_graph(...)`
3. `NNOPT_BENCH_FIRST_TOKEN()` — repurposed as "first sample
   produced"; call inside the forward graph helpfully or once at
   start since there's no real first-token concept
4. `bench.mark_end()` — after `forward_graph` returns
5. `bench.print_summary(input_ids.size(), pcm.size())`
6. `KernelProfiler::dump_summary()` — immediately before #5

Acceptance: `grep "^BENCHMARK " run.log` returns 8 lines.
`NNOPT_PROFILE=1` re-run produces a `=== KERNEL PROFILE ===` table.

## WAV output contract — what Evaluate parses

The C++ binary must:

1. Write `output.wav` (path from `--out-wav` flag or the default
   `/data/local/tmp/<model>/output.wav`) as 16-bit signed mono PCM at
   16000 Hz, RIFF format. Use the `write_wav.h` template helper —
   don't hand-roll the header.
2. Emit on stderr (parsed by Evaluate):
   ```
   TTS_OUTPUT_PCM_SAMPLES <int>
   TTS_OUTPUT_SAMPLE_RATE 16000
   ```
3. Continue to emit `NNOPT_EXIT_CLEAN exit_code=0` before `return 0;`.

Evaluate computes `|T_audio_runtime - T_audio_reference| / T_audio_reference`
and refuses convergence if the ratio is > 0.05 (5% length drift means
the duration predictor diverged — independent signal from cosine
comparison). The Build gate `tts_audio_writer_missing` catches the
case where `forward_graph` succeeds but main.cpp forgets `write_wav`.

## REQUIRED — per-op `NNOPT_CHECKPOINT` for crash bisection

OpenCL kernel crashes on Adreno are usually async: `clEnqueueNDRangeKernel`
returns CL_SUCCESS but the GPU executes after the host moves on. When the
GPU finally faults it kills the device driver and the host sees nothing
specific — only `DEVICE CRASHED DURING INFERENCE`. The only signal we get
post-crash is "Binary made progress: N NNOPT_CHECKPOINT line(s)". Without
per-op checkpoints, that "N" is a useless 6 (just the init checkpoints).

**You MUST emit one `NNOPT_CHECKPOINT` line immediately before each op
call in `model_forward_graph`.** Pattern:

    NNOPT_CHECKPOINT("forward_graph: about to call op_text_encoder");
    op_text_encoder(...);
    NNOPT_LAYER_CHECK("text_encoder_out", queue, encoder_hidden, ...);

    NNOPT_CHECKPOINT("forward_graph: about to call op_duration_predictor");
    log_durations = op_duration_predictor(...);
    NNOPT_LAYER_CHECK("log_durations", queue, log_durations, ...);

    NNOPT_CHECKPOINT("forward_graph: about to call host_compute_durations");
    int T_frames = host_compute_durations(...);

    NNOPT_CHECKPOINT("forward_graph: about to call op_length_regulator");
    expanded_stats = op_length_regulator(...);
    NNOPT_LAYER_CHECK("expanded_stats", queue, expanded_stats, ...);

    NNOPT_CHECKPOINT("forward_graph: about to call op_sample_prior");
    z_prior = op_sample_prior(...);
    NNOPT_LAYER_CHECK("z_prior", queue, z_prior, ...);

    NNOPT_CHECKPOINT("forward_graph: about to call op_flow_inverse");
    z_latent = op_flow_inverse(...);
    NNOPT_LAYER_CHECK("z_latent", queue, z_latent, ...);

    NNOPT_CHECKPOINT("forward_graph: about to call op_vocoder");
    waveform_dev = op_vocoder(...);
    NNOPT_LAYER_CHECK("waveform", queue, waveform_dev, ...);

When a crash happens, the Infer log's "last checkpoint reached" line tells
you EXACTLY which op was running. Without these, every crash forces you
to guess between 6 ops, costing 5-6 extra cycles per bug.

## ANTI-PATTERN — do NOT fake non-zero outputs to bypass the safety gate

The runtime safety gate (`NNOPT_ABORT_ON_FAIL_LAYER=1`) terminates the
binary on FAIL_ZEROS / FAIL_NAN / FAIL_INF detected by an
`NNOPT_LAYER_CHECK`. Its purpose is to prevent broken op stubs from
feeding garbage into downstream OpenCL kernels and crashing the GPU
driver.

**It is NOT a checkbox to be defeated by filling buffers with constants.**

Do NOT write code like this in any op:

    // ANTI-PATTERN — DO NOT DO THIS
    const uint16_t tiny_u16 = 0x0001;
    clEnqueueFillBuffer(queue, stats, &tiny_u16, sizeof(tiny_u16), 0,
                        stats_bytes, 0, nullptr, nullptr);

That value (`half(5.96e-08)`) is non-zero so the gate passes, but the
downstream pipeline computes on garbage. The GPU then crashes inside a
later kernel where the math overflows or addresses are bogus — exactly
the scenario the gate was supposed to prevent.

If you are tempted to fill a buffer with a constant to "make progress",
**stop and implement the real op math instead.** Reference numerics
live in `reference/layers/*_output.bin`; use SxSDebug to compare your
real implementation against them. One more build-deploy-infer cycle
that converges costs far less than a device wedge from cascading
garbage.

The Build-time gate `op_fakes_layer_output` (buildTs.ts) refuses to
build when this anti-pattern is detected in `src/ops/*.cpp`.

## Debugging axis — RNG fixtures

If you see `cos > 0.99` at `text_encoder_out` but `cos < 0.5` at
`log_durations` or `z_prior` or anything downstream, the first
hypothesis is: **runtime is sampling its own noise instead of loading
the fixtures**. See `debug-pattern-reasoning.md` axis D. Test:
`grep -r "random\|randn\|normal_distribution\|cl_rng" src/ops/`. Any
match in `duration_predictor.cpp` or `sample_prior.cpp` is the bug.
Fix by routing the precomputed `assets/*_noise.bin` buffers into the
kernel as inputs.
