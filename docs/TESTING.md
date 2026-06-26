# On-device reference testing

Two CI tiers:

| Tier | Where | What | Workflow |
| --- | --- | --- | --- |
| **Device-free** | GitHub-hosted (`ubuntu-latest`) | shellcheck lint + cross-compile every model for arm64 + self-contained link check | `.github/workflows/ci.yml` |
| **On-device** | **Self-hosted + Adreno phone** | deploy each model to a real device and assert its output matches a committed golden | `.github/workflows/device-tests.yml` |

On-device tests need a real Adreno GPU. GitHub-hosted runners and x86 emulators don't
have one, so the device suite runs only on a self-hosted runner with a phone attached
via `adb`.

## The golden-snapshot model

fp16 device output legitimately diverges from the PyTorch **fp32** reference
(`src/models/<key>/reference/reference_tokens.json`) after a few tokens — that's
accumulated fp16 error, not a bug (e.g. `mamba-130m` matches the fp32 reference for 4
tokens, then drifts; the *shipped* binary drifts identically). So the fp32 reference is
kept as **correctness documentation**, and the **CI gate is a golden snapshot of the
current known-good device output**: `tests/device_golden/<key>.json`. CI fails if a
future engine change alters a model's output — i.e. it catches *regressions*, which is
the point.

### Capturing / refreshing a golden

Run once, on a known-good build, with the phone attached, then commit the file:

```bash
# (re)capture one model's golden from the current device output
bash scripts/ci/run_device_reference.sh smollm2-135m-instruct --capture
git add tests/device_golden/smollm2-135m-instruct.json
```

Only refresh a golden when you have *intentionally* and *verifiably* changed a model's
output — otherwise you'd be rubber-stamping a regression.

### Running the test (what CI does)

```bash
bash scripts/ci/run_device_reference.sh smollm2-135m-instruct
# exit 0 = matches golden, 1 = deviates (regression) / error, 2 = modality not yet wired
```

Determinism: text/vlm/asr feed a fixed input (`reference/test_input_ids.bin` via
`--token-ids`) and force greedy decoding (`--top-k 1`), so a correct build is bit-stable
run to run. Flags: `--no-build` / `--no-deploy` skip the rebuild / adb push for fast
local iteration when the binary+weights are already on the device.

### Modality status

| Modality | Compare | Models | Status |
| --- | --- | --- | --- |
| text | token-id match (`_cmp_tokens.py`) | granite, lfm2-5-350m, mamba, mamba2, openelm, qwen2-5-0.5b, smollm2 | ✅ implemented · validated: smollm2, mamba |
| vlm | streamed caption text (`_cmp_vlm.py`) | lfm2-5-vl, smolvlm | ✅ implemented · validated: smolvlm |
| asr | `GENERATED_TEXT` match (`_cmp_text.py`) | whisper-tiny, seamless | ✅ implemented · validated: whisper-tiny |
| tts | windowed-RMS audio fingerprint | pocket-tts (out.bin), mms-tts/musicgen (output.wav) | ✅ implemented · validated: pocket-tts. kokoro/openvoice ⏳ (stdout-stream / voice-clone — need exec-out path) |
| music | windowed-RMS audio fingerprint | musicgen | ✅ wired (output.wav) · not yet captured |

Each modality is validated on-device for at least one model (golden committed). The rest
just need a one-time `--capture` on the runner. Models/modalities not yet wired exit `2`
(skipped, not failed), so the PR-blocking workflow is complete in shape and tightens as
goldens land.

**Goldens captured (12/16):** granite-4-0-350m, lfm2-5-350m, lfm2-5-vl-450m, mamba-130m,
mamba2-130m, mms-tts, musicgen-small, pocket-tts, qwen2-5-0-5b, smollm2-135m-instruct,
smolvlm-256m-instruct, whisper-tiny.

**Remaining (4) — need bespoke wiring before a golden can be captured:**
- `openelm-270m` — no `reference/test_input_ids.bin` + `reference_tokens.json` fixtures; generate them (see a sibling text model's `reference/_run_reference.py`).
- `seamless-m4t-unity-small` — translation/ASR with a different input contract (no `test_input_ids.bin`); wire its audio-in fixture.
- `kokoro-82m` — streams PCM to stdout (no on-device file); needs an `adb exec-out` capture path.
- `openvoice-v2` — voice cloning (needs reference + target speaker inputs); bespoke harness path.

For a model that *is* wired, capture is one command (with weights staged on the runner):
`bash scripts/ci/run_device_reference.sh <key> --capture` then commit `tests/device_golden/<key>.json`.

## Self-hosted runner setup (one-time)

The runner is a machine with the Adreno phone plugged in over USB (adb authorized) and
the model build toolchain installed.

**Prerequisites on the runner host**
- `adb` on `PATH`, the device authorized (`adb devices` shows it as `device`)
- Android NDK r26d (or set `ANDROID_NDK`); `python3`; the per-model build deps in
  `~/.nnopt/deps/opencl/...` (same as local builds)
- Model **weights** are fetched automatically — the workflow runs
  `scripts/fetch_weights.sh <model>` (public HF mirror, no auth, resumes/skips existing),
  so nothing to pre-stage. On a self-hosted runner the `_work` checkout persists, so only
  the first run per model downloads. Two gaps: `mms-tts` isn't on the mirror yet (upload
  via `upload_weights_to_hf.sh`), and `openelm-270m` uses `fetch_openelm_weights.sh`
  (Apple license) — but openelm is unwired/skipped anyway.

**Register the runner (GitHub → repo Settings → Actions → Runners → New self-hosted runner)**

```bash
# on the runner host, in a dedicated dir
mkdir actions-runner && cd actions-runner
# download + configure with the token GitHub shows you:
./config.sh --url https://github.com/<owner>/adreno-llms \
            --token <RUNNER_TOKEN> \
            --labels self-hosted,android
# run it (or install as a service):
./run.sh          # foreground
# ./svc.sh install && ./svc.sh start   # background service (recommended)
```

The workflow targets `runs-on: [self-hosted, android]`, so the `android` label is
required. With one phone, keep it a single runner — the workflow already pins
`max-parallel: 1` and a `concurrency` group so two jobs never touch the device at once.
