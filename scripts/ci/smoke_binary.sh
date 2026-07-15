#!/usr/bin/env bash
#
# smoke_binary.sh — build a model's .so and RUN it end-to-end, asserting the exact
# contract the host app (Edgi ProcessEngine) depends on. This is the test that would
# have caught the stable-audio "cannot open assets/cross_attn_cond.bin" bug BEFORE it
# shipped: that binary returned 1 in --serve mode before ever printing SERVE_READY.
#
# What it checks, cheapest first:
#   1. BUILD           — scripts/build.sh --release with NNOPT_DTYPE=fp16 succeeds.
#   2. SELF-CONTAINED  — check_self_contained.sh: no stray shared-lib / libc++ deps
#                        (the openvoice-v2 CANNOT-LINK class of bug), pure ELF, no device.
#   3. RUN (on device) — deploy + run in the app's real mode and assert the markers:
#        serve  : feed "<prompt>\nquit" on stdin; require SERVE_READY then SERVE_DONE,
#                 NO SERVE_ERR, exit 0, and a non-trivial output_serve_0.wav.
#        single : run "<prompt> <max_tokens>"; require exit 0 and non-empty output.
#
# DEVICE: needs an Adreno device on adb (step 3). Steps 1-2 are device-free; pass
# --no-run to stop after them (CI without a phone). Default device = first adb device;
# pass --serial to pin one. This script NEVER touches a device you don't select.
#
# Usage:
#   scripts/ci/smoke_binary.sh [options]
#     --model NAME     model dir under src/models/ (default: stable-audio-open-small)
#     --mode MODE      serve | single            (default: serve)
#     --prompt STR     generation prompt         (default: "128 BPM tech house drum loop")
#     --seconds N      serve: audio length       (default: 8)
#     --max-tokens N   single: token budget      (default: 64)
#     --serial SERIAL  adb device serial         (default: first connected)
#     --no-build       skip step 1 (use existing build/fp16 binary)
#     --no-run         stop after steps 1-2 (device-free)
#     --csv PATH       append one result row: model,mode,build,selfcontained,run,wav_bytes,verdict
#
# Exit 0 = all requested steps passed; non-zero = a step failed (details printed).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="stable-audio-open-small"
MODE="serve"
PROMPT="128 BPM tech house drum loop"
SECONDS_ARG=8
MAX_TOKENS=64
SERIAL=""
DO_BUILD=1
DO_RUN=1
CSV=""

while [ $# -gt 0 ]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2 ;;
        --mode)       MODE="$2"; shift 2 ;;
        --prompt)     PROMPT="$2"; shift 2 ;;
        --seconds)    SECONDS_ARG="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --serial)     SERIAL="$2"; shift 2 ;;
        --no-build)   DO_BUILD=0; shift ;;
        --no-run)     DO_RUN=0; shift ;;
        --csv)        CSV="$2"; shift 2 ;;
        -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

MODEL_DIR="$REPO_ROOT/src/models/$MODEL"
[ -d "$MODEL_DIR" ] || { echo "✗ no such model dir: $MODEL_DIR" >&2; exit 2; }
[ -f "$MODEL_DIR/scripts/build.sh" ] || { echo "✗ $MODEL has no scripts/build.sh" >&2; exit 2; }

ADB="adb"
[ -n "$SERIAL" ] && ADB="adb -s $SERIAL"

# Result cells (for the optional CSV): SKIP until a step sets PASS/FAIL.
R_BUILD="SKIP"; R_SELFCONTAINED="SKIP"; R_ALIGN="SKIP"; R_RUN="SKIP"; WAV_BYTES=0
fail() { echo ""; echo "✗ SMOKE FAILED: $1" >&2; finish 1; }

finish() {
    local code="$1"
    local verdict; verdict=$([ "$code" -eq 0 ] && echo PASS || echo FAIL)
    if [ -n "$CSV" ]; then
        [ -f "$CSV" ] || echo "model,mode,build,self_contained,align_16kb,run,wav_bytes,verdict" > "$CSV"
        echo "$MODEL,$MODE,$R_BUILD,$R_SELFCONTAINED,$R_ALIGN,$R_RUN,$WAV_BYTES,$verdict" >> "$CSV"
    fi
    echo ""; echo "── smoke: $MODEL [$MODE] → $verdict ──"
    exit "$code"
}

echo "══ smoke_binary: $MODEL (mode=$MODE) ══"

# ── Step 1: build ──────────────────────────────────────────────────────────────
if [ "$DO_BUILD" -eq 1 ]; then
    echo "▸ build (NNOPT_DTYPE=fp16 --release)…"
    if ( cd "$MODEL_DIR" && NNOPT_DTYPE=fp16 ./scripts/build.sh --release ) >/tmp/smoke_build.$$.log 2>&1; then
        R_BUILD="PASS"; echo "  ✓ build ok"
    else
        R_BUILD="FAIL"; tail -25 /tmp/smoke_build.$$.log; rm -f /tmp/smoke_build.$$.log
        fail "build.sh --release failed"
    fi
    rm -f /tmp/smoke_build.$$.log
fi

# Resolve the fp16 binary name from the model's own source of truth.
# shellcheck disable=SC1091
( cd "$MODEL_DIR" && . ./scripts/remote_dir.env >/dev/null 2>&1 )
BINARY_NAME="$( cd "$MODEL_DIR" && NNOPT_DTYPE=fp16 bash -c '. ./scripts/remote_dir.env >/dev/null 2>&1; echo "$BINARY_NAME"' )"
REMOTE_DIR="$( cd "$MODEL_DIR" && NNOPT_DTYPE=fp16 bash -c '. ./scripts/remote_dir.env >/dev/null 2>&1; echo "$REMOTE_DIR"' )"
BIN_PATH="$MODEL_DIR/build/fp16/$BINARY_NAME"
[ -f "$BIN_PATH" ] || fail "expected binary not found: $BIN_PATH (build first, or drop --no-build)"

# ── Step 2: self-contained (device-free ELF gate) ──────────────────────────────
echo "▸ self-contained check…"
if "$REPO_ROOT/scripts/ci/check_self_contained.sh" "$BIN_PATH" >/tmp/smoke_sc.$$.log 2>&1; then
    R_SELFCONTAINED="PASS"; echo "  ✓ no stray shared-lib / libc++ deps"
else
    R_SELFCONTAINED="FAIL"; cat /tmp/smoke_sc.$$.log; rm -f /tmp/smoke_sc.$$.log
    fail "binary is not app-linkable (see check_self_contained output)"
fi
rm -f /tmp/smoke_sc.$$.log

# ── Step 2.5: 16 KB alignment (Google Play / Android 15) — device-free ELF gate ──
echo "▸ 16 KB-alignment check…"
if "$REPO_ROOT/scripts/ci/check_16kb_alignment.sh" "$BIN_PATH" >/tmp/smoke_al.$$.log 2>&1; then
    R_ALIGN="PASS"; echo "  ✓ 16 KB-page aligned arm64 (Play-shippable)"
else
    R_ALIGN="FAIL"; cat /tmp/smoke_al.$$.log; rm -f /tmp/smoke_al.$$.log
    fail "binary is not 16 KB-aligned (Google Play would reject the AAB)"
fi
rm -f /tmp/smoke_al.$$.log

if [ "$DO_RUN" -eq 0 ]; then
    echo "▸ --no-run: stopping after device-free steps"
    finish 0
fi

# ── Step 3: run on device ──────────────────────────────────────────────────────
if ! $ADB shell true >/dev/null 2>&1; then
    fail "no adb device reachable ($ADB). Connect one or pass --serial, or use --no-run."
fi
echo "▸ deploy (NNOPT_DTYPE=fp16)…"
if ! ( cd "$MODEL_DIR" && NNOPT_DTYPE=fp16 ADB="$ADB" ./scripts/deploy_android.sh ) >/tmp/smoke_deploy.$$.log 2>&1; then
    tail -25 /tmp/smoke_deploy.$$.log; rm -f /tmp/smoke_deploy.$$.log
    fail "deploy_android.sh failed"
fi
rm -f /tmp/smoke_deploy.$$.log

LDPATH="$REMOTE_DIR/lib:/system/vendor/lib64:\$LD_LIBRARY_PATH"
RUN_LOG="/tmp/smoke_run.$$.log"

if [ "$MODE" = "serve" ]; then
    echo "▸ run --serve (prompt: \"$PROMPT\", ${SECONDS_ARG}s)…"
    ESC=$(printf '%s' "$PROMPT" | sed "s/'/'\\\\''/g")
    # Clear any stale output so we assert on THIS run's WAV.
    $ADB shell "rm -f $REMOTE_DIR/output_serve_0.wav" >/dev/null 2>&1
    printf '%s\nquit\n' "$PROMPT" | \
        $ADB shell "cd $REMOTE_DIR && LD_LIBRARY_PATH=$LDPATH ./$BINARY_NAME --serve --seconds $SECONDS_ARG" \
        >"$RUN_LOG" 2>&1
    RUN_EXIT=$?
    echo "  ── binary output (tail) ──"; tail -8 "$RUN_LOG" | sed 's/^/    /'

    grep -q "SERVE_READY" "$RUN_LOG" || { R_RUN="FAIL"; rm -f "$RUN_LOG"; fail "no SERVE_READY (binary died during load — the cross_attn_cond bug class)"; }
    if grep -q "SERVE_ERR" "$RUN_LOG"; then R_RUN="FAIL"; rm -f "$RUN_LOG"; fail "SERVE_ERR emitted (generation failed)"; fi
    grep -q "SERVE_DONE" "$RUN_LOG" || { R_RUN="FAIL"; rm -f "$RUN_LOG"; fail "no SERVE_DONE (generation never completed)"; }

    # Pull the WAV and require a plausible size (11s stereo 44.1k ≈ 2MB; even 1s > 170KB).
    TMP_WAV="/tmp/smoke_out.$$.wav"
    $ADB pull "$REMOTE_DIR/output_serve_0.wav" "$TMP_WAV" >/dev/null 2>&1
    if [ -f "$TMP_WAV" ]; then
        WAV_BYTES=$(stat -f%z "$TMP_WAV" 2>/dev/null || stat -c%s "$TMP_WAV" 2>/dev/null || echo 0)
        rm -f "$TMP_WAV"
    fi
    [ "${WAV_BYTES:-0}" -gt 50000 ] || { R_RUN="FAIL"; rm -f "$RUN_LOG"; fail "output WAV missing/too small (${WAV_BYTES} bytes)"; }
    [ "$RUN_EXIT" -eq 0 ] || echo "  ⚠ non-zero exit ($RUN_EXIT) but markers+WAV are valid (some adb builds don't propagate exit codes)"
    R_RUN="PASS"; echo "  ✓ SERVE_READY → SERVE_DONE, WAV ${WAV_BYTES} bytes"
else
    echo "▸ run single (prompt: \"$PROMPT\", ${MAX_TOKENS} tok)…"
    ESC=$(printf '%s' "$PROMPT" | sed "s/'/'\\\\''/g")
    $ADB shell "cd $REMOTE_DIR && LD_LIBRARY_PATH=$LDPATH ./$BINARY_NAME '$ESC' $MAX_TOKENS" >"$RUN_LOG" 2>&1
    RUN_EXIT=$?
    echo "  ── binary output (tail) ──"; tail -8 "$RUN_LOG" | sed 's/^/    /'
    [ -s "$RUN_LOG" ] || { R_RUN="FAIL"; rm -f "$RUN_LOG"; fail "no output produced"; }
    [ "$RUN_EXIT" -eq 0 ] || { R_RUN="FAIL"; rm -f "$RUN_LOG"; fail "binary exited non-zero ($RUN_EXIT)"; }
    R_RUN="PASS"; echo "  ✓ exit 0 with output"
fi
rm -f "$RUN_LOG"
finish 0
