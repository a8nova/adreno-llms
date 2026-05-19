#!/usr/bin/env bash
# say.sh — interactive on-device TTS demo.
#
# Loads the model on the Android device ONCE, then loops reading lines of
# text from your terminal. Each line is tokenized + synthesized on-device;
# the WAV is pulled back and played on the host as soon as it's ready.
#
# Usage:
#   ./scripts/say.sh --lang <code>
#
# Examples:
#   ./scripts/say.sh --lang amh
#   ./scripts/say.sh --lang eng
#
# Requires:
#   - adb device connected (one device, or set DEVICE=<serial>)
#   - prior build + deploy:
#       NNOPT_DTYPE=fp16 ./scripts/build.sh
#       NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh
#   - weights/<lang>/{model.fp16.bin, tokenizer_vocab.bin} present on the device
#
# Nothing in this script runs Python or touches model math. It's just a
# host wrapper around `adb shell` running the device binary in --interactive
# mode. The binary keeps its model loaded and emits a TTS_WAV_READY marker
# after each utterance; this script catches the marker, pulls the WAV, and
# plays it.
#
# Implementation note: this is bash-3-compatible (works with macOS stock
# bash). Uses a named FIFO instead of `coproc` (bash 4+).

set -uo pipefail

# ── Args ──────────────────────────────────────────────────────────────────
LANG_CODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lang) LANG_CODE="$2"; shift 2 ;;
        --lang=*) LANG_CODE="${1#--lang=}"; shift ;;
        -h|--help)
            sed -n '2,28p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
if [[ -z "$LANG_CODE" ]]; then
    echo "Usage: $0 --lang <code>  (e.g. amh, eng)" >&2; exit 2
fi

# ── Resolve device + remote layout ────────────────────────────────────────
DEVICE="${DEVICE:-$(adb get-serialno 2>/dev/null || true)}"
if [[ -z "$DEVICE" || "$DEVICE" == "unknown" ]]; then
    echo "ERROR: no adb device. plug in + enable USB debugging." >&2; exit 2
fi
REMOTE="${REMOTE:-/data/local/tmp/mms_tts_eng_inference}"
NNOPT_DTYPE="${NNOPT_DTYPE:-fp16}"
case "$NNOPT_DTYPE" in
    fp16) BIN=mms_tts_eng_inference_fp16 ;;
    fp32|"") BIN=mms_tts_eng_inference ;;
    *) echo "bad NNOPT_DTYPE=$NNOPT_DTYPE" >&2; exit 2 ;;
esac

# Sanity: weights present for this language?
# (use </dev/null — `adb shell` is known to consume bytes off its stdin at
# session-setup time, which would eat the user's first utterance otherwise.)
if ! adb -s "$DEVICE" shell "[ -f $REMOTE/weights/$LANG_CODE/model.fp16.bin ]" </dev/null 2>/dev/null; then
    cat >&2 <<EOF
ERROR: $REMOTE/weights/$LANG_CODE/model.fp16.bin missing on device.

Prep + deploy this language first:
    python3 scripts/prep_lang.py $LANG_CODE
    NNOPT_DTYPE=fp16 ./scripts/deploy_android.sh
EOF
    exit 2
fi

WAV_LOCAL="${WAV_LOCAL:-/tmp/tts_out.wav}"
echo "say.sh: lang=$LANG_CODE device=$DEVICE bin=$BIN" >&2

# ── FIFO plumbing (bash 3 compatible) ─────────────────────────────────────
FIFO="$(mktemp -u /tmp/say_in.XXXXXX)"
mkfifo "$FIFO"
cleanup() {
    # Close the keep-alive fd and wait for adb to finish.
    if [[ -n "${KEEPALIVE_FD:-}" ]]; then eval "exec ${KEEPALIVE_FD}>&-"; fi
    wait 2>/dev/null || true
    rm -f "$FIFO"
}
trap cleanup EXIT INT TERM

# ── Launch adb shell with binary in --interactive, fed by FIFO ────────────
# adb's stdout (binary stderr merged via 2>&1) feeds a watcher loop that
# pulls + plays the WAV on each TTS_WAV_READY marker.
adb -s "$DEVICE" shell "cd $REMOTE && \
    LD_LIBRARY_PATH=$REMOTE/lib:/system/vendor/lib64:\$LD_LIBRARY_PATH \
    ./$BIN --interactive --lang $LANG_CODE 2>&1" < "$FIFO" | \
while IFS= read -r line; do
    case "$line" in
        "TTS_WAV_READY "*)
            wav="${line#TTS_WAV_READY }"
            if adb -s "$DEVICE" pull "$REMOTE/$wav" "$WAV_LOCAL" >/dev/null 2>&1; then
                afplay "$WAV_LOCAL" &
            else
                echo "(could not pull $REMOTE/$wav)" >&2
            fi
            ;;
        *)
            printf '%s\n' "$line" >&2 ;;
    esac
done &

# Keep the FIFO write end open from this shell (FD 9) so the adb process
# doesn't see EOF as soon as we open + close the FIFO from each `printf`.
exec 9>"$FIFO"
KEEPALIVE_FD=9

# ── Forward user lines to the binary ──────────────────────────────────────
# The binary prints its own "> " prompt; don't double-prompt.
while IFS= read -r utt; do
    if [[ -z "$utt" ]]; then
        printf '\n' >&9
        break
    fi
    printf '%s\n' "$utt" >&9
done

# Closing FD 9 signals EOF to the binary; it prints "bye." and exits.
exec 9>&-
KEEPALIVE_FD=""

# Wait for the pull+play watcher to drain.
wait
