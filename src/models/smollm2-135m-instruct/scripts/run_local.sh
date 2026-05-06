#!/bin/bash
# Run inference locally

cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
    echo "Usage: $0 \"<prompt>\" [max_tokens]"
    exit 1
fi

PROMPT="$1"
MAX_TOKENS="${2:-64}"
shift 2 2>/dev/null || true
EXTRA_ARGS="$*"

# Dtype: NNOPT_DTYPE=fp16 picks the fp16 binary; default is fp32.
NNOPT_DTYPE="${NNOPT_DTYPE:-fp32}"
case "$NNOPT_DTYPE" in
    fp16) BINARY="build/fp16/SmolLM2_135M_Instruct_inference_fp16" ;;
    fp32|"") BINARY="build/SmolLM2_135M_Instruct_inference" ;;
    *)
        echo "ERROR: NNOPT_DTYPE must be fp32 or fp16 (got '$NNOPT_DTYPE')" >&2
        exit 1
        ;;
esac

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found: $BINARY"
    echo "Run: NNOPT_DTYPE=$NNOPT_DTYPE ./scripts/build.sh --release"
    exit 1
fi

# Env vars are inherited from the calling shell — NNOPT_PROFILE=1,
# NNOPT_DEBUG_LAYERS=0, NNOPT_NO_STREAM=1, etc. all pass through automatically.
exec "$BINARY" "$PROMPT" "$MAX_TOKENS" $EXTRA_ARGS
