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

if [ ! -f "build/mamba_130m_hf_inference" ]; then
    echo "ERROR: Binary not found. Run ./scripts/build.sh first"
    exit 1
fi

./build/mamba_130m_hf_inference "$PROMPT" "$MAX_TOKENS" $EXTRA_ARGS
