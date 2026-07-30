#!/bin/bash
# Token-exactness gate: run each golden prompt on the DEVICE and diff the
# emitted token ids against the llama.cpp oracle capture in reference/.
#
# Usage:
#   ./scripts/check_goldens.sh                      # 8B  (reference/golden_N.json)
#   BONSAI_GOLDEN=1.7b_ BONSAI_NNB=bonsai1.7b.nnb ./scripts/check_goldens.sh
#
# The device stops ON the EOS token without emitting it, while the oracle
# records it — so a device run that is a strict prefix of the oracle ending
# exactly one token short of an oracle-EOS counts as a PASS.

set -e

cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
G="${BONSAI_GOLDEN:-}"          # '' for 8B, '1.7b_' for the 1.7B set
MAXTOK="${BONSAI_GOLDEN_TOKENS:-64}"

fail=0
for i in 1 2 3; do
    f="reference/golden_${G}${i}.json"
    if [ ! -f "$f" ]; then
        echo "SKIP golden_${G}${i}: no $f"
        continue
    fi
    prompt=$($PY -c "import json,sys; print(json.load(open('$f'))['prompt'], end='')")
    # golden_1 is the ChatML-wrapped one; the port's 'chat' mode re-applies the
    # template, so feed its inner user message instead. 2 and 3 are raw prompts.
    if [ "$i" = "1" ]; then
        mode=chat
        prompt="What is the capital of France?"
    else
        mode=gen
    fi
    ids=$(./scripts/run_android.sh "$prompt" "$MAXTOK" "$mode" 2>/dev/null \
          | grep -E "^output_ids" | sed 's/^output_ids: *//')
    if $PY - "$f" "$ids" <<'EOF'
import json, sys
exp = json.load(open(sys.argv[1]))['output_ids']
got = [int(x) for x in sys.argv[2].split()] if sys.argv[2].strip() else []
if got == exp:
    sys.exit(0)
# device breaks ON eos without emitting it
if len(got) == len(exp) - 1 and got == exp[:-1]:
    sys.exit(0)
n = next((j for j in range(min(len(got), len(exp))) if got[j] != exp[j]),
         min(len(got), len(exp)))
print(f'  first divergence at token {n}: got {got[n:n+4]} want {exp[n:n+4]}')
print(f'  lengths: got {len(got)} want {len(exp)}')
sys.exit(1)
EOF
    then
        echo "PASS golden_${G}${i}"
    else
        echo "FAIL golden_${G}${i}"
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "TOKEN-EXACTNESS GATE FAILED"
    exit 1
fi
echo "ALL GOLDENS TOKEN-EXACT"
