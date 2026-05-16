#!/usr/bin/env bash
# Run a set of 10 prompts at 32 and 64 generated tokens each, capture the
# decoded output and tok/s. Greedy decode (temp=0, seed=42) so output is
# deterministic between runs. SmolLM2-135M-Instruct chat template via --chat.
set -uo pipefail

cd "$(dirname "$0")/.."

PROMPTS=(
  "What is the capital of France?"
  "List three primary colors."
  "Explain what a computer CPU does in one sentence."
  "Write a haiku about the ocean."
  "Name three planets in our solar system."
  "What is the boiling point of water in Celsius?"
  "Give me a recipe for a simple cup of tea."
  "Define gravity in simple terms."
  "Write a short greeting for a birthday card."
  "What language do they speak in Brazil?"
)

LENGTHS=(32 64)

echo "================================================================"
echo "SmolLM2-135M-Instruct prompt benchmark on Adreno 619 v2"
echo "Int8 weights, greedy (temp=0, seed=42), --chat template"
echo "================================================================"
echo

# Print table header
printf "%-3s | %-50s | %-3s | %7s | %7s | %7s\n" \
  "#" "Prompt" "Tok" "decode" "TTFT" "wall"
printf "%-3s-+-%-50s-+-%-3s-+-%7s-+-%7s-+-%7s\n" \
  "---" "$(printf '%.0s-' {1..50})" "---" "-------" "-------" "-------"

ALL_DECODE_TOKS=()

for i in "${!PROMPTS[@]}"; do
  P="${PROMPTS[$i]}"
  for N in "${LENGTHS[@]}"; do
    PROMPT_SHORT="${P:0:50}"
    OUT=$(NNOPT_DTYPE=fp16 NNOPT_QUANT=int8 NNOPT_DEBUG_LAYERS=0 \
          ./scripts/run_android.sh "$P" "$N" --temperature 0 --seed 42 --chat 2>&1)
    DEC=$(echo "$OUT" | grep "BENCHMARK decode_tokens_per_sec:" | awk '{print $3}')
    TTFT=$(echo "$OUT" | grep "BENCHMARK time_to_first_token_sec:" | awk '{print $3}')
    WALL=$(echo "$OUT" | grep "BENCHMARK total_inference_sec:" | awk '{print $3}')
    NGEN=$(echo "$OUT" | grep "BENCHMARK n_generated_tokens:" | awk '{print $3}')

    printf "%-3d | %-50s | %3d | %7s | %7s | %7s\n" \
      "$((i + 1))" "$PROMPT_SHORT" "$N" "$DEC" "$TTFT" "$WALL"
    ALL_DECODE_TOKS+=("$DEC")

    # Save the actual generated output for this prompt+length
    GENERATED=$(echo "$OUT" | awk '
      /^Sampling: temp/{flag=1; next}
      /^BENCHMARK/{flag=0}
      flag {print}
    ')
    echo "    OUTPUT (n_generated=$NGEN):"
    echo "$GENERATED" | sed 's/^/      /'
    echo
  done
done

echo
echo "================================================================"
echo "Summary"
echo "================================================================"
# Compute median of all decode tok/s values
SORTED=$(printf '%s\n' "${ALL_DECODE_TOKS[@]}" | sort -n)
N=$(echo "$SORTED" | wc -l | tr -d ' ')
MEDIAN_LINE=$(( (N + 1) / 2 ))
MED=$(echo "$SORTED" | sed -n "${MEDIAN_LINE}p")
MIN=$(echo "$SORTED" | head -1)
MAX=$(echo "$SORTED" | tail -1)
echo "decode tok/s across all 20 runs (10 prompts × 2 lengths):"
echo "  min:    $MIN"
echo "  median: $MED"
echo "  max:    $MAX"
