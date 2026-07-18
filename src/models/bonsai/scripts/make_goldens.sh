#!/bin/bash
cd "$(dirname "$0")/.."
LC=/opt/homebrew/bin/llama-cli
M=model/Bonsai-8B-Q1_0.gguf
run() {
  echo "=== golden_$1 ==="
  $LC -m $M -p "$2" -n 64 --temp 0 --top-k 1 --no-conversation --no-warmup \
      -t 8 --verbose-prompt --seed 0 > reference/golden_$1.txt 2>&1
  grep -a "Paris\|fox\|Elias" reference/golden_$1.txt | head -1
}
run 1 $'<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n'
run 2 'The quick brown fox'
run 3 'Once upon a time, in a small village nestled between two mountains, there lived a clockmaker named Elias who could repair anything except'
echo ALL_GOLDENS_DONE
