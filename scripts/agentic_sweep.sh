#!/usr/bin/env bash
# Run the standard agentic bench cell matrix.
# Usage: agentic_sweep.sh <model> <out_dir>
set -euo pipefail

MODEL="${1:?missing model}"
OUT="${2:?missing out dir}"
mkdir -p "$OUT"

# Short turn — tool-only
lightning-mlx bench "$MODEL" \
  --prompt-file evals/fixtures/agentic_short.txt \
  --num-prompts 5 --max-tokens 256 \
  --max-num-seqs 1 --prefill-batch-size 1 --completion-batch-size 1 \
  --prefill-step-size 8192 --mtp-optimistic \
  --disable-prefix-cache \
  --report-json "$OUT/short.json"

# Tool-heavy
lightning-mlx bench "$MODEL" \
  --prompt-file evals/fixtures/agentic_tool_heavy.txt \
  --num-prompts 3 --max-tokens 512 \
  --max-num-seqs 1 --prefill-batch-size 1 --completion-batch-size 1 \
  --prefill-step-size 8192 --mtp-optimistic \
  --disable-prefix-cache \
  --report-json "$OUT/tool_heavy.json"

# Long-turn artifact
lightning-mlx bench "$MODEL" \
  --prompt-file evals/fixtures/agentic_long.txt \
  --num-prompts 3 --max-tokens 2048 \
  --max-num-seqs 1 --prefill-batch-size 1 --completion-batch-size 1 \
  --prefill-step-size 8192 --mtp-optimistic \
  --disable-prefix-cache \
  --report-json "$OUT/long.json"

echo "DONE: $OUT"
