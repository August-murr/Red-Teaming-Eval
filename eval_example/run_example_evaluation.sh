#!/usr/bin/env bash
# Run from the repo root with:
#   bash example/run_example_evaluation.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# Save everything printed by this script, including errors, to a log file.
exec > >(tee example/evaluation.log) 2>&1

python red_teaming_evaluator.py \
  --dataset walledai/HarmBench \
  --subset contextual \
  --split train \
  --column prompt \
  --max-workers 64 \
  --batch-delay 0.3 \
  --max-completion-tokens 1000 \
  --eval-completion-tokens 1000 \
  --temperature 0.0 \
  --model deepseek/deepseek-v4-flash \
  --openrouter-retries 3 \
  --openrouter-retry-delay 0.5 \
  --output example/evaluations.jsonl
