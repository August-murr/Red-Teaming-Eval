#!/usr/bin/env bash
# Run the main red teaming evaluation script with a standard set of arguments.
# Usage: bash dev/run_evaluation.sh

python3 red_teaming_evaluator.py \
  --dataset walledai/HarmBench \
  --subset contextual \
  --split train \
  --column prompt \
  --row-limit 20 \
  --max-workers 10 \
  --batch-delay 0.3 \
  --max-completion-tokens 1000 \
  --eval-completion-tokens 1000 \
  --temperature 0.0 \
  --model deepseek/deepseek-v4-flash \
  --output evaluations.jsonl

# Adjust the above values if you want a different dataset, row count, or model.
# Ensure OPENROUTER_API_KEY is exported in your shell before running.
