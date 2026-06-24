#!/usr/bin/env bash
# Run the iterative red teaming prompt discovery script with a standard set of arguments.
# Usage: bash run_prompt_discovery.sh

python3 red_teaming_prompt_discovery.py \
  --dataset walledai/HarmBench \
  --subset contextual \
  --split train \
  --column prompt \
  --row-index 4 \
  --iterations 3 \
  --model deepseek/deepseek-v4-flash \
  --attacker-temperature 0.7 \
  --target-temperature 0.0 \
  --evaluator-temperature 0.0 \
  --attacker-max-completion-tokens 1000 \
  --target-max-completion-tokens 1000 \
  --eval-completion-tokens 1000 \
  --openrouter-retries 3 \
  --openrouter-retry-delay 2 \
  --output prompt_discovery.jsonl

# Adjust the above values if you want a different dataset row, iteration count, or model per role.
# Ensure OPENROUTER_API_KEY is exported in your shell before running.
