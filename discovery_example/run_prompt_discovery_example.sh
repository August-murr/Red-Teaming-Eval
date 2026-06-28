#!/usr/bin/env bash
# Run the iterative red teaming prompt discovery script with a standard set of arguments.
# Usage: bash discovery_example/run_prompt_discovery_example.sh


set -euo pipefail

cd "$(dirname "$0")/.."

# Save everything printed by this script, including errors, to a log file.
exec > >(tee discovery_example/prompt_discovery.log) 2>&1

python red_teaming_prompt_discovery.py \
  --dataset walledai/HarmBench \
  --subset contextual \
  --split train \
  --column prompt \
  --max-workers 20 \
  --iterations 15 \
  --model deepseek/deepseek-v4-flash \
  --attacker-temperature 0.7 \
  --target-temperature 0.0 \
  --evaluator-temperature 0.0 \
  --attacker-max-completion-tokens 2000 \
  --target-max-completion-tokens 1000 \
  --eval-completion-tokens 1000 \
  --openrouter-retries 3 \
  --openrouter-retry-delay 2 \
  --output /workspace/Red-Teaming-Eval/discovery_example/prompt_discovery

# Adjust the above values if you want a different dataset row, iteration count, or model per role.
# Ensure OPENROUTER_API_KEY is exported in your shell before running.
