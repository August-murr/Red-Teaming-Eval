#!/usr/bin/env bash
# Interpret the example prompt discovery output.
# Usage: bash discovery_example/run_discovery_interpretation_example.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# Save everything printed by this script, including errors, to a log file.
exec > >(tee discovery_example/discovery_interpretation.log) 2>&1

python red_teaming_discovery_interpreter.py \
  --input-dir discovery_example/prompt_discovery \
  --output-dir discovery_example/discovery_interpretation \
  --max-workers 3 \
  --model deepseek/deepseek-v4-flash \
  --temperature 0.0 \
  --max-completion-tokens 1800 \
  --final-max-completion-tokens 3000 \
  --openrouter-retries 3 \
  --openrouter-retry-delay 2 \
  --resume

# Ensure OPENROUTER_API_KEY is exported in your shell before running.
