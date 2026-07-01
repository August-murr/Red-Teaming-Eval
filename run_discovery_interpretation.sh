#!/usr/bin/env bash
# Interpret iterative red teaming prompt discovery outputs with an OpenRouter LLM.
# Usage: bash run_discovery_interpretation.sh

python red_teaming_discovery_interpreter.py \
  --input-dir prompt_discovery \
  --output-dir discovery_interpretation \
  --max-workers 20 \
  --model deepseek/deepseek-v4-flash \
  --temperature 0.0 \
  --max-completion-tokens 2000 \
  --final-max-completion-tokens 3000 \
  --openrouter-retries 3 \
  --openrouter-retry-delay 2 \
  --resume

# Adjust the input/output paths and worker count for your discovery run.
# Ensure OPENROUTER_API_KEY is exported in your shell before running.
