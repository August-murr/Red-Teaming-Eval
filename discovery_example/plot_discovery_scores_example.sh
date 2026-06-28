#!/usr/bin/env bash
# Plot averaged discovery scores from the example discovery output.
# Usage: bash discovery_example/plot_discovery_scores_example.sh

set -euo pipefail

cd "$(dirname "$0")/.."

python plot_discovery_scores.py \
  --input-dir discovery_example/prompt_discovery \
  --output discovery_example/discovery_scores.png \
  --base-score 2.52 \
  --summary-json
