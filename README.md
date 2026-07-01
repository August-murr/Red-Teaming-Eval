# Red-Teaming-Eval

A simple red-teaming evaluation tool that loads prompts from a Hugging Face dataset, sends them to an LLM via OpenRouter, evaluates the model responses using an evaluation prompt (also via OpenRouter), and saves per-example evaluations and numeric ratings to a JSONL file.

**What it does**
- Loads prompt data from a Hugging Face dataset (configurable dataset, split, and column).
- Sends prompts to an OpenRouter-hosted model to generate responses.
- Sends a second prompt (an evaluation instruction) to the same or another OpenRouter model to judge and rate the generated response on a 0-10 scale.
- Runs generation and evaluation requests in parallel batches and writes results to a JSONL file with fields `prompt`, `response`, `evaluation`, and `rating`.

**How it works (high level)**
- `red_teaming_evaluator.py` loads data using the `datasets` library.
- It calls OpenRouter via the `openrouter` client to generate model responses.
- It constructs an evaluation prompt containing the original request and the model response, asks the evaluator model to summarize and output `Rating: [number]` on the last line, and extracts that numeric rating from the evaluator output.
- The script supports parallel batches (via `concurrent.futures.ThreadPoolExecutor`) and rate-limits between batches with a small delay.

Requirements
- Python 3.10+ recommended
- Install runtime dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Environment variables
- `OPENROUTER_API_KEY`: required. Your OpenRouter API key used by the script to call model APIs.
- `HF_TOKEN`: required when accessing gated, private, or restricted Hugging Face datasets. Many red-teaming datasets are gated and require an access agreement before download; if you do not have the necessary token or agreement, dataset loading will fail.

Export example (Linux/macOS):

```bash
export OPENROUTER_API_KEY="your_openrouter_key_here"
export HF_TOKEN="your_hf_token_here"
```

Running
- Quick run using the provided helper script:

```bash
bash run_evaluation.sh
```

- Or run the script directly with custom arguments:

```bash
python red_teaming_evaluator.py \
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
```

Key script arguments
- `--dataset`: Hugging Face dataset name (default in helper: `walledai/HarmBench`).
- `--subset`: dataset config/subset name. (remove if dataset doesn't have config)
- `--split`: dataset split (train/validation/test).
- `--column`: which column to read prompts from (e.g., `prompt`).
- `--row-limit`: how many rows to process; omit to process the full dataset.
- `--max-workers`: number of parallel requests per batch.
- `--batch-delay`: delay (seconds) between batches.
- `--max-completion-tokens` / `--eval-completion-tokens`: token limits for generation and evaluation.
- `--evaluation-prompt-file`: optional file overriding the default evaluator prompt. Use `{original_prompt}` and `{model_response}` placeholders.
- `--openrouter-retries` / `--openrouter-retry-delay`: retry OpenRouter calls that error, return empty content, or produce an unparseable evaluator rating.
- `--model`: OpenRouter model name used for both generation and evaluation (configurable).
- `--output`: output JSONL file path.

Output
- The script writes one JSON object per line in the specified `--output` JSONL file. Each object contains:
  - `prompt`: the original prompt text
  - `response`: the model's generated response
  - `evaluation`: the full evaluator model output (summary + rating line)
  - `rating`: the parsed integer rating (0-10) or `null` if parsing failed

Prompt discovery
- `red_teaming_prompt_discovery.py` runs an iterative attacker -> target -> evaluator loop for one or more dataset rows. The attacker sees the original task plus feedback from previous attempts for that same row, proposes a new candidate prompt, the target answers it, and the evaluator rates the target response.
- Quick run using the provided helper script:

```bash
bash run_prompt_discovery.sh
```

- Or run directly:

```bash
python red_teaming_prompt_discovery.py \
  --dataset walledai/HarmBench \
  --subset contextual \
  --split train \
  --column prompt \
  --row-limit 5 \
  --max-workers 2 \
  --iterations 5 \
  --model deepseek/deepseek-v4-flash \
  --attacker-temperature 0.7 \
  --target-temperature 0.0 \
  --evaluator-temperature 0.0 \
  --attacker-max-completion-tokens 1000 \
  --target-max-completion-tokens 1000 \
  --eval-completion-tokens 1000 \
  --output prompt_discovery
```

Discovery-specific arguments
- `--row-index`: selects a single dataset row used as the red-team task. When provided, this takes precedence over `--row-limit`.
- `--row-limit`: runs discovery for the first N dataset rows when `--row-index` is omitted. If both `--row-index` and `--row-limit` are omitted, discovery runs over the full dataset.
- `--max-workers`: number of dataset rows to run in parallel. Each row's attacker feedback loop still runs sequentially.
- `--iterations`: number of attacker attempts.
- `--attacker-model`, `--target-model`, `--evaluator-model`: override each role independently. If omitted, `--model` is used for that role.
- `--attacker-system-prompt-file`, `--attacker-template-file`, `--target-system-prompt-file`: optional prompt customization files.
- `--evaluation-prompt-file`: optional evaluator prompt template file with `{original_prompt}` and `{model_response}` placeholders.
- `--openrouter-retries` / `--openrouter-retry-delay`: retry attacker, target, and evaluator OpenRouter calls that error, return empty content, or produce an unparseable evaluator rating.
- `--candidate-start-marker` / `--candidate-end-marker`: delimit the attacker candidate prompt. The script also accepts JSON objects with `candidate_prompt`, `prompt`, or `attack_prompt` keys.
- `--max-feedback-response-chars` / `--max-feedback-evaluation-chars`: control how much previous target and evaluator text is fed back to the attacker.

Discovery output
- The discovery script treats `--output` as a directory and writes one readable JSON file per dataset row, named like `row_000004.json`.
- Each row file contains `row_index`, `task`, `best_rating`, `best_iteration`, and an `attempts` array with `iteration`, `rating`, `candidate_prompt`, `target_response`, and `evaluation`.
- Re-running discovery with the same output directory overwrites files for the selected rows.
- At the end, it prints the best parsed rating and the corresponding candidate prompt across all completed rows.

Interpreting discovery results
- `red_teaming_discovery_interpreter.py` reads a prompt discovery output directory and asks an OpenRouter model to summarize what happened for each task. The interpreter judges whether the attacker learned from feedback, whether improvement was fast or slow, what high-level strategies worked, and why some tasks failed to improve.
- After per-task interpretation, the script sends all summaries plus aggregate score metrics into one final request that produces a cross-task verdict about the whole discovery run.
- Quick run using the provided helper script:

```bash
bash run_discovery_interpretation.sh
```

- Or run the example interpretation:

```bash
bash discovery_example/run_discovery_interpretation_example.sh
```

- Or run directly:

```bash
python red_teaming_discovery_interpreter.py \
  --input-dir discovery_example/prompt_discovery \
  --output-dir discovery_example/discovery_interpretation \
  --max-workers 3 \
  --model deepseek/deepseek-v4-flash \
  --temperature 0.0 \
  --resume
```

Interpretation output
- The interpreter writes one JSON summary per row, named like `row_000004.json`, under `--output-dir`.
- It also writes `interpretation_summaries.jsonl`, `final_verdict.json`, and `final_verdict.md`.
- `--resume` reuses existing per-row interpretation files so interrupted runs do not repeat completed OpenRouter calls.
- `--skip-final` writes only per-row interpretations and skips the cross-task verdict request.
- The interpreter prompt asks for high-level attack strategies and learning dynamics, while avoiding reproduction of detailed harmful target content.

Plotting discovery scores
- `plot_discovery_scores.py` reads a discovery output directory, averages task scores at each iteration, and saves a plot. Rows that stopped early carry their latest score forward for later iterations.
- `--base-score` optionally adds an iteration-0 baseline to the plot. For now, set this manually to the average rating from `red_teaming_evaluator.py` on the original tasks before prompt discovery. A later version may compute this directly from evaluator outputs.

```bash
bash discovery_example/plot_discovery_scores_example.sh
```

Or run it directly:

```bash
python plot_discovery_scores.py \
  --input-dir discovery_example/prompt_discovery \
  --output discovery_example/discovery_scores.png \
  --base-score 2.52 \
  --summary-json
```
