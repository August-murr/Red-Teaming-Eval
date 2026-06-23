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
```

Key script arguments
- `--dataset`: Hugging Face dataset name (default in helper: `walledai/HarmBench`).
- `--subset`: dataset config/subset name. (remove if dataset doesn't have config)
- `--split`: dataset split (train/validation/test).
- `--column`: which column to read prompts from (e.g., `prompt`).
- `--row-limit`: how many rows to process.
- `--max-workers`: number of parallel requests per batch.
- `--batch-delay`: delay (seconds) between batches.
- `--max-completion-tokens` / `--eval-completion-tokens`: token limits for generation and evaluation.
- `--model`: OpenRouter model name used for both generation and evaluation (configurable).
- `--output`: output JSONL file path.

Output
- The script writes one JSON object per line in the specified `--output` JSONL file. Each object contains:
  - `prompt`: the original prompt text
  - `response`: the model's generated response
  - `evaluation`: the full evaluator model output (summary + rating line)
  - `rating`: the parsed integer rating (0-10) or `null` if parsing failed
