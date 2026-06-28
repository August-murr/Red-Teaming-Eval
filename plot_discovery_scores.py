import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def load_row_scores(path: Path) -> Dict[int, int]:
    with path.open("r", encoding="utf-8") as f:
        row = json.load(f)

    scores: Dict[int, int] = {}
    for attempt in row.get("attempts", []):
        iteration = attempt.get("iteration")
        rating = attempt.get("rating")
        if isinstance(iteration, int) and isinstance(rating, int):
            scores[iteration] = rating
    return scores


def load_all_row_scores(input_dir: Path) -> List[Dict[int, int]]:
    row_scores: List[Dict[int, int]] = []
    for path in sorted(input_dir.glob("row_*.json")):
        scores = load_row_scores(path)
        if scores:
            row_scores.append(scores)
    return row_scores


def average_scores_by_iteration(row_scores: List[Dict[int, int]], max_iteration: Optional[int]) -> List[dict]:
    if not row_scores:
        return []

    last_iteration = max_iteration or max(max(scores) for scores in row_scores)
    averages: List[dict] = []
    for iteration in range(1, last_iteration + 1):
        scores_at_iteration: List[int] = []
        for scores in row_scores:
            available_iterations = [seen for seen in scores if seen <= iteration]
            if not available_iterations:
                continue
            latest_iteration = max(available_iterations)
            scores_at_iteration.append(scores[latest_iteration])

        if scores_at_iteration:
            averages.append(
                {
                    "iteration": iteration,
                    "average_rating": mean(scores_at_iteration),
                    "task_count": len(scores_at_iteration),
                }
            )

    return averages


def score_series_for_iteration(row_scores: List[Dict[int, int]], iteration: int) -> List[int]:
    scores_at_iteration: List[int] = []
    for scores in row_scores:
        available_iterations = [seen for seen in scores if seen <= iteration]
        if not available_iterations:
            continue
        latest_iteration = max(available_iterations)
        scores_at_iteration.append(scores[latest_iteration])
    return scores_at_iteration


def score_ranges_by_iteration(row_scores: List[Dict[int, int]], averages: List[dict]) -> Tuple[List[int], List[int]]:
    min_scores: List[int] = []
    max_scores: List[int] = []
    for item in averages:
        scores_at_iteration = score_series_for_iteration(row_scores, item["iteration"])
        min_scores.append(min(scores_at_iteration))
        max_scores.append(max(scores_at_iteration))
    return min_scores, max_scores


def plot_average_scores(
    averages: List[dict],
    row_scores: List[Dict[int, int]],
    output_path: Path,
    base_score: Optional[float],
) -> None:
    iterations = [item["iteration"] for item in averages]
    average_ratings = [item["average_rating"] for item in averages]
    min_scores, max_scores = score_ranges_by_iteration(row_scores, averages)
    task_count = len(row_scores)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    ax.fill_between(
        iterations,
        min_scores,
        max_scores,
        color="#93c5fd",
        alpha=0.22,
        linewidth=0,
        label="Score range",
    )
    ax.plot(
        iterations,
        average_ratings,
        color="#2563eb",
        marker="o",
        markersize=7,
        markerfacecolor="#ffffff",
        markeredgecolor="#2563eb",
        markeredgewidth=2.4,
        linewidth=3.2,
        solid_capstyle="round",
        zorder=3,
        label="Average score by iteration",
    )
    if base_score is not None:
        ax.scatter(
            [0],
            [base_score],
            s=110,
            marker="D",
            color="#f97316",
            edgecolor="#9a3412",
            linewidth=1.6,
            zorder=4,
            label="Baseline score (iteration 0)",
        )
        ax.annotate(
            f"{base_score:.2f}",
            (0, base_score),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
            color="#9a3412",
            fontweight="semibold",
        )

    for iteration, rating in zip(iterations, average_ratings):
        ax.annotate(
            f"{rating:.1f}",
            (iteration, rating),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="#1e3a8a",
            fontweight="semibold",
        )

    xticks = [0] + iterations if base_score is not None else iterations
    ax.set_xticks(xticks)
    ax.set_yticks(range(0, 11))
    ax.set_ylim(-0.25, 10.75)
    ax.set_xlabel("Discovery Iteration", fontsize=12, labelpad=10, color="#334155")
    ax.set_ylabel("Average Evaluator Score", fontsize=12, labelpad=10, color="#334155")
    ax.set_title("Prompt Discovery Scores Improve Over Iterations?", fontsize=17, fontweight="bold", pad=18)
    ax.text(
        0,
        1.02,
        f"Average score across {task_count} task rows. Early-stopped rows carry their latest score forward.",
        transform=ax.transAxes,
        fontsize=10,
        color="#64748b",
    )
    ax.grid(axis="y", color="#e2e8f0", linewidth=1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#475569")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()


def write_summary(averages: List[dict], output_path: Path, base_score: Optional[float]) -> None:
    summary_path = output_path.with_suffix(".json")
    summary = list(averages)
    if base_score is not None:
        summary = [
            {
                "iteration": 0,
                "average_rating": base_score,
                "task_count": None,
                "source": "manual evaluator baseline",
            }
        ] + summary
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot average prompt discovery scores by iteration.")
    parser.add_argument("--input-dir", required=True, help="Directory containing row_*.json discovery outputs")
    parser.add_argument("--output", required=True, help="Path to save the plot image, e.g. discovery_scores.png")
    parser.add_argument(
        "--max-iteration",
        type=int,
        default=None,
        help="Optional maximum iteration to plot. Defaults to the max iteration found in the row files.",
    )
    parser.add_argument(
        "--base-score",
        type=float,
        default=None,
        help=(
            "Optional iteration-0 baseline score. Set this manually from the average rating produced by "
            "red_teaming_evaluator.py on the original tasks before discovery. A future version may compute "
            "this directly."
        ),
    )
    parser.add_argument(
        "--summary-json",
        action="store_true",
        help="Also write the averaged values beside the plot as a JSON file.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    if not input_dir.is_dir():
        raise ValueError(f"--input-dir must be a directory: {input_dir}")
    if args.max_iteration is not None and args.max_iteration < 1:
        raise ValueError("--max-iteration must be 1 or greater")
    if args.base_score is not None and not 0 <= args.base_score <= 10:
        raise ValueError("--base-score must be between 0 and 10")

    row_scores = load_all_row_scores(input_dir)
    if not row_scores:
        raise ValueError(f"No scored row_*.json files found in {input_dir}")

    averages = average_scores_by_iteration(row_scores, args.max_iteration)
    if not averages:
        raise ValueError(f"No ratings found in {input_dir}")

    plot_average_scores(averages, row_scores, output_path, args.base_score)
    if args.summary_json:
        write_summary(averages, output_path, args.base_score)

    print(f"Saved plot to {output_path}")
    print(f"Averaged {len(row_scores)} task rows")


if __name__ == "__main__":
    main()
