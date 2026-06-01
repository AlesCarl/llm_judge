"""Predictive validity: Spearman p between LLM judge scores and task metrics.

Scans results/ for sessions with both llm_judge.json and eval_metrics.json,
then computes Spearman correlation between judge scores (overall + per-criterion)
and hard task metrics (rca_f1, rca_accuracy, detection_score, localization_f1).

"""

import json
from pathlib import Path

from scipy.stats import spearmanr

RESULTS_DIR = Path("results")

_JUDGE_CRITERIA  = ["relevance", "correctness", "efficiency", "clarity", "final_outcome", "overall"]
_TASK_METRICS    = ["rca_f1", "rca_accuracy", "detection_score", "localization_f1"]


### Loader

def _load_sessions(results_dir: Path) -> list[dict]:
    """Collect paired (judge scores, task metrics) from all sessions."""
    records = []
    
    for judge_path in results_dir.rglob("llm_judge.json"):
        metrics_path = judge_path.parent / "eval_metrics.json"
        if not metrics_path.exists():
            continue
        try:
            judge   = json.loads(judge_path.read_text())
            metrics = json.loads(metrics_path.read_text())
            s = judge["scores"]
            record = {
                "relevance":     s["relevance"]["score"],
                "correctness":   s["correctness"]["score"],
                "efficiency":    s["efficiency"]["score"],
                "clarity":       s["clarity"]["score"],
                "final_outcome": s["final_outcome"]["score"],
                "overall": (
                    s["relevance"]["score"]     +
                    s["correctness"]["score"]   +
                    s["efficiency"]["score"]    +
                    s["clarity"]["score"]       +
                    s["final_outcome"]["score"]
                ) / 5,
                "rca_f1":            metrics.get("rca_f1"),
                "rca_accuracy":      metrics.get("rca_accuracy"),
                "detection_score":   metrics.get("detection_score"),
                "localization_f1":   metrics.get("localization_f1"),
                "_session": str(judge_path.parent),
            }
            records.append(record)
        except (json.JSONDecodeError, KeyError):
            continue
    return records


### Compute

def compute_spearman(results_dir: Path = RESULTS_DIR) -> None:
    records = _load_sessions(results_dir)
    if len(records) < 3:
        print(f"Not enough data ({len(records)} sessions with both files). Need ≥ 3.")
        return

    print(f"Spearman ρ  —  n={len(records)} sessions\n")

    # Header
    print(f"  {'':22}", end="")
    for tm in _TASK_METRICS:
        print(f"  {tm:>22}", end="")
    print()
    print("  " + "-" * (22 + 24 * len(_TASK_METRICS)))

    for criterion in _JUDGE_CRITERIA:
        print(f"  {criterion:<22}", end="")
        for tm in _TASK_METRICS:
            # Keep only pairs where the task metric is valid (not None / -1.0)
            pairs = [
                (r[criterion], r[tm])
                for r in records
                if r[tm] is not None and r[tm] != -1.0
            ]
            if len(pairs) < 3:
                print(f"  {'N/A (< 3 pts)':>22}", end="")
                continue
            judge_vals, task_vals = zip(*pairs)
            rho, pval = spearmanr(judge_vals, task_vals)
            sig = "*" if pval < 0.05 else " "
            print(f"  {rho:>+.3f}{sig}  p={pval:.3f}  n={len(pairs):>3}", end="")
        print()

    print("\n  * p < 0.05")
    print("\n  Interpretation:")
    print("    |p| ≥ 0.7  strong correlation")
    print("    |p| ≥ 0.4  moderate")
    print("    |p| <  0.4 weak / no correlation")


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR
    compute_spearman(root)





##     python -m nika.evaluator.metrics.spearman_correlation
##     python -m nika.evaluator.metrics.spearman_correlation results/host_ip_conflict
