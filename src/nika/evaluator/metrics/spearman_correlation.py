"""Predictive validity: Spearman p between LLM judge scores and task metrics.

Scans results/ for sessions with both llm_judge.json and eval_metrics.json,
then computes Spearman correlation between judge scores (overall + per-criterion)
and hard task metrics (rca_f1, rca_accuracy, detection_score, localization_f1).

"""

import json
from pathlib import Path

from scipy.stats import spearmanr

import re
from collections import defaultdict



RESULTS_DIR = Path("results")

_JUDGE_CRITERIA  = ["relevance", "correctness", "efficiency", "clarity", "final_outcome", "overall"]
_TASK_METRICS    = ["rca_f1", "rca_accuracy", "detection_score", "localization_f1",
                    "steps", "tool_calls", "in_tokens", "out_tokens"]




def _detect_judge_type(session_dir: Path) -> str:
    """Infer which judge produced this session's llm_judge.json.
    """
    if (session_dir / "debate_rounds.json").exists():
        return "multi"
    if (session_dir / "debate_responses.json").exists():
        return "multi_role"
    return "single"



### Loader


# Map dal tag nel nome file al judge_type canonico.

_REP_RE      = re.compile(r"^llm_judge_(.+?)_rep\d+\.json$")
_TAG_TO_TYPE = {"multi": "multi", "multirole": "multi_role", "single": "single"} ## fabric : multirole": "multi_role
_CRIT5       = ["relevance", "correctness", "efficiency", "clarity", "final_outcome"]


def _read_scores(judge_path: Path) -> dict | None:
    """Read the 5 per-criterion scores from a judge json, or None if malformed."""
    try:
        s = json.loads(judge_path.read_text())["scores"]
        return {c: s[c]["score"] for c in _CRIT5}
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _make_record(crit: dict, metrics: dict, session: str, judge_type: str) -> dict:
    """Build one Spearman record from averaged criteria + shared task metrics."""
    rec = dict(crit)
    rec["overall"] = sum(crit[c] for c in _CRIT5) / 5
    for tm in _TASK_METRICS:
        rec[tm] = metrics.get(tm)
    rec["_session"]    = session
    rec["_judge_type"] = judge_type
    return rec


def _load_sessions(results_dir: Path) -> list[dict]:
    """Collect paired (judge scores, task metrics), one record per (session, judge_type).

    Iterates over sessions (one eval_metrics.json each, shared across judges).
    If the session has suffixed rep files (llm_judge_<tag>_rep*.json) it infers
    the judge_type from the FILENAME and averages repeated measurements per type.
    Otherwise it falls back to the plain llm_judge.json + artifact-based detection.
    """
    records = []

    for metrics_path in results_dir.rglob("eval_metrics.json"):
        session_dir = metrics_path.parent
        try:
            metrics = json.loads(metrics_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        # Group rep files by judge_type, collecting all reps for averaging.
        reps_by_type: dict[str, list[dict]] = defaultdict(list)
        for f in session_dir.glob("llm_judge_*_rep*.json"):
            m = _REP_RE.match(f.name)
            if not m:
                continue
            jt = _TAG_TO_TYPE.get(m.group(1))
            scores = _read_scores(f) if jt else None
            if scores is not None:
                reps_by_type[jt].append(scores)

        if reps_by_type:
            # Repeated measures: average reps per (session, judge_type).
            for jt, reps in reps_by_type.items():
                crit = {c: sum(r[c] for r in reps) / len(reps) for c in _CRIT5}
                records.append(_make_record(crit, metrics, str(session_dir), jt))
        else:
            # Fallback: classic layout (single llm_judge.json per session).
            scores = _read_scores(session_dir / "llm_judge.json")
            if scores is not None:
                records.append(
                    _make_record(scores, metrics, str(session_dir),
                                 _detect_judge_type(session_dir))
                )

    return records



### Compute

def _print_spearman_table(records: list[dict], label: str) -> None:
    """Print one Spearman table (criteria × task metrics) for a record set."""
    if len(records) < 3:
        print(f"\n[{label}]  not enough data ({len(records)} sessions, need ≥ 3).")
        return

    print(f"\n[{label}]  Spearman ρ  —  n={len(records)} sessions\n")

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


def compute_spearman(results_dir: Path = RESULTS_DIR, judge_type: str = "all") -> None:
    """Compute Spearman ρ, one table per judge architecture.

    Args:
        results_dir: root to scan for llm_judge.json / eval_metrics.json pairs.
        judge_type: "all" prints separate tables for single / multi / multi_role;
            pass a specific type to restrict the output to that architecture.
    """
    records = _load_sessions(results_dir)
    if not records:
        print("No sessions with both llm_judge.json and eval_metrics.json.")
        return

    types = ["single", "multi", "multi_role"] if judge_type == "all" else [judge_type]

    for jt in types:
        subset = [r for r in records if r["_judge_type"] == jt]
        _print_spearman_table(subset, label=jt)

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
