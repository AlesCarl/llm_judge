"""Test-retest reliability (offline) — judge consistency on identical input.

Reuses the rep files already saved by the benchmark script
(llm_judge_<type>_rep*.json): the SAME judge scored the SAME session N times.
No LLM calls. For each (session, judge_type) computes per-criterion:
  - MAD (Mean Absolute Deviation across run pairs)
  - std
then averages those over sessions.

Low MAD + low std → the judge is reliable (scores don't change much between
runs on the same trace).
"""

import json
import re
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

_CRITERIA = ["relevance", "correctness", "efficiency", "clarity", "final_outcome"]
_ALL_CRIT = _CRITERIA + ["overall"]

# Minimum reps required to compute test-retest for a (session, judge_type).
MIN_REPS = 3

# Map dal tag nel nome file al judge_type canonico (fabric: "multirole").
_REP_RE      = re.compile(r"^llm_judge_(.+?)_rep\d+\.json$")
_TAG_TO_TYPE = {"multi": "multi", "multirole": "multi_role", "single": "single"}


def _scores_from_rep(rep_path: Path) -> dict | None:
    """Read the 5 criteria (+ computed overall) from a saved rep json."""
    try:
        s = json.loads(rep_path.read_text())["scores"]
        d = {c: s[c]["score"] for c in _CRITERIA}
        d["overall"] = sum(d.values()) / len(_CRITERIA)
        return d
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def compute_test_retest_offline(results_dir: Path, judge_type: str = "all") -> None:
    """Test-retest reliability from saved rep files — no LLM calls.

    A (session, judge_type) is included only if it has >= MIN_REPS reps;
    otherwise it is skipped (the estimate would be unstable).
    """
    # per_type[jt][criterion] = {"mad": [...per-session...], "std": [...]}
    per_type: dict = defaultdict(lambda: defaultdict(lambda: {"mad": [], "std": []}))
    n_sessions: dict = defaultdict(int)
    n_skipped: dict = defaultdict(int)

    for metrics_path in Path(results_dir).rglob("eval_metrics.json"):
        session_dir = metrics_path.parent

        reps_by_type: dict[str, list[dict]] = defaultdict(list)
        for f in session_dir.glob("llm_judge_*_rep*.json"):
            m = _REP_RE.match(f.name)
            if not m:
                continue
            jt = _TAG_TO_TYPE.get(m.group(1))
            scores = _scores_from_rep(f) if jt else None
            if scores is not None:
                reps_by_type[jt].append(scores)

        for jt, reps in reps_by_type.items():
            if len(reps) < MIN_REPS:
                n_skipped[jt] += 1
                continue
            n_sessions[jt] += 1
            for c in _ALL_CRIT:
                vals  = [r[c] for r in reps]
                pairs = list(combinations(vals, 2))
                mad   = sum(abs(a - b) for a, b in pairs) / max(1, len(pairs))
                per_type[jt][c]["mad"].append(mad)
                per_type[jt][c]["std"].append(statistics.stdev(vals))

    types = ["single", "multi", "multi_role"] if judge_type == "all" else [judge_type]

    for jt in types:
        data = per_type.get(jt)
        if not data:
            skipped = n_skipped.get(jt, 0)
            note = f" ({skipped} sessions had < {MIN_REPS} reps)" if skipped else ""
            print(f"\n[test-retest:{jt}]  skipped — no session with >= {MIN_REPS} reps{note}.")
            continue
        print(f"\n[test-retest:{jt}]  n={n_sessions[jt]} sessions  (>= {MIN_REPS} reps, mean over sessions)\n")
        print(f"  {'Criterion':<20} {'mean_MAD':>9} {'mean_std':>9}")
        print("  " + "-" * 40)
        for c in _ALL_CRIT:
            mads = data[c]["mad"]
            stds = data[c]["std"]
            print(f"  {c:<20} {sum(mads)/len(mads):>9.3f} {sum(stds)/len(stds):>9.3f}")

    print("\n  MAD < 0.3  and  std < 0.5  → reliable judge")


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    compute_test_retest_offline(root, "all")

##  python -m nika.evaluator.metrics.test_retest <results_dir>
