"""Test-retest reliability: run the **same** judge N times on the same trace.

Measures judge consistency on **identical** input. Computes:
  - Per-criterion: mean, std, MAD (Mean Absolute Deviation across run pairs)
  - Pairwise Pearson r across runs (per-criterion score vector)

Low MAD + high Pearson → the judge is reliable (results don't change much
between runs on the same trace).

"""

import argparse
import json
import statistics
import tempfile
import textwrap
from itertools import combinations
from pathlib import Path

from nika.evaluator.llm_judge import LLMJudge
from nika.evaluator.multi_agent_judge import MultiAgentJudge
from nika.evaluator.multi_role_debate.multi_role_debate_judge import MultiRoleDebateJudge

_CRITERIA = ["relevance", "correctness", "efficiency", "clarity", "final_outcome"]



def _build_judge(backend: str, model: str, judge_type: str):
    if judge_type == "multi":
        return MultiAgentJudge(judge_llm_backend=backend, judge_model=model)
    elif judge_type == "multi_role":
        return MultiRoleDebateJudge(judge_llm_backend=backend, judge_model=model)
    else:
        return LLMJudge(judge_llm_backend=backend, judge_model=model)


def _ground_truth_str(gt: dict) -> str:
    return textwrap.dedent(f"""\
        The root cause is {gt['root_cause_name']}.
        The faulty devices are: {', '.join(gt['faulty_devices'])}.
    """)


## main

def run_test_retest(
    session_dir: str,
    backend: str,
    model: str,
    judge_type: str = "single",
    n: int = 3,
) -> dict:
    
    """Run the judge N times on the same session trace and measure consistency."""
    session    = Path(session_dir)
    trace_path = str(session / "conversation_diagnosis_agent.log")
    gt         = json.loads((session / "ground_truth.json").read_text())
    ground_truth = _ground_truth_str(gt)

    scores_per_run: list[dict[str, float]] = []

    for i in range(n):
        print(f"  Run {i + 1}/{n} ...", end=" ", flush=True)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        judge  = _build_judge(backend, model, judge_type)
        result = judge.evaluate_agent(ground_truth, trace_path, tmp_path)

        run_scores = {c: getattr(result.scores, c).score for c in _CRITERIA}
        run_scores["overall"] = result.scores.overall_score
        scores_per_run.append(run_scores)
        print(f"overall = {run_scores['overall']:.2f}")

    # stats
    all_criteria = _CRITERIA + ["overall"]

    criterion_stats: dict[str, dict] = {}
    for c in all_criteria:
        vals = [r[c] for r in scores_per_run]
        pairs = list(combinations(vals, 2))
        mad   = sum(abs(a - b) for a, b in pairs) / max(1, len(pairs))
        criterion_stats[c] = {
            "mean": round(statistics.mean(vals), 3),
            "std":  round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 3),
            "mad":  round(mad, 3),
        }

    return {
        "session":        str(session_dir),
        "judge_type":     judge_type,
        "n_runs":         n,
        "criterion_stats":  criterion_stats,
    }




def _print_results(result: dict) -> None:
    print(f"\nTest-retest reliability  —  {result['n_runs']} runs  |  judge: {result['judge_type']}")
    print(f"Session: {result['session']}\n")

    print(f"  {'Criterion':<20} {'mean':>6}  {'std':>6}  {'MAD':>6}")
    print("  " + "-" * 44)
    for c, s in result["criterion_stats"].items():
        print(f"  {c:<20} {s['mean']:>6.2f}  {s['std']:>6.3f}  {s['mad']:>6.3f}")


    print("\n  Interpretation:")
    print("    MAD < 0.3  and  std < 0.5  → reliable judge")
    print("    Pearson r > 0.8            → consistent scoring profile across runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--backend",     required=True)
    parser.add_argument("--model",       required=True)
    parser.add_argument("--judge-type",  default="single",
                        choices=["single", "multi", "multi_role"])
    parser.add_argument("--n",           type=int, default=3,
                        help="Number of re-runs (default: 3)")
    args = parser.parse_args()

    result = run_test_retest(
        session_dir=args.session_dir,
        backend=args.backend,
        model=args.model,
        judge_type=args.judge_type,
        n=args.n,
    )
    _print_results(result)



    """ ## Test-retest su una sessione (chiama davvero il judge 3 volte):
    
    python -m nika.evaluator.metrics.test_retest \
        --session-dir results/host_ip_conflict/[MR1-xxx] \
        --backend ollama \
        --model gpt-oss:20b-cloud \
        --judge-type multi_role \
        --n 3
        
    """