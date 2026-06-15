"""Compute all judge evaluation metrics across benchmark runs."""

from pathlib import Path
# --- DISABLED: Krippendorff alpha & Opinion Shift (multi-only diagnostics) ---
# from nika.evaluator.metrics.krippendorff_alpha import compute_krippendorff_alpha
# from nika.evaluator.metrics.opinion_shift import compute_opinion_shift

from nika.evaluator.metrics.spearman_correlation import compute_spearman
from nika.evaluator.metrics.test_retest import compute_test_retest_offline

import argparse


RESULTS_DIR = Path("results")

def compute_all(results_dir: Path = RESULTS_DIR, judge_type: str = "all") -> None:

    # ===== DISABLED: Krippendorff alpha & Opinion Shift (multi-only diagnostics) =====
    # Commented out on purpose: these are not used for the cross-architecture
    # comparison (they exist only for `multi`, so they cannot rank single/multi/
    # multi_role). Re-enable by uncommenting this block and the imports above.
    #
    # paths_multi      = list(results_dir.rglob("debate_rounds.json"))
    # paths_multi_role = list(results_dir.rglob("debate_responses.json"))
    #
    # print(f"Found {len(paths_multi)} multi runs, {len(paths_multi_role)} multi_role runs.\n")
    #
    # # --- Krippendorff alpha & Opinion Shift (multi only) ---
    # # These two metrics are defined only for the `multi` judge (iterative
    # # Critic vs Advocate). multi_role has no per-round scores (free-form
    # # discussion + a single final scoring round), so it is not supported here.
    # # Pooling both would also collide on shared rater names (e.g. "Critic").
    #
    # if judge_type in ("multi_role", "single"):
    #     print("Krippendorff alpha / Opinion Shift: skipped (multi-only metrics).")
    # elif not paths_multi:
    #     print("Krippendorff alpha / Opinion Shift: no `multi` runs found — skipped.")
    # else:
    #     multi_paths = [str(p) for p in paths_multi]
    #
    #     # --- Krippendorff alpha ---
    #     alpha = compute_krippendorff_alpha(multi_paths)
    #     print(f"Krippendorff alpha:      {alpha}")
    #
    #     # --- Opinion Shift ---
    #     opinion_shift = compute_opinion_shift(multi_paths)
    #
    #     print("\nOpinion Shift:")
    #     print("  Intra-agent shift (Round 1 → Last Round):")
    #     for debater, shifts in opinion_shift["intra_agent_shift"].items():
    #         print(f"    {debater}:")
    #         for criterion, value in shifts.items():
    #             if criterion == "overall":
    #                 continue
    #             print(f"      {criterion:<20} {value}")
    #         print(f"      {'overall':<20} {shifts['overall']}")
    #
    #     print("\n  Inter-agent divergence by round:")
    #     for round_num, div in opinion_shift["inter_agent_divergence_by_round"].items():
    #         print(f"    Round {round_num}: {div}")
    #
    #     print(f"\n  Avg rounds to consensus: {opinion_shift['avg_rounds_to_consensus']}")
    # ===== END DISABLED BLOCK =====


    # --- Spearman ---
    print("\n--- Predictive Validity (Spearman ρ) ---")
    compute_spearman(results_dir, judge_type=judge_type)


    # --- Test-retest reliability (offline, from rep files) ---
    # Reuses the saved llm_judge_<type>_rep*.json — no extra LLM calls.
    # Needs N_REPS >= 2 (>= 3 recommended) in the benchmark script.
    print("\n--- Test-Retest Reliability (offline, MAD / std) ---")
    compute_test_retest_offline(results_dir, judge_type=judge_type)


    ##  NB: la versione LIVE in test_retest.run_test_retest richiama l'LLM N volte
    # -- usala solo se NON hai i file _rep* (legge conversation_diagnosis_agent.log
    #    + ground_truth.json e rifa le run davvero).





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge-type",
        choices=["multi", "multi_role", "single", "all"],
        default="all",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Path to results directory (default: results/ in cwd).",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    compute_all(results_dir=results_dir, judge_type=args.judge_type)


## run it with:

"""
cd /Users/alessandrocarlone/Desktop/TESI/Code/nika
python -m nika.evaluator.metrics.compute_metrics \
  --results-dir /Users/alessandrocarlone/Desktop/TESI/result_fabics/  judge_compare_0610_1157 \
  --judge-type all
"""








# python -m nika.evaluator.metrics.compute_metrics --judge-type multi
# python -m nika.evaluator.metrics.compute_metrics --judge-type multi_role
# python -m nika.evaluator.metrics.compute_metrics --judge-type all  

