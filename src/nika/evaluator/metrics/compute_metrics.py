"""Compute all judge evaluation metrics across benchmark runs."""

from pathlib import Path
from nika.evaluator.metrics.krippendorff_alpha import compute_krippendorff_alpha
from nika.evaluator.metrics.opinion_shift import compute_opinion_shift

from nika.evaluator.metrics.spearman_correlation import compute_spearman

import argparse


RESULTS_DIR = Path("results")

def compute_all(results_dir: Path = RESULTS_DIR, judge_type: str = "all") -> None:

    paths_multi      = list(results_dir.rglob("debate_rounds.json"))
    paths_multi_role = list(results_dir.rglob("debate_responses.json"))

    if judge_type == "multi":
        all_paths = paths_multi
    elif judge_type == "multi_role":
        all_paths = paths_multi_role
    else:
        all_paths = paths_multi + paths_multi_role



    if not all_paths:
        print("No debate files found (debate_rounds.json or debate_responses.json).")
        return

    print(f"Found {len(paths_multi)} multi runs, {len(paths_multi_role)} multi_role runs.\n")


    # --- Krippendorff alpha ---
    alpha = compute_krippendorff_alpha([str(p) for p in all_paths])
    print(f"Krippendorff alpha:      {alpha}")


    # --- Opinion Shift ---
    opinion_shift = compute_opinion_shift([str(p) for p in all_paths])

    print("\nOpinion Shift:")
    print("  Intra-agent shift (Round 1 → Last Round):")
    for debater, shifts in opinion_shift["intra_agent_shift"].items():
        print(f"    {debater}:")
        for criterion, value in shifts.items():
            if criterion == "overall":
                continue
            print(f"      {criterion:<20} {value}")
        print(f"      {'overall':<20} {shifts['overall']}")

    print("\n  Inter-agent divergence by round:")
    for round_num, div in opinion_shift["inter_agent_divergence_by_round"].items():
        print(f"    Round {round_num}: {div}")

    print(f"\n  Avg rounds to consensus: {opinion_shift['avg_rounds_to_consensus']}")


    # --- Spearman ---
    print("\n--- Predictive Validity (Spearman ρ) ---")
    compute_spearman(results_dir)


    ##  NB: test_retest chiama davvero l'LLM N volte 
    # -- non ha senso chiamarla automaticamente su tutti i risultati
    # -- legge "conversation_diagnosis_agent.log" + "ground_truth.json"



    # latency_mult  = ...



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge-type",
        choices=["multi", "multi_role", "all"],
        default="all",
    )
    args = parser.parse_args()
    compute_all(judge_type=args.judge_type)


## run it with:
# python -m nika.evaluator.metrics.compute_metrics --judge-type multi
# python -m nika.evaluator.metrics.compute_metrics --judge-type multi_role
# python -m nika.evaluator.metrics.compute_metrics --judge-type all  

