"""Compute all judge evaluation metrics across benchmark runs."""

from pathlib import Path
from nika.evaluator.metrics.krippendorff_alpha import compute_krippendorff_alpha
from nika.evaluator.metrics.opinion_shift import compute_opinion_shift


RESULTS_DIR = Path("results")

def compute_all(results_dir: Path = RESULTS_DIR) -> None:


    paths = list(results_dir.rglob("debate_rounds.json"))
    # filter only "host_crash"
    # paths = list(Path("results/host_crash").rglob("debate_rounds.json"))

    
    if not paths:
        print("No debate_rounds.json found.")
        return

    print(f"Found {len(paths)} runs.\n")



    alpha = compute_krippendorff_alpha([str(p) for p in paths])
    print(f"Krippendorff alpha:      {alpha}")




    opinion_shift = compute_opinion_shift([str(p) for p in paths])
    
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



    # latency_mult  = ...

if __name__ == "__main__":
    compute_all()




## run it with:
# python -m nika.evaluator.metrics.compute_metrics

