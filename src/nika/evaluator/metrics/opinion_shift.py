"""
Opinion Shift — measures debate dynamics in the MultiAgentJudge panel.

Two complementary measures:
  1. Intra-agent shift: how much each debater changes their scores
     from Round 1 to the last round.
  2. Inter-agent divergence per round: average score gap between
     debaters at each round — decreasing trend means convergence.
"""

import json
from collections import defaultdict
from pathlib import Path

_CRITERIA = ["relevance", "correctness", "efficiency", "clarity", "final_outcome", "overall_score"]




def _parse_rounds(debate_rounds_path: str) -> list[dict] | None:
    """Load and parse a debate_rounds.json file.

    Returns:
        List of round dicts, or None if the file is malformed.
    """
    try:
        with open(debate_rounds_path) as f:
            rounds = json.load(f)

        parsed_rounds = []
        for r in rounds:
            parsed_assessments = {}
            for debater_name, assessment_json in r["assessments"].items():
                data = json.loads(assessment_json)
                parsed_assessments[debater_name] = {
                    c: data["scores"][c]["score"] for c in _CRITERIA
                }
            parsed_rounds.append({
                "round": r["round"],
                "assessments": parsed_assessments,
            })

        return parsed_rounds

    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return None




def compute_opinion_shift(debate_rounds_paths: list[str]) -> dict:
    """Compute Opinion Shift metrics across multiple runs.

    Args:
        debate_rounds_paths: List of paths to debate_rounds.json files.

    """


    # intra_agent_shift[debater][criterion] -- list of shifts across ru
    intra_shifts: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))


    # inter_divergence[round_num] → list of avg divergences across runs
    inter_divergence: dict[int, list[float]] = defaultdict(list)


    rounds_counts: list[int] = []

    for path in debate_rounds_paths:
        rounds = _parse_rounds(path)
        if not rounds or len(rounds) < 1:
            continue

        rounds_counts.append(len(rounds))

        round1 = next((r for r in rounds if r["round"] == 1), None)
        round_last = rounds[-1]

        if round1 is None:
            continue

        ## Intra-agent shift (round 1 --> last round) 
        for debater_name, scores_last in round_last["assessments"].items():
            scores_r1 = round1["assessments"].get(debater_name)
            if scores_r1 is None:
                continue
            for criterion in _CRITERIA:
                shift = abs(scores_last[criterion] - scores_r1[criterion])
                intra_shifts[debater_name][criterion].append(shift)

        ## Inter-agent divergence per round 
        debater_names = list(round1["assessments"].keys())
        for r in rounds:
            assessments = r["assessments"]
            if len(assessments) < 2:
                continue

            # Average |debater_i - debater_j| across all pairs and criteria
            divergences = []
            names = list(assessments.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    for criterion in _CRITERIA:
                        diff = abs(
                            assessments[names[i]][criterion] -
                            assessments[names[j]][criterion]
                        )
                        divergences.append(diff)

            if divergences:
                inter_divergence[r["round"]].append(
                    sum(divergences) / len(divergences)
                )


    result_intra: dict[str, dict[str, float]] = {}
    for debater_name, criteria_shifts in intra_shifts.items():
        result_intra[debater_name] = {}
        all_shifts = []
        for criterion in _CRITERIA:
            values = criteria_shifts[criterion]
            avg = round(sum(values) / len(values), 4) if values else 0.0
            result_intra[debater_name][criterion] = avg
            all_shifts.extend(values)
        result_intra[debater_name]["overall"] = (
            round(sum(all_shifts) / len(all_shifts), 4) if all_shifts else 0.0
        )

    result_inter: dict[int, float] = {
        round_num: round(sum(vals) / len(vals), 4)
        for round_num, vals in sorted(inter_divergence.items())
    }

    avg_rounds = (
        round(sum(rounds_counts) / len(rounds_counts), 2)
        if rounds_counts else 0.0
    )

    return {
        "intra_agent_shift": result_intra,
        "inter_agent_divergence_by_round": result_inter,
        "avg_rounds_to_consensus": avg_rounds,
    }




if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    paths = list(root.rglob("debate_rounds.json"))

    if not paths:
        print("No debate_rounds.json files found.")
    else:
        result = compute_opinion_shift([str(p) for p in paths])

        print(f"\nOpinion Shift ({len(paths)} runs)\n")

        print("Intra-agent shift (Round 1 → Last Round):")
        for debater, shifts in result["intra_agent_shift"].items():
            print(f"  {debater}: overall={shifts['overall']} | {shifts}")

        print("\nInter-agent divergence by round:")
        for round_num, div in result["inter_agent_divergence_by_round"].items():
            print(f"  Round {round_num}: {div}")

        print(f"\nAvg rounds to consensus: {result['avg_rounds_to_consensus']}")