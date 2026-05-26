"""
Krippendorff α — inter-rater reliability among MultiAgentJudge debaters.

Uses Round 1 scores (independent evaluation) since subsequent rounds are
influenced by the debate and would artificially inflate agreement.
Supports any number of debaters.

alpha = 1.0 → perfect agreement
alpha = 0.0 → agreement equivalent to chance
alpha < 0   → systematic disagreement

"""

import json
import numpy as np
import krippendorff

_CRITERIA = ["relevance", "correctness", "efficiency", "clarity", "final_outcome", "overall_score"]


def _load_round_scores(debate_rounds_path: str, round_num: int = 1) -> dict[str, dict[str, int]] | None:
    """Extract per-debater scores for a given round from debate_rounds.json.

    Returns:
        {debater_name: {criterion: score}} for all debaters present in the round,
        or None if the round does not exist or the file is malformed.
    """
    try:
        with open(debate_rounds_path) as f:
            rounds = json.load(f)

        target = next((r for r in rounds if r["round"] == round_num), None)
        if target is None:
            return None

        result = {}
        for debater_name, assessment_json in target["assessments"].items():
            data = json.loads(assessment_json)
            result[debater_name] = {c: data["scores"][c]["score"] for c in _CRITERIA}

        return result

    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return None


def compute_krippendorff_alpha(
    debate_rounds_paths: list[str],
    level_of_measurement: str = "ordinal",
) -> float:
    """Compute Krippendorff alpha across multiple runs using Round 1 scores.

    Dynamically builds the reliability matrix (n_raters , n_items) from the
    debaters found in the JSON files — works with any number of agents.
    Each run contributes len(_CRITERIA) items per rater.

    Args:
        debate_rounds_paths: List of paths to debate_rounds.json files.
        level_of_measurement: "ordinal" (default), "interval", or "nominal".

    Returns:
        Krippendorff alpha as a float.
    """

    # {debater_name: [score_item_0, score_item_1, ...]}
    scores_per_rater: dict[str, list[int]] = {}

    for path in debate_rounds_paths:
        round_scores = _load_round_scores(path, round_num=1)
        if round_scores is None:
            continue

        for debater_name, criteria_scores in round_scores.items():
            if debater_name not in scores_per_rater:
                scores_per_rater[debater_name] = []
            for criterion in _CRITERIA:
                scores_per_rater[debater_name].append(criteria_scores[criterion])

    if not scores_per_rater:
        raise ValueError("No valid data found in the provided files.")

    lengths = {name: len(s) for name, s in scores_per_rater.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Inconsistent item counts across raters: {lengths}")

    # reliability_data: shape (n_raters, n_items)
    rater_names = sorted(scores_per_rater.keys())
    reliability_data = np.array(
        [scores_per_rater[name] for name in rater_names],
        dtype=float,
    )

    alpha = krippendorff.alpha(
        reliability_data,
        level_of_measurement=level_of_measurement,
    )
    return round(float(alpha), 4)



if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    paths = list(root.rglob("debate_rounds.json"))

    if not paths:
        print("No debate_rounds.json files found.")
    else:
        alpha = compute_krippendorff_alpha([str(p) for p in paths])
        print(f"Krippendorff alpha ({len(paths)} runs, ordinal): {alpha}")
        print("Interpretation: alpha ≥ 0.8 excellent | 0.67–0.8 acceptable | < 0.67 unreliable")