"""Collapse the debaters' DebaterResponses into one JudgeResponse.

Per criterion the score is the panel mean, rounded to an integer in [1, 5]
(weights come from COMPETENCE_WEIGHTS, equal by default). Comments are
concatenated with the role name, so the different perspectives are kept.
"""

from __future__ import annotations

import logging
import statistics

from nika.evaluator.schemas import (
    DebaterResponse,
    JudgeResponse,
    Score,
    Scores,
)
from nika.evaluator.multi_role_debate.roles_config import COMPETENCE_WEIGHTS


logger = logging.getLogger(__name__)


_CRITERIA = ("relevance", "correctness", "efficiency", "clarity", "final_outcome")


def aggregate_responses(
    responses: list[DebaterResponse | None],
    role_names: list[str],
) -> JudgeResponse:
    """Aggregate the panel. None entries (parse failures) are skipped.

    Raises RuntimeError if no debater produced a valid response.
    """
    if len(responses) != len(role_names):
        raise ValueError(
            f"responses ({len(responses)}) and role_names ({len(role_names)}) "
            "must have the same length"
        )

    # Keep only the debaters that parsed.
    valid: list[tuple[str, DebaterResponse]] = [
        (name, resp)
        for name, resp in zip(role_names, responses)
        if resp is not None
    ]
    if not valid:
        raise RuntimeError(
            "No debater produced a parseable DebaterResponse — "
            "cannot aggregate into JudgeResponse."
        )

    if len(valid) < len(responses):
        skipped = [
            name for name, r in zip(role_names, responses) if r is None
        ]
        logger.warning(
            "Aggregator: skipping %d debater(s) with unparsed responses: %s",
            len(skipped),
            skipped,
        )

    # Weights are renormalised over the valid debaters; empty mapping = plain mean.
    aggregated_scores: dict[str, Score] = {}
    for criterion in _CRITERIA:
        weights_row = COMPETENCE_WEIGHTS.get(criterion, {})
        pairs = [
            (name, getattr(resp.scores, criterion).score) for name, resp in valid
        ]
        raw_w = [weights_row.get(name, 1.0) for name, _ in pairs]
        total_w = sum(raw_w)
        if total_w <= 0:
            raw_w = [1.0] * len(pairs)
            total_w = float(len(pairs))
        avg = sum(w * s for w, (_, s) in zip(raw_w, pairs)) / total_w

        rounded = max(1, min(5, round(avg)))

        # Show weights only when used.
        criterion_comments = [
            f"[{name}]"
            + (f" (w={weights_row.get(name, 1.0):.2f})" if weights_row else "")
            + f" {getattr(resp.scores, criterion).comment}"
            for name, resp in valid
        ]
        aggregated_scores[criterion] = Score(
            score=rounded,
            comment=(
                f"{'Weighted panel' if weights_row else 'Panel'} mean = "
                f"{avg:.2f} (aggregated to {rounded}).\n"
                + "\n".join(criterion_comments)
            ),
        )

    scores = Scores(**aggregated_scores)

    per_debater_lines = []
    for name, resp in valid:
        per_debater_lines.append(
            f"[{name}] overall_score={resp.scores.overall_score:.2f}\n"
            f"  reasoning: {resp.reasoning}"
        )
    audit_trail = "\n".join(per_debater_lines)

    panel_overall = statistics.mean(
        resp.scores.overall_score for _, resp in valid
    )

    overall_evaluation = (
        f"Multi-role debate aggregation across {len(valid)} debater(s):\n"
        f"{audit_trail}"
    )
    aggregation_rule = (
        "the competence-WEIGHTED mean (weights per criterion from "
        "COMPETENCE_WEIGHTS)"
        if COMPETENCE_WEIGHTS
        else "the unweighted mean (every role's vote counts the same)"
    )
    reasoning_for_overall_score = (
        f"Per-criterion scores are {aggregation_rule} of the panel's "
        f"final-round DebaterResponse values, rounded to the nearest integer "
        f"in [1, 5]. The unweighted panel-mean overall_score "
        f"(average of each debater's overall_score) is "
        f"{panel_overall:.2f}; the JudgeResponse.overall_score is "
        f"recomputed from the aggregated per-criterion scores "
        f"({scores.overall_score:.2f})."
    )

    return JudgeResponse(
        scores=scores,
        overall_evaluation=overall_evaluation,
        reasoning_for_overall_score=reasoning_for_overall_score,
    )