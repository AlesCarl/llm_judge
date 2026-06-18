"""Numeric aggregation: list[DebaterResponse] → JudgeResponse.

Each debater produces an independent judgement (scores + reasoning), 
then we collapse the panel into a single JudgeResponse via simple per-criterion averaging.

Aggregation rules:
  - Per-criterion score: competence-WEIGHTED mean across debaters
    (weights from COMPETENCE_WEIGHTS), rounded to the nearest integer
    (Scores schema requires int 1-5). final_outcome is treated like the
    other criteria — the aggregate may land on 2 or 4, which encodes
    panel disagreement (e.g. "mostly failed with one dissenter").
  - Per-criterion comment: concatenated debater comments, each prefixed
    with the role name, so the JudgeResponse keeps the diversity of
    perspectives in plain text.
  - overall_evaluation: short audit trail listing each debater's
    overall_score + their reasoning summary.
  - reasoning_for_overall_score: explicit note that the score is the
    mean of the panel, including the per-debater overall_score values.
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


# Criteria evaluated by each debater
_CRITERIA = ("relevance", "correctness", "efficiency", "clarity", "final_outcome")


def aggregate_responses(
    responses: list[DebaterResponse | None],
    role_names: list[str],
) -> JudgeResponse:
    """Collapse N DebaterResponse into a single JudgeResponse.

    Args:
        responses: Parsed final-round outputs aligned with role_names.
            Entries may be None (parsing failure) and are skipped.
        role_names: Display names of the debaters, used to label
            comments and the per-debater audit trail.

    Returns:
        JudgeResponse with averaged scores and merged textual context.

    Raises:
        RuntimeError: if no debater produced a valid response.
    """
    if len(responses) != len(role_names):
        raise ValueError(
            f"responses ({len(responses)}) and role_names ({len(role_names)}) "
            "must have the same length"
        )

    # Keep only the (role_name, DebaterResponse) pairs that parsed.
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

    # 1. Per-criterion: COMPETENCE-WEIGHTED mean score + concatenated comments.
    #    Each role's vote is weighted by its domain competence on the criterion
    #    (COMPETENCE_WEIGHTS); weights are renormalised over the debaters that
    #    actually produced a parseable response. All criteria, final_outcome
    #    included, are rounded to the nearest integer in [1, 5].
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

        criterion_comments = [
            f"[{name}] (w={weights_row.get(name, 1.0):.2f}) "
            f"{getattr(resp.scores, criterion).comment}"
            for name, resp in valid
        ]
        aggregated_scores[criterion] = Score(
            score=rounded,
            comment=(
                f"Weighted panel mean = {avg:.2f} (aggregated to {rounded}).\n"
                + "\n".join(criterion_comments)
            ),
        )

    scores = Scores(**aggregated_scores)

    # 2. Build the audit-trail texts for the JudgeResponse-level fields.
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
    reasoning_for_overall_score = (
        f"Per-criterion scores are the competence-WEIGHTED mean of the panel's "
        f"final-round DebaterResponse values (weights per criterion from "
        f"COMPETENCE_WEIGHTS), rounded to the nearest integer in [1, 5]. "
        f"The unweighted panel-mean overall_score "
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