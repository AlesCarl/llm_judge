"""Numeric aggregation: list[DebaterResponse] → JudgeResponse.

Each debater produces an independent judgement (scores + reasoning), 
then we collapse the panel into a single JudgeResponse via simple per-criterion averaging.

Aggregation rules:
  - Per-criterion score: arithmetic mean across debaters, rounded to
    the nearest integer (Scores schema requires int 1-5).
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

    # 1. Per-criterion: mean score + concatenated commentary
    aggregated_scores: dict[str, Score] = {}
    for criterion in _CRITERIA:
        criterion_scores = [
            getattr(resp.scores, criterion).score for _, resp in valid
        ]
        criterion_comments = [
            f"[{name}] {getattr(resp.scores, criterion).comment}"
            for name, resp in valid
        ]
        avg = statistics.mean(criterion_scores)
        rounded = max(1, min(5, round(avg))) 
        aggregated_scores[criterion] = Score(
            score=rounded,
            comment=(
                f"Panel mean = {avg:.2f} (rounded to {rounded}).\n"
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
        f"Per-criterion scores are the arithmetic mean of the panel's "
        f"final-round DebaterResponse values, rounded to the nearest "
        f"integer in [1, 5]. The panel-mean overall_score "
        f"(average of each debater's overall_score) is "
        f"{panel_overall:.2f}; the JudgeResponse.overall_score is "
        f"recomputed from the rounded per-criterion scores "
        f"({scores.overall_score:.2f})."
    )

    return JudgeResponse(
        scores=scores,
        overall_evaluation=overall_evaluation,
        reasoning_for_overall_score=reasoning_for_overall_score,
    )