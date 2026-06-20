"""Regression tests for the Multi-Role Debate judge design changes:

  - Blind scoring (A1): in the final round a debater must NOT receive the
    current round's peer statements (their scores), but must still receive
    the previous discussion round's statements (their arguments).
  - Competence-weighted aggregation (D2a): a role's vote counts more on the
    criteria it is the panel authority for (COMPETENCE_WEIGHTS).
  - final_outcome full scale: the aggregate is a weighted mean rounded into
    [1, 5] with no snapping, so 2/4 can encode panel disagreement.
  - Score-free discussion: discussion rounds carry the no-score instruction and
    NOT the final JSON instruction; the final round is the reverse. Numbers are
    committed for the first time only in the (blind) final round.
"""

from nika.evaluator.multi_role_debate.multi_role_debate_judge import (
    MultiRoleDebateJudge,
)
from nika.evaluator.multi_role_debate.roles_config import (
    RoleConfig,
    DEFAULT_DEBATE_CONFIG,
)
from nika.evaluator.multi_role_debate.debater import RoleDebater
from nika.evaluator.multi_role_debate.aggregator import aggregate_responses
from nika.evaluator.schemas import DebaterResponse, Score, Scores


NAMES = ["Critic", "Network Engineer", "General Operator"]


def _roles():
    return [RoleConfig(name=n, role_description="x") for n in NAMES]


def _debater_response(rel, cor, eff, cla, fin):
    return DebaterResponse(
        scores=Scores(
            relevance=Score(score=rel, comment="c"),
            correctness=Score(score=cor, comment="c"),
            efficiency=Score(score=eff, comment="c"),
            clarity=Score(score=cla, comment="c"),
            final_outcome=Score(score=fin, comment="c"),
        ),
        reasoning="r",
    )


# --------------------------------------------------------------------------
# Blind scoring (A1)
# --------------------------------------------------------------------------

def test_blind_scoring_hides_current_round_peer_scores():
    """Final round (inject_current_round=False): the debater sees the prior
    discussion of later peers but NOT any current-round peer score."""
    roles = _roles()
    deb = RoleDebater(llm=None, name="Network Engineer")  # self_idx = 1
    statements = [["disc-Critic", "disc-NetEng", "disc-Operator"]]  # round 0
    round_statements = ["FINAL-Critic"]  # Critic already committed its score

    MultiRoleDebateJudge._inject_pending_peers(
        debater=deb,
        roles=roles,
        statements=statements,
        round_statements=round_statements,
        self_idx=1,
        round_idx=1,
        inject_current_round=False,
    )
    contents = [m.content for m in deb._messages]

    # No current-round peer score leaked in:
    assert not any("FINAL-Critic" in c for c in contents)
    # But the previous discussion of the later peer (Operator, idx 2) is seen:
    assert any("disc-Operator" in c for c in contents)
    # And no self-injection (bug #1 guard):
    assert not any(c.startswith("[Network Engineer]") for c in contents)


def test_non_blind_round_does_show_current_round_peers():
    """Discussion round (inject_current_round=True): current-round peers ARE
    injected — the blind behaviour is specific to the final round."""
    roles = _roles()
    deb = RoleDebater(llm=None, name="Network Engineer")  # self_idx = 1
    statements = [["disc-Critic", "disc-NetEng", "disc-Operator"]]
    round_statements = ["DISC2-Critic"]

    MultiRoleDebateJudge._inject_pending_peers(
        debater=deb,
        roles=roles,
        statements=statements,
        round_statements=round_statements,
        self_idx=1,
        round_idx=1,
        inject_current_round=True,
    )
    contents = [m.content for m in deb._messages]
    assert any("DISC2-Critic" in c for c in contents)


# --------------------------------------------------------------------------
# Competence-weighted aggregation (D2a)
# --------------------------------------------------------------------------

def test_weighted_mean_favours_domain_expert():
    """On correctness the Network Engineer carries weight 0.6, so its high
    score pulls the aggregate above the plain (unweighted) mean."""
    # correctness: Critic=2, NetEng=5, Operator=2
    responses = [
        _debater_response(3, 2, 3, 3, 3),  # Critic
        _debater_response(3, 5, 3, 3, 3),  # Network Engineer
        _debater_response(3, 2, 3, 3, 3),  # General Operator
    ]
    out = aggregate_responses(responses, NAMES)
    # weighted = 0.2*2 + 0.6*5 + 0.2*2 = 3.8 -> 4 ; unweighted mean = 3.0 -> 3
    assert out.scores.correctness.score == 4


# --------------------------------------------------------------------------
# final_outcome: full 1-5 scale (no snapping) — 2/4 encode panel disagreement
# --------------------------------------------------------------------------

def test_final_outcome_allows_two_for_panel_disagreement():
    """final_outcome is treated like the other criteria: a weighted mean of
    1.8 rounds to 2 (no snapping to {1,3,5})."""
    # final_outcome: Critic=5, NetEng=1, Operator=1
    # weighted = 0.2*5 + 0.4*1 + 0.4*1 = 1.8  -> round() = 2
    responses = [
        _debater_response(3, 3, 3, 3, 5),
        _debater_response(3, 3, 3, 3, 1),
        _debater_response(3, 3, 3, 3, 1),
    ]
    out = aggregate_responses(responses, NAMES)
    assert out.scores.final_outcome.score == 2


# --------------------------------------------------------------------------
# Score-free discussion vs. scoring in the final round
# --------------------------------------------------------------------------

def _judge():
    return MultiRoleDebateJudge(debate_config=DEFAULT_DEBATE_CONFIG)


def test_discussion_round_is_score_free():
    """A non-final round must carry the score-free discussion instruction and
    must NOT carry the final JSON-scoring instruction."""
    judge = _judge()
    role = DEFAULT_DEBATE_CONFIG.roles[0]

    cont = judge._continuation_prompt(role, is_final=False)
    assert "DISCUSSION round" in cont
    assert "Do NOT assign any numeric" in cont
    assert "JSON object" not in cont  # no scoring instruction leaked in

    initial = judge._initial_user_prompt(
        role, ground_truth="gt", trace="tr", is_final=False
    )
    assert "Do NOT assign any numeric" in initial
    assert "JSON object" not in initial


def test_final_round_is_scoring_not_discussion():
    """The final round must carry the JSON-scoring instruction and must NOT
    carry the score-free discussion instruction."""
    judge = _judge()
    role = DEFAULT_DEBATE_CONFIG.roles[0]

    cont = judge._continuation_prompt(role, is_final=True)
    assert "JSON object" in cont
    assert "Do NOT assign any numeric" not in cont
