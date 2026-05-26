"""Multi-Role Debate evaluator (ChatEval-style).

Multiple LLM agents with distinct personas (Critic, Network Engineer,
General Operator, ...) deliberate sequentially over the agent's trace.
Each round, every debater sees the previous statements as visible chat
history. In the final round, debaters output structured per-criterion
scores (DebaterResponse). Final JudgeResponse is built by numeric
aggregation across debaters.
"""

from nika.evaluator.multi_role_debate.roles_config import (
    DebateConfig,
    RoleConfig,
    DEFAULT_DEBATE_CONFIG,
)

__all__ = ["DebateConfig", "RoleConfig", "DEFAULT_DEBATE_CONFIG"]