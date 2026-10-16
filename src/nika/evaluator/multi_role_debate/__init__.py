"""Multi-Role Debate judge (ChatEval-style): role debaters discuss, then score; scores are averaged."""

from nika.evaluator.multi_role_debate.roles_config import (
    DebateConfig,
    RoleConfig,
    DEFAULT_DEBATE_CONFIG,
)

__all__ = ["DebateConfig", "RoleConfig", "DEFAULT_DEBATE_CONFIG"]