"""Dataclass-based configuration for the Multi-Role Debate judge.


Each "RoleConfig" defines a single debate participant: name, persona
(role_description), per-role temperature, and optional model override.

"DebateConfig" groups the debate-level parameters (number of turns,
shared prompt template, final-round instruction).
"""

from dataclasses import dataclass, field


### Shared prompt fragments

# Shared template used by every debater each turn. Placeholders are filled
# at runtime by the orchestrator (string.Template safe_substitute).
#   ${ground_truth}    — task ground truth
#   ${trace}           — parsed agent action trace
#   ${role_description}— persona of the current debater
#   ${agent_name}      — name of the current debater
#   ${final_prompt}    — empty during debate rounds, set to the scoring
#                        instruction in the final round


DEFAULT_PROMPT_TEMPLATE = """\
[Ground Truth]
${ground_truth}

[Agent Action Trace]
${trace}

[System]
We would like your feedback on the performance of an autonomous network
troubleshooting agent, given the ground truth and the action trace above.
Consider relevance, correctness, efficiency, clarity, and final outcome.

There are other expert referees assigned the same task. It is your
responsibility to discuss with them and think critically before making
your final judgement.

${role_description}

Now it's your turn to speak, ${agent_name}. Keep it short and focused.

${final_prompt}
"""


# Instruction injected ONLY in the final round, forcing structured scores.
DEFAULT_FINAL_PROMPT = """\
This is the final round. Provide your final judgement as a JSON object.
Remember: you are not required to match other referees' scores — judge
independently based on the evidence in the trace.

Respond with ONLY a JSON object matching this structure exactly \
— no markdown, no extra text, no different field names:
{
  "scores": {
    "relevance":     {"score": <1-5>, "comment": "<justification>"},
    "correctness":   {"score": <1-5>, "comment": "<justification>"},
    "efficiency":    {"score": <1-5>, "comment": "<justification>"},
    "clarity":       {"score": <1-5>, "comment": "<justification>"},
    "final_outcome": {"score": <1-5>, "comment": "<justification>"}
  },
  "reasoning": "<overall summary of your assessment>"
}
"""


### Dataclasses

@dataclass
class RoleConfig:
    """Configuration for a single debate participant (persona)."""

    name: str
    role_description: str
    temperature: float = 0.3
    # Optional per-role model override. When None, the debate's default model
    # (passed to ChatEvalJudge) is used.
    model: str | None = None
    # Per-role final-round instruction. When None, falls back to
    # DebateConfig.final_prompt.
    final_prompt: str | None = None


@dataclass
class DebateConfig:
    """Top-level configuration for a multi-role debate."""

    roles: list[RoleConfig]
    num_rounds: int = 2   # 1 free-discussion round + 1 final-scoring round
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    final_prompt: str = DEFAULT_FINAL_PROMPT



### Default roster (3 roles) — Critic, Network Engineer, General Operator

_CRITIC = RoleConfig(
    name="Critic",
    temperature=0.2,
    role_description=(
        "You are the Critic, a strict and rigorous referee on this panel. "
        "Your job is to actively look for failures, inefficiencies, incorrect "
        "tool usage, and weak reasoning in the agent's trace. Do not give the "
        "benefit of the doubt: penalize mistakes clearly and challenge claims "
        "that are not supported by evidence in the trace."
    ),
)

_NETWORK_ENGINEER = RoleConfig(
    name="Network Engineer",
    temperature=0.3,
    role_description=(
        "You are the Network Engineer, a domain expert on this referee panel. "
        "Focus on technical correctness: are the diagnostic commands and tool "
        "calls appropriate for the symptoms? Are network outputs interpreted "
        "correctly? Does the agent's reasoning reflect sound networking "
        "knowledge (routing, interfaces, protocols, topology)? Bring "
        "domain-specific evidence to your judgement."
    ),
)

_GENERAL_OPERATOR = RoleConfig(
    name="General Operator",
    temperature=0.5,
    role_description=(
        "You are the General Operator, a pragmatic network operations referee. "
        "Focus on the operational value of the agent's run: is the final "
        "submission actionable? Is the root cause identification clear and "
        "useful for an on-call engineer? Reward partial progress and clear "
        "communication; penalize vague or unusable conclusions."
    ),
)


DEFAULT_DEBATE_CONFIG = DebateConfig(
    roles=[_CRITIC, _NETWORK_ENGINEER, _GENERAL_OPERATOR],
    num_rounds=2,  # 1 free-discussion round + 1 final-scoring round
)