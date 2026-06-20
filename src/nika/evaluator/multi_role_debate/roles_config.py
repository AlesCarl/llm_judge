"""Dataclass-based configuration for the Multi-Role Debate judge.


Each "RoleConfig" defines a single debate participant: name, persona
(role_description), per-role temperature, and optional model override.

"DebateConfig" groups the debate-level parameters (number of turns,
shared prompt template, final-round instruction).
"""

from dataclasses import dataclass, field

from agent.utils.template import CRITERIA_RUBRIC


### Shared prompt fragments

# Shared template used by every debater each turn. Placeholders are filled
# at runtime by the orchestrator .
#   ${ground_truth}    — task ground truth
#   ${trace}           — parsed agent action trace
#   ${role_description}— persona of the current debater
#   ${agent_name}      — name of the current debater
#   ${discussion_prompt}— score-free discussion instruction; set in the
#                        discussion rounds, empty in the final round
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

""" + CRITERIA_RUBRIC + """
There are other expert referees assigned the same task. It is your
responsibility to discuss with them and think critically before making
your final judgement.

${role_description}

Now it's your turn to speak, ${agent_name}. Keep it short and focused.

${discussion_prompt}
${final_prompt}
"""


# Instruction injected ONLY in the discussion rounds: keep them score-free so
# the panel exchanges arguments without anchoring on each other's numbers.
# Numbers are committed for the first time in the final (blind) round — this is
# the precondition that makes A1 (blind final scoring) actually effective.

DEFAULT_DISCUSSION_PROMPT = """\
This is a DISCUSSION round, not a scoring round. Do NOT assign any numeric
scores, ratings, or "X/5" values yet — not even provisional ones. Argue
qualitatively: cite specific evidence from the trace, say where the agent did
well or badly on each criterion, and engage with the other referees' arguments
(agree, push back, or build on them). You will commit your own numeric scores
privately and independently only in the final round.
"""


# Instruction injected ONLY in the final round, forcing structured scores.

DEFAULT_FINAL_PROMPT = """\
This is the final round. Provide your final judgement as a JSON object.
You will NOT see the other referees' final scores. Commit your numbers
independently, based on the discussion so far and the evidence in the trace.

Respond with ONLY a JSON object matching this structure exactly. No
markdown, no extra text, no different field names:
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
    scoring_temperature: float = 0.0   
    model: str | None = None
    final_prompt: str | None = None


@dataclass
class DebateConfig:
    """Top-level configuration for a multi-role debate."""

    roles: list[RoleConfig]
    num_rounds: int = 3   # (num_rounds - 1) discussion rounds + 1 final-scoring round
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    discussion_prompt: str = DEFAULT_DISCUSSION_PROMPT
    final_prompt: str = DEFAULT_FINAL_PROMPT



### Default roster (3 roles) — Critic, Network Engineer, General Operator

_CRITIC = RoleConfig(
    name="Critic",
    temperature=0.2,
    role_description=(
        "You are the Critic, the panel's authority on process rigor and "
        "EFFICIENCY. Score all five criteria, but bring special scrutiny to "
        "whether the agent's actions were efficient and well-ordered, without "
        "redundant or wasted steps. Across every criterion, actively look for "
        "failures, unsupported claims, and weak reasoning — do not give the "
        "benefit of the doubt."
    ),
)

_NETWORK_ENGINEER = RoleConfig(
    name="Network Engineer",
    temperature=0.2,
    role_description=(
        "You are the Network Engineer, the panel's authority on technical "
        "soundness — RELEVANCE and CORRECTNESS — and co-authority with the "
        "General Operator on the FINAL OUTCOME (does the diagnosis technically "
        "match the ground truth?). Score all five criteria, but bring deep "
        "domain expertise to whether the diagnostic commands were appropriate "
        "to the symptoms, whether network outputs were interpreted correctly, "
        "and whether the reasoning reflects sound networking knowledge "
        "(routing, interfaces, protocols, topology)."
    ),
)

_GENERAL_OPERATOR = RoleConfig(
    name="General Operator",
    temperature=0.2,
    role_description=(
        "You are the General Operator, the panel's authority on operational "
        "value: CLARITY and (jointly with the Network Engineer) FINAL OUTCOME. "
        "Score all five criteria, but focus "
        "your expertise on whether the final submission is clear, actionable, "
        "and useful to an on-call engineer, rewarding partial progress and "
        "clear communication while penalizing vague or unusable conclusions."
    ),
)


# Per-criterion competence weights used by the aggregator to compute a
# WEIGHTED panel mean. Each row sums to 1.0. A role's vote counts most on the
# criteria it is the panel authority for (see role_description above). 
COMPETENCE_WEIGHTS: dict[str, dict[str, float]] = {
    "relevance":     {"Critic": 0.2, "Network Engineer": 0.6, "General Operator": 0.2},
    "correctness":   {"Critic": 0.2, "Network Engineer": 0.6, "General Operator": 0.2},
    "efficiency":    {"Critic": 0.6, "Network Engineer": 0.2, "General Operator": 0.2},
    "clarity":       {"Critic": 0.2, "Network Engineer": 0.2, "General Operator": 0.6},
    "final_outcome": {"Critic": 0.2, "Network Engineer": 0.4, "General Operator": 0.4},
}


DEFAULT_DEBATE_CONFIG = DebateConfig(
    roles=[_CRITIC, _NETWORK_ENGINEER, _GENERAL_OPERATOR],
    num_rounds=3,  # 2 free-discussion rounds + 1 final (blind) scoring round
)