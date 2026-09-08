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
#
# The three roles differ by their FUNCTION in the debate, not by how severe they
# are: the Critic argues against the agent, the General Operator argues for it,
# and the Network Engineer checks the factual basis of both. Each prompt says
# what evidence that role must look for and what it contributes to the
# discussion; what a criterion means and what a 1/3/5 looks like is left
# entirely to the shared CRITERIA_RUBRIC, so all four judge architectures are
# measured with the same instrument.

_CRITIC = RoleConfig(
    name="Critic",
    temperature=0.2,
    role_description=(
        "You are the Critic. Your role in the panel is to identify "
        "evidence-supported weaknesses in the agent's work. "
        "Look for unsupported claims, overlooked evidence, and gaps "
        "between the agent's conclusions and its recorded observations. "
        "When another referee defends the agent, examine whether that "
        "defense is supported by the trace. Cite concrete evidence for "
        "each criticism, distinguish demonstrated errors from uncertainty, "
        "and withdraw objections when the evidence resolves them. "
        "Apply the shared rubric exactly as written when assigning scores."
    ),
)

_NETWORK_ENGINEER = RoleConfig(
    name="Network Engineer",
    temperature=0.2,
    role_description=(
        "You are the Network Engineer. Your role in the panel is to establish "
        "what is technically supported by the trace, using your networking "
        "expertise. Check the factual basis of both criticisms and defenses, "
        "and state which claims the recorded evidence supports, contradicts, "
        "or leaves unresolved. "
        "Apply the shared rubric exactly as written when assigning scores."
    ),
)

_GENERAL_OPERATOR = RoleConfig(
    name="General Operator",
    temperature=0.2,
    role_description=(
        "You are the General Operator. Your role in the panel is to identify "
        "and defend evidence-supported strengths in the agent's work. "
        "Highlight correct intermediate steps, partial progress, and "
        "supported reasoning that other referees may have overlooked. "
        "When responding to criticism, cite concrete evidence from the trace; "
        "do not invent strengths or defend unsupported claims. "
        "Apply the shared rubric exactly as written when assigning scores."
    ),
)


# Per-criterion vote weights used by the aggregator. The panel is differentiated
# by dialectical FUNCTION (the Critic accuses, the Network Engineer establishes
# what the trace supports, the General Operator credits), not by domain
# competence: weighting one function above the others on a criterion would be a
# systematic severity bias on that criterion, not an expertise argument. Votes
# are therefore equal — the empty mapping makes aggregate_responses() fall back
# to an unweighted panel mean.
COMPETENCE_WEIGHTS: dict[str, dict[str, float]] = {}


DEFAULT_DEBATE_CONFIG = DebateConfig(
    roles=[_CRITIC, _NETWORK_ENGINEER, _GENERAL_OPERATOR],
    num_rounds=3,  # 2 free-discussion rounds + 1 final (blind) scoring round
)