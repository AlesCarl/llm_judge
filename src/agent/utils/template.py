# Shared scoring rubric — IDENTICAL across all three judges (single, multi,
# multi_role). It defines the measurement instrument (what each criterion means
# and what a 1/3/5 looks like) so that differences between judge architectures
# reflect the architecture, not the prompt wording. Keep it free of '{', '}'
# and '$' so it can be concatenated into both str.format() and string.Template
# prompts without escaping.
CRITERIA_RUBRIC = """\
Assess the agent on these five criteria. Score each from 1 to 5 (1 = poor, 5 = excellent):

1. Relevance — how relevant the agent's actions were to the stated problem.
   5 = every action targeted the actual problem; 3 = mix of on-point and off-target actions; 1 = actions largely unrelated.

2. Correctness — whether the tools/commands were used correctly and their outputs interpreted soundly.
   5 = correct tools, correct usage, correct interpretation; 3 = minor errors or misreadings; 1 = wrong tools or fundamentally wrong interpretation.

3. Efficiency — how efficient and well-ordered the actions were, without redundant or wasted steps.
   5 = direct, no wasted steps; 3 = some redundancy or detours; 1 = highly redundant or aimless.

4. Clarity — how clear and well-explained the agent's reasoning and justifications were.
   5 = reasoning explicit and easy to follow; 3 = partially explained; 1 = opaque or absent.

5. Final Outcome — whether the agent produced a final submission AND it matches the ground truth (root cause and faulty devices).
   5 = submission exists and fully matches; 3 = submission exists but partially correct or incomplete; 1 = no submission, or it is wrong.
"""


LLM_JUDGE_PROMPT_TEMPLATE = """\
You are an expert networking engineer acting as a judge.
You will assess the performance of an autonomous agent given:
- Ground Truth: {ground_truth}
- Action History: {trace}

""" + CRITERIA_RUBRIC + """
Instructions:
- Briefly comment on the agent's relevance, correctness, and efficiency.
- Give an overall evaluation: what worked well and what could be improved.
- Score each of the five criteria individually, following the rubric above.
"""




## da provare ... e migliorare --- sono una bozza


## CRITIC_SYSTEM_PROMPT / ADVOCATE_SYSTEM_PROMPT -- al posto di uno solo

CRITIC_SYSTEM_PROMPT = (
    "You are a strict expert network engineer serving as a judge panel member. "
    "Your role is to rigorously evaluate an autonomous troubleshooting agent, "
    "actively looking for failures, inefficiencies, and incorrect reasoning. "
    "Penalize mistakes clearly and do not give the benefit of the doubt."
)

ADVOCATE_SYSTEM_PROMPT = (
    "You are a fair expert network engineer serving as a judge panel member. "
    "Your role is to evaluate an autonomous troubleshooting agent giving credit "
    "where actions were reasonable or partially correct, and considering the difficulty "
    "of the problem. Recognize effort and partial progress where warranted."
)

### DEBATER_SYSTEM_PROMPT = CRITIC_SYSTEM_PROMPT



_DEBATER_JSON_SCHEMA = """\
{{
  "scores": {{
    "relevance":     {{"score": <1-5>, "comment": "<justification>"}},
    "correctness":   {{"score": <1-5>, "comment": "<justification>"}},
    "efficiency":    {{"score": <1-5>, "comment": "<justification>"}},
    "clarity":       {{"score": <1-5>, "comment": "<justification>"}},
    "final_outcome": {{"score": <1-5>, "comment": "<justification>"}}
  }},
  "reasoning": "<overall summary of your assessment>"
}}\
"""


INITIAL_EVALUATION_PROMPT = """\
Evaluate the following agent run.

Ground Truth:
{ground_truth}

Agent Action Trace:
{trace}

""" + CRITERIA_RUBRIC + """
Respond with ONLY a JSON object matching this structure — no markdown, no extra text:
""" + _DEBATER_JSON_SCHEMA



REBUTTAL_PROMPT = """\
Your fellow panel member has provided the following assessment:

{other_assessments}

Review their arguments carefully. Update your position where they raise valid points \
backed by evidence from the trace, or defend your original assessment if you disagree. \
Respond with ONLY a JSON object in the same structure as before — no markdown, no extra text.\
"""



# MODERATOR_SYSTEM_PROMPT = (
#     "You are the moderator of an expert judge panel evaluating a network troubleshooting agent. "
#     "Your task is to determine whether the panel has reached sufficient consensus on the "
#     "criteria where scores still diverge."
# )
#
#
# MODERATOR_PROMPT = """\
# The panel members' scores differ on the following criteria (scores shown per member):
# {divergent_criteria}
#
# Here are their full assessments for context:
#
# {assessments}
#
# Determine whether the reasoning behind these divergent scores is fundamentally aligned \
# (members agree on the facts but weight them differently) or genuinely conflicting \
# (members interpret the evidence differently).
#
# Respond with ONLY a JSON object — no markdown, no extra text:
# {{"consensus": true or false, "summary": "brief explanation of agreements and remaining disagreements"}}\
# """



SYNTHESIS_SYSTEM_PROMPT = (
    "You are the final judge of an expert panel evaluating a network troubleshooting agent. "
    "You have access to the ground truth, the agent's action trace, the full debate transcript, "
    "and the debaters' final per-criterion scores. Render a definitive, evidence-grounded "
    "structured evaluation."
)


# Injected into SYNTHESIS_PROMPT when the debaters reached numerical consensus
# (every criterion within 1 point). The judge ratifies the agreement and must
# stay inside the band the debaters converged to.
SYNTHESIS_CONSENSUS_INSTRUCTION = """\
The debaters reached consensus: their final scores are within 1 point on every criterion.
Your role is to CONFIRM their agreement. For each criterion, your score MUST fall within the
[min, max] range of the debaters' final scores shown above. Do not overturn the consensus —
choose the value within that range that the trace evidence best supports."""


# Injected into SYNTHESIS_PROMPT when the debate ended WITHOUT consensus
# (max rounds reached, some criteria still diverge by >1). The judge arbitrates.
SYNTHESIS_NO_CONSENSUS_INSTRUCTION = """\
The debaters did NOT reach consensus: on some criteria their final scores still diverge by
more than 1 point. Your role is to ARBITRATE. For each divergent criterion, decide the score
yourself based on the evidence in the ground truth and the action trace — weigh the Critic's
concerns against the Advocate's credits. Your score may sit anywhere within the debaters'
range, but it must be justified by the facts in the trace, not by splitting the difference."""


SYNTHESIS_PROMPT = """\
[Ground Truth]
{ground_truth}

[Agent Action Trace]
{trace}

[Debate Transcript]
{debate_transcript}

[Debaters' Final Scores]
{debater_votes}

{mode_instruction}

Produce the definitive structured evaluation of the agent's performance, grounded in the
evidence above.

You MUST respond with a valid JSON object only — no markdown, no extra text, no code blocks.
The JSON must follow exactly this structure:
{{
  "scores": {{
    "relevance":     {{"score": <1-5>, "comment": "<string>"}},
    "correctness":   {{"score": <1-5>, "comment": "<string>"}},
    "efficiency":    {{"score": <1-5>, "comment": "<string>"}},
    "clarity":       {{"score": <1-5>, "comment": "<string>"}},
    "final_outcome": {{"score": <1-5>, "comment": "<string>"}}
  }},
  "overall_evaluation": "<string>",
  "reasoning_for_overall_score": "<string explaining why this overall score was given>"
}}\
"""
