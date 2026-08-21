# Shared scoring rubric — IDENTICAL across all three judges (single, multi,
# multi_role). It defines the measurement instrument (what each criterion means
# and what a 1/3/5 looks like) so that differences between judge architectures
# reflect the architecture, not the prompt wording. Keep it free of '{', '}'
# and '$' so it can be concatenated into both str.format() and string.Template
# prompts without escaping.

CRITERIA_RUBRIC = """\
Assess the agent on these five criteria. Score each from 1 to 5 (1 = poor, 5 = excellent).
Criteria 1-4 judge the agent's decision-making process, considering only the information
available to it at each step; only criterion 5 judges the outcome against the ground truth.

1. Relevance — how relevant the agent's actions were to the stated problem.
   5 = every action targeted the stated problem;
   3 = mix of on-point and off-target actions;
   1 = actions largely unrelated, or aimless.

2. Correctness — whether the tools/commands were used correctly and their outputs interpreted soundly.
   5 = correct tools, correct usage, correct interpretation;
   3 = minor errors or misreadings;
   1 = wrong tools or fundamentally wrong interpretation.

3. Efficiency — how efficient and well-ordered the actions were, without redundant or wasted steps.
   5 = no repeated steps, sensible order;
   3 = some repeated steps or avoidable back-tracking;
   1 = highly redundant or repetitive.

4. Clarity — how clear and well-explained the agent's reasoning and justifications were.
   5 = reasoning explicit and easy to follow, even if wrong;
   3 = partially explained;
   1 = opaque, or no reasoning stated.

5. Final Outcome — whether the agent produced a final submission AND it matches the ground truth (root cause and faulty devices).
   5 = submission exists and fully matches;
   3 = submission exists but partially correct or incomplete;
   1 = no submission, or it is wrong.
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




SYNTHESIS_SYSTEM_PROMPT = (
    "You are the final judge of an expert panel evaluating a network troubleshooting agent. "
    "You have access to the ground truth, the agent's action trace, the full debate transcript, "
    "and the debaters' final per-criterion scores. Render a definitive, evidence-grounded "
    "structured evaluation."
)


# Injected into SYNTHESIS_PROMPT regardless of whether the debaters reached
# numerical consensus. The debaters' scores are evidence for the judge, not a
# constraint: the judge is free to score outside their [min, max] band when the
# ground truth and action trace support it.
SYNTHESIS_FREE_INSTRUCTION = """\
You are the final authority on this evaluation. The debaters' scores shown above are
INPUT to your decision, not a constraint on it. You may agree with them, or overturn
either or both — including choosing a score outside the range they settled on — when
the ground truth and the action trace support it.

For each criterion, decide the score yourself from the evidence: what the agent actually
did in the trace, and whether it matches the ground truth. Do not defer to the debaters
because they agreed with each other, and do not split the difference when they disagreed.
Where your score departs from both debaters, state the reason in that criterion's comment.

Work through these steps, and place each one in the field named below:
1. Briefly summarise each debater's key reasoning.        -> reasoning_for_overall_score
2. State explicitly, for each debater, whether you agree
   or disagree with their assessment, and why.            -> reasoning_for_overall_score
3. Give your own full evaluation of the agent's
   performance, independent of theirs.                    -> overall_evaluation
4. Assign the final per-criterion scores.                 -> scores

Steps 1-2 are mandatory: engage with what the debaters argued before deciding. Engaging
with them is not the same as following them — you are free to conclude that both were wrong."""


SYNTHESIS_PROMPT = """\
[Ground Truth]
{ground_truth}

[Agent Action Trace]
{trace}

[Debate Transcript]
{debate_transcript}

[Debaters' Final Scores]
{debater_votes}

""" + CRITERIA_RUBRIC + """
{mode_instruction}

Produce the definitive structured evaluation of the agent's performance, grounded in the
evidence above and scored strictly according to the rubric.

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
