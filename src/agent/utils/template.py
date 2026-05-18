LLM_JUDGE_PROMPT_TEMPLATE = """
You are an expert networking engineer acting as a judge.  
You will assess the performance of an autonomous agent given:
- Ground Truth: {ground_truth}
- Action History: {trace}

Evaluation criteria (each scored 1-5):
1. Relevance of the actions to the problem  
2. Correctness of tools/commands used  
3. Efficiency and sequence of actions  
4. Clarity of justification / explanatory reasoning in the agent’s actions  
5. Final outcome: whether the final submission exists and matches the problem ground truth  

Instructions:  
– For the provided agent's actions, briefly comment on its relevance, correctness, and efficiency.  
– Then give an overall evaluation: what worked well, what could be improved.  
– Score each of the 5 criteria individually (1 = poor, 5 = excellent).  
– Provide a final overall score from 1 to 5 with reasoning.

}}
"""






## da provare ... e migliorare --- sono una bozza 



DEBATER_SYSTEM_PROMPT = (
    "You are an expert network engineer serving as a judge panel member. "
    "Your role is to critically evaluate an autonomous troubleshooting agent "
    "based on its action trace and the known ground truth for the problem it was asked to solve."
)

INITIAL_EVALUATION_PROMPT = """\
Evaluate the following agent run.

Ground Truth:
{ground_truth}

Agent Action Trace:
{trace}

Assess the agent across these five criteria (score each 1–5, where 1=poor, 5=excellent):
1. Relevance     – how relevant the agent's actions were to the problem
2. Correctness   – how correct the tools/commands used were
3. Efficiency    – how efficient and well-ordered the actions were
4. Clarity       – how clear and well-explained the reasoning was
5. Final Outcome – whether the final answer matches the ground truth

For each criterion, provide a brief justification and your proposed integer score (1–5).\
"""

REBUTTAL_PROMPT = """\
Your fellow panel member(s) have provided the following assessment(s):

{other_assessments}

Review their arguments carefully. Update your position where they raise valid points \
backed by evidence from the trace, or defend your original assessment if you disagree. \
Give your revised (or maintained) scores for all five criteria with justifications.\
"""

MODERATOR_SYSTEM_PROMPT = (
    "You are the moderator of an expert judge panel evaluating a network troubleshooting agent. "
    "Your task is to determine whether the panel has reached sufficient consensus."
)

MODERATOR_PROMPT = """\
Here are the current assessments from all panel members:

{assessments}

Determine whether the panel has reached consensus, defined as all proposed scores \
being within 1 point of each other on every criterion. \
Respond with ONLY a JSON object — no markdown, no extra text:
{{"consensus": true or false, "summary": "brief summary of agreements and remaining disagreements"}}\
"""


SYNTHESIS_SYSTEM_PROMPT = (
    "You are the final synthesizer for an expert judge panel. "
    "Given the complete debate transcript, produce a balanced and definitive structured evaluation."
)


SYNTHESIS_PROMPT = """\
Below is the full transcript of the judge panel debate:

{debate_transcript}

Based on this deliberation, produce the definitive structured evaluation of the agent's performance.

You MUST respond with a valid JSON object only — no markdown, no extra text, no code blocks.
The JSON must follow exactly this structure:
{{
  "scores": {{
    "relevance":     {{"score": <1-5>, "comment": "<string>"}},
    "correctness":   {{"score": <1-5>, "comment": "<string>"}},
    "efficiency":    {{"score": <1-5>, "comment": "<string>"}},
    "clarity":       {{"score": <1-5>, "comment": "<string>"}},
    "final_outcome": {{"score": <1-5>, "comment": "<string>"}},
    "overall_score": {{"score": <1-5>, "comment": "<string>"}}
  }},
  "overall_evaluation": "<string>",
  "reasoning_for_overall_score": "<string explaining why this overall score was given>"
}}\
"""