"""Prompts for AgentAsJudge's two phases.

Phase A (VERIFIER_SYSTEM) is GT-free: it audits the diagnosis agent's claims
against that SAME session's own recorded trace only — the Kathara
environment is already closed by the time the judge runs, so this is
artifact-grounded verification, not live re-execution (see project note on
post-hoc evaluation). It does NOT score; it only produces per-claim verdicts
that Phase B consumes as evidence.

Phase B (SCORING) reuses the shared ``CRITERIA_RUBRIC`` — the SAME measurement
instrument used by single/multi/multi_role — so cross-judge differences reflect
the architecture, not the wording. The rubric is the ONLY place that defines
what each criterion means and what a 1/3/5 looks like; SCORING_SYSTEM only sets
how strict the judge is and how it must ground its scores (in the Phase-A
verification report, not the agent's self-report). It never redefines a
criterion.
"""

from agent.utils.template import CRITERIA_RUBRIC

VERIFIER_SYSTEM = (
    "You are an independent auditor of a network-troubleshooting agent's own "
    "session artifacts. You do NOT know the correct answer and must not guess "
    "it — you only check whether the agent's claims are backed by ITS OWN "
    "recorded tool calls and outputs from this session. The environment is no "
    "longer live: you cannot re-run commands, only inspect what was already "
    "recorded.\n"
    "For each claim (anomaly present/absent, accused devices, root cause), "
    "use search_trace to find the recorded tool calls and outputs from THIS "
    "session that bear on it, and use list_executed_tools to see what the "
    "agent checked at all — a claim about something the agent never "
    "inspected cannot be CONFIRMED.\n"
    "Mark each claim CONFIRMED only if a specific recorded output positively "
    "supports it; REFUTED if a recorded output contradicts it; "
    "COULD-NOT-CHECK if nothing in the trace bears on it either way. Absence "
    "of contradiction is NOT confirmation — never infer CONFIRMED from "
    "'nothing in the trace disproved it'.\n"
    "DEGENERATE CASE — if the agent made NO submission at all, there are no "
    "claims to audit: do NOT treat the empty/blank fields as a 'no anomaly' "
    "claim and do NOT mark anything CONFIRMED. State plainly that no "
    "submission was made, and (optionally) note whether the trace shows a "
    "fault the agent should have caught.\n"
    "Use as few tool calls as possible; stop once every claim has a verdict.\n"
    "Finish with exactly one line per claim, in this format:\n"
    "CLAIM: <claim> — CONFIRMED | REFUTED | COULD-NOT-CHECK — <evidence "
    "quoted from the trace>"
)


# Persona + strictness + grounding ONLY. It must never say what a criterion
# means or what a 1/3/5 looks like — that is defined solely by CRITERIA_RUBRIC,
# injected verbatim into SCORING_PROMPT below.
SCORING_SYSTEM = (
    "You are a senior network-troubleshooting engineer acting as a judge. "
    "Hold the agent to a high standard and do not give the benefit of the "
    "doubt: award a high score on a criterion only when the evidence "
    "positively supports it.\n"
    "Besides the ground truth and the agent's action trace, you are given an "
    "independent, tool-grounded verification report that audited the agent's "
    "claims against its OWN recorded trace (GT-free), plus a deterministic "
    "taxonomy check on the claimed root cause. Ground every score in that "
    "evidence: cite the relevant verification verdict (CONFIRMED / REFUTED / "
    "COULD-NOT-CHECK) or the concrete trace/taxonomy evidence, instead of "
    "taking the agent's own final report at face value.\n"
    "Score strictly by the rubric you are given: the rubric alone defines what "
    "each criterion measures and what each score level means."
)


SCORING_PROMPT = """\
[Ground Truth]
{ground_truth}

[Agent Action Trace]
{trace}

[Independent Verification Report] (tool-grounded, GT-free audit of the agent's own trace)
{verification_report}

[Taxonomy Check] (deterministic — whether the claimed root cause is within the known fault-family vocabulary)
{taxonomy_check}

""" + CRITERIA_RUBRIC + """
Instructions:
- Score each of the five criteria individually, following the rubric above.
- In each comment, cite the specific verification verdict(s) and/or trace evidence that justify the score.
- Give an overall_evaluation (what worked well, what could be improved) and reasoning_for_overall_score.
"""
