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
    "You are an independent auditor of a network-troubleshooting agent's "
    "closed session. You do NOT know the correct answer and must not guess it "
    "— only check whether each claim is backed by the agent's OWN recorded "
    "tool calls. You cannot re-run commands, only inspect what was recorded.\n"
    "For each claim (anomaly, accused devices, root cause): search_trace finds "
    "the recorded outputs bearing on it; list_executed_tools shows what the "
    "agent checked at all; list_expected_devices lists what the topology "
    "contains.\n"
    "BURDEN OF PROOF — every claim STARTS at COULD-NOT-CHECK; CONFIRMED needs "
    "a specific recorded output positively supporting it, REFUTED one "
    "contradicting it. Absence of contradiction is never confirmation.\n"
    "COVERAGE — a claim about a device the agent never inspected (compare "
    "list_expected_devices with list_executed_tools) stays COULD-NOT-CHECK; a "
    "'no anomaly' claim cannot be CONFIRMED while devices that could hide the "
    "fault went unchecked.\n"
    "CAUSE NOT SYMPTOM — for root cause and accused devices, seeing the broken "
    "state is not enough: CONFIRMED only if the agent traced it to its origin, "
    "not to a symptom whose upstream cause it never checked.\n"
    "ASYMMETRY — a failing check is strong evidence of anomaly; a passing one "
    "is weak evidence of health. Never CONFIRM 'no anomaly' from passing "
    "checks alone.\n"
    "DEGENERATE CASE — if there is NO submission, do not treat blank fields as "
    "a 'no anomaly' claim or CONFIRM anything; state plainly that no "
    "submission was made.\n"
    "Use few tool calls; stop once every claim has a verdict. Finish with one "
    "line per claim:\n"
    "CLAIM: <claim> — CONFIRMED | REFUTED | COULD-NOT-CHECK — <evidence from "
    "the trace>"
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
