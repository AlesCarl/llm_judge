"""GT-free in-loop feedback (verifier coach) for the retry loop in ``BasicReActAgent``.

The in-loop judge NO LONGER sees the ground truth: everything it tells the
agent must be *extracted* (from the agent's own evidence or from the live
network via read-only tools), never handed down from the answer key. The GT
is used only by the offline evaluator (``nika.workflows.eval``) after the run.

Pipeline per attempt (class ``VerifierCoach``):

1. VERIFY (optional — the ablation flag ``verifier_tools`` controls it):
   a small ReAct agent with the same read-only diagnostic MCP tools audits
   the 2-4 load-bearing claims of the agent's submission against the live
   network and reports CONFIRMED / REFUTED / COULD-NOT-CHECK per claim.

2. REVIEW: a single LLM call grades each answer dimension
   (detection / localization / root_cause) as SUPPORTED / WEAK / UNSUPPORTED
   based on the evidence (verification report first, agent's own trace
   second), picks the most consistent fault family from the closed family
   list (``known_families``, for coach context only — no per-scenario
   catalog is exposed), extracts new case-file facts, and writes a short
   redirect hint.

3. ``compose_feedback`` assembles the deterministic message injected into the
   next attempt: submission echo + per-dimension verdict + verifier
   observations + the hint.

Kept deliberately generic (no Kathara-specific fault catalog): the fault
family taxonomy (``known_families``) is a defensible general vocabulary for
network faults, but the per-scenario sub-cause "differential card" (keyed to
the simulator's problem registry, one discriminator string per scenario) was
removed so this module has no simulation-specific coupling beyond the family
names themselves.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from langsmith import tracing_context

from agent.utils.loggers import AgentCallbackLogger

_DIMS = ("detection", "localization", "root_cause")
_STATUSES = ("SUPPORTED", "WEAK", "UNSUPPORTED")

# Step budget (recursion_limit) for one verification pass: ~5 tool calls.
_VERIFY_BUDGET = 12

# Cap on the accumulated cross-attempt case file (facts, newest kept).
_CASE_FILE_CAP = 25


_VERIFIER_SYSTEM = (
    "You are an independent network fault verifier. You are given a junior "
    "agent's diagnosis claims; you do NOT know the correct answer. Audit them "
    "against the live network with as few read-only tool calls as possible — "
    "inspect, never change; do not re-diagnose the whole network.\n"
    "For the 2-4 claims that carry the diagnosis (root cause, accused devices, "
    "the anomaly itself), run the single most direct check that could confirm "
    "or refute each, starting from the accused devices; if a check refutes "
    "one, note what you saw instead.\n"
    "A FAILING check confirms an anomaly claim; a PASSING one alone never "
    "refutes it (reachability varies by path and timing) — mark REFUTED only "
    "after reproducing the reported symptom from the accused devices, else "
    "COULD-NOT-CHECK.\n"
    "When the claimed cause is itself a broken state (a missing or wrong "
    "address, route, name, lease), the most direct check IS its provider: a "
    "serving daemon, a peer, or the local static config. Provider down = the "
    "claim is only an effect, REFUTED; provider serving correctly = the cause "
    "stands. A broken local config counts as the provider being down, not "
    "healthy.\n"
    "DEGENERATE CASE — if the submission claims NO anomaly (no cause AND no "
    "devices) there is nothing to refute: instead try to DISPROVE health by "
    "probing what the agent's reasoning never examined — hosts or segments it "
    "did not mention — rather than re-confirming paths it already saw. Confirm "
    "health only if those blind-spot checks ALSO pass (reachability, interface/"
    "service status, name-resolution and default-route from an end host). "
    "Add a final line: HEALTH: CONFIRMED | UNCONFIRMED — <evidence>.\n"
    "Finish with one line per claim (mark CONFIRMED only when the state you "
    "observed is the exact one named in the claim, not merely some fault):\n"
    "CLAIM: <claim> — CONFIRMED | REFUTED | COULD-NOT-CHECK — <evidence observed>"
)


_COACH_SYSTEM = (
    "You are a senior network troubleshooting reviewer. A junior agent "
    "diagnosed a network fault. You do NOT know the correct answer — judge "
    "only whether each part of its answer is backed by the available evidence "
    "(the independent verification report, when present, outweighs the "
    "agent's own reasoning).\n"
    "BURDEN OF PROOF — every dimension STARTS at WEAK. SUPPORTED = a positive, "
    "specific check actively confirms it; UNSUPPORTED = a check contradicts it; "
    "WEAK = plausible but unconfirmed. Absence of contradiction is NOT evidence "
    "— never infer SUPPORTED from 'nothing disproved it'.\n"
    "Grade three dimensions:\n"
    "- detection: is the anomaly / no-anomaly call justified by evidence?\n"
    "- localization: are the accused devices actually implicated?\n"
    "- root_cause: SUPPORTED only if the evidence shows the NAMED state "
    "itself AND that it is not just the effect of something upstream: a cause "
    "that re-states the symptom with its provider never checked stays WEAK; "
    "evidence showing a DIFFERENT broken state than the named one is a "
    "contradiction = UNSUPPORTED, and the hint must name what was seen.\n"
    "SCOPE — a refutation observed only on the accused devices refutes "
    "localization, not the mechanism: grade root_cause WEAK there and hint "
    "to test the same mechanism on sibling devices before abandoning it.\n"
    "DEGENERATE CASE — a 'no anomaly' submission with no devices and no cause "
    "makes NO falsifiable claim: grade all three WEAK, UNLESS the verifier "
    "positively confirmed health. Set health_positively_confirmed=true ONLY if "
    "the verification report contains an explicit 'HEALTH: CONFIRMED' line.\n"
    "Write a hint (2-3 sentences) ONLY about the non-SUPPORTED dimensions: name "
    "the missing check or the contradiction concretely; never write the "
    "diagnosis for the agent.\n"
    "Pick suspected_family: the fault family most consistent with the evidence, "
    "from KNOWN FAULT FAMILIES, or \"unsure\".\n"
    "List up to 5 new_facts: short factual observations established this attempt "
    "(device/interface states, outputs seen) — facts only, no conclusions.\n"
    "List in superseded_facts the text of any FACTS ALREADY ON RECORD that this "
    "attempt's evidence now contradicts or makes outdated (empty if none).\n"
    "For each dimension write \"why\" (the evidence checked) BEFORE \"status\".\n"
    "Reply with ONLY this JSON (no other text):\n"
    '{"submission_had_concrete_claims": true, '
    '"health_positively_confirmed": false, '
    '"detection": {"why": "...", "status": "..."}, '
    '"localization": {"why": "...", "status": "..."}, '
    '"root_cause": {"why": "...", "status": "..."}, '
    '"suspected_family": "...", "new_facts": ["..."], '
    '"superseded_facts": ["..."], "hint": "..."}'
)


def _head_tail(s: str, head: int = 700, tail: int = 800) -> str:
    """Keep the report's head AND tail: the CLAIM/HEALTH verdict lines the
    coach grades on live at the end and must survive truncation."""
    return s if len(s) <= head + tail else s[:head] + "\n…\n" + s[-tail:]


# --------------------------------------------------------------------------
# messages.jsonl readers (deterministic, no LLM)
# --------------------------------------------------------------------------

def _tool_events(session_dir: str) -> list[dict]:
    path = Path(session_dir) / "messages.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events


def _clean_tool_output(raw: str, max_chars: int) -> str:
    """Best-effort extraction of the content field from a str(ToolMessage)."""
    m = re.search(r"content=['\"](.*?)['\"]\s+\w+=", raw, flags=re.DOTALL)
    text = m.group(1) if m else raw
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def attempt_digest(session_dir: str, max_calls: int = 15) -> str:
    """Evidence digest of the LAST diagnosis attempt (deterministic, no LLM).

    Slices messages.jsonl at the most recent ``attempt_start`` marker (written
    by the diagnosis node), then pairs each diagnosis-agent tool_start with
    its tool_end/tool_error so the coach sees WHAT each check returned, not
    just which tools ran. Long transcripts keep the last *max_calls* calls in
    detail plus an aggregate count line.
    """
    events = _tool_events(session_dir)

    # Keep only the current attempt.
    last_marker = -1
    for i, e in enumerate(events):
        if e.get("event") == "attempt_start":
            last_marker = i
    events = events[last_marker + 1:]

    calls: list[dict] = []  # {"name":…, "input":…, "output":…}
    pending: list[dict] = []
    for e in events:
        if e.get("agent") != "diagnosis_agent":
            continue
        ev = e.get("event")
        if ev == "tool_start":
            tool = e.get("tool")
            name = tool.get("name") if isinstance(tool, dict) else str(tool)
            call = {"name": name or "unknown_tool", "input": str(e.get("input") or "")[:100], "output": ""}
            calls.append(call)
            pending.append(call)
        elif ev in ("tool_end", "tool_error") and pending:
            call = pending.pop(0)
            raw = str(e.get("output") or e.get("error") or "")
            prefix = "ERROR: " if ev == "tool_error" else ""
            call["output"] = prefix + _clean_tool_output(raw, 180)

    if not calls:
        return "Previous attempt made no tool calls."

    counts: dict[str, int] = {}
    for c in calls:
        counts[c["name"]] = counts.get(c["name"], 0) + 1
    summary = ", ".join(f"{name}×{n}" for name, n in counts.items())

    detailed = calls[-max_calls:]
    lines = [f"{len(calls)} tool calls ({summary}). Last {len(detailed)} in detail:"]
    for c in detailed:
        lines.append(f"- {c['name']}({c['input']}) -> {c['output'] or '(no output captured)'}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Fault-family knowledge (from the problem registry; GT-independent content)
# --------------------------------------------------------------------------

def known_families() -> list[str]:
    """Sorted closed list of fault-family names from the problem registry."""
    from nika.orchestrator.problems.prob_pool import _PROBLEMS

    fams: set[str] = set()
    for levels in _PROBLEMS.values():
        cls = next(iter(levels.values()), None)
        if cls is not None:
            fams.add(str(cls.META.root_cause_category))
    return sorted(fams)


# --------------------------------------------------------------------------
# Coach review
# --------------------------------------------------------------------------

@dataclass
class CoachReview:
    """Parsed outcome of one coach review (GT-free)."""

    statuses: dict[str, tuple[str, str]]  # dim -> (STATUS, why)
    suspected_family: str = ""
    new_facts: list[str] = field(default_factory=list)
    superseded_facts: list[str] = field(default_factory=list)  # earlier facts now outdated
    hint: str = ""
    verification_report: str = ""
    verified: bool = False  # True when the tool-using verification pass ran
    # Coach's self-report on the degenerate 'null answer' case (see review()).
    health_confirmed: bool = False  # verifier positively confirmed a healthy net
    raw: str = ""  # raw LLM grade reply, kept for debugging (parse failures etc.)

    @property
    def approved(self) -> bool:
        """Coach-side stop signal: every dimension graded SUPPORTED."""
        return all(st == "SUPPORTED" for st, _ in self.statuses.values())


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_review(text: str) -> CoachReview:
    """Lenient JSON parse of the coach reply; degrades to all-WEAK + raw hint."""
    text = _strip_think(text)
    fallback = CoachReview(
        statuses={dim: ("WEAK", "review unparseable") for dim in _DIMS},
        hint=text[:600],
    )
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return fallback
    try:
        data = json.loads(m.group(0))
    except Exception:
        return fallback

    statuses: dict[str, tuple[str, str]] = {}
    for dim in _DIMS:
        entry = data.get(dim) or {}
        status = str(entry.get("status", "")).strip().upper()
        if status not in _STATUSES:
            status = "WEAK"
        statuses[dim] = (status, str(entry.get("why", "")).strip()[:220])
    return CoachReview(
        statuses=statuses,
        suspected_family=str(data.get("suspected_family", "")).strip(),
        new_facts=[str(f).strip()[:160] for f in (data.get("new_facts") or []) if str(f).strip()][:5],
        superseded_facts=[str(f).strip()[:160] for f in (data.get("superseded_facts") or []) if str(f).strip()][:5],
        hint=str(data.get("hint", "")).strip()[:800],
        health_confirmed=bool(data.get("health_positively_confirmed", False)),
    )


def _is_null_answer(submission: dict) -> bool:
    """True for a 'surrender' answer: no anomaly, no accused device, no cause.

    Such a submission makes no falsifiable claim, so the coach must not be
    allowed to mark it SUPPORTED just because nothing contradicted it.
    """
    return (
        not submission.get("is_anomaly")
        and not (submission.get("faulty_devices") or [])
        and not (submission.get("root_cause_name") or [])
    )


def _fact_superseded(fact: str, superseded: list[str]) -> bool:
    """Best-effort match of a stored fact against the coach's superseded list.

    Substring both ways so a paraphrase still matches; on no match the fact is
    kept (degrades to the old behaviour — supersession never loses a fact wrongly).
    """
    fl = fact.lower()
    return any(s and (s.lower() in fl or fl in s.lower()) for s in superseded)


def merge_case_file(
    existing: list[str], new_facts: list[str], superseded: list[str] | None = None
) -> list[str]:
    """Drop coach-flagged superseded facts, append new non-duplicates, cap to newest."""
    superseded = superseded or []
    merged = [f for f in existing if not _fact_superseded(f, superseded)]
    seen = {f.lower() for f in merged}
    for fact in new_facts:
        if fact.lower() not in seen:
            merged.append(fact)
            seen.add(fact.lower())
    return merged[-_CASE_FILE_CAP:]


class VerifierCoach:
    """GT-free in-loop reviewer: optional tool-grounded verification + review.

    ``tools=None`` is the critique-only ablation arm (no verification pass);
    with tools it becomes the grounded verifier (CRITIC-style).
    """

    def __init__(self, llm, tools=None, session_dir: str = "", verify_budget: int = _VERIFY_BUDGET):
        self.llm = llm
        self.session_dir = session_dir
        self.verify_budget = verify_budget
        self.verifier_agent = None
        if tools:
            self.verifier_agent = create_react_agent(
                model=llm,
                tools=tools,
                prompt=_VERIFIER_SYSTEM,
            )

    async def _verify(self, submission: dict, diagnosis_report: str) -> str:
        """Run the tool-grounded audit of the submission's claims."""
        request = (
            "Claims to audit (the junior agent's submission):\n"
            f"  anomaly present : {submission.get('is_anomaly')}\n"
            f"  root cause      : {submission.get('root_cause_name', [])}\n"
            f"  accused devices : {submission.get('faulty_devices', [])}\n\n"
            f"Agent's reasoning for context:\n{diagnosis_report[:1500]}\n\n"
            "Audit the claims now and produce the per-claim report."
        )
        try:
            result = await self.verifier_agent.ainvoke(
                {"messages": [HumanMessage(content=request)]},
                config={
                    "callbacks": [AgentCallbackLogger(agent="verifier_coach", session_dir=self.session_dir)],
                    "recursion_limit": self.verify_budget,
                },
            )
            return _head_tail(_strip_think(str(result["messages"][-1].content)))
        except GraphRecursionError:
            return "(verification ran out of budget before completing)"
        except Exception as exc:  # audit must never kill the loop
            return f"(verification failed: {exc})"

    async def review(
        self,
        *,
        task_description: str,
        submission: dict,
        diagnosis_report: str,
        digest: str,
        no_submission: bool,
        case_file: list[str] | None = None,
    ) -> CoachReview:
        """Full review pass: (verify) + grade + hint + facts."""
        verification_report = ""
        verified = False
        if self.verifier_agent is not None and not no_submission:
            verification_report = await self._verify(submission, diagnosis_report)
            verified = not verification_report.startswith("(verification")

        sub_txt = (
            "(the agent did NOT submit — it ran out of steps)"
            if no_submission
            else (
                f"  anomaly present : {submission.get('is_anomaly')}\n"
                f"  root cause      : {submission.get('root_cause_name', [])}\n"
                f"  accused devices : {submission.get('faulty_devices', [])}"
            )
        )
        on_record = "\n".join(f"- {f}" for f in (case_file or [])) or "(none)"
        human = (
            f"TASK GIVEN TO THE AGENT:\n{task_description[:600]}\n\n"
            f"AGENT'S SUBMISSION:\n{sub_txt}\n\n"
            f"AGENT'S REASONING:\n{(diagnosis_report or '(none)')[:2200]}\n\n"
            f"AGENT'S TOOL ACTIVITY:\n{digest[:2500]}\n\n"
            f"INDEPENDENT VERIFICATION REPORT:\n{verification_report or '(not available)'}\n\n"
            f"FACTS ALREADY ON RECORD (earlier rounds):\n{on_record}\n\n"
            f"KNOWN FAULT FAMILIES: {', '.join(known_families())}\n\n"
            "Write the JSON review now."
        )
        with tracing_context(enabled=False):
            response = self.llm.invoke([
                SystemMessage(content=_COACH_SYSTEM),
                HumanMessage(content=human),
            ])
        raw = str(getattr(response, "content", ""))
        review = _parse_review(raw)
        review.raw = raw
        review.verification_report = verification_report
        review.verified = verified
        # Deterministic gate: never trust the LLM's health flag unless the
        # verifier actually wrote the confirmation line it is keyed to.
        review.health_confirmed = review.health_confirmed and ("HEALTH: CONFIRMED" in verification_report)
        if no_submission:
            # No claims were made: nothing can be SUPPORTED.
            review.statuses = {dim: ("UNSUPPORTED", "no submission was made") for dim in _DIMS}
        elif _is_null_answer(submission) and not review.health_confirmed:
            # Deterministic guard (do not trust the LLM to apply this rule):
            # a 'no anomaly / no device / no cause' answer makes no falsifiable
            # claim, so "nothing disproved it" must NOT become SUPPORTED. Cap
            # every SUPPORTED down to WEAK unless the verifier positively
            # confirmed the network is healthy. Surrender != clean diagnosis.
            review.statuses = {
                dim: (("WEAK", "null answer: no positive proof of health") if st == "SUPPORTED" else (st, why))
                for dim, (st, why) in review.statuses.items()
            }
        return review


# --------------------------------------------------------------------------
# Feedback assembly (deterministic; injected into the next attempt)
# --------------------------------------------------------------------------

_DIM_LABELS = {"detection": "detection ", "root_cause": "root cause", "localization": "devices   "}


def compose_feedback(
    *,
    review: CoachReview,
    submission: dict,
    no_submission: bool,
) -> str:
    """Assemble the self-framed retry REVIEW block from the coach review (no GT).

    Only the review lives here; the caller appends the case-file facts, the tool
    digest and the closing action line. The keep/rework instruction and the
    grounded evidence each appear exactly once (no verifier block: the verifier's
    finding is already distilled into each dimension's ``why``).
    """
    if no_submission:
        return (
            "[RETRY — last time you ran out of budget WITHOUT submitting]\n"
            "You never concluded. Commit to your best hypothesis earlier this time "
            "instead of exhausting the step budget."
        )

    verdict_lines = []
    for dim in ("detection", "root_cause", "localization"):
        status, why = review.statuses[dim]
        verdict_lines.append(f"  - {_DIM_LABELS[dim]}: {status}" + (f" — {why}" if why else ""))
    block = (
        "[RETRY — an independent reviewer (it does NOT know the answer) graded your "
        "previous attempt on evidence. KEEP the SUPPORTED dimensions; RE-WORK the "
        "WEAK/UNSUPPORTED ones by gathering the missing evidence with your tools — "
        "do not resubmit them unverified.]\n"
        "Your previous answer:\n"
        f"  detection : {submission.get('is_anomaly')}\n"
        f"  root cause: {submission.get('root_cause_name', [])}\n"
        f"  devices   : {submission.get('faulty_devices', [])}\n\n"
        "Reviewer verdict:\n" + "\n".join(verdict_lines)
    )
    if review.hint:
        block += f"\n\n[GUIDANCE — for the non-SUPPORTED parts only]\n{review.hint}"
    return block
