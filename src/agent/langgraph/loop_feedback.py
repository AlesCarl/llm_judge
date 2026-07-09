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
   second), picks the most consistent fault family from the closed registry
   list, extracts new case-file facts, and writes a short redirect hint.

3. ``compose_feedback`` assembles the deterministic message injected into the
   next attempt: submission echo + per-dimension verdict + verifier
   observations + (from the 2nd feedback on) the family differential card
   for the *coach-suspected* family + the hint.

Leak note: ``family_differential`` is unchanged (deterministic, alphabetical,
GT-independent by construction) but is now keyed by the coach's suspicion,
not by the GT family — the card may therefore be the wrong family; the lead
text says so explicitly.
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
    "You are an independent network fault verifier. A junior agent has "
    "diagnosed a fault on a live network; you are given its claims. You do "
    "NOT know the correct answer.\n"
    "Your job: audit the claims against the live network with as few tool "
    "calls as possible.\n"
    "Method:\n"
    "1. Pick the 2-4 claims that carry the diagnosis (the stated root cause, "
    "the accused devices, the anomaly itself).\n"
    "2. For each, run the single most direct check that could confirm or "
    "refute it.\n"
    "3. If a check refutes a claim, note what you observed instead.\n"
    "Rules: read-only — only inspect, never fix or change anything; audit the "
    "claims, do not re-diagnose the whole network.\n"
    "Finish with a plain report, one line per claim:\n"
    "CLAIM: <claim> — CONFIRMED | REFUTED | COULD-NOT-CHECK — <evidence observed>"
)


_COACH_SYSTEM = (
    "You are a senior network troubleshooting reviewer. A junior agent "
    "diagnosed a network fault. You do NOT know the correct answer — judge "
    "only whether each part of its answer is backed by the available evidence "
    "(the independent verification report, when present, outweighs the "
    "agent's own reasoning).\n"
    "Grade three dimensions:\n"
    "- detection: is the anomaly / no-anomaly call justified by evidence?\n"
    "- localization: are the accused devices actually implicated?\n"
    "- root_cause: does the evidence establish the named cause itself, or "
    "only a symptom of it?\n"
    "Status meaning: SUPPORTED = direct evidence backs it; WEAK = plausible "
    "but unverified; UNSUPPORTED = contradicted or no evidence.\n"
    "Then write a hint (2-3 sentences) ONLY about the non-SUPPORTED "
    "dimensions: name the missing check or the contradiction, concretely. "
    "Never write the diagnosis for the agent.\n"
    "Pick suspected_family: the fault family most consistent with the "
    "evidence, chosen from KNOWN FAULT FAMILIES, or \"unsure\".\n"
    "List up to 5 new_facts: short factual observations established this "
    "attempt (device/interface states, outputs seen) — facts only, no "
    "conclusions.\n"
    "Reply with ONLY this JSON (no other text):\n"
    '{"detection": {"status": "...", "why": "..."}, '
    '"localization": {"status": "...", "why": "..."}, '
    '"root_cause": {"status": "...", "why": "..."}, '
    '"suspected_family": "...", "new_facts": ["..."], "hint": "..."}'
)


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


def family_differential(family: str) -> str:
    """Leak-safe differential body: the sorted list of ALL known sibling sub-causes
    of a fault family, each with its observable discriminator (one " - name: sign"
    line per sibling). The instructional framing around it is added by the caller.

    Fully deterministic (no LLM). Built straight from the problem registry, so the
    output depends ONLY on *family* and is byte-identical regardless of which
    sub-cause is the real fault. Since the GT-free rework the family is the
    COACH'S SUSPICION (inferred from evidence), never the GT category.

    Returns "" when the family is unknown or has fewer than 2 members (a
    single-entry list would disclose too much).
    """
    from nika.orchestrator.problems.prob_pool import _PROBLEMS

    rows: list[tuple[str, str]] = []
    for name, levels in _PROBLEMS.items():
        cls = next(iter(levels.values()), None)
        if cls is None:
            continue
        if str(cls.META.root_cause_category) != family:
            continue
        disc = (getattr(cls, "discriminator", "") or "").strip()
        rows.append((name, disc))

    if len(rows) < 2:
        return ""
    rows.sort(key=lambda r: r[0])  # fixed alphabetical order
    return "\n".join(f" - {name}: {disc}" for name, disc in rows)


# --------------------------------------------------------------------------
# Coach review
# --------------------------------------------------------------------------

@dataclass
class CoachReview:
    """Parsed outcome of one coach review (GT-free)."""

    statuses: dict[str, tuple[str, str]]  # dim -> (STATUS, why)
    suspected_family: str = ""
    new_facts: list[str] = field(default_factory=list)
    hint: str = ""
    verification_report: str = ""
    verified: bool = False  # True when the tool-using verification pass ran

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
        hint=str(data.get("hint", "")).strip()[:800],
    )


def merge_case_file(existing: list[str], new_facts: list[str]) -> list[str]:
    """Append non-duplicate facts; keep the newest _CASE_FILE_CAP entries."""
    merged = list(existing)
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
            return _strip_think(str(result["messages"][-1].content))[:1500]
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
        human = (
            f"TASK GIVEN TO THE AGENT:\n{task_description[:600]}\n\n"
            f"AGENT'S SUBMISSION:\n{sub_txt}\n\n"
            f"AGENT'S REASONING:\n{(diagnosis_report or '(none)')[:2200]}\n\n"
            f"AGENT'S TOOL ACTIVITY:\n{digest[:2500]}\n\n"
            f"INDEPENDENT VERIFICATION REPORT:\n{verification_report or '(not available)'}\n\n"
            f"KNOWN FAULT FAMILIES: {', '.join(known_families())}\n\n"
            "Write the JSON review now."
        )
        with tracing_context(enabled=False):
            response = self.llm.invoke([
                SystemMessage(content=_COACH_SYSTEM),
                HumanMessage(content=human),
            ])
        review = _parse_review(str(getattr(response, "content", "")))
        review.verification_report = verification_report
        review.verified = verified
        if no_submission:
            # No claims were made: nothing can be SUPPORTED.
            review.statuses = {dim: ("UNSUPPORTED", "no submission was made") for dim in _DIMS}
        return review


# --------------------------------------------------------------------------
# Feedback assembly (deterministic; injected into the next attempt)
# --------------------------------------------------------------------------

_DIM_LABELS = {"detection": "detection ", "root_cause": "root cause", "localization": "devices   "}


def compose_feedback(
    *,
    review: CoachReview,
    submission: dict,
    loop_count: int,
    no_submission: bool,
) -> str:
    """Assemble the retry message from the coach review (no GT anywhere)."""
    if no_submission:
        parts = [
            "[REVIEW — you ran out of budget WITHOUT submitting]\n"
            "You never concluded last time. Commit to your best hypothesis "
            "earlier this time instead of exhausting the step budget."
        ]
    else:
        verdict_lines = []
        for dim in ("detection", "root_cause", "localization"):
            status, why = review.statuses[dim]
            verdict_lines.append(f"  - {_DIM_LABELS[dim]}: {status}" + (f" — {why}" if why else ""))
        parts = [
            "[REVIEW OF YOUR PREVIOUS ATTEMPT — an independent reviewer graded each "
            "dimension on EVIDENCE (it does not know the answer)]\n"
            "Your previous submission:\n"
            f"  detection : {submission.get('is_anomaly')}\n"
            f"  root cause: {submission.get('root_cause_name', [])}\n"
            f"  devices   : {submission.get('faulty_devices', [])}\n\n"
            "Verdict (SUPPORTED = evidence backs it, keep it; WEAK/UNSUPPORTED = re-work it):\n"
            + "\n".join(verdict_lines)
        ]

    if review.verification_report:
        parts.append(
            "[VERIFIER OBSERVATIONS — checks run against the live network]\n"
            + review.verification_report[:900]
        )

    # Escalation from the 2nd feedback on: attach the differential card for the
    # COACH-SUSPECTED family (may be wrong — the lead says so).
    rca_status = review.statuses["root_cause"][0]
    if loop_count >= 2 and rca_status != "SUPPORTED":
        card_body = family_differential(review.suspected_family)
        if card_body:
            parts.append(
                "[ROOT-CAUSE DIFFERENTIAL — from the symptoms observed so far, the "
                f"reviewer suspects the '{review.suspected_family}' fault family. Its "
                "known sub-causes and their observable signs are listed below "
                "(alphabetical; the reviewer does NOT know which is correct). Match "
                "the signs to your evidence — and if none fits what you saw, the "
                "family suspicion itself may be wrong.]\n"
                + card_body
            )

    if review.hint:
        parts.append(f"[GUIDANCE — for the non-SUPPORTED parts only]\n{review.hint}")

    return "\n\n".join(parts)
