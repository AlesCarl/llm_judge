"""In-loop feedback helpers for the retry loop in ``BasicReActAgent``.

Two responsibilities, kept deliberately on the judge side of the leak
firewall (they may read the ground truth; the agent model must not):

- "attempt_digest" — a deterministic (no-LLM, leak-free by construction)
  summary of what the agent did in the previous attempt, built from the tool
  calls recorded in messages.jsonl.

- "generate_feedback" — a single GT-aware LLM call (the judge model,
  e.g. qwen) that turns "what went wrong" into a short redirect hint at the
  fault-family granularity, never naming the exact root cause or devices.
  A mechanical "scrub_ground_truth" pass guarantees the hard constraint even
  if the model disobeys.

"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import tracing_context

# Float tolerance for the deterministic "resolved" check. 
RESOLVED_EPS = 1e-9

_COACH_SYSTEM = (
    "You are a senior network troubleshooting coach. A junior agent tried to "
    "diagnose a network fault and got PART of it wrong. You are given what the "
    "agent did, which parts of its answer are still 'to fix', and the CORRECT "
    "FAULT FAMILY. Write a SHORT hint (2-3 sentences) that helps the agent fix "
    "ONLY the parts still to fix.\n"
    "HARD RULES:\n"
    "- Give guidance ONLY on the dimensions listed as 'to fix'. Any dimension "
    "NOT listed is already CONFIRMED correct: the agent must keep it exactly as "
    "is — never suggest changing a confirmed part.\n"
    "- Steer only to the FAULT FAMILY / class of problem and the type of checks "
    "that the still-wrong part requires.\n"
    "- NEVER name a specific root cause label, and NEVER name specific device or "
    "host names. Do not reveal the answer — only point the agent in the right "
    "direction so it can find it itself.\n"
    "- Be concrete about what the agent over-explored or neglected."
)


def is_resolved(detection_score: float, loc_f1: float, rca_f1: float) -> bool:
    """Composite stop criterion: detection, localization and RCA all exact-match."""
    return (
        detection_score == 1.0
        and loc_f1 >= 1.0 - RESOLVED_EPS
        and rca_f1 >= 1.0 - RESOLVED_EPS
    )


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


def attempt_digest(session_dir: str) -> str:
    """Deterministic, leak-free digest of the previous attempt's tool activity."""
    events = _tool_events(session_dir)
    tools: list[str] = []
    for e in events:
        if e.get("event") != "tool_start":
            continue
        tool = e.get("tool")
        name = tool.get("name") if isinstance(tool, dict) else str(tool)
        tools.append(name or "unknown_tool")

    if not tools:
        return "Previous attempt made no tool calls."

    # Preserve order but collapse consecutive/aggregate counts for compactness.
    counts: dict[str, int] = {}
    for name in tools:
        counts[name] = counts.get(name, 0) + 1
    summary = ", ".join(f"{name}×{n}" for name, n in counts.items())
    return f"Previous attempt used {len(tools)} tool calls: {summary}."


def last_diagnosis_text(session_dir: str, max_chars: int = 1200) -> str:
    """Best-effort last free-text reasoning the agent produced (for coach context)."""
    events = _tool_events(session_dir)
    for e in reversed(events):
        if e.get("event") == "llm_end":
            text = str(e.get("text") or "").strip()
            if text:
                return text[:max_chars]
    return ""


def scrub_ground_truth(text: str, gt: dict) -> str:
    """Mask any literal ground-truth token that slipped into text.

    Belt-and-suspenders after the LLM: even if the coach names a device or the
    exact root cause, the literal string never reaches the agent.
    """
    tokens: list[str] = []
    tokens += [str(t) for t in gt.get("faulty_devices", []) if t]
    tokens += [str(t) for t in gt.get("root_cause_name", []) if t]
    scrubbed = text
    for tok in sorted(set(tokens), key=len, reverse=True):
        scrubbed = re.sub(re.escape(tok), "[…]", scrubbed, flags=re.IGNORECASE)
    return scrubbed


def _loc_verdict(prec: float, rec: float, f1: float, n: int) -> tuple[str, str]:
    """Per-set verdict for localization from precision/recall (STRICT, leak-free).

    Splits on precision first: only when everything the agent listed is correct
    (prec == 1) is it safe to tell it to KEEP them all. The strict variant never
    asserts that more devices exist (no cardinality hint).
    """
    if n == 0:
        return "missing", "affected devices: NONE PROVIDED — you must identify the affected device(s)."
    if f1 >= 1.0 - RESOLVED_EPS:
        return "correct", "affected devices: this list looks right — keep it unless new evidence clearly contradicts it."
    if prec >= 1.0 - RESOLVED_EPS:
        return (
            "incomplete",
            "affected devices: the ones you listed look right — keep them unless "
            "clearly contradicted; the list may be incomplete, so check whether other "
            "devices of the same kind are also involved.",
        )
    if prec <= RESOLVED_EPS:
        return (
            "wrong",
            "affected devices: none of the devices you listed appear to be involved "
            "— reconsider where you look.",
        )
    return (
        "mixed",
        "affected devices: SOME of the devices you listed are not involved — keep "
        "only the ones you are confident about and re-examine the rest.",
    )


def _rca_verdict(prec: float, rec: float, f1: float, n: int) -> tuple[str, str]:
    """Per-set verdict for the root cause (usually a single label)."""
    if n == 0:
        return "missing", "root cause: NONE PROVIDED — you must name the most likely cause."
    if f1 >= 1.0 - RESOLVED_EPS:
        return "correct", "root cause: keep your current hypothesis unless new evidence clearly contradicts it."
    if prec >= 1.0 - RESOLVED_EPS and rec < 1.0 - RESOLVED_EPS:
        return (
            "incomplete",
            "root cause: your current hypothesis looks right — keep it unless "
            "contradicted; consider whether another cause co-occurs.",
        )
    if prec <= RESOLVED_EPS:
        return "wrong", "root cause: not correct — reconsider the cause."
    return "mixed", "root cause: partly correct — keep the right one(s) and reconsider the rest."


def _det_verdict(det: float) -> tuple[str, str]:
    if det >= 1.0 - RESOLVED_EPS:
        return "correct", "detection: your 'anomaly' assessment looks right — keep it unless new evidence clearly contradicts it."
    return "wrong", "detection: there IS an anomaly to find — do not conclude 'no anomaly'."


def build_verdict(submission: dict, scores: tuple) -> tuple[str, list[str]]:
    """Deterministic, leak-controlled keep/fix header from the agent's OWN answer.

    The header echoes the agent's own submission (never GT) tagged CONFIRMED vs
    to-fix, so a retry can complete/repair the wrong parts without discarding the
    correct ones. Returns (header_text, dims_to_fix).
    """
    det = scores[0]
    dstat, dline = _det_verdict(det)
    lstat, lline = _loc_verdict(scores[2], scores[3], scores[4], len(submission.get("faulty_devices") or []))
    rstat, rline = _rca_verdict(scores[6], scores[7], scores[8], len(submission.get("root_cause_name") or []))

    header = (
        "[REVIEW OF YOUR PREVIOUS ATTEMPT — refine it, do NOT restart from zero]\n"
        "Your previous submission:\n"
        f"  detection : {submission.get('is_anomaly')}\n"
        f"  root cause: {submission.get('root_cause_name', [])}\n"
        f"  devices   : {submission.get('faulty_devices', [])}\n\n"
        "Verdict (keep what looks solid, fix only the rest):\n"
        f"  - {dline}\n  - {rline}\n  - {lline}"
    )
    to_fix = [
        name
        for name, st in (("detection", dstat), ("root cause", rstat), ("localization", lstat))
        if st != "correct"
    ]
    return header, to_fix


def generate_feedback(
    *,
    session_dir: str,
    fault_family: str,
    gt: dict,
    submission: dict,
    scores: tuple | None,
    llm,
    no_submission: bool,
) -> str:
    """Deterministic keep/fix verdict + single GT-aware family-level coach hint.

    The verdict header is built from the agent's OWN submission (no GT, so it is
    NOT scrubbed). Only the coach's LLM hint is GT-aware and gets scrubbed.
    """
    digest = attempt_digest(session_dir)
    diag_text = last_diagnosis_text(session_dir)

    if no_submission or scores is None:
        header = (
            "[REVIEW — you ran out of budget WITHOUT submitting]\n"
            "You never concluded last time. Investigate more directly this time."
        )
        to_fix = ["detection", "localization", "root cause"]
    else:
        header, to_fix = build_verdict(submission, scores)

    focus_txt = ", ".join(to_fix) if to_fix else "the remaining details"
    family = fault_family or "unknown"
    human = (
        f"What the agent did:\n{digest}\n\n"
        f"Its last reasoning (may be empty):\n{diag_text or '(none)'}\n\n"
        f"Parts still TO FIX: {focus_txt}. Do NOT give guidance on anything else "
        "— the other dimensions are already correct and must be kept.\n"
        f"CORRECT FAULT FAMILY (for your eyes only, do not name specifics): {family}\n\n"
        "Write the redirect hint now (only for the parts to fix)."
    )

    with tracing_context(enabled=False):
        response = llm.invoke([
            SystemMessage(content=_COACH_SYSTEM),
            HumanMessage(content=human),
        ])
    hint = scrub_ground_truth(str(getattr(response, "content", "")).strip(), gt)

    return (
        f"{header}\n\n"
        f"[GUIDANCE — for the parts to fix only]\n{hint}\n\n"
        "Re-investigate ONLY the parts to fix; keep the parts that look solid unless "
        "you find clear evidence against them; then conclude with an updated "
        "submission. Always submit your best hypothesis — never leave it empty."
    )
