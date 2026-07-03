"""In-loop feedback helpers for the retry loop in ``BasicReActAgent``.

Two responsibilities, kept deliberately on the *judge side* of the leak
firewall (they may read the ground truth; the agent model must not):

* ``attempt_digest`` — a **deterministic** (no-LLM, leak-free by construction)
  summary of what the agent did in the previous attempt, built from the tool
  calls recorded in ``messages.jsonl``.
* ``generate_feedback`` — a **single** GT-aware LLM call (the judge model,
  e.g. qwen) that turns "what went wrong" into a short redirect hint at the
  *fault-family* granularity, never naming the exact root cause or devices.
  A mechanical ``scrub_ground_truth`` pass guarantees the hard constraint even
  if the model disobeys.

The evaluator package (``nika.evaluator``) is intentionally *not* imported
here beyond the pure ``generic_eval``; this module only needs ``load_model``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import tracing_context

# Float tolerance for the deterministic "resolved" check. On these tasks F1 is
# quantized and never lands in (0.667, 1.0), so this only guards against float
# noise — it never turns a partial score into a pass.
RESOLVED_EPS = 1e-9

_COACH_SYSTEM = (
    "You are a senior network troubleshooting coach. A junior agent tried to "
    "diagnose a network fault and got it wrong (or ran out of budget). You are "
    "given what the agent did and the CORRECT FAULT FAMILY. Write a SHORT hint "
    "(2-3 sentences) that redirects the agent toward the right area of "
    "investigation.\n"
    "HARD RULES:\n"
    "- Steer only to the FAULT FAMILY / class of problem and the type of checks "
    "that family requires.\n"
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
    """Mask any literal ground-truth token that slipped into ``text``.

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


def generate_feedback(
    *,
    session_dir: str,
    fault_family: str,
    gt: dict,
    wrong_dims: list[str],
    llm,
    no_submission: bool,
) -> str:
    """Single GT-aware judge-side call → scrubbed, family-level redirect hint."""
    digest = attempt_digest(session_dir)
    diag_text = last_diagnosis_text(session_dir)

    if no_submission:
        situation = (
            "The agent ran out of its step budget WITHOUT producing a final "
            "submission. It never concluded."
        )
    else:
        dims = ", ".join(wrong_dims) if wrong_dims else "the diagnosis"
        situation = f"The agent submitted a WRONG answer. Incorrect on: {dims}."

    family = fault_family or "unknown"
    human = (
        f"{situation}\n\n"
        f"What the agent did:\n{digest}\n\n"
        f"Its last reasoning (may be empty):\n{diag_text or '(none)'}\n\n"
        f"CORRECT FAULT FAMILY (for your eyes only, do not name specifics): {family}\n\n"
        "Write the redirect hint now."
    )

    with tracing_context(enabled=False):
        response = llm.invoke([
            SystemMessage(content=_COACH_SYSTEM),
            HumanMessage(content=human),
        ])
    hint = str(getattr(response, "content", "")).strip()
    return scrub_ground_truth(hint, gt)
