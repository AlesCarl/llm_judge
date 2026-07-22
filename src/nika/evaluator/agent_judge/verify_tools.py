"""Artifact-grounded verification tools for AgentAsJudge's Phase A (GT-free).

The Kathara environment is torn down before the judge ever runs (session
close happens before evaluation — see project note on post-hoc evaluation),
so these tools operate on the CLOSED session's saved artifacts
(``messages.jsonl``, ``submission.json``) rather than the live network. This
is the necessary adaptation of Agent-as-a-Judge's "tool-grounded
verification" to a post-hoc evaluation setting.
"""

import ast
import json
import re
from pathlib import Path

from langchain_core.tools import tool

from nika.evaluator.result_log import SUBMISSION_FILENAME


def _read_events(session_dir: str) -> list[dict]:
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
        except json.JSONDecodeError:
            continue
    return events


def _clean_output(raw: str, max_chars: int = 400) -> str:
    """Best-effort extraction of the text content from a str(ToolMessage).

    LangChain tool outputs are logged as ``str(ToolMessage)``, whose
    ``content`` is normally a list of content blocks (``content=[{'type':
    'text', 'text': '...'}, ...] name=... tool_call_id=...``), not a bare
    string — a single quoted-string fallback misses this and would return
    the raw repr verbatim (including the ``content=[...] name=...`` noise).
    """
    m = re.search(r"^content=(\[.*?\])\s+name=", raw, flags=re.DOTALL)
    if m:
        try:
            content_list = ast.literal_eval(m.group(1))
            texts = [item.get("text", "") for item in content_list if isinstance(item, dict)]
            text = " ".join(t for t in texts if t)
        except (ValueError, SyntaxError):
            text = m.group(1)
    else:
        m2 = re.search(r"content=['\"](.*?)['\"]\s+\w+=", raw, flags=re.DOTALL)
        text = m2.group(1) if m2 else raw
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _paired_tool_calls(session_dir: str) -> list[dict]:
    """Pair each tool_start with its tool_end/tool_error, FIFO.

    The trace is written by a single sequential agent loop, so calls never
    overlap and a simple FIFO queue correctly matches each start with its own
    end/error.
    """
    calls: list[dict] = []
    pending: list[dict] = []
    for e in _read_events(session_dir):
        event = e.get("event")
        if event == "tool_start":
            tool_info = e.get("tool")
            name = tool_info.get("name") if isinstance(tool_info, dict) else str(tool_info)
            call = {
                "agent": e.get("agent"),
                "name": name or "unknown_tool",
                "input": str(e.get("input") or "")[:200],
                "output": "",
            }
            calls.append(call)
            pending.append(call)
        elif event in ("tool_end", "tool_error") and pending:
            call = pending.pop(0)
            raw = str(e.get("output") or e.get("error") or "")
            prefix = "ERROR: " if event == "tool_error" else ""
            call["output"] = prefix + _clean_output(raw)
    return calls


def check_taxonomy(root_cause_names: list[str]) -> dict:
    """Deterministic (non-LLM) check: is each claimed root cause a known name?

    Compares against the full closed-world registry of fault names
    (``list_avail_problem_names()`` from the problem pool) — this is not the
    ground truth for this scenario, just the general vocabulary of possible
    root causes, so it carries no scenario-specific leakage.
    """
    from nika.orchestrator.problems.prob_pool import list_avail_problem_names

    known = set(list_avail_problem_names())
    return {
        "known_vocabulary_size": len(known),
        "claimed": list(root_cause_names or []),
        "unknown": [n for n in (root_cause_names or []) if n not in known],
    }


def build_verify_tools(session_dir: str) -> list:
    """Build the Phase-A verification toolset bound to one closed session."""

    @tool
    def search_trace(query: str) -> str:
        """Search this session's recorded tool calls for ones whose name, input, or output mentions `query` (e.g. a device name, protocol, or keyword from a claim). Returns matching calls with their actual recorded output, so you can check whether a claim is backed by real evidence from the trace instead of trusting the agent's own report."""
        query_lower = query.lower()
        matches = [
            c
            for c in _paired_tool_calls(session_dir)
            if query_lower in c["name"].lower()
            or query_lower in c["input"].lower()
            or query_lower in c["output"].lower()
        ]
        if not matches:
            return f"No recorded tool call matches '{query}'."
        lines = [f"{len(matches)} matching call(s):"]
        for c in matches[:10]:
            lines.append(f"- [{c['agent']}] {c['name']}({c['input']}) -> {c['output'] or '(no output captured)'}")
        return "\n".join(lines)

    @tool
    def list_executed_tools() -> str:
        """List every tool call the agent actually made this session, in order, with counts. Use this to spot blind spots: a claim about a device or check the agent never actually ran cannot be marked CONFIRMED."""
        calls = _paired_tool_calls(session_dir)
        if not calls:
            return "The agent made no recorded tool calls this session."
        counts: dict[str, int] = {}
        for c in calls:
            counts[c["name"]] = counts.get(c["name"], 0) + 1
        summary = ", ".join(f"{name}×{n}" for name, n in counts.items())
        return f"{len(calls)} tool call(s) total ({summary})."

    @tool
    def get_submission_claims() -> str:
        """Return the agent's final submitted claims (is_anomaly, faulty_devices, root_cause_name) to audit — read from this session's submission.json, never the ground truth."""
        path = Path(session_dir) / SUBMISSION_FILENAME
        if not path.exists():
            return "No submission.json found — the agent submitted nothing."
        return path.read_text(encoding="utf-8")

    return [search_trace, list_executed_tools, get_submission_claims]
