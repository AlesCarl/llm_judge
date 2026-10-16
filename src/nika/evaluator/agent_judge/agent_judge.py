"""AgentAsJudge: a tool-using judge that audits the agent's own trace.

Phase A (no ground truth): a ReAct agent checks the submitted claims against
the recorded trace, within a tool-call budget, and reports CONFIRMED / REFUTED /
COULD-NOT-CHECK per claim.

Phase B (with ground truth): one structured call scores the five criteria,
citing the Phase-A verdicts, so scores rest on checked evidence and not on how
well the agent wrote its report.

The session is already closed when the judge runs, so the tools read saved
artifacts (messages.jsonl, submission.json), not the live network.
"""

import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from langsmith import tracing_context

from agent.llm.model_factory import load_model
from nika.evaluator.agent_judge.prompts import SCORING_PROMPT, SCORING_SYSTEM, VERIFIER_SYSTEM
from nika.evaluator.agent_judge.verify_tools import build_verify_tools, check_taxonomy
from nika.evaluator.base_judge import BaseJudge
from nika.evaluator.result_log import SUBMISSION_FILENAME
from nika.evaluator.schemas import JudgeResponse
from nika.evaluator.token_meter import dump_cost, meter_config, new_meter

load_dotenv()

logger = logging.getLogger(__name__)

# Phase-A budget (recursion_limit): bounds the cost.
_VERIFY_BUDGET = 15


# Matches "CLAIM: <claim> — <VERDICT> — <evidence>". The verdict anchors the split;
# markdown bold is tolerated and the evidence is optional.
_CLAIM_RE = re.compile(
    r"CLAIM:\s*\**\s*(?P<claim>.*?)\s*[—-]+\s*\**\s*"
    r"(?P<verdict>COULD-NOT-CHECK|CONFIRMED|REFUTED)\b"
    r"\**\s*(?:[—-]+\s*(?P<evidence>.*))?",
    re.IGNORECASE,
)


def _claim_key(claim: str) -> str:
    """Normalise a claim so restatements count once."""
    return re.sub(r"[\s'\"\[\]=:]+", "", claim).lower()


def _parse_claim_verdicts(report: str) -> list[dict]:
    """Parse the verifier's CLAIM lines into {claim, verdict, evidence} records.

    A restated claim keeps its last occurrence. Unparseable reports (e.g. budget
    exhausted) give an empty list.
    """
    seen: dict[str, dict] = {}
    order: list[str] = []
    for line in report.splitlines():
        if "CLAIM:" not in line.upper():
            continue
        m = _CLAIM_RE.search(line)
        if not m:
            continue
        key = _claim_key(m.group("claim"))
        if key not in seen:
            order.append(key)
        seen[key] = {
            "claim": m.group("claim").strip().strip("*").strip(),
            "verdict": m.group("verdict").upper(),
            "evidence": (m.group("evidence") or "").strip().strip("*").strip(),
        }
    return [seen[k] for k in order]


def _serialize_messages(messages) -> list[dict]:
    """Flatten the LangGraph messages into a JSON-serialisable audit trail."""
    out = []
    for m in messages:
        entry = {
            "type": getattr(m, "type", m.__class__.__name__),
            "content": str(getattr(m, "content", "")),
        }
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {"name": tc.get("name"), "args": tc.get("args")} for tc in tool_calls
            ]
        out.append(entry)
    return out


class AgentAsJudge(BaseJudge):
    """Verification (Phase A), then scoring (Phase B)."""

    def __init__(
        self,
        judge_llm_backend: str = "openai",
        judge_model: str = "gpt-5-mini",
        verify_budget: int = _VERIFY_BUDGET,
    ) -> None:
        self.judge_llm_backend = judge_llm_backend
        self.judge_model = judge_model
        self.verify_budget = verify_budget

        self._scoring_llm = load_model(
            llm_backend=judge_llm_backend, model=judge_model
        ).with_structured_output(JudgeResponse)

    @staticmethod
    def _read_submission(session_dir: str) -> dict:
        path = Path(session_dir) / SUBMISSION_FILENAME
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _run_verifier(self, session_dir: str, submission: dict, meter) -> tuple[str, list[dict]]:
        """Phase A: audit the submission against the session's trace, without ground truth."""
        verifier_llm = load_model(llm_backend=self.judge_llm_backend, model=self.judge_model)
        verifier_agent = create_react_agent(
            model=verifier_llm,
            tools=build_verify_tools(session_dir),
            prompt=VERIFIER_SYSTEM,
        )

        if submission:
            request = (
                "Claims to audit (the agent's own final submission — NOT the ground truth):\n"
                f"  anomaly present : {submission.get('is_anomaly')}\n"
                f"  root cause      : {submission.get('root_cause_name', [])}\n"
                f"  accused devices : {submission.get('faulty_devices', [])}\n\n"
                "Use the tools to audit these claims against this session's own recorded "
                "trace, then report per-claim verdicts."
            )
        else:
            request = (
                "The agent produced NO final submission this session (submission.json "
                "is absent). There are no claims to confirm — do not treat blank fields "
                "as a 'no anomaly' claim. Use the tools to check whether the recorded "
                "trace shows a fault the agent should have caught, and report that no "
                "submission was made."
            )

        config: dict = {"recursion_limit": self.verify_budget}
        mc = meter_config(meter)
        if mc:
            config["callbacks"] = mc["callbacks"]

        try:
            with tracing_context(enabled=False):
                result = verifier_agent.invoke(
                    {"messages": [HumanMessage(content=request)]}, config=config
                )
            report = str(result["messages"][-1].content)
            if "need more steps" in report.lower():
                report = "(verification ran out of budget before completing)\n" + report
            audit_trace = _serialize_messages(result["messages"])
        except GraphRecursionError:
            report = "(verification ran out of budget before completing)"
            audit_trace = []
        except Exception as exc:  # a failed verification must not kill the eval
            logger.warning("AgentAsJudge — verification pass failed: %s", exc)
            report = f"(verification failed: {exc})"
            audit_trace = []

        return report, audit_trace

    def evaluate_agent(self, ground_truth: str, trace_path: str, save_path: str) -> JudgeResponse:
        """Run both phases and save the JudgeResponse, the audit trace and the cost."""
        session_dir = os.path.dirname(trace_path)

        with open(trace_path, "r") as f:
            raw_trace = f.read()
        trace = self._parse_trace(raw_trace)

        meter = new_meter()

        submission = self._read_submission(session_dir)

        logger.info("AgentAsJudge — Phase A: tool-grounded verification (GT-free)")
        verification_report, audit_trace = self._run_verifier(session_dir, submission, meter)

        taxonomy = check_taxonomy(submission.get("root_cause_name") or [])

        logger.info("AgentAsJudge — Phase B: scoring (GT-aware)")
        scoring_prompt = SCORING_PROMPT.format(
            ground_truth=ground_truth,
            trace=trace,
            verification_report=verification_report,
            taxonomy_check=json.dumps(taxonomy),
        )
        with tracing_context(enabled=False):
            evaluation: JudgeResponse = self._scoring_llm.invoke(
                [SystemMessage(content=SCORING_SYSTEM), HumanMessage(content=scoring_prompt)],
                config=meter_config(meter),
            )

        with open(save_path, "w+") as f:
            f.write(evaluation.model_dump_json(indent=2))

        audit_path = save_path.replace("llm_judge.json", "agent_judge_trace.json")
        with open(audit_path, "w+") as f:
            json.dump(
                {
                    "verification_report": verification_report,
                    "claim_verdicts": _parse_claim_verdicts(verification_report),
                    "taxonomy_check": taxonomy,
                    "tool_call_trace": audit_trace,
                },
                f,
                indent=2,
            )

        dump_cost(meter, save_path, judge="agent", filename="agent_cost.json")

        return evaluation
